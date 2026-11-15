"""Project a clip's detections onto the 2D pitch through a per-frame calibration.

``project_players(detections, calibration)`` takes the person / ball boxes of a clip and a
:class:`footy_track.calibration.clip_calibration.ClipCalibration` (or any object with
``.H(frame)`` / ``.distortion(frame)`` / ``.row(frame)``) and returns one row per detection::

    frame, track_id (nullable), label, x_m, y_m, conf, calib_confidence, calib_ok, in_pitch, in_bounds

Anchors follow :class:`~footy_track.projection.projector.PitchProjector` (foot point = bottom-centre of the
box for players and everything else, centroid for ball labels), which does the projection itself, lens
distortion included. Pitch coordinates are metres, origin at the centre spot, +x to the right goal, +y to the
far touchline. ``x_m`` / ``y_m`` are NaN for frames without an ``ok`` calibration. ``in_pitch`` is inside the
105 x 68 m lines, ``in_bounds`` inside them plus ``margin_m`` (a player a few metres off the pitch is a throw-in
taker or a bench member, further out is a wrong calibration or a spectator).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from footy_track.constants import (
    BALL_TAG,
    COACH_TAG,
    IN_PLAY_BALL_TAG,
    OUT_OF_PLAY_BALL_TAG,
    PERSON_TAG,
    PLAYER_SUB_TAG,
    PLAYER_TAG,
    REFEREE_TAG,
)
from footy_track.projection.projector import PitchProjector, TrackedDetection

LABELS = (PLAYER_TAG, REFEREE_TAG, COACH_TAG, PERSON_TAG, PLAYER_SUB_TAG, BALL_TAG, IN_PLAY_BALL_TAG, OUT_OF_PLAY_BALL_TAG)
BALL_LABELS = (BALL_TAG, IN_PLAY_BALL_TAG, OUT_OF_PLAY_BALL_TAG)
COLUMNS = ["frame", "track_id", "label", "x_m", "y_m", "conf", "calib_confidence", "calib_ok", "in_pitch", "in_bounds"]
DETECTION_COLUMNS = ["frame", "x", "y", "w", "h", "label", "conf", "track_id"]


def load_detections(path: Path | str, labels=LABELS, min_conf: float = 0.25) -> pd.DataFrame:
    """``machine_labels/*.jsonl`` -> DataFrame ``frame, x, y, w, h`` (normalised top-left xywh), ``label``,
    ``conf``, ``track_id`` (``<NA>`` when the file has none)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lab = next((t for t in r.get("tags", []) if t in labels), None)
        conf = float(r.get("confidence", 1.0))
        if lab is None or conf < min_conf:
            continue
        b = r["bbox"]
        rows.append((int(r["frame_index"]), b["x"], b["y"], b["w"], b["h"], lab, conf, r.get("track_id")))
    df = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
    df["track_id"] = df["track_id"].astype("Int64")
    return df


def project_players(detections: pd.DataFrame, calibration, pitch_dims: tuple[float, float] = (105.0, 68.0),
                    margin_m: float = 5.0, drop_out_of_bounds: bool = False, min_calib_confidence: float = 0.0) -> pd.DataFrame:
    """Pitch positions (m) of ``detections`` (columns of :data:`DETECTION_COLUMNS`; ``track_id`` optional).

    ``drop_out_of_bounds`` removes rows outside the pitch + ``margin_m``; rows of frames without an ``ok``
    calibration (or with ``calib_confidence < min_calib_confidence``) are kept with NaN ``x_m`` / ``y_m`` and
    ``calib_ok`` False, so they stay visible in the coverage figures.
    """
    projector = PitchProjector(pitch_dims, h_quality_threshold=0.0)  # the gate is the calibration's ``ok``
    L, W = pitch_dims
    det = detections.reset_index(drop=True)
    n = len(det)
    x_m, y_m = np.full(n, np.nan), np.full(n, np.nan)
    conf_cal, ok_cal = np.zeros(n), np.zeros(n, bool)
    tid = det["track_id"] if "track_id" in det else pd.Series([pd.NA] * n, dtype="Int64")
    for frame, idx in det.groupby("frame").indices.items():
        row = calibration.row(int(frame))
        if row is None:
            continue
        conf_cal[idx] = float(row["confidence"])
        H = calibration.H(int(frame))
        if H is None or float(row["confidence"]) < min_calib_confidence:
            continue
        ok_cal[idx] = True
        sub = det.iloc[idx]
        dets = [TrackedDetection(-1, float(f), str(lab), float(x), float(y), float(w), float(h))
                for f, lab, x, y, w, h in zip(sub["frame"], sub["label"], sub["x"], sub["y"], sub["w"], sub["h"], strict=True)]
        pos = projector.project_frame(dets, H, H_quality=1.0, distortion=calibration.distortion(int(frame)))
        x_m[idx] = [p.x_pitch for p in pos]
        y_m[idx] = [p.y_pitch for p in pos]
    in_pitch = (np.abs(x_m) <= L / 2) & (np.abs(y_m) <= W / 2)
    in_bounds = (np.abs(x_m) <= L / 2 + margin_m) & (np.abs(y_m) <= W / 2 + margin_m)
    out = pd.DataFrame({
        "frame": det["frame"].to_numpy(), "track_id": tid.astype("Int64").to_numpy(), "label": det["label"].to_numpy(),
        "x_m": x_m, "y_m": y_m, "conf": det["conf"].to_numpy() if "conf" in det else np.ones(n),
        "calib_confidence": conf_cal, "calib_ok": ok_cal, "in_pitch": in_pitch, "in_bounds": in_bounds})
    out["track_id"] = out["track_id"].astype("Int64")
    if drop_out_of_bounds:
        out = out[out["in_bounds"] | ~out["calib_ok"]].reset_index(drop=True)
    return out[COLUMNS]


def project_boxes(boxes: np.ndarray, H: np.ndarray, distortion: dict | None = None, ball: np.ndarray | None = None,
                  pitch_dims: tuple[float, float] = (105.0, 68.0)) -> np.ndarray:
    """(N, 2) pitch metres of normalised top-left xywh ``boxes`` under one ``H``: foot point, or centroid where
    ``ball`` (bool (N,)) is set. Thin wrapper for evaluation code that compares two homographies on the same boxes."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    isball = np.zeros(len(b), bool) if ball is None else np.asarray(ball, bool)
    dets = [TrackedDetection(-1, 0.0, BALL_TAG if ib else PLAYER_TAG, *map(float, r)) for r, ib in zip(b, isball, strict=True)]
    pos = PitchProjector(pitch_dims, h_quality_threshold=0.0).project_frame(dets, H, H_quality=1.0, distortion=distortion)
    return np.array([[p.x_pitch, p.y_pitch] for p in pos]).reshape(-1, 2)
