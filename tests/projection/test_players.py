"""project_players: boxes of a clip -> pitch metres through a per-frame calibration (synthetic)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration.clip_calibration import ClipCalibration
from footy_track.projection.players import (
    COLUMNS,
    load_detections,
    project_boxes,
    project_players,
)
from footy_track.projection.projector import PitchProjector, TrackedDetection

# normalised image -> pitch: x spans the full length, y flips (image y down = pitch -y)
H0 = np.array([[105.0, 0, -52.5], [0, -68.0, 34.0], [0, 0, 1.0]])


def _cal(ok=(True, True, False), k1=0.0):
    rows = []
    for f, o in enumerate(ok):
        rows.append({"frame": f, "source": "propagated" if o else "none", "anchor_frame": 0, "key_frame": 0, "gap": f,
                     "line_cost": 5.0, "confidence": 0.8 if o else 0.0, "shot_id": 0, "ok": o, "k1": k1,
                     "H": list(H0.ravel()) if o else [np.nan] * 9})
    return ClipCalibration(0, pd.DataFrame(rows))


def _dets():
    return pd.DataFrame({
        "frame": [0, 0, 0, 1, 2],
        "x": [0.45, 0.80, 0.50, 0.10, 0.30], "y": [0.50, 0.40, 0.50, 0.60, 0.30], "w": [0.02, 0.02, 0.01, 0.02, 0.02],
        "h": [0.10, 0.10, 0.01, 0.10, 0.10], "label": ["player", "player", "in_play_ball", "referee", "player"],
        "conf": [0.9, 0.8, 0.5, 0.7, 0.9], "track_id": pd.array([7, pd.NA, pd.NA, 3, 1], dtype="Int64")})


def test_foot_point_ball_centroid_and_columns():
    out = project_players(_dets(), _cal())
    assert list(out.columns) == COLUMNS and len(out) == 5
    p = out.iloc[0]  # foot point = bottom-centre: (0.46, 0.60)
    assert (p.x_m, p.y_m) == pytest.approx((105 * 0.46 - 52.5, -68 * 0.60 + 34))
    b = out.iloc[2]  # ball: centroid (0.505, 0.505)
    assert (b.x_m, b.y_m) == pytest.approx((105 * 0.505 - 52.5, -68 * 0.505 + 34))
    assert out["track_id"].tolist()[:2] == [7, pd.NA] or out["track_id"].iloc[0] == 7
    assert out["calib_confidence"].iloc[0] == pytest.approx(0.8) and out["calib_ok"].iloc[0]


def test_frames_without_a_calibration_are_nan_but_kept():
    out = project_players(_dets(), _cal())
    last = out.iloc[4]
    assert np.isnan(last.x_m) and not last.calib_ok and not last.in_pitch and not last.in_bounds
    assert len(out) == 5


def test_in_pitch_in_bounds_and_drop():
    d = pd.DataFrame({"frame": [0] * 3, "x": [0.5, 0.5, 0.5], "y": [0.40, 0.40, 0.40], "w": [0.01] * 3, "h": [0.01] * 3,
                      "label": ["player"] * 3, "conf": [1.0] * 3})
    d["x"] = [0.5, 1.0 + 3.0 / 105, 1.0 + 20.0 / 105]  # on the pitch, 3 m beyond the touchline end, 20 m beyond
    out = project_players(d, _cal(), margin_m=5.0)
    assert out["in_pitch"].tolist() == [True, False, False] and out["in_bounds"].tolist() == [True, True, False]
    assert out["track_id"].isna().all()
    kept = project_players(d, _cal(), drop_out_of_bounds=True)
    assert len(kept) == 2


def test_min_calibration_confidence_gate():
    out = project_players(_dets(), _cal(), min_calib_confidence=0.9)
    assert not out["calib_ok"].any() and out["x_m"].isna().all()


def test_distortion_is_undistorted_before_h():
    k1 = 0.05
    out = project_players(_dets(), _cal(k1=k1))
    d = _dets().iloc[3]  # near the left edge, where the lens bends most
    foot = np.array([[d.x + d.w / 2, d.y + d.h]])
    expect, _ = geo.apply_homography(H0, geo.undistort_norm(foot, k1))
    assert (out.iloc[3].x_m, out.iloc[3].y_m) == pytest.approx(tuple(expect[0]))
    plain = project_players(_dets(), _cal())
    assert abs(out.iloc[3].x_m - plain.iloc[3].x_m) > 0.05


def test_matches_pitch_projector_directly():
    d = _dets().iloc[:3]
    out = project_players(d, _cal())
    pp = PitchProjector(h_quality_threshold=0.0)
    ref = pp.project_frame([TrackedDetection(-1, 0.0, r.label, r.x, r.y, r.w, r.h) for r in d.itertuples()], H0, 1.0)
    assert np.allclose(out[["x_m", "y_m"]].to_numpy(), [[p.x_pitch, p.y_pitch] for p in ref])


def test_project_boxes_two_homographies():
    boxes = _dets()[["x", "y", "w", "h"]].to_numpy()[:3]
    a = project_boxes(boxes, H0, ball=np.array([False, False, True]))
    shifted = H0 + np.array([[0, 0, 0.01], [0, 0, 0], [0, 0, 0]])  # x shifted by 0.01 * 105 / ... (affine: +0.01 m)
    b = project_boxes(boxes, shifted, ball=np.array([False, False, True]))
    assert a.shape == (3, 2) and np.allclose(b[:, 0] - a[:, 0], 0.01) and np.allclose(b[:, 1], a[:, 1])


def test_load_detections(tmp_path):
    rows = [
        {"frame_index": 3, "bbox": {"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.1}, "tags": ["player", "rtdetr"], "confidence": 0.9},
        {"frame_index": 3, "bbox": {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.01}, "tags": ["in_play_ball"], "confidence": 0.5, "track_id": 4},
        {"frame_index": 3, "bbox": {"x": 0.9, "y": 0.9, "w": 0.05, "h": 0.1}, "tags": ["player"], "confidence": 0.05},
        {"frame_index": 4, "bbox": {"x": 0.3, "y": 0.3, "w": 0.05, "h": 0.1}, "tags": ["crowd"], "confidence": 0.9},
    ]
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    df = load_detections(p)
    assert df["label"].tolist() == ["player", "in_play_ball"] and df["track_id"].tolist() == [pd.NA, 4]
    assert list(df.columns) == ["frame", "x", "y", "w", "h", "label", "conf", "track_id"]
