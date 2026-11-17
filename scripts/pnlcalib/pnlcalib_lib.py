"""Shared helpers for the PnLCalib experiment eval (mini side, our own code only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from footy_track.calibration import geometry as geo

CENTRE = np.array([[0.5, 0.5]])
FLIPS = {"+x+y": (1, 1), "+x-y": (1, -1), "-x+y": (-1, 1), "-x-y": (-1, -1)}


def load_results(path: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for ln in Path(path).read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[(r["clip"], r["frame"])] = r
    return out


def variant_H(var: dict, W: int, H: int, flip=(1, -1)) -> np.ndarray | None:
    """PnLCalib variant -> our H (normalised image -> pitch m, origin centre spot, +x right goal, +y far touchline).

    PnLCalib: pitch centred (x along the length, y across, z = 0 on the ground, z negative up), pixels of the original image.
    ``cam*``: P (3x4) maps (x, y, z, 1) -> pixels; on the ground plane that is columns 0, 1, 3. ``gh*``: ``Hinv`` is pixels -> centred
    pitch metres. ``flip`` = signs applied to (x, y) to land in our convention (determined on the GT, see ``fit_flip``)."""
    if not var or not var.get("ok"):
        return None
    D = np.diag([W, H, 1.0])
    if "P" in var:
        P = np.asarray(var["P"], float)
        Hp = P[:, [0, 1, 3]]
        if abs(np.linalg.det(Hp)) < 1e-18:
            return None
        G = np.linalg.inv(Hp) @ D
    else:
        G = np.asarray(var["Hinv"], float) @ D
    G = np.diag([flip[0], flip[1], 1.0]) @ G
    if not np.isfinite(G).all() or np.linalg.cond(G) > 1e9:
        return None
    return geo.orient_homography(G, CENTRE)


def pitch_pts(H, pts, dist=None):
    return geo.image_to_pitch(H, pts, dist)[0]


def mean_err_m(H_est: np.ndarray | None, H_gt: np.ndarray, pts: np.ndarray, dist_est=None) -> tuple[float, np.ndarray]:
    """Per-point error (m) between the estimated and the GT homography over image points ``pts`` (normalised); NaN if no H."""
    if H_est is None:
        return float("nan"), np.full(len(pts), np.nan)
    e = np.linalg.norm(pitch_pts(H_est, pts, dist_est) - pitch_pts(H_gt, pts), axis=1)
    return float(np.nanmean(e)) if np.isfinite(e).any() else float("nan"), e


def overlay(img_bgr: np.ndarray, Hs: dict[str, tuple[np.ndarray | None, tuple[int, int, int]]], title: str = "") -> np.ndarray:
    """Draw the pitch model projected through each H (name -> (H, BGR colour)) on a copy of the frame."""
    import cv2

    from footy_track.calibration.pitch_model import PitchModel

    out = img_bgr.copy()
    h, w = out.shape[:2]
    pm = PitchModel()
    for i, (name, (H, col)) in enumerate(Hs.items()):
        if H is not None:
            for runs in geo.project_pitch_model(H, pm, step_m=0.5).values():
                for run in runs:
                    pts = np.round(run * np.array([w, h])).astype(np.int32)
                    cv2.polylines(out, [pts], False, col, max(2, w // 640), cv2.LINE_AA)
        cv2.putText(out, name, (12, 36 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out
