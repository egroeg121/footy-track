"""Cheap global camera motion for the static-object filter (needs OpenCV, no GPU).

Phase correlation between consecutive frames, downscaled to ``width`` px and
windowed, summed into a cumulative image shift in native pixels. Only the size of the
shift between two frames matters to :func:`footy_track.autolabel.loop.static_candidates`,
so the sign convention and the slow drift of a cumulative sum do not.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MIN_RESPONSE = 0.03  # below this the correlation peak is noise: treat as no motion


def estimate_camera(
    video: Path, width: int = 480, max_frames: int | None = None
) -> dict[int, tuple[float, float]]:
    """``{frame: (x, y)}`` cumulative global shift in native px, frame 0 at (0, 0)."""
    cap = cv2.VideoCapture(str(video))
    out: dict[int, tuple[float, float]] = {}
    prev = None
    win = None
    x = y = 0.0
    idx = 0
    try:
        while max_frames is None or idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            h0, w0 = frame.shape[:2]
            scale = w0 / width
            small = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (width, round(h0 / scale)),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            if win is None:
                win = cv2.createHanningWindow(
                    (small.shape[1], small.shape[0]), cv2.CV_32F
                )
            if prev is not None:
                (dx, dy), resp = cv2.phaseCorrelate(prev, small, win)
                if resp >= MIN_RESPONSE:
                    x += dx * scale
                    y += dy * scale
            out[idx] = (x, y)
            prev = small
            idx += 1
    finally:
        cap.release()
    return out
