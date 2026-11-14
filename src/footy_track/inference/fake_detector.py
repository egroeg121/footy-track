"""A model-free detector for tests and for the web app's dev-only fake mode.

Two players walk across the frame and a ball moves right, with a short gap where
the ball goes missing and one stray low-confidence ball. It lets the whole
pipeline run on a machine with no GPU and no weights. DEV ONLY: its output says
nothing about real footage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from footy_track.schema import FrameDetections, ObjectDetection

GAP = range(10, 14)  # frames where the fake detector misses the ball
STRAY_FRAME = 20


def _bounce(s: float, lo: float, hi: float) -> float:
    """Reflect ``s`` back and forth inside ``[lo, hi]`` so boxes stay in the frame."""
    span = hi - lo
    u = (s - lo) % (2 * span)
    return lo + (u if u <= span else 2 * span - u)


class FakeDetector:
    """Two players walking and a ball moving right; no model involved.

    The frame number is recovered from the call order, not the pixels (mp4v is
    lossy), so frames must be fed in order from frame 0. Use one instance per
    video.
    """

    def __init__(self) -> None:
        self.seen = 0

    def predict_frames(self, frames: list[np.ndarray]) -> list[FrameDetections]:
        out = []
        for frame in frames:
            i = self.seen
            self.seen += 1
            height, width = frame.shape[:2]
            dets = [
                self._det(
                    "player", _bounce(0.10 + 0.002 * i, 0.05, 0.9), 0.5, 0.04, 0.12, 0.9
                ),
                self._det(
                    "player", _bounce(0.60 - 0.002 * i, 0.05, 0.9), 0.5, 0.04, 0.12, 0.8
                ),
            ]
            if i not in GAP:
                dets.append(
                    self._det(
                        "in_play_ball",
                        _bounce(0.1 + 0.02 * i, 0.1, 0.9),
                        0.3,
                        0.012,
                        0.02,
                        0.85,
                    )
                )
            if i == STRAY_FRAME:
                dets.append(self._det("in_play_ball", 0.9, 0.9, 0.012, 0.02, 0.3))
            out.append(
                FrameDetections(
                    uri=Path(f"f{i}"), width=width, height=height, detections=dets
                )
            )
        return out

    @staticmethod
    def _det(label, x, y, w, h, conf):
        return ObjectDetection(label=label, confidence=conf, x=x, y=y, w=w, h=h)
