"""BallTracker protocol — the common interface all bake-off methods implement.

Every competing method (ft-ztw, ft-xps, ft-1d9, ft-76b) must satisfy this
protocol so the shared harness can score them apples-to-apples.

BBox convention: (x, y, w, h) in NORMALISED coordinates [0..1], where (x, y)
is the top-left corner. This matches the rest of footy-track's bbox convention.
"""

from typing import Protocol

import numpy as np

from footy_track.ball_eval.dataset import BBox


class BallTracker(Protocol):
    """Single-object tracker that follows the ball across video frames.

    Implementors MUST:
    - Accept ``prev_bbox=None`` on the first call (cold start) or when
      re-acquiring after a tracking failure.
    - Return ``None`` when the ball is lost / not detectable.
    - Be stateful — hold Kalman state, ROI history, etc. between calls.
    - Be CPU/MPS-compatible (no hard CUDA dependency). GPU use is optional
      and measured by the harness via ``torch.cuda.max_memory_allocated()``.

    One BallTracker instance is created per eval clip. The harness calls
    ``track()`` once per frame in forward temporal order.

    Frame convention:
    - ``frame`` is a uint8 RGB numpy array, shape (H, W, 3).
    - ``prev_bbox`` is normalised (x, y, w, h) or None.
    - Return value is normalised (x, y, w, h) or None if ball lost.
    """

    def track(
        self,
        prev_bbox: BBox | None,
        frame: np.ndarray,
    ) -> BBox | None:
        """Locate the ball in *frame* given the previous position.

        Args:
            prev_bbox: Normalised (x, y, w, h) of the ball in the previous
                frame, or None on first call or after losing the ball.
            frame: uint8 RGB array (H, W, 3).

        Returns:
            Normalised (x, y, w, h) if ball found, else None.
        """
        ...

    def reset(self) -> None:
        """Reset all internal state. Called between eval clips."""
        ...
