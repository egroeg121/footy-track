"""Ball tracker implementations for the bake-off evaluation harness.

Each module exposes a class implementing the BallTracker protocol from
footy_track.ball_eval.interface.

Available trackers:
  - VitTrackSOT: ViT-based single-object tracker (ONNX, method A, ft-ztw)
  - DetectionBallTracker: tracks the detector's ball boxes (no model; used by
    ``footy_track.inference``). Takes candidates, not pixels, so it is not a
    BallTracker protocol implementation.
"""

from footy_track.ball_trackers.detection_tracker import DetectionBallTracker
from footy_track.ball_trackers.sot_vittrack import VitTrackSOT

__all__ = ["DetectionBallTracker", "VitTrackSOT"]
