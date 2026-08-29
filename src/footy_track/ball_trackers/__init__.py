"""Ball tracker implementations for the bake-off evaluation harness.

Each module exposes a class implementing the BallTracker protocol from
footy_track.ball_eval.interface.

Available trackers (bake-off ft-5hd methods A/B/C):
  - VitTrackSOT: ViT-based single-object tracker (ONNX, method A, ft-ztw)
  - Sam2BallTracker: box-prompted SAM2 image predictor (method B, ft-xps)
  - RoiYoloTracker: tiny YOLO on a Kalman-predicted ROI crop (method C, ft-1d9)

Sam2BallTracker and RoiYoloTracker require their respective extras
(ultralytics + torch) to be installed; importing this package does not
require GPU or model weights — those are lazily loaded on first ``track()``
call (Sam2BallTracker) or at construction time from a local checkpoint
(RoiYoloTracker).
"""

from footy_track.ball_trackers.roi_yolo import RoiYoloTracker
from footy_track.ball_trackers.sam2_tracker import Sam2BallTracker
from footy_track.ball_trackers.sot_vittrack import VitTrackSOT

__all__ = ["RoiYoloTracker", "Sam2BallTracker", "VitTrackSOT"]
