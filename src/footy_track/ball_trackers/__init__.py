"""Ball tracker implementations for the bake-off evaluation harness.

Each module exposes a class implementing the BallTracker protocol from
footy_track.ball_eval.interface.

Available trackers:
  - VitTrackSOT: ViT-based single-object tracker (ONNX, method A, ft-ztw)
"""

from footy_track.ball_trackers.sot_vittrack import VitTrackSOT

__all__ = ["VitTrackSOT"]
