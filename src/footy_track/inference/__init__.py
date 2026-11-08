"""End-to-end inference: video in, detections + player tracklets + ball track out."""

from footy_track.inference.pipeline import (
    FrameDetector,
    InferenceConfig,
    run_inference,
    select_device,
)

__all__ = ["FrameDetector", "InferenceConfig", "run_inference", "select_device"]
