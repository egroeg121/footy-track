"""Ball-tracking evaluation: GT dataset schema and loader."""

from footy_track.ball_eval.dataset import (
    BBox,
    Center,
    EvalClip,
    EvalDataset,
    FrameLabel,
    write_labels,
)

__all__ = [
    "BBox",
    "Center",
    "EvalClip",
    "EvalDataset",
    "FrameLabel",
    "write_labels",
]
