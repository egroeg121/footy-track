"""Ball-tracking bake-off: shared benchmark harness.

This package defines the common interface and evaluation framework used by
all competing ball-tracking methods (bake-off branches ft-ztw, ft-xps,
ft-1d9, ft-76b). Every method implements the same BallTracker protocol and
is scored against the same EvalDataset to enable apples-to-apples comparison.

Typical usage::

    from footy_track.ball_eval import BallTracker, EvalDataset, run_benchmark

    class MyTracker:
        def track(self, prev_bbox, frame):
            ...

    dataset = EvalDataset.from_dir("eval_data/")
    results = run_benchmark(MyTracker(), dataset, method_name="my_method")
    print(results.table())
"""

from footy_track.ball_eval.dataset import (
    BBox,
    Center,
    EvalClip,
    EvalDataset,
    FrameLabel,
    write_labels,
)
from footy_track.ball_eval.interface import BallTracker
from footy_track.ball_eval.metrics import ClipMetrics, MethodResult
from footy_track.ball_eval.runner import (
    compare_methods,
    render_overlay_video,
    run_benchmark,
)

__all__ = [
    "BallTracker",
    "BBox",
    "ClipMetrics",
    "EvalClip",
    "EvalDataset",
    "FrameLabel",
    "MethodResult",
    "run_benchmark",
]
