"""Benchmark runner: drives any BallTracker implementation against EvalDataset.

Usage::

    from footy_track.ball_eval import EvalDataset, run_benchmark
    from footy_track.ball_trackers.sot_vittrack import VitTrackSOT

    dataset = EvalDataset.from_dir("eval_data/clips/")
    tracker = VitTrackSOT()
    result = run_benchmark(tracker, dataset, method_name="vittrack-sot")
    print(result.table())
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from footy_track.ball_eval.dataset import BBox, EvalDataset
from footy_track.ball_eval.metrics import (
    ClipMetrics,
    FramePrediction,
    MethodResult,
    bbox_iou,
    compute_clip_metrics,
)

if TYPE_CHECKING:
    from footy_track.ball_eval.interface import BallTracker


def run_benchmark(
    tracker: BallTracker,
    dataset: EvalDataset,
    method_name: str,
    verbose: bool = True,
) -> MethodResult:
    """Score *tracker* against every clip in *dataset*.

    Args:
        tracker: Any object satisfying the BallTracker protocol.
        dataset: Labelled eval clips.
        method_name: Human-readable name for the method (shown in the table).
        verbose: Print per-clip progress to stdout.

    Returns:
        MethodResult with per-clip and aggregate metrics.
    """
    clip_metrics: list[ClipMetrics] = []

    for clip in dataset:
        if verbose:
            print(f"  [{method_name}] evaluating clip: {clip.name} ...")

        tracker.reset()
        metrics = _score_clip(tracker, clip, verbose=verbose)
        clip_metrics.append(metrics)

        if verbose:
            print(
                f"    IoU={metrics.mean_iou:.3f}  recall={metrics.recall_pct:.1f}%  "
                f"fps={metrics.fps:.1f}  failures={metrics.tracking_failures}"
            )

    return MethodResult(method_name=method_name, clip_metrics=clip_metrics)


def _score_clip(tracker: BallTracker, clip, verbose: bool = False) -> ClipMetrics:
    """Run tracker over one clip and aggregate metrics."""
    frame_preds: list[FramePrediction] = []
    total_inference_s = 0.0

    _reset_vram_counter()

    prev_bbox: BBox | None = None

    for frame_idx, frame_rgb in clip.iter_frames():
        gt_label = clip.labels.get(frame_idx)
        gt_bbox = gt_label.bbox if gt_label is not None else None

        t0 = time.perf_counter()
        pred_bbox = tracker.track(prev_bbox, frame_rgb)
        elapsed = time.perf_counter() - t0

        total_inference_s += elapsed

        iou: float | None = None
        if gt_bbox is not None and pred_bbox is not None:
            iou = bbox_iou(gt_bbox, pred_bbox)

        crop_h = getattr(tracker, "_last_crop_height", None)

        frame_preds.append(
            FramePrediction(
                frame_index=frame_idx,
                gt_bbox=gt_bbox,
                pred_bbox=pred_bbox,
                iou=iou,
                inference_time_s=elapsed,
                crop_height=crop_h,
            )
        )

        # Feed back the prediction as prev_bbox; if lost, pass None to allow re-init.
        prev_bbox = pred_bbox

    peak_vram_mb = _read_peak_vram_mb()

    occlusion_indices = clip.frames_with_tag("occlusion")

    return compute_clip_metrics(
        clip_name=clip.name,
        total_frames=clip.total_frames,
        ball_present_frames=clip.ball_present_count(),
        frame_preds=frame_preds,
        total_inference_s=total_inference_s,
        peak_vram_mb=peak_vram_mb,
        occlusion_frame_indices=occlusion_indices,
    )


def _reset_vram_counter() -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _read_peak_vram_mb() -> float:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0
