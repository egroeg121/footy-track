"""Benchmark runner: scores any BallTracker against an EvalDataset.

Usage::

    from footy_track.ball_eval import run_benchmark, EvalDataset

    dataset = EvalDataset.from_dir("eval_data/clips/")
    result = run_benchmark(my_tracker, dataset, method_name="my_method")
    print(result.table())

The runner:
  1. Iterates through each clip in the dataset.
  2. Calls tracker.reset() between clips.
  3. For each frame, calls tracker.track(prev_bbox, frame).
  4. Records inference time, VRAM, and compares to ground truth.
  5. Returns a MethodResult with per-clip and aggregate metrics.
"""

from __future__ import annotations

import io
import time
from typing import Any

from footy_track.ball_eval.dataset import BBox, EvalClip, EvalDataset
from footy_track.ball_eval.interface import BallTracker
from footy_track.ball_eval.metrics import (
    ClipMetrics,
    FramePrediction,
    MethodResult,
    bbox_iou,
    compute_clip_metrics,
)


def run_benchmark(
    tracker: BallTracker,
    dataset: EvalDataset,
    method_name: str,
    verbose: bool = True,
) -> MethodResult:
    """Score *tracker* against every clip in *dataset*.

    Args:
        tracker: Any object implementing the BallTracker protocol.
        dataset: Labelled eval clips.
        method_name: Label for this method in the output table.
        verbose: Print per-clip progress to stdout.

    Returns:
        MethodResult with per-clip ClipMetrics and aggregate scores.
    """
    clip_metrics: list[ClipMetrics] = []

    for clip in dataset:
        if verbose:
            print(f"  Evaluating {method_name} on clip '{clip.name}' ...")
        cm = _score_clip(tracker, clip, verbose=verbose)
        clip_metrics.append(cm)
        if verbose:
            print(
                f"    IoU={cm.mean_iou:.3f}, recall={cm.recall_pct:.1f}%, "
                f"fps={cm.fps:.1f}, failures={cm.tracking_failures}"
            )

    return MethodResult(method_name=method_name, clip_metrics=clip_metrics)


def _score_clip(tracker: BallTracker, clip: EvalClip, verbose: bool) -> ClipMetrics:
    """Run the tracker through one clip and return metrics."""
    tracker.reset()

    frame_preds: list[FramePrediction] = []
    prev_bbox: BBox | None = None
    total_inference_s = 0.0
    peak_vram_mb = 0.0

    _reset_vram_counter()

    for frame_idx, frame_rgb in clip.iter_frames():
        gt_label = clip.labels.get(frame_idx)
        gt_bbox = gt_label.bbox if gt_label is not None else None

        t0 = time.perf_counter()
        pred_bbox = tracker.track(prev_bbox, frame_rgb)
        elapsed = time.perf_counter() - t0

        total_inference_s += elapsed
        peak_vram_mb = max(peak_vram_mb, _current_vram_mb())

        iou: float | None = None
        if gt_bbox is not None and pred_bbox is not None:
            iou = bbox_iou(gt_bbox, pred_bbox)

        # Infer crop height from the frame if the tracker exposes it.
        crop_height = _get_crop_height(tracker)

        frame_preds.append(
            FramePrediction(
                frame_index=frame_idx,
                gt_bbox=gt_bbox,
                pred_bbox=pred_bbox,
                iou=iou,
                inference_time_s=elapsed,
                crop_height=crop_height,
            )
        )

        # Feed prediction back as next frame's prev_bbox (simulates real tracking).
        # If tracker lost the ball, keep prev_bbox so it can re-seed from last known.
        if pred_bbox is not None:
            prev_bbox = pred_bbox

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


def compare_methods(
    results: list[MethodResult],
    sort_by: str = "mean_iou",
) -> str:
    """Render a multi-method comparison table.

    Args:
        results: List of MethodResult (one per competing method).
        sort_by: Sort column. One of: "mean_iou", "mean_recall", "mean_fps",
                 "total_failures", "peak_vram_mb".

    Returns:
        Formatted ASCII table string.
    """
    _sort_key = {
        "mean_iou": lambda r: -r.mean_iou,
        "mean_recall": lambda r: -r.mean_recall,
        "mean_fps": lambda r: -r.mean_fps,
        "total_failures": lambda r: r.total_failures,
        "peak_vram_mb": lambda r: r.peak_vram_mb,
    }
    key_fn = _sort_key.get(sort_by, _sort_key["mean_iou"])
    sorted_results = sorted(results, key=key_fn)

    buf = io.StringIO()
    header = (
        f"{'Method':<24} {'IoU':>6} {'Recall%':>8} {'OccRec%':>8} "
        f"{'Failures':>9} {'FPS':>6} {'VRAM MB':>8} {'Res px':>7}"
    )
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")

    for r in sorted_results:
        eff_res = r.clip_metrics[0].effective_resolution_px if r.clip_metrics else None
        res_str = "full" if eff_res is None else str(eff_res)
        row = (
            f"{r.method_name:<24} {r.mean_iou:>6.3f} {r.mean_recall:>8.1f} "
            f"{r.mean_occlusion_recovery:>8.1f} {r.total_failures:>9} "
            f"{r.mean_fps:>6.1f} {r.peak_vram_mb:>8.1f} {res_str:>7}"
        )
        buf.write(row + "\n")

    return buf.getvalue()


# --------------------------------------------------------------------------- #
# VRAM helpers (gracefully degrade if torch not available)                    #
# --------------------------------------------------------------------------- #


try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _reset_vram_counter() -> None:
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        _torch.cuda.reset_peak_memory_stats()


def _current_vram_mb() -> float:
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        return _torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _get_crop_height(tracker: Any) -> int | None:
    """If the tracker exposes the last crop height, return it."""
    return getattr(tracker, "_last_crop_height", None)
