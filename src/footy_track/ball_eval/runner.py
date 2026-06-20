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
import pathlib
import time
from typing import Any

import cv2

from footy_track.ball_eval.dataset import BBox, EvalClip, EvalDataset
from footy_track.ball_eval.interface import BallTracker
from footy_track.ball_eval.metrics import (
    ClipMetrics,
    FramePrediction,
    MethodResult,
    bbox_center,
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
                f"    Ctr%={cm.center_within_radius_pct:.1f}, "
                f"CtrPx={cm.mean_center_dist_px:.1f}, "
                f"IoU={cm.mean_iou:.3f}, recall={cm.recall_pct:.1f}%, "
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
    frame_h: int = 1080
    frame_w: int = 1920

    _reset_vram_counter()

    for frame_idx, frame_rgb in clip.iter_frames():
        frame_h, frame_w = frame_rgb.shape[:2]
        gt_label = clip.labels.get(frame_idx)
        gt_bbox = gt_label.bbox if gt_label is not None else None
        gt_center = gt_label.center if gt_label is not None else None

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
                gt_center_norm=gt_center,
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
        frame_size=(frame_h, frame_w),
    )


def compare_methods(
    results: list[MethodResult],
    sort_by: str = "center_within_radius_pct",
    overlay_output_dir: str | pathlib.Path | None = None,
    dataset: EvalDataset | None = None,
) -> str:
    """Render a multi-method comparison table.

    Args:
        results: List of MethodResult (one per competing method).
        sort_by: Sort column. One of: "center_within_radius_pct", "mean_iou",
                 "mean_recall", "mean_fps", "total_failures", "peak_vram_mb".
        overlay_output_dir: If set, render per-method overlay videos to this
            directory. Requires ``dataset`` to also be provided.
        dataset: The EvalDataset used for scoring — needed to render overlays.

    Returns:
        Formatted ASCII table string.
    """
    _sort_key = {
        "center_within_radius_pct": lambda r: -r.mean_center_within_radius,
        "mean_iou": lambda r: -r.mean_iou,
        "mean_recall": lambda r: -r.mean_recall,
        "mean_fps": lambda r: -r.mean_fps,
        "total_failures": lambda r: r.total_failures,
        "peak_vram_mb": lambda r: r.peak_vram_mb,
    }
    key_fn = _sort_key.get(sort_by, _sort_key["center_within_radius_pct"])
    sorted_results = sorted(results, key=key_fn)

    buf = io.StringIO()
    header = (
        f"{'Method':<24} {'Ctr%':>6} {'CtrPx':>7} {'IoU':>6} {'Recall%':>8} "
        f"{'CatFail%':>9} {'OccRec%':>8} {'Failures':>9} {'FPS':>6}"
    )
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")

    for r in sorted_results:
        row = (
            f"{r.method_name:<24} {r.mean_center_within_radius:>6.1f} "
            f"{r.mean_center_dist_px:>7.1f} {r.mean_iou:>6.3f} "
            f"{r.mean_recall:>8.1f} {r.mean_catastrophic_failure_rate:>9.1f} "
            f"{r.mean_occlusion_recovery:>8.1f} {r.total_failures:>9} "
            f"{r.mean_fps:>6.1f}"
        )
        buf.write(row + "\n")

    if overlay_output_dir is not None and dataset is not None:
        out_dir = pathlib.Path(overlay_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        buf.write(f"\nOverlay videos → {out_dir}/\n")
        for result in sorted_results:
            for clip in dataset:
                clip_result = next(
                    (cm for cm in result.clip_metrics if cm.clip_name == clip.name),
                    None,
                )
                if clip_result is None:
                    continue
                out_path = out_dir / f"{result.method_name}__{clip.name}.mp4"
                render_overlay_video(clip, clip_result, out_path)
                buf.write(f"  {out_path.name}\n")

    return buf.getvalue()


def render_overlay_video(
    clip: EvalClip,
    clip_metrics: ClipMetrics,
    output_path: str | pathlib.Path,
) -> None:
    """Render a ~30s overlay video showing GT and tracker predictions.

    Draws on each frame:
      - Green circle: GT ball center
      - Red circle: predicted ball center
      - Blue box: predicted bbox (if available)
      - Per-frame center distance in top-left corner

    Args:
        clip: The eval clip providing frames.
        clip_metrics: Scored predictions from run_benchmark for this clip.
        output_path: Where to write the MP4.
    """
    output_path = pathlib.Path(output_path)
    pred_by_frame = {fp.frame_index: fp for fp in clip_metrics.frame_predictions}

    # Open source video to get FPS and frame size
    cap = cv2.VideoCapture(str(clip.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for frame_idx, frame_rgb in clip.iter_frames():
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            fp = pred_by_frame.get(frame_idx)

            if fp is not None:
                # Draw GT center (green)
                gt_ctr = fp.gt_center_norm
                if gt_ctr is None and fp.gt_bbox is not None:
                    gt_ctr = bbox_center(fp.gt_bbox)
                if gt_ctr is not None:
                    gx = int(gt_ctr[0] * width)
                    gy = int(gt_ctr[1] * height)
                    cv2.circle(bgr, (gx, gy), 8, (0, 255, 0), 2)

                # Draw predicted bbox (blue) and center (red)
                if fp.pred_bbox is not None:
                    x, y, w, h = fp.pred_bbox
                    x1, y1 = int(x * width), int(y * height)
                    x2, y2 = int((x + w) * width), int((y + h) * height)
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    pred_ctr = bbox_center(fp.pred_bbox)
                    px = int(pred_ctr[0] * width)
                    py = int(pred_ctr[1] * height)
                    cv2.circle(bgr, (px, py), 6, (0, 0, 255), 2)

                # Overlay distance text
                if fp.center_dist_px is not None:
                    label = f"dist={fp.center_dist_px:.0f}px"
                    cv2.putText(
                        bgr,
                        label,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            writer.write(bgr)
    finally:
        writer.release()


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
