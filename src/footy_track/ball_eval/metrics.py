"""Scoring metrics for ball-tracking evaluation.

All IoU calculations operate on normalised (x, y, w, h) bboxes [0..1].

Metrics per clip:
  - per_frame_iou: mean IoU across frames where both GT and prediction exist
  - recall_pct: % of ball-present frames where tracker returned a bbox
  - precision_pct: % of tracker predictions that have a GT bbox (anti-hallucination)
  - tracking_failures: number of times the tracker dropped the ball (GT present, pred None)
  - occlusion_recovery_rate: % of post-occlusion frames where tracking resumed within 3 frames
  - fps: tracker throughput (inference-only, not data loading)
  - peak_vram_mb: peak VRAM during inference (0 if no GPU)
  - effective_resolution: median resolution of the crop fed to the tracker
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class FramePrediction:
    """One frame's tracker output plus ground truth."""

    frame_index: int
    gt_bbox: tuple[float, float, float, float] | None
    pred_bbox: tuple[float, float, float, float] | None
    iou: float | None  # None if either bbox is absent
    inference_time_s: float = 0.0
    crop_height: int | None = None  # pixel height of crop fed to tracker


@dataclasses.dataclass
class ClipMetrics:
    """Aggregate metrics for one clip."""

    clip_name: str
    total_frames: int
    ball_present_frames: int

    mean_iou: float
    median_iou: float

    recall_pct: float
    precision_pct: float

    tracking_failures: int
    occlusion_recovery_rate: float

    fps: float
    peak_vram_mb: float

    effective_resolution_px: int | None

    frame_predictions: list[FramePrediction] = dataclasses.field(
        default_factory=list, repr=False
    )

    def to_dict(self) -> dict:
        return {
            "clip": self.clip_name,
            "frames": self.total_frames,
            "ball_present": self.ball_present_frames,
            "mean_iou": round(self.mean_iou, 3),
            "median_iou": round(self.median_iou, 3),
            "recall_pct": round(self.recall_pct, 1),
            "precision_pct": round(self.precision_pct, 1),
            "tracking_failures": self.tracking_failures,
            "occlusion_recovery_rate": round(self.occlusion_recovery_rate, 1),
            "fps": round(self.fps, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "effective_resolution_px": self.effective_resolution_px,
        }


@dataclasses.dataclass
class MethodResult:
    """All clips scored for one tracker method."""

    method_name: str
    clip_metrics: list[ClipMetrics]

    @property
    def mean_iou(self) -> float:
        ious = [c.mean_iou for c in self.clip_metrics if c.ball_present_frames > 0]
        return sum(ious) / len(ious) if ious else 0.0

    @property
    def mean_recall(self) -> float:
        recalls = [c.recall_pct for c in self.clip_metrics]
        return sum(recalls) / len(recalls) if recalls else 0.0

    @property
    def mean_fps(self) -> float:
        fpss = [c.fps for c in self.clip_metrics]
        return sum(fpss) / len(fpss) if fpss else 0.0

    @property
    def peak_vram_mb(self) -> float:
        return max((c.peak_vram_mb for c in self.clip_metrics), default=0.0)

    @property
    def total_failures(self) -> int:
        return sum(c.tracking_failures for c in self.clip_metrics)

    @property
    def mean_occlusion_recovery(self) -> float:
        rates = [c.occlusion_recovery_rate for c in self.clip_metrics]
        return sum(rates) / len(rates) if rates else 0.0

    def table(self) -> str:
        """Return a formatted comparison table."""
        import io  # noqa: PLC0415

        buf = io.StringIO()
        header = (
            f"{'Method':<24} {'IoU':>6} {'Recall%':>8} {'OccRecov%':>10} "
            f"{'Failures':>9} {'FPS':>6} {'VRAM MB':>8} {'Res px':>7}"
        )
        buf.write(header + "\n")
        buf.write("-" * len(header) + "\n")

        row = (
            f"{self.method_name:<24} {self.mean_iou:>6.3f} {self.mean_recall:>8.1f} "
            f"{self.mean_occlusion_recovery:>10.1f} {self.total_failures:>9} "
            f"{self.mean_fps:>6.1f} {self.peak_vram_mb:>8.1f} "
        )
        eff_res = (
            self.clip_metrics[0].effective_resolution_px if self.clip_metrics else None
        )
        row += f"{'full' if eff_res is None else str(eff_res):>7}"
        buf.write(row + "\n")

        if len(self.clip_metrics) > 1:
            buf.write("\nPer-clip breakdown:\n")
            for cm in self.clip_metrics:
                buf.write(
                    f"  {cm.clip_name}: IoU={cm.mean_iou:.3f}, "
                    f"recall={cm.recall_pct:.1f}%, "
                    f"failures={cm.tracking_failures}, "
                    f"fps={cm.fps:.1f}\n"
                )
        return buf.getvalue()

    def to_dict(self) -> dict:
        return {
            "method": self.method_name,
            "aggregate": {
                "mean_iou": round(self.mean_iou, 3),
                "mean_recall_pct": round(self.mean_recall, 1),
                "mean_fps": round(self.mean_fps, 1),
                "peak_vram_mb": round(self.peak_vram_mb, 1),
                "total_failures": self.total_failures,
                "mean_occlusion_recovery_pct": round(self.mean_occlusion_recovery, 1),
            },
            "clips": [c.to_dict() for c in self.clip_metrics],
        }


def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two normalised (x, y, w, h) bboxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def compute_clip_metrics(
    clip_name: str,
    total_frames: int,
    ball_present_frames: int,
    frame_preds: list[FramePrediction],
    total_inference_s: float,
    peak_vram_mb: float,
    occlusion_frame_indices: list[int],
) -> ClipMetrics:
    """Aggregate FramePredictions into ClipMetrics."""
    import statistics  # noqa: PLC0415

    ious = [fp.iou for fp in frame_preds if fp.iou is not None]
    mean_iou = statistics.mean(ious) if ious else 0.0
    median_iou = statistics.median(ious) if ious else 0.0

    gt_present = [fp for fp in frame_preds if fp.gt_bbox is not None]
    recall_hits = [fp for fp in gt_present if fp.pred_bbox is not None]
    recall_pct = 100.0 * len(recall_hits) / len(gt_present) if gt_present else 0.0

    pred_present = [fp for fp in frame_preds if fp.pred_bbox is not None]
    precision_hits = [fp for fp in pred_present if fp.gt_bbox is not None]
    precision_pct = (
        100.0 * len(precision_hits) / len(pred_present) if pred_present else 0.0
    )

    failures = len(gt_present) - len(recall_hits)

    fps = len(frame_preds) / total_inference_s if total_inference_s > 0 else 0.0

    crop_heights = [fp.crop_height for fp in frame_preds if fp.crop_height is not None]
    effective_res = int(statistics.median(crop_heights)) if crop_heights else None

    pred_by_idx = {fp.frame_index: fp for fp in frame_preds}
    recovery_count = 0
    occlusion_events = 0
    for occ_idx in occlusion_frame_indices:
        occlusion_events += 1
        recovered = False
        for offset in range(1, 4):
            check_idx = occ_idx + offset
            fp = pred_by_idx.get(check_idx)
            if fp is not None and fp.pred_bbox is not None:
                recovered = True
                break
        if recovered:
            recovery_count += 1

    recovery_rate = (
        100.0 * recovery_count / occlusion_events if occlusion_events > 0 else 100.0
    )

    return ClipMetrics(
        clip_name=clip_name,
        total_frames=total_frames,
        ball_present_frames=ball_present_frames,
        mean_iou=mean_iou,
        median_iou=median_iou,
        recall_pct=recall_pct,
        precision_pct=precision_pct,
        tracking_failures=failures,
        occlusion_recovery_rate=recovery_rate,
        fps=fps,
        peak_vram_mb=peak_vram_mb,
        effective_resolution_px=effective_res,
        frame_predictions=frame_preds,
    )
