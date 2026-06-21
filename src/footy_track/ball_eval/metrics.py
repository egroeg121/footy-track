"""Scoring metrics for ball-tracking evaluation.

All IoU calculations operate on normalised (x, y, w, h) bboxes [0..1].
Center distances are in normalised coordinates [0..1] and in pixels.

Metrics per clip:
  - center_within_radius_pct: PRIMARY — % of labelled frames where predicted
    center is within one ball-radius (or FALLBACK_RADIUS_PX pixels) of GT center
  - mean_center_dist_px: mean pixel distance between predicted and GT centers
  - per_frame_iou: mean IoU across frames where both GT and prediction exist
    (secondary — unreliable for tiny balls)
  - recall_pct: % of ball-present frames where tracker returned a bbox
  - precision_pct: % of tracker predictions that have a GT bbox (anti-hallucination)
  - tracking_failures: number of times the tracker dropped the ball (GT present, pred None)
  - max_track_streak: longest consecutive run of correctly-tracked frames (continuity)
  - catastrophic_failure_rate: % of tracker predictions where center error > CATASTROPHIC_PX
  - occlusion_recovery_rate: % of post-occlusion frames where tracking resumed within 3 frames
  - fps: tracker throughput (inference-only, not data loading)
  - peak_vram_mb: peak VRAM during inference (0 if no GPU)
  - effective_resolution: median resolution of the crop fed to the tracker
"""

from __future__ import annotations

import dataclasses
import io
import math
import statistics

# Fallback ball-radius in pixels when GT only has a center (no bbox).
# Roughly 10px radius at 1080p — adjust if typical ball size differs.
FALLBACK_RADIUS_PX: int = 10

# A prediction is a "catastrophic failure" if its center is > this many pixels
# from the GT center. Catches "tracking the wrong object entirely."
CATASTROPHIC_DIST_PX: int = 50


@dataclasses.dataclass
class FramePrediction:
    """One frame's tracker output plus ground truth."""

    frame_index: int
    gt_bbox: tuple[float, float, float, float] | None
    pred_bbox: tuple[float, float, float, float] | None
    iou: float | None  # None if either bbox is absent
    inference_time_s: float = 0.0
    crop_height: int | None = None  # pixel height of crop fed to tracker
    # Center-distance fields (filled by compute_clip_metrics)
    gt_center_norm: tuple[float, float] | None = None
    pred_center_norm: tuple[float, float] | None = None
    center_dist_px: float | None = None  # pixel distance at frame resolution
    # True when prev_bbox was seeded from GT for this frame
    seeded_from_gt: bool = False


@dataclasses.dataclass
class ClipMetrics:
    """Aggregate metrics for one clip."""

    clip_name: str
    total_frames: int
    ball_present_frames: int

    # PRIMARY metric: % frames where predicted center is within ball-radius of GT
    center_within_radius_pct: float
    # Mean pixel distance between predicted and GT centers (lower is better)
    mean_center_dist_px: float

    # Per-frame IoU (only frames where GT and pred both exist)
    # Secondary for small balls — IoU is brittle when balls are tiny
    mean_iou: float
    median_iou: float

    # Recall: GT present and tracker returned a bbox
    recall_pct: float

    # Precision: tracker returned a bbox and GT present
    precision_pct: float

    # Failure/recovery
    tracking_failures: int  # GT present, pred is None
    # Longest consecutive run of correctly-tracked frames (center within radius)
    max_track_streak: int
    # % of tracker predictions where center error > CATASTROPHIC_DIST_PX
    catastrophic_failure_rate: float
    occlusion_recovery_rate: (
        float  # % of occlusion events where tracking resumed ≤3 frames later
    )

    # GT-seeding statistics (how often the harness had to inject GT)
    seeded_frames: int  # frames where tracker was given GT seed as prev_bbox
    reseed_count: (
        int  # number of distinct reseed events (first seed + re-seeds after failure)
    )

    # Speed / resource
    fps: float
    peak_vram_mb: float

    # Effective resolution: median crop height (None → full frame)
    effective_resolution_px: int | None

    # Per-frame details (not printed in table but available for debugging)
    frame_predictions: list[FramePrediction] = dataclasses.field(
        default_factory=list, repr=False
    )

    def to_dict(self) -> dict:
        return {
            "clip": self.clip_name,
            "frames": self.total_frames,
            "ball_present": self.ball_present_frames,
            "center_within_radius_pct": round(self.center_within_radius_pct, 1),
            "mean_center_dist_px": round(self.mean_center_dist_px, 1),
            "mean_iou": round(self.mean_iou, 3),
            "median_iou": round(self.median_iou, 3),
            "recall_pct": round(self.recall_pct, 1),
            "precision_pct": round(self.precision_pct, 1),
            "tracking_failures": self.tracking_failures,
            "max_track_streak": self.max_track_streak,
            "catastrophic_failure_rate": round(self.catastrophic_failure_rate, 1),
            "occlusion_recovery_rate": round(self.occlusion_recovery_rate, 1),
            "fps": round(self.fps, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "effective_resolution_px": self.effective_resolution_px,
            "seeded_frames": self.seeded_frames,
            "reseed_count": self.reseed_count,
        }


@dataclasses.dataclass
class MethodResult:
    """All clips scored for one tracker method."""

    method_name: str
    clip_metrics: list[ClipMetrics]

    @property
    def mean_center_within_radius(self) -> float:
        vals = [
            c.center_within_radius_pct
            for c in self.clip_metrics
            if c.ball_present_frames > 0
        ]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_center_dist_px(self) -> float:
        vals = [
            c.mean_center_dist_px
            for c in self.clip_metrics
            if c.ball_present_frames > 0
        ]
        return sum(vals) / len(vals) if vals else 0.0

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

    @property
    def mean_catastrophic_failure_rate(self) -> float:
        rates = [c.catastrophic_failure_rate for c in self.clip_metrics]
        return sum(rates) / len(rates) if rates else 0.0

    def table(self) -> str:
        """Return a formatted comparison table."""
        buf = io.StringIO()
        header = (
            f"{'Method':<24} {'Ctr%':>6} {'CtrPx':>7} {'IoU':>6} {'Recall%':>8} "
            f"{'CatFail%':>9} {'OccRec%':>8} {'Failures':>9} {'FPS':>6}"
        )
        buf.write(header + "\n")
        buf.write("-" * len(header) + "\n")

        row = (
            f"{self.method_name:<24} {self.mean_center_within_radius:>6.1f} "
            f"{self.mean_center_dist_px:>7.1f} {self.mean_iou:>6.3f} "
            f"{self.mean_recall:>8.1f} {self.mean_catastrophic_failure_rate:>9.1f} "
            f"{self.mean_occlusion_recovery:>8.1f} {self.total_failures:>9} "
            f"{self.mean_fps:>6.1f}"
        )
        buf.write(row + "\n")

        # Per-clip breakdown
        if len(self.clip_metrics) > 1:
            buf.write("\nPer-clip breakdown:\n")
            for cm in self.clip_metrics:
                buf.write(
                    f"  {cm.clip_name}: Ctr%={cm.center_within_radius_pct:.1f}, "
                    f"CtrPx={cm.mean_center_dist_px:.1f}, IoU={cm.mean_iou:.3f}, "
                    f"recall={cm.recall_pct:.1f}%, streak={cm.max_track_streak}, "
                    f"failures={cm.tracking_failures}, fps={cm.fps:.1f}\n"
                )
        return buf.getvalue()

    def to_dict(self) -> dict:
        return {
            "method": self.method_name,
            "aggregate": {
                "center_within_radius_pct": round(self.mean_center_within_radius, 1),
                "mean_center_dist_px": round(self.mean_center_dist_px, 1),
                "mean_iou": round(self.mean_iou, 3),
                "mean_recall_pct": round(self.mean_recall, 1),
                "mean_fps": round(self.mean_fps, 1),
                "peak_vram_mb": round(self.peak_vram_mb, 1),
                "total_failures": self.total_failures,
                "mean_catastrophic_failure_rate_pct": round(
                    self.mean_catastrophic_failure_rate, 1
                ),
                "mean_occlusion_recovery_pct": round(self.mean_occlusion_recovery, 1),
            },
            "clips": [c.to_dict() for c in self.clip_metrics],
        }


# --------------------------------------------------------------------------- #
# Metric computation helpers                                                   #
# --------------------------------------------------------------------------- #


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


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the normalised center (cx, cy) of a bbox (x, y, w, h)."""
    return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)


def bbox_radius_px(
    bbox: tuple[float, float, float, float], frame_h: int, frame_w: int
) -> float:
    """Approximate ball radius in pixels from bbox size."""
    w_px = bbox[2] * frame_w
    h_px = bbox[3] * frame_h
    return max(w_px, h_px) / 2


def center_dist_px(
    a: tuple[float, float],
    b: tuple[float, float],
    frame_h: int,
    frame_w: int,
) -> float:
    """Euclidean pixel distance between two normalised centers."""
    dx = (a[0] - b[0]) * frame_w
    dy = (a[1] - b[1]) * frame_h
    return math.sqrt(dx * dx + dy * dy)


def compute_clip_metrics(  # noqa: PLR0912, PLR0915
    clip_name: str,
    total_frames: int,
    ball_present_frames: int,
    frame_preds: list[FramePrediction],
    total_inference_s: float,
    peak_vram_mb: float,
    occlusion_frame_indices: list[int],
    frame_size: tuple[int, int] = (1080, 1920),  # (height, width) for pixel metrics
    seeded_frames: int = 0,
    reseed_count: int = 0,
) -> ClipMetrics:
    """Aggregate FramePredictions into ClipMetrics.

    Args:
        frame_size: (height, width) of video frames — used to convert normalised
            center coordinates to pixel distances. Defaults to 1080p.
    """
    frame_h, frame_w = frame_size

    ious = [fp.iou for fp in frame_preds if fp.iou is not None]
    mean_iou = statistics.mean(ious) if ious else 0.0
    median_iou = statistics.median(ious) if ious else 0.0

    # Recall: GT present, pred returned something
    gt_present = [
        fp
        for fp in frame_preds
        if fp.gt_bbox is not None or fp.gt_center_norm is not None
    ]
    recall_hits = [fp for fp in gt_present if fp.pred_bbox is not None]
    recall_pct = 100.0 * len(recall_hits) / len(gt_present) if gt_present else 0.0

    # Precision: tracker predicted something, GT present
    pred_present = [fp for fp in frame_preds if fp.pred_bbox is not None]
    precision_hits = [
        fp
        for fp in pred_present
        if fp.gt_bbox is not None or fp.gt_center_norm is not None
    ]
    precision_pct = (
        100.0 * len(precision_hits) / len(pred_present) if pred_present else 0.0
    )

    # Tracking failures: GT present but pred absent
    failures = len(gt_present) - len(recall_hits)

    # FPS — exclude data-loading time
    fps = len(frame_preds) / total_inference_s if total_inference_s > 0 else 0.0

    # Effective resolution
    crop_heights = [fp.crop_height for fp in frame_preds if fp.crop_height is not None]
    effective_res = int(statistics.median(crop_heights)) if crop_heights else None

    # Occlusion recovery: after each occlusion tag frame, did tracking resume ≤3 frames later?
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

    # ------------------------------------------------------------------ #
    # Center-distance metrics                                              #
    # ------------------------------------------------------------------ #
    center_dists_px: list[float] = []
    within_radius_hits = 0
    within_radius_total = 0
    catastrophic_count = 0
    catastrophic_total = 0

    # Track continuity: longest streak of within-radius predictions
    current_streak = 0
    max_streak = 0

    for fp in frame_preds:
        # Derive GT center from bbox or explicit center
        gt_ctr: tuple[float, float] | None = fp.gt_center_norm
        if gt_ctr is None and fp.gt_bbox is not None:
            gt_ctr = bbox_center(fp.gt_bbox)

        # Derive pred center from bbox
        pred_ctr: tuple[float, float] | None = None
        if fp.pred_bbox is not None:
            pred_ctr = bbox_center(fp.pred_bbox)

        if gt_ctr is not None and pred_ctr is not None:
            dist = center_dist_px(gt_ctr, pred_ctr, frame_h, frame_w)
            center_dists_px.append(dist)
            fp.center_dist_px = dist
            fp.gt_center_norm = gt_ctr
            fp.pred_center_norm = pred_ctr

            # Radius: derive from GT bbox if available, else fallback
            if fp.gt_bbox is not None:
                radius_px = bbox_radius_px(fp.gt_bbox, frame_h, frame_w)
                radius_px = max(radius_px, FALLBACK_RADIUS_PX)
            else:
                radius_px = float(FALLBACK_RADIUS_PX)

            within_radius_total += 1
            if dist <= radius_px:
                within_radius_hits += 1
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

            catastrophic_total += 1
            if dist > CATASTROPHIC_DIST_PX:
                catastrophic_count += 1
        else:
            current_streak = 0

    center_within_radius_pct = (
        100.0 * within_radius_hits / within_radius_total
        if within_radius_total > 0
        else 0.0
    )
    mean_center_dist = statistics.mean(center_dists_px) if center_dists_px else 0.0
    catastrophic_rate = (
        100.0 * catastrophic_count / catastrophic_total
        if catastrophic_total > 0
        else 0.0
    )

    return ClipMetrics(
        clip_name=clip_name,
        total_frames=total_frames,
        ball_present_frames=ball_present_frames,
        center_within_radius_pct=center_within_radius_pct,
        mean_center_dist_px=mean_center_dist,
        mean_iou=mean_iou,
        median_iou=median_iou,
        recall_pct=recall_pct,
        precision_pct=precision_pct,
        tracking_failures=failures,
        max_track_streak=max_streak,
        catastrophic_failure_rate=catastrophic_rate,
        occlusion_recovery_rate=recovery_rate,
        seeded_frames=seeded_frames,
        reseed_count=reseed_count,
        fps=fps,
        peak_vram_mb=peak_vram_mb,
        effective_resolution_px=effective_res,
        frame_predictions=frame_preds,
    )
