"""SAM3 GT-seeded tight-crop ball tracking trial.

Runs SAM3VideoPredictor on a spatially-cropped version of the clip so the ball
is large at imgsz=512, seeded from the first GT ball mark.  Results are
persisted incrementally to docs/sam3_trial_results.json.

Strategy: crop approach (a) — compute the bounding extent of all GT ball marks,
add a margin, write that ROI as a temp clip, run SAM3 on it, map centroid back
to full-frame coords.  This is the simplest approach that fits SAM3's whole-clip
pipeline.

Usage:
    uv run python scripts/run_sam3_tight_crop_trial.py [--clip mancity_seg010]

Key verdict metrics:
  - max consecutive frames tracked before drift
  - drift onset frame (first frame where center error > DRIFT_THRESHOLD_PX)
  - overall center-within-radius-pct vs GT
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import tempfile
import time
from typing import NamedTuple

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).parents[1]

# GT marks and video locations
GT_DIR = (
    pathlib.Path.home()
    / "Library"
    / "Mobile Documents"
    / "com~apple~CloudDocs"
    / "footy_data"
    / "ball_gt_marks"
)
VIDEO_DIR = (
    pathlib.Path.home()
    / "Library"
    / "Mobile Documents"
    / "com~apple~CloudDocs"
    / "footy_data"
    / "arsenal_mancity"
    / "split_video_broadcast_frames"
)

OUTPUT_PATH = PROJECT_ROOT / "docs" / "sam3_trial_results.json"

# Clip name → (video stem, gt jsonl stem) mapping
CLIP_VIDEO_MAP = {
    "mancity_seg010": ("arsenal_mancity_20250925_seg010", "mancity_seg010"),
    "mancity_seg050": ("arsenal_mancity_20250925_seg050", "mancity_seg050"),
}

DRIFT_THRESHOLD_PX = 30  # pixels — center error above this = "drift"
ROI_MARGIN = 0.08        # extra margin fraction around GT bbox extent
IMGSZ = 512


class BallMark(NamedTuple):
    frame_index: int
    cx_norm: float
    cy_norm: float
    bbox_norm: tuple[float, float, float, float]  # (x, y, w, h) normalised


def load_gt_ball_marks(gt_path: pathlib.Path) -> list[BallMark]:
    """Load in_play_ball marks from a JSONL file, sorted by frame_index."""
    marks = []
    with gt_path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            d = json.loads(stripped)
            if "in_play_ball" not in d.get("tags", []):
                continue
            bbox = d.get("bbox")
            center = d.get("center")
            if bbox is None and center is None:
                continue
            if isinstance(bbox, dict):
                bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            elif bbox is not None:
                bx, by, bw, bh = bbox
            else:
                bx, by, bw, bh = None, None, None, None

            if isinstance(center, dict):
                cx, cy = center["x"], center["y"]
            elif center is not None:
                cx, cy = center
            elif bx is not None:
                cx, cy = bx + bw / 2, by + bh / 2
            else:
                continue

            if bx is None:
                bx, by, bw, bh = cx - 0.005, cy - 0.005, 0.01, 0.01

            marks.append(BallMark(
                frame_index=int(d["frame_index"]),
                cx_norm=float(cx),
                cy_norm=float(cy),
                bbox_norm=(float(bx), float(by), float(bw), float(bh)),
            ))
    return sorted(marks, key=lambda m: m.frame_index)


def compute_roi(
    marks: list[BallMark],
    margin: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) pixel ROI enclosing all GT ball marks + margin."""
    min_x = min(m.bbox_norm[0] for m in marks)
    min_y = min(m.bbox_norm[1] for m in marks)
    max_x = max(m.bbox_norm[0] + m.bbox_norm[2] for m in marks)
    max_y = max(m.bbox_norm[1] + m.bbox_norm[3] for m in marks)

    # Add margin
    pad_x = margin * (max_x - min_x)
    pad_y = margin * (max_y - min_y)
    min_x = max(0.0, min_x - pad_x)
    min_y = max(0.0, min_y - pad_y)
    max_x = min(1.0, max_x + pad_x)
    max_y = min(1.0, max_y + pad_y)

    return (
        int(min_x * frame_w),
        int(min_y * frame_h),
        int(max_x * frame_w),
        int(max_y * frame_h),
    )


def write_cropped_clip(
    video_path: pathlib.Path,
    roi: tuple[int, int, int, int],
    out_path: str,
) -> tuple[int, int]:
    """Write a spatially cropped copy of video_path to out_path.

    Returns (crop_w, crop_h).
    """
    x0, y0, x1, y1 = roi
    crop_w, crop_h = x1 - x0, y1 - y0

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (crop_w, crop_h))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cropped = frame[y0:y1, x0:x1]
            writer.write(cropped)
    finally:
        writer.release()
        cap.release()
    return crop_w, crop_h


def mask_poly_to_centroid(poly: np.ndarray) -> tuple[float, float] | None:
    """Return normalised centroid (cx, cy) from a SAM3 mask polygon (xyn format)."""
    if poly is None or len(poly) == 0:
        return None
    pts = np.array(poly)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    return (cx, cy)


def run_sam3_on_clip(
    cropped_video: str,
    seed_bbox_crop: list[float],  # [x1, y1, x2, y2] in crop coords (pixels)
    model_uri: str,
    imgsz: int = 512,
    checkpoint_callback=None,
) -> list[tuple[float, float] | None]:
    """Run SAM3VideoPredictor and return list of (cx, cy) in crop-normalised coords.

    Returns one entry per frame; None if no mask produced.
    """
    import torch
    from ultralytics.models.sam import SAM3VideoPredictor

    dev_fn = None
    try:
        from footy_track.detectors.utils import _available_device
        dev_fn = _available_device
    except ImportError:
        pass

    if dev_fn is not None:
        dev = dev_fn()
        device = dev.type if isinstance(dev, torch.device) else str(dev)
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"  SAM3 device: {device}")

    overrides = {
        "conf": 0.25,
        "task": "segment",
        "mode": "predict",
        "model": model_uri,
        "imgsz": imgsz,
        "verbose": False,
        "device": device,
        "save": False,
    }
    predictor = SAM3VideoPredictor(overrides=overrides)
    predictor.set_prompts({"bboxes": [seed_bbox_crop]})

    centroids: list[tuple[float, float] | None] = []
    for fi, result in enumerate(predictor(source=cropped_video, stream=True)):
        masks = getattr(result, "masks", None)
        polys = masks.xyn if masks is not None else []
        if len(polys) == 0:
            centroids.append(None)
        else:
            # Take the first (and only) mask — we seeded one object
            c = mask_poly_to_centroid(polys[0])
            centroids.append(c)
        if checkpoint_callback is not None:
            checkpoint_callback(fi, centroids[-1])
        if fi % 50 == 0:
            print(f"    frame {fi}: {'tracked' if centroids[-1] else 'LOST'}")
    return centroids


def score_results(
    centroids_crop_norm: list[tuple[float, float] | None],
    gt_marks: list[BallMark],
    roi: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> dict:
    """Score SAM3 centroid predictions against GT marks.

    Centroids are in crop-normalised [0,1] coords; we map back to full-frame
    pixel coords for scoring.
    """
    x0, y0, x1, y1 = roi
    crop_w = x1 - x0
    crop_h = y1 - y0

    gt_by_frame = {m.frame_index: m for m in gt_marks}
    labelled_frames = sorted(gt_by_frame.keys())

    per_frame: list[dict] = []
    center_dists: list[float] = []
    within_radius_hits = 0
    within_radius_total = 0
    consecutive_streak = 0
    max_streak = 0
    drift_onset_frame: int | None = None

    for frame_idx in labelled_frames:
        mark = gt_by_frame[frame_idx]
        gt_cx_px = mark.cx_norm * frame_w
        gt_cy_px = mark.cy_norm * frame_h

        pred_entry: dict = {"frame_index": frame_idx, "gt_cx_norm": mark.cx_norm, "gt_cy_norm": mark.cy_norm}

        if frame_idx < len(centroids_crop_norm):
            c = centroids_crop_norm[frame_idx]
        else:
            c = None

        if c is not None:
            # Map crop-normalised → full-frame pixels
            pred_cx_px = c[0] * crop_w + x0
            pred_cy_px = c[1] * crop_h + y0
            dist_px = math.sqrt((pred_cx_px - gt_cx_px) ** 2 + (pred_cy_px - gt_cy_px) ** 2)
            pred_cx_norm = pred_cx_px / frame_w
            pred_cy_norm = pred_cy_px / frame_h
            pred_entry["pred_cx_norm"] = pred_cx_norm
            pred_entry["pred_cy_norm"] = pred_cy_norm
            pred_entry["center_dist_px"] = round(dist_px, 1)

            # Ball radius from GT bbox
            _, _, bw, bh = mark.bbox_norm
            radius_px = max(max(bw * frame_w, bh * frame_h) / 2, 10.0)
            pred_entry["gt_radius_px"] = round(radius_px, 1)

            center_dists.append(dist_px)
            within_radius_total += 1
            if dist_px <= radius_px:
                within_radius_hits += 1
                consecutive_streak += 1
                max_streak = max(max_streak, consecutive_streak)
            else:
                if drift_onset_frame is None and dist_px > DRIFT_THRESHOLD_PX:
                    drift_onset_frame = frame_idx
                consecutive_streak = 0
            pred_entry["within_radius"] = dist_px <= radius_px
            pred_entry["drifted"] = dist_px > DRIFT_THRESHOLD_PX
        else:
            pred_entry["pred_cx_norm"] = None
            pred_entry["pred_cy_norm"] = None
            pred_entry["center_dist_px"] = None
            pred_entry["within_radius"] = False
            pred_entry["drifted"] = None
            consecutive_streak = 0

        per_frame.append(pred_entry)

    mean_dist = sum(center_dists) / len(center_dists) if center_dists else None
    center_within_radius_pct = (
        100.0 * within_radius_hits / within_radius_total if within_radius_total > 0 else 0.0
    )

    return {
        "labelled_frames": len(labelled_frames),
        "frames_with_prediction": sum(1 for p in per_frame if p.get("pred_cx_norm") is not None),
        "center_within_radius_pct": round(center_within_radius_pct, 1),
        "mean_center_dist_px": round(mean_dist, 1) if mean_dist is not None else None,
        "max_track_streak": max_streak,
        "drift_onset_frame": drift_onset_frame,
        "tracking_failures": within_radius_total - within_radius_hits,
        "per_frame": per_frame,
    }


def resolve_model_uri() -> str:
    candidates = [
        pathlib.Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "footy_data"
        / "model_saves"
        / "sam3"
        / "sam3.pt",
        PROJECT_ROOT / "model_saves" / "sam3" / "sam3.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "sam3.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="mancity_seg010")
    ap.add_argument("--model", default=None, help="SAM3 model path (default: auto-resolve)")
    ap.add_argument("--output", type=pathlib.Path, default=OUTPUT_PATH)
    ap.add_argument("--margin", type=float, default=ROI_MARGIN)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args = ap.parse_args()

    clip_name = args.clip
    clip_entry = CLIP_VIDEO_MAP.get(clip_name)
    if clip_entry is None:
        print(f"Unknown clip: {clip_name}. Known clips: {list(CLIP_VIDEO_MAP.keys())}")
        return
    video_stem, gt_stem = clip_entry

    video_path = VIDEO_DIR / f"{video_stem}.mp4"
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    gt_path = GT_DIR / f"{gt_stem}.jsonl"
    if not gt_path.exists():
        print(f"GT file not found: {gt_path}")
        return

    model_uri = args.model or resolve_model_uri()
    print(f"Model: {model_uri}")
    print(f"Video: {video_path}")
    print(f"GT:    {gt_path}")

    # Load GT marks
    gt_marks = load_gt_ball_marks(gt_path)
    if not gt_marks:
        print("No in_play_ball marks found in GT file!")
        return
    print(f"GT ball marks: {len(gt_marks)} frames, first={gt_marks[0].frame_index}, last={gt_marks[-1].frame_index}")

    # Video dimensions
    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Frame: {frame_w}x{frame_h}, total={total_frames}")

    # Compute tight crop ROI from GT ball extent
    roi = compute_roi(gt_marks, margin=args.margin, frame_w=frame_w, frame_h=frame_h)
    x0, y0, x1, y1 = roi
    crop_w, crop_h = x1 - x0, y1 - y0
    print(f"ROI (pixels): ({x0},{y0}) -> ({x1},{y1}) = {crop_w}x{crop_h}")
    print(f"ROI coverage: {crop_w/frame_w*100:.1f}% x {crop_h/frame_h*100:.1f}% of frame")
    print(f"Effective ball size at imgsz={args.imgsz}: ~{args.imgsz * crop_h / frame_h:.0f}px crop height")

    # Seed bbox in crop coordinates (absolute pixels within crop)
    seed_mark = gt_marks[0]
    sx, sy, sw, sh = seed_mark.bbox_norm
    # Convert to crop-relative pixel coords
    seed_x1_px = sx * frame_w - x0
    seed_y1_px = sy * frame_h - y0
    seed_x2_px = (sx + sw) * frame_w - x0
    seed_y2_px = (sy + sh) * frame_h - y0
    seed_bbox_crop = [seed_x1_px, seed_y1_px, seed_x2_px, seed_y2_px]
    print(f"Seed frame: {seed_mark.frame_index}, bbox_crop={[round(v, 1) for v in seed_bbox_crop]}")

    # Write cropped clip to temp file
    print(f"\nWriting cropped clip ({crop_w}x{crop_h})...")
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp") as f:
        tmp_path = f.name
    write_cropped_clip(video_path, roi, tmp_path)
    print(f"  done in {time.perf_counter()-t0:.1f}s -> {tmp_path}")

    # Persist intermediate state
    intermediate = {
        "clip": clip_name,
        "video": str(video_path),
        "gt_file": str(gt_path),
        "model": model_uri,
        "imgsz": args.imgsz,
        "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "crop_w": crop_w, "crop_h": crop_h},
        "seed_frame": seed_mark.frame_index,
        "gt_marks_count": len(gt_marks),
        "status": "running_sam3",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(intermediate, indent=2))
    print(f"Intermediate state saved to {args.output}")

    # Run SAM3
    print(f"\nRunning SAM3 on cropped clip (imgsz={args.imgsz})...")
    t0 = time.perf_counter()
    centroids: list[tuple[float, float] | None] = []

    def checkpoint(fi, centroid):
        # Incremental save every 50 frames
        if fi % 50 == 0 and fi > 0:
            tmp_out = {
                **intermediate,
                "status": f"running_frame_{fi}/{total_frames}",
                "centroids_so_far": fi + 1,
            }
            args.output.write_text(json.dumps(tmp_out, indent=2))

    centroids = run_sam3_on_clip(
        tmp_path,
        seed_bbox_crop,
        model_uri,
        imgsz=args.imgsz,
        checkpoint_callback=checkpoint,
    )
    sam3_time = time.perf_counter() - t0
    fps = len(centroids) / sam3_time if sam3_time > 0 else 0
    print(f"  SAM3 done: {len(centroids)} frames in {sam3_time:.1f}s ({fps:.1f} fps)")

    # Clean up temp file
    pathlib.Path(tmp_path).unlink(missing_ok=True)

    # Score results
    print("\nScoring vs GT marks...")
    scores = score_results(centroids, gt_marks, roi, frame_w, frame_h)
    print(f"  GT labelled frames: {scores['labelled_frames']}")
    print(f"  Frames with prediction: {scores['frames_with_prediction']}")
    print(f"  Center within radius %: {scores['center_within_radius_pct']:.1f}%")
    print(f"  Mean center dist px: {scores['mean_center_dist_px']}")
    print(f"  Max consecutive track streak: {scores['max_track_streak']}")
    print(f"  Drift onset frame: {scores['drift_onset_frame']}")
    print(f"  Tracking failures: {scores['tracking_failures']}")

    # Full result
    result = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clip": clip_name,
        "video": str(video_path),
        "gt_file": str(gt_path),
        "model": model_uri,
        "imgsz": args.imgsz,
        "roi_margin": args.margin,
        "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "crop_w": crop_w, "crop_h": crop_h},
        "seed_frame": seed_mark.frame_index,
        "total_frames": total_frames,
        "sam3_fps": round(fps, 1),
        "sam3_time_s": round(sam3_time, 1),
        "scores": scores,
        "verdict": {
            "center_within_radius_pct": scores["center_within_radius_pct"],
            "max_track_streak": scores["max_track_streak"],
            "drift_onset_frame": scores["drift_onset_frame"],
            "mean_center_dist_px": scores["mean_center_dist_px"],
            "beats_roi_yolo_baseline": scores["center_within_radius_pct"] > 1.1,
        },
        "status": "complete",
    }

    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to: {args.output}")

    # Summary verdict
    print("\n=== VERDICT ===")
    print(f"  SAM3 tight-crop (imgsz={args.imgsz}) on {clip_name}")
    print(f"  Crop: {crop_w}x{crop_h} ({crop_w/frame_w*100:.1f}%x{crop_h/frame_h*100:.1f}% of frame)")
    print(f"  Center within radius: {scores['center_within_radius_pct']:.1f}% (ROI-YOLO baseline: 1.1%)")
    print(f"  Max track streak: {scores['max_track_streak']} frames")
    print(f"  Drift onset: frame {scores['drift_onset_frame']}")
    print(f"  Mean center dist: {scores['mean_center_dist_px']} px")
    beats = scores["center_within_radius_pct"] > 1.1
    print(f"  Beats ROI-YOLO baseline: {'YES' if beats else 'NO'}")


if __name__ == "__main__":
    main()
