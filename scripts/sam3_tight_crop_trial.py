"""SAM3 GT-seeded tight-crop ball tracking trial.

Tests SAM3's ball-tracking ability in isolation, seeded from ball location on
frame 0 (via YOLO when GT marks not available), with tight cropping at ~512px
effective resolution.

Research question: given a correct(ish) starting position, how long does SAM3
hold the ball — especially through crowd/occlusion scenes?

Usage:
    uv run python scripts/sam3_tight_crop_trial.py [--clip PATH] [--seed-frame N]

Results are incrementally persisted to docs/sam3_trial_results.json.

Design notes:
- Tight-crop approach (a): crop the whole clip to a fixed ROI around YOLO's
  frame-0 ball position, then run SAM3 on that. Simpler than per-frame crop.
  ROI is 512x512px centred on the ball (clamped to frame edges).
- SAM3 imgsz=512 so the cropped 512x512 region fills the whole input window,
  giving maximum effective resolution on the ball region.
- Metrics: max consecutive frames tracked, drift-onset frame, center-distance
  per frame (vs initial ball position as proxy, since no GT JSONL available).
- Run serialized, alone — no GPU contention from other processes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Pin torch.compile cache before torch loads (avoids recompile on session restart).
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path.home() / ".cache" / "footy_torch_inductor"),
)

import cv2
import numpy as np
import torch

# Resolve project root and ensure src/ is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from footy_track.detectors.ultralytics import get_current_best_detector
from footy_track.labeller.video_utils import (
    _available_device,
    _default_model_uri,
)

# Output path for persisted results.
RESULTS_PATH = _REPO_ROOT / "docs" / "sam3_trial_results.json"

# Tight-crop size in pixels.
CROP_SIZE = 512

# Ball label tags from YOLO.
BALL_LABELS = {"ball", "in_play_ball", "out_of_play_ball"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_ball_yolo(
    video_path: Path,
    frame_idx: int = 0,
    conf_threshold: float = 0.10,
) -> tuple[float, float, float, float, int] | None:
    """Return (cx_norm, cy_norm, w_norm, h_norm, found_frame) of first YOLO ball.

    Searches from frame_idx up to 100 frames. Returns None if not found.
    """
    detector = get_current_best_detector(min_confidence=conf_threshold)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    search_limit = min(100, total)
    for fi in range(frame_idx, frame_idx + search_limit):
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            break

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = Path(f.name)
        cv2.imwrite(str(tmp_path), frame_bgr)
        try:
            fd = detector.predict_from_path(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        balls = [d for d in fd.detections if d.label in BALL_LABELS]
        if balls:
            b = max(balls, key=lambda d: d.confidence)
            cx = b.x + b.w / 2.0
            cy = b.y + b.h / 2.0
            print(
                f"  Ball found at frame {fi}: center=({cx:.3f}, {cy:.3f}), "
                f"size=({b.w:.3f}x{b.h:.3f}), conf={b.confidence:.2f}"
            )
            return cx, cy, b.w, b.h, fi

    print(f"  WARNING: No ball found in {search_limit} frames starting from frame {frame_idx} via YOLO.")
    return None


def build_tight_crop(
    cx_norm: float,
    cy_norm: float,
    frame_w: int,
    frame_h: int,
    crop_px: int = CROP_SIZE,
) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) pixel ROI centred on (cx_norm, cy_norm).

    Clamped so the crop stays within the frame.
    """
    cx_px = int(cx_norm * frame_w)
    cy_px = int(cy_norm * frame_h)
    half = crop_px // 2

    x1 = max(0, cx_px - half)
    y1 = max(0, cy_px - half)
    x2 = min(frame_w, x1 + crop_px)
    y2 = min(frame_h, y1 + crop_px)
    # Adjust if frame smaller than crop_px in either dimension.
    x1 = max(0, x2 - crop_px)
    y1 = max(0, y2 - crop_px)
    return x1, y1, x2, y2


def write_cropped_video(
    video_path: Path,
    roi: tuple[int, int, int, int],
    out_path: Path,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> int:
    """Write a cropped video (optionally starting at start_frame) and return the frame count."""
    x1, y1, x2, y2 = roi
    crop_w = x2 - x1
    crop_h = y2 - y1

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total = total - start_frame
    if max_frames is not None:
        total = min(total, max_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (crop_w, crop_h))

    frame_count = 0
    while frame_count < total:
        ok, frame = cap.read()
        if not ok:
            break
        cropped = frame[y1:y2, x1:x2]
        writer.write(cropped)
        frame_count += 1

    cap.release()
    writer.release()
    return frame_count


def run_sam3_on_cropped(
    cropped_video_path: Path,
    seed_bbox_in_crop: tuple[float, float, float, float],
    model_uri: str,
    crop_w: int,
    crop_h: int,
    imgsz: int = 512,
    device: str = "cpu",
    max_frames: int | None = None,
) -> list[dict]:
    """Run SAM3VideoPredictor on a cropped video, seeded from a bbox in crop coords.

    Returns list of per-frame dicts:
        {frame_index, tracked: bool, cx_norm: float|None, cy_norm: float|None,
         mask_area_norm: float|None}
    Coords are normalised within the CROP (not the full frame).
    """
    from ultralytics.models.sam import SAM3VideoPredictor  # noqa: PLC0415

    # Convert normalized crop coords to absolute pixels in the crop video.
    bbox_abs = [
        seed_bbox_in_crop[0] * crop_w,
        seed_bbox_in_crop[1] * crop_h,
        seed_bbox_in_crop[2] * crop_w,
        seed_bbox_in_crop[3] * crop_h,
    ]

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
    predictor.set_prompts({"bboxes": [bbox_abs]})

    per_frame = []
    for fi, result in enumerate(predictor(source=str(cropped_video_path), stream=True)):
        if max_frames is not None and fi >= max_frames:
            break

        masks = getattr(result, "masks", None)
        polys = masks.xyn if masks is not None else []

        tracked = False
        cx_norm = cy_norm = mask_area_norm = None

        if polys and len(polys) > 0:
            poly = polys[0]  # first (and only) tracked object
            if poly is not None and len(poly) > 0:
                pts = np.array(poly, dtype=np.float32)
                if pts.size > 0:
                    cx_norm = float(pts[:, 0].mean())
                    cy_norm = float(pts[:, 1].mean())
                    # Shoelace area (normalized)
                    if len(pts) >= 3:
                        x_ = pts[:, 0]
                        y_ = pts[:, 1]
                        area = abs(
                            np.dot(x_, np.roll(y_, 1)) - np.dot(y_, np.roll(x_, 1))
                        ) / 2.0
                        mask_area_norm = float(area)
                    tracked = True

        per_frame.append(
            {
                "frame_index": fi,
                "tracked": tracked,
                "cx_norm": cx_norm,
                "cy_norm": cy_norm,
                "mask_area_norm": mask_area_norm,
            }
        )

    return per_frame


def compute_metrics(
    per_frame: list[dict],
    ball_cx_in_crop: float,
    ball_cy_in_crop: float,
    crop_w: int,
    crop_h: int,
) -> dict:
    """Compute tracking metrics from per-frame results.

    - max_track_streak: longest consecutive run of tracked=True frames
    - drift_onset_frame: first frame where tracked=False after an initial tracked run
    - pct_tracked: % frames where SAM3 returned a mask
    - mean_dist_from_seed_px: mean pixel distance from initial ball center
      (measures drift tendency, not accuracy vs GT)
    """
    tracked_flags = [f["tracked"] for f in per_frame]
    total = len(tracked_flags)
    n_tracked = sum(tracked_flags)
    pct = 100.0 * n_tracked / total if total > 0 else 0.0

    # Max consecutive streak
    max_streak = cur_streak = 0
    for t in tracked_flags:
        if t:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    # First drift-onset (first False after at least one True)
    drift_onset = None
    seen_true = False
    for i, t in enumerate(tracked_flags):
        if t:
            seen_true = True
        elif seen_true:
            drift_onset = i
            break

    # Mean distance from seed center (in crop pixels)
    dists = []
    seed_x_px = ball_cx_in_crop * crop_w
    seed_y_px = ball_cy_in_crop * crop_h
    for f in per_frame:
        if f["tracked"] and f["cx_norm"] is not None:
            px_x = f["cx_norm"] * crop_w
            px_y = f["cy_norm"] * crop_h
            dists.append(((px_x - seed_x_px) ** 2 + (px_y - seed_y_px) ** 2) ** 0.5)

    mean_dist = float(np.mean(dists)) if dists else None

    return {
        "total_frames": total,
        "n_tracked": n_tracked,
        "pct_tracked": round(pct, 1),
        "max_track_streak": max_streak,
        "drift_onset_frame": drift_onset,
        "mean_dist_from_seed_px": round(mean_dist, 1) if mean_dist is not None else None,
    }


def save_checkpoint(results: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"  [checkpoint] saved to {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM3 tight-crop ball tracking trial")
    ap.add_argument(
        "--clip",
        type=Path,
        default=None,
        help="Path to clip MP4. Defaults to mancity_seg010 in CloudDrive.",
    )
    ap.add_argument(
        "--seed-frame",
        type=int,
        default=0,
        help="Frame index to search for YOLO ball seed (default: 0)",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames processed (None = all)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="SAM3 model path. Defaults to iCloud sam3.pt.",
    )
    args = ap.parse_args()

    clip_default = (
        Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "footy_data"
        / "arsenal_mancity"
        / "split_video_broadcast_frames"
        / "arsenal_mancity_20250925_seg010.mp4"
    )
    clip_path = args.clip or clip_default
    if not clip_path.exists():
        sys.exit(f"ERROR: Clip not found: {clip_path}")

    model_uri = args.model or _default_model_uri()
    print(f"SAM3 trial — clip: {clip_path.name}")
    print(f"             model: {model_uri}")

    # Detect device
    dev = _available_device()
    device = dev.type if isinstance(dev, torch.device) else str(dev)
    print(f"             device: {device}")

    # Get clip dimensions
    cap = cv2.VideoCapture(str(clip_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"             resolution: {frame_w}x{frame_h}, {total_frames} frames @ {fps}fps")

    max_frames = args.max_frames or total_frames

    results: dict = {
        "trial_metadata": {
            "clip": str(clip_path),
            "clip_name": clip_path.stem,
            "model": str(model_uri),
            "device": device,
            "imgsz": CROP_SIZE,
            "frame_w": frame_w,
            "frame_h": frame_h,
            "total_frames_in_clip": total_frames,
            "max_frames_processed": max_frames,
            "seed_method": "yolo_frame0",
            "seed_note": (
                "GT JSONL files not found; seeding from YOLO detection. "
                "Searched up to 100 frames from seed_frame to find ball."
            ),
            "crop_size_px": CROP_SIZE,
            "run_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "status": "running",
        "seed": None,
        "roi": None,
        "per_frame": [],
        "metrics": None,
    }
    save_checkpoint(results)

    # --- Step 1: Find ball seed ---
    print("\n[1/4] Finding ball seed via YOLO...")
    t0 = time.time()
    seed = find_ball_yolo(clip_path, frame_idx=args.seed_frame)
    if seed is None:
        results["status"] = "failed_no_seed"
        results["error"] = "YOLO could not find ball in first 100 frames"
        save_checkpoint(results)
        sys.exit("ERROR: No ball found by YOLO. Cannot proceed.")

    cx_norm, cy_norm, w_norm, h_norm, found_frame = seed
    results["seed"] = {
        "cx_norm": cx_norm,
        "cy_norm": cy_norm,
        "w_norm": w_norm,
        "h_norm": h_norm,
        "found_frame": found_frame,
    }
    save_checkpoint(results)
    print(f"  Seed: center=({cx_norm:.3f}, {cy_norm:.3f}) at frame {found_frame}, time={time.time()-t0:.1f}s")

    # --- Step 2: Build ROI and write cropped video ---
    print("\n[2/4] Building tight crop ROI and writing cropped video...")
    t1 = time.time()
    roi = build_tight_crop(cx_norm, cy_norm, frame_w, frame_h, CROP_SIZE)
    x1, y1, x2, y2 = roi
    crop_w = x2 - x1
    crop_h = y2 - y1
    results["roi"] = {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "crop_w": crop_w, "crop_h": crop_h,
        "start_frame_in_clip": found_frame,
    }
    print(f"  ROI: ({x1},{y1})-({x2},{y2}) = {crop_w}x{crop_h}px")

    # Seed ball center in crop-normalised coordinates
    ball_cx_in_crop = (cx_norm * frame_w - x1) / crop_w
    ball_cy_in_crop = (cy_norm * frame_h - y1) / crop_h

    # Seed bbox (crop coords, normalized): use a small box around the ball center
    ball_size_in_crop = max(w_norm * frame_w, h_norm * frame_h, 20) / crop_w
    half_box = ball_size_in_crop / 2.0 + 0.05  # add a bit of padding
    seed_bbox_norm = (
        max(0.0, ball_cx_in_crop - half_box),
        max(0.0, ball_cy_in_crop - half_box),
        min(1.0, ball_cx_in_crop + half_box),
        min(1.0, ball_cy_in_crop + half_box),
    )
    print(
        f"  Ball in crop (norm): center=({ball_cx_in_crop:.3f}, {ball_cy_in_crop:.3f}), "
        f"seed_bbox={[round(v, 3) for v in seed_bbox_norm]}"
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        cropped_tmp = Path(f.name)
    print(f"  Writing cropped video (starting at frame {found_frame}) to {cropped_tmp}...")
    n_written = write_cropped_video(
        clip_path, roi, cropped_tmp, start_frame=found_frame, max_frames=max_frames
    )
    print(f"  Wrote {n_written} frames in {time.time()-t1:.1f}s")

    # --- Step 3: Run SAM3 ---
    print(f"\n[3/4] Running SAM3VideoPredictor on {n_written} cropped frames...")
    print(f"  Model: {model_uri}")
    print(f"  This may take a while (3.45GB model, torch.compile on first run)...")
    t2 = time.time()

    try:
        per_frame = run_sam3_on_cropped(
            cropped_tmp,
            seed_bbox_in_crop=seed_bbox_norm,
            model_uri=model_uri,
            crop_w=crop_w,
            crop_h=crop_h,
            imgsz=CROP_SIZE,
            device=device,
            max_frames=max_frames,
        )
    except Exception as e:
        results["status"] = "failed_sam3"
        results["error"] = str(e)
        save_checkpoint(results)
        cropped_tmp.unlink(missing_ok=True)
        raise
    finally:
        cropped_tmp.unlink(missing_ok=True)

    elapsed_sam3 = time.time() - t2
    fps_sam3 = len(per_frame) / elapsed_sam3 if elapsed_sam3 > 0 else 0
    print(f"  SAM3 done: {len(per_frame)} frames in {elapsed_sam3:.1f}s ({fps_sam3:.2f} fps)")

    results["per_frame"] = per_frame
    results["sam3_elapsed_s"] = round(elapsed_sam3, 1)
    results["sam3_fps"] = round(fps_sam3, 3)
    save_checkpoint(results)

    # --- Step 4: Metrics ---
    print("\n[4/4] Computing metrics...")
    metrics = compute_metrics(per_frame, ball_cx_in_crop, ball_cy_in_crop, crop_w, crop_h)
    results["metrics"] = metrics
    results["status"] = "done"
    results["trial_metadata"]["run_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    save_checkpoint(results)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Clip:                  {clip_path.name}")
    print(f"Total frames:          {metrics['total_frames']}")
    print(f"Frames tracked:        {metrics['n_tracked']} ({metrics['pct_tracked']}%)")
    print(f"Max track streak:      {metrics['max_track_streak']} consecutive frames")
    print(f"Drift onset:           frame {metrics['drift_onset_frame']} (None = never lost)")
    print(f"Mean dist from seed:   {metrics['mean_dist_from_seed_px']}px (in {CROP_SIZE}px crop)")
    print(f"SAM3 throughput:       {fps_sam3:.2f} fps")
    print(f"")
    print(f"VERDICT:")
    pct = metrics["pct_tracked"]
    streak = metrics["max_track_streak"]
    if pct >= 80:
        print(f"  STRONG tracking ({pct}% frames, max streak {streak})")
    elif pct >= 40:
        print(f"  MODERATE tracking ({pct}% frames, max streak {streak})")
    else:
        print(f"  WEAK tracking ({pct}% frames, max streak {streak})")
    print(f"  ROI-YOLO baseline: 0-12% hit rate")
    print(f"  Results saved to: {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
