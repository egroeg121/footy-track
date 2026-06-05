"""Prototype: seed SAM3 video tracking with POINT prompts (not boxes).

De-risks the planned rework. Confirms that feeding ``set_prompts({"points",
"labels"})`` through the existing streaming wrapper:
  1. works (propagates objects across frames), and
  2. doesn't produce the whole-frame box we see with bbox prompts on sam3.1.

Usage:
  uv run python scripts/proto_sam3_points.py <video.mp4> [--model PATH]

Marks one point per object near the YOLO-detected box centre on frame 0, runs a
few frames, and prints the resulting box sizes (a sane result = small boxes that
track; the bug = boxes ~= whole frame).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from footy_track.labeller.video_utils import (
    _available_device,
    _default_model_uri,
    mask_poly_to_norm_xywh,
    yolo_seed_objects,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--model", default=None)
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()

    from ultralytics.models.sam import SAM3VideoPredictor  # noqa: PLC0415

    cap = cv2.VideoCapture(str(args.video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Seed points = centres of YOLO boxes on frame 0 (positive/foreground points).
    seeds = yolo_seed_objects(args.video, "", 0.35, w, h, 0.5, 0)
    points = []
    labels = []
    for o in seeds:
        x1, y1, x2, y2 = o.bbox_xyxy_abs
        points.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        labels.append(1)  # 1 = foreground
    print(f"Seeding {len(points)} positive points from YOLO on frame 0")

    dev = _available_device()
    device = dev.type if hasattr(dev, "type") else str(dev)
    overrides = {
        "conf": 0.25,
        "task": "segment",
        "mode": "predict",
        "model": args.model or _default_model_uri(),
        "imgsz": 512,
        "verbose": False,
        "device": device,
        "save": False,
    }
    predictor = SAM3VideoPredictor(overrides=overrides)
    # POINT prompts: shape (N, 2) = N objects, one point each; labels (N,).
    # (Passing [points] would mean ONE object with N points -> whole-frame mask.)
    predictor.set_prompts({"points": points, "labels": labels})

    print(f"{'frame':>5} {'obj':>4} {'label':>10} {'w%':>6} {'h%':>6}")
    for fi, result in enumerate(predictor(source=str(args.video), stream=True)):
        if fi >= args.frames:
            break
        masks = getattr(result, "masks", None)
        polys = masks.xyn if masks is not None else []
        for oi, poly in enumerate(polys):
            box = mask_poly_to_norm_xywh(poly)
            if box is None:
                continue
            _, _, bw, bh = box
            lbl = seeds[oi].label if oi < len(seeds) else "?"
            # flag a whole-frame blowup
            flag = "  <-- WHOLE FRAME?" if (bw > 0.8 and bh > 0.8) else ""
            print(f"{fi:>5} {oi:>4} {lbl:>10} {bw * 100:>5.1f} {bh * 100:>5.1f}{flag}")


if __name__ == "__main__":
    main()
