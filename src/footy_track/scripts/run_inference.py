"""Run the simple end-to-end inference pipeline on a video or clip.

    python -m footy_track.scripts.run_inference \
        --video data/arsenal_mancity_example_video.mp4 \
        --weights model_saves/detector/rtdetr_1920_v11/best.pt \
        --out runs/example --max-frames 300 --overlay

Writes detections.jsonl, tracklets.jsonl, ball_track.jsonl and summary.json (plus
overlay.mp4 with --overlay) to --out. Formats are in
``footy_track.inference.pipeline``; usage and thresholds in docs/inference.md.

Ball defaults come from the Ball Check precision bands (0.04 at conf 0.1-0.2,
0.42 at 0.5-0.6, 0.79 at 0.6-0.7, 0.96 at 0.7-0.8, 1.0 at 0.8-0.9): a track may
only START on a candidate that is usually right (--ball-start-conf 0.6), and a
candidate of --ball-jump-conf 0.8 or more is believed even outside the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from footy_track.ball_trackers.factory import TRACKER_KINDS
from footy_track.detectors.ultralytics import (
    CURRENT_BEST_DETECTOR_CHECKPOINT,
    DEFAULT_IMGSZ,
    UltralyticsObjectDetector,
)
from footy_track.inference.pipeline import InferenceConfig, run_inference, select_device
from footy_track.utils import get_project_root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    d = InferenceConfig()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--weights",
        type=Path,
        default=get_project_root() / CURRENT_BEST_DETECTOR_CHECKPOINT,
        help="detector checkpoint (not in git)",
    )
    ap.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--detect-conf", type=float, default=d.detect_conf)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--start", type=int, default=d.start, help="first frame index")
    ap.add_argument("--max-frames", type=int, default=None, help="frames to process")
    ap.add_argument(
        "--stride", type=int, default=d.stride, help="process every Nth frame"
    )
    ap.add_argument("--overlay", action="store_true", help="also write overlay.mp4")
    ap.add_argument("--player-conf", type=float, default=d.player_conf)
    ap.add_argument("--track-iou", type=float, default=d.track_iou)
    ap.add_argument("--track-max-age", type=int, default=d.track_max_age)
    ap.add_argument("--ball-start-conf", type=float, default=d.ball_start_conf)
    ap.add_argument("--ball-jump-conf", type=float, default=d.ball_jump_conf)
    ap.add_argument("--ball-base-radius-px", type=float, default=d.ball_base_radius_px)
    ap.add_argument("--ball-growth-px", type=float, default=d.ball_growth_px)
    ap.add_argument("--ball-max-missed", type=int, default=d.ball_max_missed)
    ap.add_argument(
        "--ball-tracker",
        choices=TRACKER_KINDS,
        default=d.ball_tracker,
        help="online (streaming, default) or an offline tracker that sees the whole clip",
    )
    args = ap.parse_args(argv)

    if not args.weights.exists():
        ap.error(f"weights not found: {args.weights} (pass --weights)")
    if args.stride < 1:
        ap.error("--stride must be >= 1")

    device = args.device or select_device()
    detector = UltralyticsObjectDetector(
        model_uri=str(args.weights),
        min_confidence=args.detect_conf,
        imgsz=args.imgsz,
        use_model_names=True,
        device=device,
    )
    config = InferenceConfig(
        start=args.start,
        max_frames=args.max_frames,
        stride=args.stride,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        detect_conf=args.detect_conf,
        player_conf=args.player_conf,
        track_iou=args.track_iou,
        track_max_age=args.track_max_age,
        ball_start_conf=args.ball_start_conf,
        ball_jump_conf=args.ball_jump_conf,
        ball_base_radius_px=args.ball_base_radius_px,
        ball_growth_px=args.ball_growth_px,
        ball_max_missed=args.ball_max_missed,
        ball_tracker=args.ball_tracker,
        overlay=args.overlay,
    )
    source_tag = "rtdetr" if "rtdetr" in str(args.weights).lower() else "yolo"
    summary = run_inference(
        args.video,
        args.out,
        detector,
        config,
        checkpoint=str(args.weights),
        device=device,
        source_tag=source_tag,
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("frames_processed", "detect_fps", "players", "ball")
            },
            indent=2,
        )
    )
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
