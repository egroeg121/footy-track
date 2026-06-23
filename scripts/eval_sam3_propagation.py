"""Evaluate SAM3 point-prompt propagation for ball occlusion handling (ft-55d.4).

Tests whether SAM3 can propagate a ball mask forward through frames where
YOLO fails, given a confident seed detection.

Usage:
    uv run python scripts/eval_sam3_propagation.py [--clips-dir PATH] [--gt-dir PATH] [--output PATH]

Output:
    JSON report committed to bead notes (accuracy, latency, failure modes).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).parents[1]

CLIPS_DIR_DEFAULT = pathlib.Path(
    "/Users/georgebarnett/code/footy/footy_track/refinery/rig/eval_data/clips"
)
GT_DIR_DEFAULT = (
    pathlib.Path.home()
    / "Library"
    / "Mobile Documents"
    / "com~apple~CloudDocs"
    / "footy_data"
    / "ball_gt_marks"
)
OUTPUT_DEFAULT = PROJECT_ROOT / "docs" / "sam3_propagation_eval.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate SAM3 propagation for ball occlusion")
    ap.add_argument("--clips-dir", type=pathlib.Path, default=CLIPS_DIR_DEFAULT)
    ap.add_argument("--gt-dir", type=pathlib.Path, default=GT_DIR_DEFAULT)
    ap.add_argument("--output", type=pathlib.Path, default=OUTPUT_DEFAULT)
    ap.add_argument(
        "--max-clips",
        type=int,
        default=5,
        help="Limit number of clips evaluated (default 5 for speed)",
    )
    ap.add_argument(
        "--prop-window",
        type=int,
        default=30,
        help="SAM3 propagation window in frames (default 30)",
    )
    args = ap.parse_args()

    if not args.clips_dir.exists():
        print(f"ERROR: clips dir not found: {args.clips_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.gt_dir.exists():
        print(f"ERROR: GT dir not found: {args.gt_dir}", file=sys.stderr)
        sys.exit(1)

    from footy_track.ball_eval import EvalDataset, run_benchmark
    from footy_track.ball_trackers.sam3_propagation import Sam3PropagationTracker

    print("Loading dataset...")
    dataset = EvalDataset.from_dirs(args.clips_dir, args.gt_dir)
    clips = dataset.clips[: args.max_clips]
    from footy_track.ball_eval.dataset import EvalDataset as DS
    subset = DS(clips)

    print(f"Evaluating {len(clips)} clips:")
    for c in clips:
        print(f"  {c.name}: {c.total_frames} frames, {c.ball_present_count()} ball-present")

    tracker = Sam3PropagationTracker(propagation_window=args.prop_window)

    print(f"\n--- Running SAM3 propagation (window={args.prop_window}) ---")
    t0 = time.perf_counter()
    result = run_benchmark(tracker, subset, method_name="sam3-propagation")
    total_time = time.perf_counter() - t0

    summary = tracker.summary()

    print(f"\n=== Results (total wall-time: {total_time:.1f}s) ===")
    print(result.table() if hasattr(result, "table") else "")

    print("\n=== Propagation Statistics ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Build output report.
    clip_data = []
    for cm in result.clip_metrics:
        clip_data.append(
            {
                "clip": cm.clip_name,
                "frames": cm.total_frames,
                "ball_present": cm.ball_present_frames,
                "center_within_radius_pct": round(cm.center_within_radius_pct, 2),
                "mean_center_dist_px": round(cm.mean_center_dist_px, 1),
                "mean_iou": round(cm.mean_iou, 3),
                "recall_pct": round(cm.recall_pct, 1),
                "tracking_failures": cm.tracking_failures,
                "fps": round(cm.fps, 1),
                "peak_vram_mb": round(cm.peak_vram_mb, 0),
            }
        )

    report = {
        "method": "sam3-propagation",
        "propagation_window": args.prop_window,
        "total_wall_time_s": round(total_time, 1),
        "aggregate": {
            "center_within_radius_pct": round(result.mean_center_within_radius, 2),
            "mean_center_dist_px": round(result.mean_center_dist_px, 1),
            "mean_iou": round(result.mean_iou, 3),
            "mean_recall_pct": round(result.mean_recall, 1),
            "total_failures": result.total_failures,
            "mean_fps": round(result.mean_fps, 1),
        },
        "propagation_stats": summary,
        "clips": clip_data,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
