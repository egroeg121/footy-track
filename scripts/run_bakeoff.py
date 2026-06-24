"""Bake-off: run cheap ball-tracking methods on the user's real GT marks.

Runs ROI-YOLO (trained model) and SOT-VitTrack on the 5 marked clips.
Results are persisted to docs/bakeoff_results.json and printed as a table.

Usage:
    uv run python scripts/run_bakeoff.py [--clips-dir PATH] [--output PATH]

Compute note: heavy methods (SAM3/SAM2) are NOT run here — they must be
serialized and run separately. This script covers the cheap methods only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from footy_track.ball_eval import EvalDataset, compare_methods, run_benchmark
from footy_track.ball_trackers.roi_yolo import RoiYoloTracker

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
OUTPUT_DEFAULT = PROJECT_ROOT / "docs" / "bakeoff_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bake-off on real marks")
    parser.add_argument(
        "--clips-dir",
        type=pathlib.Path,
        default=CLIPS_DIR_DEFAULT,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--gt-dir",
        type=pathlib.Path,
        default=GT_DIR_DEFAULT,
        help="Directory containing JSONL GT mark files (default: iCloud ball_gt_marks)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=OUTPUT_DEFAULT,
        help="JSON output path for persisted results",
    )
    parser.add_argument(
        "--skip-sot",
        action="store_true",
        help="Skip VitTrack SOT (saves time if only testing ROI-YOLO)",
    )
    args = parser.parse_args()

    if not args.clips_dir.exists():
        print(f"ERROR: clips directory not found: {args.clips_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.gt_dir.exists():
        print(f"ERROR: GT directory not found: {args.gt_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from clips: {args.clips_dir}")
    print(f"                   GT dir: {args.gt_dir}")
    dataset = EvalDataset.from_dirs(args.clips_dir, args.gt_dir)

    print(f"\nDataset: {len(dataset.clips)} clips")
    for clip in dataset:
        print(
            f"  {clip.name}: {clip.total_frames} frames total, "
            f"{clip.labelled_frame_count()} labelled, "
            f"{clip.ball_present_count()} ball-present"
        )

    results = []

    # --- Method 1: ROI-YOLO with trained detector ---
    print("\n--- Running ROI-YOLO (trained model) ---")
    t0 = time.perf_counter()
    roi_tracker = RoiYoloTracker()  # uses trained model by default
    roi_result = run_benchmark(roi_tracker, dataset, method_name="roi-yolo-trained")
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    results.append(roi_result)

    # --- Method 2: VitTrack SOT ---
    if not args.skip_sot:
        print("\n--- Running VitTrack SOT ---")
        try:
            from footy_track.ball_trackers.sot_vittrack import VitTrackSOT  # noqa: PLC0415, I001

            t0 = time.perf_counter()
            sot_tracker = VitTrackSOT()
            sot_result = run_benchmark(sot_tracker, dataset, method_name="sot-vittrack")
            print(f"  done in {time.perf_counter() - t0:.1f}s")
            results.append(sot_result)
        except Exception as e:
            print(f"  VitTrack SOT failed: {e}")

    # --- Comparison table ---
    print("\n=== Bake-off Results ===")
    table = compare_methods(results)
    print(table)

    # Per-clip center-distance breakdown
    print("\n=== Per-clip center-distance breakdown ===")
    header = f"{'Clip':<25} {'Method':<24} {'Ctr%':>6} {'CtrPx':>7} {'Recall%':>8} {'FPS':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        for cm in r.clip_metrics:
            print(
                f"{cm.clip_name:<25} {r.method_name:<24} "
                f"{cm.center_within_radius_pct:>6.1f} "
                f"{cm.mean_center_dist_px:>7.1f} "
                f"{cm.recall_pct:>8.1f} "
                f"{cm.fps:>6.1f}"
            )

    # Persist results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clips_dir": str(args.clips_dir),
        "methods": [r.to_dict() for r in results],
    }
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults persisted to: {args.output}")


if __name__ == "__main__":
    main()
