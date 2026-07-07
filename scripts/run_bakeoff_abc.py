"""Bake-off ft-5hd.2: run the harness against methods A/B/C.

This is the assembled scratch entrypoint for the ball-tracking bake-off
(parent ft-5hd). It wires together:

  - Harness:      footy_track.ball_eval (ft-1my)
  - Method A SOT: footy_track.ball_trackers.sot_vittrack.VitTrackSOT (ft-ztw)
  - Method B SAM2: footy_track.ball_trackers.sam2_tracker.Sam2BallTracker (ft-xps)
  - Method C ROI-YOLO: footy_track.ball_trackers.roi_yolo.RoiYoloTracker (ft-1d9)

Eval data comes from ``eval_data/clips/`` (videos) with GT sidecars produced
by ``scripts/build_eval_sidecars.py`` (ft-5hd.1) from the ball GT marks at
``~/code/footy_data/ball_gt_marks``.

Usage::

    # List the registered methods without running anything (smoke check).
    uv run python scripts/run_bakeoff_abc.py --list

    # Run all three methods against every eval clip in eval_data/clips.
    uv run python scripts/run_bakeoff_abc.py

    # Run only a subset (e.g. skip the GPU-heavy SAM2 method, ft-5hd.3
    # tracks unblocking it) against a single clip for a fast smoke run.
    uv run python scripts/run_bakeoff_abc.py --methods sot,roi-yolo --max-clips 1

Method B (SAM2) needs the ``sam2_b.pt`` checkpoint downloaded (see ft-5hd.3);
if the checkpoint / GPU isn't available in this environment, use
``--methods`` to exclude it, or expect it to fail per-clip and be reported
as 0 scored clips rather than crashing the whole run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections.abc import Callable

from footy_track.ball_eval import (
    EvalDataset,
    MethodResult,
    compare_methods,
    run_benchmark,
)
from footy_track.ball_eval.interface import BallTracker

PROJECT_ROOT = pathlib.Path(__file__).parents[1]
CLIPS_DIR_DEFAULT = PROJECT_ROOT / "eval_data" / "clips"
OUTPUT_DEFAULT = PROJECT_ROOT / "docs" / "bakeoff_results.json"


def _make_sot() -> BallTracker:
    from footy_track.ball_trackers.sot_vittrack import VitTrackSOT  # noqa: PLC0415

    return VitTrackSOT()


def _make_sam2() -> BallTracker:
    from footy_track.ball_trackers.sam2_tracker import Sam2BallTracker  # noqa: PLC0415

    return Sam2BallTracker()


def _make_roi_yolo() -> BallTracker:
    from footy_track.ball_trackers.roi_yolo import RoiYoloTracker  # noqa: PLC0415

    return RoiYoloTracker()


# The bake-off method registry (ft-5hd methods A/B/C). Keys are the
# `--methods` CLI values; values are (display name, lazy-constructor).
# Constructors are lazy so `--list` and import-time checks never touch
# GPU/network/weights.
METHOD_REGISTRY: dict[str, tuple[str, Callable[[], BallTracker]]] = {
    "sot": ("sot-vittrack (method A, ft-ztw)", _make_sot),
    "sam2": ("sam2-ball (method B, ft-xps)", _make_sam2),
    "roi-yolo": ("roi-yolo-trained (method C, ft-1d9)", _make_roi_yolo),
}
DEFAULT_METHOD_ORDER = ["sot", "sam2", "roi-yolo"]


def _parse_methods(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_METHOD_ORDER)
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    unknown = [k for k in keys if k not in METHOD_REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown method(s): {unknown}. Choose from {list(METHOD_REGISTRY)}"
        )
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips-dir",
        type=pathlib.Path,
        default=CLIPS_DIR_DEFAULT,
        help="Directory with <clip>.mp4 + <clip>.jsonl sidecars (default: eval_data/clips)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help=f"Comma-separated subset of {list(METHOD_REGISTRY)} (default: all three)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limit to the first N eval clips (useful for a fast smoke run)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=OUTPUT_DEFAULT,
        help="JSON output path for persisted results",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Enumerate the registered methods and available eval clips, then exit "
        "without running any inference. Use to smoke-check the assembly.",
    )
    args = parser.parse_args()

    method_keys = _parse_methods(args.methods)

    if args.list:
        print("Registered bake-off methods:")
        for key in method_keys:
            name, _ = METHOD_REGISTRY[key]
            print(f"  {key:<10} -> {name}")
        if args.clips_dir.exists():
            n_clips = len(list(args.clips_dir.glob("*.jsonl")))
            print(f"\nEval clips dir: {args.clips_dir} ({n_clips} .jsonl sidecars found)")
        else:
            print(f"\nEval clips dir does not exist: {args.clips_dir}")
        return

    if not args.clips_dir.exists():
        print(f"ERROR: clips directory not found: {args.clips_dir}", file=sys.stderr)
        sys.exit(1)

    dataset = EvalDataset.from_dir(args.clips_dir)
    clips = list(dataset)
    if args.max_clips is not None:
        clips = clips[: args.max_clips]
        dataset = EvalDataset(clips=clips)

    print(f"Dataset: {len(clips)} clips from {args.clips_dir}")
    for clip in clips:
        print(
            f"  {clip.name}: {clip.total_frames} frames total, "
            f"{clip.labelled_frame_count()} labelled, "
            f"{clip.ball_present_count()} ball-present"
        )

    results: list[MethodResult] = []
    for key in method_keys:
        name, make_tracker = METHOD_REGISTRY[key]
        print(f"\n--- Running {name} ---")
        t0 = time.perf_counter()
        try:
            tracker = make_tracker()
            result = run_benchmark(tracker, dataset, method_name=name)
            print(f"  done in {time.perf_counter() - t0:.1f}s")
            results.append(result)
        except Exception as e:  # noqa: BLE001
            print(f"  {name} failed: {e!r}")

    if not results:
        print("\nNo methods produced results.")
        return

    print("\n=== Bake-off Results ===")
    print(compare_methods(results))

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
