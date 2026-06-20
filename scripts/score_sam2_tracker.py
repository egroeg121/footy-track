"""Score the SAM2 box-prompted ball tracker (bake-off method B) against eval data.

Usage::

    uv run python scripts/score_sam2_tracker.py eval_data/clips/ [--model sam2_b.pt] [--roi-scale 3.0]

Prints the MethodResult table when done.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score SAM2 ball tracker (bake-off method B)"
    )
    ap.add_argument(
        "clips_dir", type=Path, help="Directory with video + .jsonl label files"
    )
    ap.add_argument("--model", default="sam2_b.pt", help="Ultralytics SAM2 model tag")
    ap.add_argument(
        "--roi-scale", type=float, default=3.0, help="ROI crop scale factor"
    )
    ap.add_argument("--device", default=None, help="torch device (cpu/mps/cuda)")
    ap.add_argument(
        "--conf", type=float, default=0.5, help="SAM2 mask confidence threshold"
    )
    ap.add_argument(
        "--output", type=Path, default=None, help="Save JSON results to this path"
    )
    args = ap.parse_args()

    from footy_track.ball_eval import EvalDataset, run_benchmark  # noqa: PLC0415
    from footy_track.ball_tracking import Sam2BallTracker  # noqa: PLC0415

    print(f"Loading eval dataset from {args.clips_dir} ...")
    dataset = EvalDataset.from_dir(args.clips_dir)
    print(dataset.summary())

    tracker = Sam2BallTracker(
        model_name=args.model,
        roi_scale=args.roi_scale,
        device=args.device,
        conf_threshold=args.conf,
    )

    print(
        f"\nRunning SAM2 tracker (model={args.model}, roi_scale={args.roi_scale}) ..."
    )
    result = run_benchmark(
        tracker, dataset, method_name=f"sam2-b ({args.model})", verbose=True
    )

    print("\n" + "=" * 70)
    print(result.table())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
