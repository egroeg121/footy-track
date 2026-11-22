"""Write review proposals for one match from its combined calibration (design ``calibration_match_flow.md`` 3b).

    PYTHONPATH=src python -m footy_track.scripts.calibration_proposals_export \\
        --match arsenal_mancity --gt-dir /path/to/scratch_gt [--seed 0] [--n 300] [--dry-run]

Reads ``<calibration root>/<version>/<match>.parquet`` (``combine_match`` output), samples frames (stratified, fixed
seed) and writes ``<gt-dir>/calibration_proposals/<segment clip>.json`` in the format the review page loads. There is
deliberately no default ``--gt-dir``: pass the live ``ball_gt_marks`` dir only when you mean to. Details and
conventions: ``footy_track.calibration.proposals_export``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from footy_track.calibration import match_calibration as mc
from footy_track.calibration import proposals_export as pe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True, help=f"match dir name, e.g. {', '.join(mc.KNOWN_MATCHES)}")
    ap.add_argument("--gt-dir", type=Path, required=True, help="dir that holds calibration_proposals/ (the labeller's GT dir)")
    ap.add_argument("--version", default=mc.DEFAULT_VERSION)
    ap.add_argument("--root", type=Path, default=None, help="calibration root (default $FOOTY_CALIBRATION_ROOT or <data>/calibration)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=pe.TARGET_FRAMES, help="target sample size per match")
    ap.add_argument("--dry-run", action="store_true", help="sample and project, write nothing")
    ap.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = ap.parse_args(argv)

    rows = pe.load_rows(args.match, version=args.version, root=args.root)
    s = pe.export_proposals(rows, args.gt_dir, seed=args.seed, target=args.n, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(s, indent=1))
        return 0
    print(f"{args.match} {s['run_id']}: picked {s['n_picked']} frames, wrote {s['n_written']} proposals in "
          f"{s['n_clips']} clips ({s['skipped_few_points']} skipped: < {pe.MIN_POINTS} visible keypoints)"
          + (" [dry run]" if args.dry_run else ""))
    for name, d in s["design"].items():
        if not name.startswith("_"):
            print(f"  {name:<26} {d['kind']:<9} pool={d['pool_frames']:>6} expected={d['expected']:>7} picked={d['picked']:>4}")
    print(f"  {s['design']['_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
