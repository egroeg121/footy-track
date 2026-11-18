"""Combine the per-segment PTZ V3 calibration outputs into one file per match.

    PYTHONPATH=src nice -n 19 env OMP_NUM_THREADS=1 python -m footy_track.scripts.calibration_match_combine \\
        --all [--update] [--dry-run]
    ... --match arsenal_mancity --out-root /mnt/storage/footy_data/calibration --version ptz_v3

For every segment of ``<data>/<match>/split_video_broadcast_frames/segments.json`` the best result is picked
(the weak-segment recovery in ``<temporal>/rerun_weak`` only if its accepted fraction is higher than the main
run's) and ``<out-root>/<version>/<match>.parquet`` + ``<match>.manifest.json`` are written atomically. Re-running
with unchanged inputs is a no-op; a changed result is refused unless ``--update`` (old files go to ``_history/``).
Format and conventions: ``footy_track.calibration.match_calibration``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from footy_track.calibration import match_calibration as mc


def _summary(m: dict) -> str:
    t = m["totals"]
    return (
        f"{m['match']:<30} {m['status']:<9} {m['version_id']:<18} frames={t['frames']:>7} "
        f"segments={t['segments']:>3} recovered={t['segments_recovered']:>2} missing={t['segments_missing']:>2} "
        f"accepted={100 * t['accepted_frac']:5.1f}% calibrated={100 * t['calibrated_frac']:5.1f}%"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", action="append", default=[], help="match dir name (repeatable)")
    ap.add_argument("--all", action="store_true", help=f"every known match: {', '.join(mc.KNOWN_MATCHES)}")
    ap.add_argument("--rig", default=None, help="rig stem of the segment dirs (after '@'); default per known match")
    ap.add_argument("--version", default=mc.DEFAULT_VERSION)
    ap.add_argument("--out-root", type=Path, default=None, help="default: $FOOTY_CALIBRATION_ROOT or <data>/calibration")
    ap.add_argument("--data-dir", type=Path, default=None, help="footy_data root (segments.json); default $FOOTY_DATA_DIR")
    ap.add_argument("--temporal", type=Path, default=None, help="dir of the main runs (<data>/scratch/temporal)")
    ap.add_argument("--rerun", type=Path, default=None, help="dir of the recovery runs (default <temporal>/rerun_weak)")
    ap.add_argument("--update", action="store_true", help="replace a changed file (the old one is archived)")
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    args = ap.parse_args(argv)

    matches = list(args.match) + (list(mc.KNOWN_MATCHES) if args.all else [])
    if not matches:
        ap.error("give --match <name> or --all")
    rc = 0
    for match in dict.fromkeys(matches):
        try:
            m = mc.combine_match(
                match, out_root=args.out_root, version=args.version, rig_stem=args.rig, data_dir=args.data_dir,
                temporal=args.temporal, rerun_dir=args.rerun, update=args.update, dry_run=args.dry_run,
            )
        except mc.WouldOverwrite as exc:
            print(f"{match}: REFUSED {exc}", file=sys.stderr)
            rc = 2
            continue
        print(_summary(m))
        for w in m["warnings"]:
            print(f"    warning: {w}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
