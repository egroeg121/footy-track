"""Inspect or create the labeller's dataset registry.

    python -m footy_track.scripts.labeller_datasets list
    python -m footy_track.scripts.labeller_datasets init [--data-root DIR] [--force]

The registry is ``$FOOTY_LABELLER_DATASETS`` (default
``/mnt/storage/footy_data/labeller_datasets.json``); see
``footy_track/labeller/datasets.py``. ``init`` writes the seed (``calib`` as the
default, plus ``nottinghamforest``) only if the file does not exist yet, and never
overwrites an existing registry without ``--force``. The running server
re-reads the file when it changes, so no restart is needed to add a dataset.
"""

from __future__ import annotations

import argparse
import sys

from footy_track.labeller import datasets as D

DEFAULT_DATA_ROOT = "/mnt/storage/footy_data"


def seed_entries(data: str = DEFAULT_DATA_ROOT) -> dict[str, dict]:
    """The two datasets that used to be two servers (:8011 calib, :8012 nottinghamforest)."""
    return {
        "calib": {
            "label": "Calibration / Ball Check (calib)",
            "clips_dir": f"{data}/eval_clips_kf10",
            "marks_dir": f"{data}/ball_gt_marks",
            # No candidates_dir on purpose: the :8011 server never had one, so Ball Check
            # there serves the sidecar queue. Add "candidates_dir" to switch it on.
            "extra_clips_dirs": [f"{data}/ballcheck_clips_kf10"],
        },
        "nottinghamforest": {
            "label": "Arsenal v Nottingham Forest (held-out)",
            "clips_dir": f"{data}/arsenal_nottinghamforest/split_video_broadcast_frames_kf10",
            "marks_dir": f"{data}/ball_gt_marks_nottinghamforest",
            "candidates_dir": f"{data}/ball_candidates_nottinghamforest",
            "ball_check_list": f"{data}/nottinghamforest_label_queue/ball_check_list_v1.json",
        },
    }


def cmd_list(_args: argparse.Namespace) -> int:
    print(f"# {D.registry_path()}")
    if D.registry_error():
        print(f"ERROR: {D.registry_error()}", file=sys.stderr)
        return 1
    for ds in D.all_datasets():
        star = "*" if ds.is_default else " "
        print(f"{star} {ds.name:<12} marks={ds.marks_dir}  clips={ds.clips_dir}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    path = D.registry_path()
    if path.exists() and not args.force:
        print(f"{path} exists; leaving it alone (use --force to overwrite)")
        return 0
    D.write_registry(path, "calib", seed_entries(args.data_root))
    print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    i = sub.add_parser("init")
    i.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_init)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
