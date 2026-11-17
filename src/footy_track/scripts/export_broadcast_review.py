"""Export Broadcast-review verdicts as classifier training data.

    python -m footy_track.scripts.export_broadcast_review \\
        --dataset broadcast_review --out data/classifier_dataset/broadcast_review_v1

Writes the ultralytics folder layout ``<out>/train/{Yes,No}/<match>_f<frame>.jpg`` (class names =
``EnumBroadcastClassification``) that ``scripts/train_classifier.py --local-dataset`` reads. ``yes`` ->
``Yes``, ``no`` -> ``No``; ``unsure`` is skipped. Matches in ``--valid-matches`` go to ``valid/`` (keep a
match whole on one side so near-identical frames cannot leak). Pass ``--into`` an existing
folder dataset (e.g. the Roboflow export) to add the reviewed frames to it. Re-running is idempotent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from footy_track.labeller import broadcast_review as br
from footy_track.labeller import datasets

CLASS = {"yes": "Yes", "no": "No"}


def export(ds: datasets.Dataset, out: Path, valid_matches: set[str] | None = None) -> dict:
    valid_matches = valid_matches or set()
    counts = {"Yes": 0, "No": 0, "unsure": 0, "missing_image": 0}
    root = ds.marks_dir / "broadcast_review"
    for store in sorted(root.glob("*.json")) if root.is_dir() else []:
        match = store.stem
        for frame, rec in sorted(br.read_store(ds, match).items()):
            cls = CLASS.get(rec["verdict"])
            if cls is None:
                counts["unsure"] += 1
                continue
            src = br.frame_path(ds, match, frame)
            if not src.is_file():
                counts["missing_image"] += 1
                continue
            dst_dir = out / ("valid" if match in valid_matches else "train") / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{match}_f{frame}.jpg"
            if not dst.exists():
                shutil.copy2(src, dst)
            counts[cls] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="broadcast_review", help="labeller dataset name in the registry")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--into", action="store_true", help="--out is an existing dataset: add to it")
    ap.add_argument("--valid-matches", default="", help="comma-separated match keys to put in valid/")
    args = ap.parse_args(argv)
    if not args.into and args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} is not empty (use --into to add to an existing dataset)", file=sys.stderr)
        return 2
    counts = export(datasets.get(args.dataset), args.out, {m for m in args.valid_matches.split(",") if m})
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
