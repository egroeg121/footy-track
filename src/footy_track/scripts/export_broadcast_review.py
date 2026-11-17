"""Export Broadcast-review verdicts as classifier training data.

    python -m footy_track.scripts.export_broadcast_review \\
        --dataset broadcast_review --out data/classifier_dataset/broadcast_review_v1

Writes the ultralytics folder layout ``<out>/<split>/{Yes,No}/<match>_f<frame>.jpg`` (class names =
``EnumBroadcastClassification``) that ``scripts/train_classifier.py --local-dataset`` reads. ``yes`` ->
``Yes``, ``no`` -> ``No``; ``unsure`` is skipped.

The split comes from the global authority (``footy_track.splits``; docs/data.md "Train/eval
partitioning"), not from hand-picked matches. A reviewed frame belongs to a *window* of consecutive
frames (``<match key>_w<frame // 6000>``, the clip-equivalent here), so near-identical neighbours
always land on the same side:

* ``train``              -> ``train/``
* ``eval_in_match``      -> ``valid/`` (windows of train matches registered/hashed as eval)
* ``eval_heldout_match`` -> left out; with ``--heldout-as-test`` -> ``test/``

A match is identified by its full-match video stem (the review's ``<match>`` key), resolved to the
register key through ``full_matches/manifest.jsonl``. An unresolvable match is an error (it might
be a held-out match): add it to the manifest/register, or pass ``--allow-unmapped`` to treat it as
a plain train-pool match keyed by its own name. ``--valid-matches`` stays as an explicit override
that forces whole matches into ``valid/``. Pass ``--into`` an existing folder dataset (e.g. the
Roboflow export) to add the reviewed frames to it. Re-running is idempotent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from footy_track import splits as authority
from footy_track.labeller import broadcast_review as br
from footy_track.labeller import datasets

CLASS = {"yes": "Yes", "no": "No"}


_DIR = {"train": "train", "val": "valid", "test": "test"}


def export(
    ds: datasets.Dataset,
    out: Path,
    valid_matches: set[str] | None = None,
    *,
    heldout_as_test: bool = False,
    allow_unmapped: bool = False,
) -> dict:
    valid_matches = valid_matches or set()
    counts = {"Yes": 0, "No": 0, "unsure": 0, "missing_image": 0, "heldout_skipped": 0}
    root = ds.marks_dir / "broadcast_review"
    stores = sorted(root.glob("*.json")) if root.is_dir() else []
    unmapped: dict[str, None] = {}
    plan: list[tuple[str, int, str, Path]] = []
    for store in stores:
        match = store.stem
        key = authority.match_key_of(match)
        if key is None and match not in valid_matches:
            unmapped[match] = None
        for frame, rec in sorted(br.read_store(ds, match).items()):
            cls = CLASS.get(rec["verdict"])
            if cls is None:
                counts["unsure"] += 1
                continue
            if match in valid_matches:
                sub: str | None = "valid"
            else:
                split = authority.split_of(
                    key or match, authority.broadcast_window_id(key or match, frame)
                )
                sub = authority.export_dir(split, heldout_as_test=heldout_as_test)
                sub = _DIR.get(sub) if sub else None
            if sub is None:
                counts["heldout_skipped"] += 1
                continue
            plan.append((match, frame, cls, out / sub / cls))
    if unmapped and not allow_unmapped:
        raise ValueError(
            f"cannot map match(es) {sorted(unmapped)} to the split register (not in "
            "full_matches/manifest.jsonl or the register); fix that or pass --allow-unmapped"
        )
    for match, frame, cls, dst_dir in plan:
        src = br.frame_path(ds, match, frame)
        if not src.is_file():
            counts["missing_image"] += 1
            continue
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
    ap.add_argument("--valid-matches", default="", help="override: comma-separated match keys forced into valid/")
    ap.add_argument("--heldout-as-test", action="store_true", help="write held-out-match frames to test/ (default: leave out)")
    ap.add_argument("--allow-unmapped", action="store_true", help="treat matches missing from the register as train-pool")
    args = ap.parse_args(argv)
    if not args.into and args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} is not empty (use --into to add to an existing dataset)", file=sys.stderr)
        return 2
    try:
        counts = export(
            datasets.get(args.dataset), args.out, {m for m in args.valid_matches.split(",") if m},
            heldout_as_test=args.heldout_as_test, allow_unmapped=args.allow_unmapped,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
