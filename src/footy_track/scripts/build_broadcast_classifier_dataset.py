"""Build the authority-based training set for the broadcast-view classifier (Yes/No).

    python -m footy_track.scripts.build_broadcast_classifier_dataset \\
        --roboflow /mnt/storage/footy_data/classifier_dataset/roboflow_dataset_10 \\
        --out /mnt/storage/footy_data/classifier_dataset/broadcast_20261118_ab12cd

Layout written (ultralytics folder format, what ``train_broadcast_classifier`` reads)::

    <out>/train/{Yes,No}/   Roboflow ``train`` images that the split authority allows + Broadcast-review
                            frames whose window is ``train``
    <out>/valid/{Yes,No}/   Broadcast-review frames of ``eval_in_match`` windows ONLY (never held-out)
    <out>/manifest.json     counts per source / match / split / class, file-list hashes, the split-register
                            hashes and the broadcast dataset version this was built against

Nothing here decides a split: every decision is ``footy_track.splits`` (via
``export_broadcast_review.export`` for the review frames, ``splits.parse_image_name`` + ``split_of`` for
the Roboflow images). Roboflow ``valid`` / ``test`` images are not used: they cannot be placed in an eval
window, so they would be neither a clean validation set nor safe training data.

Roboflow images (``<match>_<frame>_jpg.rf.<hash>.jpg``) name a match but usually not a clip:

* from a held-out match            -> excluded (``heldout``);
* from a train match, clip known   -> kept only if the clip's split is ``train``;
* from a train match, clip unknown -> kept, counted as ``match_only`` (an in-match eval window cannot be
  ruled out; the review-frame eval windows are ``<match>_wNNNN`` of full-match videos, which the
  Roboflow frames of those matches may overlap);
* names no known match             -> excluded (``unmapped``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from footy_track import splits as authority
from footy_track.datasets import snapshot
from footy_track.labeller import datasets
from footy_track.scripts import export_broadcast_review as review_export

CLASSES = ("Yes", "No")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def _sha(paths: list[Path], root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def add_roboflow_train(rf_dir: Path, out: Path) -> dict:
    """Copy the allowed Roboflow ``train`` images into ``out/train``; return counts."""
    counts: Counter = Counter()
    by_match: dict[str, Counter] = defaultdict(Counter)
    excluded_clips: Counter = Counter()
    for cls in CLASSES:
        for src in sorted((rf_dir / "train" / cls).glob("*")):
            if src.suffix.lower() not in IMG_EXT:
                continue
            ref = authority.parse_image_name(src.name)
            if ref.match_key is None and ref.clip_id is None:
                cat = "unmapped"
            elif ref.clip_id is None:
                cat = "heldout" if authority.split_of(ref.match_key) == authority.EVAL_HELDOUT else "match_only"
            else:
                sp = authority.split_of(ref.match_key, ref.clip_id)
                cat = {authority.TRAIN: "train_clip", authority.EVAL_HELDOUT: "heldout"}.get(sp, "eval_clip")
            counts[cat] += 1
            by_match[ref.match_key or ref.stem][f"{cat}:{cls}"] += 1
            if cat in ("heldout", "eval_clip", "unmapped"):
                excluded_clips[ref.clip_id or ref.match_key or ref.stem] += 1
                continue
            dst = out / "train" / cls / src.name  # keep the Roboflow name: splits.parse_image_name reads it back
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
    return {"counts": dict(counts), "by_match": {k: dict(v) for k, v in sorted(by_match.items())},
            "excluded": dict(excluded_clips)}


def tally(out: Path) -> dict:
    """Per sub-split / class / source / match file counts and a content hash per sub-split."""
    res: dict = {}
    for sub in ("train", "valid"):
        files = [p for p in (out / sub).rglob("*") if p.suffix.lower() in IMG_EXT]
        per_class = Counter(p.parent.name for p in files)
        per_source: Counter = Counter()
        per_match: dict[str, Counter] = defaultdict(Counter)
        for p in files:
            if ".rf." in p.name:  # Roboflow export name (<clip>_<frame>_jpg.rf.<hash>.jpg)
                ref = authority.parse_image_name(p.name)
                src, match = "roboflow", ref.match_key or ref.stem
            else:
                stem = p.name.rsplit("_f", 1)[0]
                src, match = "broadcast_review", authority.match_key_of(stem) or stem
            per_source[src] += 1
            per_match[match][p.parent.name] += 1
        res[sub] = {"n": len(files), "per_class": dict(per_class), "per_source": dict(per_source),
                    "per_match": {m: dict(c) for m, c in sorted(per_match.items())}, "sha256": _sha(files, out)}
    return res


def build(rf_dir: Path, out: Path, *, labeller_dataset: str = "broadcast_review", allow_unmapped: bool = False) -> dict:
    rf = add_roboflow_train(rf_dir, out)
    review = review_export.export(datasets.get(labeller_dataset), out, allow_unmapped=allow_unmapped)
    ver = snapshot.latest_version("broadcast")
    snap_manifest = json.loads((snapshot.version_dir("broadcast", ver) / "manifest.json").read_text()) if ver else None
    t = tally(out)
    manifest = {
        "kind": "broadcast_classifier_dataset",
        "roboflow_dir": str(rf_dir),
        "roboflow": rf,
        "broadcast_review_export": review,
        "tally": t,
        "broadcast_dataset": {
            "version": ver,
            "content_hash": snap_manifest and snap_manifest["content_hash"],
            "snapshot_counts_by_split": snap_manifest and snap_manifest["counts"]["by_split"],
            "snapshot_counts_by_split_kind": snap_manifest and snap_manifest["counts"]["by_split_kind"],
            "split_registers": snap_manifest and snap_manifest["split_registers"],
        },
        "note": "validation = eval_in_match review frames only; held-out never used; Roboflow valid/test unused",
    }
    # the review export reads the label files directly; flag it when they have moved past the snapshot
    if snap_manifest:
        rev_train = t["train"]["per_source"].get("broadcast_review", 0)
        rev_valid = t["valid"]["per_source"].get("broadcast_review", 0)
        want = snap_manifest["counts"]["by_split"]
        manifest["matches_snapshot"] = bool(rev_train == want.get("train", 0) and rev_valid == want.get("eval_in_match", 0))
        manifest["review_counts_vs_snapshot"] = {"train": [rev_train, want.get("train", 0)],
                                                 "eval_in_match": [rev_valid, want.get("eval_in_match", 0)]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roboflow", type=Path, required=True, help="Roboflow/YOLO folder export (train/{Yes,No}/...)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="broadcast_review", help="labeller dataset name in the registry")
    ap.add_argument("--allow-unmapped", action="store_true")
    args = ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} is not empty", file=sys.stderr)
        return 2
    try:
        m = build(args.roboflow, args.out, labeller_dataset=args.dataset, allow_unmapped=args.allow_unmapped)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps({k: m[k] for k in ("roboflow", "broadcast_review_export", "matches_snapshot") if k in m}, indent=1))
    print(json.dumps({s: {k: v for k, v in t.items() if k != "per_match"} for s, t in m["tally"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
