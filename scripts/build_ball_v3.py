"""Build ``ball_v3``: ONE single-class (ball) YOLO dataset from Roboflow v11 + ``ball_v2``.

Read-only on every source (images are hard-linked, labels rewritten); never touches the feature
store, the panel dirs or ball_v2. Train/val membership comes from ``footy_track.splits``.

Rules
-----
* ``ball_v2/train``                     -> train (as is).
* ``ball_v2/val``                       -> val   (in-match eval clips).
* ``ball_v2/test``                      -> DROPPED (heldout matches / panel frames must not sit in any split).
* Roboflow v11 train/valid/test images  -> train, when they carry a ball box (classes ``ball``/``in_play_ball`` -> 0) and
    - the (clip, frame) is not already in ball_v2 (ball_v2's label wins, it carries Ball Check corrections), and
    - the clip is a train clip. v11 images from ``eval_in_match`` clips go to VAL instead (extra val coverage),
      heldout-match images are dropped. Frames cut from the full Man City video are mapped to their
      segment clip via ``segments.json``; unresolvable ones are dropped.
    - v11 images with NO ball box are dropped (the ball may simply be unlabelled there).
* Nothing from ``ball_gt_marks_ball_eval_v1`` (the blind panel marks) is read at all.
* Ball Check ``not_ball`` boxes are NOT used (a rejected candidate says nothing about the rest of the frame).

    python scripts/build_ball_v3.py --out /mnt/storage/footy_data/training_datasets/ball_v3
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shutil
from pathlib import Path

import yaml
from PIL import Image

from footy_track import splits as S
from footy_track.splits import EVAL_HELDOUT, EVAL_IN_MATCH, TRAIN

FD = Path(os.environ.get("FOOTY_DATA", "/mnt/storage/footy_data"))
V11 = FD / "detection_dataset" / "roboflow_dataset_11"
V2 = FD / "training_datasets" / "ball_v2"
BALL_V11_IDS = {"0", "2"}  # ball, in_play_ball (v11 data.yaml order)
FULL_FRAME_STRIDE = 25
FULL_STEM = "arsenal_mancity_20250925"  # parse_image_name splits the frame number off: stem + frame
SEGMENTS = FD / "arsenal_mancity" / "split_video_broadcast_frames" / "segments.json"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _full_match_clip(stem: str, n: int | None, segs: list[dict]) -> str | None:
    if stem != FULL_STEM or n is None:
        return None
    n *= FULL_FRAME_STRIDE  # full_video_frames/ holds every 25th video frame (7143 pngs = 178584 / 25)
    for s in segs:
        if s["start_frame"] <= n <= s["end_frame"]:
            return S.canonical_clip_id(f"arsenal_mancity_20250925_seg{s['index']:03d}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="ball_v3")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    segs = json.loads(SEGMENTS.read_text())["segments"]

    plan: list[dict] = []  # {split, src_img, name, boxes(list of 'cx cy w h'), source, clip, frame}
    taken: set[tuple[str, int]] = set()
    drops: collections.Counter = collections.Counter()

    # ---- ball_v2 (train + val; test dropped) ----
    for sp in ("train", "val", "test"):
        for im in sorted(glob.glob(f"{V2}/images/{sp}/*.jpg")):
            stem = Path(im).stem
            clip, fr = stem.rsplit("_", 1)
            cid = S.canonical_clip_id(clip)
            key = (cid, int(fr))
            taken.add(key)
            if sp == "test":
                drops["ball_v2/test (heldout match, dropped)"] += 1
                continue
            lab = Path(im.replace("/images/", "/labels/")).with_suffix(".txt")
            boxes = [ln for ln in lab.read_text().split("\n") if ln.strip()]
            plan.append(
                dict(
                    split=sp,
                    src=Path(im),
                    name=Path(im).name,
                    boxes=boxes,
                    source=f"ball_v2/{sp}",
                    clip=cid,
                    frame=int(fr),
                )
            )

    # ---- Roboflow v11 ----
    seen_v11: set[tuple[str, int]] = set()
    for sub in ("train", "valid", "test"):
        for im in sorted(glob.glob(f"{V11}/{sub}/images/*")):
            if Path(im).suffix.lower() not in IMG_EXT:
                continue
            ref = S.parse_image_name(Path(im).name)
            cid = ref.clip_id or _full_match_clip(ref.stem, ref.frame, segs)
            tag = f"v11/{sub}"
            if cid is None or ref.frame is None:
                drops[f"{tag}: clip unresolvable"] += 1
                continue
            key = (cid, ref.frame) if ref.clip_id else (ref.stem, ref.frame)
            lab = Path(im.replace("/images/", "/labels/")).with_suffix(".txt")
            boxes = []
            for ln in lab.read_text().split("\n") if lab.exists() else []:
                p = ln.split()
                if len(p) >= 5 and p[0] in BALL_V11_IDS:
                    boxes.append("0 " + " ".join(p[1:5]))
            if not boxes:
                drops[f"{tag}: no ball box (ball may be unlabelled)"] += 1
                continue
            if key in taken or key in seen_v11:
                drops[f"{tag}: duplicate (clip,frame) already in ball_v2/v11"] += 1
                continue
            seen_v11.add(key)
            mk = ref.match_key or S.match_key_of(cid)
            sp = S.split_of(mk, cid)
            if sp == EVAL_HELDOUT:
                drops[f"{tag}: heldout match"] += 1
                continue
            split = "val" if sp == EVAL_IN_MATCH else "train"
            plan.append(
                dict(
                    split=split,
                    src=Path(im),
                    name="rf11_" + Path(im).name,
                    boxes=boxes,
                    source=tag,
                    clip=cid,
                    frame=ref.frame,
                )
            )

    # ---- sanity: no panel frames / heldout in any emitted split ----
    bad = [p for p in plan if S.split_of(S.match_key_of(p["clip"]), p["clip"]) == EVAL_HELDOUT]
    assert not bad, f"heldout items in plan: {bad[:3]}"
    train_eval = [
        p for p in plan
        if p["split"] == "train" and S.split_of(S.match_key_of(p["clip"]), p["clip"]) == EVAL_IN_MATCH
    ]
    assert not train_eval, f"eval_in_match clip in train: {train_eval[:3]}"
    if any("ball_eval_v1" in str(p["src"]) for p in plan):
        raise SystemExit("ball_eval_v1 content leaked into plan")

    summ: dict = {"counts": {}, "by_source": {}, "by_clip_train": {}, "sizes": {}}
    for p in plan:
        k = (p["split"], p["source"])
        d = summ["by_source"].setdefault(f"{p['split']}|{p['source']}", {"images": 0, "boxes": 0, "empty": 0})
        d["images"] += 1
        d["boxes"] += len(p["boxes"])
        d["empty"] += not p["boxes"]
    for sp in ("train", "val"):
        ps = [p for p in plan if p["split"] == sp]
        summ["counts"][sp] = {
            "images": len(ps),
            "boxes": sum(len(p["boxes"]) for p in ps),
            "empty_frames": sum(not p["boxes"] for p in ps),
        }
    summ["by_clip_train"] = dict(
        collections.Counter(p["clip"] for p in plan if p["split"] == "train").most_common()
    )
    summ["by_match_train"] = dict(
        collections.Counter(
            S.match_key_of(p["clip"]) or p["clip"] for p in plan if p["split"] == "train"
        ).most_common()
    )
    print(json.dumps({"counts": summ["counts"], "by_source": summ["by_source"], "drops": drops}, indent=1))
    if a.dry_run:
        return

    out = a.out
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    sizes: collections.Counter = collections.Counter()
    prov = []
    for p in plan:
        sp = p["split"]
        dst = out / "images" / sp / p["name"]
        _link(p["src"], dst)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp / Path(p["name"]).with_suffix(".txt").name).write_text(
            "\n".join(p["boxes"]) + ("\n" if p["boxes"] else "")
        )
        with Image.open(p["src"]) as im:
            sizes[f"{sp}|{im.size[0]}x{im.size[1]}"] += 1
        prov.append({k: str(v) if k == "src" else v for k, v in p.items() if k != "boxes"} | {"n_boxes": len(p["boxes"])})
    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(out), "train": "images/train", "val": "images/val", "nc": 1, "names": ["ball"]},
            sort_keys=False,
        )
    )
    (out / "provenance.jsonl").write_text("\n".join(json.dumps(r) for r in prov) + "\n")
    manifest = {
        "tag": a.tag,
        "built_from": {"ball_v2": str(V2), "roboflow_v11": str(V11)},
        "split_authority": "footy_track.splits.split_of",
        "rules": __doc__,
        "counts": summ["counts"],
        "by_source": summ["by_source"],
        "drops": dict(drops),
        "image_sizes": dict(sizes),
        "train_by_match": summ["by_match_train"],
        "train_by_clip": summ["by_clip_train"],
        "excluded": {
            "ball_eval_v1": "never read",
            "heldout_matches": "dropped from every split (incl. ball_v2/test)",
            "not_ball_boxes": "not used (no safe empty-frame inference)",
        },
    }
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=120))
    print("wrote", out)


if __name__ == "__main__":
    main()
