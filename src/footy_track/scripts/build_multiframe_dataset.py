"""Build the camera-motion-compensated multi-frame dataset (CPU only).

For every Roboflow v11 image (same train/valid/test image ids as the existing
detector) and for human-GT frames of clips unseen by training (held-out), decode
frames t-k, t, t+k from the source clip, warp the neighbours into frame t's
coordinates and write one multi-page TIFF per image:

    page 0   gray(t-k) aligned to t        (Ultralytics stacks pages on axis 2)
    page 1-3 B, G, R of frame t            (decoded from video, not the v11 JPEG)
    page 4   gray(t+k) aligned to t

A neighbour that is unavailable (clip edge), crosses a cut or fails alignment is
replaced by gray(t) (diff == 0), recorded per image in ``meta.jsonl``.
Labels are copied unchanged from v11 (same pixel grid), so ids/splits are identical.

    uv run python -m footy_track.scripts.build_multiframe_dataset \
        --k 2 --out /mnt/storage/footy_data/multiframe_v1/k2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from footy_track.clip_id import canonical_clip_id
from footy_track.multiframe.align import (
    boxes_to_mask,
    estimate_pair,
    estimate_pair_lk,
    hist_distance,
    warp_to_ref,
)
from footy_track.multiframe.frames import (
    BALL_ID,
    V11_NAMES,
    ClipReader,
    machine_person_boxes,
    person_boxes,
)
from footy_track.multiframe.mapping import FOOTY_DATA, frame_count, stem_video
from footy_track.multiframe.v11 import V11_DIR, load_v11

DEFAULT_OUT = FOOTY_DATA / "multiframe_v1"
#: Bhattacharyya distance between consecutive frames above which we call it a cut.
CUT_THRESHOLD = 0.2  # normal frames <=0.09; every cut seen was >=0.37 (multiframe_cut_scan)
SPLIT_DIR = {"train": "train", "valid": "val", "test": "test", "heldout": "heldout"}
GT_DIR = FOOTY_DATA / "ball_gt_marks"
GT_CLASS = {"in_play_ball": 2, "player": 4, "referee": 6, "coach": 1, "person": 3, "player_sub": 5}


def gt_tier(tags: set[str]) -> int:
    if "labeller" in tags:
        return 1
    if "vittrack" in tags and "human_checked" in tags:
        return 2
    return 3


def heldout_items(train_clips: set[str], other_split_clips: set[str], other_split_frames: set):
    """Human-GT ball frames (tier 1/2) of clips absent from the v11 *train* split.

    ``strict`` = clip appears in no v11 split at all. Frames that are already v11 val
    or test images are skipped (evaluated there). Returns a list of dicts.
    """
    by_frame: dict[tuple[str, int], dict] = {}
    for gt in sorted(GT_DIR.glob("*.jsonl")):
        cid = canonical_clip_id(gt.stem)
        if cid in train_clips:
            continue
        video = stem_video(gt.stem)
        if video is None:
            continue
        for line in gt.read_text().splitlines():
            r = json.loads(line)
            if r.get("bbox") is None:
                continue
            tags = set(r.get("tags", []))
            cls = next((GT_CLASS[t] for t in tags if t in GT_CLASS), None)
            if cls is None or gt_tier(tags) > 2:
                continue
            b = r["bbox"]
            if isinstance(b, list):
                b = dict(zip("xywh", b))
            rec = by_frame.setdefault((cid, r["frame_index"]), {
                "clip_id": cid, "frame": r["frame_index"], "video": video, "stem": gt.stem,
                "labels": [], "tiers": []})
            rec["labels"].append((cls, b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, b["w"], b["h"]))
            if cls == BALL_ID:
                rec["tiers"].append(gt_tier(tags))
    out = []
    for (cid, frame), rec in sorted(by_frame.items()):
        if not rec["tiers"] or (cid, frame) in other_split_frames:
            continue
        n = frame_count(rec["video"])
        rec["n_frames"] = n
        rec["strict"] = cid not in other_split_clips
        # de-duplicate identical rows written under two aliases
        rec["labels"] = sorted(set(rec["labels"]))
        out.append(rec)
    return out


def write_stack(path: Path, planes: list[np.ndarray]) -> None:
    # 5-page TIFF: each page a 2-D uint8 plane. Ultralytics' imread (cv2.imdecodemulti)
    # stacks 2-D pages on axis 2 -> HxWx5. LZW(5) keeps it lossless and ~2x smaller.
    ok = cv2.imwritemulti(str(path), planes, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
    if not ok:
        raise OSError(f"failed to write {path}")


def align_neighbour(rd: ClipReader, a, t: int, tn: int, n_frames: int, mask, boxes, method: str, scale: int):
    """Return (aligned gray plane or None, status dict) for neighbour frame ``tn``."""
    st = {"frame": tn, "status": "ok", "ms": 0.0}
    if tn < 0 or tn >= n_frames:
        st["status"] = "edge"
        return None, st
    b = rd.get(tn)
    if b is None:
        st["status"] = "decode_fail"
        return None, st
    lo, hi = sorted((t, tn))
    hd = [hist_distance(rd.get(g), rd.get(g + 1)) for g in range(lo, hi)]
    st["hist_max"] = max(hd)
    if st["hist_max"] > CUT_THRESHOLD:
        st["status"] = "cut"
        return None, st
    t0 = time.perf_counter()
    if method == "lk":  # boxes (not a pixel mask) so the shared module builds its own mask
        pm = estimate_pair_lk(a, b, scale=scale, boxes_a=boxes, boxes_b=boxes)
    else:
        pm = estimate_pair(a, b, mask, mask, scale=scale, method=method)
    st["ms"] = (time.perf_counter() - t0) * 1000
    st.update(inliers=pm.inliers, residual=round(pm.residual, 3), reason=pm.reason)
    if not pm.ok:
        st["status"] = "align_fail"
        return None, st
    return cv2.cvtColor(warp_to_ref(b, pm.H_px), cv2.COLOR_BGR2GRAY), st


def process_clip(job):
    """Process all items of one video. job = (video, items, k, out_dir, method, scale)."""
    video, items, k, out_dir, method, scale = job
    rd = ClipReader(video, cache=8)
    n_frames = frame_count(video)
    metas = []
    for it in sorted(items, key=lambda x: x["frame"]):
        t = it["frame"]
        a = rd.get(t)
        if a is None:
            metas.append({"id": it["id"], "split": it["split"], "error": "decode_fail"})
            continue
        boxes = person_boxes(it["labels"]) if it["split"] != "heldout" else \
            [(x, y, w, h) for c, x, y, w, h in it["labels"] if c != BALL_ID]
        src = "labels"
        if not boxes:
            boxes = machine_person_boxes(video.stem, t) or []
            src = "machine" if boxes else "none"
        mask = boxes_to_mask(a.shape[:2], boxes, 0.25) if boxes else None
        gray_t = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        planes, nb = [], {}
        for name, tn in (("prev", t - k), ("next", t + k)):
            g, st = align_neighbour(rd, a, t, tn, n_frames, mask, boxes, method, scale)
            nb[name] = st
            planes.append(g if g is not None else gray_t)
        stack = [planes[0], *cv2.split(a), planes[1]]
        img_path = out_dir / "images" / SPLIT_DIR[it["split"]] / f"{it['id']}.tiff"
        write_stack(img_path, stack)
        lab_path = out_dir / "labels" / SPLIT_DIR[it["split"]] / f"{it['id']}.txt"
        lab_path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in it["labels"]))
        metas.append({"id": it["id"], "split": it["split"], "clip_id": it["clip_id"],
                      "video": str(video), "frame": t, "k": k, "mask_src": src,
                      "n_ball": sum(1 for c, *_ in it["labels"] if c == BALL_ID),
                      "strict_heldout": it.get("strict"), "prev": nb["prev"], "next": nb["next"]})
    rd.close()
    return metas


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--method", default="lk", help="lk (shared calibration.temporal module) | ecc | orb | ...")
    ap.add_argument("--scale", type=int, default=480)
    ap.add_argument("--workers", type=int, default=1, help="processes; keep 1 on the shared mini")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N items per split")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test", "heldout"])
    args = ap.parse_args(argv)
    cv2.setNumThreads(2)  # shared box: cap threads
    out_dir = args.out or DEFAULT_OUT / f"k{args.k}"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} exists and is not empty; refusing to overwrite")
    for s in args.splits:
        (out_dir / "images" / SPLIT_DIR[s]).mkdir(parents=True)
        (out_dir / "labels" / SPLIT_DIR[s]).mkdir(parents=True)

    v11 = load_v11()
    items, unmapped = [], Counter()
    for it in v11:
        if it.split not in args.splits:
            continue
        if not it.ref.mapped:
            unmapped[it.split] += 1
            continue
        items.append({"id": it.stem, "split": it.split, "frame": it.ref.frame, "video": it.ref.video,
                      "clip_id": it.ref.clip_id, "labels": it.labels})
    if "heldout" in args.splits:
        by = defaultdict(set)
        for it in v11:
            by[it.split].add(it.ref.clip_id)
        frames = {(it.ref.clip_id, it.ref.frame) for it in v11}  # already a v11 image
        for h in heldout_items(by["train"], by["valid"] | by["test"], frames):
            items.append({"id": f"{h['clip_id']}_{h['frame']:06d}", "split": "heldout",
                          "frame": h["frame"], "video": h["video"], "clip_id": h["clip_id"],
                          "labels": h["labels"], "strict": h["strict"]})
    if args.limit:
        cnt, keep = Counter(), []
        for it in items:
            if cnt[it["split"]] < args.limit:
                cnt[it["split"]] += 1
                keep.append(it)
        items = keep

    groups = defaultdict(list)
    for it in items:
        groups[it["video"]].append(it)
    jobs = [(v, its, args.k, out_dir, args.method, args.scale) for v, its in groups.items()]
    jobs.sort(key=lambda j: -len(j[1]))
    t0 = time.time()
    metas = []
    if args.workers == 1:
        results = map(process_clip, jobs)
    else:
        results = mp.Pool(args.workers).imap_unordered(process_clip, jobs)
    for i, m in enumerate(results, 1):
        metas.extend(m)
        print(f"[{i}/{len(jobs)}] clips done, {len(metas)} images, {time.time() - t0:.0f}s", flush=True)
    metas.sort(key=lambda m: (m["split"], m["id"]))
    with (out_dir / "meta.jsonl").open("w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")

    names = {i: n for i, n in enumerate(V11_NAMES)}
    base = {"path": str(out_dir), "names": names, "nc": len(names), "channels": 5}
    yaml = lambda d: "".join(f"{k}: {json.dumps(v)}\n" for k, v in d.items())  # JSON scalars are valid YAML
    (out_dir / "data.yaml").write_text(yaml({**base, "train": "images/train", "val": "images/val",
                                             "test": "images/test"}))
    (out_dir / "data_heldout.yaml").write_text(yaml({**base, "train": "images/train", "val": "images/heldout"}))
    status = Counter()
    for m in metas:
        for side in ("prev", "next"):
            if side in m:
                status[(m["split"], m[side]["status"])] += 1
    per_split = {s: {"images": sum(1 for m in metas if m["split"] == s and "error" not in m),
                     "balls": sum(m.get("n_ball", 0) for m in metas if m["split"] == s)}
                 for s in args.splits}
    manifest = {"k": args.k, "method": args.method, "scale": args.scale, "cut_threshold": CUT_THRESHOLD,
                "channels": ["gray(t-k)_aligned", "B(t)", "G(t)", "R(t)", "gray(t+k)_aligned"],
                "unmapped_v11_images": dict(unmapped), "per_split": per_split,
                "neighbour_status": {f"{s}/{st}": n for (s, st), n in sorted(status.items())},
                "seconds": round(time.time() - t0, 1)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
