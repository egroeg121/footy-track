"""Build the pseudo-label dataset in the exact v1 multiframe format (CPU, one process).

Format (identical to multiframe_v1/k2, see docs/experiments/multiframe_dataset.md): one 5-page 8-bit
LZW TIFF per image, full 1920x1080, pages = [gray(t-2) aligned to t, B(t), G(t), R(t), gray(t+2) aligned
to t]; a neighbour that crosses a cut (consecutive-frame Bhattacharyya distance > 0.2) or fails
alignment is replaced by gray(t). Alignment = LK + RANSAC from calibration/temporal through
``multiframe.align.estimate_pair_lk``, person boxes (machine_labels conf >= 0.3) excluded from the
motion estimate. Differences to the original builder: frames are decoded with a single sequential
``ffmpeg -fps_mode passthrough`` pass per clip (ring buffer of 5 frames; cv2 seeking gave black frames
on some of these videos), and RGB comes from the video exactly as in v1.

Labels: pseudo ball (class 2, in_play_ball) + machine player/other-class boxes at conf >= 0.5 (class ids
as v11). ``meta.jsonl`` carries the provenance of every pseudo ball.

usage: python pseudo_build.py --sel selection.jsonl --out /mnt/storage/footy_data/multiframe_pseudo_v1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from footy_track.multiframe.align import estimate_pair_lk, hist_distance, warp_to_ref  # noqa: E402
from footy_track.multiframe.frames import machine_person_boxes  # noqa: E402

K = 2
CUT_THRESHOLD = 0.2
W, H = 1920, 1080
V11_NAMES = ["ball", "coach", "in_play_ball", "person", "player", "player_sub", "referee"]
FOOTY = Path("/mnt/storage/footy_data")


def frames_of(video: str):
    nbytes = W * H * 3
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=nbytes * 2)
    i = 0
    try:
        while True:
            buf = p.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            i += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def align_neighbour(a, b, t, tn, hd, stem):
    st = {"frame": tn, "status": "ok", "ms": 0.0}
    lo, hi = sorted((t, tn))
    st["hist_max"] = max(hd[g] for g in range(lo, hi))
    if st["hist_max"] > CUT_THRESHOLD:
        st["status"] = "cut"
        return None, st
    boxes = machine_person_boxes(stem, t) or []
    t0 = time.perf_counter()
    pm = estimate_pair_lk(a, b, scale=480, boxes_a=boxes, boxes_b=boxes)
    st["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    st.update(inliers=pm.inliers, residual=round(pm.residual, 3), reason=pm.reason)
    if not pm.ok:
        st["status"] = "align_fail"
        return None, st
    return cv2.cvtColor(warp_to_ref(b, pm.H_px), cv2.COLOR_BGR2GRAY), st


def label_lines(r: dict) -> list[str]:
    cx, cy = r["x"] / W, r["y"] / H
    bw, bh = r["w"] / W, r["h"] / H
    x0, y0, x1, y1 = max(cx - bw / 2, 0.0), max(cy - bh / 2, 0.0), min(cx + bw / 2, 1.0), min(cy + bh / 2, 1.0)
    lines = [f"2 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n"]
    for cls, ox, oy, ow, oh, _conf in r["other_boxes"]:
        lines.append(f"{cls} {ox:.6f} {oy:.6f} {ow:.6f} {oh:.6f}\n")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N images")
    a = ap.parse_args()
    cv2.setNumThreads(1)
    sel = [json.loads(x) for x in Path(a.sel).read_text().splitlines() if x.strip()]
    pool_stem = {}
    pb = json.loads((Path(a.sel).parent / "stems.json").read_text())
    by_clip = defaultdict(list)
    for r in sel:
        by_clip[r["clip"]].append(r)
    out = a.out
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    meta_p = out / "meta.jsonl"
    done = set()
    if meta_p.exists():
        done = {json.loads(x)["id"] for x in meta_p.read_text().splitlines() if x.strip()}
    n_done = len(done)
    t_start = time.time()
    with meta_p.open("a") as mf:
        for ci, (cid, rs) in enumerate(sorted(by_clip.items()), 1):
            todo = {r["t"]: r for r in rs if f"{cid}_{r['t']:06d}" not in done}
            if not todo:
                continue
            stem = pb[cid]["stem"]
            video = pb[cid]["video"]
            ring, hd, prev_hd_frame = {}, {}, None
            need = sorted(todo)
            idx = 0
            for i, fr in frames_of(video):
                ring[i] = fr
                if i - 1 in ring and idx < len(need) and i - 1 >= need[idx] - K:
                    hd[i - 1] = hist_distance(ring[i - 1], fr)
                while idx < len(need) and need[idx] + K <= i:
                    t = need[idx]
                    idx += 1
                    r = todo[t]
                    a_img = ring[t]
                    gray_t = cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY)
                    planes, nb = [], {}
                    for name, tn in (("prev", t - K), ("next", t + K)):
                        g, st = align_neighbour(a_img, ring[tn], t, tn, hd, stem)
                        nb[name] = st
                        planes.append(g if g is not None else gray_t)
                    stack = [planes[0], *cv2.split(a_img), planes[1]]
                    iid = f"{cid}_{t:06d}"
                    if not cv2.imwritemulti(str(out / "images" / "train" / f"{iid}.tiff"), stack, [cv2.IMWRITE_TIFF_COMPRESSION, 5]):
                        raise OSError(iid)
                    lines = label_lines(r)
                    (out / "labels" / "train" / f"{iid}.txt").write_text("".join(lines))
                    meta = {"id": iid, "split": "train", "clip_id": cid, "video": video, "frame": t, "k": K,
                            "mask_src": "machine", "n_ball": 1, "prev": nb["prev"], "next": nb["next"],
                            "pseudo": {k: r.get(k) for k in ("category", "source", "rule", "det_conf", "solver_p", "agree_px", "dist_anchor",
                                                              "anchor_conf", "x", "y", "w", "h", "near_human_gt_5", "near_v11_train_5")},
                            "n_other_boxes": len(r["other_boxes"]), "other_boxes_source": "machine_labels conf>=0.5"}
                    mf.write(json.dumps(meta) + "\n")
                    mf.flush()
                    n_done += 1
                    if n_done % 25 == 0:
                        print(f"[{ci}/{len(by_clip)}] {n_done} images, {time.time() - t_start:.0f}s", flush=True)
                    if a.limit and n_done >= a.limit:
                        return
                thr = min(i - 1, need[idx] - K) if idx < len(need) else i
                for g in [g for g in ring if g < thr]:
                    del ring[g]
                if idx >= len(need):
                    break
    base = {"path": str(out), "names": dict(enumerate(V11_NAMES)), "nc": 7, "channels": 5}
    y = lambda d: "".join(f"{k}: {json.dumps(v)}\n" for k, v in d.items())
    (out / "data.yaml").write_text(y({**base, "train": "images/train", "val": "/mnt/storage/footy_data/multiframe_v1/k2/images/val"}))
    print(f"done {n_done} images in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
