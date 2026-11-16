"""Round 3 step 5 (desktop CPU): 3-channel PNG + YOLO-label dataset and contact sheets from a selection.jsonl (r3_fuse.py).

Frames are decoded with a single sequential `ffmpeg -fps_mode passthrough` pass per clip and the source frame `t_orig` of every
selected row is written losslessly (BGR PNG, pixels identical to what the detectors see). Labels: pseudo ball (class 2) + machine
player/other boxes at conf >= 0.5 (v11 class ids). 3ch only: no neighbour alignment is computed.

    python r3_build.py --selection selection.jsonl --videos <dir with <clip>.mp4 or --map clip=path> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

W, H = 1920, 1080


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


def label_lines(r: dict) -> list[str]:
    cx, cy, bw, bh = r["x"] / W, r["y"] / H, r["w"] / W, r["h"] / H
    x0, y0, x1, y1 = max(cx - bw / 2, 0.0), max(cy - bh / 2, 0.0), min(cx + bw / 2, 1.0), min(cy + bh / 2, 1.0)
    lines = [f"2 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n"]
    for cls, ox, oy, ow, oh, _c in r["other_boxes"]:
        lines.append(f"{cls} {ox:.6f} {oy:.6f} {ow:.6f} {oh:.6f}\n")
    return lines


def contact_sheet(rows, out_png: Path, img_dir: Path, n=48, cols=8, crop=64, zoom=3, seed=0):
    rng = random.Random(seed)
    sel = rng.sample(rows, min(n, len(rows)))
    tiles = []
    for r in sel:
        img = cv2.imread(str(img_dir / f"{r['clip']}_{r['t_orig']:06d}.png"))
        cx, cy = int(r["x"]), int(r["y"])
        x0, y0 = int(np.clip(cx - crop // 2, 0, W - crop)), int(np.clip(cy - crop // 2, 0, H - crop))
        t = cv2.resize(img[y0:y0 + crop, x0:x0 + crop], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        bw, bh = r["w"] * zoom / 2, r["h"] * zoom / 2
        c = ((cx - x0) * zoom, (cy - y0) * zoom)
        cv2.rectangle(t, (int(c[0] - bw), int(c[1] - bh)), (int(c[0] + bw), int(c[1] + bh)), (0, 255, 255), 1)
        cap = (f"{r['source']} c={r['det_conf']:.2f}" if r["source"] == "B1_det" else f"S1 {r.get('rule')} d={r['dist_anchor']}")
        bar = np.full((16, t.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"{cap} {r['clip'][-6:]}:{r['t_orig']}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, t]))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows_img = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_png), np.vstack(rows_img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True, help="dir with <clip>.mp4 (frame t_orig is read from it)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match", default="")
    a = ap.parse_args()
    cv2.setNumThreads(2)
    sel = [json.loads(x) for x in a.selection.read_text().splitlines() if x.strip()]
    by_clip = defaultdict(list)
    for r in sel:
        by_clip[r["clip"]].append(r)
    img_dir, lab_dir = a.out / "images" / "train", a.out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for ci, (cid, rs) in enumerate(sorted(by_clip.items()), 1):
        todo = {r["t_orig"]: r for r in rs if not (img_dir / f"{cid}_{r['t_orig']:06d}.png").exists()}
        if todo:
            last = max(todo)
            for i, fr in frames_of(str(a.videos / f"{cid}.mp4")):
                if i in todo:
                    r = todo[i]
                    tmp = img_dir / f"{cid}_{i:06d}.tmp.png"
                    cv2.imwrite(str(tmp), fr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    tmp.replace(img_dir / f"{cid}_{i:06d}.png")
                    (lab_dir / f"{cid}_{i:06d}.txt").write_text("".join(label_lines(r)))
                if i >= last:
                    break
            missing = [t for t in todo if not (img_dir / f"{cid}_{t:06d}.png").exists()]
            if missing:
                raise RuntimeError(f"{cid}: could not decode frames {missing[:5]}")
        for r in rs:
            (lab_dir / f"{cid}_{r['t_orig']:06d}.txt").write_text("".join(label_lines(r)))
            metas.append({"id": f"{cid}_{r['t_orig']:06d}", **{k: v for k, v in r.items() if k not in ("other_boxes",)}, "n_other_boxes": len(r["other_boxes"])})
        print(f"[{ci}/{len(by_clip)}] {cid}: {len(rs)} images", flush=True)
    (a.out / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas))
    easy = [r for r in sel if r["category"] == "easy"]
    hard = [r for r in sel if r["category"] != "easy"]
    if easy:
        contact_sheet(easy, a.out / "contact_sheet_easy.png", img_dir)
    if hard:
        contact_sheet(hard, a.out / "contact_sheet_hard.png", img_dir)
    (a.out / ".complete").write_text(f"{len(sel)}\n")
    print("done", len(sel))


if __name__ == "__main__":
    main()
