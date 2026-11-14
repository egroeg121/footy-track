"""Montage of WASB failures for eyeballing (CPU, mini): 96x96 crops of the source frame at 3x.

For a mode, picks random ball frames that were MISSED (top-1 candidate >10 px from GT) and
random false positives (top-1 candidate with score>=thr that is wrong or on a no_ball frame).
Each cell: crop centred on the GT (green ring) or on the FP (red ring), the other marker drawn if
inside the crop. Run: uv run --with numpy --with opencv-python-headless python viz_failures.py ...
"""
import argparse
import glob
import json
import math
import os
import random
import subprocess

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="", help="substring of clip ids to skip")
    ap.add_argument("--only", default="", help="substring: keep only matching clip ids")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ev = json.load(open(a.evalset))["clips"]
    miss, fp, hit = [], [], []
    for f in sorted(glob.glob(os.path.join(a.pred_dir, "*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if a.skip and a.skip in r["clip"]:
                continue
            if a.only and a.only not in r["clip"]:
                continue
            c = [(x, y, s) for x, y, s, _ in r["pred"][a.mode] if s >= a.thr]
            top = max(c, key=lambda t: t[2]) if c else None
            gts = r["gt"]
            ok = bool(top and gts and min(math.hypot(top[0] - g[0], top[1] - g[1]) for g in gts) <= 10)
            if r["kind"] == "ball" and not ok:
                miss.append((r, top))
            if top and not ok:
                fp.append((r, top))
            if ok:
                hit.append((r, top))
    random.Random(a.seed).shuffle(miss)
    random.Random(a.seed).shuffle(fp)
    random.Random(a.seed).shuffle(hit)

    def cell(r, top, centre, label):
        cap = cv2.VideoCapture(ev[r["clip"]]["video"])
        for _ in range(r["frame"]):   # sequential decode: cap.set() seeks land on black frames in some clips
            cap.grab()
        ok, bgr = cap.read()
        cap.release()
        if ok and bgr.mean() < 1.0:   # pip OpenCV 5 decodes some clips (colour space "unknown") as all black
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", ev[r["clip"]]["video"], "-vf", f"select=eq(n\\,{r['frame']})",
                                  "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
            bgr = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3) if len(raw) == 1080 * 1920 * 3 else bgr
        if not ok:
            return np.zeros((288, 288, 3), np.uint8)
        if bgr.shape[:2] != (1080, 1920):   # 720p clips: same 1080p-equivalent space as run_wasb.py
            bgr = cv2.resize(bgr, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        cx, cy = int(centre[0]), int(centre[1])
        pad = cv2.copyMakeBorder(bgr, 48, 48, 48, 48, cv2.BORDER_CONSTANT)
        crop = pad[max(cy, 0):max(cy, 0) + 96, max(cx, 0):max(cx, 0) + 96]
        if crop.shape[:2] != (96, 96):
            return np.zeros((288, 288, 3), np.uint8)
        crop = cv2.resize(crop, (288, 288), interpolation=cv2.INTER_NEAREST)

        def mark(p, col):
            x, y = (p[0] - cx + 48) * 3, (p[1] - cy + 48) * 3
            if 0 <= x < 288 and 0 <= y < 288:
                cv2.circle(crop, (int(x), int(y)), 22, col, 1)
        for g in r["gt"]:
            mark(g, (0, 255, 0))
        if top:
            mark(top, (0, 0, 255))
        cv2.putText(crop, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return crop

    cells = []
    for kind, lst in (("MISS", miss), ("FP", fp)):
        for r, top in lst[:a.n]:
            centre = r["gt"][0] if (kind == "MISS" and r["gt"]) else (top if top else r["gt"][0])
            lab = f"{kind} {r['clip'][-14:]} f{r['frame']}" + (f" s{top[2]:.2f}" if top else " none")
            cells.append(cell(r, top, centre, lab))
    cols = 6
    while len(cells) % cols:
        cells.append(np.zeros((288, 288, 3), np.uint8))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    cv2.imwrite(a.out, np.vstack(rows))
    print(f"miss={len(miss)} fp={len(fp)} hit={len(hit)} -> {a.out}")


if __name__ == "__main__":
    main()
