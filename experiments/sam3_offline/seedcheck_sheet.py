"""Candidate seeds for the seed-verification study (mini CPU): random RT-DETR conf >= 0.8 ball detections of the held-out clips (stride-1 labels), a contact sheet to label them by eye
(ball / not ball), and cands.json for seedcheck.py (GPU).

    python seedcheck_sheet.py --spec spec.json --labels DIR --out-dir DIR [--per-clip 8]
"""
import argparse, json, random, subprocess
from pathlib import Path
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--spec"); ap.add_argument("--labels"); ap.add_argument("--out-dir"); ap.add_argument("--per-clip", type=int, default=8); ap.add_argument("--thr", type=float, default=0.8)
a = ap.parse_args()
spec = json.load(open(a.spec)); random.seed(0)
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
cands = []
for name, c in spec["clips"].items():
    p = Path(a.labels) / f"{name}.jsonl"
    if c["set"] != "final" or name.startswith("astonvilla") or not p.exists(): continue
    W, H = c["w"], c["h"]; pool = []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if any(t in ("in_play_ball", "ball") for t in r["tags"]) and r["confidence"] >= a.thr:
            b = r["bbox"]; pool.append((r["frame_index"], [b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W, (b["y"] + b["h"]) * H], r["confidence"]))
    random.shuffle(pool); chosen = []
    for f, box, conf in pool:
        if all(abs(f - g) >= 12 for g, _, _ in chosen): chosen.append((f, box, conf))
        if len(chosen) >= a.per_clip: break
    for f, box, conf in sorted(chosen):
        cands.append({"id": len(cands), "clip": name, "frame": f, "box": [round(v, 1) for v in box], "conf": round(conf, 4), "n_pool": len(pool)})
json.dump(cands, open(out / "cands.json", "w"))
tiles = []
by = {}
for c in cands: by.setdefault(c["clip"], []).append(c)
S, Z = 64, 4
for name, cs in by.items():
    c0 = spec["clips"][name]; w, h = c0["w"], c0["h"]; want = {c["frame"] for c in cs}
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", c0["video"], "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=w*h*6)
    imgs, i = {}, 0
    while i <= max(want):
        buf = proc.stdout.read(w*h*3)
        if len(buf) < w*h*3: break
        if i in want: imgs[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        i += 1
    proc.kill()
    for c in cs:
        fr = imgs[c["frame"]]; b = c["box"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0 = int(min(max(cx - S / 2, 0), w - S)); y0 = int(min(max(cy - S / 2, 0), h - S))
        t = cv2.resize(fr[y0:y0 + S, x0:x0 + S], (S * Z, S * Z), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(t, (int((b[0] - x0) * Z) - 4, int((b[1] - y0) * Z) - 4), (int((b[2] - x0) * Z) + 4, int((b[3] - y0) * Z) + 4), (0, 0, 255), 1)
        cv2.putText(t, f"#{c['id']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
cols = 8
rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
cv2.imwrite(str(out / "cands_sheet.jpg"), np.vstack(rows))
print(len(cands), "candidates", {k: len(v) for k, v in by.items()})
