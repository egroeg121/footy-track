"""Contact sheet of a sam3_label.py output (mini CPU): one tile per sampled labelled frame, crop around the label, text = frame / gate / score, GT box green when present.

    python viz_labels.py --spec spec.json --labels DIR/<clip>.labels.jsonl --out sheet.jpg [--step 6] [--max 48] [--gate weak]
"""
import argparse, json, subprocess
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--spec"); ap.add_argument("--labels"); ap.add_argument("--out"); ap.add_argument("--step", type=int, default=6)
ap.add_argument("--max", type=int, default=48); ap.add_argument("--cols", type=int, default=8); ap.add_argument("--size", type=int, default=128); ap.add_argument("--gate", default="")
a = ap.parse_args()
spec = json.load(open(a.spec)); rows = [json.loads(l) for l in open(a.labels)]
clip = spec["clips"][rows[0]["clip"]]; st, w, h = clip["stride"], clip["w"], clip["h"]
if a.gate: rows = [r for r in rows if r["gate"] == a.gate]
sel = rows[::a.step][:a.max]
gt = {int(k) * st: v for k, v in clip["gt"].items()} | {int(k) * st + 1: v for k, v in clip.get("gt_odd", {}).items()}
want = {r["frame"] for r in sel} | {r["frame"] + 1 for r in sel}
proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip["video"], "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=w*h*6)
imgs, i = {}, 0
while i <= max(want):
    buf = proc.stdout.read(w*h*3)
    if len(buf) < w*h*3: break
    if i in want: imgs[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    i += 1
proc.kill()
S, Z = a.size, 3; tiles = []
col = {"seed": (255, 0, 255), "s1": (0, 200, 0), "score": (0, 215, 255), "weak": (0, 0, 255)}
for r in sel:
    fr = imgs.get(r["frame"])
    if fr is None: continue
    x0 = int(min(max(r["x"] - S / 2, 0), w - S)); y0 = int(min(max(r["y"] - S / 2, 0), h - S))
    t = cv2.resize(fr[y0:y0 + S, x0:x0 + S], (S * Z, S * Z), interpolation=cv2.INTER_CUBIC)
    px, py = int((r["x"] - x0) * Z), int((r["y"] - y0) * Z)
    cv2.circle(t, (px, py), 22, col[r["gate"]], 1)
    g = gt.get(r["frame"])
    msg = ""
    if g is not None:
        cv2.rectangle(t, (int((g[0] - x0) * Z), int((g[1] - y0) * Z)), (int((g[2] - x0) * Z), int((g[3] - y0) * Z)), (0, 255, 0), 1)
        msg = " err %.0f" % (np.hypot(r["x"] - (g[0] + g[2]) / 2, r["y"] - (g[1] + g[3]) / 2) * 1080 / h)
    cv2.putText(t, f"{r['frame']} {r['gate']} {r['score']:.1f}{msg}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    tiles.append(t)
rows_ = [np.hstack(tiles[k:k + a.cols] + [np.zeros_like(tiles[0])] * (a.cols - len(tiles[k:k + a.cols]))) for k in range(0, len(tiles), a.cols)]
cv2.imwrite(a.out, cv2.resize(np.vstack(rows_), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
