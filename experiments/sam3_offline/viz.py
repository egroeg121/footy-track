"""Contact sheet / GIF of a seed-propagated track (mini CPU): crops around the predicted position, GT box in green, prediction cross in red, text = frame, distance
from the seed, score logit, error to GT (px, 1080p-equiv).

    python viz.py --spec spec.json --tracks DIR --tag TAG --clip C --seed o0 --dir fwd --out sheet.jpg [--step 12] [--gif out.gif] [--size 128]
"""
import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--spec"); ap.add_argument("--tracks"); ap.add_argument("--tag"); ap.add_argument("--clip"); ap.add_argument("--seed"); ap.add_argument("--dir", default="fwd")
ap.add_argument("--protocol", default="oracle"); ap.add_argument("--out"); ap.add_argument("--gif", default=""); ap.add_argument("--step", type=int, default=12)
ap.add_argument("--size", type=int, default=128); ap.add_argument("--cols", type=int, default=8); ap.add_argument("--max", type=int, default=48)
a = ap.parse_args()
spec = json.load(open(a.spec)); clip = spec["clips"][a.clip]
doc = json.load(open(Path(a.tracks) / f"tracks__{a.clip}__{a.tag}__{a.protocol}.json"))
tr = next(t for t in doc["tracks"] if t["seed"] == a.seed and t["dir"] == a.dir)
st, w, h = clip["stride"], clip["w"], clip["h"]
fs = sorted(int(k) for k in tr["frames"])
if a.dir == "bwd":
    fs = fs[::-1]
sel = fs[::a.step][:a.max]
gt = {int(k): v for k, v in clip["gt"].items()}
want = {f * st for f in sel}
proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip["video"], "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=w * h * 6)
imgs, i, last = {}, 0, max(want)
while i <= last:
    buf = proc.stdout.read(w * h * 3)
    if len(buf) < w * h * 3: break
    if i in want: imgs[i // st] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    i += 1
proc.kill()
S, Z, sc = a.size, 3, 1080.0 / h
tiles, centre = [], None
for f in sel:
    r = tr["frames"][str(f)]
    fr = imgs.get(f)
    if fr is None: continue
    if r[0]: centre = (r[1], r[2])
    c = centre or ((clip["seeds_oracle"] + clip["seeds_rtdetr"])[0]["box"][0], (clip["seeds_oracle"] + clip["seeds_rtdetr"])[0]["box"][1]) if centre is None else centre
    x0 = int(min(max(c[0] - S / 2, 0), w - S)); y0 = int(min(max(c[1] - S / 2, 0), h - S))
    t = cv2.resize(fr[y0:y0 + S, x0:x0 + S], (S * Z, S * Z), interpolation=cv2.INTER_CUBIC)
    err = ""
    if f in gt:
        b = gt[f]; cv2.rectangle(t, (int((b[0] - x0) * Z), int((b[1] - y0) * Z)), (int((b[2] - x0) * Z), int((b[3] - y0) * Z)), (0, 255, 0), 1)
        if r[0]: err = " err %.0f" % (np.hypot(r[1] - (b[0] + b[2]) / 2, r[2] - (b[1] + b[3]) / 2) * sc)
        else: err = " ABSENT"
    if r[0]:
        px, py = int((r[1] - x0) * Z), int((r[2] - y0) * Z)
        col = (0, 0, 255) if r[4] > 0 else (0, 165, 255)
        cv2.line(t, (px - 8, py), (px + 8, py), col, 1); cv2.line(t, (px, py - 8), (px, py + 8), col, 1)
    cv2.putText(t, f"{f} d{abs(f - tr['t'])} s{r[4]:.1f}{err}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    tiles.append(t)
rows = [np.hstack(tiles[k:k + a.cols] + [np.zeros_like(tiles[0])] * (a.cols - len(tiles[k:k + a.cols]))) for k in range(0, len(tiles), a.cols)]
sheet = np.vstack(rows)
cv2.imwrite(a.out, cv2.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
if a.gif:
    from PIL import Image
    fr = [Image.fromarray(cv2.cvtColor(cv2.resize(t, (S * 2, S * 2)), cv2.COLOR_BGR2RGB)) for t in tiles]
    fr[0].save(a.gif, save_all=True, append_images=fr[1:], duration=250, loop=0)
