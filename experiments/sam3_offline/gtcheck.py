"""GT sanity check (CPU, mini): contact sheet of 96 px crops around the human GT ball boxes of one clip (original frame indices).

    python gtcheck.py --spec spec.json --clip NAME --out sheet.jpg [--n 24]
"""
import argparse, json, subprocess
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--spec"); ap.add_argument("--clip"); ap.add_argument("--out"); ap.add_argument("--n", type=int, default=24); ap.add_argument("--frames", default="")
a = ap.parse_args()
spec = json.load(open(a.spec)); c = spec["clips"][a.clip]
st = c["stride"]
items = [(int(k) * st, v) for k, v in c["gt"].items()] + [(int(k) * st + 1, v) for k, v in c.get("gt_odd", {}).items()]
items.sort()
if a.frames:
    keep = set(json.load(open(a.frames))[a.clip])
    items = [(f, b) for f, b in items if f in keep]
if len(items) > a.n:
    items = [items[int(i * (len(items) - 1) / (a.n - 1))] for i in range(a.n)]
w, h = c["w"], c["h"]
want = {f for f, _ in items}
proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", c["video"], "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=w*h*6)
tiles, i = {}, 0
last = max(want)
while i <= last:
    buf = proc.stdout.read(w*h*3)
    if len(buf) < w*h*3: break
    if i in want:
        tiles[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    i += 1
proc.kill()
S, Z = 96, 3
sheet = []
for f, b in items:
    fr = tiles.get(f)
    if fr is None: continue
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    x0 = int(min(max(cx - S/2, 0), w - S)); y0 = int(min(max(cy - S/2, 0), h - S))
    t = cv2.resize(fr[y0:y0+S, x0:x0+S], (S*Z, S*Z), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(t, (int((b[0]-x0)*Z)-3, int((b[1]-y0)*Z)-3), (int((b[2]-x0)*Z)+3, int((b[3]-y0)*Z)+3), (0, 0, 255), 1)
    cv2.putText(t, str(f), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    sheet.append(t)
cols = 6
rows = [np.hstack(sheet[i:i+cols] + [np.zeros_like(sheet[0])] * (cols - len(sheet[i:i+cols]))) for i in range(0, len(sheet), cols)]
cv2.imwrite(a.out, np.vstack(rows))
