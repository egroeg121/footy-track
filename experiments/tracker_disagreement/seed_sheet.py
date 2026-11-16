"""Contact sheet of seeds (crop around the seed box) to eyeball seeds on non-balls (mini; numpy + opencv).

    python seed_sheet.py --seeds seeds.jsonl --out sheet.png --source text --n 48
"""
import argparse, json, random, subprocess
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--source", default="text")
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--cols", type=int, default=8)
ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()
seeds = [json.loads(l) for l in a.seeds.read_text().splitlines() if l.strip()]
seeds = [s for s in seeds if s["source"] == a.source]
random.Random(a.seed).shuffle(seeds)
tiles = []
for s in seeds[:a.n]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"/mnt/storage/footy_data/arsenal_norwich/split_video_broadcast_frames/{s['clip']}.mp4", "-vf", f"select=eq(n\\,{s['frame']})",
                          "-fps_mode", "passthrough", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    cx, cy = (s["box"][0] + s["box"][2]) / 2, (s["box"][1] + s["box"][3]) / 2
    side = 96
    x0, y0 = int(min(max(cx - side / 2, 0), 1920 - side)), int(min(max(cy - side / 2, 0), 1080 - side))
    crop = cv2.resize(img[y0:y0 + side, x0:x0 + side], (288, 288), interpolation=cv2.INTER_CUBIC)
    c = (int((cx - x0) * 3), int((cy - y0) * 3))
    r = int(max(s["box"][2] - s["box"][0], s["box"][3] - s["box"][1]) * 1.5 + 6)
    cv2.circle(crop, c, r, (0, 255, 255), 1)
    cv2.putText(crop, f"{s['clip'][-5:]} f{s['frame']} {s.get('conf', '')}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    tiles.append(crop)
while len(tiles) % a.cols:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(a.out), np.vstack([np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]))
print("wrote", a.out, len(seeds), "seeds of source", a.source)
