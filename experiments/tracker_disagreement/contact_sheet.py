"""Contact sheet of random cards of the batch with ALL three tracker positions drawn (mini; numpy + opencv).

    uv run --no-project --python /usr/bin/python3 --with numpy --with opencv-python-headless python contact_sheet.py --list LIST.json --out sheet.png --n 24 --stratum disagree

Green = A++/B1, cyan = SAM2.1, red = SAM3 (circles of 14 px diameter around the position; a filled dot where two trackers coincide); the card's bbox tracker is marked with a
square.  Tile = crop around the positions (side = max(180, span * 1.5 + 60) px, resized to 360), header = clip / frame / class / stratum.
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

COL = {"t1": (60, 200, 60), "t2": (230, 200, 40), "t3": (50, 50, 240)}     # BGR
LAB = {"t1": "A++/B1", "t2": "SAM2.1", "t3": "SAM3"}
VID = "/mnt/storage/footy_data/arsenal_norwich/split_video_broadcast_frames/{}.mp4"


def frame(clip: str, f: int):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", VID.format(clip), "-vf", f"select=eq(n\\,{f})", "-fps_mode", "passthrough", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def tile(it: dict, out: int = 360):
    img = frame(it["clip"], it["frame_index"])
    pos = {k: v for k, v in it["positions"].items() if v}
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    side = int(max(180, span * 1.5 + 60))
    x0 = int(min(max(cx - side / 2, 0), img.shape[1] - side)) if side < img.shape[1] else 0
    y0 = int(min(max(cy - side / 2, 0), img.shape[0] - side)) if side < img.shape[0] else 0
    crop = img[y0:y0 + side, x0:x0 + side].copy()
    s = out / crop.shape[1]
    crop = cv2.resize(crop, (out, int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    for k, (px, py) in pos.items():
        c = (int((px - x0) * s), int((py - y0) * s))
        cv2.circle(crop, c, 11, COL[k], 2)
        if it.get("src_bbox") == k:
            cv2.rectangle(crop, (c[0] - 14, c[1] - 14), (c[0] + 14, c[1] + 14), (255, 255, 255), 1)
    head = np.full((34, out, 3), 30, np.uint8)
    cv2.putText(head, f"{it['clip'][-6:]} f{it['frame_index']} {it['class']}" + (f":{it['class_detail']}" if it.get("class_detail") else ""), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(head, f"{it['stratum']} mag {it['magnitude_px']:.0f}px seg {it.get('segment_frames', '-')}", (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    t = np.vstack([head, crop])
    return t if t.shape[0] == out + 34 else cv2.copyMakeBorder(t, 0, out + 34 - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--stratum", default="disagree")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    items = [it for it in json.loads(a.list.read_text())["items"] if it["stratum"] == a.stratum and sum(1 for p in it["positions"].values() if p) >= 2]
    random.Random(a.seed).shuffle(items)
    tiles = [tile(it) for it in items[:a.n]]
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
    sheet = np.vstack(rows)
    leg = np.full((26, sheet.shape[1], 3), 20, np.uint8)
    x = 6
    for k in ("t1", "t2", "t3"):
        cv2.circle(leg, (x + 8, 13), 8, COL[k], 2)
        cv2.putText(leg, LAB[k], (x + 22, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x += 120
    cv2.putText(leg, "white square = tracker shown as the card's yellow box", (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.imwrite(str(a.out), np.vstack([leg, sheet]), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print("wrote", a.out, sheet.shape)


if __name__ == "__main__":
    main()
