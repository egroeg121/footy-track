"""Contact sheet of one track: crops around the GT ball (green = GT centre, red = SAM2 mask centre)."""
from __future__ import annotations

import argparse
import json

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks")
    ap.add_argument("video")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--anchor", type=int, required=True)
    ap.add_argument("--dir", default="fwd")
    ap.add_argument("--frames", default="1,3,5,8,12,16,20,25")
    ap.add_argument("--half", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.tracks))
    t = next(t for t in d["tracks"] if (t["run"], t["anchor"], t["dir"]) == (a.run, a.anchor, a.dir))
    run = d["runs"][a.run]
    sgn = 1 if a.dir == "fwd" else -1
    want = [a.anchor + sgn * int(k) for k in a.frames.split(",")]
    want = [f for f in want if str(f) in t["frames"]]
    from sam2_tracks import read_frames, video_size
    tiles = []
    frames = {f: im[:, :, ::-1].copy() for f, im in read_frames(a.video, set(want + [a.anchor]), *video_size(a.video)).items()}
    for f in [a.anchor] + want:
        img = frames[f]
        b = run["boxes"].get(str(f))
        cx, cy = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) if b else (None, None)
        gx, gy = (cx, cy) if cx is not None else (run["boxes"][str(a.anchor)][0], run["boxes"][str(a.anchor)][1])
        x0, y0 = int(gx) - a.half, int(gy) - a.half
        pad = cv2.copyMakeBorder(img, a.half * 2, a.half * 2, a.half * 2, a.half * 2, cv2.BORDER_CONSTANT)
        crop = pad[y0 + a.half * 2:y0 + a.half * 2 + 2 * a.half, x0 + a.half * 2:x0 + a.half * 2 + 2 * a.half].copy()
        crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        if b:
            cv2.circle(crop, (int((cx - x0) * 4), int((cy - y0) * 4)), 22, (0, 255, 0), 1)
        p = t["frames"][str(f)]
        if p[0]:
            cv2.circle(crop, (int((p[1] - x0) * 4), int((p[2] - y0) * 4)), 10, (0, 0, 255), 1)
        cv2.putText(crop, f"f{f} {'ABS' if not p[0] else ''} s{p[4]:.1f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        tiles.append(crop)
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(a.out, np.vstack(rows))


if __name__ == "__main__":
    main()
