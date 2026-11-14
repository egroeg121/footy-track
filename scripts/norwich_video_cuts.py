#!/usr/bin/env python
"""Per-frame cheap video features for the Norwich held-out clips (CPU, 1 thread-ish).

For every clip writes ``<out>/<clip>.npz`` with, per frame: mean gray level,
L1 distance between consecutive gray histograms (``hist_d``) and mean absolute
difference of 64x36 gray thumbnails (``mad``).  Used to find shot cuts
independently of the player-box camera estimate, and to detect black decodes.

    python scripts/norwich_video_cuts.py --out <dir>
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

VIDEO_DIR = Path(
    "/mnt/storage/footy_data/arsenal_norwich/split_video_broadcast_frames"
)


def clip_features(path: Path) -> dict[str, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    means, hd, mad, ok_flags = [], [], [], []
    prev_h = prev_s = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        s = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        h = cv2.calcHist([g], [0], None, [32], [0, 256]).ravel()
        h /= h.sum()
        means.append(float(g.mean()))
        hd.append(0.0 if prev_h is None else float(np.abs(h - prev_h).sum()))
        mad.append(0.0 if prev_s is None else float(np.abs(s - prev_s).mean()))
        prev_h, prev_s = h, s
    cap.release()
    return {
        "mean": np.array(means, np.float32),
        "hist_d": np.array(hd, np.float32),
        "mad": np.array(mad, np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cv2.setNumThreads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for p in sorted(VIDEO_DIR.glob("arsenal_norwich_seg*.mp4")):
        dst = args.out / f"{p.stem}.npz"
        if dst.exists():
            continue
        feats = clip_features(p)
        np.savez(dst, **feats)
        print(f"{p.stem} {len(feats['mean'])} frames, min mean {feats['mean'].min():.1f}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
