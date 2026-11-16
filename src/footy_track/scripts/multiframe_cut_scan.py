"""Scan whole clips for camera cuts: adjacent-frame histogram distance, plus what the
ECC aligner does on pairs straddling the biggest jumps (calibrates ``CUT_THRESHOLD``).

    uv run python -m footy_track.scripts.multiframe_cut_scan CLIP.mp4 [CLIP.mp4 ...]
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from footy_track.multiframe.align import estimate_pair, hist_distance


def scan(path: str, top: int = 6) -> dict:
    cap = cv2.VideoCapture(path)
    prev, frames, dist = None, [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.resize(fr, (480, 270), interpolation=cv2.INTER_AREA)
        if prev is not None:
            dist.append(hist_distance(prev, small))
        frames.append(small)
        prev = small
    d = np.array(dist)
    out = {"clip": path, "frames": len(frames), "dist_median": float(np.median(d)),
           "dist_p99": float(np.percentile(d, 99)), "n_over": {str(t): int((d > t).sum()) for t in (0.1, 0.2, 0.35, 0.5)},
           "jumps": []}
    for i in np.argsort(-d)[:top]:
        # pair (i, i+1) straddles the jump; ECC on the 270-px frames (scale 270)
        pm = estimate_pair(frames[i + 1], frames[i], scale=270, method="ecc")
        out["jumps"].append({"between": [int(i), int(i) + 1], "hist": round(float(d[i]), 3),
                             "ecc_ok": pm.ok, "ecc_resid": round(pm.residual, 2), "reason": pm.reason})
    return out


if __name__ == "__main__":
    cv2.setNumThreads(2)
    for p in sys.argv[1:]:
        print(json.dumps(scan(p)), flush=True)
