#!/usr/bin/env python
"""Sanity-check the player-box camera-shift estimate against phase correlation of
the video itself on a few Norwich clips (CPU, 250 frames each).

    PYTHONPATH=src python scripts/norwich_camera_check.py <diagnostics cache.pkl> [clip suffixes...]
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from footy_track.ball_offline.data import _load_machine, estimate_camera_shift  # noqa: E402

VIDEO_DIR = Path("/mnt/storage/footy_data/arsenal_norwich/split_video_broadcast_frames")
LABEL_DIR = Path("/mnt/storage/footy_data/machine_labels_norwich")


def main() -> None:
    cv2.setNumThreads(2)
    store = pickle.loads(Path(sys.argv[1]).read_bytes())
    sfx = sys.argv[2:] or ["000", "010", "024", "026", "044", "060"]
    n = 250
    for s in sfx:
        stem = f"arsenal_norwich_seg{s}"
        nf = len(store[stem]["present"])
        m = min(n, nf)
        _, _, _, _, players = _load_machine(LABEL_DIR / f"{stem}.jsonl", nf)
        shift, _ = estimate_camera_shift(players[:m])
        cap = cv2.VideoCapture(str(VIDEO_DIR / f"{stem}.mp4"))
        prev = None
        pc = []
        for _ in range(m):
            ok, f = cap.read()
            g = cv2.cvtColor(cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is None:
                pc.append((0.0, 0.0))
            else:
                # phaseCorrelate returns the shift of cur relative to prev (static content)
                (dx, dy), _ = cv2.phaseCorrelate(prev, g)
                pc.append((dx * 4, dy * 4))
            prev = g
        cap.release()
        pc = np.array(pc)
        a, b = shift[1:, 0], pc[1:, 0]
        moving = np.abs(b) > 3
        print(
            f"{stem[-6:]} corr_x={np.corrcoef(a, b)[0, 1]:.3f} corr_y={np.corrcoef(shift[1:, 1], pc[1:, 1])[0, 1]:.3f} "
            f"median|dx| players={np.median(np.abs(a)):.1f} video={np.median(np.abs(b)):.1f}px "
            f"MAE_x={np.mean(np.abs(a - b)):.2f}px  (frames with |video dx|>3px: {moving.sum()}, "
            f"MAE there {np.mean(np.abs(a - b)[moving]) if moving.any() else float('nan'):.2f})"
        )


if __name__ == "__main__":
    main()
