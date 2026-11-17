"""Extract the frames the PnLCalib zero-shot experiment needs (mini side).

    PYTHONPATH=src python scripts/pnlcalib/prep_frames.py --out DIR [--sweep-clips a,b] [--sweep-step 5]

Writes ``DIR/frames/<canonical clip>/<frame:06d>.jpg`` (full resolution, JPEG q95) for every calibration GT frame of every clip, plus
every ``--sweep-step``-th frame of the sweep clips, and ``DIR/manifest.json`` (list of {clip, frame, w, h, gt}).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from footy_track.calibration.prelabel.pipeline import find_video
from footy_track.clip_id import canonical_clip_id
from footy_track.scripts.calibrate_clip import CLIPS_DIR, GT_DIR, load_gt_keyframes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sweep-clips", default="arsenal_mancity_seg010,arsenal_mancity_seg020")
    ap.add_argument("--sweep-step", type=int, default=5)
    a = ap.parse_args()
    sweep = {canonical_clip_id(c) for c in a.sweep_clips.split(",") if c}
    manifest = []
    for p in sorted((GT_DIR / "calibration_gt").glob("*.json")):
        cid = p.stem
        video = find_video(CLIPS_DIR, cid)
        if video is None:
            print("no video for", cid)
            continue
        gt = set(load_gt_keyframes(cid))
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = set(gt)
        if cid in sweep:
            want |= set(range(0, n, a.sweep_step))
        d = a.out / "frames" / cid
        d.mkdir(parents=True, exist_ok=True)
        last = max(want)
        for i in range(last + 1):
            ok, fr = cap.read()
            if not ok:
                print("short read", cid, i)
                break
            if i in want:
                h, w = fr.shape[:2]
                cv2.imwrite(str(d / f"{i:06d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                manifest.append({"clip": cid, "frame": i, "w": w, "h": h, "gt": i in gt})
        cap.release()
        print(cid, len(want), "frames", flush=True)
    (a.out / "manifest.json").write_text(json.dumps(manifest))
    print(len(manifest), "images")


if __name__ == "__main__":
    main()
