"""Run the project RT-DETR detector (rtdetr-l_1920_v11, imgsz 1280, conf 0.1) over one clip and
write ``machine_labels``-format JSONL (same rows as the committed-by-hand producer of
``footy_data/machine_labels``; mirrors ``batch_detect_clips.py`` of the Norwich held-out work).

usage: python detect_clip.py VIDEO --weights best.pt --out DIR [--name arsenal_example]
Writes DIR/<name>.jsonl and DIR/_manifest.json (frames, sha256_16 of the weights).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    from ultralytics import YOLO

    name = args.name or args.video.stem
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    names = model.names
    lines, frames, t0 = [], 0, time.time()
    for idx, res in enumerate(model.predict(source=str(args.video), stream=True, imgsz=args.imgsz, conf=args.conf,
                                            device=0, half=True, batch=args.batch, verbose=False)):
        frames = idx + 1
        b = res.boxes
        if b is None or len(b) == 0:
            continue
        for c, s, (cx, cy, w, h) in zip(b.cls.int().tolist(), b.conf.tolist(), b.xywhn.tolist()):
            lines.append(json.dumps({"frame_index": idx, "bbox": {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h},
                                     "center": {"x": cx, "y": cy}, "tags": [names[c], "rtdetr"],
                                     "confidence": round(float(s), 4), "model_id": "rtdetr-l_1920_v11",
                                     "run_id": "regen-" + time.strftime("%Y%m%d"), "reviewed": False}))
    sec = time.time() - t0
    tmp = args.out / f"{name}.jsonl.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, args.out / f"{name}.jsonl")
    (args.out / "_manifest.json").write_text(json.dumps({
        "model_sha256_16": sha256_16(args.weights), "imgsz": args.imgsz, "conf_threshold": args.conf, "half": True,
        "clips": {name: {"frames": frames, "detections": len(lines), "seconds": round(sec, 1)}}}))
    print(f"{name}: {frames} frames, {len(lines)} detections, {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
