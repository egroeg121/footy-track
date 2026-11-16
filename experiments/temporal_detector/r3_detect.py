"""Round 3: RT-DETR-L v11 @1280 machine labels for unlabelled broadcast clips (desktop GPU).

Writes `<out>/<clip>.jsonl` in the `machine_labels` format the harness / pseudo-label code reads (one row per detection, all
classes, conf >= 0.1, normalised top-left bbox + centre, tags [class, "rtdetr"]) and `<out>/<clip>.meta.json`.
Decoding is the ffmpeg CLI with `-fps_mode passthrough` (never cv2). `--stride 2` processes every 2nd decoded frame and writes
frame_index = original_index // 2: the 50 fps Bournemouth clips are thereby handled as 25 fps clips, so B1's per-frame parameters
(vmax px/frame, max_gap, windows, stride) keep their meaning; the original frame of subsampled index t is 2 t.

    python r3_detect.py --clips arsenal_bournemouth_1st_seg012:/path/Bournemouth..._seg012.mp4 --stride 2 --out labels/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

W, H = 1920, 1080


def ffprobe_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip().split(",")[0])


def read_frames(path: Path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 26)
    n = W * H * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="name:video")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.weights)
    names = model.names
    a.out.mkdir(parents=True, exist_ok=True)
    for spec in a.clips:
        name, video = spec.split(":", 1)
        dst = a.out / f"{name}.jsonl"
        if dst.exists():
            print("skip", name)
            continue
        n_all = ffprobe_frames(Path(video))
        t0 = time.time()
        rows, n_proc, n_dec = [], 0, 0
        for t, fr in enumerate(read_frames(Path(video))):
            n_dec += 1
            if t % a.stride:
                continue
            r = model.predict(fr, imgsz=1280, conf=0.1, max_det=300, verbose=False, device=0, half=True)[0]
            b = r.boxes
            if b is not None and len(b):
                xywhn, cf, cl = b.xywhn.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
                for (cx, cy, w, h), c, k in zip(xywhn, cf, cl, strict=True):
                    rows.append(json.dumps({"frame_index": t // a.stride,
                                            "bbox": {"x": float(cx - w / 2), "y": float(cy - h / 2), "w": float(w), "h": float(h)},
                                            "center": {"x": float(cx), "y": float(cy)}, "tags": [names[int(k)], "rtdetr"],
                                            "confidence": round(float(c), 4), "model_id": "rtdetr-l_1920_v11"}))
            n_proc += 1
            if n_proc % 200 == 0:
                print(f"{name} {n_proc} {time.time() - t0:.0f}s", flush=True)
        if n_dec != n_all:
            raise RuntimeError(f"{name}: decoded {n_dec} frames, ffprobe says {n_all}")
        tmp = a.out / f"{name}.tmp"
        tmp.write_text("\n".join(rows) + ("\n" if rows else ""))
        (a.out / f"{name}.meta.json").write_text(json.dumps({"frames": n_proc, "orig_frames": n_all, "stride": a.stride, "detections": len(rows),
                                                             "seconds": round(time.time() - t0, 1)}))
        tmp.replace(dst)
        print(f"{name}: {n_proc} frames processed ({n_all} decoded), {len(rows)} detections, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
