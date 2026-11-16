"""Shared helpers for the SAM 3 offline ball labeller (desktop GPU side + mini CPU side).

Pure numpy / cv2 / subprocess; torch is imported lazily by the GPU modules.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def video_info(video: str) -> dict:
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_frames", "-of", "csv=p=0",
                        video], capture_output=True, text=True, check=True).stdout.strip().split(",")
    num, den = o[2].split("/")
    return {"w": int(o[0]), "h": int(o[1]), "fps": float(num) / float(den), "n": int(o[3]) if o[3].isdigit() else 0}


def decode_window(video: str, w: int, h: int, stride: int, lo: int, hi: int) -> dict[int, np.ndarray]:
    """Decode processed frames lo..hi (inclusive; processed index p = original frame p*stride) with the ffmpeg CLI.

    cv2.VideoCapture returns all-black frames for these clips and ffmpeg needs ``-fps_mode passthrough`` (see KB footy-offline-ball-tracking-2026-11-14).
    """
    nbytes = w * h * 3
    vf = ["-vf", f"select='not(mod(n,{stride}))'"] if stride > 1 else []
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-fps_mode", "passthrough", *vf, "-frames:v", str(hi + 1), "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=nbytes * 2)
    out: dict[int, np.ndarray] = {}
    i = 0
    while i <= hi:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        if i >= lo:
            out[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        i += 1
    proc.stdout.close()
    proc.kill()
    proc.wait()
    if not out:
        raise RuntimeError(f"{video}: no frames decoded in [{lo},{hi}]")
    if float(np.mean([f.mean() for f in list(out.values())[:3]])) < 1.0:
        raise RuntimeError(f"{video}: decoded frames are black")
    return out


def crop_at(frame: np.ndarray, cx: float, cy: float, c: int):
    """c x c native-pixel window centred on (cx, cy), clamped inside the frame. Returns (crop, x0, y0)."""
    h, w = frame.shape[:2]
    c = min(c, h, w)
    x0 = int(min(max(round(cx - c / 2), 0), w - c))
    y0 = int(min(max(round(cy - c / 2), 0), h - c))
    return frame[y0:y0 + c, x0:x0 + c], x0, y0


def load_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def iter_seeds(spec_clip: dict, protocol: str) -> list[dict]:
    key = {"oracle": "seeds_oracle", "rtdetr": "seeds_rtdetr"}[protocol]
    return [dict(s, id=f"{protocol[0]}{i}") for i, s in enumerate(spec_clip[key])]


def pick(seeds: list[dict], n: int) -> list[dict]:
    """n evenly spaced seeds (all when n <= 0 or n >= len)."""
    if n <= 0 or n >= len(seeds):
        return seeds
    return [seeds[round(i * (len(seeds) - 1) / max(n - 1, 1))] for i in range(n)] if n > 1 else [seeds[len(seeds) // 2]]
