"""SAM 3 text-prompt ball detection on the GT frames of the spec (seed-finder hit rate vs RT-DETR) -- GPU, desktop venv.

For every GT ball / no_ball frame (original frame index of the video; the spec's processed index * stride, odd GT frames included) the frame is run through
SAM 3's image detector with several text prompts and tiling schemes; all detections (top ``--topk`` by score, boxes in full-frame px) are written to JSONL
for CPU scoring (seedfind_score.py).  Image features are computed once per tile and reused for all prompts.

Schemes: full = whole frame squashed to 1008; t540 = 3x2 tiles of 540 px (1.87x up); t384 = 5x3 tiles of 384 px (2.6x up).
Optional ``--exemplar``: text + one positive box on the same tile (a 3x-inflated, jittered GT box = a rough human click / tracker box) via add_geometric_prompt.

    python sam3_seedfind.py --spec spec.json --videos DIR --out DIR --clips a,b --schemes full,t540 --prompts "ball,soccer ball"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import decode_window, load_spec  # noqa: E402

PROMPTS = ["ball", "football", "soccer ball", "white ball", "small white ball", "yellow ball"]


def tiles_for(scheme: str, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) tiles of the scheme."""
    if scheme == "full":
        return [(0, 0, w, h)]
    size, nx, ny = {"t540": (540, 3, 2), "t384": (384, 5, 3), "t640": (640, 3, 2)}[scheme]
    size = min(size, h, w)
    xs = [round(i * (w - size) / (nx - 1)) for i in range(nx)]
    ys = [round(i * (h - size) / (ny - 1)) for i in range(ny)]
    return [(x, y, x + size, y + size) for y in ys for x in xs]


def nms(dets: list[list[float]], thr: float = 0.5) -> list[list[float]]:
    dets = sorted(dets, key=lambda d: -d[4])
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            ix0, iy0, ix1, iy1 = max(d[0], k[0]), max(d[1], k[1]), min(d[2], k[2]), min(d[3], k[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            u = (d[2] - d[0]) * (d[3] - d[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter
            if u > 0 and inter / u > thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


class TextDetector:
    def __init__(self, conf: float = 0.02, patch_turing: bool = True):
        import torch
        from sam3.model_builder import build_sam3_video_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3_track import CKPT, patch_turing as pt
        if patch_turing:
            pt()
        self.torch = torch
        t0 = time.time()
        m = build_sam3_video_model(checkpoint_path=str(CKPT["sam3"]), load_from_HF=False)
        self.model = m
        self.proc = Sam3Processor(m.detector, confidence_threshold=conf)
        self.model.eval()
        self.load_sec = time.time() - t0

    def detect_tile(self, rgb: np.ndarray, prompts: list[str], exemplar_box=None) -> dict[str, list[list[float]]]:
        """rgb tile HxWx3 uint8 -> {prompt: [[x0,y0,x1,y1,score], ...]} in tile px."""
        from PIL import Image
        torch = self.torch
        out = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = self.proc.set_image(Image.fromarray(rgb))
            h, w = rgb.shape[:2]
            for p in prompts:
                self.proc.reset_all_prompts(state)
                st = self.proc.set_text_prompt(prompt=p, state=state)
                if exemplar_box is not None:
                    x0, y0, x1, y1 = exemplar_box
                    st = self.proc.add_geometric_prompt([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h], True, st)
                out[p] = [[*(float(v) for v in b), float(s)] for b, s in zip(st["boxes"].float().cpu().tolist(), st["scores"].float().cpu().tolist())]
        return out

    def detect(self, frame: np.ndarray, scheme: str, prompts: list[str], exemplar_box=None, topk: int = 15) -> dict[str, list[list[float]]]:
        h, w = frame.shape[:2]
        merged = {p: [] for p in prompts}
        for (x0, y0, x1, y1) in tiles_for(scheme, w, h):
            ex = None
            if exemplar_box is not None:
                ex = [exemplar_box[0] - x0, exemplar_box[1] - y0, exemplar_box[2] - x0, exemplar_box[3] - y0]
                if ex[0] < 0 or ex[1] < 0 or ex[2] > x1 - x0 or ex[3] > y1 - y0:
                    continue   # exemplar must lie inside the tile
            res = self.detect_tile(np.ascontiguousarray(frame[y0:y1, x0:x1]), prompts, ex)
            for p, ds in res.items():
                merged[p] += [[d[0] + x0, d[1] + y0, d[2] + x0, d[3] + y0, d[4]] for d in ds]
        return {p: [[round(v, 2) for v in d] for d in nms(ds)[:topk]] for p, ds in merged.items()}


def gt_frames(clip: dict) -> list[tuple[int, str, list | None]]:
    st = clip["stride"]
    rows = [(int(k) * st, "ball", v) for k, v in clip["gt"].items()] + [(int(k) * st + 1, "ball", v) for k, v in clip.get("gt_odd", {}).items()]
    rows += [(f * st, "noball", None) for f in clip["no_ball"]]
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--schemes", default="full,t540")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--exemplar", action="store_true")
    ap.add_argument("--only-frames", default="", help="json {clip: [orig frames]} to restrict (e.g. the arsenal_example held-out frames)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-per-clip", type=int, default=40, help="dense clips (astonvilla_seg080: 568 GT frames): evenly spaced subset")
    a = ap.parse_args()
    import torch
    spec = load_spec(a.spec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = json.loads(a.only_frames) if a.only_frames else {}
    schemes, prompts = a.schemes.split(","), [p for p in a.prompts.split(",") if p]
    det = None
    for name in a.clips.split(","):
        clip = spec["clips"][name]
        frames = gt_frames(clip)
        if name in only:
            keep = set(only[name])
            frames = [r for r in frames if r[0] in keep]
        if len(frames) > a.max_per_clip > 0:
            frames = [frames[round(i * (len(frames) - 1) / (a.max_per_clip - 1))] for i in range(a.max_per_clip)]
        if a.max_frames:
            frames = frames[:a.max_frames]
        if not frames:
            continue
        dst = out / f"seedfind__{name}{'__ex' if a.exemplar else ''}.jsonl"
        done = set()
        if dst.exists():
            for line in dst.read_text().splitlines():
                r = json.loads(line)
                done.add((r["frame"], r["scheme"]))
        if det is None:
            det = TextDetector()
            torch.cuda.reset_peak_memory_stats()
        lo, hi = frames[0][0], frames[-1][0]
        vid = decode_window(str(Path(a.videos) / f"{name}.mp4"), clip["w"], clip["h"], 1, lo, hi)
        with open(dst, "a") as fh:
            for f, kind, box in frames:
                for sc in schemes:
                    if (f, sc) in done:
                        continue
                    t0 = time.time()
                    res = det.detect(vid[f], sc, prompts, exemplar_box=(None if not a.exemplar or box is None else
                                                                          [box[0] - 1.0 * (box[2] - box[0]), box[1] - 1.0 * (box[3] - box[1]), box[2] + 1.0 * (box[2] - box[0]), box[3] + 1.0 * (box[3] - box[1])]))
                    fh.write(json.dumps({"clip": name, "frame": f, "kind": kind, "scheme": sc, "dets": res, "sec": round(time.time() - t0, 2), "gt": box}) + "\n")
                    fh.flush()
                print(f"{name} f{f} {kind} done", flush=True)
        del vid
    print("peak vram MiB", torch.cuda.max_memory_allocated() / 2**20, flush=True)


if __name__ == "__main__":
    main()
