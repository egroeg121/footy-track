"""Run SAM2.1 from sparse human ball anchors, forward and backward (GPU).

For every anchor (union over the anchor spacings N) a *separate* inference state is built
for the forward window ``[a, a+reach]`` and another for the backward window ``[a-reach, a]``
(propagated with ``propagate_in_video(reverse=True)``), so forward and backward tracks are
independent and no track ever sees memory of another anchor. The anchor prompt is the GT
box (optionally + its centre point). Per-frame outputs are written to JSON; scoring is a
separate CPU step (``score.py``).

Variants
  full      the 1920x1080 frame squashed to 1024x1024 by SAM2 (its default)
  cropNNN   a native-resolution NNN x NNN window (1080p pixels, no downscale) around the ball,
            resized to 1024; the window is re-centred on the last predicted ball position
            before each frame is fed (frames are injected into the state one at a time, so
            this is a custom driver around ``propagate_in_video``)

Peak VRAM / runtime are recorded. Resumable: finished (run, anchor, direction) tracks in an
existing output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt as G  # noqa: E402

MODELS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG = 1024


def video_size(video: str) -> tuple[int, int]:
    import subprocess
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", video], capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in o.split(",")[:2])
    return w, h


def read_frames(video: str, wanted: set[int], w: int, h: int) -> dict[int, np.ndarray]:
    """Decode with the ffmpeg CLI (rgb24 pipe). cv2.VideoCapture returns all-black frames for
    these clips (swscaler 'Cannot convert interlaced to progressive')."""
    import subprocess
    nbytes = w * h * 3
    last = max(wanted)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-frames:v", str(last + 1),
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=nbytes * 2)
    out, i = {}, 0
    while i <= last:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        if i in wanted:
            out[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        i += 1
    proc.stdout.close()
    proc.kill()
    proc.wait()
    missing = wanted - set(out)
    if missing:
        raise RuntimeError(f"{video}: could not decode frames {sorted(missing)[:5]}...")
    if out and float(np.mean([f.mean() for f in list(out.values())[:5]])) < 1.0:
        raise RuntimeError(f"{video}: decoded frames are black")
    return out


def to_input(rgb: np.ndarray):
    """uint8 HxWx3 RGB -> normalised float32 3x1024x1024 tensor (SAM2's own preprocessing)."""
    import torch
    from PIL import Image
    im = Image.fromarray(rgb).resize((IMG, IMG))
    a = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def crop_at(frame: np.ndarray, cx: float, cy: float, wc: int):
    h, w = frame.shape[:2]
    x0 = int(min(max(round(cx - wc / 2), 0), w - wc))
    y0 = int(min(max(round(cy - wc / 2), 0), h - wc))
    return frame[y0:y0 + wc, x0:x0 + wc], x0, y0


def analyse_mask(m):
    """m: bool tensor (GPU). -> None if empty, else dict(cx, cy, area, n_cc) in mask pixel coords."""
    nz = m.nonzero()
    if nz.shape[0] == 0:
        return None
    y0, x0 = (int(v) for v in nz.min(0).values)
    y1, x1 = (int(v) for v in nz.max(0).values)
    sub = m[y0:y1 + 1, x0:x1 + 1].cpu().numpy().astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(sub, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, 4]))
    return {"cx": x0 + float(cents[k][0]) + 0.5, "cy": y0 + float(cents[k][1]) + 0.5,
            "area": int(stats[k, 4]), "n_cc": n - 1}


def torch_autocast(dtype):
    import torch
    return torch.autocast("cuda", dtype=dtype)


class Runner:
    def __init__(self, model: str, ckpt_dir: str, variant: str, dtype: str, prompt: str,
                 hole_area: int | None):
        import torch
        import sam2.sam2_video_predictor as svp
        from sam2.build_sam import build_sam2_video_predictor
        self.torch = torch
        cfg, ck = MODELS[model]
        extra = [] if hole_area is None else [f"++model.fill_hole_area={hole_area}"]
        # hydra_overrides_extra is applied BEFORE build_sam's own fill_hole_area=8, so for a
        # non-default value patch the attribute after construction instead.
        self.pred = build_sam2_video_predictor(cfg, str(Path(ckpt_dir) / ck), device="cuda")
        if hole_area is not None:
            self.pred.fill_hole_area = hole_area
        del extra
        self.variant, self.prompt = variant, prompt
        self.crop = int(variant[4:]) if variant.startswith("crop") else 0
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[dtype]
        self.holder: dict = {}
        svp.load_video_frames = lambda **kw: (self.holder["images"], self.holder["h"], self.holder["w"])
        self.buf = torch.zeros(128, 3, IMG, IMG, dtype=torch.float32)

    # -------------------------------------------------------------- one track
    def track(self, frames_rgb, run: G.Run, a: int, direction: str, reach: int) -> dict | None:
        import contextlib
        ac = (torch_autocast(self.dtype) if self.dtype is not None else contextlib.nullcontext())
        with self.torch.inference_mode(), ac:
            return self._track(frames_rgb, run, a, direction, reach)

    def _track(self, frames_rgb, run: G.Run, a: int, direction: str, reach: int) -> dict | None:
        torch = self.torch
        lo, hi = run.window
        fr = (list(range(a, min(a + reach, hi) + 1)) if direction == "fwd"
              else list(range(max(a - reach, lo), a + 1)))
        if len(fr) < 2:
            return None
        T = len(fr)
        ai = 0 if direction == "fwd" else T - 1
        step = 1 if direction == "fwd" else -1
        imgs = self.buf[:T]
        origin = {}  # frame idx in window -> (x0, y0) crop origin
        t0 = time.time()
        gx0, gy0, gx1, gy1 = run.boxes[a]
        gcx, gcy = run.centre(a)
        if self.crop:
            c, ox, oy = crop_at(frames_rgb[a], gcx, gcy, self.crop)
            imgs[ai] = to_input(c)
            origin[ai] = (ox, oy)
            vh = vw = self.crop
            box = np.array([gx0 - ox, gy0 - oy, gx1 - ox, gy1 - oy], dtype=np.float32)
            pt = np.array([[gcx - ox, gcy - oy]], dtype=np.float32)
        else:
            for i, f in enumerate(fr):
                imgs[i] = to_input(frames_rgb[f])
                origin[i] = (0, 0)
            vh, vw = frames_rgb[a].shape[:2]
            box = np.array([gx0, gy0, gx1, gy1], dtype=np.float32)
            pt = np.array([[gcx, gcy]], dtype=np.float32)
        self.holder.update(images=imgs, h=vh, w=vw)
        st = self.pred.init_state("<injected>", offload_video_to_cpu=True)
        kw = {}
        if self.prompt in ("box", "boxpoint"):
            kw["box"] = box
        if self.prompt in ("point", "boxpoint"):
            kw["points"] = pt
            kw["labels"] = np.array([1], dtype=np.int32)
        self.pred.add_new_points_or_box(st, frame_idx=ai, obj_id=1, **kw)
        out: dict[str, list] = {}
        centre = (gcx, gcy)
        for fidx, _ids, masks in self.pred.propagate_in_video(st, start_frame_idx=ai,
                                                              reverse=(direction == "bwd")):
            f = fr[fidx]
            ox, oy = origin[fidx]
            res = analyse_mask(masks[0, 0] > 0)
            od = st["output_dict_per_obj"][0]
            rec = od["cond_frame_outputs"].get(fidx) or od["non_cond_frame_outputs"].get(fidx)
            score = float(rec["object_score_logits"].reshape(-1)[0]) if rec else float("nan")
            if res is None:
                out[str(f)] = [0, None, None, 0, score]
            else:
                out[str(f)] = [1, round(res["cx"] + ox, 2), round(res["cy"] + oy, 2), res["area"], round(score, 3)]
                if f != a:
                    centre = (res["cx"] + ox, res["cy"] + oy)
            if self.crop:
                nxt = fidx + step
                if 0 <= nxt < T and nxt not in origin:
                    c, ox2, oy2 = crop_at(frames_rgb[fr[nxt]], centre[0], centre[1], self.crop)
                    imgs[nxt] = to_input(c)
                    origin[nxt] = (ox2, oy2)
        if self.torch.cuda.is_available():
            torch.cuda.synchronize()
        self.pred.reset_state(st)
        del st
        return {"anchor": a, "dir": direction, "reach": reach, "frames": out,
                "sec": round(time.time() - t0, 3), "n_frames": T - 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="comma list of clip names (videos/<c>.mp4, gt/<c>.jsonl)")
    ap.add_argument("--data", required=True, help="dir with videos/ and gt/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--models", default="small")
    ap.add_argument("--variants", default="full,crop512")
    ap.add_argument("--ns", default="10,25,50")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--prompt", default="box", choices=["box", "point", "boxpoint"])
    ap.add_argument("--hole-area", type=int, default=None)
    ap.add_argument("--min-run", type=int, default=9)
    ap.add_argument("--max-anchors", type=int, default=0, help="debug: only first K anchors per run")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    Ns = [int(x) for x in args.ns.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = args.clips.split(",")
    sizes = {c: video_size(str(Path(args.data) / "videos" / f"{c}.mp4")) for c in clips}
    runs_by_clip = {c: G.find_runs(c, Path(args.data) / "gt" / f"{c}.jsonl", args.min_run, *sizes[c]) for c in clips}
    for c, rs in runs_by_clip.items():
        print(c, [(r.start, r.end, len(r.aux)) for r in rs], flush=True)
    frames_cache = {}
    for model in args.models.split(","):
        for variant in args.variants.split(","):
            tag = f"{model}__{variant}__{args.dtype}__{args.prompt}{args.tag}"
            runner = None
            for c in clips:
                dst = out_dir / f"tracks__{c}__{tag}.json"
                done = json.loads(dst.read_text()) if dst.exists() else None
                doc = done or {"meta": {"clip": c, "model": model, "variant": variant, "dtype": args.dtype,
                                        "prompt": args.prompt, "ns": Ns, "video_wh": sizes[c], "scale1080": 1080 / sizes[c][1], "host": os.uname().nodename,
                                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                        "torch": torch.__version__},
                               "runs": [], "tracks": []}
                have = {(t["run"], t["anchor"], t["dir"]): t for t in doc["tracks"]}
                if runner is None:
                    runner = Runner(model, args.ckpt, variant, args.dtype, args.prompt, args.hole_area)
                    torch.cuda.reset_peak_memory_stats()
                if c not in frames_cache:
                    wanted = set()
                    for r in runs_by_clip[c]:
                        lo, hi = r.window
                        wanted.update(range(lo, hi + 1))
                    t0 = time.time()
                    frames_cache = {c: read_frames(str(Path(args.data) / "videos" / f"{c}.mp4"), wanted, *sizes[c])}
                    print(f"decoded {len(wanted)} frames of {c} in {time.time()-t0:.1f}s", flush=True)
                frames = frames_cache[c]
                doc["runs"] = [{"start": r.start, "end": r.end, "aux": {str(k): v for k, v in r.aux.items()},
                                "boxes": {str(f): [round(v, 2) for v in b] for f, b in r.boxes.items()},
                                "anchors": {str(n): G.anchors(r, n) for n in Ns}} for r in runs_by_clip[c]]
                for ri, r in enumerate(runs_by_clip[c]):
                    need: dict[int, int] = {}
                    for n in Ns:
                        for a in G.anchors(r, n):
                            need[a] = max(need.get(a, 0), G.reach_for(n))
                    items = sorted(need.items())
                    if args.max_anchors:
                        items = items[:args.max_anchors]
                    for a, reach in items:
                        for d in ("fwd", "bwd"):
                            if (ri, a, d) in have and have[(ri, a, d)]["reach"] >= reach:
                                continue
                            t = runner.track(frames, r, a, d, reach)
                            if t is None:
                                continue
                            t["run"] = ri
                            doc["tracks"] = [x for x in doc["tracks"] if (x["run"], x["anchor"], x["dir"]) != (ri, a, d)]
                            doc["tracks"].append(t)
                            have[(ri, a, d)] = t
                    tf = sum(t["n_frames"] for t in doc["tracks"])
                    ts = sum(t["sec"] for t in doc["tracks"])
                    doc["perf"] = {"peak_vram_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                                   "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
                                   "frames_propagated": tf, "seconds": round(ts, 1),
                                   "sec_per_frame": round(ts / max(tf, 1), 4)}
                    tmp = dst.with_name(dst.name + ".tmp")
                    tmp.write_text(json.dumps(doc))
                    os.replace(tmp, dst)
                    print(f"[{tag}] {c} run {ri} ({r.start}-{r.end}) done; {doc['perf']}", flush=True)
            del runner
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
