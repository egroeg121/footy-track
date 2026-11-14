"""Zero-shot WASB-SBDT soccer model over our GT frames (GPU; run on the desktop).

For every GT frame t (ball frames + explicit no_ball frames) a 3-frame window
(t-1, t, t+1; edges clamped) is fed to the pretrained WASB HRNet (frames_in=3,
frames_out=3, input 288x512, ImageNet normalisation, warpAffine bilinear resize as in
the WASB data loader). The heatmap of the *centre* frame is turned into candidates
(local maxima, >= 0.1) and mapped back to 1080p coordinates.

Modes (see MODES): full frame resized to 512x288 (WASB default), plus overlapping
tiles at native / 1.5x / 2x downscale, NMS-merged across tiles.

Inputs  (--inputs): evalset.json, wasb_src/ (a checkout of nttcom/WASB-SBDT, only
                    src/models/hrnet.py is imported), weights/wasb_soccer_best.pth.tar,
                    videos/<video_rel>
Outputs (--out):    pred/<clip>.jsonl  (one line per eval frame, candidates per mode),
                    runtime.json (ms per window / frame per mode, VRAM per mode)
Idempotent: clips with an existing pred file are skipped; files are renamed into place.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

INP_W, INP_H = 512, 288
SRC_W, SRC_H = 1920, 1080
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PEAK_MIN = 0.1      # lowest heatmap value kept as a candidate
NMS_RADIUS = 8.0    # source px, cross-tile NMS
MAX_CAND = 30
TILE_BATCH = 4      # tiles per forward pass (8 GB card)


def grid_starts(total, size, n):
    if n == 1:
        return [0]
    return [round(i * (total - size) / (n - 1)) for i in range(n)]


def make_regions(k, ncols, nrows):
    """Source regions (x0, y0, w, h) of size (512k x 288k), k = downscale factor."""
    rw, rh = round(INP_W * k), round(INP_H * k)
    xs, ys = grid_starts(SRC_W, rw, ncols), grid_starts(SRC_H, rh, nrows)
    return [(x, y, rw, rh) for y in ys for x in xs]


# name -> (regions, resize method)
MODES = {
    "full_bilinear": ([(0, 0, SRC_W, SRC_H)], "warp"),     # WASB default (3.75x down)
    "full_area": ([(0, 0, SRC_W, SRC_H)], "area"),         # ablation: antialiased resize
    "tile_2x": (make_regions(2.0, 3, 2), "warp"),          # 1024x576 regions -> 512x288
    "tile_1p5x": (make_regions(1.5, 4, 3), "warp"),        # 768x432 regions
    "tile_native": (make_regions(1.0, 6, 4), "warp"),      # 512x288 regions, 1:1
}


class AttrDict(dict):
    """dict with attribute access (hrnet.py reads cfg.MODEL.EXTRA like an OmegaConf node)."""

    def __getattr__(self, k):
        v = self[k]
        return AttrDict(v) if isinstance(v, dict) else v


def load_model(src_dir, ckpt, device):
    spec = importlib.util.spec_from_file_location("wasb_hrnet", os.path.join(src_dir, "models", "hrnet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = lambda n, b, nb, nc: dict(NUM_MODULES=1, NUM_BRANCHES=nb, BLOCK=b, NUM_BLOCKS=n, NUM_CHANNELS=nc, FUSE_METHOD="SUM")  # noqa: E731
    cfg = dict(name="hrnet", frames_in=3, frames_out=3, inp_height=288, inp_width=512, out_height=288, out_width=512,
               rgb_diff=False, out_scales=[0],
               MODEL=dict(EXTRA=dict(
                   FINAL_CONV_KERNEL=1, PRETRAINED_LAYERS=["*"], STEM=dict(INPLANES=64, STRIDES=[1, 1]),
                   STAGE1=s([1], "BOTTLENECK", 1, [32]), STAGE2=s([2, 2], "BASIC", 2, [16, 32]),
                   STAGE3=s([2, 2, 2], "BASIC", 3, [16, 32, 64]), STAGE4=s([2, 2, 2, 2], "BASIC", 4, [16, 32, 64, 128]),
                   DECONV=dict(NUM_DECONVS=0, KERNEL_SIZE=[], NUM_BASIC_BLOCKS=2)), INIT_WEIGHTS=True))
    model = mod.HRNet(AttrDict(cfg))
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval(), sorted(ck.keys()) if isinstance(ck, dict) else None


def prep_tile(frames_rgb, region, method):
    """3 source frames -> (9, 288, 512) float32 normalised, plus (x0, y0, k)."""
    x0, y0, rw, rh = region
    k = rw / INP_W
    chans = []
    for im in frames_rgb:
        if method == "area":
            t = cv2.resize(im[y0:y0 + rh, x0:x0 + rw], (INP_W, INP_H), interpolation=cv2.INTER_AREA)
        else:
            # dst = (src + .5)/k - .5 (pixel-centre aligned), same bilinear warp as WASB's loader
            M = np.array([[1 / k, 0, (0.5 - x0) / k - 0.5], [0, 1 / k, (0.5 - y0) / k - 0.5]], np.float32)
            t = cv2.warpAffine(im, M, (INP_W, INP_H), flags=cv2.INTER_LINEAR)
        t = (t.astype(np.float32) / 255.0 - MEAN) / STD
        chans.append(t.transpose(2, 0, 1))
    return np.concatenate(chans, 0), (x0, y0, k)


@torch.no_grad()
def peaks(hm, topk=10):
    """hm (B,H,W) sigmoid heatmaps -> per tile list of (x, y, peak, sum7x7) in tile input px."""
    B, H, W = hm.shape
    mx = F.max_pool2d(hm[:, None], 9, 1, 4)[:, 0]
    cand = hm * ((hm == mx) & (hm >= PEAK_MIN))
    vals, idx = cand.view(B, -1).topk(topk, dim=1)
    ys, xs = idx // W, idx % W
    pad = F.pad(hm, (2, 2, 2, 2))
    d = torch.arange(-2, 3, device=hm.device)
    bi = torch.arange(B, device=hm.device)[:, None, None, None]
    patch = pad[bi, (ys + 2)[:, :, None, None] + d[None, None, :, None], (xs + 2)[:, :, None, None] + d[None, None, None, :]]
    wsum = patch.sum((2, 3)).clamp_min(1e-6)
    cx = xs.float() + (patch * d[None, None, None, :]).sum((2, 3)) / wsum
    cy = ys.float() + (patch * d[None, None, :, None]).sum((2, 3)) / wsum
    ssum = F.avg_pool2d(hm[:, None], 7, 1, 3, count_include_pad=True)[:, 0] * 49
    s7 = ssum.view(B, -1).gather(1, idx)
    out = []
    v, cx, cy, s7 = vals.cpu().numpy(), cx.cpu().numpy(), cy.cpu().numpy(), s7.cpu().numpy()
    for b in range(B):
        out.append([(cx[b, i], cy[b, i], v[b, i], s7[b, i]) for i in range(topk) if v[b, i] > 0])
    return out


def nms(cands, radius=NMS_RADIUS):
    cands = sorted(cands, key=lambda c: -c[2])
    kept = []
    for c in cands:
        if all((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 > radius ** 2 for k in kept):
            kept.append(c)
            if len(kept) >= MAX_CAND:
                break
    return kept


def run_window(model, frames_rgb, mode, device, tim):
    regions, method = MODES[mode]
    t0 = time.perf_counter()
    arrs, meta = zip(*[prep_tile(frames_rgb, r, method) for r in regions])
    t1 = time.perf_counter()
    pk = []
    for i in range(0, len(arrs), TILE_BATCH):           # full-res HRNet: ~1 GB/tile, so chunk the tiles
        x = torch.from_numpy(np.stack(arrs[i:i + TILE_BATCH])).to(device)
        out = model(x)[0]                               # (B, 3, 288, 512) logits, one per input frame
        hm = out[:, 1].sigmoid().float()                # centre-frame heatmap
        pk += peaks(hm)
        del x, out, hm
    if device == 'cuda':
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    allc = []
    for (x0, y0, k), cl in zip(meta, pk):
        for cx, cy, v, s7 in cl:
            allc.append((float((cx + 0.5) * k - 0.5 + x0), float((cy + 0.5) * k - 0.5 + y0), float(v), float(s7)))
    kept = nms(allc)
    t3 = time.perf_counter()
    tim[mode]["prep"] += t1 - t0
    tim[mode]["gpu"] += t2 - t1
    tim[mode]["post"] += t3 - t2
    tim[mode]["n"] += 1
    return [[round(a, 2), round(b, 2), round(c, 4), round(d, 3)] for a, b, c, d in kept]


def calibrate_vram(model, device, modes):
    res = {}
    dummy = [np.random.randint(0, 255, (SRC_H, SRC_W, 3), np.uint8) for _ in range(3)]
    for m in modes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        run_window(model, dummy, m, device, {m: dict(prep=0, gpu=0, post=0, n=0)})
        res[m] = dict(tiles=len(MODES[m][0]), peak_alloc_MiB=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                      peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
                      model_resident_MiB=round(base / 2 ** 20, 1))
    return res


def process_clip(model, device, cid, c, inputs, out_dir, modes, tim, max_frames):
    dst = os.path.join(out_dir, "pred", f"{cid}.jsonl")
    if os.path.exists(dst):
        print(f"skip {cid} (done)")
        return 0
    targets = {int(k): "ball" for k in c["ball"]}
    targets.update({int(k): "no_ball" for k in c["no_ball"]})
    tl = sorted(targets)
    if max_frames:
        tl = tl[:max_frames]
    tset, last = set(tl), max(tl)
    cap = cv2.VideoCapture(os.path.join(inputs, "videos", c["video_rel"]))
    buf, rows, i = {}, [], -1

    def emit(t):
        w = [buf[max(j, 0) if max(j, 0) in buf else t] for j in (t - 1, t, t + 1)]
        row = {"clip": cid, "frame": t, "kind": targets[t], "gt": c["ball"].get(str(t), []), "pred": {}}
        for m in modes:
            row["pred"][m] = run_window(model, w, m, device, tim)
        rows.append(row)

    while i <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        if bgr.shape[:2] != (SRC_H, SRC_W):   # the 3 Aston Villa clips are 1280x720: work in 1080p-equivalent space
            bgr = cv2.resize(bgr, (SRC_W, SRC_H), interpolation=cv2.INTER_CUBIC)
        buf[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for old in [k for k in buf if k < i - 3]:
            del buf[old]
        if (i - 1) in tset:
            emit(i - 1)
            tset.discard(i - 1)
    cap.release()
    for t in sorted(tset):          # clip end: t+1 missing -> clamp to t
        if t in buf:
            buf[t + 1] = buf[t]
            emit(t)
    tmp = dst + ".tmp"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(tmp, "w") as f:
        for r in sorted(rows, key=lambda r: r["frame"]):
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, dst)
    print(f"{cid}: {len(rows)}/{len(tl)} frames", flush=True)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--max-frames", type=int, default=0, help="per clip, smoke tests")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clips", default="", help="comma list of canonical ids (default all)")
    a = ap.parse_args()
    device = a.device
    modes = a.modes.split(",")
    ev = json.load(open(os.path.join(a.inputs, "evalset.json")))["clips"]
    if a.clips:
        ev = {k: v for k, v in ev.items() if k in a.clips.split(",")}
    model, keys = load_model(os.path.join(a.inputs, "wasb_src"),
                             os.path.join(a.inputs, "weights", "wasb_soccer_best.pth.tar"), device)
    nparam = sum(p.numel() for p in model.parameters())
    print("model params", nparam, "ckpt keys", keys, "torch", torch.__version__, (torch.cuda.get_device_name(0) if device == "cuda" else "cpu"), flush=True)
    os.makedirs(a.out, exist_ok=True)
    vram = calibrate_vram(model, device, modes) if device == "cuda" else {}
    print("VRAM", json.dumps(vram), flush=True)
    tim = {m: dict(prep=0.0, gpu=0.0, post=0.0, n=0) for m in modes}
    t0 = time.time()
    nf = 0
    for cid, c in ev.items():
        nf += process_clip(model, device, cid, c, a.inputs, a.out, modes, tim, a.max_frames)
    rt = {"params": nparam, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch": torch.__version__, "frames": nf,
          "wall_s": round(time.time() - t0, 1), "vram": vram, "modes": {}}
    for m in modes:
        n = max(tim[m]["n"], 1)
        rt["modes"][m] = {"windows": tim[m]["n"], "tiles_per_window": len(MODES[m][0]),
                          "ms_prep_cpu": round(1000 * tim[m]["prep"] / n, 2), "ms_gpu": round(1000 * tim[m]["gpu"] / n, 2),
                          "ms_post_cpu": round(1000 * tim[m]["post"] / n, 2),
                          "ms_total": round(1000 * (tim[m]["prep"] + tim[m]["gpu"] + tim[m]["post"]) / n, 2)}
    json.dump(rt, open(os.path.join(a.out, "runtime.json"), "w"), indent=1)
    print(json.dumps(rt, indent=1))


if __name__ == "__main__":
    main()
