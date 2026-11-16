"""SAM 3 / SAM 3.1 single-ball tracker driven from one seed (GPU, desktop venv ~/sam3-exp/venv with the upstream facebookresearch/sam3 package).

Mirrors the SAM2.1 ``crop512`` driver of ~/sam2-exp/code/sam2_tracks.py (one independent inference state per (seed, direction); a native-resolution crop
that follows the ball: it is re-centred on the last predicted position before every frame is fed; frames are injected one at a time).  Differences:
the model input is 1008 px (the crop is upscaled to it), fp16 autocast (RTX 2070 has no bf16; the upstream code enters a bf16 autocast in its constructors,
which is exited right after building), SAM 3.1 uses the multiplex tracker.

CLI (used by the Prefect flow ``sam3-track-eval``):  python sam3_track.py --spec spec.json --videos DIR --out DIR --config '{...}' --clips a,b --protocol oracle
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import crop_at, decode_window, iter_seeds, load_spec, pick  # noqa: E402

IMG = 1008
SAM3_DIR = Path(os.environ.get("SAM3_HOME", "/home/george/sam3-exp"))
CKPT = {"sam3": SAM3_DIR / "weights" / "sam3.pt", "sam3.1": SAM3_DIR / "weights" / "sam3.1_multiplex.pt"}

DEFAULT_CFG = {"model": "sam3", "prompt": "box", "crop": 512, "dtype": "fp16", "recentre": "mask",   # mask: every non-empty mask; conf: only score > 0
               "memsel": True, "interp": "linear", "offload": False}


def cfg_tag(c: dict) -> str:
    return "__".join(f"{k}-{str(c[k]).replace(' ', '_').replace(':', '-')}" for k in ("model", "prompt", "crop") if k in c) + (f"__{c['tag']}" if c.get("tag") else "")


class LazyImages:
    """Stand-in for the (T, 3, 1008, 1008) frame tensor of SAM2/SAM3 video states: frames are injected one by one, old ones dropped."""

    def __init__(self, T: int, keep: int = 4):
        self.T, self.keep, self.d = T, keep, {}

    def __len__(self):
        return self.T

    def __getitem__(self, i):
        return self.d[i]

    def put(self, i, t):
        self.d[i] = t
        for k in [k for k in self.d if abs(k - i) > self.keep]:
            del self.d[k]


def to_input(rgb: np.ndarray, interp: str = "linear"):
    import torch
    im = cv2.resize(rgb, (IMG, IMG), interpolation={"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}[interp])
    a = (im.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def analyse_mask(m):
    """m: bool tensor HxW on GPU -> None if empty, else dict(cx, cy, area, box, n_cc) of the largest connected component (mask px)."""
    nz = m.nonzero()
    if nz.shape[0] == 0:
        return None
    y0, x0 = (int(v) for v in nz.min(0).values)
    y1, x1 = (int(v) for v in nz.max(0).values)
    sub = m[y0:y1 + 1, x0:x1 + 1].cpu().numpy().astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(sub, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, 4]))
    bx, by, bw, bh = (int(v) for v in stats[k, :4])
    return {"cx": x0 + float(cents[k][0]) + 0.5, "cy": y0 + float(cents[k][1]) + 0.5, "area": int(stats[k, 4]),
            "box": [x0 + bx, y0 + by, x0 + bx + bw, y0 + by + bh], "n_cc": n - 1}


def patch_turing() -> None:
    """Upstream ViT MLPs call sam3.perflib.fused.addmm_act, which hard-casts to bfloat16 (fused cublasLt GELU epilogue).  The RTX 2070 (sm_75) has no bf16
    tensor cores: the emulated magma sgemmEx kernel was 59 % of the backbone time (0.63 s per 1008 px frame).  Replace it with the plain fp16-autocast path."""
    import sam3.model.vitdet as v
    v.addmm_act = lambda act, lin, x: act()(lin(x))
    # SAM 3.1 memory attention (decoder.functional_attention) forces the FlashAttention backend: not available on sm_75 ("No available kernel")
    import sam3.model.decoder as dec
    from torch.nn.attention import SDPBackend, sdpa_kernel
    dec.sdpa_kernel = lambda *a, **k: sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])


class Sam3Tracker:
    """Adapter over upstream SAM 3 (tracker of the full video model) or SAM 3.1 multiplex."""

    def __init__(self, cfg: dict):
        import torch
        self.torch = torch
        self.cfg = {**DEFAULT_CFG, **cfg}
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[self.cfg["dtype"]]
        self.version = self.cfg["model"]
        t0 = time.time()
        if self.cfg.get("patch_turing", True):
            patch_turing()
        if self.version == "sam3":
            from sam3.model_builder import build_sam3_video_model
            m = build_sam3_video_model(checkpoint_path=str(CKPT["sam3"]), load_from_HF=False, apply_temporal_disambiguation=bool(self.cfg["memsel"]))
            self.model = m
            self.tracker = m.tracker
            self.tracker.backbone = m.detector.backbone
            self._exit_bf16(self.tracker)
        elif self.version == "sam3.1":
            from sam3.model_builder import build_sam3_multiplex_video_predictor
            pred = build_sam3_multiplex_video_predictor(checkpoint_path=str(CKPT["sam3.1"]), use_fa3=False, use_rope_real=True, async_loading_frames=False)
            demo = pred.model
            self.model = demo
            self.tracker = demo.tracker.model        # Sam3VideoTrackingMultiplexDemo (backbone deleted by the builder: features come from the detector's tri backbone)
            self.tracker.backbone = demo.detector.backbone
            orig_fwd = self.tracker.forward_image

            def fwd(img_batch, *, need_sam3_out=False, need_interactive_out=False, need_propagation_out=False):
                # the tracker's own demo path asks for the detector ("sam3") head as well, whose flat output breaks its post-processing loop; the standalone
                # tracker only needs the interactive and propagation heads
                return orig_fwd(img_batch, need_sam3_out=False, need_interactive_out=need_interactive_out, need_propagation_out=need_propagation_out)
            self.tracker.forward_image = fwd
            for o in (pred, demo, demo.tracker, self.tracker):
                self._exit_bf16(o)
        else:
            raise NotImplementedError(self.version)
        self.model.eval()
        self.load_sec = time.time() - t0

    @staticmethod
    def _exit_bf16(obj):
        ctx = getattr(obj, "bf16_context", None)
        if ctx is not None:
            ctx.__exit__(None, None, None)
            obj.bf16_context = None

    def autocast(self):
        return self.torch.autocast("cuda", dtype=self.dtype) if self.dtype is not None else contextlib.nullcontext()

    # ------------------------------------------------------------------ one track
    def track(self, get_frame, seed_t: int, box, direction: str, lo: int, hi: int, prompt: str | None = None, crop: int | None = None, cands: dict | None = None) -> dict:
        """get_frame(p) -> HxWx3 uint8 RGB for processed frame p in [lo, hi].  box = seed box (x0,y0,x1,y1) px of the full frame.

        Window: forward seed..hi, backward lo..seed.  Returns {frames: {p: [present, cx, cy, area, score, x0, y0, x1, y1]}, sec, ...} in full-frame px.
        """
        with self.torch.inference_mode(), self.autocast():
            return self._track(get_frame, seed_t, box, direction, lo, hi, prompt or self.cfg["prompt"], self.cfg["crop"] if crop is None else crop, cands)

    def _track_text(self, get_frame, a, box, direction, lo, hi, prompt, crop):
        """Detector-only tracking: no SAM memory.  On every frame the text prompt (``text:<prompt>``) is run on the crop around the last accepted position and the
        nearest candidate (score >= text_min, within ``text_radius`` px) is taken; score field = logit of the detection probability."""
        fr = list(range(a, hi + 1)) if direction == "fwd" else list(range(lo, a + 1))
        if len(fr) < 2:
            return {"frames": {}, "sec": 0.0, "n_frames": 0}
        if direction == "bwd":
            fr = fr[::-1]
        t0 = time.time()
        centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        out = {str(a): [1, round(centre[0], 2), round(centre[1], 2), int((box[2] - box[0]) * (box[3] - box[1])), 9.0, *[round(v, 1) for v in box]]}
        if getattr(self, "_proc", None) is None:
            from sam3.model.sam3_image_processor import Sam3Processor
            self._proc = Sam3Processor(self.model.detector, confidence_threshold=self.cfg.get("text_conf", 0.05))
        from PIL import Image
        txt = prompt.split(":", 1)[1]
        rad, mn, C = self.cfg.get("text_radius", 150), self.cfg.get("text_min", 0.2), crop or 512
        for p in fr[1:]:
            frame = get_frame(p)
            c, ox, oy = crop_at(frame, centre[0], centre[1], C)
            st = self._proc.set_image(Image.fromarray(np.ascontiguousarray(c)))
            st = self._proc.set_text_prompt(prompt=txt, state=st)
            best = None
            for b, sc in zip(st["boxes"].float().cpu().tolist(), st["scores"].float().cpu().tolist()):
                bx = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
                d = float(np.hypot((bx[0] + bx[2]) / 2 - centre[0], (bx[1] + bx[3]) / 2 - centre[1]))
                if sc >= mn and d <= rad and (bx[2] - bx[0]) * (bx[3] - bx[1]) <= self.cfg.get("text_max_area", 2500) and (best is None or d < best[0]):
                    best = (d, bx, sc)
            if best is None:
                out[str(p)] = [0, None, None, 0, round(float(np.log(max(mn / 2, 1e-4) / (1 - mn / 2))), 3), None, None, None, None]
            else:
                _, bx, sc = best
                centre = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
                out[str(p)] = [1, round(centre[0], 2), round(centre[1], 2), int((bx[2] - bx[0]) * (bx[3] - bx[1])), round(float(np.log(sc / (1 - sc + 1e-6))), 3), *[round(v, 1) for v in bx]]
        return {"anchor": a, "dir": direction, "frames": out, "sec": round(time.time() - t0, 3), "n_frames": len(fr) - 1, "recoveries": []}

    def _track(self, get_frame, a, box, direction, lo, hi, prompt, crop, cands=None):
        if prompt.startswith("text:"):
            return self._track_text(get_frame, a, box, direction, lo, hi, prompt, crop)
        torch = self.torch
        k = int(self.cfg.get("frame_step", 1))   # k > 1: track every k-th frame only (throughput trick), the frames in between are interpolated below
        fr = list(range(a, hi + 1, k)) if direction == "fwd" else list(range(a, lo - 1, -k))[::-1]
        if len(fr) < 2:
            return {"frames": {}, "sec": 0.0, "n_frames": 0}
        T = len(fr)
        ai = 0 if direction == "fwd" else T - 1
        step = 1 if direction == "fwd" else -1
        t0 = time.time()
        x0b, y0b, x1b, y1b = box
        gcx, gcy = (x0b + x1b) / 2, (y0b + y1b) / 2
        f0 = get_frame(a)
        H, W = f0.shape[:2]
        C = crop if crop else None   # None = whole frame (squashed to 1008)
        images = LazyImages(T)
        origin = {}

        def feed(j, frame, cx, cy):
            if C:
                c, ox, oy = crop_at(frame, cx, cy, C)
                images.put(j, to_input(c, self.cfg["interp"]))
                origin[j] = (ox, oy, c.shape[1], c.shape[0])
            else:
                images.put(j, to_input(frame, self.cfg["interp"]))
                origin[j] = (0, 0, W, H)

        feed(ai, f0, gcx, gcy)
        ox, oy, cw, ch = origin[ai]
        st = self.tracker.init_state(video_height=ch, video_width=cw, num_frames=T, offload_video_to_cpu=self.cfg["offload"],
                                     offload_state_to_cpu=self.cfg["offload"])
        st["images"] = images
        bx = np.array([[(x0b - ox) / cw, (y0b - oy) / ch, (x1b - ox) / cw, (y1b - oy) / ch]], dtype=np.float32)
        pt = np.array([[(gcx - ox) / cw, (gcy - oy) / ch]], dtype=np.float32)
        if self.version == "sam3":
            kw = {}
            if prompt in ("box", "boxpoint"):
                kw["box"] = bx
            if prompt in ("point", "boxpoint"):
                kw["points"] = pt
                kw["labels"] = np.array([1], dtype=np.int32)
            self.tracker.add_new_points_or_box(st, frame_idx=ai, obj_id=1, **kw)
        else:   # SAM 3.1 multiplex tracker: point or mask prompts only; a "box" prompt becomes the ellipse inscribed in the box
            if prompt == "point":
                self.tracker.add_new_points(st, frame_idx=ai, obj_id=1, points=torch.tensor(pt, dtype=torch.float32), labels=torch.tensor([1], dtype=torch.int32),
                                            clear_old_points=True)
            else:
                yy, xx = np.mgrid[0:ch, 0:cw]
                ex0, ey0, ex1, ey1 = (x0b - ox), (y0b - oy), (x1b - ox), (y1b - oy)
                ell = (((xx + 0.5 - (ex0 + ex1) / 2) / max((ex1 - ex0) / 2, 1)) ** 2 + ((yy + 0.5 - (ey0 + ey1) / 2) / max((ey1 - ey0) / 2, 1)) ** 2) <= 1.0
                self.tracker.add_new_masks(st, frame_idx=ai, obj_ids=[1], masks=torch.from_numpy(ell[None]).to(torch.float32))
        seed_check = None
        if self.cfg.get("seed_check") and self.version == "sam3":
            seed_check = self._seed_check(f0, box, self.cfg["seed_check"])
        out: dict[str, list] = {}
        centre = (gcx, gcy)          # last confident / last mask centre (crop follows it)
        last_good = (gcx, gcy)       # last position with a positive object score (recovery search centre)
        pkw = {"propagate_preflight": True} if self.version == "sam3" else {}
        rec_mode, loss_thr, loss_k = self.cfg.get("recover", "none"), self.cfg.get("loss_thr", 0.0), self.cfg.get("loss_k", 3)
        max_rec, cooldown = self.cfg.get("max_recover", 8), self.cfg.get("recover_cooldown", 5)
        n_rec, lost_run, next_try = 0, 0, -1
        start = ai
        events = []
        while True:
            if self.version == "sam3.1":
                self.tracker.propagate_in_video_preflight(st)
            restart = None
            for fidx, _ids, _low, vmasks, scores in self.tracker.propagate_in_video(st, start_frame_idx=start, max_frame_num_to_track=T, reverse=(direction == "bwd"),
                                                                                    tqdm_disable=True, **pkw):
                p = fr[fidx]
                ox, oy, cw, ch = origin[fidx]
                res = analyse_mask(vmasks[0, 0] > 0)
                score = float(scores.reshape(-1)[0])
                if res is None:
                    out[str(p)] = [0, None, None, 0, round(score, 3), None, None, None, None]
                else:
                    bxs = res["box"]
                    out[str(p)] = [1, round(res["cx"] + ox, 2), round(res["cy"] + oy, 2), res["area"], round(score, 3),
                                   bxs[0] + ox, bxs[1] + oy, bxs[2] + ox, bxs[3] + oy]
                    if p != a and (self.cfg["recentre"] == "mask" or score > 0):
                        centre = (res["cx"] + ox, res["cy"] + oy)
                    if score > 0:
                        last_good = (res["cx"] + ox, res["cy"] + oy)
                lost = res is None or score < loss_thr
                lost_run = lost_run + 1 if lost else 0
                nxt = fidx + step
                if rec_mode != "none" and lost_run >= loss_k and n_rec < max_rec and abs(fidx - ai) >= next_try and 0 <= nxt < T:
                    hit = self._recover(get_frame(p), p, last_good, origin[fidx], cands, rec_mode)
                    next_try = abs(fidx - ai) + cooldown
                    if hit is not None:
                        n_rec += 1
                        events.append({"p": p, "box": [round(v, 1) for v in hit["box"]], "score": round(hit["score"], 3)})
                        ox2, oy2, cw2, ch2 = origin[fidx]
                        rb = np.array([[(hit["box"][0] - ox2) / cw2, (hit["box"][1] - oy2) / ch2, (hit["box"][2] - ox2) / cw2, (hit["box"][3] - oy2) / ch2]], dtype=np.float32)
                        self.tracker.add_new_points_or_box(st, frame_idx=fidx, obj_id=1, box=rb)
                        centre = ((hit["box"][0] + hit["box"][2]) / 2, (hit["box"][1] + hit["box"][3]) / 2)
                        last_good, lost_run = centre, 0
                        restart = fidx      # re-run this frame as a conditioning frame, then continue
                        break
                if 0 <= nxt < T:
                    feed(nxt, get_frame(fr[nxt]), centre[0], centre[1])
            if restart is None:
                break
            start = restart
        torch.cuda.synchronize()
        self.tracker.clear_all_points_in_video(st)
        del st
        if k > 1:   # linear interpolation of the skipped frames between two present predictions
            ps = sorted(int(x) for x in out)
            for p1, p2 in zip(ps, ps[1:]):
                r1, r2 = out[str(p1)], out[str(p2)]
                for q in range(p1 + 1, p2):
                    w2 = (q - p1) / (p2 - p1)
                    if r1[0] and r2[0]:
                        out[str(q)] = [1] + [round(r1[i] * (1 - w2) + r2[i] * w2, 2) for i in (1, 2)] + [int(r1[3]) , round(min(r1[4], r2[4]), 3)] + [round(r1[i] * (1 - w2) + r2[i] * w2, 1) for i in (5, 6, 7, 8)]
                    else:
                        out[str(q)] = [0, None, None, 0, round(min(r1[4], r2[4]), 3), None, None, None, None]
        return {"anchor": a, "dir": direction, "frames": out, "sec": round(time.time() - t0, 3), "n_frames": (T - 1) * k, "n_steps": T - 1, "recoveries": events, "seed_check": seed_check}

    def _seed_check(self, frame, box, prompt):
        """Is the seed box a "ball" according to the SAM 3 text detector?  Best IoU of any text detection (score >= 0.05) on a 512 px crop around the seed with the
        seed box, and that detection's score (the yellow-boot-vs-yellow-ball case: a track from a false seed is confident and consistent, only the seed can be checked)."""
        if getattr(self, "_proc", None) is None:
            from sam3.model.sam3_image_processor import Sam3Processor
            self._proc = Sam3Processor(self.model.detector, confidence_threshold=0.05)
        from PIL import Image
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        c, ox, oy = crop_at(frame, cx, cy, 512)
        st = self._proc.set_image(Image.fromarray(np.ascontiguousarray(c)))
        st = self._proc.set_text_prompt(prompt=prompt, state=st)
        best = {"iou": 0.0, "score": 0.0}
        for b, sc in zip(st["boxes"].float().cpu().tolist(), st["scores"].float().cpu().tolist()):
            bx = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
            iw, ih = min(bx[2], box[2]) - max(bx[0], box[0]), min(bx[3], box[3]) - max(bx[1], box[1])
            inter = max(iw, 0) * max(ih, 0)
            iou = inter / max((bx[2] - bx[0]) * (bx[3] - bx[1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter, 1e-9)
            if iou > best["iou"] or (iou == best["iou"] and sc > best["score"]):
                best = {"iou": round(iou, 3), "score": round(sc, 3)}
        return best

    # ------------------------------------------------------------------ lost-track recovery (SAM 3 only: text detector shares the tracker's backbone)
    def _recover(self, frame, p, centre, org, cands, mode):
        """Candidate box (full-frame px) near ``centre`` on frame ``p``: text-prompt detection on a crop (mode text:<prompt>) or an external candidate list."""
        rad = self.cfg.get("recover_radius", 150)
        if mode == "cands":
            best = None
            for c in (cands or {}).get(p, []):
                d = float(np.hypot((c[0] + c[2]) / 2 - centre[0], (c[1] + c[3]) / 2 - centre[1]))
                if c[4] >= self.cfg.get("cand_min_conf", 0.5) and d <= rad and (best is None or d < best[0]):
                    best = (d, {"box": c[:4], "score": c[4]})
            return best[1] if best else None
        if self.version != "sam3":
            return None
        if getattr(self, "_proc", None) is None:
            from sam3.model.sam3_image_processor import Sam3Processor
            self._proc = Sam3Processor(self.model.detector, confidence_threshold=self.cfg.get("recover_conf", 0.1))
        from PIL import Image
        prompt = mode.split(":", 1)[1] if ":" in mode else "ball"
        c, ox, oy = crop_at(frame, centre[0], centre[1], self.cfg.get("recover_crop", 512))
        st = self._proc.set_image(Image.fromarray(np.ascontiguousarray(c)))
        st = self._proc.set_text_prompt(prompt=prompt, state=st)
        best = None
        for b, s in zip(st["boxes"].float().cpu().tolist(), st["scores"].float().cpu().tolist()):
            bx = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
            d = float(np.hypot((bx[0] + bx[2]) / 2 - centre[0], (bx[1] + bx[3]) / 2 - centre[1]))
            area = (b[2] - b[0]) * (b[3] - b[1])
            if s >= self.cfg.get("recover_min_score", 0.3) and d <= rad and area <= self.cfg.get("recover_max_area", 2500) and (best is None or d < best[0]):
                best = (d, {"box": bx, "score": s})
        return best[1] if best else None


# ---------------------------------------------------------------------------------------------- CLI
def load_cands(path: Path, w: int, h: int, stride: int) -> dict[int, list]:
    """RT-DETR machine-label jsonl (r3_detect format, every original frame) -> {processed frame: [[x0,y0,x1,y1,conf], ...]} for ball-class rows."""
    out: dict[int, list] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not any(t in ("in_play_ball", "out_of_play_ball", "ball") for t in r["tags"]) or r["frame_index"] % stride:
            continue
        b = r["bbox"]
        out.setdefault(r["frame_index"] // stride, []).append([b["x"] * w, b["y"] * h, (b["x"] + b["w"]) * w, (b["y"] + b["h"]) * h, r["confidence"]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--videos", required=True, help="dir with <clip>.mp4 (staged copies)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--clips", default="")
    ap.add_argument("--protocol", default="oracle", choices=["oracle", "rtdetr"])
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--max-reach", type=int, default=0, help="override spec max_reach (frames each way)")
    ap.add_argument("--n-seeds", type=int, default=0, help="use only N evenly spaced seeds per clip (0 = all)")
    a = ap.parse_args()
    import torch
    cfg = {**DEFAULT_CFG, **json.loads(a.config)}
    spec = load_spec(a.spec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.clips.split(",")) if a.clips else None
    runner = None
    tag = cfg_tag(cfg)
    for name, clip in sorted(spec["clips"].items()):
        if only and name not in only:
            continue
        seeds = pick(iter_seeds(clip, a.protocol), a.n_seeds)
        if not seeds:
            continue
        dst = out / f"tracks__{name}__{tag}__{a.protocol}.json"
        doc = json.loads(dst.read_text()) if dst.exists() else {"meta": {"clip": name, "cfg": cfg, "protocol": a.protocol, "host": os.uname().nodename,
                                                                          "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                                                          "w": clip["w"], "h": clip["h"], "n_frames": clip["n_frames"], "stride": clip["stride"]},
                                                                "tracks": []}
        have = {(t["seed"], t["dir"]) for t in doc["tracks"]}
        mr = a.max_reach or spec.get("max_reach", 400)
        todo = [(s, d) for s in seeds for d in ("fwd", "bwd") if (s["id"], d) not in have]
        if a.max_tracks:
            todo = todo[:a.max_tracks]
        if not todo:
            print(f"{name}: nothing to do", flush=True)
            continue
        if runner is None:
            runner = Sam3Tracker(cfg)
            torch.cuda.reset_peak_memory_stats()
            print(f"model {cfg['model']} loaded in {runner.load_sec:.1f}s", flush=True)
        lo = max(0, min(s["t"] for s, _ in todo) - mr)
        hi = min(clip["n_frames"] - 1, max(s["t"] for s, _ in todo) + mr)
        cands = next((load_cands(p, clip["w"], clip["h"], clip["stride"]) for p in (SAM3_DIR / "out" / d / f"{name}.jsonl" for d in ("rtdetr_full", "rtdetr_dev"))
                     if p.exists()), None) if cfg.get("recover") == "cands" else None
        t0 = time.time()
        frames = decode_window(str(Path(a.videos) / f"{name}.mp4"), clip["w"], clip["h"], clip["stride"], lo, hi)
        print(f"{name}: decoded {len(frames)} frames in {time.time() - t0:.1f}s; {len(todo)} tracks", flush=True)
        for i, (s, d) in enumerate(todo):
            tlo, thi = max(lo, s["t"] - mr), min(hi, s["t"] + mr)
            rec = runner.track(frames.__getitem__, s["t"], s["box"], d, tlo, thi, cands=cands)
            rec.update({"seed": s["id"], "dir": d, "t": s["t"], "source": s["source"]})
            doc["tracks"].append(rec)
            tf = sum(x["n_frames"] for x in doc["tracks"])
            ts = sum(x["sec"] for x in doc["tracks"])
            doc["perf"] = {"peak_vram_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20), "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
                           "frames_propagated": tf, "seconds": round(ts, 1), "sec_per_frame": round(ts / max(tf, 1), 4), "load_sec": round(runner.load_sec, 1)}
            tmp = dst.with_name(dst.name + ".tmp")
            tmp.write_text(json.dumps(doc))
            os.replace(tmp, dst)
            print(f"  {name} seed {s['id']}@{s['t']} {d}: {rec['n_frames']} fr in {rec['sec']}s -> {doc['perf']}", flush=True)
        del frames


if __name__ == "__main__":
    main()
