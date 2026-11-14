"""Dump per-image ball detections of one model on val / test / heldout (GPU part of the evaluation).

    python evaluate.py --kind yolo --arm B --weights runs/b_s0/weights/last.pt --data-root ~/exp/temporal/data --out preds.json
    python evaluate.py --kind rtdetr --weights rtdetr_1920_v11/best.pt ...        # production reference, frame t only

Inference is explicit (no Ultralytics predictor) because the predictor reverses all channels of a numpy input, which
would scramble the 5-channel layout: letterbox to 1280 (INTER_AREA, pad 114 on every channel, rect 736x1280 like val),
fp16 forward, NMS (conf 0.001, IoU 0.7, multi-label, ball class only), boxes mapped back to the original 1920x1080 frame.
Metrics (AP50, recall@10px, ...) are computed from this file by report.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from common import BALL_CLS, EVAL_SPLITS, IMG_H, IMG_W, bgr_of, load_meta, read_labels, read_tiff5

MOTION_WIN = 7  # half window (px) around the GT centre for the temporal-difference score


def motion_scores(img5: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """p90 of |gray(t) - neighbour| in a 15x15 window at the GT ball (the neighbours are camera-aligned already)."""
    g = cv2.cvtColor(bgr_of(img5), cv2.COLOR_BGR2GRAY).astype(np.int16)
    x0, x1 = int(max(cx - MOTION_WIN, 0)), int(min(cx + MOTION_WIN + 1, IMG_W))
    y0, y1 = int(max(cy - MOTION_WIN, 0)), int(min(cy + MOTION_WIN + 1, IMG_H))
    win = (slice(y0, y1), slice(x0, x1))
    dp = np.abs(g[win] - img5[win][..., 0].astype(np.int16))
    dn = np.abs(img5[win][..., 4].astype(np.int16) - g[win])
    return float(np.percentile(dp, 90)), float(np.percentile(dn, 90))


class YoloRunner:
    def __init__(self, weights: str, arm: str, imgsz: int = 1280):
        import torch
        from ultralytics import YOLO

        self.torch, self.arm, self.imgsz = torch, arm, imgsz
        self.model = YOLO(weights).model.cuda().half().eval()
        self.in_ch = self.model.yaml["channels"]
        assert self.in_ch == (3 if arm == "A" else 5), f"arm {arm} but model has {self.in_ch} input channels"

    @staticmethod
    def letterbox(x: np.ndarray, imgsz: int):
        h, w = x.shape[:2]
        r = imgsz / max(h, w)
        nw, nh = round(w * r), round(h * r)
        x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
        if x.ndim == 2:
            x = x[..., None]
        ph = (-nh) % 32
        top, left = ph // 2, 0
        out = np.full((nh + ph, nw, x.shape[2]), 114, np.uint8)
        out[top : top + nh, left : left + nw] = x
        return out, r, left, top

    def __call__(self, img5: np.ndarray, conf: float = 0.001):
        from ultralytics.utils.nms import non_max_suppression

        x = img5[..., [3, 2, 1]] if self.arm == "A" else img5  # A: RGB (what stock training feeds); B: stored order
        x, r, left, top = self.letterbox(np.ascontiguousarray(x), self.imgsz)
        t = self.torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).cuda().half() / 255
        with self.torch.no_grad():
            y = self.model(t)
        y = y[0] if isinstance(y, (list, tuple)) else y
        det = non_max_suppression(y, conf_thres=conf, iou_thres=0.7, classes=[BALL_CLS], multi_label=True, max_det=100)[0]
        det = det.float().cpu().numpy()
        res = []
        for x1, y1, x2, y2, c, _ in det:
            x1, x2 = (x1 - left) / r, (x2 - left) / r
            y1, y2 = (y1 - top) / r, (y2 - top) / r
            res.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(c)])
        return res


class RtdetrRunner:
    def __init__(self, weights: str, imgsz: int = 1280):
        from ultralytics import YOLO

        self.model, self.imgsz = YOLO(weights), imgsz

    def __call__(self, img5: np.ndarray, conf: float = 0.001):
        r = self.model.predict(bgr_of(img5), imgsz=self.imgsz, conf=conf, classes=[BALL_CLS], max_det=300, verbose=False,
                               device=0, half=True)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xywh, cf = b.xywh.cpu().numpy(), b.conf.cpu().numpy()
        return [[*map(float, xywh[i]), float(cf[i])] for i in range(len(cf))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["yolo", "rtdetr"], required=True)
    ap.add_argument("--arm", choices=["A", "B"], default="A")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", type=Path, required=True, help="dir containing k2/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", default=",".join(EVAL_SPLITS))
    ap.add_argument("--limit", type=int, default=0, help="first N images per split (smoke test)")
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()

    root = a.data_root / "k2"
    runner = YoloRunner(a.weights, a.arm, a.imgsz) if a.kind == "yolo" else RtdetrRunner(a.weights, a.imgsz)
    meta = load_meta(root)
    out = {"kind": a.kind, "arm": a.arm, "weights": a.weights, "imgsz": a.imgsz, "images": []}
    t0 = time.time()
    for split in a.splits.split(","):
        files = sorted((root / "images" / split).glob("*.tiff"))
        if a.limit:
            files = files[: a.limit]
        for i, p in enumerate(files):
            img5 = read_tiff5(p)
            lab = read_labels(root / "labels" / split / (p.stem + ".txt"), BALL_CLS)
            gt = [[c * IMG_W, d * IMG_H, w * IMG_W, h * IMG_H] for _, c, d, w, h in lab]
            m = meta.get((split, p.stem), {})
            rec = {"id": p.stem, "split": split, "gt": gt, "dets": runner(img5)[:60],
                   "prev_status": m.get("prev", {}).get("status"), "next_status": m.get("next", {}).get("status"),
                   "strict_heldout": bool(m.get("strict_heldout", False)), "clip_id": m.get("clip_id")}
            if gt:
                rec["motion_prev"], rec["motion_next"] = motion_scores(img5, gt[0][0], gt[0][1])
            out["images"].append(rec)
            if i % 50 == 0:
                print(f"{split} {i}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(a.out)
    print("wrote", a.out, len(out["images"]), "images", out["seconds"], "s")


if __name__ == "__main__":
    main()
