"""Render augmented 5-channel training samples and check the neighbour channels stay aligned with frame t.

For each sample (taken from the *same* Ultralytics train dataset / augmentation pipeline the trainer uses) it writes

  left   frame t as RGB with the GT boxes (ball = red)
  middle temporal composite  R = gray(t-2) aligned, G = gray(t), B = gray(t+2) aligned  (static, aligned content is
         grey; motion shows as coloured fringes; a misaligned neighbour would show fringes over the whole picture)
  right  4x zoom around the ball: gray(t-2) | gray(t) | gray(t+2)

and prints the mean |gray(t) - neighbour| of augmented vs non-augmented samples (a geometric transform applied to the
t channels only would blow the augmented number up).

    python viz_aug.py --data-root ~/exp/temporal/data --out ~/exp/temporal/aug_check --n 8
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from common import BALL_CLS, write_data_yaml
from multichannel import patch_ultralytics_multichannel


def make_dataset(root: Path, tmp: Path, augment: bool, imgsz: int):
    from ultralytics.models.yolo.detect import DetectionTrainer

    data_yaml = write_data_yaml(root, tmp / "data.yaml", 5)
    tr = DetectionTrainer(overrides=dict(data=str(data_yaml), imgsz=imgsz, batch=4, model="yolo11n.yaml", device="cpu",
                                         workers=0, plots=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, project=str(tmp),
                                         name="viz", exist_ok=True))
    from ultralytics.nn.tasks import DetectionModel

    tr.model = DetectionModel("yolo11n.yaml", ch=5, nc=7, verbose=False)  # build_dataset reads the stride from it
    return tr.build_dataset(tr.data["train"], mode="train" if augment else "val", batch=4)


def to_hwc(t):
    return t.numpy().transpose(1, 2, 0)


def diffs(img5):
    g = cv2.cvtColor(np.ascontiguousarray(img5[..., 1:4]), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(g - img5[..., 0]).mean()), float(np.abs(img5[..., 4] - g).mean())


def render(item, s=640):
    img5 = to_hwc(item["img"])
    h, w = img5.shape[:2]
    bgr = np.ascontiguousarray(img5[..., 1:4])
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    comp = np.dstack([img5[..., 4], g, img5[..., 0]])  # BGR order: B=next, G=t, R=prev
    boxes = item["bboxes"].numpy()
    cls = item["cls"].numpy().ravel()
    left = bgr.copy()
    ball_c = None
    for (cx, cy, bw, bh), c in zip(boxes, cls):
        x1, y1, x2, y2 = int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        col = (0, 0, 255) if int(c) == BALL_CLS else (255, 255, 0)
        cv2.rectangle(left, (x1, y1), (x2, y2), col, 2 if int(c) == BALL_CLS else 1)
        if int(c) == BALL_CLS and ball_c is None:
            ball_c = (int(cx * w), int(cy * h))
    zoom = np.zeros((s, s, 3), np.uint8)
    if ball_c is not None:
        r = 24
        x0, y0 = max(ball_c[0] - r, 0), max(ball_c[1] - r, 0)
        x1, y1 = min(x0 + 2 * r, w), min(y0 + 2 * r, h)
        tiles = []
        for ch in (img5[..., 0], g, img5[..., 4]):
            crop = cv2.resize(ch[y0:y1, x0:x1], (s // 3, s // 3), interpolation=cv2.INTER_NEAREST)
            tiles.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
        zoom[: s // 3, : 3 * (s // 3)] = np.hstack(tiles)
        # difference row
        gt_, gp_, gn_ = (img5[y0:y1, x0:x1, 0].astype(np.int16), g[y0:y1, x0:x1].astype(np.int16),
                         img5[y0:y1, x0:x1, 4].astype(np.int16))
        d = [np.clip(np.abs(a - b) * 4, 0, 255).astype(np.uint8) for a, b in ((gp_, gt_), (gn_, gp_))]
        row = [cv2.cvtColor(cv2.resize(x, (s // 3, s // 3), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR) for x in d]
        zoom[s // 3 : 2 * s // 3, : 2 * (s // 3)] = np.hstack(row)
    panels = [cv2.resize(left, (s, s)), cv2.resize(comp, (s, s)), zoom]
    return np.hstack(panels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--stat-n", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    patch_ultralytics_multichannel()
    a.out.mkdir(parents=True, exist_ok=True)
    root = a.data_root / "k2"
    random.seed(a.seed)
    np.random.seed(a.seed)
    aug = make_dataset(root, a.out / "_tmp", True, a.imgsz)
    plain = make_dataset(root, a.out / "_tmp", False, a.imgsz)
    ball_idx = [i for i, lb in enumerate(plain.labels) if (lb["cls"] == BALL_CLS).any()]
    sel = random.sample(ball_idx, min(a.stat_n, len(ball_idx)))
    # channel-order check on a real (non-augmented) batch item: tensor channel c must be stored channel c (B,G,R stay in
    # that order for 5 channels), i.e. the pretrained R/G/B stem filters belong on input channels 3/2/1
    i0 = ball_idx[0]
    t = plain[i0]["img"].numpy()
    im, _, (rh, rw) = plain.load_image(i0)
    order_ok = False
    for top in range(0, t.shape[1] - rh + 1):  # the val letterbox pads to a stride multiple (here 736x1312)
        for left in range(0, t.shape[2] - rw + 1):
            if np.array_equal(t[0, top : top + rh, left : left + rw], im[..., 0]):
                order_ok = all(np.array_equal(t[c, top : top + rh, left : left + rw], im[..., c]) for c in range(5))
                break
        if order_ok:
            break
    assert order_ok, "batch tensor channels are not the stored channel order"
    st_aug, st_plain = [], []
    for k, i in enumerate(sel):
        it = aug[i]
        st_aug.append(diffs(to_hwc(it["img"])))
        st_plain.append(diffs(to_hwc(plain[i]["img"])))
        if k < a.n:
            cv2.imwrite(str(a.out / f"aug_{k:02d}.jpg"), render(it), [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(a.out / f"plain_{k:02d}.jpg"), render(plain[i]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    rep = {"n": len(sel), "mean_absdiff_prev_aug": float(np.mean([x[0] for x in st_aug])),
           "mean_absdiff_next_aug": float(np.mean([x[1] for x in st_aug])),
           "mean_absdiff_prev_plain": float(np.mean([x[0] for x in st_plain])),
           "mean_absdiff_next_plain": float(np.mean([x[1] for x in st_plain])),
           "img_shape_aug": list(to_hwc(aug[sel[0]]["img"]).shape), "tensor_channel_order_is_stored_order": order_ok,
           "means_bgr_t_first_item": [float(im[..., c].mean()) for c in range(5)]}
    (a.out / "aug_stats.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
