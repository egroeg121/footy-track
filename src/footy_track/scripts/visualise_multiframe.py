"""Eyeball aligned stacks / diffs with the ball GT marked.

Reads samples from a built dataset (``meta.jsonl`` + TIFF stacks) and writes a few PNGs.
Top row: frame t (ball boxed) and the aligned absolute difference |t - (t-k)| over the
whole frame (x4 gain). Bottom row, crops around the ball (x3): aligned gray(t-k), frame t,
aligned gray(t+k), |t-prev| aligned, |next-t| aligned, |t-prev| with NO alignment.

    uv run python -m footy_track.scripts.visualise_multiframe \
        --dataset /mnt/storage/footy_data/multiframe_v1/k2 --split heldout --n 3 \
        --out docs/experiments/multiframe_dataset
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from footy_track.multiframe.frames import BALL_ID, ClipReader, read_yolo_labels

GAIN = 4
HALF = 48  # crop half-size in px at 1080p
ZOOM = 3


def load_stack(path: Path) -> np.ndarray:
    ok, pages = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
    assert ok and len(pages) == 5, path
    return np.stack(pages, axis=2)  # H W 5


def crop(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = min(max(cx - HALF, 0), w - 2 * HALF)
    y0 = min(max(cy - HALF, 0), h - 2 * HALF)
    c = img[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]
    c = cv2.resize(c, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    if c.ndim == 2:
        c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
    cv2.circle(c, ((cx - x0) * ZOOM, (cy - y0) * ZOOM), 7 * ZOOM, (0, 255, 255), 1)
    return c


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img


def render(dataset: Path, m: dict) -> np.ndarray | None:
    split_dir = {"valid": "val"}.get(m["split"], m["split"])
    st = load_stack(dataset / "images" / split_dir / f"{m['id']}.tiff")
    labs = read_yolo_labels(dataset / "labels" / split_dir / f"{m['id']}.txt")
    balls = [(x, y, w, h) for c, x, y, w, h in labs if c == BALL_ID]
    if not balls:
        return None
    prev, bgr, nxt = st[..., 0], st[..., 1:4], st[..., 4]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    cx, cy = int(balls[0][0] * W), int(balls[0][1] * H)
    d_prev = cv2.absdiff(gray, prev)
    d_next = cv2.absdiff(nxt, gray)
    # unaligned: the raw neighbour straight from the video
    rd = ClipReader(Path(m["video"]), cache=4)
    raw = rd.get(m["frame"] - m["k"])
    rd.close()
    d_raw = cv2.absdiff(gray, cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)) if raw is not None else np.zeros_like(gray)

    def amp(d):
        return np.clip(d.astype(np.int32) * GAIN, 0, 255).astype(np.uint8)

    full = bgr.copy()
    bx, by, bw, bh = balls[0]
    cv2.rectangle(full, (int((bx - bw / 2) * W) - 6, int((by - bh / 2) * H) - 6),
                  (int((bx + bw / 2) * W) + 6, int((by + bh / 2) * H) + 6), (0, 255, 255), 2)
    top = np.hstack([
        label(cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA), f"frame t  ({m['id']})"),
        label(cv2.cvtColor(cv2.resize(amp(d_prev), (960, 540), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_GRAY2BGR), f"aligned |t - (t-{m['k']})| x{GAIN}  [{m['prev']['status']}]"),
    ])
    panels = [
        label(crop(prev, cx, cy), "gray(t-k) aligned"), label(crop(bgr, cx, cy), "RGB(t)"),
        label(crop(nxt, cx, cy), "gray(t+k) aligned"), label(crop(amp(d_prev), cx, cy), "|t-prev| aligned"),
        label(crop(amp(d_next), cx, cy), "|next-t| aligned"), label(crop(amp(d_raw), cx, cy), "|t-prev| UNaligned"),
    ]
    bottom = np.hstack(panels)
    pad = max(top.shape[1], bottom.shape[1])
    pad_to = lambda x: np.pad(x, ((0, 0), (0, pad - x.shape[1]), (0, 0)))
    return np.vstack([pad_to(top), pad_to(bottom)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", nargs="*", help="explicit image ids")
    args = ap.parse_args(argv)
    metas = [json.loads(line) for line in (args.dataset / "meta.jsonl").read_text().splitlines()]
    pool = [m for m in metas if m["split"] == args.split and m.get("n_ball")
            and m["prev"]["status"] == "ok" and m["next"]["status"] == "ok"]
    if args.ids:
        pool = [m for m in metas if m["id"] in args.ids]
    else:
        random.Random(args.seed).shuffle(pool)
    args.out.mkdir(parents=True, exist_ok=True)
    for m in pool[: args.n]:
        img = render(args.dataset, m)
        if img is not None:
            p = args.out / f"stack_{args.split}_{m['id'].split('.rf.')[0]}.png"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(p)


if __name__ == "__main__":
    main()
