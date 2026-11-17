"""Offline render of Ball Check cards: still, wide context and a 3x3 filmstrip.

Imports the crop/box helpers from ``footy_track.labeller.ball_check`` so the window
and box drawing are George's own. One process, one card at a time (the host throttles):
run as ``OMP_NUM_THREADS=2 nice -n 10 python render.py``. Never touches the live server.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import cv2  # noqa: E402

cv2.setNumThreads(2)
from footy_track.labeller import ball_check as bc  # noqa: E402

OUT = Path("/mnt/storage/exp/opus_ballcheck")
PAD_STILL = 6.0      # exactly George's still
PAD_WIDE = 14.0      # ~2.3x the window
OFFSETS = [-12, -8, -4, -2, 0, 2, 4, 8, 12]
TILE = 360
YELLOW, MAGENTA = (0, 255, 255), (255, 0, 255)


def _label(img, text):
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_boxes(crop, bbox, ob, frame_wh, win, out_wh):
    if ob is not None:
        bc._draw_box(crop, *bc._box_in_crop(ob, *frame_wh, win, out_wh), *out_wh,
                     MAGENTA, 2, bracket=False)
    bc._draw_box(crop, *bc._box_in_crop(bbox, *frame_wh, win, out_wh), *out_wh, YELLOW, 2)


def _up(crop, w, interp):
    s = w / crop.shape[1]
    return cv2.resize(crop, (w, max(1, round(crop.shape[0] * s))), interpolation=interp)


def render_card(card: dict, outdir: Path) -> bool:
    b = card["bbox"]
    bbox = (b["x"], b["y"], b["w"], b["h"])
    o = card.get("other_bbox")
    ob = (o["x"], o["y"], o["w"], o["h"]) if o else None
    cap = cv2.VideoCapture(card["video"])
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fi = card["frame_index"]
        first = max(0, fi + OFFSETS[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        want = {fi + d: d for d in OFFSETS}
        frames = {}
        for idx in range(first, fi + OFFSETS[-1] + 1):
            ok, fr = cap.read()
            if not ok:
                break
            if idx in want:
                frames[idx] = fr
    finally:
        cap.release()
    if fi not in frames:
        return False
    h_px, w_px = frames[fi].shape[:2]
    wh = (w_px, h_px)
    win = bc._crop_window(*bbox, w_px, h_px, PAD_STILL)
    x1, y1, x2, y2 = win
    # still: George's crop (nearest upscale to 640), box drawn on the upscaled crop
    still = _up(frames[fi][y1:y2, x1:x2], 640, cv2.INTER_NEAREST)
    _draw_boxes(still, bbox, ob, wh, win, (still.shape[1], still.shape[0]))
    # wide context
    wwin = bc._crop_window(*bbox, w_px, h_px, PAD_WIDE)
    wx1, wy1, wx2, wy2 = wwin
    wide = _up(frames[fi][wy1:wy2, wx1:wx2], 640, cv2.INTER_LANCZOS4)
    _draw_boxes(wide, bbox, ob, wh, wwin, (wide.shape[1], wide.shape[0]))
    # filmstrip: same window as the still on every tile; box only on the centre frame
    tiles = []
    blank = None
    for d in OFFSETS:
        fr = frames.get(fi + d)
        if fr is None:
            if blank is None:
                blank = _up(frames[fi][y1:y2, x1:x2], TILE, cv2.INTER_AREA) * 0
            t = blank.copy()
        else:
            t = _up(fr[y1:y2, x1:x2], TILE, cv2.INTER_LANCZOS4)
            if d == 0:
                _draw_boxes(t, bbox, ob, wh, win, (t.shape[1], t.shape[0]))
        _label(t, "t=0 (boxed)" if d == 0 else f"{d:+d} frames")
        cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), (60, 60, 60), 1)
        tiles.append(t)
    th, tw = tiles[0].shape[:2]
    strip = cv2.vconcat([cv2.hconcat([cv2.resize(t, (tw, th)) for t in tiles[r * 3:r * 3 + 3]])
                         for r in range(3)])
    outdir.mkdir(parents=True, exist_ok=True)
    for name, img in (("still", still), ("wide", wide), ("strip", strip)):
        cv2.imwrite(str(outdir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


def main() -> None:
    cards = json.load(open(OUT / "cards.json"))
    only = set(sys.argv[1:])
    for i, c in enumerate(cards):
        if only and c["id"] not in only and c["split"] not in only:
            continue
        d = OUT / "renders" / c["id"]
        if (d / "strip.jpg").exists():
            continue
        ok = render_card(c, d)
        print(i, c["id"], "ok" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
