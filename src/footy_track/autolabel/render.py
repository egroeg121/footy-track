"""Draw the question image for the Claude verify pilot (needs OpenCV; no GPU)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from footy_track.autolabel.verify_pilot import (
    CROP_PX,
    GRID,
    VIEW_PX,
    Cand,
    cell_label,
)

PALETTE = [  # BGR, distinct on grass and on shirts
    (0, 0, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 200, 255),
    (255, 255, 0),
    (0, 255, 128),
]
SCALE = VIEW_PX / CROP_PX
THUMB = (480, 270)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale, colour, thick):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def _grid(view: np.ndarray) -> None:
    step = VIEW_PX // GRID
    layer = view.copy()
    for i in range(1, GRID):
        cv2.line(layer, (i * step, 0), (i * step, VIEW_PX), (255, 255, 255), 1)
        cv2.line(layer, (0, i * step), (VIEW_PX, i * step), (255, 255, 255), 1)
    cv2.addWeighted(layer, 0.3, view, 0.7, 0, view)
    for i in range(GRID):
        c = cell_label(i, 0)[0]
        for y in (18, VIEW_PX - 6):
            _text(view, c, (i * step + step // 2 - 8, y), 0.6, (210, 210, 210), 1)
        r = str(i + 1)
        for x in (4, VIEW_PX - 18):
            _text(view, r, (x, i * step + step // 2 + 6), 0.6, (210, 210, 210), 1)


def _place_number(
    taken: list, box: tuple, n: int, size: tuple[int, int]
) -> tuple[int, int]:
    """Top-left for the number beside ``box`` that overlaps no other box or number."""
    (tw, th), _ = cv2.getTextSize(str(n), FONT, 1.3, 3)
    x1, y1, x2, y2 = box
    gap = 6
    options = [
        (x2 + gap, y1 + th),
        (x1 - gap - tw, y1 + th),
        (x2 + gap, y2 + th + gap),
        (x1 - gap - tw, y2 + th + gap),
        (x1, y1 - gap),
        (x1, y2 + th + gap),
    ]
    for ox, oy in options:
        r = (ox, oy - th, ox + tw, oy)
        inside = r[0] >= 0 and r[1] >= 0 and r[2] <= size[0] and r[3] <= size[1]
        if inside and not any(
            r[0] < t[2] and t[0] < r[2] and r[1] < t[3] and t[1] < r[3] for t in taken
        ):
            taken.append(r)
            return ox, oy
    ox, oy = options[0]
    taken.append((ox, oy - th, ox + tw, oy))
    return ox, oy


def render_question(
    frame_bgr: np.ndarray,
    origin: tuple[int, int],
    cands: Sequence[Cand],
    out_path: Path,
) -> None:
    """Write a PNG: the crop at 3x with numbered boxes and a grid, plus a context thumbnail."""
    x0, y0 = origin
    crop = frame_bgr[y0 : y0 + CROP_PX, x0 : x0 + CROP_PX]
    view = cv2.resize(crop, (VIEW_PX, VIEW_PX), interpolation=cv2.INTER_CUBIC)
    _grid(view)
    boxes = []
    for c in cands:
        bx1 = int((c.cx - c.w / 2 - x0) * SCALE) - 3
        by1 = int((c.cy - c.h / 2 - y0) * SCALE) - 3
        bx2 = int((c.cx + c.w / 2 - x0) * SCALE) + 3
        by2 = int((c.cy + c.h / 2 - y0) * SCALE) + 3
        boxes.append((bx1, by1, bx2, by2))
    taken = list(boxes)
    for i, b in enumerate(boxes):
        col = PALETTE[i % len(PALETTE)]
        cv2.rectangle(view, b[:2], b[2:], col, 1, cv2.LINE_AA)
        _text(
            view,
            str(i + 1),
            _place_number(taken, b, i + 1, (VIEW_PX, VIEW_PX)),
            1.3,
            col,
            3,
        )
    side = np.full((VIEW_PX, THUMB[0], 3), 32, np.uint8)
    thumb = cv2.resize(frame_bgr, THUMB, interpolation=cv2.INTER_AREA)
    sx, sy = THUMB[0] / frame_bgr.shape[1], THUMB[1] / frame_bgr.shape[0]
    cv2.rectangle(
        thumb,
        (int(x0 * sx), int(y0 * sy)),
        (int((x0 + CROP_PX) * sx), int((y0 + CROP_PX) * sy)),
        (0, 255, 255),
        2,
    )
    side[: THUMB[1]] = thumb
    _text(
        side,
        "full frame (crop in yellow)",
        (8, THUMB[1] + 26),
        0.55,
        (230, 230, 230),
        1,
    )
    if not cands:
        _text(side, "no boxes drawn", (8, THUMB[1] + 70), 0.7, (230, 230, 230), 1)
    for i, c in enumerate(cands):
        y = THUMB[1] + 70 + i * 40
        col = PALETTE[i % len(PALETTE)]
        cv2.rectangle(side, (10, y - 22), (34, y + 2), col, 2)
        _text(side, f"{i + 1}  conf {c.conf:.2f}", (46, y), 0.7, (240, 240, 240), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.hstack([view, side]))


def contact_sheet(
    items: Sequence[tuple[Path, str, tuple[float, float] | None]],
    out_path: Path,
    *,
    origin_of: Sequence[tuple[int, int]],
    cols: int = 4,
    tile: int = 480,
) -> None:
    """Mosaic of question images (crop panel only), caption on top, GT as a green ring."""
    tiles = []
    for (path, caption, gt), (x0, y0) in zip(items, origin_of, strict=True):
        img = cv2.imread(str(path))[:VIEW_PX, :VIEW_PX].copy()
        if gt is not None:
            p = (int((gt[0] - x0) * SCALE), int((gt[1] - y0) * SCALE))
            cv2.circle(img, p, 34, (0, 255, 0), 3, cv2.LINE_AA)
        t = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        bar = np.full((44, tile, 3), 24, np.uint8)
        cv2.putText(
            bar, caption[:60], (6, 28), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        tiles.append(np.vstack([bar, t]))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
