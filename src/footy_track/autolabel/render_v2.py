"""Draw the v2 question image: a tight 4x zoom on one candidate plus a context panel.

Needs OpenCV; no GPU. The candidate is always at the centre of the zoom (the patch is
padded with black where it leaves the frame).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from footy_track.autolabel.verify_v2 import (
    CONTEXT_PX,
    PATCH_PX,
    VIEW_PX,
    Candidate,
)

YELLOW = (0, 255, 255)  # BGR
SCALE = VIEW_PX / PATCH_PX
HEADER = 26
GAP = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_BOX_PX = 10.0  # native; keeps the box visible for tiny detections


def crop_padded(frame: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    """``size`` square at (x0, y0); outside the frame is black."""
    h, w = frame.shape[:2]
    out = np.zeros((size, size, 3), np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + size, w), min(y0 + size, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out


def _label(img: np.ndarray, s: str, org: tuple[int, int]) -> None:
    cv2.putText(img, s, org, FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)


def render_question(
    frame_bgr: np.ndarray, cand: Candidate, out_path: Path
) -> tuple[int, int]:
    """Write the PNG; returns the native top-left of the zoomed patch (for the scorer)."""
    px0 = round(cand.cx - PATCH_PX / 2)
    py0 = round(cand.cy - PATCH_PX / 2)
    zoom = cv2.resize(
        crop_padded(frame_bgr, px0, py0, PATCH_PX),
        (VIEW_PX, VIEW_PX),
        interpolation=cv2.INTER_CUBIC,
    )
    bw, bh = max(cand.w, MIN_BOX_PX), max(cand.h, MIN_BOX_PX)
    p1 = (
        int((cand.cx - bw / 2 - px0) * SCALE) - 3,
        int((cand.cy - bh / 2 - py0) * SCALE) - 3,
    )
    p2 = (
        int((cand.cx + bw / 2 - px0) * SCALE) + 3,
        int((cand.cy + bh / 2 - py0) * SCALE) + 3,
    )
    cv2.rectangle(zoom, p1, p2, YELLOW, 1, cv2.LINE_AA)

    cx0 = round(cand.cx - CONTEXT_PX / 2)
    cy0 = round(cand.cy - CONTEXT_PX / 2)
    ctx = crop_padded(frame_bgr, cx0, cy0, CONTEXT_PX)
    cv2.rectangle(
        ctx,
        (px0 - cx0, py0 - cy0),
        (px0 - cx0 + PATCH_PX, py0 - cy0 + PATCH_PX),
        YELLOW,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        ctx,
        (int(cand.cx - bw / 2 - cx0) - 1, int(cand.cy - bh / 2 - cy0) - 1),
        (int(cand.cx + bw / 2 - cx0) + 1, int(cand.cy + bh / 2 - cy0) + 1),
        YELLOW,
        1,
        cv2.LINE_AA,
    )
    width = VIEW_PX + GAP + CONTEXT_PX
    img = np.full((HEADER + VIEW_PX, width, 3), 32, np.uint8)
    img[HEADER:, :VIEW_PX] = zoom
    img[HEADER : HEADER + CONTEXT_PX, VIEW_PX + GAP :] = ctx
    _label(img, f"zoom 4x ({PATCH_PX} px patch)", (6, 18))
    _label(img, f"context ({CONTEXT_PX} px, 1x)", (VIEW_PX + GAP + 6, 18))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp.png")
    cv2.imwrite(str(tmp), img)
    tmp.replace(out_path)
    return px0, py0


def contact_sheet(
    items: Sequence[tuple[Path, str, tuple[float, float] | None, tuple[int, int]]],
    out_path: Path,
    *,
    cols: int = 4,
    tile: int = 400,
) -> None:
    """Mosaic of zoom panels, caption on top, the human mark as a green ring.

    ``items``: (image path, caption, human mark in native px or None, patch origin).
    """
    tiles = []
    for path, caption, gt, (x0, y0) in items:
        img = cv2.imread(str(path))[HEADER:, :VIEW_PX].copy()
        if gt is not None:
            p = (int((gt[0] - x0) * SCALE), int((gt[1] - y0) * SCALE))
            cv2.circle(img, p, 34, (0, 255, 0), 3, cv2.LINE_AA)
        t = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        bar = np.full((40, tile, 3), 24, np.uint8)
        cv2.putText(
            bar, caption[:56], (6, 26), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        tiles.append(np.vstack([bar, t]))
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
