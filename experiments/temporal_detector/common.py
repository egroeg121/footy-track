"""Shared constants / IO for the temporal-detector ablation (stdlib + numpy + cv2)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

BALL_CLS = 2  # in_play_ball (class 0 'ball' has zero instances)
NAMES = {0: "ball", 1: "coach", 2: "in_play_ball", 3: "person", 4: "player", 5: "player_sub", 6: "referee"}
SPLITS_TRAIN_VAL = ("train", "val")
EVAL_SPLITS = ("val", "test", "heldout")
IMG_W, IMG_H = 1920, 1080


def read_tiff5(path: str | Path) -> np.ndarray:
    """(H, W, 5) uint8 in stored order [gray(t-k), B, G, R, gray(t+k)]."""
    ok, pages = cv2.imdecodemulti(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if not ok or len(pages) != 5:
        raise RuntimeError(f"{path}: expected 5 pages, got {len(pages) if ok else 'decode failure'}")
    return np.stack(pages, axis=2)


def bgr_of(img5: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img5[..., 1:4])


def read_labels(path: Path, cls: int | None = None) -> np.ndarray:
    """YOLO label file -> (n, 5) [cls, cx, cy, w, h] normalised."""
    if not path.exists() or not path.read_text().strip():
        return np.zeros((0, 5), np.float32)
    a = np.loadtxt(str(path), ndmin=2, dtype=np.float64)
    return a if cls is None else a[a[:, 0] == cls]


def load_meta(root: Path) -> dict[str, dict]:
    out = {}
    with open(root / "meta.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out[(r["split"], r["id"])] = r
    return out


def write_data_yaml(root: Path, dst: Path, channels: int, images_subdir: str = "images", extra_train: list[str] | None = None) -> Path:
    """Local-path data.yaml (the dataset's own yaml names the mini's absolute path).

    ``extra_train``: absolute image directories added to the train list (round 2: pseudo-labelled images)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    names = ", ".join(f'"{k}": {v}' for k, v in NAMES.items())
    train = f"{images_subdir}/train" if not extra_train else json.dumps([f"{images_subdir}/train", *extra_train])
    dst.write_text(
        f'path: "{root}"\nnames: {{{names}}}\nnc: 7\nchannels: {channels}\n'
        f"train: {train}\nval: {images_subdir}/val\ntest: {images_subdir}/test\n"
        f"heldout: {images_subdir}/heldout\n"
    )
    return dst
