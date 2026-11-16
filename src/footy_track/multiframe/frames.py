"""Frame access, label/mask helpers shared by the multi-frame scripts."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from footy_track.multiframe.mapping import FOOTY_DATA

#: Roboflow v11 class ids (data.yaml order).
V11_NAMES = ["ball", "coach", "in_play_ball", "person", "player", "player_sub", "referee"]
BALL_ID = 2
PERSON_IDS = {1, 3, 4, 5, 6}


class ClipReader:
    """Frame-accurate reader for one video with a small LRU cache.

    Sequential reads reuse the decoder; any other access seeks (``CAP_PROP_POS_FRAMES``
    is frame accurate on these mp4s, verified against exact pixel matches).
    """

    def __init__(self, path: Path, cache: int = 24):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(path))
        self.pos = 0
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.max = cache

    def get(self, i: int) -> np.ndarray | None:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        if i != self.pos:
            if 0 < i - self.pos <= 30:
                for _ in range(i - self.pos):
                    self.cap.grab()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i
        ok, fr = self.cap.read()
        if not ok:
            return None
        self.pos = i + 1
        self.cache[i] = fr
        if len(self.cache) > self.max:
            self.cache.popitem(last=False)
        return fr

    def close(self) -> None:
        self.cap.release()


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append((int(p[0]), *map(float, p[1:5])))
    return out


def person_boxes(labels) -> list[tuple[float, float, float, float]]:
    return [(x, y, w, h) for c, x, y, w, h in labels if c in PERSON_IDS]


_MACHINE_CACHE: dict[str, dict[int, list]] = {}
_PERSON_TAGS = {"player", "referee", "person", "coach", "player_sub"}


def machine_person_boxes(clip_stem: str, frame: int, min_conf: float = 0.3):
    """Person-like boxes (cx, cy, w, h normalised) from RT-DETR machine labels, or None.

    ``clip_stem`` is the dated clip name (``arsenal_mancity_20250925_segNNN``); only
    those 162 clips have machine labels. Boxes are tier-3 and used only as masks.
    """
    if clip_stem not in _MACHINE_CACHE:
        p = FOOTY_DATA / "machine_labels" / f"{clip_stem}.jsonl"
        d: dict[int, list] = {}
        if p.exists():
            with p.open() as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("confidence", 1) < min_conf or not (_PERSON_TAGS & set(r.get("tags", []))):
                        continue
                    c, b = r["center"], r["bbox"]
                    d.setdefault(r["frame_index"], []).append((c["x"], c["y"], b["w"], b["h"]))
        _MACHINE_CACHE[clip_stem] = d
    return _MACHINE_CACHE[clip_stem].get(frame)
