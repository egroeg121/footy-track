"""Roboflow v11 image name -> (source video, frame index) for multi-frame input.

Two filename families exist in ``roboflow_dataset_11`` (verified by pixel
comparison against the decoded video frame, see docs/experiments/multiframe_dataset.md):

``<clip_stem>_<NNNNNN>_(png|jpg).rf.<hash>.jpg``
    ``clip_stem`` is a GT/alias stem (``astonvilla_seg080``, ``arsenal_mancity_seg010``,
    ``arsenal_example`` ...). ``NNNNNN`` is the **frame index inside that clip**
    (mean abs diff to decoded frame ~0.35 grey levels = JPEG noise; off-by-one is
    ~13). The clip file is found through ``ballcheck_clips/<stem>.mp4``.

``arsenal_mancity_20250925_<NNNNNN>_png.rf.<hash>.jpg``
    ``NNNNNN`` is the ordinal of ``arsenal_mancity/full_video_frames/*.png``, a
    1 fps extraction of the 25 fps source webm, i.e. absolute source frame
    ``25 * NNNNNN + 12`` (ffmpeg ``fps=1`` takes the frame at ``s + 0.5`` s).
    OpenCV cannot decode the webm (every seek returns the same frame), so the
    frame is located inside the broadcast segment clip that contains it, using
    ``clips/segments.json`` (``start_frame``); frames in non-broadcast gaps have no
    clip and cannot be mapped.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2

from footy_track.clip_id import canonical_clip_id

FOOTY_DATA = Path(os.environ.get("FOOTY_DATA", "/mnt/storage/footy_data"))
FULL_MATCH_PREFIX = "arsenal_mancity_20250925"
#: absolute source frame = FPS * ordinal + FULL_VIDEO_OFFSET (fps=1 extraction).
FULL_VIDEO_FPS = 25
FULL_VIDEO_OFFSET = 12

_NAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)_(?:png|jpg|jpeg)\.rf\.[0-9a-f]+\.(?:jpg|txt)$")


@dataclass(frozen=True)
class SourceRef:
    video: Path | None
    frame: int | None
    n_frames: int | None
    clip_id: str  # canonical clip id ("" if unmappable)
    reason: str = ""  # why unmapped, empty if mapped

    @property
    def mapped(self) -> bool:
        return self.video is not None and self.frame is not None

    def has_neighbours(self, k: int) -> bool:
        return (
            self.mapped
            and self.frame - k >= 0
            and self.n_frames is not None
            and self.frame + k < self.n_frames
        )


def parse_name(name: str) -> tuple[str, int]:
    m = _NAME_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"not a Roboflow v11 file name: {name}")
    return m.group("prefix"), int(m.group("idx"))


@lru_cache(maxsize=None)
def frame_count(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


@lru_cache(maxsize=None)
def _segments() -> list[dict]:
    return json.loads((FOOTY_DATA / "clips" / "segments.json").read_text())["segments"]


def stem_video(stem: str, root: Path = FOOTY_DATA) -> Path | None:
    """Video file for a clip stem (any alias). ``None`` if no alias resolves."""
    if stem == "arsenal_example":
        cand = [root / "arsenal_mancity_example_video.mp4"]
    else:
        cand = [
            root / "ballcheck_clips" / f"{stem}.mp4",
            root / "clips" / f"{stem}.mp4",
            root / "eval_data" / "clips" / f"{stem}.mp4",
        ]
    for p in cand:
        if p.exists():  # follows symlinks; a broken link is "not found"
            return p
    return None


def resolve(prefix: str, idx: int, root: Path = FOOTY_DATA) -> SourceRef:
    if prefix == FULL_MATCH_PREFIX:
        absolute = FULL_VIDEO_FPS * idx + FULL_VIDEO_OFFSET
        for seg in _segments():
            if seg["start_frame"] <= absolute < seg["start_frame"] + seg["frame_count"]:
                video = root / "clips" / seg["file"]
                if not video.exists():
                    return SourceRef(None, None, None, "", "segment clip missing")
                return SourceRef(
                    video,
                    absolute - seg["start_frame"],
                    frame_count(video),
                    canonical_clip_id(video.stem),
                )
        return SourceRef(None, None, None, "", "frame outside every broadcast segment")
    video = stem_video(prefix, root)
    if video is None:
        return SourceRef(None, None, None, canonical_clip_id(prefix), "no video for clip stem")
    return SourceRef(video, idx, frame_count(video), canonical_clip_id(prefix))


def resolve_name(name: str, root: Path = FOOTY_DATA) -> SourceRef:
    prefix, idx = parse_name(name)
    return resolve(prefix, idx, root)
