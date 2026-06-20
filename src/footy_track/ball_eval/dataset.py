"""Eval dataset schema and loader.

Ground-truth data is stored as a directory containing:
  - One or more video files (mp4/avi/etc.)
  - A sidecar ``<clip_name>.jsonl`` per clip, one JSON object per labelled frame:

    Full bbox (automated or careful manual annotation)::

        {"frame_index": 0, "bbox": [x, y, w, h], "tags": ["occlusion"]}

    Sparse human center-point (faster to label, primary for bake-off GT)::

        {"frame_index": 5, "center": [cx, cy], "tags": ["small_ball"]}

    Ball absent / not visible::

        {"frame_index": 1, "bbox": null, "tags": ["ball_not_visible"]}

Sparse center-point GT is the primary format for human-anchored ground truth.
A human only needs to click the ball center — no box drawing required.
Frames without a JSONL entry are unlabelled (the harness skips them for metrics).

BBox: normalised (x, y, w, h) top-left, matching footy-track convention.
Center: normalised (cx, cy) ball center.
Tags are free strings; standard values: "occlusion", "motion_blur",
"small_ball", "crowd_background", "ball_not_visible".

Example layout::

    eval_data/
        clips/
            hard_occlusion_01.mp4
            hard_occlusion_01.jsonl
            motion_blur_02.mp4
            motion_blur_02.jsonl
        README.md
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Generator
from typing import NamedTuple

import cv2
import numpy as np

# Normalised (x, y, w, h) top-left. Using a plain tuple for zero-overhead.
BBox = tuple[float, float, float, float]
# Normalised (cx, cy) ball center.
Center = tuple[float, float]

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


class FrameLabel(NamedTuple):
    """Ground-truth label for one frame.

    One of ``bbox`` or ``center`` will be set when the ball is visible.
    ``bbox`` gives a full box; ``center`` gives just the ball center (used for
    sparse human-anchored GT where drawing a box is impractical).
    ``center`` is always derivable from ``bbox`` via ``bbox_center()``.
    """

    frame_index: int
    bbox: BBox | None  # None → ball absent / not visible, or center-only label
    tags: tuple[str, ...]  # e.g. ("occlusion",)
    center: Center | None = None  # explicit center when bbox not available

    @classmethod
    def from_dict(cls, d: dict) -> FrameLabel:
        raw_bbox = d.get("bbox")
        bbox: BBox | None = tuple(raw_bbox) if raw_bbox is not None else None  # type: ignore[arg-type]
        raw_center = d.get("center")
        center: Center | None = tuple(raw_center) if raw_center is not None else None  # type: ignore[arg-type]
        return cls(
            frame_index=int(d["frame_index"]),
            bbox=bbox,
            tags=tuple(d.get("tags", [])),
            center=center,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "frame_index": self.frame_index,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "tags": list(self.tags),
        }
        if self.center is not None:
            d["center"] = list(self.center)
        return d

    def ball_center(self) -> Center | None:
        """Return the ball center, derived from bbox if no explicit center."""
        if self.center is not None:
            return self.center
        if self.bbox is not None:
            return (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2)
        return None

    def is_ball_visible(self) -> bool:
        """True when ball is present in this frame."""
        return self.bbox is not None or self.center is not None


class EvalClip:
    """One labelled video clip used for evaluation.

    Attributes:
        name: Short human-readable identifier (stem of the video file).
        video_path: Path to the video file.
        labels: Per-frame ground-truth labels, keyed by frame_index.
        total_frames: Number of frames in the clip.
    """

    def __init__(
        self,
        name: str,
        video_path: pathlib.Path,
        labels: dict[int, FrameLabel],
        total_frames: int,
    ) -> None:
        self.name = name
        self.video_path = video_path
        self.labels = labels
        self.total_frames = total_frames

    @classmethod
    def from_video_and_jsonl(
        cls,
        video_path: pathlib.Path,
        jsonl_path: pathlib.Path,
    ) -> EvalClip:
        labels: dict[int, FrameLabel] = {}
        with jsonl_path.open() as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                d = json.loads(stripped)
                lbl = FrameLabel.from_dict(d)
                labels[lbl.frame_index] = lbl

        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception:
            total_frames = max(labels.keys()) + 1 if labels else 0

        return cls(
            name=video_path.stem,
            video_path=video_path,
            labels=labels,
            total_frames=total_frames,
        )

    def iter_frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, rgb_frame) pairs for every frame in the clip."""
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            frame_idx = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield frame_idx, rgb
                frame_idx += 1
        finally:
            cap.release()

    def labelled_frame_count(self) -> int:
        """Number of frames with a ground-truth label (bbox or absent)."""
        return len(self.labels)

    def ball_present_count(self) -> int:
        """Number of labelled frames where the ball is visible."""
        return sum(1 for lbl in self.labels.values() if lbl.is_ball_visible())

    def frames_with_tag(self, tag: str) -> list[int]:
        """Return frame indices that carry the given tag."""
        return [lbl.frame_index for lbl in self.labels.values() if tag in lbl.tags]


class EvalDataset:
    """Collection of labelled eval clips.

    Load from a directory::

        dataset = EvalDataset.from_dir("eval_data/clips/")

    Or construct manually::

        dataset = EvalDataset([clip1, clip2])
    """

    def __init__(self, clips: list[EvalClip]) -> None:
        self.clips = clips

    @classmethod
    def from_dir(cls, directory: str | pathlib.Path) -> EvalDataset:
        """Load all labelled clips from *directory*.

        For each ``.jsonl`` sidecar found, the corresponding video file
        (same stem, any video suffix) must exist alongside it.
        """
        directory = pathlib.Path(directory)
        clips: list[EvalClip] = []

        for jsonl_path in sorted(directory.glob("*.jsonl")):
            video_path = _find_video(directory, jsonl_path.stem)
            if video_path is None:
                raise FileNotFoundError(
                    f"No video file found for ground-truth file {jsonl_path}. "
                    f"Expected one of: {[jsonl_path.stem + s for s in _VIDEO_SUFFIXES]}"
                )
            clips.append(EvalClip.from_video_and_jsonl(video_path, jsonl_path))

        if not clips:
            raise ValueError(f"No labelled clips found in {directory}")

        return cls(clips)

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self):
        return iter(self.clips)

    def summary(self) -> str:
        lines = [f"EvalDataset: {len(self.clips)} clips"]
        for clip in self.clips:
            lines.append(
                f"  {clip.name}: {clip.total_frames} frames, "
                f"{clip.ball_present_count()} labelled ball-present"
            )
        return "\n".join(lines)


def _find_video(directory: pathlib.Path, stem: str) -> pathlib.Path | None:
    for suffix in _VIDEO_SUFFIXES:
        candidate = directory / (stem + suffix)
        if candidate.exists():
            return candidate
    return None


def write_labels(labels: list[FrameLabel], path: str | pathlib.Path) -> None:
    """Write a list of FrameLabels to a JSONL file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for lbl in labels:
            f.write(json.dumps(lbl.to_dict()) + "\n")
