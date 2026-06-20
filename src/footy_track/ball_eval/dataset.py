"""Eval dataset schema and loader.

Ground-truth data is stored as a directory containing:
  - One or more video files (mp4/avi/etc.)
  - A sidecar ``<clip_name>.jsonl`` per clip, one JSON object per frame:
      {"frame_index": 0, "bbox": [x, y, w, h], "tags": ["occlusion"]}
    or {"frame_index": 1, "bbox": null, "tags": ["ball_not_visible"]} if ball
    is absent / out of frame.

BBox: normalised (x, y, w, h) top-left, matching footy-track convention.
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
from typing import NamedTuple

# Normalised (x, y, w, h) top-left. Using a plain tuple for zero-overhead.
BBox = tuple[float, float, float, float]

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


class FrameLabel(NamedTuple):
    """Ground-truth label for one frame."""

    frame_index: int
    bbox: BBox | None  # None → ball absent / not visible
    tags: tuple[str, ...]  # e.g. ("occlusion",)

    @classmethod
    def from_dict(cls, d: dict) -> FrameLabel:
        raw_bbox = d.get("bbox")
        bbox: BBox | None = tuple(raw_bbox) if raw_bbox is not None else None  # type: ignore[arg-type]
        return cls(
            frame_index=int(d["frame_index"]),
            bbox=bbox,
            tags=tuple(d.get("tags", [])),
        )

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "tags": list(self.tags),
        }


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
            import cv2  # noqa: PLC0415

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

    def iter_frames(self):
        """Yield (frame_index, rgb_frame) pairs for every frame in the clip."""
        import cv2  # noqa: PLC0415

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
        return len(self.labels)

    def ball_present_count(self) -> int:
        return sum(1 for lbl in self.labels.values() if lbl.bbox is not None)

    def frames_with_tag(self, tag: str) -> list[int]:
        return [lbl.frame_index for lbl in self.labels.values() if tag in lbl.tags]


class EvalDataset:
    """Collection of labelled eval clips.

    Load from a directory::

        dataset = EvalDataset.from_dir("eval_data/clips/")
    """

    def __init__(self, clips: list[EvalClip]) -> None:
        self.clips = clips

    @classmethod
    def from_dir(cls, directory: str | pathlib.Path) -> EvalDataset:
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
