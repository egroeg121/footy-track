# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408
"""Loaders for the offline ball-tracking harness.

Joins, per canonical clip (``footy_track.clip_id``):

* machine candidates: ball-class rows (``in_play_ball``) with real detector
  confidences from an :class:`~footy_track.ball_offline.sources.EvalSetSpec`
  (``machine_labels/`` for Man City, ``ball_candidates_norwich/`` for Norwich);
* human ground truth, through a pluggable :class:`GTSource` (a labeller marks
  directory such as ``ball_gt_marks/`` or ``ball_gt_marks_norwich/``, or a frozen
  snapshot): every ``*.jsonl`` alias of the clip merged into one frame -> point
  map (machine-only marks are dropped, explicit ``no_ball`` frames kept,
  ``not_broadcast`` frames ignored).

Everything is in pixel coordinates at 1920x1080 (the machine labels are
normalised).  The GT frame index space equals the machine-label frame index
space (T-03, ``clip_id`` docstring).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from footy_track.ball_offline.sources import data_root
from footy_track.clip_id import canonical_clip_id

if TYPE_CHECKING:
    from footy_track.ball_offline.sources import EvalSetSpec

W_PX = 1920.0
H_PX = 1080.0
BALL_CLASSES = ("in_play_ball",)
PLAYER_CLASSES = ("player",)

#: GT provenance tiers
TIER_HUMAN = "human"  # ``labeller`` tag, human-placed
TIER_VITTRACK = "vittrack_checked"  # propagated then reviewed (looser geometry)


@dataclasses.dataclass
class GTFrame:
    xy: tuple[float, float]  # pixel centre
    tier: str


@dataclasses.dataclass
class ClipData:
    """Everything the harness needs for one clip."""

    clip_id: str
    n_frames: int
    # candidates, CSR layout: frame t owns rows ``ptr[t]:ptr[t+1]``
    ptr: np.ndarray  # (n_frames + 1,)
    xy: np.ndarray  # (n_cand, 2) pixels
    conf: np.ndarray  # (n_cand,)
    wh: np.ndarray  # (n_cand, 2) pixels
    gt: dict[int, GTFrame]  # frames with a human ball position
    no_ball: set[int]  # frames explicitly marked no_ball
    gt_aliases: list[str] = dataclasses.field(default_factory=list)
    gt_conflicts: int = 0  # alias disagreements > 10 px on the same frame

    def cands(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.ptr[t], self.ptr[t + 1]
        return self.xy[s:e], self.conf[s:e]


def _read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_machine(cand_path: Path, n_frames: int):
    per_cand: list[tuple[int, float, float, float, float, float]] = []
    for r in _read_jsonl(cand_path):
        t = r["frame_index"]
        if t >= n_frames or not any(c in r["tags"] for c in BALL_CLASSES):
            continue
        b = r["bbox"]
        cx = (b["x"] + b["w"] / 2) * W_PX
        cy = (b["y"] + b["h"] / 2) * H_PX
        per_cand.append((t, cx, cy, r["confidence"], b["w"] * W_PX, b["h"] * H_PX))
    per_cand.sort(key=lambda x: (x[0], -x[3]))
    arr = np.array(per_cand, dtype=float).reshape(-1, 6)
    ptr = np.zeros(n_frames + 1, dtype=np.int64)
    if len(arr):
        counts = np.bincount(arr[:, 0].astype(int), minlength=n_frames)
        ptr[1:] = np.cumsum(counts)
    return ptr, arr[:, 1:3].copy(), arr[:, 3].copy(), arr[:, 4:6].copy()


GTLoad = tuple[dict[int, GTFrame], set[int], list[str], int]


class GTSource(Protocol):
    """Where ground truth comes from.  ``clip_ids`` lists canonical clip ids that
    have any GT; ``load`` returns ``(gt, no_ball, aliases, conflicts)``."""

    name: str

    def clip_ids(self) -> list[str]: ...

    def load(self, clip_id: str) -> GTLoad: ...


class MarksDirGT:
    """Labeller GT sidecars: ``<dir>/<stem>.jsonl`` (what ``FOOTY_GT_MARKS_DIR``
    points at).  Human-placed marks win over vittrack+human_checked marks on the
    same frame; machine-only marks (e.g. ``yolo``) are ignored; ``out_of_play_ball``
    marks and ``not_broadcast`` frames are unscored; every alias file of a clip
    (see :func:`footy_track.clip_id.canonical_clip_id`) is merged."""

    def __init__(self, directory: Path | str, name: str | None = None):
        self.dir = Path(directory)
        self.name = name or self.dir.name

    def clip_ids(self) -> list[str]:
        return sorted({canonical_clip_id(p.name) for p in self.dir.glob("*.jsonl")})

    def source_files(self, clip_id: str) -> list[Path]:
        return [
            p
            for p in sorted(self.dir.glob("*.jsonl"))
            if canonical_clip_id(p.name) == clip_id
        ]

    def load(self, clip_id: str) -> GTLoad:
        gt: dict[int, GTFrame] = {}
        no_ball: set[int] = set()
        aliases: list[str] = []
        conflicts = 0
        for p in self.source_files(clip_id):
            aliases.append(p.stem)
            for r in _read_jsonl(p):
                tags = r["tags"]
                t = r["frame_index"]
                if "no_ball" in tags:
                    no_ball.add(t)
                    continue
                if "in_play_ball" not in tags:
                    continue
                if "labeller" in tags:
                    tier = TIER_HUMAN
                elif "vittrack" in tags and "human_checked" in tags:
                    tier = TIER_VITTRACK
                else:
                    continue  # machine-only provenance: not ground truth
                c = r.get("center")
                if isinstance(c, dict):
                    cx, cy = c["x"], c["y"]
                elif c is not None:
                    cx, cy = c
                else:
                    b = r["bbox"]
                    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
                new = GTFrame((cx * W_PX, cy * H_PX), tier)
                old = gt.get(t)
                if old is not None:
                    if np.hypot(old.xy[0] - new.xy[0], old.xy[1] - new.xy[1]) > 10:
                        conflicts += 1
                    if old.tier == TIER_HUMAN and new.tier != TIER_HUMAN:
                        continue
                gt[t] = new
        # a frame both marked ball and no_ball in different aliases: ball wins
        no_ball -= set(gt)
        return gt, no_ball, aliases, conflicts


def load_gt(gt_dir: Path | str, clip_id: str) -> GTLoad:
    """Merge all GT alias files of ``clip_id`` found in the marks directory ``gt_dir``."""
    return MarksDirGT(gt_dir).load(clip_id)


def machine_stem_map(candidates_dir: Path | str) -> dict[str, Path]:
    """canonical clip id -> candidate file."""
    return {
        canonical_clip_id(p.name): p
        for p in sorted(Path(candidates_dir).glob("*.jsonl"))
    }


def _manifest_frames(directory: Path | None, stem: str) -> int | None:
    if directory is None:
        return None
    m = directory / "_manifest.json"
    if not m.is_file():
        return None
    entry = json.loads(m.read_text()).get("clips", {}).get(stem)
    return None if entry is None else int(entry["frames"])


def _max_frame(path: Path) -> int:
    top = -1
    for r in _read_jsonl(path):
        top = max(top, r["frame_index"])
    return top + 1


def load_clip(clip_id: str, spec: EvalSetSpec, *, gt: GTLoad | None = None) -> ClipData:
    """Candidates of one clip.  ``gt`` overrides the spec's live GT directory
    (pass a snapshot's GT to score against frozen labels)."""
    stems = machine_stem_map(spec.candidates_dir)
    if clip_id not in stems:
        raise KeyError(f"no candidates for {clip_id} in {spec.candidates_dir}")
    cand_path = stems[clip_id]
    n_frames = frame_count(spec, cand_path)
    ptr, xy, conf, wh = _load_machine(cand_path, n_frames)
    gt_map, no_ball, aliases, conflicts = (
        gt if gt is not None else MarksDirGT(spec.gt_dir).load(clip_id)
    )
    return ClipData(
        clip_id=clip_id,
        n_frames=n_frames,
        ptr=ptr,
        xy=xy,
        conf=conf,
        wh=wh,
        gt={t: g for t, g in gt_map.items() if t < n_frames},
        no_ball={t for t in no_ball if t < n_frames},
        gt_aliases=aliases,
        gt_conflicts=conflicts,
    )


def frame_count(spec: EvalSetSpec, cand_path: Path) -> int:
    """Frames of a clip: the detector run manifest if there is one (``manifest_dir``
    or the candidate dir), else the highest candidate frame index + 1."""
    n = _manifest_frames(spec.candidates_dir, cand_path.stem)
    if n is None:
        n = _manifest_frames(spec.manifest_dir, cand_path.stem)
    return n if n is not None else _max_frame(cand_path)


# --------------------------------------------------------------------------
# detector-training contamination
# --------------------------------------------------------------------------

_RF_NAME = re.compile(r"^(?P<stem>.*)_(?P<frame>\d{6})_(?:png|jpg)\.rf\.")


def roboflow_frames(root: Path | None = None, versions=("11", "3")):
    """``{(version, split): {canonical clip: set(frame)}}`` of the exported
    Roboflow datasets.  Image names encode ``<clip stem>_<frame>_<ext>.rf.<hash>``."""
    root = root or data_root()
    out: dict[tuple[str, str], dict[str, set[int]]] = {}
    for ver in versions:
        for split in ("train", "valid", "test"):
            d = (
                root
                / "detection_dataset"
                / f"roboflow_dataset_{ver}"
                / split
                / "images"
            )
            clips: dict[str, set[int]] = defaultdict(set)
            if d.is_dir():
                for f in os.listdir(d):
                    m = _RF_NAME.match(f)
                    if m:
                        clips[canonical_clip_id(m["stem"])].add(int(m["frame"]))
            out[(ver, split)] = dict(clips)
    return out


def contaminated_frames(
    clip_id: str, rf: dict, *, near: int = 5, splits=("train", "valid")
) -> set[int]:
    """Frames of ``clip_id`` within ``near`` frames of a detector-training image.

    ``valid`` is included because ``best.pt`` is selected on it.  ``test`` is
    not training data and is treated as clean."""
    out: set[int] = set()
    for (_, split), clips in rf.items():
        if split in splits:
            for f in clips.get(clip_id, ()):
                out.update(range(f - near, f + near + 1))
    return out
