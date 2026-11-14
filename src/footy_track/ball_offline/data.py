# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408
"""Loaders for the offline ball-tracking harness.

Joins, per canonical clip (``footy_track.clip_id``):

* machine candidates: ``machine_labels/<date-stamped stem>.jsonl`` ball-class rows
  (``in_play_ball``) with real detector confidences, plus player rows used only
  to estimate global camera translation;
* human ground truth: every ``ball_gt_marks/*.jsonl`` alias of the clip merged
  into one frame -> point map (machine-only marks are dropped, explicit
  ``no_ball`` frames kept, ``not_broadcast`` frames ignored).

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

import numpy as np

from footy_track.clip_id import canonical_clip_id

W_PX = 1920.0
H_PX = 1080.0
BALL_CLASSES = ("in_play_ball",)
PLAYER_CLASSES = ("player",)

#: GT provenance tiers
TIER_HUMAN = "human"  # ``labeller`` tag, human-placed
TIER_VITTRACK = "vittrack_checked"  # propagated then reviewed (looser geometry)


def data_root() -> Path:
    return Path(os.environ.get("FOOTY_DATA", "/mnt/storage/footy_data"))


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
    cam_shift: np.ndarray  # (n_frames, 2) estimated global translation t-1 -> t
    cam_cut: np.ndarray  # (n_frames,) bool, True if frame t looks like a cut
    players: list[np.ndarray] = dataclasses.field(
        default_factory=list
    )  # per frame (n, 4) cx, cy, w, h px
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


def estimate_camera_shift(
    players: list[np.ndarray], bin_px: float = 4.0, max_px: float = 120.0
) -> tuple[np.ndarray, np.ndarray]:
    """Global translation between consecutive frames from player-box centres.

    For every pair of consecutive frames take all pairwise displacement vectors
    between player centres (|d| < ``max_px``), find the densest 2-D bin (the
    camera pan moves every player by the same vector, individual running adds
    scatter) and refine with the mean of the vectors near that mode.  Support
    is the fraction of players voting for the mode; very low support with many
    players on both frames flags a probable shot cut.

    Returns ``(shift[t], cut[t])`` where ``shift[t]`` is the translation of
    static scene content from frame t-1 to t (so ``p - cumsum(shift)`` is a
    camera-stabilised coordinate).
    """
    n = len(players)
    shift = np.zeros((n, 2))
    cut = np.zeros(n, dtype=bool)
    for t in range(1, n):
        a, b = players[t - 1][:, :2], players[t][:, :2]
        if len(a) < 4 or len(b) < 4:
            continue
        d = (b[None, :, :] - a[:, None, :]).reshape(-1, 2)
        d = d[np.hypot(d[:, 0], d[:, 1]) < max_px]
        if len(d) < 4:
            continue
        nb = int(2 * max_px / bin_px) + 2
        ij = np.floor((d + max_px) / bin_px).astype(int)
        hist = np.bincount(ij[:, 0] * nb + ij[:, 1], minlength=nb * nb).reshape(nb, nb)
        pad = np.pad(hist, 1)
        smooth = sum(
            pad[1 + dx : 1 + dx + nb, 1 + dy : 1 + dy + nb]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        )
        kx, ky = np.unravel_index(np.argmax(smooth), smooth.shape)
        best_n = int(smooth[kx, ky])
        best = (kx - max_px / bin_px, ky - max_px / bin_px)
        centre = (np.array(best) + 0.5) * bin_px
        sel = np.hypot(*(d - centre).T) < 1.5 * bin_px
        if not sel.any():
            continue
        shift[t] = d[sel].mean(axis=0)
        support = best_n / max(1, min(len(a), len(b)))
        if support < 0.25 and min(len(a), len(b)) >= 6:
            cut[t] = True
            shift[t] = 0.0
    return shift, cut


def _load_machine(path: Path, n_frames: int):
    per_cand: list[tuple[int, float, float, float, float, float]] = []
    players: list[list[tuple[float, float, float, float]]] = [
        [] for _ in range(n_frames)
    ]
    for r in _read_jsonl(path):
        tags = r["tags"]
        t = r["frame_index"]
        if t >= n_frames:
            continue
        b = r["bbox"]
        cx = (b["x"] + b["w"] / 2) * W_PX
        cy = (b["y"] + b["h"] / 2) * H_PX
        if any(c in tags for c in BALL_CLASSES):
            per_cand.append((t, cx, cy, r["confidence"], b["w"] * W_PX, b["h"] * H_PX))
        elif any(c in tags for c in PLAYER_CLASSES):
            players[t].append((cx, cy, b["w"] * W_PX, b["h"] * H_PX))
    per_cand.sort(key=lambda x: (x[0], -x[3]))
    arr = np.array(per_cand, dtype=float).reshape(-1, 6)
    ptr = np.zeros(n_frames + 1, dtype=np.int64)
    if len(arr):
        counts = np.bincount(arr[:, 0].astype(int), minlength=n_frames)
        ptr[1:] = np.cumsum(counts)
    return (
        ptr,
        arr[:, 1:3].copy(),
        arr[:, 3].copy(),
        arr[:, 4:6].copy(),
        [np.array(p).reshape(-1, 4) for p in players],
    )


def load_gt(root: Path, clip_id: str):
    """Merge all GT alias files of ``clip_id``.

    Returns ``(gt, no_ball, aliases, conflicts)``.  Human-placed marks win over
    vittrack+human_checked marks on the same frame.  Marks that are neither
    (machine-only, e.g. ``yolo``) are ignored; ``out_of_play_ball`` marks and
    ``not_broadcast`` frames are unscored.
    """
    gt: dict[int, GTFrame] = {}
    no_ball: set[int] = set()
    aliases: list[str] = []
    conflicts = 0
    for p in sorted((root / "ball_gt_marks").glob("*.jsonl")):
        if canonical_clip_id(p.name) != clip_id:
            continue
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


def machine_stem_map(root: Path) -> dict[str, Path]:
    """canonical clip id -> machine-label file."""
    return {
        canonical_clip_id(p.name): p
        for p in sorted((root / "machine_labels").glob("*.jsonl"))
    }


def load_clip(
    clip_id: str, root: Path | None = None, *, with_camera: bool = True
) -> ClipData:
    root = root or data_root()
    stems = machine_stem_map(root)
    if clip_id not in stems:
        raise KeyError(f"no machine labels for {clip_id}")
    manifest = json.loads((root / "machine_labels" / "_manifest.json").read_text())
    n_frames = manifest["clips"][stems[clip_id].stem]["frames"]
    ptr, xy, conf, wh, players = _load_machine(stems[clip_id], n_frames)
    if with_camera:
        shift, cut = estimate_camera_shift(players)
    else:
        shift, cut = np.zeros((n_frames, 2)), np.zeros(n_frames, dtype=bool)
    gt, no_ball, aliases, conflicts = load_gt(root, clip_id)
    return ClipData(
        clip_id=clip_id,
        n_frames=n_frames,
        ptr=ptr,
        xy=xy,
        conf=conf,
        wh=wh,
        gt={t: g for t, g in gt.items() if t < n_frames},
        no_ball={t for t in no_ball if t < n_frames},
        cam_shift=shift,
        cam_cut=cut,
        players=players,
        gt_aliases=aliases,
        gt_conflicts=conflicts,
    )


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
