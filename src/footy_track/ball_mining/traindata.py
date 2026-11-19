"""The training miner's own labeller dataset (``ball_train_v1``): where its Ball Check cards load from and where the verdicts land.

A round's cards name clips that live in no existing labeller dataset: pool cards point at segment mp4s of train matches
(``<root>/<match>/split_video_broadcast_frames/<clip>.mp4``), gold cards at GT clips of the Man City / Bournemouth /
Aston Villa datasets. Ball Check finds a card's video by *exact stem in the dataset's clips dir*, so the round gets a dataset
of its own, laid out like the ball evaluation panel's (``ball_eval.panel.link_clips`` / ``register_dataset``)::

    <root>/ball_train/clips/<clip>.mp4          symlinks to the source videos (this module)
    <root>/ball_gt_marks_ball_train_v1/         verdicts (``ball_checks/<clip>.jsonl``), state, round logs

Unlike the panel's, the dataset keeps ``ingest`` on: the verdicts are training data, and ``feature_store.ingest_all`` reads
every registered marks dir that does not say ``"ingest": false`` (sidecars, ``ball_checks/`` via ``import_ball_checks``).

Clip names. A pool card keeps its pool clip name (``<match>_segNNN``: unique per match, and the file name it already has).
A gold card is renamed to ``clip_id.canonical_clip_id(clip)``, so the several aliases of one physical GT clip (detector stems
``arsenal_mancity_20250925_segNNN``, sidecar stems ``mancity_segNNN`` / ``arsenal_mancity_segNNN``) all verdict into one
``ball_checks/<canonical>.jsonl``; the original sidecar stem rides along as the hidden card key ``source_clip``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import datasets

LOGGER = logging.getLogger(__name__)

TRAIN_DATASET = "ball_train_v1"
TRAIN_DIRNAME = "ball_train"
SEGMENTS_DIR = "split_video_broadcast_frames"
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
GOLD_BUCKET = "G"


class ClipLinkError(RuntimeError):
    """A card's clip cannot be linked into the dataset's clips dir (message says which and why)."""


# ---------------------------------------------------------------------------
# Paths and registration
# ---------------------------------------------------------------------------


def clips_dir(root: str | Path, name: str = TRAIN_DATASET) -> Path:
    return Path(root) / TRAIN_DIRNAME / "clips"


def marks_dir(root: str | Path, name: str = TRAIN_DATASET) -> Path:
    return Path(root) / f"ball_gt_marks_{name}"


def register_dataset(root: str | Path, name: str = TRAIN_DATASET) -> bool:
    """Add the training dataset to ``labeller_datasets.json`` if missing (``datasets.register``: locked, atomic, idempotent).
    Creates the clips and marks dirs. Returns True when the registry changed."""
    cdir, mdir = clips_dir(root, name), marks_dir(root, name)
    cdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    return datasets.register(
        name,
        {
            "label": f"Ball training miner {name} (Ball Check rounds; verdicts are training data)",
            "clips_dir": str(cdir),
            "marks_dir": str(mdir),
        },
    )


# ---------------------------------------------------------------------------
# Finding source videos
# ---------------------------------------------------------------------------


class ClipResolver:
    """Find the video of a clip stem in an ordered list of directories: an exact stem first, else any file whose canonical clip
    id matches (``mancity_seg010`` -> ``arsenal_mancity_seg010.mp4``). Directory listings are read once."""

    def __init__(self, dirs: Iterable[str | Path]):
        self.dirs = [Path(d) for d in dirs]
        self._index: dict[Path, tuple[dict[str, Path], dict[str, Path]]] = {}

    def _listing(self, d: Path) -> tuple[dict[str, Path], dict[str, Path]]:
        if d not in self._index:
            exact: dict[str, Path] = {}
            canon: dict[str, Path] = {}
            try:
                files = sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS)
            except OSError:
                files = []
            for p in files:
                if p.exists():  # a dangling link is not a video
                    exact.setdefault(p.stem, p)
                    canon.setdefault(canonical_clip_id(p.stem), p)
            self._index[d] = (exact, canon)
        return self._index[d]

    def find(self, clip: str) -> Path | None:
        for d in self.dirs:
            exact, _ = self._listing(d)
            if clip in exact:
                return exact[clip]
        cid = canonical_clip_id(clip)
        for d in self.dirs:
            _, canon = self._listing(d)
            if cid in canon:
                return canon[cid]
        return None


def pool_video(root: str | Path, match: str, clip: str) -> Path | None:
    """Segment mp4 of a pool clip: ``<root>/<match>/split_video_broadcast_frames/<clip>.mp4``, else any
    ``<root>/*/split_video_broadcast_frames*/<clip>.mp4``."""
    root = Path(root)
    for ext in VIDEO_EXTS:
        p = root / match / SEGMENTS_DIR / f"{clip}{ext}"
        if p.is_file():
            return p
    for ext in VIDEO_EXTS:
        for p in sorted(root.glob(f"*/{SEGMENTS_DIR}*/{clip}{ext}")):
            if p.is_file():
                return p
    return None


def gold_dataset_dirs(names: Iterable[str]) -> list[Path]:
    """Clips dir then extra clips dirs of each named registry dataset (unknown names skipped)."""
    out: list[Path] = []
    for n in names:
        try:
            ds = datasets.get(n)
        except datasets.UnknownDatasetError:
            continue
        out += [ds.clips_dir, *ds.extra_clips_dirs]
    return out


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------


def _link(src: Path, dst: Path) -> bool:
    """Symlink ``dst`` -> ``src``. True when created; False when it is already there (idempotent). Raises on a conflict."""
    if src == dst:
        return False  # the source already is the file in the dataset's clips dir
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return False
        if dst.exists() and os.path.realpath(dst) == os.path.realpath(src):
            return False  # a different path to the same file
        raise ClipLinkError(f"{dst} exists and is not a link to {src}")
    try:
        dst.symlink_to(src)
    except FileExistsError:  # lost a race with another round: fine if it points at the same video
        if os.path.realpath(dst) == os.path.realpath(src):
            return False
        raise ClipLinkError(f"{dst} was created concurrently and points elsewhere") from None
    return True


def link_cards(
    cards: Sequence[dict],
    dst_dir: str | Path,
    *,
    data_root: str | Path,
    resolver: ClipResolver | None = None,
) -> tuple[list[dict], dict]:
    """Link every card's clip into ``dst_dir`` and return ``(cards, report)``.

    Pool cards link ``<dst>/<clip>.mp4`` to their segment mp4; gold cards (bucket G) are renamed to the canonical clip id and
    link to the video found by ``resolver`` (the dataset their GT came from); the sidecar stem is kept as ``source_clip``.
    The input cards are not modified. Nothing is created for a card set that cannot be fully resolved (every clip is
    resolved first, then linked): ``ClipLinkError`` lists them all.
    """
    dst = Path(dst_dir)
    plan: dict[str, Path] = {}  # link name -> source
    out: list[dict] = []
    missing: list[str] = []
    n_gold = 0
    for c in cards:
        gold = (c.get("sampling") or {}).get("bucket") == GOLD_BUCKET
        clip = c["clip"]
        if gold:
            name = canonical_clip_id(clip)
            # Aliases of one physical clip (``mancity_seg010`` and ``arsenal_mancity_seg010`` are two kf10 files of it) share one
            # link: reuse what this call already chose, else what an earlier round linked, else look it up.
            src = (
                next((p for n, p in plan.items() if Path(n).stem == name), None)
                or next((dst / f"{name}{e}" for e in VIDEO_EXTS if (dst / f"{name}{e}").is_file()), None)
                or (resolver.find(clip) if resolver else None)
                or pool_video(data_root, c.get("match", ""), clip)
            )
        else:
            src = pool_video(data_root, c.get("match", ""), clip) or (resolver.find(clip) if resolver else None)
            name = clip
        if src is None:
            missing.append(f"{clip} ({'gold' if gold else c.get('match')})")
            continue
        name = name + src.suffix.lower()
        if os.path.realpath(plan.setdefault(name, src)) != os.path.realpath(src):
            raise ClipLinkError(f"two different videos for one clip name {name}: {plan[name]} and {src}")
        card = dict(c)
        if gold:
            n_gold += 1
            card["source_clip"] = clip
            card["clip"] = Path(name).stem
        out.append(card)
    if missing:
        raise ClipLinkError(f"no video for {len(missing)} card clip(s): {sorted(set(missing))[:10]}")
    dst.mkdir(parents=True, exist_ok=True)
    created = 0
    for name, src in sorted(plan.items()):
        created += _link(src, dst / name)
        if not (dst / name).is_file():
            raise ClipLinkError(f"{dst / name} does not resolve to a file (link to {src})")
    return out, {"clips": len(plan), "linked_new": created, "already_linked": len(plan) - created, "gold_renamed": n_gold,
                 "clips_dir": str(dst)}  # fmt: skip
