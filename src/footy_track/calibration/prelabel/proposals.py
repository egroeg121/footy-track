"""Proposal sidecars: ``<gt dir>/calibration_proposals/<canonical clip>.json``.

Format is the shared contract (see ``docs/design/calibration.md`` §6): one file
per clip, ``{"clip", "runs": {run_id: {source, created, params, frames}}}``,
each frame a GT-shaped record plus ``status``/``score``/``failure``/``view``.
Writes are atomic (tmp + rename) and keep an existing file's mode (0664 new).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from footy_track.clip_id import canonical_clip_id

_NEW_FILE_MODE = 0o664
STATUSES = ("proposed", "accepted", "edited", "rejected", "failed", "not_broadcast")
_lock = threading.Lock()


def proposals_dir(gt_dir: Path) -> Path:
    return Path(gt_dir) / "calibration_proposals"


def proposals_path(gt_dir: Path, clip: str) -> Path:
    return proposals_dir(gt_dir) / f"{canonical_clip_id(clip)}.json"


def read_proposals(gt_dir: Path, clip: str) -> dict:
    p = proposals_path(gt_dir, clip)
    if not p.exists():
        return {"clip": canonical_clip_id(clip), "runs": {}}
    return json.loads(p.read_text())


def write_proposals(gt_dir: Path, clip: str, doc: dict) -> None:
    path = proposals_path(gt_dir, clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = _NEW_FILE_MODE
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def update_proposals(gt_dir: Path, clip: str, fn) -> dict:
    """Read-modify-write under a process lock; ``fn(doc)`` mutates ``doc``."""
    with _lock:
        doc = read_proposals(gt_dir, clip)
        fn(doc)
        write_proposals(gt_dir, clip, doc)
        return doc


def list_clips_with_proposals(gt_dir: Path) -> list[str]:
    d = proposals_dir(gt_dir)
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []
