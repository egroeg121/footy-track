"""Frame-level "not a broadcast pitch view" flags (LAB-1222).

``<gt dir>/frame_type/<canonical clip>.json`` -> ``{"clip", "frames": {"<idx>": {"type":
"not_broadcast", "ts", "run_id"?, "note"?}}}``. A separate sidecar (not a section of
``calibration_gt``) so the GT file keeps meaning "frames with a calibration", every
existing reader of it (pre-label ``--skip-gt``, the feature-store GT import, GT list) is
untouched, and deleting the last GT frame never drops the flag. Writes are atomic (tmp +
rename) and keep an existing file's mode (0664 new), like the other calibration sidecars.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from footy_track.clip_id import canonical_clip_id

_NEW_FILE_MODE = 0o664
NOT_BROADCAST = "not_broadcast"
_lock = threading.Lock()


def frame_type_dir(gt_dir: Path) -> Path:
    return Path(gt_dir) / "frame_type"


def frame_type_path(gt_dir: Path, clip: str) -> Path:
    return frame_type_dir(gt_dir) / f"{canonical_clip_id(clip)}.json"


def read_frame_types(gt_dir: Path, clip: str) -> dict:
    p = frame_type_path(gt_dir, clip)
    if not p.exists():
        return {"clip": canonical_clip_id(clip), "frames": {}}
    return json.loads(p.read_text())


def write_frame_types(gt_dir: Path, clip: str, doc: dict) -> None:
    path = frame_type_path(gt_dir, clip)
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


def update_frame_types(gt_dir: Path, clip: str, fn) -> dict:
    """Read-modify-write under a process lock; ``fn(doc)`` mutates ``doc``."""
    with _lock:
        doc = read_frame_types(gt_dir, clip)
        fn(doc)
        write_frame_types(gt_dir, clip, doc)
        return doc


def not_broadcast_frames(gt_dir: Path, clip: str) -> set[int]:
    """Frame indices of ``clip`` flagged not-broadcast (empty when there is no sidecar)."""
    try:
        frames = read_frame_types(gt_dir, clip).get("frames", {})
    except (OSError, ValueError):
        return set()
    return {int(k) for k, v in frames.items() if (v or {}).get("type") == NOT_BROADCAST}
