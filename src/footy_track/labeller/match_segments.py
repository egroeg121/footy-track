"""Index of the full-match broadcast segments: canonical clip id -> segment mp4.

``<footy_data>/<match>/split_video_broadcast_frames/segments.json`` lists the segment files of a match (the same
files ``/calibration/match`` addresses by ``(segment, local frame)``). The calibration editor uses this index as a
fallback for clips that are not in the active dataset's clip directories, so every segment of a match can be opened
at ``/calibration?clip=<canonical_clip_id(stem)>&frame=<local>``.

Only paths listed in a ``segments.json`` under the data root are ever returned (the clip id is a dictionary key,
never joined into a path), and a listed ``file`` must be a bare file name.

When two files share one canonical id the lexicographically first match directory wins (deterministic; checked
at index build, see ``ambiguous``).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import matches_status

SEGMENTS_DIR = "split_video_broadcast_frames"
SEGMENTS_JSON = "segments.json"
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
#: The index is rebuilt at most this often (a directory listing + one stat per match).
_TTL_S = 30.0

_lock = threading.Lock()
#: root -> (built at (monotonic), signature, {canonical id: path}, ambiguous canonical ids)
_CACHE: dict[str, tuple[float, tuple, dict[str, Path], dict[str, list[Path]]]] = {}


def _bare_name(name: object) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and name == Path(name).name
        and not name.startswith(".")
        and Path(name).suffix.lower() in _VIDEO_SUFFIXES
    )


def _match_dirs(root: Path) -> list[Path]:
    try:
        return sorted(p for p in root.iterdir() if (p / SEGMENTS_DIR / SEGMENTS_JSON).is_file())
    except OSError:
        return []


def _signature(dirs: list[Path]) -> tuple:
    sig = []
    for d in dirs:
        try:
            st = (d / SEGMENTS_DIR / SEGMENTS_JSON).stat()
        except OSError:
            continue
        sig.append((d.name, st.st_mtime_ns, st.st_size))
    return tuple(sig)


def _build(dirs: list[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    index: dict[str, Path] = {}
    ambiguous: dict[str, list[Path]] = {}
    for d in dirs:
        seg_dir = d / SEGMENTS_DIR
        try:
            doc = json.loads((seg_dir / SEGMENTS_JSON).read_text())
        except (OSError, ValueError):
            continue
        segs = doc.get("segments") if isinstance(doc, dict) else None
        if not isinstance(segs, list):
            continue
        for seg in segs:
            name = seg.get("file") if isinstance(seg, dict) else None
            if not _bare_name(name):
                continue
            path = seg_dir / name
            cid = canonical_clip_id(Path(name).stem)
            if cid in index:
                if index[cid] != path:
                    ambiguous.setdefault(cid, [index[cid]]).append(path)
                continue
            index[cid] = path
    return index, ambiguous


def _entry(root: Path | None = None) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    root = Path(root) if root is not None else matches_status.data_root()
    key = str(root)
    now = time.monotonic()
    with _lock:
        hit = _CACHE.get(key)
        if hit is not None and now - hit[0] < _TTL_S:
            return hit[2], hit[3]
    dirs = _match_dirs(root)
    sig = _signature(dirs)
    with _lock:
        hit = _CACHE.get(key)
        if hit is not None and hit[1] == sig:
            _CACHE[key] = (now, sig, hit[2], hit[3])
            return hit[2], hit[3]
    index, ambiguous = _build(dirs)
    with _lock:
        _CACHE[key] = (now, sig, index, ambiguous)
    return index, ambiguous


def segment_index(root: Path | None = None) -> dict[str, Path]:
    """``canonical clip id -> segment mp4 path`` over every match under the data root (cached)."""
    return dict(_entry(root)[0])


def ambiguous(root: Path | None = None) -> dict[str, list[Path]]:
    """Canonical ids claimed by more than one file (the first one is what ``find_segment_video`` returns)."""
    return dict(_entry(root)[1])


def find_segment_video(clip: str, root: Path | None = None) -> Path | None:
    """The full-match segment video of ``clip`` (any alias of its canonical id), or ``None`` when unknown or the
    file is missing."""
    if not clip or not isinstance(clip, str):
        return None
    path = _entry(root)[0].get(canonical_clip_id(clip))
    return path if path is not None and path.is_file() else None


def clear_cache() -> None:
    with _lock:
        _CACHE.clear()
