"""One frame-index convention for reading videos: the *decoded* index.

Frame ``k`` is the k-th frame of a sequential decode (the k-th smallest presentation time). That is what the PTZ match
files, ``/calibration/match`` and every sequential reader number frames by.

``cv2.CAP_PROP_POS_FRAMES`` is only equal to that on files with a uniform pts timeline. On a file with a pts hole
(a dropped stretch of the broadcast recording; Bournemouth ``seg060``/``seg073``/``seg075``/``seg087``) the seek is
time-based, so it lands tens to ~170 frames away from decoded index ``k``, and different re-encodes of the same video
disagree. So every reader goes through here:

* :func:`pts_report` / :func:`seek_is_exact`: parse the mp4 ``moov`` atom (a few hundred KB, no decoding; cached by
  path + mtime + size) and flag a pts gap > 1.5 frames or a duplicate pts;
* :func:`read_frame_decoded`: one frame by decoded index (cv2 seek on a uniform file, exact pts-ordered read otherwise);
* :func:`seek_decoded`: position an open ``cv2.VideoCapture`` so the next ``read()`` returns decoded frame ``idx``
  (for the seek-once-then-sequential readers).

Anything that cannot be assessed (not an mp4, unreadable, fragmented) is treated as seek-exact, i.e. unchanged
behaviour.
"""

from __future__ import annotations

import os
import struct
import threading
from pathlib import Path

import numpy as np

#: a pts step above this many median frame intervals is a hole
GAP_FRAMES = 1.5
#: a step below this fraction of the median interval is a duplicate / double-pts
DUP_FRACTION = 0.5

_lock = threading.Lock()
#: ``(resolved path, mtime_ns, size)`` -> report dict or None (cannot be assessed)
_REPORT_CACHE: dict[tuple, dict | None] = {}
#: same key -> sorted pts (seconds, edit-list applied) as float64 array
_PTS_CACHE: dict[tuple, np.ndarray | None] = {}


# --------------------------------------------------------------------------- moov parsing


def _boxes(f, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        h = f.read(8)
        if len(h) < 8:
            return
        size, typ = struct.unpack(">I4s", h)
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            return
        yield typ.decode("latin1"), pos + hdr, pos + size
        pos += size


def _video_track_info(f, s: int, e: int, info: dict) -> None:
    for t, a, b in _boxes(f, s, e):
        if t in ("mdia", "minf", "stbl", "edts"):
            _video_track_info(f, a, b, info)
        elif t == "hdlr":
            f.seek(a + 8)
            info["hdlr"] = f.read(4).decode("latin1")
        elif t == "mdhd":
            f.seek(a)
            v = f.read(1)[0]
            f.seek(a + (20 if v == 1 else 12))
            info["ts"] = struct.unpack(">I", f.read(4))[0]
        elif t == "stts":
            f.seek(a + 4)
            n = struct.unpack(">I", f.read(4))[0]
            info["stts"] = np.frombuffer(f.read(8 * n), ">u4").reshape(-1, 2).astype(np.int64)
        elif t == "ctts":
            f.seek(a)
            v = f.read(1)[0]
            f.seek(a + 4)
            n = struct.unpack(">I", f.read(4))[0]
            d = f.read(8 * n)
            cnt = np.frombuffer(d, ">u4").reshape(-1, 2)[:, 0].astype(np.int64)
            off = np.frombuffer(d, ">i4" if v == 1 else ">u4").reshape(-1, 2)[:, 1].astype(np.int64)
            info["ctts"] = (cnt, off)
        elif t == "elst":
            f.seek(a)
            v = f.read(1)[0]
            f.seek(a + 4)
            n = struct.unpack(">I", f.read(4))[0]
            for _ in range(n):
                if v == 1:
                    _sd, mt, _ = struct.unpack(">QqH", f.read(18))
                    f.read(2)
                else:
                    _sd, mt = struct.unpack(">Ii", f.read(8))
                    f.read(4)
                info.setdefault("edits", []).append(mt)


def moov_pts(path: Path | str) -> np.ndarray | None:
    """Sorted presentation times (seconds, edit-list start applied) of every sample of the first video track, read from
    the ``moov`` atom only. ``None`` when ``path`` is not a parseable mp4/mov with a sample table."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            moov = next(((s, e) for t, s, e in _boxes(f, 0, size) if t == "moov"), None)
            if moov is None:
                return None
            info = None
            for t, s, e in _boxes(f, *moov):
                if t != "trak":
                    continue
                cand: dict = {}
                _video_track_info(f, s, e, cand)
                if cand.get("hdlr") == "vide":
                    info = cand
                    break
    except (OSError, struct.error, IndexError, ValueError):
        return None
    if not info or "stts" not in info or not info.get("ts"):
        return None
    counts, deltas = info["stts"][:, 0], info["stts"][:, 1]
    n = int(counts.sum())
    if n == 0:
        return None
    step = np.repeat(deltas, counts)
    pts = np.concatenate([[0], np.cumsum(step)[:-1]]).astype(np.int64)
    if "ctts" in info:
        cnt, off = info["ctts"]
        o = np.repeat(off, cnt)[:n]
        pts[: len(o)] += o
    shift = 0
    edits = [m for m in info.get("edits", []) if m >= 0]
    if edits:
        shift = edits[0]
    # ffmpeg drops samples with pts before the edit start
    return np.sort((pts[pts >= shift] - shift) / float(info["ts"]))


def _key(path: Path | str) -> tuple | None:
    try:
        p = Path(path)
        st = p.stat()
        return (str(p.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def pts_table(path: Path | str) -> np.ndarray | None:
    """Cached :func:`moov_pts` (key: path + mtime + size)."""
    key = _key(path)
    if key is None:
        return None
    with _lock:
        if key in _PTS_CACHE:
            return _PTS_CACHE[key]
    pts = moov_pts(path)
    with _lock:
        _PTS_CACHE[key] = pts
    return pts


def analyse_pts(pts: np.ndarray) -> dict:
    """Gaps (> :data:`GAP_FRAMES` median intervals) and duplicate pts of a sorted pts array."""
    d = np.diff(pts)
    dt = float(np.median(d)) if len(d) else 0.0
    if dt <= 0:
        return {"n": len(pts), "dt": dt, "gaps": [], "dups": int((d <= 0).sum()), "uniform": len(d) == 0}
    gap_idx = np.flatnonzero(d > GAP_FRAMES * dt)
    dup_idx = np.flatnonzero(d < DUP_FRACTION * dt)
    return {
        "n": int(len(pts)),
        "dt": dt,
        "gaps": [{"after_frame": int(i), "t": float(pts[i]), "gap_s": float(d[i]), "gap_frames": float(d[i] / dt)} for i in gap_idx],
        "dups": int(len(dup_idx)),
        "uniform": bool(len(gap_idx) == 0 and len(dup_idx) == 0),
    }


def pts_report(path: Path | str) -> dict | None:
    """Cached pts-timeline report ``{n, dt, gaps, dups, uniform}`` or ``None`` when the file cannot be assessed."""
    key = _key(path)
    if key is None:
        return None
    with _lock:
        if key in _REPORT_CACHE:
            return _REPORT_CACHE[key]
    pts = pts_table(path)
    rep = None if pts is None else analyse_pts(pts)
    with _lock:
        _REPORT_CACHE[key] = rep
    return rep


def seek_is_exact(video: Path | str) -> bool:
    """Whether ``CAP_PROP_POS_FRAMES`` == decoded index for ``video`` (uniform pts, or not assessable)."""
    rep = pts_report(video)
    return rep is None or rep["uniform"]


def clear_cache() -> None:
    with _lock:
        _REPORT_CACHE.clear()
        _PTS_CACHE.clear()


# --------------------------------------------------------------------------- readers


def read_frame_decoded(video: Path | str, idx: int):
    """BGR frame number ``idx`` as a sequential decode numbers it, or ``None``.

    cv2 seek (with the black-frame ffmpeg fallback of ``calibration.video``) when the pts timeline is uniform,
    otherwise the exact pts-ordered read."""
    from footy_track.calibration import video as cv  # noqa: PLC0415 (video imports this module)

    if idx < 0:
        return None
    if seek_is_exact(video):
        return cv.read_frame_seek(Path(video), idx)
    return cv.read_frame_exact(video, idx)


_POS_FRAMES = 1  # cv2.CAP_PROP_POS_FRAMES
_POS_MSEC = 0  # cv2.CAP_PROP_POS_MSEC


def seek_decoded(cap, video: Path | str, idx: int, *, max_tries: int = 8) -> bool:
    """Position ``cap`` (an open ``cv2.VideoCapture`` on ``video``) so the next ``read()`` returns decoded frame ``idx``.

    A plain ``cap.set(CAP_PROP_POS_FRAMES, idx)`` on a uniform file. On a file with a pts hole the seek is time-based
    and misses, so the landing frame's timestamp (``CAP_PROP_POS_MSEC``) is mapped to its decoded index through the pts
    table and corrected: seek early, then ``grab`` forward to ``idx - 1``. Returns ``False`` if it could not be
    positioned (the capture is then at an unspecified place)."""
    idx = max(0, int(idx))
    pts = None if seek_is_exact(video) else pts_table(video)
    if pts is None or idx == 0:
        cap.set(_POS_FRAMES, idx)
        return True
    if idx >= len(pts):
        return False
    half = 0.5 * float(np.median(np.diff(pts)))
    guess = idx
    for _ in range(max_tries):
        cap.set(_POS_FRAMES, max(0, guess))
        if not cap.grab():
            return False
        t = cap.get(_POS_MSEC) / 1000.0
        got = int(np.clip(np.searchsorted(pts, t - half), 0, len(pts) - 1))  # first frame with pts >= t - half
        if abs(float(pts[got]) - t) > 1.5 * half * 2:  # timestamp does not map to a frame of this file
            return False
        if got <= idx - 1:
            for _ in range(idx - 1 - got):
                if not cap.grab():
                    return False
            return True
        guess -= (got - idx) + 1  # landed on/after idx: seek earlier by the overshoot (+1 frame of slack)
        if guess < 0:
            guess = 0
            cap.set(_POS_FRAMES, 0)
    return _seek_by_decoding(cap, idx)


def _seek_by_decoding(cap, idx: int) -> bool:
    cap.set(_POS_FRAMES, 0)
    for _ in range(idx):
        if not cap.grab():
            return False
    return True
