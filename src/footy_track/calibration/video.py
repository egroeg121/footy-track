"""Single frame reader for calibration (routes and the pre-labelling pipeline).

OpenCV 5.0 decodes some clips (e.g. ``arsenal_example``, ``mancity_part050``,
``mancity_part093``: swscaler logs "Cannot convert interlaced to progressive
frames") as all-black frames while reporting success. The ffmpeg CLI decodes
them fine, so when cv2 fails or yields an all-black frame we fall back to
ffmpeg, seeking frame-accurately, and remember per video which path works.
"""

from __future__ import annotations

import collections
import json
import shutil
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

#: A frame whose brightest pixel is at or below this is treated as "all black".
BLACK_MAX = 8

_lock = threading.Lock()
#: resolved video path -> "cv2" | "ffmpeg" (which reader produced a real frame)
_PATH_CACHE: dict[str, str] = {}


def is_black(frame: np.ndarray | None) -> bool:
    return frame is None or frame.size == 0 or int(frame.max()) <= BLACK_MAX


def _key(video: Path) -> str:
    return str(Path(video).resolve())


def _cv2_frame(video: Path, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    return fps if fps > 0 else 25.0


def _ffmpeg_frame(video: Path, idx: int, fps: float | None = None) -> np.ndarray | None:
    """Frame ``idx`` via the ffmpeg CLI.

    Seeks half a frame *early*, ``(idx - 0.5) / fps``: with ``-ss`` before
    ``-i`` ffmpeg decodes up to the target and outputs the first frame whose
    pts is ``>= t``, so that is frame ``idx`` and rounding of container
    timestamps cannot land on a neighbour. (Seeking to ``(idx + 0.5) / fps``
    returns frame ``idx + 1`` and nothing at all for the last frame.)
    """
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    fps = fps or _fps(video)
    t = max(0.0, (idx - 0.5) / fps)
    cmd = [
        exe, "-v", "error", "-nostdin", "-ss", f"{t:.6f}", "-i", str(video),
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)


def read_frame(video: Path | str, idx: int) -> np.ndarray | None:
    """BGR frame ``idx`` of ``video``, or ``None`` if it cannot be read.

    Uses cv2 unless it fails or returns an all-black frame, then ffmpeg. The
    working reader is cached per video; a video known to need ffmpeg skips cv2.
    """
    video = Path(video)
    key = _key(video)
    with _lock:
        known = _PATH_CACHE.get(key)
    if known != "ffmpeg":
        frame = _cv2_frame(video, idx)
        if frame is not None and not is_black(frame):
            with _lock:
                _PATH_CACHE[key] = "cv2"
            return frame
    else:
        frame = None
    alt = _ffmpeg_frame(video, idx)
    if alt is not None and not is_black(alt):
        with _lock:
            _PATH_CACHE[key] = "ffmpeg"
        return alt
    # Both black or failed: a genuinely dark frame (fade) is still a frame.
    return frame if frame is not None else alt


#: ``(resolved path, mtime_ns, size)`` -> (start_time, sorted pts per decoded frame, width, height) or None
_PTS_CACHE: dict[tuple, tuple[float, np.ndarray, int, int] | None] = {}


def _pts_table(video: Path) -> tuple[float, np.ndarray, int, int] | None:
    """Presentation time of every decoded frame of ``video`` (one ffprobe packet scan, no decoding), cached."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    try:
        st = video.stat()
    except OSError:
        return None
    key = (_key(video), st.st_mtime_ns, st.st_size)
    with _lock:
        if key in _PTS_CACHE:
            return _PTS_CACHE[key]
    table = None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "packet=pts_time:stream=width,height:format=start_time", "-of", "json", str(video)],
            capture_output=True, text=True, timeout=120, check=True).stdout
        doc = json.loads(out)
        pts = np.sort(np.array([float(p["pts_time"]) for p in doc.get("packets", []) if "pts_time" in p]))
        stream = (doc.get("streams") or [{}])[0]
        start = float((doc.get("format") or {}).get("start_time") or 0.0)
        if len(pts) and stream.get("width") and stream.get("height"):
            table = (start, pts, int(stream["width"]), int(stream["height"]))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        table = None
    with _lock:
        _PTS_CACHE[key] = table
    return table


#: Last few exact frames: one editor view asks for the same frame several times (image, warp, auto points) and an
#: ffmpeg seek + decode costs seconds on a long-GOP segment.
_EXACT_CACHE_MAX = 6
_EXACT_CACHE: collections.OrderedDict[tuple, np.ndarray] = collections.OrderedDict()


def read_frame_exact(video: Path | str, idx: int) -> np.ndarray | None:
    """Cached :func:`_read_frame_exact` (the returned array is a private copy)."""
    video = Path(video)
    try:
        st = video.stat()
    except OSError:
        return None
    key = (_key(video), st.st_mtime_ns, st.st_size, int(idx))
    with _lock:
        hit = _EXACT_CACHE.get(key)
        if hit is not None:
            _EXACT_CACHE.move_to_end(key)
            return hit.copy()
    frame = _read_frame_exact(video, idx)
    if frame is not None:
        with _lock:
            _EXACT_CACHE[key] = frame
            while len(_EXACT_CACHE) > _EXACT_CACHE_MAX:
                _EXACT_CACHE.popitem(last=False)
        return frame.copy()
    return None


def _read_frame_exact(video: Path | str, idx: int) -> np.ndarray | None:
    """BGR frame number ``idx`` *as a sequential decode numbers it*, even when the file cannot be seeked by index.

    ``CAP_PROP_POS_FRAMES`` seeks are not frame-accurate on long-GOP / variable-frame-rate broadcast segments
    (Bournemouth ``seg087``: tens to ~170 frames off), while the PTZ calibration run numbers frames by a sequential
    decode (``Frame.local_frame``). Here the k-th decoded frame is the one with the k-th smallest pts: ffmpeg is
    asked for the first frame at or after the midpoint between pts[k-1] and pts[k] (pixels identical to the cv2
    decode). Falls back to :func:`read_frame`
    when ffprobe/ffmpeg are unavailable. ``None`` when ``idx`` is outside the file."""
    video = Path(video)
    if idx < 0:
        return None
    table = _pts_table(video)
    exe = shutil.which("ffmpeg")
    if table is None or exe is None:
        return read_frame(video, idx)
    start, pts, w, h = table
    if idx >= len(pts):
        return None
    t = 0.0 if idx == 0 else float((pts[idx - 1] + pts[idx]) / 2.0) - start
    cmd = [exe, "-v", "error", "-nostdin", "-ss", f"{max(t, 0.0):.6f}", "-i", str(video),
           # cv2 converts YUV with the BT.601 matrix whatever the stream says; match it so pixels are identical
           "-vf", "scale=in_color_matrix=bt601:in_range=tv", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=120, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return read_frame(video, idx)
    if len(raw) != w * h * 3:
        return read_frame(video, idx)
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()


def reader_for(video: Path | str) -> str | None:
    """Which reader last worked for ``video`` (``"cv2"``/``"ffmpeg"``/None)."""
    with _lock:
        return _PATH_CACHE.get(_key(Path(video)))


def clear_cache() -> None:
    with _lock:
        _PATH_CACHE.clear()
        _PTS_CACHE.clear()
        _EXACT_CACHE.clear()
