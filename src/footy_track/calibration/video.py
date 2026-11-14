"""Single frame reader for calibration (routes and the pre-labelling pipeline).

OpenCV 5.0 decodes some clips (e.g. ``arsenal_example``, ``mancity_part050``,
``mancity_part093``: swscaler logs "Cannot convert interlaced to progressive
frames") as all-black frames while reporting success. The ffmpeg CLI decodes
them fine, so when cv2 fails or yields an all-black frame we fall back to
ffmpeg, seeking frame-accurately, and remember per video which path works.
"""

from __future__ import annotations

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


def reader_for(video: Path | str) -> str | None:
    """Which reader last worked for ``video`` (``"cv2"``/``"ffmpeg"``/None)."""
    with _lock:
        return _PATH_CACHE.get(_key(Path(video)))


def clear_cache() -> None:
    with _lock:
        _PATH_CACHE.clear()
