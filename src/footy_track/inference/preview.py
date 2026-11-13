"""Small "preview" copies of the overlay videos, for slow networks.

The full-quality overlays are about 5-6 Mbit/s at 1080p. Over a weak Wi-Fi link a
browser can fetch them slower than they play, so the player shows the first frame
and stalls. A preview is the same video at about 720p and 1-1.5 Mbit/s: it starts
quickly and plays on a slow link. It is only an extra copy; the full-quality file
is never replaced.

Same frame count and frame rate as the input (``-fps_mode passthrough``), no
audio, H.264 yuv420p with ``+faststart`` so the browser can start playing before
the whole file has arrived. Unlike the stitch module's raw-H.264 path, the input
here is a normal mp4 and the output a normal mp4, so timestamps are kept and the
muxer does not trim B-frame delay (``-bf 0`` is not needed; frame counts are
checked in the tests).
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path

from footy_track.inference.stitch import StitchCancelled, _run

PREVIEW_HEIGHT = 720
# Fit inside 720 pixels of height (never enlarge), keep the aspect ratio, and
# round the width to an even number for yuv420p.
PREVIEW_SCALE = f"scale=-2:min(ih\\,{PREVIEW_HEIGHT})"
PREVIEW_ARGS = [
    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "29",
    "-maxrate", "1500k", "-bufsize", "3M", "-pix_fmt", "yuv420p",
    "-vf", PREVIEW_SCALE, "-fps_mode", "passthrough", "-movflags", "+faststart",
]  # fmt: skip


def make_preview(
    src: Path, dst: Path, is_cancelled: Callable[[], bool] = lambda: False
) -> dict:
    """Write a preview of ``src`` to ``dst``. Returns ``{"bytes", "seconds"}``.

    Raises ``StitchCancelled`` on cancel and ``RuntimeError`` on ffmpeg failure;
    a partial file is never left at ``dst``.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed on the server.")
    t0 = time.time()
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    cmd = [ffmpeg, "-hide_banner", "-y", "-loglevel", "error", "-i", str(src)]
    cmd += [*PREVIEW_ARGS, "-f", "mp4", str(tmp)]
    try:
        _run(cmd, is_cancelled, [])
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError("preview came out empty")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return {"bytes": dst.stat().st_size, "seconds": round(time.time() - t0, 2)}


__all__ = ["PREVIEW_ARGS", "StitchCancelled", "make_preview"]
