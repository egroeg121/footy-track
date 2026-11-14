"""Stitch a job's overlay clips and the untouched gaps into one ``full.mp4``.

The cutter keeps only broadcast shots, so the clips do not cover the whole
upload. This module works out what is *between* them (the timeline) and, after
tracking, builds one video that plays the whole upload in order: the overlay
for each tracked clip and the plain source footage for everything else.

Everything is counted in **source frame indices**, not seconds, so the cuts are
frame-accurate and the output has exactly as many frames as the source:

* a clip starts at frame ``round(start_s * fps)`` (the cutter wrote
  ``start_s = start_frame / fps`` rounded to milliseconds, far below half a
  frame) and is ``frame_count`` frames long;
* a clip that was only partly tracked (``max_frames`` / ``stride``) is tracked
  for ``frames_processed * stride`` frames and the rest is plain footage;
* a clip that failed, and every gap, is plain footage ("skipped").

Each piece is encoded on its own to raw H.264 with the same settings and an
exact frame count, the pieces are byte-concatenated and wrapped into an mp4 at
the source frame rate. Raw H.264 carries no timestamps, so there is nothing to
drift or to mismatch between pieces. Audio is dropped.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

logger = logging.getLogger(__name__)

TRACKED, SKIPPED = "tracked", "skipped"
# -bf 0: no B-frames. With them the mp4 muxer trims the reorder delay at the start
# of the stream and the output ends up a couple of frames short.
ENCODE_ARGS = [
    "-an", "-c:v", "libx264", "-preset", "veryfast", "-bf", "0", "-crf", "20",
    "-pix_fmt", "yuv420p",
]  # fmt: skip
SCALE_EVEN = "scale=trunc(iw/2)*2:trunc(ih/2)*2"


class StitchCancelled(Exception):
    """The job was cancelled while stitching."""


def fps_fraction(fps: float) -> Fraction:
    """Exact frame rate from a (possibly rounded) float: 29.97 -> 30000/1001.

    NTSC rates (24000/1001, 30000/1001, ...) are recognised even when rounded to
    three decimals; anything else is taken as a simple fraction (25 -> 25/1).
    """
    k = round(fps * 1.001)
    if k > 0 and abs(fps - k / 1.001) < 0.01:
        return Fraction(k * 1000, 1001)
    return Fraction(fps).limit_denominator(1000)


def _tracked_frames(clip: dict, stride: int) -> int:
    """How many source frames of this clip the overlay covers."""
    if clip.get("status") != "done" or not clip.get("summary"):
        return 0
    processed = int(clip["summary"].get("frames_processed") or 0)
    return min(int(clip["frame_count"]), processed * max(stride, 1))


def compute_timeline(
    clips: list[dict], total_frames: int, fps: float, stride: int = 1
) -> list[dict]:
    """Split the whole video into ``tracked`` and ``skipped`` spans.

    ``clips`` need ``id, status, start_s, frame_count, summary``. Spans are in
    time order, cover ``[0, total_frames)`` without overlap, and adjacent
    skipped spans are merged. The last span ends at ``total_frames`` (or at the
    last clip's end if the clips run past the reported frame count). Each span:
    ``start_frame, end_frame, start_s, end_s, kind, clip`` (clip id or None).
    """
    spans: list[dict] = []

    def add(start: int, end: int, kind: str, clip: str | None) -> None:
        if end <= start:
            return
        if kind == SKIPPED and spans and spans[-1]["kind"] == SKIPPED:
            spans[-1]["end_frame"] = end
        else:
            spans.append(
                {"start_frame": start, "end_frame": end, "kind": kind, "clip": clip}
            )

    cursor = 0
    for clip in sorted(clips, key=lambda c: c["start_s"]):
        start = max(round(clip["start_s"] * fps), cursor)
        end = start + int(clip["frame_count"])
        tracked = min(_tracked_frames(clip, stride), max(end - start, 0))
        add(cursor, start, SKIPPED, None)
        add(start, start + tracked, TRACKED, clip["id"] if tracked else None)
        add(start + tracked, end, SKIPPED, None)
        cursor = max(cursor, end)
    add(cursor, max(total_frames, cursor), SKIPPED, None)

    for s in spans:
        s["start_s"] = round(s["start_frame"] / fps, 3)
        s["end_s"] = round(s["end_frame"] / fps, 3)
    return spans


def _run(cmd: list[str], is_cancelled: Callable[[], bool], log: list[str]) -> None:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    err: list[str] = []
    reader = threading.Thread(
        target=lambda: err.extend((proc.stderr.read() or "")[-2000:].splitlines()),
        daemon=True,
    )
    reader.start()
    try:
        while True:
            try:
                rc = proc.wait(timeout=0.3)
                break
            except subprocess.TimeoutExpired:
                if is_cancelled():
                    proc.kill()
                    proc.wait()
                    raise StitchCancelled from None
    finally:
        if proc.poll() is None:
            proc.kill()
    reader.join(timeout=2)
    if rc != 0:
        log.extend(err)
        raise RuntimeError(f"ffmpeg failed (exit {rc}): {' | '.join(err[-3:])}")


def stitch_full_video(
    source: Path,
    job_dir: Path,
    spans: list[dict],
    fps: float,
    is_cancelled: Callable[[], bool],
    on_progress: Callable[[int, int], None] = lambda _d, _t: None,
) -> dict:
    """Build ``<job_dir>/full.mp4`` from ``spans``. Returns ``{"seconds", "pieces"}``.

    Raises ``StitchCancelled`` on cancel and ``RuntimeError`` on any ffmpeg
    failure; the caller decides what that means for the job.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed on the server.")
    rate = fps_fraction(fps)
    rate_s = f"{rate.numerator}/{rate.denominator}"
    t0 = time.time()
    out = job_dir / "full.mp4"
    tmp_out = job_dir / "full.tmp.mp4"
    log: list[str] = []
    with tempfile.TemporaryDirectory(dir=job_dir, prefix="stitch_") as td:
        work = Path(td)
        pieces: list[Path] = []
        for i, span in enumerate(spans):
            if is_cancelled():
                raise StitchCancelled
            n = span["end_frame"] - span["start_frame"]
            piece = work / f"p{i:04d}.h264"
            base = [ffmpeg, "-hide_banner", "-y", "-loglevel", "error"]
            if span["kind"] == TRACKED:
                overlay = job_dir / "out" / span["clip"] / "overlay.mp4"
                # A strided overlay has fewer frames; fps= repeats each one so
                # it lasts as long as the source frames it stands for.
                vf = f"fps={rate_s},{SCALE_EVEN}"
                cmd = [*base, "-i", str(overlay), *ENCODE_ARGS, "-vf", vf]
            else:
                # Seek half a frame early so the first frame decoded is exactly
                # ``start_frame`` whatever the container's timestamp rounding.
                ss = max(span["start_frame"] - 0.5, 0) / float(rate)
                cmd = [*base, "-ss", f"{ss:.6f}", "-i", str(source), *ENCODE_ARGS]
                cmd += ["-vf", SCALE_EVEN]
            # passthrough: no duplicated or dropped frames to "fix" timestamps
            cmd += ["-fps_mode", "passthrough", "-frames:v", str(n)]
            cmd += ["-f", "h264", str(piece)]
            _run(cmd, is_cancelled, log)
            if not piece.exists() or piece.stat().st_size == 0:
                raise RuntimeError(f"piece {i} ({span['kind']}) came out empty")
            pieces.append(piece)
            on_progress(i + 1, len(spans) + 1)

        joined = work / "all.h264"
        with joined.open("wb") as dst:
            for piece in pieces:
                with piece.open("rb") as src:
                    shutil.copyfileobj(src, dst, 1 << 20)
                piece.unlink()
        _run(
            [
                ffmpeg,
                "-hide_banner",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                rate_s,
                "-i",
                str(joined),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(tmp_out),
            ],  # fmt: skip
            is_cancelled,
            log,
        )
    tmp_out.replace(out)
    on_progress(len(spans) + 1, len(spans) + 1)
    return {"seconds": round(time.time() - t0, 2), "pieces": len(spans)}
