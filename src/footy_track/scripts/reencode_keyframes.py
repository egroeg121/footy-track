"""Re-encode clips with a keyframe every N frames so labeller seeks are cheap.

Segment clips cut by stream copy can have only one or two keyframes, so every
labeller seek (Ball Check stills and context clips, the main labeller) decodes
1080p from the start of the clip: seconds per card on the Mac mini. With a
keyframe every 10 frames a seek decodes at most 9 extra frames.

Frames map 1:1 (``-fps_mode passthrough``); each output is checked to have the
same video packet count as its source before it replaces anything, so frame
indices in GT sidecars, candidates and verdicts stay valid. The source dir is
never modified. Point ``FOOTY_CLIPS_DIR`` at the output dir.

    python -m footy_track.scripts.reencode_keyframes SRC_DIR OUT_DIR [--gop 10]

The output dir starts as symlinks to the sources and each clip is swapped in
atomically, so a server already pointed at OUT_DIR keeps working throughout
and gets faster clip by clip. ``--first stem1,stem2`` re-encodes those clips
first (e.g. the clips of the next cards in a review list).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

_SUFFIXES = (".mp4", ".mkv", ".mov", ".webm", ".avi")


def video_packets(path: Path) -> int:
    """Number of video packets (frames) in ``path``, without decoding."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return int(out.strip())


def reencode(
    src: Path, dst: Path, gop: int, crf: int, preset: str, threads: int
) -> str:
    """Re-encode one clip into ``dst`` (atomically). Returns a status string."""
    # Temp output lives in a sibling dir (same filesystem, so the swap stays
    # atomic) rather than in OUT_DIR, where a server listing clips would see it.
    tmp_dir = dst.parent.parent / f".{dst.parent.name}.tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp = tmp_dir / dst.name
    cmd = [
        "nice", "-n", "5", "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-map", "0:v:0", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-pix_fmt", "yuv420p", "-an", "-threads", str(threads),
        "-movflags", "+faststart", str(tmp),
    ]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        return f"FAILED: {res.stderr.strip()[-300:]}"
    n_src, n_out = video_packets(src), video_packets(tmp)
    if n_src != n_out:
        tmp.unlink(missing_ok=True)
        return f"FRAME MISMATCH {n_src} vs {n_out}; kept source"
    os.replace(tmp, dst)
    return f"ok ({n_src} frames)"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--gop", type=int, default=10, help="frames between keyframes")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="ultrafast")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--first", default="", help="comma-separated stems to do first")
    args = ap.parse_args(argv)

    srcs = sorted(p for p in args.src_dir.iterdir() if p.suffix.lower() in _SUFFIXES)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in srcs:  # symlink first so OUT_DIR is complete and servable at once
        d = args.out_dir / s.name
        if not d.exists():
            d.symlink_to(s.resolve())
    first = [f for f in args.first.split(",") if f]
    srcs.sort(
        key=lambda p: (first.index(p.stem) if p.stem in first else len(first), p.name)
    )
    for s in srcs:
        d = args.out_dir / s.name
        if d.exists() and not d.is_symlink():
            continue  # already re-encoded
        t = time.time()
        status = reencode(s, d, args.gop, args.crf, args.preset, args.threads)
        print(f"{s.stem}: {status} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
