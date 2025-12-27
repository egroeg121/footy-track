#!/usr/bin/env python3
"""
Split a video file into fixed-length chunks using ffmpeg.

Default chunk length is 10 seconds. By default this uses stream copy (-c copy)
for speed, which creates segments at the nearest keyframes around each boundary.
If you need exact boundaries, use --reencode to force keyframes at segment
boundaries (slower and re-encodes the video).

Usage examples:
  uvx scripts/split_video.py path/to/video.mp4
  uvx scripts/split_video.py path/to/video.mp4 --chunk 5 --outdir ./out
  uvx scripts/split_video.py path/to/video.mp4 --reencode --bitrate 4M

Requires: ffmpeg to be installed and available on PATH.
macOS install hint: brew install ffmpeg
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(
            "Error: ffmpeg not found on PATH.\nInstall it (e.g., on macOS: brew install ffmpeg) and try again.",
            file=sys.stderr,
        )
        sys.exit(1)
    return ffmpeg


def build_output_dir(input_path: Path, outdir: str | None) -> Path:
    out = Path(outdir) if outdir else input_path.parent / f"{input_path.stem}_chunks"
    out.mkdir(parents=True, exist_ok=True)
    return out


def split_video(  # noqa: PLR0913
    input_path: Path,
    output_dir: Path,
    chunk_seconds: float,
    re_encode: bool,
    vcodec: str | None,
    acodec: str | None,
    bitrate: str | None,
    extra_ffmpeg_args: list[str],
) -> int:
    ffmpeg = check_ffmpeg()

    # Output filename pattern: <name>_part000.mp4
    # Keep original extension if mp4-like, otherwise default to .mp4
    ext = input_path.suffix.lower() or ".mp4"
    if ext not in {".mp4", ".mov", ".mkv", ".ts", ".m4v"}:
        ext = ".mp4"
    out_pattern = output_dir / f"{input_path.stem}_part%03d{ext}"

    base_cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",  # overwrite without prompt
        "-loglevel",
        "info",
        "-i",
        str(input_path),
    ]

    if re_encode:
        # Re-encode video so we can force keyframes at exact segment boundaries.
        # This ensures chunks are very close to the requested duration.
        # Choose sensible defaults if codecs not provided.
        vcodec_flag = vcodec or "libx264"
        acodec_flag = acodec or "aac"
        # Force keyframes at multiples of chunk_seconds
        force_kf = f"expr:gte(t,n_forced*{chunk_seconds})"
        reencode_args = [
            "-map",
            "0",
            "-c:v",
            vcodec_flag,
            "-c:a",
            acodec_flag,
            "-force_key_frames",
            force_kf,
        ]
        if bitrate:
            reencode_args += ["-b:v", bitrate]
        stream_args = reencode_args
    else:
        # Fast path: stream copy. Boundaries align to keyframes nearest each segment point.
        stream_args = [
            "-map",
            "0",
            "-c",
            "copy",
        ]

    segment_args = [
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
    ]

    cmd = base_cmd + stream_args + extra_ffmpeg_args + segment_args + [str(out_pattern)]

    print("Running:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    except KeyboardInterrupt:
        return 130


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Split a video into fixed-length chunks using ffmpeg."
    )
    p.add_argument("input", help="Path to input video file")
    p.add_argument(
        "--outdir",
        "-o",
        help="Output directory (default: <input_stem>_chunks next to input)",
    )
    p.add_argument(
        "--chunk",
        type=float,
        default=10.0,
        help="Chunk length in seconds (default: 10)",
    )
    p.add_argument(
        "--re-encode",
        dest="re_encode",
        action="store_true",
        help="Re-encode to force exact boundaries (slower)",
    )
    p.add_argument("--vcodec", help="Video codec when re-encoding (default: libx264)")
    p.add_argument("--acodec", help="Audio codec when re-encoding (default: aac)")
    p.add_argument("--bitrate", help="Target video bitrate when re-encoding, e.g., 4M")
    p.add_argument(
        "--ffmpeg-arg",
        action="append",
        default=[],
        help="Extra raw ffmpeg arg(s). Can be used multiple times, e.g., --ffmpeg-arg -vf --ffmpeg-arg scale=1280:-2",
    )

    args = p.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    output_dir = build_output_dir(input_path, args.outdir)

    rc = split_video(
        input_path=input_path,
        output_dir=output_dir,
        chunk_seconds=args.chunk,
        re_encode=args.re_encode,
        vcodec=args.vcodec,
        acodec=args.acodec,
        bitrate=args.bitrate,
        extra_ffmpeg_args=args.ffmpeg_arg,
    )

    if rc == 0:
        print(f"Done. Chunks written to: {output_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
