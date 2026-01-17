#!/usr/bin/env python3
"""
Extract frames from a video and save them as images using ffmpeg.

By default, extracts at 1 FPS into <input_stem>_frames/ as JPEGs.

Examples:
  uvx scripts/extract_frames.py path/to/video.mp4
  uvx scripts/extract_frames.py path/to/video.mp4 --fps 5 --outdir ./frames
  uvx scripts/extract_frames.py path/to/video.mp4 --start 30 --duration 120
  uvx scripts/extract_frames.py path/to/video.mp4 --width 1280 --format png
  uvx scripts/extract_frames.py path/to/video.mp4 --format webp --quality 85

Requires: ffmpeg on PATH (macOS: brew install ffmpeg)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import piexif
from piexif import ExifIFD
from piexif.helper import UserComment

from footy_track import constants


def check_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(
            "Error: ffmpeg not found on PATH.\nInstall it (e.g., on macOS: brew install ffmpeg) and try again.",
            file=sys.stderr,
        )
        sys.exit(1)
    return ffmpeg


def check_exiftool() -> str | None:
    """Return exiftool path if available, else None."""
    return shutil.which("exiftool")


def check_ffprobe() -> str | None:
    return shutil.which("ffprobe")


def ffprobe_avg_fps(input_path: Path) -> float | None:
    ffprobe = check_ffprobe()
    if not ffprobe:
        return None
    # Get avg_frame_rate like 30000/1001
    cmd = [
        ffprobe,
        "-v",
        "0",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return None
    try:
        if "/" in out:
            num, den = out.split("/", 1)
            num, den = float(num), float(den)
            return num / den if den != 0 else None
        val = float(out)
        return val if val > 0 else None
    except Exception:
        return None


def build_output_dir(input_path: Path, outdir: str | None) -> Path:
    out = Path(outdir) if outdir else input_path.parent / f"{input_path.stem}_frames"
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_vf_chain(
    fps: float | None, width: int | None, height: int | None, keyframes_only: bool
) -> str | None:
    parts: list[str] = []
    if keyframes_only:
        # Only keep I-frames
        parts.append("select='eq(pict_type,\"I\")'")
    if fps and fps > 0:
        parts.append(f"fps={fps}")
    if width or height:
        w = str(width) if width else "-2"
        h = str(height) if height else "-2"
        parts.append(f"scale={w}:{h}")
    if not parts:
        return None
    return ",".join(parts)


# New helper to embed EXIF metadata in JPEGs
def _embed_metadata_jpeg(
    files: list[Path],
    *,
    video_id: str,
    start_number: int,
    fps: float,
    start: float | None,
) -> None:
    if not piexif:
        print(
            "Warning: piexif not available, skipping metadata embed (pip install piexif)",
            file=sys.stderr,
        )
        return

    # Avoid division by zero
    fps_val = fps if fps and fps > 0 else 1.0
    t0 = float(start or 0.0)

    for i, fpath in enumerate(sorted(files)):
        frame_index = start_number + i
        # Compute timestamp from fps and start offset
        timestamp_seconds = t0 + (frame_index - start_number) / fps_val

        payload = {
            "video_id": video_id,
            "frame_index": int(frame_index),
            "timestamp_seconds": float(timestamp_seconds),
        }
        try:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            # Store a stable unique id combining video id and frame index
            exif_dict["Exif"][ExifIFD.ImageUniqueID] = str(f"{video_id}:{frame_index}")
            # Store JSON in UserComment
            exif_dict["Exif"][ExifIFD.UserComment] = UserComment.dump(
                json.dumps(payload), encoding="unicode"
            )
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(fpath))
        except Exception as e:  # pragma: no cover
            print(
                f"Warning: failed to embed metadata for {fpath.name}: {e}",
                file=sys.stderr,
            )


def extract_frames(  # noqa: PLR0913
    input_path: Path,
    output_dir: Path,
    fps: float,
    img_format: str,
    quality: int | None,
    start: float | None,
    duration: float | None,
    width: int | None,
    height: int | None,
    prefix: str | None,
    start_number: int,
    keyframes_only: bool,
    extra_ffmpeg_args: list[str],
    *,
    embed_metadata: bool = False,
    video_id: str | None = None,
) -> int:
    ffmpeg = check_ffmpeg()

    ext = {"jpg": ".jpg", "jpeg": ".jpg", "png": ".png", "webp": ".webp"}[img_format]
    name_prefix = prefix or input_path.stem
    out_pattern = output_dir / f"{name_prefix}_%06d{ext}"

    cmd: list[str] = [ffmpeg, "-hide_banner", "-y", "-loglevel", "info"]

    # Faster (but slightly less accurate) seek by placing -ss before -i
    if start is not None and start > 0:
        cmd += ["-ss", str(start)]

    cmd += ["-i", str(input_path)]

    if duration is not None and duration > 0:
        cmd += ["-t", str(duration)]

    vf = make_vf_chain(
        fps=fps, width=width, height=height, keyframes_only=keyframes_only
    )
    if vf:
        cmd += ["-vf", vf]

    # Image format/quality settings
    if img_format in {"jpg", "jpeg", "webp"} and quality is not None:
        cmd += ["-q:v", str(quality)]
    elif img_format == "png" and quality is not None:
        # Map 0-100 quality to PNG compression level 0-9 (inverse scale: lower is better quality)
        q = max(0, min(100, quality))
        compression_level = int(round((100 - q) * 9 / 100))
        cmd += ["-compression_level", str(compression_level)]

    cmd += ["-vsync", "vfr", "-start_number", str(start_number)]

    if extra_ffmpeg_args:
        cmd += list(extra_ffmpeg_args)

    cmd += [str(out_pattern)]

    print("Running:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False)
        rc = proc.returncode
    except KeyboardInterrupt:
        return 130

    if rc != 0:
        return rc

    # Optionally embed metadata (JPEG only)
    if embed_metadata and ext == ".jpg":
        files = sorted(output_dir.glob(f"{name_prefix}_*.jpg"))
        if files:
            _embed_metadata_jpeg(
                files,
                video_id=video_id or input_path.stem,
                start_number=start_number,
                fps=fps,
                start=start,
            )
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract frames from a video and save as images (ffmpeg)"
    )
    p.add_argument("input", help="Path to input video file")
    p.add_argument(
        "--outdir",
        "-o",
        help="Output directory (default: <input_stem>_frames next to input)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)",
    )
    p.add_argument(
        "--format",
        choices=["jpg", "jpeg", "png", "webp"],
        default=constants.IMAGE_FORMAT,
        help="Image format (default: jpg)",
    )
    p.add_argument(
        "--quality",
        type=int,
        help="Image quality (jpg/webp: 2-31 lower=better; png: 0-100 higher=better)",
    )
    p.add_argument("--start", type=float, help="Start time in seconds (fast seek)")
    p.add_argument("--duration", type=float, help="Duration in seconds to process")
    p.add_argument("--width", type=int, help="Scale output to width (keeps aspect)")
    p.add_argument("--height", type=int, help="Scale output to height (keeps aspect)")
    p.add_argument("--prefix", help="Output filename prefix (default: input stem)")
    p.add_argument(
        "--start-number",
        type=int,
        default=0,
        help="Start index for image numbering (default: 0)",
    )
    p.add_argument(
        "--keyframes", action="store_true", help="Extract only keyframes (I-frames)"
    )
    p.add_argument(
        "--ffmpeg-arg",
        action="append",
        default=[],
        help="Extra raw ffmpeg arg(s). Can be used multiple times, e.g., --ffmpeg-arg -an",
    )
    # New CLI options
    p.add_argument(
        "--embed-metadata",
        action="store_true",
        help="Embed video_id, frame_index, timestamp_seconds into JPEG EXIF",
    )
    p.add_argument(
        "--video-id", help="Video identifier to embed (default: input file stem)"
    )

    args = p.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    output_dir = build_output_dir(input_path, args.outdir)

    rc = extract_frames(
        input_path=input_path,
        output_dir=output_dir,
        fps=args.fps,
        img_format=args.format,
        quality=args.quality,
        start=args.start,
        duration=args.duration,
        width=args.width,
        height=args.height,
        prefix=args.prefix,
        start_number=args.start_number,
        keyframes_only=args.keyframes,
        extra_ffmpeg_args=args.ffmpeg_arg,
        embed_metadata=args.embed_metadata,
        video_id=args.video_id,
    )

    if rc == 0:
        print(f"Done. Frames written to: {output_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
