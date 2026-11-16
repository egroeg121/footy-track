#!/usr/bin/env python3
"""Make a 1080p25 SDR (bt709) proxy of a match video, tone-mapping HDR sources.

The training matches are 1080p25 (or 720p25) SDR. A 4K50 HDR10/PQ broadcast
(e.g. ``arsenal_nottinghamforest.mkv``: HEVC Main10, bt2020nc, smpte2084) looks washed out if
simply scaled, and is 8x the pixels and 2x the frames, so it is first converted to a
proxy that matches the existing data, then handed to ``split_broadcast_segments``.

Frame indices of every clip cut from the proxy refer to the PROXY (25 fps). The proxy is
cut from the source by dropping every other frame (``fps=25`` on 50p), so proxy frame
``n`` is source frame ``2n`` and time ``n / 25 s`` is the same in both.

Usage:
  uv run python -m footy_track.scripts.make_sdr_proxy SRC.mkv DST.mp4 [--height 1080] [--fps 25]

Writes ``DST.tmp.mp4`` then renames, so an interrupted run never leaves a partial DST.
Requires ffmpeg with zscale (libzimg). No audio is kept.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}

# scale first (cheap at 10 bit, 4x fewer pixels), then linear-light hable tone-map to bt709.
TONEMAP = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)


def probe(src: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,color_transfer,pix_fmt", "-of", "json", str(src)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)["streams"][0]


def build_filter(info: dict, height: int, fps: int) -> str:
    chain = [f"fps={fps}", f"scale=-2:{height}:flags=bicubic"]
    if info.get("color_transfer") in HDR_TRANSFERS:
        chain.append(TONEMAP)
    else:
        chain.append("format=yuv420p")
    return ",".join(chain)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--threads", type=int, default=0, help="ffmpeg threads (0 = all)")
    args = ap.parse_args(argv)
    if args.dst.exists():
        print(f"{args.dst} exists; not overwriting", file=sys.stderr)
        return 0
    info = probe(args.src)
    vf = build_filter(info, args.height, args.fps)
    print(f"source {info}\nfilter {vf}", flush=True)
    tmp = args.dst.with_name(args.dst.stem + ".tmp.mp4")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error", "-stats",
           "-threads", str(args.threads), "-i", str(args.src), "-map", "0:v:0", "-an",
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
           "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-colorspace", "bt709", "-movflags", "+faststart", str(tmp)]
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        tmp.unlink(missing_ok=True)
        return rc
    os.replace(tmp, args.dst)
    print(f"wrote {args.dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
