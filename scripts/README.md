# Scripts

This directory contains small helper scripts used during development and data preparation.

## `split_video.py`
Split a video file into fixed-length chunks using ffmpeg.

- Default chunk length: 10 seconds
- Fast by default using stream copy (`-c copy`) — segment cuts align to nearest keyframe
- Use `--reencode` to force keyframes at boundaries for more exact segment lengths (slower)
- Output files are written to `<input_stem>_chunks/` (or `--outdir`) using the pattern `<input_stem>_part000.<ext>`

Requirements:
- ffmpeg available on PATH (macOS: `brew install ffmpeg`)

Run with uvx (recommended):

```bash
# Split into ~10s chunks (default)
uvx scripts/split_video.py /path/to/video.mp4

# Custom chunk size and output directory
uvx scripts/split_video.py /path/to/video.mp4 --chunk 5 --outdir ./out

# Re-encode to force exact boundaries (slower), with optional bitrate
uvx scripts/split_video.py /path/to/video.mp4 --reencode --bitrate 4M

# Show help
uvx scripts/split_video.py -h
```
