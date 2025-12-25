````markdown
# Scripts

This directory contains small helper scripts used during development and data preparation.

## `split_video.py`
Split a video file into fixed-length chunks using ffmpeg.

- Default chunk length: 10 seconds
- Fast by default using stream copy (`-c copy`) — segment cuts align to nearest keyframe
- Use `--re_encode` to force keyframes at boundaries for more exact segment lengths (slower)
- Output files are written to `<input_stem>_chunks/` (or `--outdir`) using the pattern `<input_stem>_part000.<ext>`

Requirements:
- ffmpeg available on PATH (macOS: `brew install ffmpeg`)

Run with uv run:

```bash
# Split into ~10s chunks (default)
uv run footy-track-split-video /path/to/video.mp4

# Custom chunk size and output directory
uv run footy-track-split-video /path/to/video.mp4 --chunk 5 --outdir ./out

# Re-encode to force exact boundaries (slower), with optional bitrate
uv run footy-track-split-video /path/to/video.mp4 --re_encode --bitrate 4M

# Show help
uv run footy-track-split-video -h
```

## `extract_frames.py`
Extract frames from a video into images using ffmpeg.

- Default: 1 FPS, JPEG images into `<input_stem>_frames/`
- Supports PNG and WEBP, adjustable quality
- Optional resize with `--width/--height`
- Can extract only keyframes with `--keyframes`

Examples:

```bash
# Default: 1 FPS JPEGs next to the video
uv run footy-track-extract-frames /path/to/video.mp4

# Faster sampling and custom output dir
uv run footy-track-extract-frames /path/to/video.mp4 --fps 5 --outdir ./frames

# Start at 30s for 2 minutes
uv run footy-track-extract-frames /path/to/video.mp4 --start 30 --duration 120

# Resize while keeping aspect (use one of width/height or both)
uv run footy-track-extract-frames /path/to/video.mp4 --width 1280

# Different format and quality
uv run footy-track-extract-frames /path/to/video.mp4 --format webp --quality 85

# Only keyframes (I-frames)
uv run footy-track-extract-frames /path/to/video.mp4 --keyframes

# Show help
uv run footy-track-extract-frames -h
```

````
