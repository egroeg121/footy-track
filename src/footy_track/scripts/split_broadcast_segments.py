#!/usr/bin/env python3
"""Split a full match video into broadcast-only segments.

Instead of fixed-length chunks, this runs the best broadcast-frame classifier
over the video and cuts it into contiguous runs of *broadcast* frames (where the
classifier says "Yes"), dropping the non-broadcast frames (replays, crowd,
studio, ads). Each contiguous broadcast run is written as its own MP4, plus a
manifest JSON describing the segments.

Sampling: classifying every frame of a full match is slow, so by default we
classify every Nth frame (``--sample 5``). Whenever a sample's label differs from
the previous sample's, the whole interval between them is classified frame-by-
frame to pin the *exact* transition frame(s), so the cut boundaries stay frame-
accurate even with coarse sampling. Frames inside stable intervals inherit the
last sampled label. Use ``--sample 1`` to classify every frame unconditionally.

Smoothing: brief misclassifications are bridged — non-broadcast gaps shorter
than ``--merge-gap-s`` are absorbed into the surrounding broadcast run, and
segments shorter than ``--min-seg-s`` are dropped.

Usage:
  uv run python -m footy_track.scripts.split_broadcast_segments match.mp4
  uv run python -m footy_track.scripts.split_broadcast_segments match.mp4 \
      --sample 1 --merge-gap-s 0.5 --min-seg-s 2.0 --outdir ./segments

Requires: ffmpeg on PATH (used to cut the segments).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from rich.logging import RichHandler
from tqdm import tqdm

from footy_track.classifier import get_current_best_guess_classifier
from footy_track.schema import EnumBroadcastClassification

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def _check_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.error("ffmpeg not found on PATH (macOS: brew install ffmpeg).")
        sys.exit(1)
    return ffmpeg


def _classify_frame(model, frame: np.ndarray) -> bool:
    """Return True if the classifier labels this frame as broadcast."""
    result = model.predict(source=frame, device="mps", verbose=False)[0]
    label = model.names[result.probs.top1]
    return label == EnumBroadcastClassification.YES.value


def classify_broadcast_mask(
    video_path: Path, sample: int
) -> tuple[np.ndarray, float, int]:
    """Classify the video and return (per-frame broadcast mask, fps, total_frames).

    Done in a single forward decode pass (no random seeks — important for ``.ts``
    sources, which have no clean frame index). Every ``sample``-th frame is
    classified. When a sample's label differs from the previous sample's, the
    entire interval between them is classified frame-by-frame so the exact flip
    frame(s) are found — this stays robust even if the classifier flickers
    (multiple transitions) inside a single interval. Frames in stable intervals
    inherit the last classified label.

    The classifier's YOLO model is fed numpy frames directly to avoid writing a
    temp image per frame.
    """
    classifier = get_current_best_guess_classifier()
    model = classifier.model  # underlying ultralytics YOLO classification model

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Buffer the frames of the current (not-yet-on-a-sample-boundary) interval so
    # that, if the upcoming sample reveals a label change, we can go back and
    # classify each buffered frame — all from this single forward pass.
    mask_list: list[bool] = []
    buffer: list[np.ndarray] = []  # frames since the last sample boundary
    prev_sample_label = False  # label of the most recent sample boundary
    frame_idx = 0

    pbar = tqdm(total=total or None, unit="frame", desc="Classifying")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample == 0:
            cur_label = _classify_frame(model, frame)
            if buffer:
                # Flush the preceding interval [prev_sample .. this_sample).
                if cur_label == prev_sample_label:
                    # No change across the interval: every in-between frame
                    # inherits that stable label (one classification for many).
                    mask_list.extend([prev_sample_label] * len(buffer))
                else:
                    # Change detected: classify each buffered frame to pin the
                    # exact transition frame(s) within the interval.
                    mask_list.extend(_classify_frame(model, f) for f in buffer)
                buffer = []
            mask_list.append(cur_label)
            prev_sample_label = cur_label
        else:
            buffer.append(frame)
        frame_idx += 1
        pbar.update(1)
    # Trailing frames after the final sample boundary inherit its label.
    if buffer:
        mask_list.extend([prev_sample_label] * len(buffer))
    pbar.close()
    cap.release()

    mask = np.array(mask_list, dtype=bool)
    return mask, fps, len(mask_list)


def _mask_to_segments(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Return [(start_frame, end_frame_inclusive), ...] for runs of True."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def smooth_mask(
    mask: np.ndarray, fps: float, merge_gap_s: float, min_seg_s: float
) -> list[tuple[int, int]]:
    """Bridge short non-broadcast gaps and drop short segments.

    Returns the final list of (start_frame, end_frame_inclusive) segments.
    """
    merge_gap_frames = int(round(merge_gap_s * fps))
    min_seg_frames = int(round(min_seg_s * fps))

    segments = _mask_to_segments(mask)
    if not segments:
        return []

    # Bridge gaps: merge two broadcast runs separated by a short non-broadcast gap.
    merged: list[tuple[int, int]] = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1
        if gap <= merge_gap_frames:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Drop segments shorter than the minimum length.
    kept = [(s, e) for (s, e) in merged if (e - s + 1) >= min_seg_frames]
    return kept


def cut_segment(
    ffmpeg: str,
    input_path: Path,
    out_path: Path,
    start_s: float,
    duration_s: float,
) -> int:
    """Cut [start_s, start_s+duration_s) from the source, re-encoding for exact cuts."""
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration_s:.3f}",
        "-map",
        "0",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(out_path),
    ]
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split a video into broadcast-only segments via the classifier."
    )
    parser.add_argument("input", type=Path, help="Path to the input match video.")
    parser.add_argument(
        "--outdir",
        "-o",
        type=Path,
        default=None,
        help="Output dir (default: <input_stem>_broadcast next to input).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help=(
            "Classify every Nth frame (default 5; try 24 for ~1/sec). "
            "Intervals where the label changes are refined frame-by-frame, so "
            "boundaries stay exact. Use 1 to classify every frame."
        ),
    )
    parser.add_argument(
        "--merge-gap-s",
        type=float,
        default=0.5,
        help="Bridge non-broadcast gaps shorter than this many seconds (default 0.5).",
    )
    parser.add_argument(
        "--min-seg-s",
        type=float,
        default=2.0,
        help="Drop broadcast segments shorter than this many seconds (default 2.0).",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        return 2
    if args.sample < 1:
        logger.error("--sample must be >= 1")
        return 2

    ffmpeg = _check_ffmpeg()
    outdir = args.outdir or args.input.parent / f"{args.input.stem}_broadcast"
    outdir.mkdir(parents=True, exist_ok=True)

    mask, fps, total = classify_broadcast_mask(args.input, args.sample)
    if total == 0:
        logger.error("No frames read from video.")
        return 1

    broadcast_frames = int(mask.sum())
    logger.info(
        f"Classified {total} frames @ {fps:.2f} fps "
        f"(sample every {args.sample}): {broadcast_frames} broadcast "
        f"({100 * broadcast_frames / total:.1f}%)."
    )

    segments = smooth_mask(mask, fps, args.merge_gap_s, args.min_seg_s)
    logger.info(f"{len(segments)} broadcast segment(s) after smoothing.")

    manifest: list[dict] = []
    for i, (start_f, end_f) in enumerate(
        tqdm(segments, unit="segment", desc="Cutting")
    ):
        start_s = start_f / fps
        # end_f is inclusive; duration spans through the end of that frame.
        duration_s = (end_f - start_f + 1) / fps
        out_path = outdir / f"{args.input.stem}_seg{i:03d}.mp4"
        rc = cut_segment(ffmpeg, args.input, out_path, start_s, duration_s)
        if rc != 0:
            logger.warning(f"ffmpeg failed (rc={rc}) for segment {i}; skipping.")
            continue
        manifest.append(
            {
                "index": i,
                "file": out_path.name,
                "start_frame": start_f,
                "end_frame": end_f,
                "frame_count": end_f - start_f + 1,
                "start_s": round(start_s, 3),
                "end_s": round((end_f + 1) / fps, 3),
                "duration_s": round(duration_s, 3),
            }
        )

    manifest_path = outdir / "segments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": str(args.input),
                "fps": fps,
                "total_frames": total,
                "sample": args.sample,
                "merge_gap_s": args.merge_gap_s,
                "min_seg_s": args.min_seg_s,
                "broadcast_frames": broadcast_frames,
                "segments": manifest,
            },
            indent=2,
        )
    )

    logger.info(f"Wrote {len(manifest)} segment(s) → {outdir}")
    logger.info(f"Manifest → {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
