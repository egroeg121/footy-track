"""Simple end-to-end inference: broadcast video -> tracked players and ball.

    python -m footy_track.scripts.run_inference --video clip.mp4 --out runs/clip

Steps: read frames (OpenCV) -> detect (RT-DETR via ``UltralyticsObjectDetector``,
imgsz 1280, conf >= 0.1, nothing filtered) -> track players/referees with
``LapTracker`` (through ``scripts.track_detections``, same defaults) -> track the
ball with ``DetectionBallTracker`` -> write everything to ``--out``.

Outputs (all JSONL, one row per line; ``frame_index`` is the index in the source
video, so ``--start`` / ``--stride`` runs line up with the video):

``detections.jsonl``
    Machine-label format, as ``machine_labels/*.jsonl`` and readable by the same
    tools: ``{frame_index, bbox{x,y,w,h}, center{x,y}, tags:[label, "rtdetr"],
    confidence, model_id, run_id, reviewed}``. Every detection is kept.
``tracklets.jsonl``
    Players and referees, the ``scripts.track_detections`` format that
    ``/identity/tracklets`` reads: ``{frame_index, track_id, tags:[label, "lap"],
    bbox, confidence, is_interpolated}``.
``ball_track.jsonl``
    The only new format. One row per processed frame: ``{frame_index, track_id,
    source, bbox, center, confidence}``. ``source`` is ``detected``,
    ``interpolated`` (gap bridged between two detections; ``confidence`` null) or
    ``none`` (no ball; ``bbox``, ``center`` and ``track_id`` null). bbox/center are
    normalised dicts like the labeller sidecars.
``summary.json``
    Frame counts, fps, detection and track counts, ball coverage, timings, the run
    config, checkpoint and git SHA.
``overlay.mp4``
    Optional (``overlay=True``): boxes, track ids and the ball with a short trail.

Not handled yet: broadcast cuts (tracks and the ball track just break at a cut)
and any projection to pitch coordinates.
"""

from __future__ import annotations

import collections
import json
import subprocess
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch

from footy_track.ball_trackers.detection_tracker import (
    DETECTED,
    INTERPOLATED,
    BallCandidate,
    BallPoint,
    DetectionBallTracker,
)
from footy_track.constants import BALL_TAG, IN_PLAY_BALL_TAG
from footy_track.schema import FrameDetections
from footy_track.scripts.track_detections import (
    DEFAULT_LABELS,
    load_detections,
    summarise,
    track_clip,
)
from footy_track.utils import get_project_root

BALL_LABELS = (IN_PLAY_BALL_TAG, BALL_TAG)


class FrameDetector(Protocol):
    """What the pipeline needs from a detector: BGR frames in, detections out."""

    def predict_frames(self, frames: list[np.ndarray]) -> list[FrameDetections]: ...


@dataclass
class InferenceConfig:
    """Everything that shapes a run; written verbatim to ``summary.json``."""

    start: int = 0
    max_frames: int | None = None
    stride: int = 1
    batch_size: int = 8
    imgsz: int = 1280
    detect_conf: float = 0.1  # keep everything; the trackers decide what to trust
    # players / referees: scripts.track_detections measured defaults
    player_conf: float = 0.5
    track_iou: float = 0.15
    track_max_age: int = 90
    # ball: see DetectionBallTracker for what each threshold does
    ball_start_conf: float = 0.6
    ball_jump_conf: float = 0.8
    ball_base_radius_px: float = 40.0
    ball_growth_px: float = 25.0
    ball_max_missed: int = 12
    overlay: bool = False


def select_device() -> str:
    """cuda, then mps, then cpu.

    Not ``detectors.utils._available_device`` (mps first) and not the training
    script's ``--device mps`` default, which fails on every non-Apple machine.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def iter_frames(
    cap: cv2.VideoCapture, start: int, max_frames: int | None, stride: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` for every ``stride``-th frame from ``start``."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    index, yielded = start, 0
    while max_frames is None or yielded < max_frames:
        ok, frame = cap.read()
        if not ok:
            return
        if (index - start) % stride == 0:
            yield index, frame
            yielded += 1
        index += 1


def detection_rows(
    frame_index: int,
    frame: FrameDetections,
    *,
    source_tag: str,
    model_id: str,
    run_id: str,
) -> list[dict]:
    """Machine-label rows for one frame (the ``machine_labels/*.jsonl`` schema)."""
    return [
        {
            "frame_index": frame_index,
            "bbox": {"x": d.x, "y": d.y, "w": d.w, "h": d.h},
            "center": {"x": d.x + d.w / 2, "y": d.y + d.h / 2},
            "tags": [d.label, source_tag],
            "confidence": round(d.confidence, 4),
            "model_id": model_id,
            "run_id": run_id,
            "reviewed": False,
        }
        for d in frame.detections
    ]


def ball_candidates(frame: FrameDetections) -> list[BallCandidate]:
    return [
        BallCandidate(d.x, d.y, d.w, d.h, d.confidence)
        for d in frame.detections
        if d.label in BALL_LABELS
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _ball_coverage(points: list[BallPoint]) -> dict:
    counts = collections.Counter(p.source for p in points)
    n = max(len(points), 1)
    return {
        "frames": len(points),
        "detected": counts[DETECTED],
        "interpolated": counts[INTERPOLATED],
        "none": len(points) - counts[DETECTED] - counts[INTERPOLATED],
        "coverage": round((counts[DETECTED] + counts[INTERPOLATED]) / n, 4),
        "tracks": len({p.track_id for p in points if p.track_id is not None}),
    }


def write_overlay(
    video: Path,
    out_path: Path,
    frame_indices: list[int],
    tracklets: list[dict],
    ball: list[BallPoint],
    fps: float,
    trail: int = 15,
) -> None:
    """Re-read the processed frames and draw players, referees and the ball."""
    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    by_frame: dict[int, list[dict]] = collections.defaultdict(list)
    for row in tracklets:
        by_frame[row["frame_index"]].append(row)
    ball_by_frame = {p.frame_index: p for p in ball}
    wanted = set(frame_indices)
    history: list[tuple[int, int]] = []
    for index, frame in iter_frames(cap, frame_indices[0], None, 1):
        if index not in wanted:
            continue
        for row in by_frame[index]:
            b = row["bbox"]
            p1 = (int(b["x"] * width), int(b["y"] * height))
            p2 = (int((b["x"] + b["w"]) * width), int((b["y"] + b["h"]) * height))
            colour = (0, 165, 255) if row["tags"][0] == "referee" else (255, 144, 30)
            cv2.rectangle(frame, p1, p2, colour, 2)
            cv2.putText(
                frame, str(row["track_id"]), (p1[0], p1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
            )  # fmt: skip
        point = ball_by_frame[index]
        if point.x is None:
            history = []
        else:
            centre = (
                int((point.x + point.w / 2) * width),
                int((point.y + point.h / 2) * height),
            )
            history = (history + [centre])[-trail:]
            colour = (0, 255, 255) if point.source == DETECTED else (0, 200, 0)
            cv2.circle(frame, centre, 14, colour, 2)
        for a, b in zip(history, history[1:], strict=False):
            cv2.line(frame, a, b, (0, 255, 255), 2)
        writer.write(frame)
    writer.release()
    cap.release()


def run_inference(
    video: Path,
    out_dir: Path,
    detector: FrameDetector,
    config: InferenceConfig | None = None,
    *,
    checkpoint: str = "unknown",
    device: str = "unknown",
    source_tag: str = "rtdetr",
) -> dict:
    """Run the whole pipeline on ``video`` and write its outputs to ``out_dir``.

    Returns the ``summary.json`` dict.
    """
    config = config or InferenceConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model_id = Path(checkpoint).parent.name or Path(checkpoint).stem
    run_id = f"{model_id}__{datetime.now():%Y%m%d-%H%M%S}"
    ball_tracker = DetectionBallTracker(
        frame_size=(width, height),
        start_conf=config.ball_start_conf,
        jump_conf=config.ball_jump_conf,
        base_radius_px=config.ball_base_radius_px,
        growth_px=config.ball_growth_px,
        max_missed=config.ball_max_missed,
    )

    t0 = time.time()
    rows: list[dict] = []
    frame_indices: list[int] = []
    batch: list[tuple[int, np.ndarray]] = []

    def flush() -> None:
        results = detector.predict_frames([f for _, f in batch])
        for (index, _), result in zip(batch, results, strict=True):
            rows.extend(
                detection_rows(
                    index,
                    result,
                    source_tag=source_tag,
                    model_id=model_id,
                    run_id=run_id,
                )
            )
            ball_tracker.update(index, ball_candidates(result))
            frame_indices.append(index)
        batch.clear()

    for item in iter_frames(cap, config.start, config.max_frames, config.stride):
        batch.append(item)
        if len(batch) >= config.batch_size:
            flush()
    if batch:
        flush()
    cap.release()
    detect_s = time.time() - t0

    detections_path = out_dir / "detections.jsonl"
    _write_jsonl(detections_path, rows)

    # Players and referees: re-read what we just wrote, exactly as
    # scripts.track_detections does for machine_labels/, so both paths agree.
    by_frame = load_detections(
        detections_path, min_confidence=config.player_conf, keep_labels=DEFAULT_LABELS
    )
    tracklets = track_clip(
        by_frame,
        iou_threshold=config.track_iou,
        max_age=config.track_max_age,
        fps=fps,
        width=width,
        height=height,
    )
    _write_jsonl(out_dir / "tracklets.jsonl", tracklets)

    ball = ball_tracker.points()
    _write_jsonl(out_dir / "ball_track.jsonl", [p.to_row() for p in ball])

    if config.overlay and frame_indices:
        write_overlay(
            video, out_dir / "overlay.mp4", frame_indices, tracklets, ball,
            fps / config.stride,
        )  # fmt: skip

    summary = {
        "video": str(video),
        "video_fps": fps,
        "video_frames": total_frames,
        "frames_processed": len(frame_indices),
        "detections": len(rows),
        "detections_by_label": dict(collections.Counter(r["tags"][0] for r in rows)),
        "players": summarise(tracklets),
        "ball": _ball_coverage(ball),
        "detect_seconds": round(detect_s, 2),
        "detect_fps": round(len(frame_indices) / max(detect_s, 1e-6), 2),
        "total_seconds": round(time.time() - t0, 2),
        "checkpoint": checkpoint,
        "model_id": model_id,
        "run_id": run_id,
        "device": device,
        "git_sha": _git_sha(),
        "config": asdict(config),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
