"""Track objects in a video using UltralyticsTracker (ByteTrack/BoT-SORT).

Reads a video file frame-by-frame, runs YOLO detection + tracking on each frame,
writes annotated frames to a run directory, and saves tracks.parquet +
tracks_meta.json via TrackingWriter.

Usage
-----
uv run python -m footy_track.scripts.track_objects <video_path> --model_path model_saves/detector/best.pt
"""

import argparse
import logging
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from rich.logging import RichHandler
from tqdm import tqdm

from footy_track.detectors.ultralytics import CURRENT_BEST_DETECTOR_CHECKPOINT
from footy_track.trackers import TrackingWriter, UltralyticsTracker
from footy_track.utils import get_project_root

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Colour palette per label (BGR for OpenCV)
_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "player": (255, 144, 30),
    "referee": (225, 105, 65),
    "coach": (237, 149, 100),
    "ball": (0, 200, 255),
    "in_play_ball": (0, 215, 255),
    "out_of_play_ball": (0, 140, 255),
    "player_sub": (180, 130, 70),
    "person": (180, 130, 70),
}
_DEFAULT_COLOR = (128, 0, 128)


def _label_color(label: str) -> tuple[int, int, int]:
    return _LABEL_COLORS.get(label.lower(), _DEFAULT_COLOR)


def _draw_tracked_frame(frame, tracked_detections) -> None:
    """Draw bounding boxes + track IDs onto *frame* in-place (BGR)."""
    h, w = frame.shape[:2]
    for td in tracked_detections:
        x1 = int(td.x * w)
        y1 = int(td.y * h)
        x2 = int((td.x + td.w) * w)
        y2 = int((td.y + td.h) * h)
        color = _label_color(td.label)
        thickness = max(2, int(min(w, h) * 0.003))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text = f"#{td.track_id} {td.label} {td.confidence:.2f}"
        font_scale = max(0.4, min(w, h) * 0.0008)
        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )


def _run_tracking(
    tracker: UltralyticsTracker,
    writer: TrackingWriter,
    video_path: Path,
    run_dir: Path,
    fps: float,
    total_frames: int,
    vid_w: int,
    vid_h: int,
    save_video: bool,
) -> int:
    """Process video frames, annotate, and buffer detections. Returns frame count."""
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(run_dir / "tracks_overlay.mp4"), fourcc, fps, (vid_w, vid_h)
        )

    total_by_label: Counter[str] = Counter()
    frame_idx = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        pbar = tqdm(total=total_frames or None, unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tmp_path = Path(tmpdir) / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(tmp_path), frame)

            frame_t = frame_idx / fps
            tracked = tracker.update_from_path(tmp_path, frame_t)

            for td in tracked:
                writer.write(td)
            by_label = Counter(td.label for td in tracked)
            total_by_label.update(by_label)

            _draw_tracked_frame(frame, tracked)
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), frame)
            if video_writer is not None:
                video_writer.write(frame)

            logging.info(
                f"Frame {frame_idx:4d} | t={frame_t:.2f}s | "
                f"{len(tracked)} tracked | {dict(by_label)}"
            )
            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    if video_writer is not None:
        video_writer.release()

    logging.info(f"Total per-label: {dict(total_by_label)}")
    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track objects in a video using YOLO + ByteTrack."
    )
    parser.add_argument(
        "video_path",
        type=Path,
        nargs="?",
        default=Path("data/arsenal_mancity_test_15frames.mp4"),
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=get_project_root() / CURRENT_BEST_DETECTOR_CHECKPOINT,
        help="Path to the YOLO checkpoint (.pt).",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracker algorithm config.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.3,
        help="Minimum detection confidence threshold.",
    )
    parser.add_argument(
        "--match_id",
        type=str,
        default=None,
        help="Match identifier written into tracks_meta.json. Defaults to the video stem.",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Also write an annotated output video (tracks_overlay.mp4).",
    )
    args = parser.parse_args()

    if not args.video_path.exists():
        logging.error(f"Video not found: {args.video_path}")
        return
    if not args.model_path.exists():
        logging.error(f"Model checkpoint not found: {args.model_path}")
        return

    match_id = args.match_id or args.video_path.stem
    run_dir = (
        Path("runs/track_objects")
        / f"{args.video_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO  # noqa: PLC0415

    _probe = YOLO(str(args.model_path))
    classes: dict[int, str] = dict(_probe.names)
    logging.info(f"Model classes: {classes}")
    del _probe

    tracker = UltralyticsTracker(
        model_uri=str(args.model_path),
        tracker=args.tracker,
        classes=classes,
        min_confidence=args.min_confidence,
        verbose=False,
    )
    writer = TrackingWriter()

    cap = cv2.VideoCapture(str(args.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_count = _run_tracking(
        tracker,
        writer,
        args.video_path,
        run_dir,
        fps,
        total_frames,
        vid_w,
        vid_h,
        args.save_video,
    )

    meta = tracker.finalise()
    parquet_path, meta_path = writer.finalise(
        output_dir=run_dir,
        match_id=match_id,
        meta=meta,
        detector=str(args.model_path),
        tracker_name=args.tracker,
        fps=fps,
        width=vid_w,
        height=vid_h,
    )

    logging.info(f"Frames processed : {frame_count}")
    logging.info(f"Unique tracks     : {len(meta)}")
    logging.info(f"Parquet           : {parquet_path}")
    logging.info(f"Meta              : {meta_path}")
    logging.info(f"Annotated frames  : {run_dir / 'frames'}")
    if args.save_video:
        logging.info(f"Annotated video   : {run_dir / 'tracks_overlay.mp4'}")


if __name__ == "__main__":
    main()
