"""Detect objects in frames using SAM3 and optionally push results to the feature store.

Usage
-----
uv run python -m footy_track.scripts.detect_objects <frames_folder> \
    [--model_path model_saves/sam3/sam3.pt] \
    [--prompts football player] \
    [--store_path data/feature_store.duckdb] \
    [--game_id arsenal_mancity] \
    [--fps 25.0]

Pass ``--store_path`` to push detections into the feature store after each frame.
``--game_id`` defaults to the frames folder name; ``--fps`` is used to derive
ContinuousTime from the frame index embedded in the filename.
"""

import argparse
import logging
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler
from tqdm import tqdm

from footy_track.detectors.ultralytics import UltralyticsSam3Detector
from footy_track.detectors.utils import visualise_detections_on_image

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Matches filenames like ``<stem>_000042.jpg`` or ``<stem>_frame_000042.jpg``
_FRAME_INDEX_RE = re.compile(r"_(?:frame_)?(\d+)\.[^.]+$")


def _parse_frame_index(filename: str) -> int | None:
    """Extract the frame index from a filename, or None if not parseable."""
    m = _FRAME_INDEX_RE.search(filename)
    return int(m.group(1)) if m else None


def main():
    """Run the SAM3 detector on a folder of frames and print the detection summary."""
    parser = argparse.ArgumentParser(description="Detect objects in frames using SAM3.")
    parser.add_argument(
        "frames_folder",
        type=Path,
        nargs="?",
        default=Path("tests/data/tmp_extracted_frames"),
        help="Path to the folder of frames. Defaults to tests/data/tmp_extracted_frames.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_saves/sam3/sam3.pt",
        help="Path to the SAM3 model.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["football"],
        help="Text prompts for detection. Defaults to ['football'].",
    )
    # Feature store integration
    parser.add_argument(
        "--store_path",
        type=Path,
        default=None,
        help="Path to the feature store DuckDB file. If given, push detections after each frame.",
    )
    parser.add_argument(
        "--game_id",
        type=str,
        default=None,
        help="Game identifier written to the feature store. Defaults to the frames folder name.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames-per-second of the source video; used to derive ContinuousTime from frame index.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame width in pixels (auto-detected from image if omitted).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Frame height in pixels (auto-detected from image if omitted).",
    )
    args = parser.parse_args()

    if not args.frames_folder.exists():
        logging.error(f"Frames folder {args.frames_folder} does not exist.")
        return

    model_input = args.model_path
    if Path(model_input).is_file():
        model_input = Path(model_input)

    detector = UltralyticsSam3Detector(model_input)
    frames = sorted(
        list(args.frames_folder.glob("*.png")) + list(args.frames_folder.glob("*.jpg"))
    )

    if not frames:
        logging.warning(f"No image files found in {args.frames_folder}")
        return

    # Where to save annotated images
    save_root = (
        Path("runs/detect_objects")
        / f"{args.frames_folder.name}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    save_root.mkdir(parents=True, exist_ok=True)

    # Feature store setup (optional)
    store = None
    game_id = args.game_id or args.frames_folder.name
    run_id: str | None = None
    if args.store_path is not None:
        from footy_track.feature_store import FeatureStore
        from footy_track.feature_store.ingest import detector_run
        from footy_track.feature_store.schema import GameRow

        store = FeatureStore.open(args.store_path)
        run_id = f"sam3_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        store.upsert_games([GameRow(game_id=game_id, fps=args.fps)])
        store.upsert_runs([detector_run(run_id, model_name=str(args.model_path), source="sam3")])
        logging.info(f"Feature store: {args.store_path} | game={game_id!r} | run={run_id!r}")

    total_detections = 0
    total_by_label: Counter[str] = Counter()
    for frame in tqdm(frames):
        fd = detector.predict_from_path(frame)
        labels = [d.label for d in fd.detections]
        by_label = Counter(labels)
        num_detections = len(fd.detections)
        total_detections += num_detections
        total_by_label.update(by_label)
        # Save visualisation; do not pop a viewer
        out_path = save_root / frame.name
        visualise_detections_on_image(fd, save_path=out_path, show=False)
        logging.info(
            f"{frame.name}: {num_detections} detections | per-label {dict(by_label)}"
        )

        # Push to feature store if configured
        if store is not None and run_id is not None:
            frame_index = _parse_frame_index(frame.name)
            if frame_index is None:
                logging.warning(f"Cannot parse frame index from {frame.name!r}; skipping store write")
                continue

            from footy_track.feature_store.ingest import ingest_frame

            w = args.width
            h = args.height
            if w is None or h is None:
                try:
                    from PIL import Image  # noqa: PLC0415
                    with Image.open(frame) as im:
                        w, h = im.width, im.height
                except Exception:
                    w = w or 0
                    h = h or 0

            ingest_frame(
                store,
                game_id=game_id,
                frame_index=frame_index,
                frame_uri=str(frame),
                width=w,
                height=h,
                continuous_time_s=frame_index / args.fps,
                detections=fd,
                detection_source="sam3",
                detection_run_id=run_id,
            )

    if store is not None:
        store.close()
        logging.info(f"Feature store written: {args.store_path}")

    logging.info(
        f"Total detections across {len(frames)} frames: {total_detections} | per-label {dict(total_by_label)}"
    )


if __name__ == "__main__":
    main()
