import argparse
import logging
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from rich.logging import RichHandler

from footy_track.detectors.ultralytics import UltralyticsSam3Detector
from footy_track.detectors.utils import visualise_detections_on_image

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)


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
    args = parser.parse_args()

    if not args.frames_folder.exists():
        logging.error(f"Frames folder {args.frames_folder} does not exist.")
        return

    model_input = args.model_path
    if Path(model_input).is_file():
        model_input = Path(model_input)

    detector = UltralyticsSam3Detector(model_input)
    frames = list(args.frames_folder.glob("*.png")) + list(
        args.frames_folder.glob("*.jpg")
    )

    if not frames:
        logging.warning(f"No image files found in {args.frames_folder}")
        return

    # Where to save annotated images
    save_root = Path("runs/detect_objects") / args.frames_folder.name
    save_root.mkdir(parents=True, exist_ok=True)

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

    logging.info(
        f"Total detections across {len(frames)} frames: {total_detections} | per-label {dict(total_by_label)}"
    )


if __name__ == "__main__":
    main()
