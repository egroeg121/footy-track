import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from rich.logging import RichHandler

from footy_track.detectors.ultralytics import UltralyticsSam3Detector

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

    total_detections = 0
    for frame in tqdm(frames):
        detections = detector.predict_from_path(frame)
        num_detections = len(detections.detections)
        total_detections += num_detections
        logging.info(f"{frame.name}: {num_detections} detections")

    logging.info(f"Total detections across {len(frames)} frames: {total_detections}")


if __name__ == "__main__":
    main()
