import argparse
import logging
from collections import Counter
from pathlib import Path

from rich.logging import RichHandler

from footy_track.classifier import UltralyticsClassifier

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)


def main():
    """Run the classifier on a folder of frames and print the breakdown."""
    parser = argparse.ArgumentParser(description="Classify frames in a folder.")
    parser.add_argument(
        "frames_folder", type=Path, help="Path to the folder of frames."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_saves/classifier/20251226-yolo11n-cls/0.987.pt",
        help="Path or name of the classifier model.",
    )
    args = parser.parse_args()

    model_input = args.model_path
    if Path(model_input).is_file():
        model_input = Path(model_input)

    classifier = UltralyticsClassifier(model_input)
    frames = list(args.frames_folder.glob("*.png")) + list(
        args.frames_folder.glob("*.jpg")
    )

    if not frames:
        logging.warning(f"No image files found in {args.frames_folder}")
        return

    classifications = [classifier.predict_from_path(frame) for frame in frames]
    labels = [c.classification.label for c in classifications]
    counts = Counter(labels)

    logging.info(f"Classification summary for {args.frames_folder}:")
    for label, count in counts.items():
        logging.info(f"  {label.value}: {count}")


if __name__ == "__main__":
    main()
