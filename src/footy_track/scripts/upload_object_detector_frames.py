"""
This script uploads a directory of images to a Roboflow project for object detection.

It allows for sampling a subset of images and assigns them to a specific batch.
The main purpose is to facilitate the process of adding new images to a Roboflow
dataset for annotation and training.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from footy_track import constants, labelling

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
_logger = logging.getLogger(__name__)


def get_image_paths(frames_dir: Path, num_samples: int | None) -> list[Path]:
    """
    Gets a list of image paths from a directory, with an option to sample them evenly.

    Args:
        frames_dir (Path): The directory containing the image files.
        num_samples (int | None): The number of images to sample. If None, all images are returned.

    Returns:
        list[Path]: A list of paths to the selected images.
    """
    _logger.info(f"Searching for images in: {frames_dir}")
    all_image_files = sorted(frames_dir.glob(f"*.{constants.IMAGE_FORMAT}"))

    if not all_image_files:
        _logger.warning("No images found in the specified directory.")
        return []

    if num_samples is None or num_samples <= 0 or num_samples >= len(all_image_files):
        _logger.info(f"Using all {len(all_image_files)} images.")
        return all_image_files

    _logger.info(
        f"Sampling {num_samples} images evenly from {len(all_image_files)} total images."
    )
    total_files = len(all_image_files)
    if num_samples > total_files:
        return all_image_files

    # Generate evenly spaced indices to sample images across the whole set
    indices = [
        round(i * (total_files - 1) / (num_samples - 1)) for i in range(num_samples)
    ]
    unique_indices = sorted(set(indices))
    return [all_image_files[i] for i in unique_indices]


def main(
    frames_dir: Path,
    num_samples: int | None,
    batch_name: str,
    batch_size: int | None = None,
) -> None:
    """
    Main function to handle the process of uploading frames to Roboflow.

    Args:
        frames_dir (Path): Directory containing the video frames.
        num_samples (int | None): Number of images to sample.
        batch_name (str): Name of the batch for the upload in Roboflow.
        batch_size (int): The size of batches to upload.
    """
    if not frames_dir.exists():
        _logger.error(f"Frames directory not found: {frames_dir}")
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    _logger.info(f"Processing frames from: {frames_dir}")

    image_paths = get_image_paths(frames_dir, num_samples)

    if batch_size is None or batch_size <= 0:
        batch_size = len(image_paths)

    if not image_paths:
        _logger.info("No images to upload.")
        return

    # Initialize the Roboflow handler
    handler = labelling.RoboflowObjectDetectionHandler()
    model_name = handler.detector_name
    current_time = datetime.now().strftime("%Y-%m-%d_%H%M")
    batch_name = f"{batch_name}__{model_name}__{current_time}"

    _logger.info(f"Uploading {len(image_paths)} images to Roboflow...")
    num_images = len(image_paths)
    for i in tqdm(range(0, num_images, batch_size), desc="Uploading batches"):
        sub_batch_paths = image_paths[i : i + batch_size]
        sub_batch_name = f"{batch_name}__batch_{i // batch_size + 1}"
        _logger.info(
            f"Uploading {len(sub_batch_paths)} images to Roboflow with batch name '{sub_batch_name}'..."
        )
        handler.upload_images(image_paths=sub_batch_paths, batch_name=sub_batch_name)

    _logger.info("Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload frames to Roboflow for object detection labelling."
    )
    parser.add_argument(
        "frames_dir",
        type=str,
        help="Directory containing the video frames to upload.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of images to sample. If not provided, all images will be uploaded.",
    )
    parser.add_argument(
        "--batch_name",
        type=str,
        default="script_upload",
        help="Name of the batch for the upload in Roboflow.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of images per batch for uploading.",
    )
    args = parser.parse_args()

    main(
        frames_dir=Path(args.frames_dir),
        num_samples=args.num_samples,
        batch_name=args.batch_name,
        batch_size=args.batch_size,
    )
