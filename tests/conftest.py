"""Pytest fixtures for object detector tests.

Provides repository root, sample image path, and detector instances. Skips
cleanly if the sample image or optional `transformers` dependency is missing.
"""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from footy_track.scripts import extract_frames


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def video_path(repo_root: Path) -> Path:
    """Fixture for the video file path."""
    path = repo_root / "tests" / "data" / "video" / "arsenal_mancity_20250925_part192.mp4"
    assert path.exists(), f"Video file not found at {path}"
    return path


@pytest.fixture(scope="session")
def frames_path(repo_root: Path) -> Path:
    """Fixture for the frames directory path."""
    path = repo_root / "tests/data/tmp_extracted_frames"
    assert path.exists(), f"Frames directory not found at {path}"
    return path


@pytest.fixture(scope="session", autouse=True)
def extracted_frames(video_path: Path, frames_path: Path) -> list[Path]:
    """Fixture for the frames directory path."""

    if frames_path.exists() and any(frames_path.iterdir()):
        # Frames already extracted
        frames = list(frames_path.glob("*.png"))
        return frames

    # Extract frames at 1 FPS
    extract_frames.extract_frames(
        input_path=video_path,
        output_dir=frames_path,
        fps=1,
        img_format="png",
        quality=None,
        start=None,
        duration=None,
        width=None,
        height=None,
        prefix=None,
        start_number=0,
        keyframes_only=False,
        extra_ffmpeg_args=[],
    )

    # Check that 30 frames were created
    frames = list(frames_path.glob("*.png"))
    return frames


@pytest.fixture(scope="session", autouse=True)
def random_seed():
    """Fixture to set a fixed random seed for tests."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def roboflow_test_project_name() -> str:
    """Returns the name of the test roboflow broadcast project."""
    return "footy-track-broadcast-frame-test"
