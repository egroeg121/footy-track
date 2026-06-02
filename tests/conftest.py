"""Pytest fixtures for object detector tests.

Provides repository root, sample image path, and detector instances. Skips
cleanly if the sample image or optional `transformers` dependency is missing.
"""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from footy_track.constants import ROBOFLOW_BROADCAST_PROJECT_TEST_PROJECT
from footy_track.scripts import extract_frames


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def video_path(repo_root: Path) -> Path:
    """Fixture for the video file path."""
    path = (
        repo_root / "tests" / "data" / "video" / "arsenal_mancity_20250925_part192.mp4"
    )
    if not path.exists():
        pytest.skip(f"Video file not found at {path}")
    return path


@pytest.fixture(scope="session")
def frames_path(repo_root: Path) -> Path:
    """Fixture for the frames directory path."""
    path = repo_root / "tests/data/tmp_extracted_frames"
    if not path.exists():
        pytest.skip(f"Frames directory not found at {path}")
    return path


@pytest.fixture(scope="session", autouse=True)
def extracted_frames(repo_root: Path) -> list[Path]:
    """Pre-extract video frames if the source video exists; no-op otherwise."""
    video = (
        repo_root / "tests" / "data" / "video" / "arsenal_mancity_20250925_part192.mp4"
    )
    frames_dir = repo_root / "tests/data/tmp_extracted_frames"

    if not video.exists():
        return []

    if frames_dir.exists() and any(frames_dir.iterdir()):
        return list(frames_dir.glob("*.png"))

    extract_frames.extract_frames(
        input_path=video,
        output_dir=frames_dir,
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
    return list(frames_dir.glob("*.png"))


@pytest.fixture(scope="session", autouse=True)
def random_seed():
    """Fixture to set a fixed random seed for tests."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def roboflow_test_project_name() -> str:
    """Returns the name of the test roboflow broadcast project."""
    return ROBOFLOW_BROADCAST_PROJECT_TEST_PROJECT
