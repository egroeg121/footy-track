"""Pytest fixtures for object detector tests.

Provides repository root, sample image path, and detector instances. Skips
cleanly if the sample image or optional `transformers` dependency is missing.
"""

import os
from pathlib import Path
import pytest

from footy_track.object_detections import (
    UltralyticsObjectDetector,
    GroundingDinoObjectDetector,
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def image_path(repo_root: Path) -> Path:
    path = repo_root / "data" / "arsenal_mancity_frames_1fps" / "arsenal_mancity_20250925_000629.jpg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return path


@pytest.fixture(scope="session")
def ultralytics_detector() -> UltralyticsObjectDetector:
    # Prefer local model file if present
    model_path = "yolo11n.pt"
    if not Path(model_path).exists():
        # Allow Ultralytics to download if missing
        pass
    return UltralyticsObjectDetector(model_uri=model_path, verbose=False, compile=False)


@pytest.fixture(scope="session")
def grounding_dino_detector():
    # Skip if transformers not installed (dev dependency)
    transformers = pytest.importorskip("transformers")
    return GroundingDinoObjectDetector(
        model_id="IDEA-Research/grounding-dino-tiny",
        box_threshold=0.40,
        text_threshold=0.30,
        prompts=("soccer ball", "football", "ball"),
    )
