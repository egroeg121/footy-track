"""Pytest fixtures for object detector tests.

Provides repository root, sample image path, and detector instances. Skips
cleanly if the sample image or optional `transformers` dependency is missing.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from footy_track.object_detections import (
    Detection,
    FrameDetections,
    _clamp01,
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def arsenal_mancity_image_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "data" / "arsenal_mancity_test_detection.jpg"


@pytest.fixture(scope="session")
def arsenal_mancity_json_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "data" / "arsenal_mancity_test_detections.json"


@pytest.fixture(scope="session")
def arsenal_mancity_test_detection_as_frame(
    repo_root: Path, arsenal_mancity_json_path: Path
) -> FrameDetections:
    """Return the curated Arsenal/Man City frame as FrameDetections.

    - JSON bboxes are in top-left [x, y, w, h] normalized format (matches schema)
    - No coordinate conversion is performed here to avoid double-conversion bugs
    """
    image_path = repo_root / "tests" / "data" / "arsenal_mancity_test_detection.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    with arsenal_mancity_json_path.open("r") as f:
        data = json.load(f)

    objects = data.get("objects") or []

    detections: list[Detection] = []
    for obj in objects:
        label = str(obj.get("label", "")).strip()
        bbox = obj.get("bbox", [0, 0, 0, 0])
        try:
            x, y, w, h = [float(v) for v in bbox]
        except Exception:
            x = y = w = h = 0.0
        detections.append(
            Detection(
                label=label,
                confidence=1.0,
                x=_clamp01(x),
                y=_clamp01(y),
                w=_clamp01(w),
                h=_clamp01(h),
            )
        )

    # Get image size
    with Image.open(image_path) as im:
        img = im.convert("RGB")
        width, height = img.size

    return FrameDetections(
        uri=image_path,
        width=int(width),
        height=int(height),
        detections=detections,
    )
