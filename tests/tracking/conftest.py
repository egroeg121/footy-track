"""Fixtures for tracking tests.

Synthetic FrameDetections are used for unit tests so no model inference is needed.
Real FrameDetections (via UltralyticsObjectDetector on footy_data frames) are used
for integration tests and are marked slow.
"""

from __future__ import annotations

import pathlib

import pytest

from footy_track.schema import FrameDetections, ObjectDetection

FOOTY_DATA_FRAMES = (
    pathlib.Path.home() / "code/footy/footy_data/arsenal_mancity/full_video_frames"
)
FRAME_GLOB = "*.png"
N_INTEGRATION_FRAMES = 15  # Must be > 10 per bead spec


def _det(
    label: str, x: float, y: float, w: float = 0.05, h: float = 0.1, conf: float = 0.9
) -> ObjectDetection:
    return ObjectDetection(label=label, confidence=conf, x=x, y=y, w=w, h=h)


def _fd(uri: str, dets: list[ObjectDetection]) -> FrameDetections:
    return FrameDetections(
        uri=pathlib.Path(uri), width=1920, height=1080, detections=dets
    )


@pytest.fixture
def two_player_frames() -> list[FrameDetections]:
    """10 frames with two slowly-moving players — IDs should stay stable."""
    frames = []
    for i in range(10):
        dx = i * 0.02
        frames.append(
            _fd(
                f"frame_{i:04d}.png",
                [
                    _det("person", 0.1 + dx, 0.3),
                    _det("person", 0.6 + dx, 0.3),
                ],
            )
        )
    return frames


@pytest.fixture
def disappearing_player_frames() -> list[FrameDetections]:
    """Player A is visible in all 10 frames; Player B disappears after frame 4."""
    frames = []
    for i in range(10):
        dets = [_det("person", 0.1 + i * 0.01, 0.3)]
        if i < 5:
            dets.append(_det("person", 0.6 + i * 0.01, 0.3))
        frames.append(_fd(f"frame_{i:04d}.png", dets))
    return frames


@pytest.fixture
def single_frame_two_players() -> FrameDetections:
    return _fd("frame_0000.png", [_det("person", 0.1, 0.3), _det("person", 0.6, 0.3)])


@pytest.fixture
def empty_frame() -> FrameDetections:
    return _fd("frame_empty.png", [])


@pytest.fixture
def footy_data_frames() -> list[pathlib.Path]:
    """Sorted frame paths from footy_data — skipped if directory absent."""
    if not FOOTY_DATA_FRAMES.exists():
        pytest.skip(f"footy_data frames not found at {FOOTY_DATA_FRAMES}")
    paths = sorted(FOOTY_DATA_FRAMES.glob(FRAME_GLOB))[:N_INTEGRATION_FRAMES]
    if len(paths) < N_INTEGRATION_FRAMES:
        pytest.skip(
            f"Not enough frames — need {N_INTEGRATION_FRAMES}, found {len(paths)}"
        )
    return paths
