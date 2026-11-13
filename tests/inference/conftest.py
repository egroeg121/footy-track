"""Shared fixtures: a tiny synthetic video and a job runner with fake parts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from footy_track.inference.jobs import (
    JobRunner,
    fake_detector_factory,
    make_chunk_segmenter,
)

W, H, N = 160, 90, 30


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "match.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    for i in range(N):
        writer.write(np.full((H, W, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def runner(tmp_path: Path):
    r = JobRunner(
        tmp_path / "jobs",
        detector_factory=fake_detector_factory,
        segmenter=make_chunk_segmenter(chunk_frames=15),
    )
    yield r
    r.shutdown()
