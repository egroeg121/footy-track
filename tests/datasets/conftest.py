"""Fixtures for the dataset-version tests. They are self-contained (tmp dirs, in-process DuckDB),
so the top-level autouse that extracts DVC-managed test video frames is overridden with a no-op."""

from __future__ import annotations

import pytest

from tests.datasets.fixtures import World, box, calibration_doc, skip, world, write_broadcast  # noqa: F401


@pytest.fixture(scope="session", autouse=True)
def extracted_frames() -> list:  # noqa: PT004 - overrides parent autouse fixture
    return []
