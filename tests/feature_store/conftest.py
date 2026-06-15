"""Local fixtures for feature-store tests.

The feature store is self-contained (in-memory DuckDB) and does not need the
extracted-video-frames autouse fixture from the top-level ``tests/conftest.py``
(which requires a DVC-managed test video). Override it with a no-op here so
these tests run without test-media dependencies.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def extracted_frames() -> list:  # noqa: PT004 - overrides parent autouse fixture
    return []
