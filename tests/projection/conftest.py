"""Local conftest for projection tests.

Overrides the session-scoped autouse fixtures from the root conftest that
require a real video file (not needed for projection unit tests).
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def frames_path(repo_root, tmp_path_factory):
    """No-op override — projection tests don't need extracted frames."""
    path = tmp_path_factory.mktemp("frames")
    return path


@pytest.fixture(scope="session", autouse=True)
def extracted_frames(frames_path):
    """No-op override — projection tests don't extract frames."""
    return []
