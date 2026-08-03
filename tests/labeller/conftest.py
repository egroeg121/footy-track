"""Shared fixtures for labeller server tests.

Everything here is designed so tests never open a real video file and never
run real ONNX/YOLO inference (both are slow and can be SIGKILLed in sandboxed
shells):

- ``fake_cv2`` replaces the ``cv2`` name inside the labeller server modules
  with a stand-in whose ``VideoCapture`` serves synthetic metadata/frames.
- ``gt_marks_dir`` / ``clips_dir`` redirect the module-level data dirs to tmp.
- ``fresh_session`` swaps the global ``SESSION`` singleton for a clean one.
- ``client`` is a FastAPI ``TestClient`` over the real app.
"""

from __future__ import annotations

import collections
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from footy_track.labeller import server as labeller_server
from footy_track.labeller.server import Session
from footy_track.schema import ObjectDetection

# ---------------------------------------------------------------------------
# Fake cv2 layer
# ---------------------------------------------------------------------------

# Real cv2 property ids (stable integers in OpenCV's API).
CAP_PROP_FPS = 5
CAP_PROP_FRAME_COUNT = 7
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_POS_FRAMES = 1

FAKE_JPEG = b"\xff\xd8\xe0fake-jpeg-bytes\xff\xd9"


class _FakeBuf:
    """Mimics the ndarray cv2.imencode returns — only .tobytes() is used."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def tobytes(self) -> bytes:
        return self._data


class _FakeFrame:
    """Mimics an ndarray frame — only .shape[:2] and .size are used."""

    def __init__(self, width: int, height: int) -> None:
        self.shape = (height, width, 3)
        self.size = width * height * 3

    def __getitem__(self, _key) -> _FakeFrame:
        # frame[y1:y2, x1:x2] crop — return a non-empty fake crop.
        return _FakeFrame(8, 8)


class FakeVideoCapture:
    """Synthetic cv2.VideoCapture: fixed metadata, readable fake frames."""

    # Class-level config so tests can tune before an endpoint opens a capture.
    fps: float = 25.0
    total_frames: int = 10
    width: int = 640
    height: int = 360

    def __init__(self, path: str) -> None:
        self.path = path
        self.pos = 0
        self.released = False

    def get(self, prop: int) -> float:
        return {
            CAP_PROP_FPS: self.fps,
            CAP_PROP_FRAME_COUNT: float(self.total_frames),
            CAP_PROP_FRAME_WIDTH: float(self.width),
            CAP_PROP_FRAME_HEIGHT: float(self.height),
        }.get(prop, 0.0)

    def set(self, prop: int, value: float) -> None:
        if prop == CAP_PROP_POS_FRAMES:
            self.pos = int(value)

    def read(self):
        if 0 <= self.pos < self.total_frames:
            self.pos += 1
            return True, _FakeFrame(self.width, self.height)
        return False, None

    def release(self) -> None:
        self.released = True


def make_fake_cv2() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        VideoCapture=FakeVideoCapture,
        imencode=lambda _ext, _frame, *args: (True, _FakeBuf(FAKE_JPEG)),
        imwrite=lambda _path, _frame, *args: True,
        CAP_PROP_FPS=CAP_PROP_FPS,
        CAP_PROP_FRAME_COUNT=CAP_PROP_FRAME_COUNT,
        CAP_PROP_FRAME_WIDTH=CAP_PROP_FRAME_WIDTH,
        CAP_PROP_FRAME_HEIGHT=CAP_PROP_FRAME_HEIGHT,
        CAP_PROP_POS_FRAMES=CAP_PROP_POS_FRAMES,
        IMWRITE_JPEG_QUALITY=1,
    )


@pytest.fixture
def fake_cv2(monkeypatch):
    """Swap ``cv2`` for the fake in every labeller server module; reset config."""
    FakeVideoCapture.fps = 25.0
    FakeVideoCapture.total_frames = 10
    FakeVideoCapture.width = 640
    FakeVideoCapture.height = 360
    fake = make_fake_cv2()
    patch_labeller_attr(monkeypatch, "cv2", fake, exclude=_CV2_PATCH_EXCLUDE)
    return fake


# ---------------------------------------------------------------------------
# Patch-point helper
# ---------------------------------------------------------------------------

# Modules whose real cv2 must be left alone (they do actual frame processing
# exercised by other tests, not stubbed HTTP video IO).
_CV2_PATCH_EXCLUDE = ("video_utils", "motion_tracker", "app", "_canvas_compat")


def patch_labeller_attr(
    monkeypatch, name: str, value, exclude: tuple[str, ...] = ()
) -> None:
    """Monkeypatch ``name`` in every imported ``footy_track.labeller`` module
    that defines it.

    This keeps the test patch points stable across refactors: whether an
    endpoint lives in ``server.py`` or an extracted submodule, redirecting
    e.g. ``_GT_MARKS_DIR`` or ``cv2`` hits every copy of the name.
    """
    patched = False
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("footy_track.labeller"):
            continue
        if mod_name.rsplit(".", 1)[-1] in exclude:
            continue
        if name in vars(mod):
            monkeypatch.setattr(mod, name, value)
            patched = True
    assert patched, f"no footy_track.labeller module defines {name!r}"


# ---------------------------------------------------------------------------
# Data-dir redirection
# ---------------------------------------------------------------------------


@pytest.fixture
def gt_marks_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "ball_gt_marks"
    d.mkdir()
    patch_labeller_attr(monkeypatch, "_GT_MARKS_DIR", d)
    return d


@pytest.fixture
def clips_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "clips"
    d.mkdir()
    patch_labeller_attr(monkeypatch, "_CLIPS_DIR", d)
    return d


# ---------------------------------------------------------------------------
# Session + client
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_session(monkeypatch) -> Session:
    """Replace the global SESSION with a clean one for this test."""
    session = Session()
    patch_labeller_attr(monkeypatch, "SESSION", session)
    return session


@pytest.fixture
def client() -> TestClient:
    return TestClient(labeller_server.app)


@pytest.fixture
def crop_cache(monkeypatch):
    """Isolate the review crop LRU cache."""
    cache: collections.OrderedDict = collections.OrderedDict()
    patch_labeller_attr(monkeypatch, "_CROP_CACHE", cache)
    return cache


# ---------------------------------------------------------------------------
# Helpers shared across test modules
# ---------------------------------------------------------------------------


def make_box(
    label: str = "player",
    model: str = "labeller",
    confidence: float = 1.0,
    x: float = 0.1,
    y: float = 0.2,
    w: float = 0.05,
    h: float = 0.08,
) -> ObjectDetection:
    return ObjectDetection(
        label=label, confidence=confidence, x=x, y=y, w=w, h=h, model=model
    )


def load_fake_clip(
    session: Session,
    clips_dir: Path,
    name: str = "clip.mp4",
    total_frames: int = 10,
) -> Path:
    """Create an empty stand-in video file and load it into the session.

    Requires the ``fake_cv2`` fixture to be active (Session.load reads
    metadata through the server module's cv2 name).
    """
    video = clips_dir / name
    video.touch()
    FakeVideoCapture.total_frames = total_frames
    session.load(str(video))
    return video
