"""Tests for FOOTY_DEBUG lightweight mode and env-based directory resolution.

Debug mode exists so the labeller can run on a box with no GPU and little RAM.
The hard requirement is that it loads NO model weights — if it ever constructs
a real detector, startup pulls a checkpoint and the box swaps itself to death.
"""

from __future__ import annotations

import importlib

from footy_track.detectors.ultralytics import (
    StubObjectDetector,
    debug_mode,
    get_current_best_detector,
)


def test_debug_mode_reads_env(monkeypatch):
    monkeypatch.delenv("FOOTY_DEBUG", raising=False)
    assert debug_mode() is False
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("FOOTY_DEBUG", truthy)
        assert debug_mode() is True
    monkeypatch.setenv("FOOTY_DEBUG", "0")
    assert debug_mode() is False


def test_debug_detector_loads_no_weights(monkeypatch):
    """The whole point: no checkpoint is touched under FOOTY_DEBUG."""
    monkeypatch.setenv("FOOTY_DEBUG", "1")

    def _explode(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("debug mode must not construct a real detector")

    monkeypatch.setattr(
        "footy_track.detectors.ultralytics.UltralyticsObjectDetector", _explode
    )
    detector = get_current_best_detector()
    assert isinstance(detector, StubObjectDetector)
    assert detector.model_tag == "stub"


def test_stub_returns_no_detections(tmp_path):
    """Empty, not fake: invented boxes could be saved and corrupt provenance."""
    img = tmp_path / "frame.png"
    img.write_bytes(b"not a real image")
    result = StubObjectDetector().predict_from_path(img)
    assert result.detections == []


def test_gt_marks_dir_honours_env(monkeypatch, tmp_path):
    """Machine-specific hardcoded paths made Linux hosts silently show no labels."""
    gt = tmp_path / "gt"
    gt.mkdir()
    clips = tmp_path / "clips"
    clips.mkdir()
    monkeypatch.setenv("FOOTY_GT_MARKS_DIR", str(gt))
    monkeypatch.setenv("FOOTY_CLIPS_DIR", str(clips))

    from footy_track.labeller import server

    importlib.reload(server)
    try:
        assert server._GT_MARKS_DIR == gt
        assert server._CLIPS_DIR == clips
    finally:
        monkeypatch.delenv("FOOTY_GT_MARKS_DIR", raising=False)
        monkeypatch.delenv("FOOTY_CLIPS_DIR", raising=False)
        importlib.reload(server)
