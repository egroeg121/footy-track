"""Tests for FOOTY_DEBUG lightweight mode and env-based directory resolution.

Debug mode exists so the labeller can run on a box with no GPU and little RAM.
The hard requirement is that it loads NO model weights — if it ever constructs
a real detector, startup pulls a checkpoint and the box swaps itself to death.
"""

from __future__ import annotations

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

    from footy_track.labeller import datasets  # noqa: PLC0415

    assert datasets.current().marks_dir == gt
    assert datasets.current().clips_dir == clips


def test_detector_checkpoint_resolution_order(monkeypatch, tmp_path):
    """env override > project root > shared models dir > (missing) project root."""
    from footy_track.detectors import ultralytics as det  # noqa: PLC0415

    root = tmp_path / "root"
    shared = tmp_path / "shared"
    monkeypatch.setattr(det, "get_project_root", lambda: root)
    monkeypatch.setattr(det, "SHARED_MODELS_DIR", shared)
    monkeypatch.delenv("FOOTY_DETECTOR_WEIGHTS", raising=False)
    rel = det.CURRENT_BEST_DETECTOR_CHECKPOINT
    in_project = root / rel
    in_shared = shared / "rtdetr_1920_v11" / "best.pt"

    assert det.current_best_detector_path() == in_project  # nothing exists

    in_shared.parent.mkdir(parents=True)
    in_shared.write_bytes(b"w")
    assert det.current_best_detector_path() == in_shared  # shared fallback

    in_project.parent.mkdir(parents=True)
    in_project.write_bytes(b"w")
    assert det.current_best_detector_path() == in_project  # project root wins

    monkeypatch.setenv("FOOTY_DETECTOR_WEIGHTS", str(tmp_path / "mine.pt"))
    assert det.current_best_detector_path() == tmp_path / "mine.pt"
