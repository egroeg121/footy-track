"""Tests for the ROI-YOLO ball tracker (Method C, ft-1d9).

These tests do NOT load a real YOLO model — they patch ultralytics.YOLO so
the tracker logic (Kalman prediction, crop geometry, coordinate remapping) can
be verified in a pure-Python, zero-dependency-download environment.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_frame(h: int = 720, w: int = 1280) -> np.ndarray:
    """Return a black RGB frame of given shape."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mock_yolo_result(
    x1n: float, y1n: float, x2n: float, y2n: float, conf: float = 0.9
):
    """Return a fake Ultralytics Results object for one sports-ball detection."""
    boxes = SimpleNamespace(
        conf=torch.tensor([conf]),
        xyxyn=torch.tensor([[x1n, y1n, x2n, y2n]]),
        __len__=lambda self: 1,
    )
    return SimpleNamespace(boxes=boxes)


def _make_tracker(model_mock=None, **kwargs):
    """Construct a RoiYoloTracker with YOLO patched out."""
    with patch("footy_track.ball_trackers.roi_yolo.torch") as mock_torch, patch(
        "ultralytics.YOLO"
    ) as mock_yolo_cls:
        # Set up device selection to return CPU
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad.return_value.__enter__ = lambda s: None
        mock_torch.no_grad.return_value.__exit__ = lambda s, *a: None

        if model_mock is None:
            model_mock = MagicMock()
        mock_yolo_cls.return_value = model_mock

        from footy_track.ball_trackers.roi_yolo import RoiYoloTracker

        tracker = RoiYoloTracker(**kwargs)
        tracker._model = model_mock
        tracker._device = "cpu"
        return tracker


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_implements_ball_tracker_protocol():
    """RoiYoloTracker must have track() and reset() methods."""
    from footy_track.ball_trackers.roi_yolo import RoiYoloTracker

    assert hasattr(RoiYoloTracker, "track")
    assert hasattr(RoiYoloTracker, "reset")


def test_reset_clears_state():
    model_mock = MagicMock()
    model_mock.predict.return_value = [_mock_yolo_result(0.4, 0.3, 0.5, 0.4)]

    tracker = _make_tracker(model_mock=model_mock)
    frame = _make_frame()
    tracker.track((0.4, 0.3, 0.1, 0.1), frame)  # warm up Kalman

    tracker.reset()
    assert tracker._kf is None
    assert tracker._last_bbox is None
    assert tracker._last_crop_height is None


# ---------------------------------------------------------------------------
# Kalman initialisation and cold start
# ---------------------------------------------------------------------------


def test_cold_start_uses_full_frame():
    """On first call (no prev_bbox), tracker should search the whole frame."""
    model_mock = MagicMock()
    model_mock.predict.return_value = [_mock_yolo_result(0.48, 0.38, 0.52, 0.42)]

    tracker = _make_tracker(model_mock=model_mock)
    frame = _make_frame()
    result = tracker.track(None, frame)

    # Tracker called YOLO on the full frame (no crop restriction)
    call_args = model_mock.predict.call_args
    assert call_args is not None
    # crop passed = the full frame (same shape)
    crop_arr = call_args[0][0]
    assert crop_arr.shape == (720, 1280, 3)
    # Result remapped correctly — YOLO saw full frame so coords unchanged
    assert result is not None
    assert pytest.approx(result[0], abs=0.01) == 0.48
    assert pytest.approx(result[1], abs=0.01) == 0.38


def test_cold_start_no_detection_returns_none():
    model_mock = MagicMock()
    model_mock.predict.return_value = [SimpleNamespace(boxes=None)]

    tracker = _make_tracker(model_mock=model_mock)
    result = tracker.track(None, _make_frame())
    assert result is None


# ---------------------------------------------------------------------------
# ROI crop geometry
# ---------------------------------------------------------------------------


def test_roi_crops_around_predicted_position():
    """With a prev_bbox, the crop should be smaller than the full frame."""
    model_mock = MagicMock()
    # Detection at center of the crop (0.5, 0.5 in crop coords)
    model_mock.predict.return_value = [_mock_yolo_result(0.4, 0.4, 0.6, 0.6)]

    tracker = _make_tracker(model_mock=model_mock, roi_scale=3.0, min_roi_frac=0.05)
    frame = _make_frame(h=720, w=1280)
    prev = (0.45, 0.45, 0.03, 0.03)  # 3% wide ball
    tracker.track(prev, frame)

    crop_arr = model_mock.predict.call_args[0][0]
    # Crop must be smaller than full frame
    assert crop_arr.shape[0] < 720
    assert crop_arr.shape[1] < 1280


def test_last_crop_height_set_after_track():
    model_mock = MagicMock()
    model_mock.predict.return_value = [_mock_yolo_result(0.4, 0.4, 0.6, 0.6)]

    tracker = _make_tracker(model_mock=model_mock)
    tracker.track((0.4, 0.3, 0.05, 0.05), _make_frame())
    assert tracker._last_crop_height is not None
    assert tracker._last_crop_height > 0


def test_last_crop_height_none_before_track():
    tracker = _make_tracker()
    assert tracker._last_crop_height is None


# ---------------------------------------------------------------------------
# Coordinate remapping
# ---------------------------------------------------------------------------


def test_coord_remapping_full_frame_cold_start():
    """When the whole frame is fed, crop-normalized = full-frame-normalized."""
    model_mock = MagicMock()
    # Exact detection at (0.3, 0.2, 0.1, 0.08) in full frame
    model_mock.predict.return_value = [_mock_yolo_result(0.3, 0.2, 0.4, 0.28)]

    tracker = _make_tracker(model_mock=model_mock)
    result = tracker.track(None, _make_frame(h=720, w=1280))

    assert result is not None
    x, y, w, h = result
    assert pytest.approx(x, abs=0.01) == 0.3
    assert pytest.approx(y, abs=0.01) == 0.2
    assert pytest.approx(w, abs=0.01) == 0.1
    assert pytest.approx(h, abs=0.01) == 0.08


def test_coord_remapping_with_roi_crop():
    """Detection inside a crop is correctly mapped back to full-frame coords.

    Setup:
    - Frame: 720×1280
    - prev_bbox: ball at (0.5, 0.5) normalised, size 0.03×0.03
    - Crop: ~80px square centred on (0.5, 0.5) = (640, 360) pixel
    - YOLO reports detection at crop-normalised (0.5, 0.5, 0.55, 0.55)
      (i.e. dead centre of crop)
    - We expect full-frame coords ≈ (0.5, 0.5) with small size
    """
    model_mock = MagicMock()
    # Ball detected at centre of crop
    model_mock.predict.return_value = [_mock_yolo_result(0.45, 0.45, 0.55, 0.55)]

    tracker = _make_tracker(model_mock=model_mock, roi_scale=3.0, min_roi_frac=0.05)
    prev = (0.485, 0.485, 0.03, 0.03)  # centred at (0.5, 0.5)
    result = tracker.track(prev, _make_frame(h=720, w=1280))

    assert result is not None
    x, y, w, h = result
    # Full-frame centre ≈ 0.5, 0.5
    cx = x + w / 2.0
    cy = y + h / 2.0
    assert pytest.approx(cx, abs=0.05) == 0.5
    assert pytest.approx(cy, abs=0.05) == 0.5


# ---------------------------------------------------------------------------
# No detection case
# ---------------------------------------------------------------------------


def test_no_detection_returns_none_with_prev_bbox():
    model_mock = MagicMock()
    model_mock.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(
        conf=torch.tensor([]),
        xyxyn=torch.tensor([]).reshape(0, 4),
        __len__=lambda self: 0,
    ))]

    tracker = _make_tracker(model_mock=model_mock)
    result = tracker.track((0.4, 0.3, 0.05, 0.05), _make_frame())
    assert result is None


# ---------------------------------------------------------------------------
# Confidence threshold filtering
# ---------------------------------------------------------------------------


def test_low_confidence_detection_rejected():
    model_mock = MagicMock()
    # Very low confidence — below the tracker's threshold
    model_mock.predict.return_value = [_mock_yolo_result(0.4, 0.3, 0.5, 0.4, conf=0.05)]

    tracker = _make_tracker(model_mock=model_mock, min_confidence=0.20)
    result = tracker.track(None, _make_frame())
    # conf=0.05 < min_confidence=0.20 → should be None
    assert result is None


def test_sufficient_confidence_detection_accepted():
    model_mock = MagicMock()
    model_mock.predict.return_value = [_mock_yolo_result(0.4, 0.3, 0.5, 0.4, conf=0.80)]

    tracker = _make_tracker(model_mock=model_mock, min_confidence=0.20)
    result = tracker.track(None, _make_frame())
    assert result is not None


# ---------------------------------------------------------------------------
# Harness integration (no real YOLO, no real video)
# ---------------------------------------------------------------------------


def test_harness_runner_integration():
    """RoiYoloTracker plugs into the bake-off harness without errors."""
    import pathlib
    import tempfile

    from footy_track.ball_eval.dataset import BBox, EvalClip, EvalDataset, FrameLabel
    from footy_track.ball_eval.runner import run_benchmark

    bbox: BBox = (0.45, 0.45, 0.05, 0.05)
    num_frames = 5

    labels = {
        i: FrameLabel(frame_index=i, bbox=bbox, tags=())
        for i in range(num_frames)
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        clip = EvalClip(
            name="synthetic",
            video_path=pathlib.Path(tmpdir) / "synthetic.mp4",
            labels=labels,
            total_frames=num_frames,
        )
        frames = [_make_frame(h=64, w=64) for _ in range(num_frames)]
        frame_data = list(enumerate(frames))

        def _iter():
            yield from frame_data

        clip.iter_frames = _iter  # type: ignore[method-assign]

        dataset = EvalDataset([clip])

        model_mock = MagicMock()
        model_mock.predict.return_value = [_mock_yolo_result(0.4, 0.4, 0.6, 0.6)]

        tracker = _make_tracker(model_mock=model_mock)
        result = run_benchmark(tracker, dataset, method_name="roi_yolo_c", verbose=False)

        assert result.method_name == "roi_yolo_c"
        assert len(result.clip_metrics) == 1
        cm = result.clip_metrics[0]
        # Tracker returned something every frame → recall should be 100%
        assert cm.recall_pct == pytest.approx(100.0)
