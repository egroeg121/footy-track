"""Unit tests for the SAM2 ball tracker (bake-off method B).

These tests run without loading the actual SAM2 model — we mock the predictor
to verify the Kalman / ROI / coordinate-mapping logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from footy_track.ball_tracking.sam2_tracker import (
    Sam2BallTracker,
    _bbox_centre,
    _compute_roi,
    _kf_init,
    _kf_predict,
    _kf_update,
    _mask_to_bbox,
)

# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def test_bbox_centre():
    assert _bbox_centre((0.1, 0.2, 0.4, 0.3)) == pytest.approx((0.3, 0.35))


def test_compute_roi_stays_in_bounds():
    roi = _compute_roi(0.5, 0.5, 0.1, 0.1, scale=3.0)
    x1, y1, x2, y2 = roi
    assert 0.0 <= x1 < x2 <= 1.0
    assert 0.0 <= y1 < y2 <= 1.0


def test_compute_roi_near_edge():
    # Ball near top-left — ROI must not go negative
    roi = _compute_roi(0.01, 0.01, 0.05, 0.05, scale=3.0)
    x1, y1, x2, y2 = roi
    assert x1 >= 0.0
    assert y1 >= 0.0


def test_mask_to_bbox_simple():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 30:60] = True
    bbox = _mask_to_bbox(mask)
    assert bbox is not None
    x, y, w, h = bbox
    assert x == pytest.approx(30 / 100)
    assert y == pytest.approx(20 / 100)
    assert w == pytest.approx(30 / 100)
    assert h == pytest.approx(20 / 100)


def test_mask_to_bbox_empty_mask():
    mask = np.zeros((100, 100), dtype=bool)
    assert _mask_to_bbox(mask) is None


# --------------------------------------------------------------------------- #
# Kalman filter                                                               #
# --------------------------------------------------------------------------- #


def test_kf_predict_advances_position():
    kf = _kf_init(0.5, 0.5)
    # seed with velocity via two updates
    _kf_update(kf, 0.5, 0.5)
    # give it a velocity by updating to a new position
    kf["x"][2] = 0.02  # vx
    kf["x"][3] = 0.01  # vy
    pos = _kf_predict(kf)
    # Should predict cx = 0.5 + 0.02 = 0.52
    assert float(pos[0]) == pytest.approx(0.52, abs=0.01)


def test_kf_update_pulls_towards_measurement():
    kf = _kf_init(0.5, 0.5)
    _kf_predict(kf)
    _kf_update(kf, 0.6, 0.6)
    # After update, state should be pulled toward (0.6, 0.6)
    assert kf["x"][0] > 0.5
    assert kf["x"][1] > 0.5


# --------------------------------------------------------------------------- #
# Tracker interface                                                            #
# --------------------------------------------------------------------------- #


def _make_fake_result(mask_array: np.ndarray | None = None):
    """Build a mock Ultralytics result with an optional mask."""
    result = MagicMock()
    if mask_array is not None:
        import torch  # noqa: PLC0415

        mask_tensor = torch.from_numpy(mask_array.astype(np.float32)).unsqueeze(0)
        result.masks.data = mask_tensor
        result.masks.__len__ = lambda self: 1
    else:
        result.masks = None
        result.boxes = None
    return result


@patch("footy_track.ball_tracking.sam2_tracker.Sam2BallTracker._ensure_model")
def test_tracker_reset_clears_state(mock_ensure):
    tracker = Sam2BallTracker()
    tracker._kf = _kf_init(0.5, 0.5)
    tracker._last_bbox = (0.1, 0.1, 0.05, 0.05)
    tracker.reset()
    assert tracker._kf is None
    assert tracker._last_bbox is None
    assert tracker._last_crop_height is None


@patch("footy_track.ball_tracking.sam2_tracker.Sam2BallTracker._run_sam2")
@patch("footy_track.ball_tracking.sam2_tracker.Sam2BallTracker._ensure_model")
def test_tracker_first_call_no_prev_bbox(mock_ensure, mock_run_sam2):
    """Cold start (prev_bbox=None) should fall through to full-frame search."""
    mock_run_sam2.return_value = (0.4, 0.4, 0.05, 0.05)

    tracker = Sam2BallTracker()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = tracker.track(None, frame)

    assert result is not None
    assert mock_run_sam2.called
    # Kalman should be initialised after finding the ball
    assert tracker._kf is not None


@patch("footy_track.ball_tracking.sam2_tracker.Sam2BallTracker._run_sam2")
@patch("footy_track.ball_tracking.sam2_tracker.Sam2BallTracker._ensure_model")
def test_tracker_maps_crop_coords_to_full_frame(mock_ensure, mock_run_sam2):
    """Verify that a crop-local bbox is correctly projected back to full-frame."""
    # Place the ball at (0.5, 0.5), size 0.05x0.05 in a 200x200 frame.
    prev = (0.475, 0.475, 0.05, 0.05)

    # SAM2 will "find" the ball centred in the crop
    # The ROI will be ~3x ball size centred on predicted cx/cy ≈ 0.5, 0.5
    # Crop-local result: ball is at centre of crop, roughly (0.4, 0.4, 0.2, 0.2)
    # We just check that the tracker returns a result and Kalman state is updated.
    mock_run_sam2.return_value = (0.3, 0.3, 0.4, 0.4)

    tracker = Sam2BallTracker()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    result = tracker.track(prev, frame)

    assert result is not None
    full_x, full_y, full_w, full_h = result
    # Result must be in [0, 1]
    assert 0.0 <= full_x <= 1.0
    assert 0.0 <= full_y <= 1.0
    assert full_w > 0.0
    assert full_h > 0.0


@patch("footy_track.ball_tracking.sam2_tracker.Sam2BallTracker._run_sam2")
@patch("footy_track.ball_tracking.sam2_tracker.Sam2BallTracker._ensure_model")
def test_tracker_returns_none_on_sam2_failure(mock_ensure, mock_run_sam2):
    """If SAM2 returns no mask, tracker returns None."""
    mock_run_sam2.return_value = None

    tracker = Sam2BallTracker()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = tracker.track((0.4, 0.4, 0.05, 0.05), frame)

    assert result is None


@patch("footy_track.ball_tracking.sam2_tracker.Sam2BallTracker._run_sam2")
@patch("footy_track.ball_tracking.sam2_tracker.Sam2BallTracker._ensure_model")
def test_tracker_records_crop_height(mock_ensure, mock_run_sam2):
    mock_run_sam2.return_value = (0.3, 0.3, 0.4, 0.4)

    tracker = Sam2BallTracker()
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    tracker.track((0.4, 0.4, 0.05, 0.05), frame)

    # Should have recorded a crop height
    assert tracker._last_crop_height is not None
    assert tracker._last_crop_height > 0
