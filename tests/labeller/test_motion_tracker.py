"""Tests for ft-v43: FrameTracker/CropRunner protocol + MotionGuidedTracker.

GPU-independent by construction: all math here is plain numpy (no torch, no
model weights), so these run identically on CPU/MPS/CUDA-less CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from footy_track.labeller.motion_tracker import (
    Detection,
    KalmanBoxTracker,
    MotionGuidedTracker,
    box_to_cxcywh,
    compute_roi,
    crop_frame,
    cxcywh_to_box,
    map_crop_to_frame,
    map_frame_to_crop,
)

# ---------------------------------------------------------------------------
# Coordinate helpers / crop <-> frame round trip
# ---------------------------------------------------------------------------


def test_cxcywh_box_round_trip():
    box = (10.0, 20.0, 50.0, 80.0)
    cx, cy, w, h = box_to_cxcywh(box)
    assert (cx, cy, w, h) == pytest.approx((30.0, 50.0, 40.0, 60.0))
    assert cxcywh_to_box(cx, cy, w, h) == pytest.approx(box)


def test_compute_roi_centered_when_within_bounds():
    pred_box = (490.0, 490.0, 510.0, 510.0)  # 20x20 box centred at (500, 500)
    roi = compute_roi(pred_box, frame_w=1920, frame_h=1080, margin_scale=2.0, min_size=10.0)
    x1, y1, x2, y2 = roi
    # Fully inside frame -> should stay centred on the predicted box's centre.
    assert (x1 + x2) / 2.0 == pytest.approx(500.0, abs=1e-6)
    assert (y1 + y2) / 2.0 == pytest.approx(500.0, abs=1e-6)
    assert x2 - x1 == pytest.approx(y2 - y1)  # square ROI


def test_compute_roi_clamped_to_frame_near_edge():
    pred_box = (0.0, 0.0, 20.0, 20.0)  # ball right at the top-left corner
    roi = compute_roi(pred_box, frame_w=640, frame_h=480, margin_scale=3.0, min_size=100.0)
    x1, y1, x2, y2 = roi
    assert x1 >= 0.0
    assert y1 >= 0.0
    assert x2 <= 640.0
    assert y2 <= 480.0


def test_compute_roi_grows_with_velocity():
    pred_box = (490.0, 490.0, 510.0, 510.0)
    slow = compute_roi(pred_box, 1920, 1080, velocity=(0.0, 0.0), margin_scale=2.0)
    fast = compute_roi(pred_box, 1920, 1080, velocity=(200.0, 0.0), margin_scale=2.0)

    def side(roi):
        x1, y1, x2, y2 = roi
        return x2 - x1

    assert side(fast) > side(slow)


def test_crop_frame_extracts_expected_pixels():
    frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    roi = (10.0, 20.0, 40.0, 60.0)
    crop = crop_frame(frame, roi)
    assert crop.shape == (40, 30, 3)
    np.testing.assert_array_equal(crop, frame[20:60, 10:40])


def test_crop_to_frame_coordinate_round_trip():
    """The key ft-v43 requirement: crop-local box -> frame box -> back."""
    roi = (100.0, 150.0, 356.0, 406.0)  # 256x256 crop starting at (100, 150)
    box_in_frame_original = (180.0, 220.0, 240.0, 280.0)

    box_in_crop = map_frame_to_crop(box_in_frame_original, roi)
    box_in_frame_recovered = map_crop_to_frame(box_in_crop, roi)

    assert box_in_frame_recovered == pytest.approx(box_in_frame_original)
    # And the crop-local box should be within the crop's own dimensions.
    cx1, cy1, cx2, cy2 = box_in_crop
    assert 0.0 <= cx1 <= 256.0
    assert 0.0 <= cy1 <= 256.0
    assert 0.0 <= cx2 <= 256.0
    assert 0.0 <= cy2 <= 256.0


def test_crop_to_frame_round_trip_matches_pixel_crop():
    """Round-trip using an actual cropped ndarray, not just the box math."""
    frame = np.random.default_rng(0).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    roi = (50.0, 60.0, 306.0, 316.0)  # 256x256
    crop = crop_frame(frame, roi)
    assert crop.shape == (256, 256, 3)

    # A box detected inside the crop...
    box_in_crop = (10.0, 20.0, 60.0, 70.0)
    # ...maps to the same pixels in the original frame.
    box_in_frame = map_crop_to_frame(box_in_crop, roi)
    x1, y1, x2, y2 = (int(v) for v in box_in_frame)
    np.testing.assert_array_equal(
        frame[y1:y2, x1:x2],
        crop[20:70, 10:60],
    )


# ---------------------------------------------------------------------------
# KalmanBoxTracker predict/update
# ---------------------------------------------------------------------------


def test_kalman_init_predict_returns_same_box_with_zero_velocity():
    kf = KalmanBoxTracker()
    seed = (100.0, 100.0, 140.0, 140.0)  # 40x40 box
    kf.init(seed)
    assert kf.box == pytest.approx(seed)
    predicted = kf.predict()
    # Zero initial velocity -> prediction should match the seed box.
    assert predicted == pytest.approx(seed, abs=1e-6)


def test_kalman_tracks_constant_velocity_motion():
    """Feed a ball moving at constant velocity; predictions should converge
    to track it, and velocity should be recovered close to the true value."""
    kf = KalmanBoxTracker(process_noise=0.5, measurement_noise=2.0, velocity_process_noise=2.0)
    true_vx, true_vy = 8.0, -4.0
    w, h = 20.0, 20.0
    cx0, cy0 = 100.0, 100.0

    kf.init(cxcywh_to_box(cx0, cy0, w, h))

    cx, cy = cx0, cy0
    for step in range(1, 15):
        cx += true_vx
        cy += true_vy
        pred = kf.predict()
        observed = cxcywh_to_box(cx, cy, w, h)
        kf.update(observed)
        if step > 5:
            # After a few steps the filter should have "caught up" and be
            # tracking closely (constant velocity is exactly its model, so
            # the pre-update prediction for this step should already be
            # close to the true current position).
            pred_cx, pred_cy, _, _ = box_to_cxcywh(pred)
            assert pred_cx == pytest.approx(cx, abs=3.0)
            assert pred_cy == pytest.approx(cy, abs=3.0)

    vx, vy = kf.velocity
    assert vx == pytest.approx(true_vx, abs=1.0)
    assert vy == pytest.approx(true_vy, abs=1.0)


def test_kalman_predict_before_init_raises():
    kf = KalmanBoxTracker()
    with pytest.raises(RuntimeError):
        kf.predict()


def test_kalman_update_before_init_behaves_like_init():
    kf = KalmanBoxTracker()
    box = (0.0, 0.0, 10.0, 10.0)
    kf.update(box)
    assert kf.initialized
    assert kf.box == pytest.approx(box)


# ---------------------------------------------------------------------------
# MotionGuidedTracker: reset/step wiring, refine + reacquire-on-miss
# ---------------------------------------------------------------------------


class _FakeRunner:
    """A scripted CropRunner: returns pre-programmed detections/misses in order."""

    def __init__(self, name: str, responses: list[Detection | None]):
        self.name = name
        self._responses = list(responses)
        self.warmup_calls = 0
        self.detect_calls: list[tuple] = []

    def warmup(self) -> None:
        self.warmup_calls += 1

    def detect(self, crop, prior):
        self.detect_calls.append((crop.shape, prior))
        return self._responses.pop(0)


def _blank_frame(w=640, h=480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_motion_guided_tracker_reset_calls_warmup_and_seeds_kalman():
    runner = _FakeRunner("fake", [])
    tracker = MotionGuidedTracker(runner=runner)
    frame = _blank_frame()
    seed = (100.0, 100.0, 140.0, 140.0)

    tracker.reset(frame, seed)

    assert runner.warmup_calls == 1
    assert tracker.kalman.initialized
    assert tracker.kalman.box == pytest.approx(seed)


def test_motion_guided_tracker_step_refines_and_maps_back_to_frame():
    # Runner reports a detection near the crop's local origin; the tracker
    # must map it back to frame-absolute coordinates.
    seed = (100.0, 100.0, 140.0, 140.0)
    frame = _blank_frame()

    # We don't know the exact ROI a priori (it depends on compute_roi), so
    # compute it the same way the tracker will, to construct a runner response
    # that we can verify maps back correctly.
    predicted = seed  # zero velocity -> predict() == seed on first step
    roi = compute_roi(predicted, frame.shape[1], frame.shape[0])
    # Detection in CROP-local coords: same box as the seed, translated into
    # crop-local space.
    crop_local_box = (
        seed[0] - roi[0],
        seed[1] - roi[1],
        seed[2] - roi[0],
        seed[3] - roi[1],
    )
    runner = _FakeRunner("fake", [Detection(box=crop_local_box, confidence=0.9, label="ball")])
    tracker = MotionGuidedTracker(runner=runner, miss_confidence_thresh=0.3)
    tracker.reset(frame, seed)

    det = tracker.step(frame)

    assert det is not None
    assert det.confidence == pytest.approx(0.9)
    assert det.box == pytest.approx(seed, abs=1e-6)
    assert tracker.last_provenance == "fake"


def test_motion_guided_tracker_reacquires_on_miss():
    seed = (100.0, 100.0, 140.0, 140.0)
    frame = _blank_frame()

    # Per-step runner reports low confidence -> should trigger full-frame
    # re-acquire via a separate reacquire_runner.
    step_runner = _FakeRunner("cheap", [Detection(box=(0, 0, 1, 1), confidence=0.05)])
    reacquire_box = (300.0, 300.0, 340.0, 340.0)
    reacquire_runner = _FakeRunner(
        "sam3-fullframe", [Detection(box=reacquire_box, confidence=0.95, label="ball")]
    )
    tracker = MotionGuidedTracker(
        runner=step_runner,
        reacquire_runner=reacquire_runner,
        miss_confidence_thresh=0.3,
    )
    tracker.reset(frame, seed)

    det = tracker.step(frame)

    assert det is not None
    assert det.box == pytest.approx(reacquire_box)
    assert tracker.last_provenance == "sam3-fullframe-reacquire"
    # Kalman state should have been reinitialised at the reacquired box.
    assert tracker.kalman.box == pytest.approx(reacquire_box)
    # The re-acquire runner was called on the full frame, not a small crop.
    (call_shape, call_prior) = reacquire_runner.detect_calls[0]
    assert call_shape == frame.shape
    assert call_prior is None


def test_motion_guided_tracker_returns_none_when_reacquire_also_fails():
    seed = (100.0, 100.0, 140.0, 140.0)
    frame = _blank_frame()
    step_runner = _FakeRunner("cheap", [None])
    reacquire_runner = _FakeRunner("sam3-fullframe", [None])
    tracker = MotionGuidedTracker(runner=step_runner, reacquire_runner=reacquire_runner)
    tracker.reset(frame, seed)

    det = tracker.step(frame)

    assert det is None
    assert tracker.last_provenance is None


def test_motion_guided_tracker_step_before_reset_raises():
    tracker = MotionGuidedTracker(runner=_FakeRunner("fake", []))
    with pytest.raises(RuntimeError):
        tracker.step(_blank_frame())
