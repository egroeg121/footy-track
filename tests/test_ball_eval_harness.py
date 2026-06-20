"""Tests for the ball-eval bake-off harness and VitTrack SOT tracker.

These tests use synthetic data only — no video files or real model weights needed.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pytest

from footy_track.ball_eval.dataset import BBox, FrameLabel, write_labels
from footy_track.ball_eval.metrics import (
    ClipMetrics,
    FramePrediction,
    MethodResult,
    bbox_iou,
    compute_clip_metrics,
)
from footy_track.ball_trackers.sot_vittrack import (
    _HANN_WINDOW,
    VitTrackSOT,
    _crop_region,
    _preprocess,
)

# --------------------------------------------------------------------------- #
# BBox IoU                                                                     #
# --------------------------------------------------------------------------- #


def test_iou_identical_boxes():
    b = (0.1, 0.1, 0.2, 0.2)
    assert bbox_iou(b, b) == pytest.approx(1.0, abs=1e-6)


def test_iou_non_overlapping():
    a = (0.0, 0.0, 0.1, 0.1)
    b = (0.9, 0.9, 0.1, 0.1)
    assert bbox_iou(a, b) == pytest.approx(0.0, abs=1e-6)


def test_iou_partial_overlap():
    # a: [0, 0, 0.2, 0.2]  b: [0.1, 0.1, 0.2, 0.2]
    # intersection: [0.1,0.1,0.1,0.1] = 0.01
    # union: 0.04 + 0.04 - 0.01 = 0.07
    a = (0.0, 0.0, 0.2, 0.2)
    b = (0.1, 0.1, 0.2, 0.2)
    expected = 0.01 / 0.07
    assert bbox_iou(a, b) == pytest.approx(expected, abs=1e-4)


def test_iou_contained_box():
    outer = (0.0, 0.0, 1.0, 1.0)
    inner = (0.25, 0.25, 0.5, 0.5)
    # inner area = 0.25; union = outer area = 1.0 → IoU = 0.25
    assert bbox_iou(outer, inner) == pytest.approx(0.25, abs=1e-6)


# --------------------------------------------------------------------------- #
# FrameLabel round-trip                                                        #
# --------------------------------------------------------------------------- #


def test_frame_label_with_bbox():
    d = {"frame_index": 5, "bbox": [0.1, 0.2, 0.3, 0.4], "tags": ["occlusion"]}
    lbl = FrameLabel.from_dict(d)
    assert lbl.frame_index == 5
    assert lbl.bbox == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert "occlusion" in lbl.tags
    assert lbl.to_dict() == d


def test_frame_label_ball_absent():
    d = {"frame_index": 3, "bbox": None, "tags": ["ball_not_visible"]}
    lbl = FrameLabel.from_dict(d)
    assert lbl.bbox is None


def test_write_labels_roundtrip():
    labels = [
        FrameLabel(frame_index=0, bbox=(0.1, 0.1, 0.2, 0.2), tags=()),
        FrameLabel(frame_index=1, bbox=None, tags=("occlusion",)),
    ]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = pathlib.Path(f.name)
    try:
        write_labels(labels, path)
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        assert len(rows) == 2
        assert rows[0]["frame_index"] == 0
        assert rows[1]["bbox"] is None
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# compute_clip_metrics                                                         #
# --------------------------------------------------------------------------- #


def _make_pred(frame_idx: int, gt: BBox | None, pred: BBox | None) -> FramePrediction:
    iou = bbox_iou(gt, pred) if gt is not None and pred is not None else None
    return FramePrediction(frame_index=frame_idx, gt_bbox=gt, pred_bbox=pred, iou=iou)


def test_perfect_tracking():
    gt_box: BBox = (0.1, 0.1, 0.2, 0.2)
    preds = [_make_pred(i, gt_box, gt_box) for i in range(10)]
    metrics = compute_clip_metrics(
        "clip_a",
        10,
        10,
        preds,
        total_inference_s=1.0,
        peak_vram_mb=0.0,
        occlusion_frame_indices=[],
    )
    assert metrics.mean_iou == pytest.approx(1.0, abs=1e-6)
    assert metrics.recall_pct == pytest.approx(100.0)
    assert metrics.tracking_failures == 0
    assert metrics.fps == pytest.approx(10.0)


def test_tracking_failure_counts():
    gt_box: BBox = (0.1, 0.1, 0.2, 0.2)
    preds = [
        _make_pred(0, gt_box, gt_box),
        _make_pred(1, gt_box, None),  # failure
        _make_pred(2, gt_box, gt_box),
        _make_pred(3, gt_box, None),  # failure
    ]
    metrics = compute_clip_metrics("clip_b", 4, 4, preds, 0.4, 0.0, [])
    assert metrics.tracking_failures == 2
    assert metrics.recall_pct == pytest.approx(50.0)


def test_occlusion_recovery():
    gt_box: BBox = (0.1, 0.1, 0.2, 0.2)
    # Occlusion at frame 2; tracker recovers at frame 4 (2 frames later ≤ 3)
    preds = [
        _make_pred(i, gt_box, gt_box if i not in {2, 3} else None) for i in range(6)
    ]
    metrics = compute_clip_metrics(
        "clip_c", 6, 6, preds, 0.6, 0.0, occlusion_frame_indices=[2]
    )
    assert metrics.occlusion_recovery_rate == pytest.approx(100.0)


def test_precision_anti_hallucination():
    gt_box: BBox = (0.1, 0.1, 0.2, 0.2)
    # 3 GT-present frames, 2 GT-absent frames; tracker predicts on all 5
    preds = [
        _make_pred(0, gt_box, gt_box),
        _make_pred(1, gt_box, gt_box),
        _make_pred(2, gt_box, gt_box),
        _make_pred(3, None, gt_box),  # hallucination
        _make_pred(4, None, gt_box),  # hallucination
    ]
    metrics = compute_clip_metrics("clip_d", 5, 3, preds, 0.5, 0.0, [])
    # 3 predictions with GT, 5 total predictions → 60% precision
    assert metrics.precision_pct == pytest.approx(60.0)
    assert metrics.recall_pct == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# MethodResult                                                                 #
# --------------------------------------------------------------------------- #


def _dummy_clip_metrics(
    name: str, iou: float, recall: float, fps: float
) -> ClipMetrics:
    return ClipMetrics(
        clip_name=name,
        total_frames=100,
        ball_present_frames=80,
        mean_iou=iou,
        median_iou=iou,
        recall_pct=recall,
        precision_pct=recall,
        tracking_failures=0,
        occlusion_recovery_rate=100.0,
        fps=fps,
        peak_vram_mb=0.0,
        effective_resolution_px=256,
    )


def test_method_result_aggregates():
    result = MethodResult(
        method_name="test",
        clip_metrics=[
            _dummy_clip_metrics("a", 0.8, 90.0, 100.0),
            _dummy_clip_metrics("b", 0.6, 70.0, 80.0),
        ],
    )
    assert result.mean_iou == pytest.approx(0.7)
    assert result.mean_recall == pytest.approx(80.0)
    assert result.mean_fps == pytest.approx(90.0)


def test_method_result_table_contains_method_name():
    result = MethodResult(
        method_name="vittrack-sot",
        clip_metrics=[_dummy_clip_metrics("clip1", 0.75, 85.0, 50.0)],
    )
    table = result.table()
    assert "vittrack-sot" in table


def test_method_result_to_dict():
    result = MethodResult(
        method_name="vittrack-sot",
        clip_metrics=[_dummy_clip_metrics("clip1", 0.75, 85.0, 50.0)],
    )
    d = result.to_dict()
    assert d["method"] == "vittrack-sot"
    assert "mean_iou" in d["aggregate"]
    assert len(d["clips"]) == 1


# --------------------------------------------------------------------------- #
# VitTrack SOT (no model — unit tests for preprocessing and geometry)         #
# --------------------------------------------------------------------------- #


def test_crop_region_centred():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Ball in the centre of the frame
    bbox_px = (270.0, 190.0, 100.0, 100.0)
    crop, crop_sz = _crop_region(frame, bbox_px, factor=2.0)
    assert crop_sz == 200  # ceil(sqrt(100*100) * 2) = 200
    assert crop.shape == (200, 200, 3)


def test_crop_region_with_border_padding():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Ball near top-left corner — crop extends outside frame
    bbox_px = (5.0, 5.0, 30.0, 30.0)
    crop, crop_sz = _crop_region(frame, bbox_px, factor=4.0)
    # Should pad rather than crash
    assert crop.shape[0] == crop_sz
    assert crop.shape[1] == crop_sz


def test_preprocess_output_shape_and_dtype():
    crop = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    blob = _preprocess(crop, (128, 128))
    assert blob.shape == (1, 3, 128, 128)
    assert blob.dtype == np.float32


def test_preprocess_normalisation_range():
    # All-white crop → should be positive after normalization
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    blob = _preprocess(white, (128, 128))
    assert blob.min() > 1.5  # (1 - 0.406) / 0.225 ≈ 2.64


def test_tracker_reset_clears_state():
    tracker = VitTrackSOT.__new__(VitTrackSOT)
    tracker._template_blob = np.zeros((1, 3, 128, 128), dtype=np.float32)
    tracker._last_bbox_px = (0.1, 0.1, 0.2, 0.2)
    tracker._last_crop_height = 200
    tracker.reset()
    assert tracker._template_blob is None
    assert tracker._last_bbox_px is None
    assert tracker._last_crop_height is None


def test_tracker_returns_none_on_cold_start():
    """Without a model, test that cold-start (prev_bbox=None) returns None."""
    tracker = VitTrackSOT.__new__(VitTrackSOT)
    tracker._template_blob = None
    tracker._last_bbox_px = None
    tracker._last_crop_height = None
    # No session — but track() should return None before hitting session
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = tracker.track(None, frame)
    assert result is None


def test_hann_window_properties():
    assert _HANN_WINDOW.shape == (16, 16)
    # Centre should be close to 1.0 (peak of Hann)
    assert _HANN_WINDOW[8, 8] > 0.9
    # Corners should be near 0
    assert _HANN_WINDOW[0, 0] < 0.1
    assert _HANN_WINDOW[15, 15] < 0.1
