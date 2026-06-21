"""Tests for the ball-tracking benchmark harness (ft-1my).

Uses synthetic video frames and ground-truth labels — no real video files needed.
Tests cover:
  - BBox IoU calculation
  - FrameLabel serialisation round-trip
  - EvalClip metric helpers
  - ClipMetrics computation with a perfect / zero tracker
  - Runner end-to-end with a dummy BallTracker
  - compare_methods table rendering
"""

from __future__ import annotations

import inspect
import json
import pathlib

import numpy as np
import pytest

from footy_track.ball_eval.dataset import (
    BBox,
    EvalClip,
    EvalDataset,
    FrameLabel,
    write_labels,
)
from footy_track.ball_eval.metrics import (
    FramePrediction,
    MethodResult,
    bbox_center,
    bbox_iou,
    center_dist_px,
    compute_clip_metrics,
)
from footy_track.ball_eval.runner import compare_methods, run_benchmark
from footy_track.ball_trackers.roi_yolo import RoiYoloTracker, _compute_roi

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a black RGB frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _synthetic_clip(
    tmp_dir: pathlib.Path,
    name: str = "test_clip",
    num_frames: int = 10,
    ball_bbox: BBox = (0.4, 0.3, 0.05, 0.05),
    occlusion_frames: list[int] | None = None,
    absent_frames: list[int] | None = None,
) -> EvalClip:
    """Create an EvalClip from synthetic labels (no real video)."""
    occlusion_frames = occlusion_frames or []
    absent_frames = absent_frames or []

    labels: dict[int, FrameLabel] = {}
    for i in range(num_frames):
        if i in absent_frames:
            labels[i] = FrameLabel(frame_index=i, bbox=None, tags=("ball_not_visible",))
        else:
            tags: list[str] = []
            if i in occlusion_frames:
                tags.append("occlusion")
            labels[i] = FrameLabel(frame_index=i, bbox=ball_bbox, tags=tuple(tags))

    return EvalClip(
        name=name,
        video_path=tmp_dir / f"{name}.mp4",  # path won't be opened in these tests
        labels=labels,
        total_frames=num_frames,
    )


def _make_frame_preds(
    clip: EvalClip,
    predict_fn,  # Callable[[int], BBox | None]
) -> list[FramePrediction]:
    preds = []
    for i in range(clip.total_frames):
        gt_label = clip.labels.get(i)
        gt_bbox = gt_label.bbox if gt_label else None
        pred_bbox = predict_fn(i)
        iou = bbox_iou(gt_bbox, pred_bbox) if gt_bbox and pred_bbox else None
        preds.append(
            FramePrediction(
                frame_index=i,
                gt_bbox=gt_bbox,
                pred_bbox=pred_bbox,
                iou=iou,
                inference_time_s=0.001,
            )
        )
    return preds


# --------------------------------------------------------------------------- #
# IoU tests                                                                    #
# --------------------------------------------------------------------------- #


def test_iou_perfect_overlap():
    box = (0.1, 0.1, 0.3, 0.3)
    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = (0.0, 0.0, 0.2, 0.2)
    b = (0.5, 0.5, 0.2, 0.2)
    assert bbox_iou(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    a = (0.0, 0.0, 0.4, 0.4)
    b = (0.2, 0.2, 0.4, 0.4)
    # intersection: 0.2x0.2=0.04; union: 0.16+0.16-0.04=0.28
    assert bbox_iou(a, b) == pytest.approx(0.04 / 0.28, rel=1e-4)


def test_iou_adjacent_boxes():
    a = (0.0, 0.0, 0.3, 0.3)
    b = (0.3, 0.0, 0.3, 0.3)  # touches but doesn't overlap
    assert bbox_iou(a, b) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# FrameLabel serialisation                                                     #
# --------------------------------------------------------------------------- #


def test_frame_label_round_trip():
    lbl = FrameLabel(
        frame_index=5, bbox=(0.1, 0.2, 0.05, 0.04), tags=("occlusion", "small_ball")
    )
    restored = FrameLabel.from_dict(lbl.to_dict())
    assert restored.frame_index == lbl.frame_index
    assert restored.bbox == pytest.approx(lbl.bbox)
    assert restored.tags == lbl.tags


def test_frame_label_absent_ball():
    lbl = FrameLabel(frame_index=3, bbox=None, tags=("ball_not_visible",))
    restored = FrameLabel.from_dict(lbl.to_dict())
    assert restored.bbox is None


def test_write_labels_round_trip(tmp_path):
    labels = [
        FrameLabel(frame_index=0, bbox=(0.3, 0.3, 0.05, 0.05), tags=()),
        FrameLabel(frame_index=1, bbox=None, tags=("ball_not_visible",)),
        FrameLabel(frame_index=2, bbox=(0.31, 0.31, 0.05, 0.05), tags=("occlusion",)),
    ]
    path = tmp_path / "test.jsonl"
    write_labels(labels, path)

    restored = []
    with path.open() as f:
        for line in f:
            restored.append(FrameLabel.from_dict(json.loads(line)))

    assert len(restored) == 3
    assert restored[0].bbox == pytest.approx(labels[0].bbox)
    assert restored[1].bbox is None
    assert "occlusion" in restored[2].tags


# --------------------------------------------------------------------------- #
# EvalClip helpers                                                             #
# --------------------------------------------------------------------------- #


def test_eval_clip_ball_present_count(tmp_path):
    clip = _synthetic_clip(tmp_path, absent_frames=[2, 5])
    assert clip.ball_present_count() == 8  # 10 - 2 absent


def test_eval_clip_frames_with_tag(tmp_path):
    clip = _synthetic_clip(tmp_path, num_frames=8, occlusion_frames=[1, 3])
    assert clip.frames_with_tag("occlusion") == [1, 3]
    assert clip.frames_with_tag("motion_blur") == []


# --------------------------------------------------------------------------- #
# EvalDataset JSONL loading                                                    #
# --------------------------------------------------------------------------- #


def test_eval_dataset_from_dir_no_clips(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    with pytest.raises(ValueError, match="No labelled clips"):
        EvalDataset.from_dir(clips_dir)


def test_eval_dataset_from_dir_missing_video(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "myclip.jsonl").write_text(
        '{"frame_index": 0, "bbox": [0.1, 0.2, 0.03, 0.03], "tags": []}\n'
    )
    with pytest.raises(FileNotFoundError, match="No video file found"):
        EvalDataset.from_dir(clips_dir)


# --------------------------------------------------------------------------- #
# compute_clip_metrics                                                         #
# --------------------------------------------------------------------------- #


def test_perfect_tracker_metrics(tmp_path):
    clip = _synthetic_clip(tmp_path, num_frames=10, occlusion_frames=[3])
    bbox = (0.4, 0.3, 0.05, 0.05)
    preds = _make_frame_preds(clip, lambda i: bbox)
    cm = compute_clip_metrics(
        clip_name=clip.name,
        total_frames=clip.total_frames,
        ball_present_frames=clip.ball_present_count(),
        frame_preds=preds,
        total_inference_s=1.0,
        peak_vram_mb=0.0,
        occlusion_frame_indices=clip.frames_with_tag("occlusion"),
    )
    assert cm.mean_iou == pytest.approx(1.0)
    assert cm.recall_pct == pytest.approx(100.0)
    assert cm.precision_pct == pytest.approx(100.0)
    assert cm.tracking_failures == 0
    assert cm.occlusion_recovery_rate == pytest.approx(100.0)


def test_zero_tracker_metrics(tmp_path):
    clip = _synthetic_clip(tmp_path, num_frames=10)
    preds = _make_frame_preds(clip, lambda i: None)
    cm = compute_clip_metrics(
        clip_name=clip.name,
        total_frames=clip.total_frames,
        ball_present_frames=clip.ball_present_count(),
        frame_preds=preds,
        total_inference_s=1.0,
        peak_vram_mb=0.0,
        occlusion_frame_indices=[],
    )
    assert cm.recall_pct == pytest.approx(0.0)
    assert cm.tracking_failures == 10


def test_partial_tracker_metrics(tmp_path):
    clip = _synthetic_clip(tmp_path, num_frames=10, absent_frames=[9])
    bbox = (0.4, 0.3, 0.05, 0.05)
    # Tracker gets it right on even frames, misses on odd frames
    preds = _make_frame_preds(clip, lambda i: bbox if i % 2 == 0 else None)
    cm = compute_clip_metrics(
        clip_name=clip.name,
        total_frames=clip.total_frames,
        ball_present_frames=clip.ball_present_count(),
        frame_preds=preds,
        total_inference_s=2.0,
        peak_vram_mb=500.0,
        occlusion_frame_indices=[],
    )
    # 9 ball-present frames, 5 even (0,2,4,6,8) predicted
    assert cm.recall_pct == pytest.approx(100.0 * 5 / 9, rel=0.01)
    assert cm.fps == pytest.approx(10 / 2.0)
    assert cm.peak_vram_mb == pytest.approx(500.0)


def test_occlusion_recovery_rate(tmp_path):
    clip = _synthetic_clip(tmp_path, num_frames=10, occlusion_frames=[2, 6])
    bbox = (0.4, 0.3, 0.05, 0.05)

    # Tracker resumes after occlusion at frame 2 (frame 3 has prediction)
    # but FAILS to resume after occlusion at frame 6 (frames 7,8,9 all None)
    def predict(i: int):
        if i in (7, 8, 9):
            return None
        return bbox

    preds = _make_frame_preds(clip, predict)
    cm = compute_clip_metrics(
        clip_name=clip.name,
        total_frames=clip.total_frames,
        ball_present_frames=clip.ball_present_count(),
        frame_preds=preds,
        total_inference_s=1.0,
        peak_vram_mb=0.0,
        occlusion_frame_indices=[2, 6],
    )
    # 1 of 2 occlusion events recovered
    assert cm.occlusion_recovery_rate == pytest.approx(50.0)


# --------------------------------------------------------------------------- #
# MethodResult.table()                                                         #
# --------------------------------------------------------------------------- #


def test_method_result_table_contains_method_name(tmp_path):
    clip = _synthetic_clip(tmp_path, num_frames=5)
    preds = _make_frame_preds(clip, lambda i: (0.4, 0.3, 0.05, 0.05))
    cm = compute_clip_metrics(
        clip_name=clip.name,
        total_frames=clip.total_frames,
        ball_present_frames=clip.ball_present_count(),
        frame_preds=preds,
        total_inference_s=0.5,
        peak_vram_mb=0.0,
        occlusion_frame_indices=[],
    )
    result = MethodResult(method_name="test_method", clip_metrics=[cm])
    table = result.table()
    assert "test_method" in table
    assert "IoU" in table


# --------------------------------------------------------------------------- #
# compare_methods                                                              #
# --------------------------------------------------------------------------- #


def test_compare_methods_sorted(tmp_path):
    clip = _synthetic_clip(tmp_path, num_frames=5)
    bbox = (0.4, 0.3, 0.05, 0.05)

    def _result(name: str, predict_fn) -> MethodResult:
        preds = _make_frame_preds(clip, predict_fn)
        cm = compute_clip_metrics(
            clip_name=clip.name,
            total_frames=clip.total_frames,
            ball_present_frames=clip.ball_present_count(),
            frame_preds=preds,
            total_inference_s=1.0,
            peak_vram_mb=0.0,
            occlusion_frame_indices=[],
        )
        return MethodResult(method_name=name, clip_metrics=[cm])

    perfect = _result("method_perfect", lambda i: bbox)
    zero = _result("method_zero", lambda i: None)

    table = compare_methods([zero, perfect], sort_by="mean_iou")
    # After sorting by IoU descending, perfect should appear first
    assert table.index("method_perfect") < table.index("method_zero")


# --------------------------------------------------------------------------- #
# run_benchmark with a dummy tracker (no real video)                          #
# --------------------------------------------------------------------------- #


class _ConstantTracker:
    """Always returns the same fixed bbox. Used to test the runner scaffold."""

    def __init__(self, bbox: BBox) -> None:
        self._bbox = bbox
        self._last_crop_height = None

    def track(self, prev_bbox: BBox | None, frame: np.ndarray) -> BBox | None:
        self._last_crop_height = frame.shape[0]
        return self._bbox

    def reset(self) -> None:
        pass


class _SyntheticEvalDataset(EvalDataset):
    """EvalDataset that yields pre-baked frames instead of loading video."""

    def __init__(self, clip: EvalClip, frames: list[np.ndarray]) -> None:
        super().__init__([clip])
        self._frames = frames
        # Monkey-patch clip.iter_frames to return synthetic frames
        frame_data = list(enumerate(frames))

        def _iter():
            yield from frame_data

        clip.iter_frames = _iter  # type: ignore[method-assign]


def test_run_benchmark_perfect_tracker(tmp_path):
    bbox: BBox = (0.4, 0.3, 0.05, 0.05)
    clip = _synthetic_clip(tmp_path, num_frames=6, occlusion_frames=[2])
    frames = [_make_frame() for _ in range(6)]
    dataset = _SyntheticEvalDataset(clip, frames)

    tracker = _ConstantTracker(bbox)
    result = run_benchmark(tracker, dataset, method_name="constant", verbose=False)

    assert result.method_name == "constant"
    assert len(result.clip_metrics) == 1
    assert result.mean_iou == pytest.approx(1.0)
    assert result.mean_recall == pytest.approx(100.0)


def test_run_benchmark_effective_resolution_captured(tmp_path):
    bbox: BBox = (0.4, 0.3, 0.05, 0.05)
    clip = _synthetic_clip(tmp_path, num_frames=4)
    frames = [_make_frame(h=128, w=128) for _ in range(4)]
    dataset = _SyntheticEvalDataset(clip, frames)

    tracker = _ConstantTracker(bbox)
    result = run_benchmark(tracker, dataset, method_name="res_test", verbose=False)

    cm = result.clip_metrics[0]
    assert cm.effective_resolution_px == 128


# --------------------------------------------------------------------------- #
# Center-distance and new metrics                                              #
# --------------------------------------------------------------------------- #


def test_bbox_center():
    assert bbox_center((0.4, 0.3, 0.1, 0.1)) == pytest.approx((0.45, 0.35))


def test_center_dist_px_same_point():
    assert center_dist_px((0.5, 0.5), (0.5, 0.5), 1080, 1920) == pytest.approx(0.0)


def test_center_dist_px_known_distance():
    # 0.1 * 1920 = 192 px horizontal offset
    d = center_dist_px((0.0, 0.5), (0.1, 0.5), 1080, 1920)
    assert d == pytest.approx(192.0)


def test_frame_label_center_only_round_trip():
    lbl = FrameLabel(frame_index=10, bbox=None, tags=(), center=(0.45, 0.32))
    restored = FrameLabel.from_dict(lbl.to_dict())
    assert restored.center == pytest.approx((0.45, 0.32))
    assert restored.bbox is None
    assert restored.is_ball_visible()


def test_frame_label_ball_center_from_bbox():
    lbl = FrameLabel(frame_index=0, bbox=(0.4, 0.3, 0.1, 0.1), tags=())
    ctr = lbl.ball_center()
    assert ctr == pytest.approx((0.45, 0.35))


def test_frame_label_ball_center_explicit_wins():
    lbl = FrameLabel(
        frame_index=0, bbox=(0.4, 0.3, 0.1, 0.1), tags=(), center=(0.9, 0.9)
    )
    ctr = lbl.ball_center()
    assert ctr == pytest.approx((0.9, 0.9))


def test_frame_label_is_ball_visible_absent():
    lbl = FrameLabel(frame_index=0, bbox=None, tags=("ball_not_visible",))
    assert not lbl.is_ball_visible()


def test_perfect_tracker_center_metrics(tmp_path):
    """A perfect tracker should score 100% center_within_radius."""
    bbox: BBox = (0.4, 0.3, 0.05, 0.05)
    clip = _synthetic_clip(tmp_path, num_frames=10)
    preds = _make_frame_preds(clip, lambda i: bbox)
    cm = compute_clip_metrics(
        clip_name=clip.name,
        total_frames=clip.total_frames,
        ball_present_frames=clip.ball_present_count(),
        frame_preds=preds,
        total_inference_s=1.0,
        peak_vram_mb=0.0,
        occlusion_frame_indices=[],
        frame_size=(1080, 1920),
    )
    assert cm.center_within_radius_pct == pytest.approx(100.0)
    assert cm.mean_center_dist_px == pytest.approx(0.0)
    assert cm.catastrophic_failure_rate == pytest.approx(0.0)
    assert cm.max_track_streak == 10


def test_catastrophic_failure_detected(tmp_path):
    """A wildly wrong prediction counts as catastrophic."""
    wrong_bbox: BBox = (0.9, 0.9, 0.05, 0.05)  # far corner
    clip = _synthetic_clip(tmp_path, num_frames=5)
    preds = _make_frame_preds(clip, lambda i: wrong_bbox)
    cm = compute_clip_metrics(
        clip_name=clip.name,
        total_frames=clip.total_frames,
        ball_present_frames=clip.ball_present_count(),
        frame_preds=preds,
        total_inference_s=1.0,
        peak_vram_mb=0.0,
        occlusion_frame_indices=[],
        frame_size=(1080, 1920),
    )
    # Distance between (0.425,0.325) and (0.925,0.925) should be >> CATASTROPHIC_DIST_PX
    assert cm.catastrophic_failure_rate == pytest.approx(100.0)


def test_track_streak_resets_on_failure(tmp_path):
    """Streak resets when tracker leaves the radius."""
    bbox: BBox = (0.4, 0.3, 0.05, 0.05)
    wrong_bbox: BBox = (0.9, 0.9, 0.05, 0.05)

    # frames 0-3 correct, frame 4 wrong, frames 5-9 correct → streak 5
    def _predict(i):
        return wrong_bbox if i == 4 else bbox

    clip = _synthetic_clip(tmp_path, num_frames=10)
    preds = _make_frame_preds(clip, _predict)
    cm = compute_clip_metrics(
        clip_name=clip.name,
        total_frames=clip.total_frames,
        ball_present_frames=clip.ball_present_count(),
        frame_preds=preds,
        total_inference_s=1.0,
        peak_vram_mb=0.0,
        occlusion_frame_indices=[],
        frame_size=(1080, 1920),
    )
    assert cm.max_track_streak == 5  # frames 5-9


def test_center_only_gt_in_benchmark(tmp_path):
    """Center-only labels (no bbox) work in the runner."""
    center_bbox: BBox = (0.4, 0.3, 0.05, 0.05)
    # Label frames with center-only (no bbox)
    labels: dict[int, FrameLabel] = {
        i: FrameLabel(frame_index=i, bbox=None, tags=(), center=(0.425, 0.325))
        for i in range(6)
    }
    clip = EvalClip(
        name="center_clip",
        video_path=tmp_path / "center_clip.mp4",
        labels=labels,
        total_frames=6,
    )
    frames = [_make_frame(h=128, w=128) for _ in range(6)]
    dataset = _SyntheticEvalDataset(clip, frames)

    tracker = _ConstantTracker(center_bbox)
    result = run_benchmark(tracker, dataset, method_name="center_test", verbose=False)

    cm = result.clip_metrics[0]
    # Prediction center = (0.425, 0.325), GT center = (0.425, 0.325) → 0px error
    assert cm.mean_center_dist_px == pytest.approx(0.0, abs=1.0)
    assert cm.center_within_radius_pct == pytest.approx(100.0)


def test_compare_methods_sorted_by_center(tmp_path):
    """compare_methods defaults to center_within_radius_pct sort."""
    clip = _synthetic_clip(tmp_path, num_frames=5)
    bbox = (0.4, 0.3, 0.05, 0.05)

    def _result(name: str, predict_fn) -> MethodResult:
        preds = _make_frame_preds(clip, predict_fn)
        cm = compute_clip_metrics(
            clip_name=clip.name,
            total_frames=clip.total_frames,
            ball_present_frames=clip.ball_present_count(),
            frame_preds=preds,
            total_inference_s=1.0,
            peak_vram_mb=0.0,
            occlusion_frame_indices=[],
        )
        return MethodResult(method_name=name, clip_metrics=[cm])

    perfect = _result("method_perfect", lambda i: bbox)
    zero = _result("method_zero", lambda i: None)

    table = compare_methods([zero, perfect])
    # perfect has 100% center accuracy, should appear first
    assert table.index("method_perfect") < table.index("method_zero")


def test_write_labels_center_field(tmp_path):
    """write_labels preserves center field in round-trip."""
    labels = [
        FrameLabel(frame_index=0, bbox=None, tags=(), center=(0.45, 0.32)),
        FrameLabel(frame_index=5, bbox=(0.4, 0.3, 0.05, 0.05), tags=("occlusion",)),
    ]
    path = tmp_path / "centers.jsonl"
    write_labels(labels, path)

    restored = []
    with path.open() as f:
        for line in f:
            restored.append(FrameLabel.from_dict(json.loads(line)))

    assert restored[0].center == pytest.approx((0.45, 0.32))
    assert restored[0].bbox is None
    assert restored[1].center is None
    assert restored[1].bbox is not None


# --------------------------------------------------------------------------- #
# GT-seeding tests (ft-019)                                                   #
# --------------------------------------------------------------------------- #


class _RecordingTracker:
    """Tracker that records what prev_bbox it received and optionally fails."""

    def __init__(self, bbox: BBox, fail_on_frames: set[int] | None = None) -> None:
        self._bbox = bbox
        self._fail_on_frames = fail_on_frames or set()
        self._last_crop_height = None
        self.calls: list[
            tuple[int, BBox | None]
        ] = []  # (frame_idx, received_prev_bbox)
        self._call_count = 0

    def track(self, prev_bbox: BBox | None, frame: np.ndarray) -> BBox | None:
        self._last_crop_height = frame.shape[0]
        idx = self._call_count
        self.calls.append((idx, prev_bbox))
        self._call_count += 1
        return None if idx in self._fail_on_frames else self._bbox

    def reset(self) -> None:
        self._call_count = 0
        self.calls.clear()


def test_runner_seeds_from_gt_on_first_frame(tmp_path):
    """Harness must inject GT bbox as prev_bbox on frame 0."""
    bbox: BBox = (0.4, 0.3, 0.05, 0.05)
    clip = _synthetic_clip(tmp_path, num_frames=4)
    frames = [_make_frame() for _ in range(4)]
    dataset = _SyntheticEvalDataset(clip, frames)

    tracker = _RecordingTracker(bbox)
    run_benchmark(tracker, dataset, method_name="seed_test", verbose=False)

    # Frame 0: tracker should have received GT bbox (not None)
    _, received = tracker.calls[0]
    assert received is not None, "Frame 0 prev_bbox must be seeded from GT"
    assert received == pytest.approx(bbox)


def test_runner_reseeds_after_failure(tmp_path):
    """After tracker returns None, next frame with GT must re-seed."""
    bbox: BBox = (0.4, 0.3, 0.05, 0.05)
    clip = _synthetic_clip(tmp_path, num_frames=5)
    frames = [_make_frame() for _ in range(5)]
    dataset = _SyntheticEvalDataset(clip, frames)

    # Tracker fails on frame 1 (returns None)
    tracker = _RecordingTracker(bbox, fail_on_frames={1})
    run_benchmark(tracker, dataset, method_name="reseed_test", verbose=False)

    # Frame 2 must receive GT seed because frame 1 produced None
    _, frame2_prev = tracker.calls[2]
    assert frame2_prev is not None, (
        "Frame 2 must be re-seeded from GT after failure on frame 1"
    )


def test_runner_no_seed_when_tracking_succeeds(tmp_path):
    """When tracking is continuous, only frame 0 should receive a GT seed."""
    bbox: BBox = (0.4, 0.3, 0.05, 0.05)
    clip = _synthetic_clip(tmp_path, num_frames=5)
    frames = [_make_frame() for _ in range(5)]
    dataset = _SyntheticEvalDataset(clip, frames)

    tracker = _RecordingTracker(bbox)
    run_benchmark(tracker, dataset, method_name="noseed_test", verbose=False)

    # Frames 1-4 should receive tracker's own previous output (the fixed bbox), not GT
    for call_idx in range(1, 5):
        _, received = tracker.calls[call_idx]
        assert received == pytest.approx(bbox), (
            f"Frame {call_idx} should have received tracker output, not a GT seed"
        )


def test_seeded_frame_flag_set_on_seed_frames(tmp_path):
    """FramePrediction.seeded_from_gt must be True only on seeded frames."""
    bbox: BBox = (0.4, 0.3, 0.05, 0.05)
    clip = _synthetic_clip(tmp_path, num_frames=4)
    frames = [_make_frame() for _ in range(4)]
    dataset = _SyntheticEvalDataset(clip, frames)

    tracker = _RecordingTracker(bbox, fail_on_frames={1})
    result = run_benchmark(tracker, dataset, method_name="flag_test", verbose=False)

    preds = result.clip_metrics[0].frame_predictions
    # Frame 0: seeded (initial seed)
    assert preds[0].seeded_from_gt is True
    # Frame 1: NOT seeded (prev_bbox was available from frame 0's output)
    assert preds[1].seeded_from_gt is False
    # Frame 2: seeded (re-seed after frame 1 failure)
    assert preds[2].seeded_from_gt is True
    # Frame 3: NOT seeded (frame 2 produced tracker output)
    assert preds[3].seeded_from_gt is False


# --------------------------------------------------------------------------- #
# ROI crop sizing tests (ft-019)                                              #
# --------------------------------------------------------------------------- #


def test_roi_yolo_default_min_crop_is_512():
    """RoiYoloTracker must default to min_roi_px=512."""
    # Check via signature — does not instantiate the model
    sig = inspect.signature(RoiYoloTracker.__init__)
    default = sig.parameters["min_roi_px"].default
    assert default == 512, f"min_roi_px default must be 512, got {default}"


def test_roi_yolo_compute_roi_respects_absolute_min():
    """_compute_roi must enforce min_roi_px floor regardless of bbox size."""
    # Small bbox, 1080p frame, roi_scale=3.0, min_roi_frac=0.0 (no frac floor),
    # min_roi_px=512 — ROI side must be >= 512 even if bbox diagonal is tiny
    tiny_bbox: BBox = (0.5, 0.5, 0.005, 0.005)  # ~5x5 px ball
    x0, y0, x1, y1 = _compute_roi(
        prev_bbox=tiny_bbox,
        pred_cx=0.5,
        pred_cy=0.5,
        H=1080,
        W=1920,
        roi_scale=3.0,
        min_roi_frac=0.0,
        min_roi_px=512,
    )
    roi_side = max(x1 - x0, y1 - y0)
    assert roi_side >= 512, f"ROI side {roi_side} must be >= 512 when min_roi_px=512"
