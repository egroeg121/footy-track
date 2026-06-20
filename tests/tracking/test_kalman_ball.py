"""Tests for KalmanBallTracker."""

from __future__ import annotations

import pathlib

import pytest

from footy_track.schema import FrameDetections, ObjectDetection
from footy_track.trackers.kalman_ball import KalmanBallTracker


def _det(label: str, x: float, y: float, w: float = 0.02, h: float = 0.02, conf: float = 0.8) -> ObjectDetection:
    return ObjectDetection(label=label, confidence=conf, x=x, y=y, w=w, h=h)


def _fd(dets: list[ObjectDetection]) -> FrameDetections:
    return FrameDetections(uri=pathlib.Path("frame.png"), width=1920, height=1080, detections=dets)


class TestKalmanBallTrackerBasic:
    def test_no_detections_returns_empty(self):
        tracker = KalmanBallTracker()
        result = tracker.update(_fd([]), 0.0)
        assert result == []

    def test_first_ball_detection_spawns_track(self):
        tracker = KalmanBallTracker()
        result = tracker.update(_fd([_det("ball", 0.5, 0.5)]), 0.0)
        assert len(result) == 1
        assert result[0].track_id == 1
        assert result[0].label == "ball"
        assert not result[0].is_interpolated

    def test_stable_track_across_frames(self):
        tracker = KalmanBallTracker()
        track_ids = []
        for i in range(5):
            x = 0.1 + i * 0.05
            result = tracker.update(_fd([_det("ball", x, 0.5)]), float(i))
            assert len(result) == 1
            track_ids.append(result[0].track_id)
        # All frames get the same track ID
        assert len(set(track_ids)) == 1

    def test_gap_fill_emitted_on_redetection(self):
        tracker = KalmanBallTracker(max_age=5, gap_fill=True)
        # Frame 0: detect ball
        tracker.update(_fd([_det("ball", 0.3, 0.5)]), 0.0)
        # Frames 1-2: no detection (gap)
        tracker.update(_fd([]), 1.0)
        tracker.update(_fd([]), 2.0)
        # Frame 3: ball re-detected nearby
        result = tracker.update(_fd([_det("ball", 0.35, 0.5)]), 3.0)
        # Should get 2 interpolated + 1 real
        assert len(result) == 3
        interpolated = [r for r in result if r.is_interpolated]
        real = [r for r in result if not r.is_interpolated]
        assert len(interpolated) == 2
        assert len(real) == 1
        # All same track ID
        ids = {r.track_id for r in result}
        assert len(ids) == 1

    def test_gap_fill_disabled(self):
        tracker = KalmanBallTracker(max_age=5, gap_fill=False)
        tracker.update(_fd([_det("ball", 0.3, 0.5)]), 0.0)
        tracker.update(_fd([]), 1.0)
        tracker.update(_fd([]), 2.0)
        result = tracker.update(_fd([_det("ball", 0.35, 0.5)]), 3.0)
        # No gap fill — only the real detection
        assert len(result) == 1
        assert not result[0].is_interpolated

    def test_track_lost_after_max_age(self):
        tracker = KalmanBallTracker(max_age=2, gap_fill=False)
        tracker.update(_fd([_det("ball", 0.5, 0.5)]), 0.0)
        tracker.update(_fd([]), 1.0)
        tracker.update(_fd([]), 2.0)
        # age == max_age, track finalised
        tracker.update(_fd([]), 3.0)
        # Re-detect ball far away — should spawn a new track
        result = tracker.update(_fd([_det("ball", 0.5, 0.5)]), 4.0)
        assert len(result) == 1
        assert result[0].track_id == 2  # new track

    def test_finalise_returns_track_meta(self):
        tracker = KalmanBallTracker()
        tracker.update(_fd([_det("ball", 0.5, 0.5)]), 0.0)
        tracker.update(_fd([_det("ball", 0.52, 0.5)]), 1.0)
        metas = tracker.finalise()
        assert len(metas) == 1
        assert metas[0].track_id == 1
        assert metas[0].label == "ball"
        assert metas[0].start_frame == 0
        assert metas[0].end_frame == 1

    def test_in_play_ball_label_tracked(self):
        tracker = KalmanBallTracker()
        result = tracker.update(_fd([_det("in_play_ball", 0.5, 0.5)]), 0.0)
        assert len(result) == 1
        assert result[0].label == "in_play_ball"

    def test_out_of_play_ball_label_tracked(self):
        tracker = KalmanBallTracker()
        result = tracker.update(_fd([_det("out_of_play_ball", 0.5, 0.5)]), 0.0)
        assert len(result) == 1
        assert result[0].label == "out_of_play_ball"

    def test_non_ball_labels_ignored(self):
        tracker = KalmanBallTracker()
        result = tracker.update(_fd([_det("person", 0.5, 0.5)]), 0.0)
        assert result == []

    def test_picks_closest_detection_to_prediction(self):
        """When multiple ball detections exist, pick the one nearest the predicted position."""
        tracker = KalmanBallTracker(max_match_dist=1.0)
        # Seed tracker at (0.1, 0.5)
        tracker.update(_fd([_det("ball", 0.1, 0.5)]), 0.0)
        # Two balls: one close, one far
        close = _det("ball", 0.12, 0.5)
        far = _det("ball", 0.9, 0.5)
        result = tracker.update(_fd([close, far]), 1.0)
        assert len(result) == 1
        # Should pick the close detection (x ≈ 0.12)
        assert result[0].x == pytest.approx(close.x)

    def test_interpolated_positions_monotone_in_time(self):
        """Gap-filled frames should have monotonically increasing frame_index."""
        tracker = KalmanBallTracker(max_age=10, gap_fill=True)
        tracker.update(_fd([_det("ball", 0.3, 0.5)]), 0.0)
        tracker.update(_fd([]), 1.0)
        tracker.update(_fd([]), 2.0)
        tracker.update(_fd([]), 3.0)
        result = tracker.update(_fd([_det("ball", 0.4, 0.5)]), 4.0)
        frame_indices = [r.frame_index for r in result]
        assert frame_indices == sorted(frame_indices)
