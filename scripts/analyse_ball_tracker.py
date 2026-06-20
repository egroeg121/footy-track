"""Analyse ball detector + tracker output to characterise failure modes.

Runs the existing YOLO detector + LapTracker on representative broadcast clips
and logs per-frame ball detection confidence and tracker ID continuity.

Produces a summary: what fraction of missed-ball frames are due to
  (a) detection failure  — YOLO confidence = 0 (no ball box at all)
  (b) tracker failure    — YOLO confidence > threshold but tracker dropped it

Usage:
    uv run python scripts/analyse_ball_tracker.py
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from footy_track.constants import BALL_TAG, IN_PLAY_BALL_TAG, OUT_OF_PLAY_BALL_TAG
from footy_track.detectors.ultralytics import get_current_best_detector
from footy_track.schema import FrameDetections
from footy_track.trackers.lap import LapTracker

BALL_LABELS = {BALL_TAG, IN_PLAY_BALL_TAG, OUT_OF_PLAY_BALL_TAG}
CONF_THRESHOLD = 0.3  # min_confidence used in detector


@dataclass
class FrameRecord:
    clip: str
    frame_idx: int
    time_s: float
    ball_detected: bool  # YOLO found at least one ball
    max_ball_conf: float  # highest ball confidence in this frame (0.0 if none)
    ball_tracked: bool  # tracker emitted a ball in this frame
    track_ids: list[int]  # all ball track IDs emitted this frame


@dataclass
class ClipSummary:
    clip: str
    total_frames: int
    frames_with_ball_detected: int  # YOLO found ball
    frames_with_ball_tracked: int  # tracker emitted ball
    frames_ball_missed: int  # neither detected nor tracked
    detection_failures: int  # ball not detected (conf=0)
    tracker_failures: int  # detected but tracker dropped
    detection_failure_pct: float  # as % of missed-ball frames
    tracker_failure_pct: float  # as % of missed-ball frames
    track_id_switches: int  # number of times ball track_id changed
    unique_ball_track_ids: int  # unique track IDs assigned to ball across clip
    example_detection_failure_frames: list[int]
    example_tracker_failure_frames: list[int]


def extract_frames(
    video_path: Path, max_frames: int = 250
) -> list[tuple[int, float, Path]]:
    """Extract frames to temp directory, returning (frame_idx, time_s, path) tuples."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample evenly if clip is long, else take all frames
    step = max(1, total // max_frames)
    frames = []
    tmpdir = Path(tempfile.mkdtemp(prefix="ft_analyse_"))

    frame_idx = 0
    saved_count = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_path = tmpdir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), img)
            frames.append((frame_idx, frame_idx / fps, out_path))
            saved_count += 1
            if saved_count >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return frames


def analyse_clip(
    clip_path: Path,
    detector,
    max_frames: int = 250,
    detection_threshold: float = CONF_THRESHOLD,
) -> tuple[list[FrameRecord], ClipSummary]:
    """Run detector+tracker on a clip and return per-frame records + summary."""
    tracker = LapTracker(max_age=30, iou_threshold=0.3)
    print(f"\n[{clip_path.name}] Extracting frames...")
    frame_tuples = extract_frames(clip_path, max_frames=max_frames)
    print(
        f"[{clip_path.name}] Running detector+tracker on {len(frame_tuples)} frames..."
    )

    records: list[FrameRecord] = []
    prev_ball_track_id: int | None = None
    track_id_switches = 0
    all_ball_track_ids: set[int] = set()

    for i, (fidx, t, img_path) in enumerate(frame_tuples):
        if i % 50 == 0:
            print(f"  frame {i}/{len(frame_tuples)}")

        frame_dets: FrameDetections = detector.predict_from_path(img_path)
        tracked = tracker.update(frame_dets, t)

        # Ball detections from raw YOLO output
        ball_dets = [d for d in frame_dets.detections if d.label in BALL_LABELS]
        ball_detected = len(ball_dets) > 0
        max_conf = max((d.confidence for d in ball_dets), default=0.0)

        # Ball tracks from tracker output
        ball_tracks = [td for td in tracked if td.label in BALL_LABELS]
        ball_tracked = len(ball_tracks) > 0
        track_ids_this_frame = [td.track_id for td in ball_tracks]

        # Track ID continuity
        if ball_tracks:
            current_id = ball_tracks[0].track_id
            all_ball_track_ids.add(current_id)
            if prev_ball_track_id is not None and current_id != prev_ball_track_id:
                track_id_switches += 1
            prev_ball_track_id = current_id
        else:
            prev_ball_track_id = None

        records.append(
            FrameRecord(
                clip=clip_path.name,
                frame_idx=fidx,
                time_s=t,
                ball_detected=ball_detected,
                max_ball_conf=max_conf,
                ball_tracked=ball_tracked,
                track_ids=track_ids_this_frame,
            )
        )

        # Clean up frame file
        img_path.unlink(missing_ok=True)

    # Compute summary
    total = len(records)
    n_detected = sum(1 for r in records if r.ball_detected)
    n_tracked = sum(1 for r in records if r.ball_tracked)

    # Missed = not tracked (tracker is the final output; detected-but-not-tracked is a tracker failure)
    missed_records = [r for r in records if not r.ball_tracked]
    n_missed = len(missed_records)

    # Detection failure: not detected at all (conf=0)
    detection_failures = [r for r in missed_records if not r.ball_detected]
    # Tracker failure: detected above threshold but tracker dropped it
    tracker_failures = [
        r
        for r in missed_records
        if r.ball_detected and r.max_ball_conf >= detection_threshold
    ]

    det_fail_pct = len(detection_failures) / n_missed * 100 if n_missed else 0.0
    trk_fail_pct = len(tracker_failures) / n_missed * 100 if n_missed else 0.0

    summary = ClipSummary(
        clip=clip_path.name,
        total_frames=total,
        frames_with_ball_detected=n_detected,
        frames_with_ball_tracked=n_tracked,
        frames_ball_missed=n_missed,
        detection_failures=len(detection_failures),
        tracker_failures=len(tracker_failures),
        detection_failure_pct=round(det_fail_pct, 1),
        tracker_failure_pct=round(trk_fail_pct, 1),
        track_id_switches=track_id_switches,
        unique_ball_track_ids=len(all_ball_track_ids),
        example_detection_failure_frames=[r.frame_idx for r in detection_failures[:5]],
        example_tracker_failure_frames=[r.frame_idx for r in tracker_failures[:5]],
    )
    return records, summary


def print_summary(summary: ClipSummary) -> None:
    print(f"\n{'=' * 60}")
    print(f"Clip: {summary.clip}")
    print(f"{'=' * 60}")
    print(f"  Total frames analysed:        {summary.total_frames}")
    print(
        f"  Frames ball detected (YOLO):  {summary.frames_with_ball_detected} "
        f"({summary.frames_with_ball_detected / summary.total_frames * 100:.1f}%)"
    )
    print(
        f"  Frames ball tracked (output): {summary.frames_with_ball_tracked} "
        f"({summary.frames_with_ball_tracked / summary.total_frames * 100:.1f}%)"
    )
    print(
        f"  Frames ball MISSED:           {summary.frames_ball_missed} "
        f"({summary.frames_ball_missed / summary.total_frames * 100:.1f}%)"
    )
    print()
    print("  Missed-ball breakdown:")
    print(
        f"    Detection failures (conf=0): {summary.detection_failures} "
        f"({summary.detection_failure_pct:.1f}% of missed)"
    )
    print(
        f"    Tracker failures (det>thr):  {summary.tracker_failures} "
        f"({summary.tracker_failure_pct:.1f}% of missed)"
    )
    other = (
        summary.frames_ball_missed
        - summary.detection_failures
        - summary.tracker_failures
    )
    other_pct = (
        other / summary.frames_ball_missed * 100 if summary.frames_ball_missed else 0
    )
    print(f"    Low-conf detections dropped: {other} ({other_pct:.1f}% of missed)")
    print()
    print("  Track ID continuity:")
    print(f"    Unique ball track IDs:       {summary.unique_ball_track_ids}")
    print(f"    Track ID switches:           {summary.track_id_switches}")
    if summary.example_detection_failure_frames:
        print(
            f"  Example detection failure frames: {summary.example_detection_failure_frames}"
        )
    if summary.example_tracker_failure_frames:
        print(
            f"  Example tracker failure frames:   {summary.example_tracker_failure_frames}"
        )


def main() -> None:
    worktree_root = Path(__file__).parent.parent

    clips = [
        worktree_root / "tests/data/video/arsenal_mancity_20250925_part192.mp4",
        worktree_root / "data/arsenal_mancity_example_video.mp4",
    ]
    clips = [c for c in clips if c.exists() and c.stat().st_size > 10_000]

    if not clips:
        print("No valid clips found. Run `git lfs pull` to fetch video test data.")
        return

    print("Loading detector...")
    detector = get_current_best_detector(min_confidence=CONF_THRESHOLD, verbose=False)

    all_summaries: list[ClipSummary] = []
    all_records: list[FrameRecord] = []

    for clip in clips:
        records, summary = analyse_clip(clip, detector, max_frames=250)
        all_records.extend(records)
        all_summaries.append(summary)
        print_summary(summary)

    # Aggregate across all clips
    if len(all_summaries) > 1:
        total_frames = sum(s.total_frames for s in all_summaries)
        total_missed = sum(s.frames_ball_missed for s in all_summaries)
        total_det_fail = sum(s.detection_failures for s in all_summaries)
        total_trk_fail = sum(s.tracker_failures for s in all_summaries)
        det_fail_pct = total_det_fail / total_missed * 100 if total_missed else 0
        trk_fail_pct = total_trk_fail / total_missed * 100 if total_missed else 0

        print(f"\n{'=' * 60}")
        print("AGGREGATE ACROSS ALL CLIPS")
        print(f"{'=' * 60}")
        print(f"  Total frames:          {total_frames}")
        print(
            f"  Total missed:          {total_missed} ({total_missed / total_frames * 100:.1f}%)"
        )
        print(
            f"  Detection failures:    {total_det_fail} ({det_fail_pct:.1f}% of missed)"
        )
        print(
            f"  Tracker failures:      {total_trk_fail} ({trk_fail_pct:.1f}% of missed)"
        )

    # Save JSON results
    out_path = worktree_root / "runs" / "ball_tracker_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "summaries": [asdict(s) for s in all_summaries],
        "per_frame_records": [asdict(r) for r in all_records],
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
