"""Timeline maths and the stitched full video (fake detector, gappy chunk cutter)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from footy_track.inference import jobs as jobs_mod
from footy_track.inference.jobs import (
    JobRunner,
    fake_detector_factory,
    make_chunk_segmenter,
)
from footy_track.inference.stitch import compute_timeline, fps_fraction
from footy_track.inference.web.app import create_app

from .test_jobs import submit
from .test_web import wait_done

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)
FPS = 25.0


def clip(cid, start_f, n, processed=None, status="done"):
    return {
        "id": cid,
        "status": status,
        "start_s": round(start_f / FPS, 3),
        "frame_count": n,
        "summary": {"frames_processed": n if processed is None else processed},
    }


def shape(spans):
    return [(s["kind"], s["start_frame"], s["end_frame"], s["clip"]) for s in spans]


# ---------------------------------------------------------------- timeline maths


def test_timeline_no_segments_is_one_skipped_span():
    assert shape(compute_timeline([], 100, FPS)) == [("skipped", 0, 100, None)]


def test_timeline_segment_at_start_and_end_leaves_no_empty_gaps():
    spans = compute_timeline([clip("c000", 0, 40), clip("c001", 60, 40)], 100, FPS)
    assert shape(spans) == [
        ("tracked", 0, 40, "c000"),
        ("skipped", 40, 60, None),
        ("tracked", 60, 100, "c001"),
    ]


def test_timeline_gap_before_first_and_after_last():
    spans = compute_timeline([clip("c000", 10, 20)], 50, FPS)
    assert shape(spans) == [
        ("skipped", 0, 10, None),
        ("tracked", 10, 30, "c000"),
        ("skipped", 30, 50, None),
    ]
    assert spans[0]["end_s"] == 0.4 and spans[1]["start_s"] == 0.4


def test_timeline_adjacent_segments_have_no_gap_between():
    spans = compute_timeline([clip("c000", 0, 20), clip("c001", 20, 20)], 40, FPS)
    assert shape(spans) == [("tracked", 0, 20, "c000"), ("tracked", 20, 40, "c001")]


def test_timeline_partly_tracked_and_failed_clips_become_plain_footage():
    clips = [
        clip("c000", 0, 20, processed=5),  # max_frames: 5 tracked, 15 plain
        clip("c001", 20, 10, status="failed"),
        clip("c002", 30, 20, processed=4),  # stride 2 -> 8 tracked frames
    ]
    spans = compute_timeline(clips, 50, FPS, stride=2)
    assert shape(spans) == [
        ("tracked", 0, 10, "c000"),
        ("skipped", 10, 30, None),  # rest of c000 + failed c001, merged
        ("tracked", 30, 38, "c002"),
        ("skipped", 38, 50, None),
    ]


def test_timeline_spans_always_tile_the_video():
    clips = [clip("c000", 3, 7), clip("c001", 12, 1), clip("c002", 30, 5)]
    spans = compute_timeline(clips, 40, FPS)
    assert spans[0]["start_frame"] == 0 and spans[-1]["end_frame"] == 40
    for a, b in zip(spans, spans[1:], strict=False):
        assert a["end_frame"] == b["start_frame"]


def test_timeline_survives_frame_count_shorter_than_clips():
    spans = compute_timeline([clip("c000", 0, 50)], 30, FPS)
    assert shape(spans) == [("tracked", 0, 50, "c000")]


def test_fps_fraction_recovers_ntsc_rates():
    assert fps_fraction(29.97) == fps_fraction(30000 / 1001)  # rounded as in job.json
    assert str(fps_fraction(23.976)) == "24000/1001"
    assert str(fps_fraction(59.94)) == "60000/1001"
    assert str(fps_fraction(30.0)) == "30"
    assert str(fps_fraction(25.0)) == "25"


# -------------------------------------------------------------- stitched output


def frame_count(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return int(out.stdout.strip())


@pytest.fixture
def gappy_runner(tmp_path):
    r = JobRunner(
        tmp_path / "jobs",
        detector_factory=fake_detector_factory,
        segmenter=make_chunk_segmenter(chunk_frames=8, gap_frames=4),
    )
    yield r
    r.shutdown()


def test_chunk_segmenter_leaves_gaps(gappy_runner, video_path):
    job_id = submit(gappy_runner, video_path)
    assert gappy_runner.wait_idle()
    clips = gappy_runner.get(job_id)["clips"]
    assert [c["frame_count"] for c in clips] == [8, 8, 6]
    assert clips[1]["start_s"] == round(12 / FPS, 3)


@needs_ffmpeg
def test_full_video_has_every_source_frame(gappy_runner, video_path):
    job_id = submit(gappy_runner, video_path)
    assert gappy_runner.wait_idle()
    job = gappy_runner.get(job_id)
    assert job["status"] == "done", job["error"]
    assert job["stages"]["stitch"]["status"] == "done"
    assert shape(job["timeline"]) == [
        ("tracked", 0, 8, "c000"),
        ("skipped", 8, 12, None),
        ("tracked", 12, 20, "c001"),
        ("skipped", 20, 24, None),
        ("tracked", 24, 30, "c002"),
    ]
    assert job["full_video"]["frames"] == 30
    full = gappy_runner.full_video_file(job_id)
    assert frame_count(full) == frame_count(video_path) == 30


def gray_levels(path: Path) -> list[int]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    levels = []
    while True:
        ok, frame = cap.read()
        if not ok:
            return levels
        levels.append(int(frame.mean()))


@pytest.fixture
def h264_video(video_path, tmp_path) -> Path:
    """Same frames, but H.264 with B-frames like a real upload (nonzero start pts)."""
    out = tmp_path / "match_h264.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video_path),
         "-c:v", "libx264", "-bf", "2", "-crf", "12", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )  # fmt: skip
    return out


@needs_ffmpeg
def test_full_video_frames_are_the_right_source_frames(gappy_runner, h264_video):
    """Source frame i is a flat gray of i*8, so a shifted or repeated frame shows."""
    video_path = h264_video
    job_id = submit(gappy_runner, video_path)
    assert gappy_runner.wait_idle()
    full = gray_levels(gappy_runner.full_video_file(job_id))
    assert len(full) == 30
    # Gaps are plain source footage: frames 8-11 and 20-23 must be the originals.
    for i in [8, 9, 10, 11, 20, 21, 22, 23]:
        assert abs(full[i] - i * 8) <= 6, (i, full[i])


@needs_ffmpeg
def test_full_video_with_quick_run_options_keeps_frame_count(gappy_runner, video_path):
    job_id = submit(gappy_runner, video_path, max_frames=3, stride=2)
    assert gappy_runner.wait_idle()
    job = gappy_runner.get(job_id)
    assert job["status"] == "done", job["error"]
    tracked = [s for s in job["timeline"] if s["kind"] == "tracked"]
    # 3 processed frames x stride 2 = 6 source frames per clip (the last clip has 6)
    assert [s["end_frame"] - s["start_frame"] for s in tracked] == [6, 6, 6]
    assert frame_count(gappy_runner.full_video_file(job_id)) == 30


@needs_ffmpeg
def test_gap_only_video_still_stitches(tmp_path, video_path):
    r = JobRunner(
        tmp_path / "jobs",
        detector_factory=fake_detector_factory,
        segmenter=lambda *_a: [],
    )
    try:
        job_id = submit(r, video_path)
        assert r.wait_idle()
        # No clips: the job fails as before and nothing is stitched.
        job = r.get(job_id)
        assert job["status"] == "failed"
        assert job["timeline"] is None
    finally:
        r.shutdown()


def test_stitch_failure_does_not_fail_the_job(gappy_runner, video_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(jobs_mod, "stitch_full_video", boom)
    job_id = submit(gappy_runner, video_path)
    assert gappy_runner.wait_idle()
    job = gappy_runner.get(job_id)
    assert job["status"] == "done"
    assert job["stages"]["stitch"]["status"] == "failed"
    assert "exploded" in job["stages"]["stitch"]["error"]
    assert job["full_video"] is None
    assert job["timeline"]  # still recorded
    assert all(c["status"] == "done" for c in job["clips"])
    with pytest.raises(jobs_mod.JobNotFound):
        gappy_runner.full_video_file(job_id)


def test_cancel_during_stitch_cancels_the_job(gappy_runner, video_path, monkeypatch):
    def cancel_me(*_a, **_k):
        raise jobs_mod.StitchCancelled

    monkeypatch.setattr(jobs_mod, "stitch_full_video", cancel_me)
    job_id = submit(gappy_runner, video_path)
    assert gappy_runner.wait_idle()
    assert gappy_runner.get(job_id)["status"] == "cancelled"


# --------------------------------------------------------------------------- API


@needs_ffmpeg
def test_api_exposes_timeline_and_full_video(gappy_runner, video_path):
    client = TestClient(create_app(gappy_runner, max_upload_bytes=5 * 1024 * 1024))
    job_id = client.post(
        "/infer/api/jobs?name=m.mp4", content=video_path.read_bytes()
    ).json()["id"]
    job = wait_done(client, job_id)
    assert [s["kind"] for s in job["timeline"]] == [
        "tracked", "skipped", "tracked", "skipped", "tracked",
    ]  # fmt: skip
    assert {"start_s", "end_s", "kind", "clip"} <= set(job["timeline"][0])
    res = client.get(f"/infer/api/jobs/{job_id}/full.mp4")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("video/mp4")
    assert b"ftyp" in res.content[:64]


def test_api_full_video_404_when_missing_and_needs_token(gappy_runner, video_path):
    client = TestClient(
        create_app(gappy_runner, max_upload_bytes=5 * 1024 * 1024, token="s3cret")
    )
    h = {"x-footy-token": "s3cret"}
    job_id = client.post(
        "/infer/api/jobs?name=m.mp4", content=video_path.read_bytes(), headers=h
    ).json()["id"]
    gappy_runner.wait_idle()
    assert client.get(f"/infer/api/jobs/{job_id}/full.mp4").status_code == 401
    (gappy_runner.job_dir(job_id) / "full.mp4").unlink(missing_ok=True)
    assert (
        client.get(f"/infer/api/jobs/{job_id}/full.mp4", headers=h).status_code == 404
    )


@needs_ffmpeg
def test_gaps_are_frame_exact_in_a_long_gop_h264_source(tmp_path):
    """Seeking into the middle of a real encode must not shift or repeat frames."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    n = 150
    raw = tmp_path / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 90))
    for i in range(n):
        writer.write(np.full((90, 160, 3), (i * 7) % 250, dtype=np.uint8))
    writer.release()
    src = tmp_path / "long.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264",
         "-bf", "2", "-g", "250", "-crf", "12", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )  # fmt: skip
    r = JobRunner(
        tmp_path / "jobs",
        detector_factory=fake_detector_factory,
        segmenter=make_chunk_segmenter(chunk_frames=30, gap_frames=17),
    )
    try:
        job_id = submit(r, src)
        assert r.wait_idle(120)
        job = r.get(job_id)
        assert job["status"] == "done", job["error"]
        levels = gray_levels(r.full_video_file(job_id))
        assert len(levels) == n
        for span in job["timeline"]:
            if span["kind"] == "skipped":
                for i in range(span["start_frame"], span["end_frame"]):
                    assert abs(levels[i] - (i * 7) % 250) <= 6, (i, levels[i])
    finally:
        r.shutdown()
