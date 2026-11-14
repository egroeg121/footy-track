"""Job runner tests: fake detector, chunk segmenter, tiny synthetic video."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest

from footy_track.detectors.ultralytics import CURRENT_BEST_DETECTOR_CHECKPOINT
from footy_track.inference import jobs as jobs_mod
from footy_track.inference.fake_detector import FakeDetector
from footy_track.inference.jobs import (
    BadVideo,
    DetectorHandle,
    JobBusy,
    JobNotFound,
    JobRunner,
    fake_detector_factory,
    make_chunk_segmenter,
    make_overlay_browser_playable,
)
from footy_track.utils import get_project_root


def submit(runner: JobRunner, video: Path, **options) -> str:
    job_id, dest = runner.allocate(
        "match.mp4", ".mp4", {"max_frames": None, "stride": 1, **options}
    )
    shutil.copy(video, dest)
    runner.submit(job_id)
    return job_id


def test_full_job_writes_results_for_every_clip(runner, video_path):
    job_id = submit(runner, video_path)
    assert runner.wait_idle()
    job = runner.get(job_id)
    assert job["status"] == "done", job["error"]
    assert job["stages"]["segment"]["status"] == "done"
    assert job["stages"]["track"] == {
        "status": "done",
        "seconds": job["stages"]["track"]["seconds"],
        "clips_done": 2,
        "clips_total": 2,
    }
    assert [c["id"] for c in job["clips"]] == ["c000", "c001"]
    assert sum(c["summary"]["frames_processed"] for c in job["clips"]) == 30
    for clip in job["clips"]:
        assert clip["status"] == "done"
        for name in jobs_mod.CLIP_FILES:
            assert runner.clip_file(job_id, clip["id"], name).is_file()
        summary = json.loads(
            runner.clip_file(job_id, clip["id"], "summary.json").read_text()
        )
        assert summary["video"] == f"{clip['id']}.mp4"  # no server paths
        assert clip["summary"]["ball"]["detected"] > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_overlay_is_reencoded_to_h264(runner, video_path):
    job_id = submit(runner, video_path)
    assert runner.wait_idle()
    clip = runner.get(job_id)["clips"][0]
    assert clip["overlay"] == {"codec": "h264", "browser_playable": True}
    head = runner.clip_file(job_id, "c000", "overlay.mp4").read_bytes()[:4096]
    assert b"ftyp" in head


def test_overlay_falls_back_without_ffmpeg(tmp_path, monkeypatch):
    f = tmp_path / "overlay.mp4"
    f.write_bytes(b"not really a video")
    monkeypatch.setattr(jobs_mod.shutil, "which", lambda _name: None)
    assert make_overlay_browser_playable(f) == {
        "codec": "mp4v",
        "browser_playable": False,
    }
    assert f.read_bytes() == b"not really a video"


def test_quick_run_options_cap_frames(runner, video_path):
    job_id = submit(runner, video_path, max_frames=5, stride=2)
    assert runner.wait_idle()
    job = runner.get(job_id)
    assert [c["summary"]["frames_processed"] for c in job["clips"]] == [5, 5]


def test_no_clips_found_is_a_clear_failure(tmp_path, video_path):
    r = JobRunner(
        tmp_path / "jobs",
        detector_factory=fake_detector_factory,
        segmenter=lambda v, o, c: [],
    )
    try:
        job_id = submit(r, video_path)
        assert r.wait_idle()
        job = r.get(job_id)
        assert job["status"] == "failed"
        assert "No broadcast shots" in job["error"]
    finally:
        r.shutdown()


def test_segmenter_error_fails_job_cleanly(tmp_path, video_path):
    def boom(video, outdir, cancel):
        raise RuntimeError("ffmpeg is not installed")

    r = JobRunner(
        tmp_path / "jobs", detector_factory=fake_detector_factory, segmenter=boom
    )
    try:
        job_id = submit(r, video_path)
        assert r.wait_idle()
        job = r.get(job_id)
        assert job["status"] == "failed" and "ffmpeg" in job["error"]
        assert job["stages"]["segment"]["status"] == "failed"
    finally:
        r.shutdown()


def test_detector_load_failure_fails_before_cutting(tmp_path, video_path):
    called = []

    def bad_factory():
        raise FileNotFoundError("weights missing")

    r = JobRunner(
        tmp_path / "jobs",
        detector_factory=bad_factory,
        segmenter=lambda v, o, c: called.append(1) or [],
    )
    try:
        job_id = submit(r, video_path)
        assert r.wait_idle()
        assert "could not be loaded" in r.get(job_id)["error"]
        assert not called
    finally:
        r.shutdown()


def test_one_bad_clip_does_not_sink_the_job(tmp_path, video_path):
    calls = {"n": 0}

    class Flaky(FakeDetector):
        def predict_frames(self, frames):
            if calls["n"] == 0:
                calls["n"] += 1
                raise RuntimeError("cuda out of memory")
            return super().predict_frames(frames)

    r = JobRunner(
        tmp_path / "jobs",
        detector_factory=lambda: DetectorHandle(Flaky(), "x", "cpu", "fake"),
        segmenter=make_chunk_segmenter(15),
    )
    try:
        job_id = submit(r, video_path)
        assert r.wait_idle()
        job = r.get(job_id)
        assert job["status"] == "done"
        assert [c["status"] for c in job["clips"]] == ["failed", "done"]
        assert "out of memory" in job["clips"][0]["error"]
    finally:
        r.shutdown()


def test_bad_file_is_rejected_and_cleaned_up(runner, tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 50)
    job_id, dest = runner.allocate("junk.mp4", ".mp4", {})
    shutil.copy(junk, dest)
    with pytest.raises(BadVideo):
        runner.submit(job_id)
    assert not (runner.jobs_dir / job_id).exists()
    assert runner.list() == []


def test_unknown_extension_is_replaced_with_safe_one(runner):
    _, dest = runner.allocate("../../etc/passwd.sh", ".sh", {})
    assert dest.name == "input.mp4"
    assert dest.parent.parent == runner.jobs_dir


def test_state_survives_restart_and_running_jobs_are_marked_failed(
    tmp_path, video_path
):
    jobs = tmp_path / "jobs"
    r1 = JobRunner(
        jobs,
        detector_factory=fake_detector_factory,
        segmenter=make_chunk_segmenter(15),
    )
    job_id = submit(r1, video_path)
    assert r1.wait_idle()
    r1.shutdown()
    # forge an interrupted job
    stuck_id, _ = r1.allocate("stuck.mp4", ".mp4", {})
    stuck = r1._jobs[stuck_id]
    stuck["status"] = "running"
    stuck["stages"]["segment"]["status"] = "running"
    r1._save(stuck)
    # and a half-uploaded one
    up_id, _ = r1.allocate("half.mp4", ".mp4", {})

    r2 = JobRunner(
        jobs,
        detector_factory=fake_detector_factory,
        segmenter=make_chunk_segmenter(15),
        start_worker=False,
    )
    assert r2.get(job_id)["status"] == "done"
    assert r2.get(stuck_id)["status"] == "failed"
    assert "restarted" in r2.get(stuck_id)["error"]
    with pytest.raises(JobNotFound):
        r2.get(up_id)
    assert not (jobs / up_id).exists()


def test_cancel_running_job_then_delete(tmp_path, video_path):
    started, release = threading.Event(), threading.Event()

    class Slow(FakeDetector):
        def predict_frames(self, frames):
            started.set()
            release.wait(5)
            return super().predict_frames(frames)

    r = JobRunner(
        tmp_path / "jobs",
        detector_factory=lambda: DetectorHandle(Slow(), "x", "cpu", "fake"),
        segmenter=make_chunk_segmenter(15),
    )
    try:
        job_id = submit(r, video_path)
        assert started.wait(5)
        with pytest.raises(JobBusy):
            r.delete(job_id)
        r.cancel(job_id)
        release.set()
        assert r.wait_idle()
        assert r.get(job_id)["status"] == "cancelled"
        r.delete(job_id)
        assert not (r.jobs_dir / job_id).exists()
    finally:
        release.set()
        r.shutdown()


def test_cancel_queued_job(tmp_path, video_path):
    r = JobRunner(
        tmp_path / "jobs",
        detector_factory=fake_detector_factory,
        segmenter=make_chunk_segmenter(15),
        start_worker=False,
    )
    job_id = submit(r, video_path)
    assert r.get(job_id)["status"] == "queued"
    assert r.cancel(job_id)["status"] == "cancelled"


def test_ids_are_validated_against_traversal(runner):
    for bad in ("../x", "..", "a" * 12 + "/..", "ZZZZZZZZZZZZ", ""):
        with pytest.raises(JobNotFound):
            runner.get(bad)
        with pytest.raises(JobNotFound):
            runner.job_dir(bad)


def test_clip_file_rejects_unknown_names(runner, video_path):
    job_id = submit(runner, video_path)
    assert runner.wait_idle()
    for clip, name in [
        ("c000", "../job.json"),
        ("c000", "job.json"),
        ("../c000", "overlay.mp4"),
        ("c999", "overlay.mp4"),
    ]:
        with pytest.raises(JobNotFound):
            runner.clip_file(job_id, clip, name)


@pytest.mark.slow
@pytest.mark.needs_weights
def test_real_weights_smoke(tmp_path, video_path):  # pragma: no cover
    weights = get_project_root() / CURRENT_BEST_DETECTOR_CHECKPOINT
    if not weights.exists():
        pytest.skip("detector weights not available")
    r = JobRunner(
        tmp_path / "jobs",
        detector_factory=jobs_mod.make_real_detector_factory(weights),
        segmenter=make_chunk_segmenter(15),
    )
    try:
        job_id = submit(r, video_path, max_frames=4)
        assert r.wait_idle(600)
        assert r.get(job_id)["status"] == "done"
    finally:
        r.shutdown()
