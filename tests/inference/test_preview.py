"""Preview copies for slow networks (fake detector, no GPU, needs ffmpeg)."""

from __future__ import annotations

import json
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
from footy_track.inference.preview import make_preview
from footy_track.inference.stitch import StitchCancelled
from footy_track.inference.web.app import build_from_args, create_app

from .test_web import upload, wait_done

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)


def probe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height,pix_fmt,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    st = json.loads(out)["streams"][0]
    return {
        "frames": int(st["nb_read_frames"]),
        "w": st["width"],
        "h": st["height"],
        "pix": st["pix_fmt"],
        "rate": st["r_frame_rate"],
    }


def make_runner(tmp_path: Path, **kw) -> JobRunner:
    return JobRunner(
        tmp_path / "jobs",
        detector_factory=fake_detector_factory,
        segmenter=make_chunk_segmenter(chunk_frames=15, gap_frames=5),
        **kw,
    )


def test_make_preview_scales_down_and_keeps_frames(tmp_path):
    src = tmp_path / "big.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=25:duration=2",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],  # fmt: skip
        check=True,
    )
    dst = tmp_path / "small.mp4"
    info = make_preview(src, dst)
    a, b = probe(src), probe(dst)
    assert (b["w"], b["h"]) == (1280, 720)
    assert b["frames"] == a["frames"] == 50
    assert b["pix"] == "yuv420p" and b["rate"] == a["rate"]
    assert dst.stat().st_size < src.stat().st_size / 2
    assert info["bytes"] == dst.stat().st_size
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_make_preview_never_upscales_and_keeps_even_dims(tmp_path):
    src = tmp_path / "odd.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=300x202:rate=25:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],  # fmt: skip
        check=True,
    )
    dst = tmp_path / "p.mp4"
    make_preview(src, dst)
    b = probe(dst)
    assert (b["w"], b["h"]) == (300, 202)
    assert b["frames"] == 25


def test_make_preview_cancel_leaves_nothing(tmp_path):
    src = tmp_path / "long.mp4"  # long enough that ffmpeg is still running at the check
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=25:duration=40",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],  # fmt: skip
        check=True,
    )
    with pytest.raises(StitchCancelled):
        make_preview(src, tmp_path / "p.mp4", lambda: True)
    assert not list(tmp_path.glob("p*.mp4"))


def test_job_makes_previews_and_api_serves_them(tmp_path, video_path):
    runner = make_runner(tmp_path)
    try:
        c = TestClient(create_app(runner, token="s3cret"))
        c.cookies.set("footy_token", "s3cret")
        assert c.get("/infer/api/config").json()["preview"] is True
        job_id = upload(c, video_path).json()["id"]
        job = wait_done(c, job_id)
        assert job["status"] == "done", job["error"]
        assert job["full_preview"]["status"] == "done"
        for clip in job["clips"]:
            assert clip["preview"]["status"] == "done"
            full = runner.clip_file(job_id, clip["id"], "overlay.mp4")
            small = runner.clip_file(job_id, clip["id"], "overlay_preview.mp4")
            assert probe(small)["frames"] == probe(full)["frames"]
            assert small.stat().st_size < full.stat().st_size
            url = f"/infer/api/jobs/{job_id}/clips/{clip['id']}/overlay_preview.mp4"
            r = c.get(url)
            assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
            assert c.get(url, headers={"Range": "bytes=0-9"}).status_code == 206
        full = runner.full_video_file(job_id)
        small = runner.full_preview_file(job_id)
        assert probe(small)["frames"] == probe(full)["frames"] == 30
        assert small.stat().st_size < full.stat().st_size
        url = f"/infer/api/jobs/{job_id}/full_preview.mp4"
        assert c.get(url).status_code == 200
        anon = TestClient(create_app(runner, token="s3cret"))
        assert anon.get(url).status_code == 401
        assert (
            anon.get(
                f"/infer/api/jobs/{job_id}/clips/c000/overlay_preview.mp4"
            ).status_code
            == 401
        )
    finally:
        runner.shutdown()


def test_preview_off_makes_none(tmp_path, video_path):
    runner = make_runner(tmp_path, preview=False)
    try:
        c = TestClient(create_app(runner))
        assert c.get("/infer/api/config").json()["preview"] is False
        job_id = upload(c, video_path).json()["id"]
        job = wait_done(c, job_id)
        assert job["status"] == "done"
        assert job["full_preview"] == {"status": "off"}
        assert all(cl["preview"] == {"status": "off"} for cl in job["clips"])
        assert not list((tmp_path / "jobs" / job_id).rglob("*preview*"))
        assert c.get(f"/infer/api/jobs/{job_id}/full_preview.mp4").status_code == 404
        assert (
            c.get(
                f"/infer/api/jobs/{job_id}/clips/c000/overlay_preview.mp4"
            ).status_code
            == 404
        )
        assert c.get(f"/infer/api/jobs/{job_id}/full.mp4").status_code == 200
    finally:
        runner.shutdown()


def test_preview_failure_does_not_fail_clip_or_job(tmp_path, video_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(jobs_mod, "make_preview", boom)
    runner = make_runner(tmp_path)
    try:
        c = TestClient(create_app(runner))
        job_id = upload(c, video_path).json()["id"]
        job = wait_done(c, job_id)
        assert job["status"] == "done", job["error"]
        assert job["stages"]["stitch"]["status"] == "done"
        assert job["full_preview"]["status"] == "failed"
        assert "exploded" in job["full_preview"]["error"]
        for clip in job["clips"]:
            assert clip["status"] == "done"
            assert clip["preview"]["status"] == "failed"
        assert (
            c.get(f"/infer/api/jobs/{job_id}/clips/c000/overlay.mp4").status_code == 200
        )
        assert c.get(f"/infer/api/jobs/{job_id}/full.mp4").status_code == 200
        assert c.get(f"/infer/api/jobs/{job_id}/full_preview.mp4").status_code == 404
    finally:
        runner.shutdown()


def test_cancel_during_preview_cancels_job(tmp_path, video_path, monkeypatch):
    def cancelled(*_a, **_k):
        raise StitchCancelled

    monkeypatch.setattr(jobs_mod, "make_preview", cancelled)
    runner = make_runner(tmp_path)
    try:
        c = TestClient(create_app(runner))
        job = wait_done(c, upload(c, video_path).json()["id"])
        assert job["status"] == "cancelled"
    finally:
        runner.shutdown()


def test_old_job_without_preview_fields_still_works(tmp_path, video_path):
    runner = make_runner(tmp_path)
    try:
        c = TestClient(create_app(runner))
        job_id = upload(c, video_path).json()["id"]
        wait_done(c, job_id)
        base = tmp_path / "jobs" / job_id
        for f in base.rglob("*preview*"):
            f.unlink()
        assert c.get(f"/infer/api/jobs/{job_id}").status_code == 200
        assert (
            c.get(f"/infer/api/jobs/{job_id}/clips/c000/overlay.mp4").status_code == 200
        )
        assert c.get(f"/infer/api/jobs/{job_id}/full_preview.mp4").status_code == 404
    finally:
        runner.shutdown()


def test_cli_preview_flag_and_env(tmp_path, monkeypatch):
    jd = str(tmp_path / "j")
    monkeypatch.delenv("FOOTY_INFERENCE_PREVIEW", raising=False)
    app, args = build_from_args(["--fake", "--jobs-dir", jd])
    assert args.preview and app.state.runner.preview
    app.state.runner.shutdown()
    app, args = build_from_args(["--fake", "--jobs-dir", jd, "--no-preview"])
    assert not args.preview and not app.state.runner.preview
    app.state.runner.shutdown()
    monkeypatch.setenv("FOOTY_INFERENCE_PREVIEW", "0")
    app, args = build_from_args(["--fake", "--jobs-dir", jd])
    assert not args.preview
    app.state.runner.shutdown()
    app, args = build_from_args(["--fake", "--jobs-dir", jd, "--preview"])
    assert args.preview  # the flag wins over the environment
    app.state.runner.shutdown()
