"""API tests with FastAPI's TestClient: fake detector, chunk segmenter."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from footy_track.inference.fake_detector import FakeDetector
from footy_track.inference.web.app import build_from_args, create_app


@pytest.fixture
def client(runner):
    return TestClient(create_app(runner, max_upload_bytes=5 * 1024 * 1024))


def upload(client, video_path, query="name=match.mp4", **kw):
    return client.post(
        f"/infer/api/jobs?{query}", content=video_path.read_bytes(), **kw
    )


def wait_done(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/infer/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_upload_to_results_flow(client, video_path):
    res = upload(client, video_path)
    assert res.status_code == 201
    job_id = res.json()["id"]
    job = wait_done(client, job_id)
    assert job["status"] == "done", job["error"]
    assert len(job["clips"]) == 2

    listing = client.get("/infer/api/jobs").json()
    assert [j["id"] for j in listing] == [job_id]
    assert listing[0]["clips"] == 2

    base = f"/infer/api/jobs/{job_id}/clips/c000"
    for name, ctype in [
        ("overlay.mp4", "video/mp4"),
        ("clip.mp4", "video/mp4"),
        ("detections.jsonl", "application/x-ndjson"),
        ("tracklets.jsonl", "application/x-ndjson"),
        ("ball_track.jsonl", "application/x-ndjson"),
        ("summary.json", "application/json"),
    ]:
        r = client.get(f"{base}/{name}")
        assert r.status_code == 200, name
        assert r.headers["content-type"].startswith(ctype)
    assert (
        client.get(f"{base}/overlay.mp4", headers={"Range": "bytes=0-9"}).status_code
        == 206
    )

    for path in (
        "/infer",
        "/infer/jobs",
        f"/infer/jobs/{job_id}",
        "/infer/static/infer.js",
    ):
        assert client.get(path).status_code == 200
    page = client.get(f"/infer/jobs/{job_id}").text
    assert "recall about 0.6" in page and "no team" in page and "restart" in page

    assert client.delete(f"/infer/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/infer/api/jobs/{job_id}").status_code == 404
    assert client.get("/infer/api/jobs").json() == []


def test_quick_run_fields(client, video_path):
    job_id = upload(client, video_path, "name=a.mp4&max_frames=3&stride=2").json()["id"]
    job = wait_done(client, job_id)
    assert job["options"] == {"max_frames": 3, "stride": 2}
    assert job["clips"][0]["summary"]["frames_processed"] == 3
    assert upload(client, video_path, "name=a.mp4&stride=0").status_code == 422


def test_bad_file_is_a_400_and_leaves_nothing(client, runner):
    res = client.post(
        "/infer/api/jobs?name=x.mp4", content=b"definitely not video" * 100
    )
    assert res.status_code == 400
    assert client.get("/infer/api/jobs").json() == []
    assert list(runner.jobs_dir.iterdir()) == []
    assert client.post("/infer/api/jobs", content=b"").status_code == 400


def test_upload_size_limit(runner, video_path):
    small = TestClient(create_app(runner, max_upload_bytes=1000))
    res = upload(small, video_path)
    assert res.status_code == 413
    assert list(runner.jobs_dir.iterdir()) == []


def test_traversal_attempts_are_rejected(client, video_path):
    job_id = upload(client, video_path).json()["id"]
    wait_done(client, job_id)
    bad = [
        "/infer/api/jobs/..%2F..%2Fetc/clips/c000/overlay.mp4",
        "/infer/api/jobs/..",
        "/infer/api/jobs/%2e%2e",
        f"/infer/api/jobs/{job_id}/clips/..%2F/overlay.mp4",
        f"/infer/api/jobs/{job_id}/clips/c000/..%2Fjob.json",
        f"/infer/api/jobs/{job_id}/clips/c000/job.json",
        f"/infer/api/jobs/{job_id}/clips/c000/%2e%2e%2f%2e%2e%2fjob.json",
        f"/infer/api/jobs/{job_id}/clips/c999/overlay.mp4",
        f"/infer/api/jobs/{job_id}/clips/c0/overlay.mp4",
    ]
    for path in bad:
        assert client.get(path).status_code in (400, 404), path
    assert client.delete("/infer/api/jobs/..%2F..").status_code in (404, 405)
    assert client.post("/infer/api/jobs/nope/cancel").status_code == 404


def test_hostile_filename_is_only_a_display_name(client, runner, video_path):
    res = upload(client, video_path, "name=..%2F..%2Fevil%00.sh")
    assert res.status_code == 201
    job = res.json()
    assert "/" not in job["name"] and "\x00" not in job["name"]
    files = {p.name for p in (runner.jobs_dir / job["id"]).iterdir()}
    assert "input.mp4" in files
    wait_done(client, job["id"])
    assert not any(p.suffix == ".sh" for p in runner.jobs_dir.rglob("*"))


def test_delete_running_job_conflicts_then_cancel(runner, video_path, monkeypatch):
    runner.shutdown()  # no worker: the job stays queued
    r2 = type(runner)(
        runner.jobs_dir / "q",
        detector_factory=runner.detector_factory,
        segmenter=runner.segmenter,
        start_worker=False,
    )
    c = TestClient(create_app(r2))
    job_id = upload(c, video_path).json()["id"]
    assert c.delete(f"/infer/api/jobs/{job_id}").status_code == 409
    assert c.post(f"/infer/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    assert c.delete(f"/infer/api/jobs/{job_id}").status_code == 204


def test_token_protects_api_and_downloads(runner, video_path):
    c = TestClient(create_app(runner, token="s3cret"))
    assert c.get("/infer/api/jobs").status_code == 401
    assert c.get("/infer/api/jobs?token=wrong").status_code == 401
    assert (
        c.get("/infer/api/jobs", headers={"Authorization": "Bearer s3cret"}).status_code
        == 200
    )
    assert c.get("/infer/api/jobs?token=s3cret").status_code == 200
    assert upload(c, video_path).status_code == 401
    c.cookies.set("footy_token", "s3cret")  # what the pages do, for <video> and links
    job_id = upload(c, video_path).json()["id"]
    wait_done(c, job_id)
    assert c.get(f"/infer/api/jobs/{job_id}/clips/c000/overlay.mp4").status_code == 200
    assert c.get("/infer").status_code == 200  # pages hold no data


def test_cli_fake_mode_builds_app(tmp_path):
    app, args = build_from_args(["--fake", "--jobs-dir", str(tmp_path / "j")])
    assert args.fake
    with TestClient(app) as c:
        assert c.get("/infer/api/config").json()["fake"] is True
    app.state.runner.shutdown()


def test_cli_errors_when_weights_missing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        build_from_args(
            ["--weights", str(tmp_path / "nope.pt"), "--jobs-dir", str(tmp_path)]
        )
    assert "weights not found" in capsys.readouterr().err


def test_fake_detector_stays_in_frame_on_long_clips():
    frames = [np.zeros((9, 16, 3), np.uint8)] * 600
    for fd in FakeDetector().predict_frames(frames):
        assert all(0 <= d.x <= 1 for d in fd.detections)


def test_demo_disabled_by_default(client):
    assert client.get("/infer/api/config").json()["demo"] is False
    assert client.post("/infer/api/demo").status_code == 404


def test_demo_creates_job_without_leaking_path(runner, video_path):
    c = TestClient(create_app(runner, demo_clip=video_path))
    cfg = c.get("/infer/api/config").json()
    assert cfg["demo"] is True and cfg["demo_name"] == "match.mp4"
    assert str(video_path.parent) not in str(cfg)
    res = c.post("/infer/api/demo?max_frames=20")
    assert res.status_code == 201
    assert str(video_path.parent) not in res.text
    job = wait_done(c, res.json()["id"])
    assert job["status"] == "done", job["error"]
    assert job["options"]["max_frames"] == 20
    assert video_path.exists()  # copied, not moved


def test_demo_requires_token(runner, video_path):
    c = TestClient(create_app(runner, token="s3cret", demo_clip=video_path))
    assert c.post("/infer/api/demo").status_code == 401
    res = c.post("/infer/api/demo", headers={"Authorization": "Bearer s3cret"})
    assert res.status_code == 201


def test_demo_missing_file_hidden(runner, tmp_path):
    c = TestClient(create_app(runner, demo_clip=tmp_path / "gone.mp4"))
    assert c.get("/infer/api/config").json()["demo"] is False
    assert c.post("/infer/api/demo").status_code == 404
    assert c.get("/infer/api/jobs").json() == []


def test_demo_clip_env_and_flag(tmp_path, video_path, monkeypatch):
    monkeypatch.setenv("FOOTY_INFERENCE_DEMO_CLIP", str(video_path))
    _, args = build_from_args(["--fake", "--jobs-dir", str(tmp_path / "j")])
    assert args.demo_clip == video_path
