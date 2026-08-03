"""Ingest endpoint tests (docs/labeller_requirements.md §3, Ingest).

The SSE run endpoint is only exercised on its no-subprocess error path — the
happy path shells out to split_broadcast_segments on a real video, which is
out of bounds for unit tests.
"""

from __future__ import annotations

from .conftest import patch_labeller_attr


def test_upload_saves_file_and_returns_metadata(client, tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    patch_labeller_attr(monkeypatch, "_INGEST_UPLOADS", uploads)

    r = client.post(
        "/ingest/upload",
        files={"file": ("match.mp4", b"fake-video-bytes", "video/mp4")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "match.mp4"
    assert data["size"] == len(b"fake-video-bytes")
    assert data["path"] == str(uploads / "match.mp4")
    assert (uploads / "match.mp4").read_bytes() == b"fake-video-bytes"


def test_ingest_run_missing_file_streams_error_and_done(client, tmp_path):
    r = client.get("/ingest/run", params={"path": str(tmp_path / "missing.mp4")})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "ERROR: file not found" in r.text
    assert "data: [DONE]" in r.text
    assert "Running:" not in r.text  # no subprocess was attempted
