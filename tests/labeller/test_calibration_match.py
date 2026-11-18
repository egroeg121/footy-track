"""/calibration/match page + API: status strip, frame info with overlays, next-uncertain, sampled grid, frame JPEGs.

Synthetic match file (written by the real combine code from fake PTZ outputs) and a synthetic video frame."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from footy_track.calibration import match_calibration as mcal
from footy_track.labeller import calibration_match as cm

from ..calibration.test_match_calibration import (  # noqa: F401  (fixture)
    _combine,
    world,
)

# rows: seg0 = 0..19 (0-14 accepted, 15-18 calibrated, 19 uncalibrated), seg1 = 20..49 (recovered; 25-29 calibrated),
# seg2 = 50..64 (10-14 calibrated); seg3 has no result


@pytest.fixture
def env(world, monkeypatch, client):  # noqa: F811
    _combine(world)
    monkeypatch.setenv("FOOTY_CALIBRATION_ROOT", str(world["out"]))
    monkeypatch.setenv("FOOTY_DATA_DIR", str(world["data"]))
    cm._MC_CACHE.clear()
    cm._JPEG_CACHE.clear()
    vdir = world["data"] / "m" / "split_video_broadcast_frames"
    for i in range(3):
        (vdir / f"m_seg{i:03d}.mp4").touch()
    reads: list[tuple[str, int]] = []

    def fake_read(video, idx):
        reads.append((video.name, idx))
        return np.full((1080, 1920, 3), 90, np.uint8)

    monkeypatch.setattr(cm, "_read_frame", fake_read)
    monkeypatch.setattr(cm, "_editor_has_clip", lambda clip: clip == "m_seg000")
    client.reads = reads
    return client


def test_page_served_no_store(env):
    r = env.get("/calibration/match")
    assert r.status_code == 200 and "Match calibration" in r.text
    assert "no-store" in r.headers["cache-control"]


def test_linked_from_hub_and_editor_pages(env):
    for path in ("/main", "/calibration", "/calibration/review"):
        assert "/calibration/match" in env.get(path).text


def test_list_matches(env):
    j = env.get("/calibration/match/api/matches").json()
    assert [m["match"] for m in j["matches"]] == ["m"]
    m = j["matches"][0]
    assert m["frames"] == 65 and m["accepted_frac"] == pytest.approx(50 / 65, abs=1e-5)
    assert m["version_id"].startswith("ptz_v3+")


def test_summary_strip_rle_and_segments(env):
    j = env.get("/calibration/match/api/summary", params={"match": "m"}).json()
    assert j["n_rows"] == 65 and j["totals"]["segments_recovered"] == 1
    segs = {s["segment"]: s for s in j["segments"]}
    assert segs[0]["rle"] == [[2, 15], [1, 4], [0, 1]]
    assert segs[1]["rle"] == [[2, 25], [1, 5]] and segs[1]["source"] == "recovered"
    assert segs[1]["row0"] == 20 and segs[2]["row0"] == 50 and segs[1]["n"] == 30
    assert segs[3]["n"] == 0 and segs[3]["rle"] == []
    assert segs[0]["clip"] == "m_seg000" and segs[0]["editor_ok"] is True and segs[1]["editor_ok"] is False
    assert sum(n for s in j["segments"] for _, n in s["rle"]) == 65


def test_frame_info_overlay_status_and_editor_link(env):
    acc = env.get("/calibration/match/api/frame_info", params={"match": "m", "segment": 0, "local": 3}).json()
    assert acc["status"] == "accepted" and acc["frame"] == 103 and acc["source"] == "main"
    assert acc["editor_url"] == "/calibration?clip=m_seg000&frame=3" and acc["editor_ok"] is True
    assert acc["k1"] == pytest.approx(0.0147) and acc["evidence_f1"] is not None
    ov = acc["overlay"]
    assert "CENTRE_CIRCLE" in "".join(ov) or any("CIRCLE" in k for k in ov)
    pts = np.concatenate([np.array(r) for runs in ov.values() for r in runs])
    assert pts.shape[1] == 2 and np.isfinite(pts).all() and len(pts) > 20
    cal = env.get("/calibration/match/api/frame_info", params={"match": "m", "segment": 0, "local": 16}).json()
    assert cal["status"] == "calibrated" and cal["overlay"]
    unc = env.get("/calibration/match/api/frame_info", params={"match": "m", "segment": 0, "local": 19}).json()
    assert unc["status"] == "uncalibrated" and unc["overlay"] == {} and "pan_deg" not in unc
    rec = env.get("/calibration/match/api/frame_info", params={"match": "m", "segment": 1, "local": 3}).json()
    assert rec["source"] == "recovered" and rec["status"] == "accepted"


def test_overlay_uses_the_lens(env):
    mc = mcal.load_match_calibration("m")
    H = mc.H_for_frame(100)
    with_k1 = cm.overlay_runs(H, 0.0147)
    plain = cm.overlay_runs(H, 0.0)
    a = np.concatenate([np.array(r) for runs in with_k1.values() for r in runs])
    b = np.concatenate([np.array(r) for runs in plain.values() for r in runs])
    assert a.shape != b.shape or not np.allclose(a, b)


def test_frame_info_errors(env):
    assert env.get("/calibration/match/api/frame_info", params={"match": "m", "segment": 9, "local": 0}).status_code == 404
    assert env.get("/calibration/match/api/frame_info", params={"match": "m", "segment": 0, "local": 99}).status_code == 404
    assert env.get("/calibration/match/api/frame_info", params={"match": "nope", "segment": 0, "local": 0}).status_code == 404
    assert env.get("/calibration/match/api/summary", params={"match": "../x"}).status_code in (400, 404)
    assert env.get("/calibration/match/api/summary", params={"match": ".."}).status_code == 400
    assert env.get("/calibration/match/api/summary", params={"match": "m", "version": "a/b"}).status_code == 400
    assert env.get("/calibration/match/api/summary").status_code == 422


def test_next_uncertain_walks_runs_both_ways(env):
    def nxt(seg, local, d):
        return env.get("/calibration/match/api/next", params={"match": "m", "segment": seg, "local": local, "dir": d}).json()

    j = nxt(0, 0, 1)
    assert (j["segment"], j["local"], j["status"]) == (0, 15, "calibrated")
    j = nxt(0, 15, 1)  # inside a run: skip it
    assert (j["segment"], j["local"]) == (1, 25)
    j = nxt(1, 25, 1)
    assert (j["segment"], j["local"]) == (2, 10)
    assert nxt(2, 10, 1) == {"found": False}
    j = nxt(2, 12, -1)
    assert (j["segment"], j["local"]) == (1, 25)
    j = nxt(1, 25, -1)
    assert (j["segment"], j["local"]) == (0, 15)
    assert nxt(0, 15, -1) == {"found": False}
    assert env.get("/calibration/match/api/next", params={"match": "m", "segment": 5, "local": 0}).status_code == 404


def test_grid_modes_and_limits(env):
    g = env.get("/calibration/match/api/grid", params={"match": "m", "n": 10}).json()
    assert len(g["frames"]) == 10
    keys = [(f["segment"], f["local"]) for f in g["frames"]]
    assert keys == sorted(keys) and len(set(keys)) == 10  # even: ordered, distinct
    assert g["frames"][0]["img"].startswith("/calibration/match/api/frame/m/")
    assert all("overlay" in f for f in g["frames"])
    unc = env.get("/calibration/match/api/grid", params={"match": "m", "n": 24, "mode": "uncertain"}).json()
    assert {f["status"] for f in unc["frames"]} <= {"calibrated", "uncalibrated"} and len(unc["frames"]) == 15
    acc = env.get("/calibration/match/api/grid", params={"match": "m", "n": 5, "mode": "accepted", "seed": 3}).json()
    assert len(acc["frames"]) == 5 and {f["status"] for f in acc["frames"]} == {"accepted"}
    again = env.get("/calibration/match/api/grid", params={"match": "m", "n": 5, "mode": "accepted", "seed": 3}).json()
    assert again == acc  # seeded: reproducible
    other = env.get("/calibration/match/api/grid", params={"match": "m", "n": 5, "mode": "random", "seed": 4}).json()
    assert len(other["frames"]) == 5
    one = env.get("/calibration/match/api/grid", params={"match": "m", "n": 50, "segment": 2}).json()
    assert {f["segment"] for f in one["frames"]} == {2} and len(one["frames"]) == 15
    assert len(env.get("/calibration/match/api/grid", params={"match": "m", "n": 9999}).json()["frames"]) == cm.GRID_MAX  # capped
    assert env.get("/calibration/match/api/grid", params={"match": "m", "mode": "bogus"}).status_code == 400
    assert env.get("/calibration/match/api/grid", params={"match": "m", "mode": "uncertain", "segment": 3}).json()["frames"] == []


def test_frame_jpeg_resized_cached_and_errors(env):
    r = env.get("/calibration/match/api/frame/m/1/7.jpg", params={"w": 480})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    assert "max-age" in r.headers["cache-control"]
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (270, 480)  # 16:9 kept
    assert env.reads == [("m_seg001.mp4", 7)]
    env.get("/calibration/match/api/frame/m/1/7.jpg", params={"w": 480})
    assert len(env.reads) == 1  # LRU hit: no second decode
    assert env.get("/calibration/match/api/frame/m/1/99.jpg").status_code == 404
    assert env.get("/calibration/match/api/frame/m/3/0.jpg").status_code == 404  # segment without rows
    assert env.get("/calibration/match/api/frame/zzz/0/0.jpg").status_code == 404


def test_frame_jpeg_missing_video_and_unreadable_frame(env, world, monkeypatch):  # noqa: F811
    (world["data"] / "m" / "split_video_broadcast_frames" / "m_seg002.mp4").unlink()
    assert env.get("/calibration/match/api/frame/m/2/0.jpg").status_code == 404
    monkeypatch.setattr(cm, "_read_frame", lambda v, i: None)
    assert env.get("/calibration/match/api/frame/m/0/0.jpg").status_code == 404


def test_pure_helpers():
    assert cm.rle(np.array([2, 2, 1, 1, 1, 0, 2], np.int8)) == [[2, 2], [1, 3], [0, 1], [2, 1]]
    assert cm.rle(np.array([], np.int8)) == []
