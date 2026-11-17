"""Regression tests for the labelling architecture review's step-1 bug-fix pass.

README §19d (LAB-1427..1434): Ball Check never loses a card to a failed verdict save, slow
cv2/file work stays off the event loop, jobs read each store once, calibration GT
read-modify-write is locked, request-body clip names cannot escape the marks dir, review
writes are atomic, ``/ingest`` is not a 500 page, and the sidecar box scan is cached.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from footy_track.labeller import broadcast_review as br
from footy_track.labeller import calibration, datasets, jobs, review
from footy_track.labeller import identity_routes as ir
from footy_track.labeller import server as labeller_server

from .conftest import FAKE_JPEG  # noqa: F401  (fixtures import side effects)

PAGE = Path(__file__).resolve().parents[2] / "src/footy_track/labeller/web/ball_check.html"

# ---------------------------------------------------------------------------
# LAB-1427 (B3): a verdict that was not saved never silently drops the card
# ---------------------------------------------------------------------------

JUDGE_CHECK = r"""
const assert = require("assert");
const REFILL_AT = 12, RESAMPLE_EVERY = 15;
const srcOf = (i) => i.source || 'sidecar';
const keyOf = (i) => `${srcOf(i)}|${i.clip}|${i.frame_index}|${i.box_index}`;
let queue = [], history = [], remaining = 0, judged = 0, busy = false, sinceRefresh = 0;
const els = {};
const $ = (id) => (els[id] = els[id] || { textContent: '' });
let renders = 0, loads = 0;
const flash = () => {}, render = () => { renders++; }, refreshStats = () => {};
const loadQueue = () => { loads++; };
const card = (n) => ({ clip: 'c', frame_index: n, box_index: 0, source: 'cand', confidence: 0.5, bbox: {}, label: 'ball', provenance: 'x' });
const reset = () => { queue = [card(1), card(2), card(3)]; history = []; remaining = 3; judged = 0; busy = false; sinceRefresh = 0; loads = 0; };
const reply = (status, body) => async () => ({ ok: status >= 200 && status < 300, status, json: async () => { if (body === undefined) throw new Error('no json'); return body; } });

(async () => {
  // saved: optimistic advance, counters move once, no error
  reset(); global.fetch = reply(200, { ok: true });
  await judge('ball');
  assert.deepEqual(queue.map((q) => q.frame_index), [2, 3]);
  assert.equal(judged, 1); assert.equal(remaining, 2); assert.equal(history.length, 1);
  assert.equal(busy, false); assert.equal($('status').textContent, '');

  // the card is already gone while the POST is in flight (optimistic advance is kept)
  reset(); let release; const gate = new Promise((r) => { release = r; });
  global.fetch = async () => { await gate; return { ok: true, status: 200, json: async () => ({ ok: true }) }; };
  const p = judge('ball');
  assert.deepEqual(queue.map((q) => q.frame_index), [2, 3]); assert.equal(busy, true);
  release(); await p;

  // every failure mode puts the card back at the front and rolls the counters back
  const failures = {
    'http 500': reply(500, { detail: 'boom' }),
    'http 500 without json': reply(500, undefined),
    'ok:false in a 200': reply(200, { ok: false, error: 'clip, frame_index and box_index are required' }),
    '200 without ok': reply(200, {}),
    'network error': async () => { throw new TypeError('Failed to fetch'); },
  };
  for (const [name, f] of Object.entries(failures)) {
    reset(); global.fetch = f;
    await judge('not_ball');
    assert.deepEqual(queue.map((q) => q.frame_index), [1, 2, 3], name);
    assert.equal(judged, 0, name); assert.equal(remaining, 3, name); assert.equal(history.length, 0, name);
    assert.equal(busy, false, name); assert.equal(loads, 0, name);
    assert.match($('status').textContent, /NOT saved/, name);
  }

  // retry after a failure works and clears the message
  reset(); global.fetch = reply(500, undefined); await judge('ball');
  global.fetch = reply(200, { ok: true }); await judge('ball');
  assert.deepEqual(queue.map((q) => q.frame_index), [2, 3]); assert.equal(judged, 1);
  assert.equal($('status').textContent, '');
  console.log('ok');
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_ball_check_judge_keeps_the_card_when_the_save_fails(tmp_path):
    html = PAGE.read_text()
    a, b = html.index("// ---- verdict save helpers"), html.index("// ---- end verdict save helpers")
    f = tmp_path / "judge.js"
    # the helpers are declared after the stubs they use, as in the page
    stubs, check = JUDGE_CHECK.split("(async () => {", 1)
    f.write_text(stubs + html[a:b] + "(async () => {" + check)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 0, r.stderr + r.stdout


# ---------------------------------------------------------------------------
# LAB-1428 (B1/B8): blocking cv2/file work does not stall the event loop
# ---------------------------------------------------------------------------

BLOCK_S = 0.6


async def _probe_while(slow_path: str, probe_path: str = "/api/server_info") -> tuple[float, float]:
    """Longest event-loop stall (and a cheap request's latency) while ``slow_path`` is in flight."""
    transport = httpx.ASGITransport(app=labeller_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=30) as c:
        stalls: list[float] = []
        stop = asyncio.Event()

        async def ticker() -> None:
            while not stop.is_set():
                t = time.monotonic()
                await asyncio.sleep(0.02)
                stalls.append(time.monotonic() - t - 0.02)

        async def timed(path: str) -> float:
            r = await c.get(path)
            assert r.status_code == 200, (path, r.status_code, r.text[:200])
            return 0.0

        tick = asyncio.create_task(ticker())
        t0 = time.monotonic()
        slow = asyncio.create_task(timed(slow_path))
        await asyncio.sleep(0.15)  # a blocked loop only wakes this after the block is over
        probe_start = time.monotonic()
        await timed(probe_path)
        probe = time.monotonic() - min(probe_start, t0 + 0.15)
        await slow
        slow_s = time.monotonic() - t0
        stop.set()
        await tick
        return max(probe, max(stalls, default=0.0)), slow_s


def test_clips_status_does_not_block_the_event_loop(clips_dir, gt_marks_dir, monkeypatch):
    (clips_dir / "a.mp4").touch()

    def slow_completion(stem, path):
        time.sleep(BLOCK_S)  # stands in for cv2.VideoCapture + the sidecar parse
        return {"marked": False, "complete": False, "label_count": 0}

    monkeypatch.setattr(labeller_server, "_clip_completion", slow_completion)
    probe, slow = asyncio.run(_probe_while("/clips/status"))
    assert slow >= BLOCK_S
    assert probe < BLOCK_S / 2, f"a cheap request waited {probe:.2f}s behind /clips/status"


def test_identity_routes_do_not_block_the_event_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(ir, "_labels_dir", lambda: tmp_path)

    def slow_labels(_d):
        time.sleep(BLOCK_S)
        return []

    monkeypatch.setattr(ir, "load_pair_labels", slow_labels)
    probe, slow = asyncio.run(_probe_while("/identity/clusters"))
    assert slow >= BLOCK_S
    assert probe < BLOCK_S / 2, f"a cheap request waited {probe:.2f}s behind /identity/clusters"


def test_identity_data_routes_are_not_async_with_blocking_bodies():
    """The page routes stay ``async def``; every data route (cv2 / label scans) is a threadpool ``def``."""
    for name in ("list_tracklets", "risky_frames", "best_frame", "crop", "track_video",
                 "label_pair", "submit_review", "clusters"):
        assert not inspect.iscoroutinefunction(getattr(ir, name)), name


# ---------------------------------------------------------------------------
# LAB-1429 (B2): each match's store is read once per progress computation
# ---------------------------------------------------------------------------


def _br_job(two, n_frames=40):
    ds = datasets.get("calib")
    items = [{"clip": f"match_{m}", "frame": f} for m in ("a", "b") for f in range(n_frames)]
    doc = {
        "id": "job1", "title": "t", "kind": "broadcast_review", "dataset": "calib", "status": "open",
        "items": items, "created": "2026-11-17T00:00:00Z",
    }
    br.set_verdict(ds, "match_a", 3, "yes")
    br.set_verdict(ds, "match_b", 5, "no")
    return doc


def test_jobs_compute_reads_each_broadcast_review_store_once(two, monkeypatch):
    doc = _br_job(two)
    calls: list[str] = []
    real = br.read_store

    def spy(ds, match):
        calls.append(match)
        return real(ds, match)

    monkeypatch.setattr(br, "read_store", spy)
    out = jobs.compute(doc)
    assert out["done"] == 2 and out["total"] == 80
    assert sorted(calls) == ["match_a", "match_b"], calls  # was one read per item (80)


def test_broadcast_job_view_reads_each_store_once(two, monkeypatch):
    doc = _br_job(two)
    monkeypatch.setattr(jobs, "resolve", lambda ref: doc)
    calls: list[str] = []
    real = br.read_store

    def spy(ds, match):
        calls.append(match)
        return real(ds, match)

    monkeypatch.setattr(br, "read_store", spy)
    r = TestClient(labeller_server.app).get("/broadcast_review/api/job/job1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["done"] == 2 and body["total"] == 80
    assert sorted(calls) == ["match_a", "match_b"], calls


# ---------------------------------------------------------------------------
# LAB-1430 (B5): calibration GT read-modify-write is serialised
# ---------------------------------------------------------------------------


def test_concurrent_calibration_gt_updates_lose_nothing(gt_marks_dir):
    """40 threads each add one frame to the same clip's GT; a lost update would drop some."""
    n = 40
    start = threading.Barrier(n)

    def add(i: int) -> None:
        start.wait()
        calibration.update_gt("clipx", lambda doc: doc["frames"].update({str(i): {"n": i}}))

    ts = [threading.Thread(target=add, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(int(k) for k in calibration._read_gt("clipx")["frames"]) == list(range(n))


def test_unlocked_read_modify_write_would_lose_updates(gt_marks_dir, monkeypatch):
    """Guards the test above: with the lock removed the same race does lose frames."""
    monkeypatch.setattr(calibration, "_GT_LOCK", contextlib.nullcontext())
    real_read = calibration._read_gt

    def slow_read(clip):
        doc = real_read(clip)
        time.sleep(0.01)  # widen the read -> write window
        return doc

    monkeypatch.setattr(calibration, "_read_gt", slow_read)
    n = 8
    start = threading.Barrier(n)

    def add(i: int) -> None:
        start.wait()
        calibration.update_gt("clipy", lambda doc: doc["frames"].update({str(i): {"n": i}}))

    ts = [threading.Thread(target=add, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(real_read("clipy")["frames"]) < n


def test_update_gt_writes_nothing_when_the_document_is_unchanged(gt_marks_dir):
    assert calibration.update_gt("ghost", lambda doc: doc["frames"].pop("1", None) is not None) is False
    assert not calibration._gt_path("ghost").exists()


# ---------------------------------------------------------------------------
# LAB-1431 (B6): clip names from request bodies cannot escape the marks dir
# ---------------------------------------------------------------------------

BAD_NAMES = ["../evil", "a/b", "..", "a\\b", "/etc/passwd", "", ".hidden", "x" * 300, "a..b"]


@pytest.mark.parametrize("bad", BAD_NAMES)
def test_ball_check_verdict_rejects_unsafe_clip_names(client, gt_marks_dir, bad):
    body = {"clip": bad, "frame_index": 1, "box_index": 0, "verdict": "ball"}
    assert client.post("/ball_check/verdict", json=body).status_code == 422
    assert client.post("/ball_check/undo", json={k: v for k, v in body.items() if k != "verdict"}).status_code == 422
    assert not (gt_marks_dir / "ball_checks").exists()
    assert list(gt_marks_dir.parent.rglob("evil*")) == []


@pytest.mark.parametrize("bad", BAD_NAMES)
def test_review_and_save_box_reject_unsafe_clip_names(client, gt_marks_dir, bad):
    box = {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}
    base = {"clip": bad, "frame_index": 1, "box_index": 0}
    assert client.post("/review/correct", json={**base, "label": "ball", "bbox": box}).status_code == 422
    assert client.post("/review/delete", json=base).status_code == 422
    assert client.post("/ball_check/save_box", json={**base, "bbox": box, "source": "cand"}).status_code == 422
    assert list(gt_marks_dir.parent.rglob("evil*")) == []
    assert list(gt_marks_dir.iterdir()) == []


def test_real_clip_names_still_pass(client, gt_marks_dir):
    for good in ("arsenal_mancity_seg000", "clip.mp4", "arsenal_mancity_20250925_seg000", "a-b_c.d"):
        r = client.post("/ball_check/verdict", json={"clip": good, "frame_index": 1, "box_index": 0, "verdict": "ball"})
        assert r.status_code == 200 and r.json() == {"ok": True}, good
        assert (gt_marks_dir / "ball_checks" / f"{good}.jsonl").exists()


# ---------------------------------------------------------------------------
# LAB-1432: review edits are atomic
# ---------------------------------------------------------------------------


def _sidecar(gt_marks_dir, stem="clip", n=3):
    lines = [json.dumps({"frame_index": 4, "bbox": {"x": 0.1 * i, "y": 0.1, "w": 0.05, "h": 0.05},
                         "center": {"x": 0.1, "y": 0.1}, "tags": ["in_play_ball", "yolo"]}) for i in range(n)]
    p = gt_marks_dir / f"{stem}.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_review_correct_and_delete_replace_the_file_atomically(client, gt_marks_dir, monkeypatch):
    p = _sidecar(gt_marks_dir)
    inode = p.stat().st_ino
    mode = p.stat().st_mode & 0o777
    r = client.post("/review/delete", json={"clip": "clip", "frame_index": 4, "box_index": 0})
    assert r.json() == {"ok": True}
    assert len(p.read_text().splitlines()) == 2
    assert p.stat().st_ino != inode  # replaced via rename, not truncated in place
    assert p.stat().st_mode & 0o777 == mode
    assert not list(gt_marks_dir.glob("*.tmp"))
    r = client.post("/review/correct", json={"clip": "clip", "frame_index": 4, "box_index": 0,
                                             "label": "in_play_ball", "bbox": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}})
    assert r.json() == {"ok": True}
    assert json.loads(p.read_text().splitlines()[0])["bbox"]["x"] == 0.5


def test_a_crash_mid_write_leaves_the_sidecar_intact(client, gt_marks_dir, monkeypatch):
    p = _sidecar(gt_marks_dir)
    before = p.read_text()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(review.os, "replace", boom)
    with pytest.raises(OSError):
        review._write_text_atomic(p, "garbage\n")
    assert p.read_text() == before
    assert not list(gt_marks_dir.glob("*.tmp"))


# ---------------------------------------------------------------------------
# LAB-1433 (B7): /ingest is not a 500 page
# ---------------------------------------------------------------------------


def test_ingest_page_route_is_gone_and_the_hub_does_not_link_to_it(client):
    assert client.get("/ingest").status_code == 404  # was a 500 (missing web/ingest.html)
    hub = client.get("/").text
    assert 'href="/ingest"' not in hub
    # the working API endpoints (LAB-501/502) are untouched
    assert client.get("/ingest/run", params={"path": "/nonexistent.mp4"}).status_code == 200


# ---------------------------------------------------------------------------
# LAB-1434 (B13): the sidecar box scan only re-parses sidecars that changed
# ---------------------------------------------------------------------------


def test_sidecar_scan_reparses_only_changed_files(gt_marks_dir, clips_dir, monkeypatch):
    for stem in ("a", "b"):
        (clips_dir / f"{stem}.mp4").touch()
        _sidecar(gt_marks_dir, stem)
    review._BOX_CACHE.clear()
    parsed: list[str] = []
    real = review._parse_jsonl_box

    def spy(raw, stem, video):
        parsed.append(stem)
        return real(raw, stem, video)

    monkeypatch.setattr(review, "_parse_jsonl_box", spy)
    first = review._read_all_boxes()
    assert len(first) == 6 and sorted(set(parsed)) == ["a", "b"]
    parsed.clear()
    assert review._read_all_boxes() == first and parsed == []  # all cached
    # an atomic edit of one file invalidates just that file
    review._write_text_atomic(gt_marks_dir / "a.jsonl", (gt_marks_dir / "a.jsonl").read_text().split("\n", 1)[1])
    again = review._read_all_boxes()
    assert sorted(set(parsed)) == ["a"] and len(again) == 5
    # a clip that appears later re-keys the entry (video path changed)
    parsed.clear()
    (clips_dir / "c.mp4").touch()
    _sidecar(gt_marks_dir, "c", n=1)
    assert len(review._read_all_boxes()) == 6 and sorted(set(parsed)) == ["c"]
