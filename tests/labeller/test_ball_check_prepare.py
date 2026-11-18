"""Ball Check preparation: inline cards, the disk render cache, prepare job states, and the bounds
(card cap, queued-card cap, free-disk floor, idle/finished render expiry, LRU size cap).

Fixtures ``two`` / ``web`` (conftest) give two datasets (calib default, nottinghamforest) in tmp dirs;
``FOOTY_BALLCHECK_RENDERS_DIR`` points into tmp (tests/conftest.py).
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import tarfile
import threading
import time
from pathlib import Path

import pytest

from footy_track.labeller import ball_check, jobs, jobs_client, prefect_client
from footy_track.labeller import ball_check_cache as bcc
from footy_track.scripts import ballcheck_prepare as prep

DS = "nottinghamforest"
CLIP = "arsenal_nottinghamforest_seg003"
DAY = 86400


def _card(frame: int, clip: str = CLIP, **kw) -> dict:
    return {
        "clip": clip,
        "frame_index": frame,
        "bbox": {"x": 0.4, "y": 0.2, "w": 0.01, "h": 0.02},
        **kw,
    }


def _submit(web, cards, **over):
    body = {
        "title": "Blind batch",
        "kind": "ball_check",
        "dataset": DS,
        "items": cards,
        **over,
    }
    return web.post("/api/jobs", json=body)


@pytest.fixture
def dispatched(monkeypatch):
    """Replace the Prefect call; records ``(job id, where)`` and hands out run ids."""
    calls: list[tuple[str, str]] = []

    def fake(job, where):
        calls.append((job["id"], where))
        return f"run-{len(calls)}"

    monkeypatch.setattr(jobs, "DISPATCH", fake)
    return calls


@pytest.fixture
def clip_file(two):
    p = (two[DS] / "clips") / f"{CLIP}.mp4"
    p.touch()
    return p


def _verdict(web, card, verdict="ball"):
    return web.post(
        f"/ball_check/verdict?dataset={DS}",
        json={"clip": card["clip"], "frame_index": card["frame_index"], "box_index": card["box_index"],
              "source": "list", "verdict": verdict},
    ).json()  # fmt: skip


def _queue(web, job_id, **q):
    extra = "".join(f"&{k}={v}" for k, v in q.items())
    return web.get(
        f"/ball_check/queue?dataset={DS}&source=list&job={job_id}{extra}"
    ).json()


# ---------------------------------------------------------------------------
# Inline cards
# ---------------------------------------------------------------------------


def test_inline_cards_are_served_in_item_order_without_a_list_file_or_registry_edit(
    web, two, clip_file, tmp_path
):
    reg = Path(os.environ["FOOTY_LABELLER_DATASETS"])
    before = reg.read_bytes()
    r = _submit(web, [_card(30), _card(10), _card(20)])
    assert r.status_code == 201
    jid = r.json()["id"]
    assert f"job={jid}" in r.json()["url"] and "source=list" in r.json()["url"]
    q = _queue(web, jid)
    assert [i["frame_index"] for i in q["items"]] == [30, 10, 20]
    assert q["remaining"] == 3 and q["judged"] == 0
    boxes = [i["box_index"] for i in q["items"]]
    assert all(b >= jobs.INLINE_BOX_BASE for b in boxes) and len(set(boxes)) == 3
    assert q["items"][0]["image_url"].endswith(f"/{boxes[0]}.jpg?src=list")
    assert reg.read_bytes() == before  # no registry growth
    assert not list(
        (two[DS] / "clips").glob("*.json")
    )  # nor anything in the clips listing


def test_inline_verdict_lands_in_the_marks_dir_like_a_list_verdict_and_counts_as_progress(
    web, two, clip_file
):
    jid = _submit(web, [_card(10), _card(20)]).json()["id"]
    first = _queue(web, jid)["items"][0]
    assert _verdict(web, first, "not_ball")["ok"]
    log = (two[DS] / "marks") / "ball_checks" / f"{CLIP}.jsonl"
    rec = json.loads(log.read_text().splitlines()[-1])
    assert (rec["source"], rec["clip"], rec["frame_index"], rec["box_index"], rec["verdict"]) == (
        "list", CLIP, 10, first["box_index"], "not_ball",
    )  # fmt: skip
    assert _queue(web, jid)["items"][0]["frame_index"] == 20
    prog = web.get(f"/api/jobs/{jid}").json()["progress"]
    assert (prog["done"], prog["total"]) == (1, 2)
    # a list-source verdict is not a detector precision sample, same as list mode
    assert web.get(f"/ball_check/stats?dataset={DS}").json()["judged"] == 0
    res = web.get(f"/api/jobs/{jid}/results").json()
    assert [x["verdict"] for x in res["labels"]] == ["not_ball"]


def test_two_jobs_on_one_dataset_never_collide_even_on_the_same_frame(
    web, two, clip_file
):
    a = _submit(web, [_card(10)], title="A").json()["id"]
    b = _submit(web, [_card(10)], title="B").json()["id"]
    ca, cb = _queue(web, a)["items"][0], _queue(web, b)["items"][0]
    assert ca["box_index"] != cb["box_index"]
    assert _verdict(web, ca)["ok"]
    assert _queue(web, a)["remaining"] == 0
    assert _queue(web, b)["remaining"] == 1  # B's card is not judged by A's verdict
    assert web.get(f"/api/jobs/{b}").json()["progress"]["done"] == 0


def test_same_frame_twice_in_one_job_gets_distinct_box_indexes(web, two, clip_file):
    jid = _submit(
        web, [_card(10), _card(10, bbox={"x": 0.1, "y": 0.1, "w": 0.01, "h": 0.01})]
    ).json()["id"]
    boxes = [i["box_index"] for i in _queue(web, jid)["items"]]
    assert len(set(boxes)) == 2


def test_hidden_keys_are_never_sent_to_the_page_but_pass_through_to_results(
    web, two, clip_file
):
    secret = {
        "class": "all3_agree",
        "positions": {"t1": [1, 2]},
        "v2_id": "abc",
        "stratum": "s",
        "t1_conf": 0.99,
    }
    jid = _submit(web, [_card(10, **secret)]).json()["id"]
    item = _queue(web, jid)["items"][0]
    assert not set(secret) & set(item)  # blind: nothing but geometry reaches the phone
    assert (
        item["provenance"] == "list"
        and item["confidence"] is None
        and "note" not in item
    )
    assert _verdict(web, item)["ok"]
    res = web.get(f"/api/jobs/{jid}/results").json()
    assert res["labels"][0]["meta"] == secret
    assert res["items"][0]["item"]["class"] == "all3_agree"


def test_only_label_provenance_confidence_note_and_other_bbox_are_shown(
    web, two, clip_file
):
    other = {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.01}
    jid = _submit(
        web,
        [
            _card(
                10,
                note="why",
                role="gt",
                confidence=0.5,
                label="ball",
                other_bbox=other,
                junk=1,
            )
        ],
    ).json()["id"]
    item = _queue(web, jid)["items"][0]
    assert (item["note"], item["role"], item["confidence"], item["label"]) == (
        "why",
        "gt",
        0.5,
        "ball",
    )
    assert item["other_bbox"] == other and "junk" not in item


def test_inline_item_validation(web, two):
    assert (
        _submit(web, [_card(10, bbox={"x": 2, "y": 0, "w": 0.1, "h": 0.1})]).status_code
        == 422
    )
    assert (
        _submit(
            web, [_card(10, bbox={"x": 0.1, "y": 0.1, "w": 0, "h": 0.1})]
        ).status_code
        == 422
    )
    assert _submit(web, [_card(10, source="sidecar")]).status_code == 422
    bad = _card(10)
    del bad["frame_index"]
    assert _submit(web, [bad]).status_code == 422


def test_pointer_cards_still_work_unchanged(web, two):
    r = _submit(
        web, [{"clip": CLIP, "frame_index": 5, "box_index": 3, "source": "list"}]
    )
    assert r.status_code == 201
    j = web.get(f"/api/jobs/{r.json()['id']}").json()
    assert j["items"][0]["box_index"] == 3 and jobs.is_inline(j["items"][0]) is False


def test_inline_crop_clip_and_meta_use_the_cards_own_geometry(
    web, two, clip_file, fake_cv2, monkeypatch
):
    jid = _submit(
        web, [_card(5, other_bbox={"x": 0.1, "y": 0.1, "w": 0.01, "h": 0.01})]
    ).json()["id"]
    item = _queue(web, jid)["items"][0]
    b = item["box_index"]
    meta = web.get(f"/ball_check/crop_meta/{CLIP}/5/{b}?dataset={DS}&src=list").json()
    assert meta["ok"] and meta["bbox"]["x"] == pytest.approx(0.4)
    r = web.get(
        f"/ball_check/crop/{CLIP}/5/{b}.jpg?dataset={DS}&src=list&pad=6&reticle=false"
    )
    assert r.status_code == 200 and r.content
    assert (
        web.get(f"/ball_check/crop/{CLIP}/6/{b}.jpg?dataset={DS}&src=list").status_code
        == 404
    )  # wrong frame


# ---------------------------------------------------------------------------
# Disk render cache
# ---------------------------------------------------------------------------


@pytest.fixture
def rc(tmp_path):
    return bcc.RenderCache(tmp_path / "rc")


def test_disk_cache_miss_then_hit_and_lru_touch(rc):
    calls = []

    def render():
        calls.append(1)
        return b"jpeg", {}

    (data, _), how = rc.get_or_render("k" * 40, "jpg", render)
    assert (data, how, len(calls)) == (b"jpeg", "rendered", 1)
    (data, _), how = rc.get_or_render("k" * 40, "jpg", render)
    assert (data, how, len(calls)) == (b"jpeg", "disk", 1)
    p = rc.blob_path("k" * 40, "jpg")
    os.utime(p, (1, 1))
    rc.get("k" * 40, "jpg")
    assert p.stat().st_mtime > 1000  # a hit refreshes the LRU time


def test_disk_write_is_atomic_and_stages_outside_the_served_folder(rc, monkeypatch):
    key = "a" * 40
    real = bcc.os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(bcc.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        rc.put(key, "jpg", b"x")
    monkeypatch.setattr(bcc.os, "replace", real)
    assert not rc.has(key, "jpg")  # no half-written blob is ever visible
    assert not list(rc.tmp.glob("*"))  # and the temp file was cleaned up
    rc.put(key, "mp4", b"y", {"n": 3})
    assert rc.blob_path(key, "mp4").read_bytes() == b"y"
    assert (
        rc.tmp not in rc.blob_path(key, "mp4").parents
        and rc.blobs not in rc.tmp.parents
    )
    assert rc.get(key, "mp4")[1] == {"n": 3}


def test_concurrent_renders_of_one_key_are_a_single_render(rc):
    n = {"calls": 0}
    gate = threading.Event()

    def render():
        n["calls"] += 1
        gate.wait(2)
        return b"v", {}

    out = []
    threads = [
        threading.Thread(
            target=lambda: out.append(rc.get_or_render("c" * 40, "jpg", render))
        )
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    time.sleep(0.2)
    gate.set()
    for t in threads:
        t.join(5)
    assert n["calls"] == 1 and len(out) == 6 and all(o[0][0] == b"v" for o in out)


def test_key_covers_everything_that_changes_the_bytes(tmp_path):
    v = tmp_path / "c.mp4"
    v.write_bytes(b"1234")
    base = {
        "video_path": v,
        "frame_idx": 5,
        "at_frame": 5,
        "bbox": (0.1, 0.2, 0.01, 0.02),
        "ob": None,
        "pad": 6.0,
        "reticle": False,
    }
    k0 = ball_check.crop_disk_key(**base)
    assert k0 == ball_check.crop_disk_key(**base)
    for change in ({"pad": 14.0}, {"reticle": True}, {"at_frame": 6}, {"frame_idx": 6}, {"bbox": (0.1, 0.2, 0.011, 0.02)},
                   {"ob": (0.5, 0.5, 0.01, 0.01)}):  # fmt: skip
        assert ball_check.crop_disk_key(**{**base, **change}) != k0, change
    v.write_bytes(b"12345")  # the clip file changed
    assert ball_check.crop_disk_key(**base) != k0
    c0 = ball_check.clip_disk_key(v, 5, base["bbox"], None, 6.0, "mp4")
    assert c0 != ball_check.clip_disk_key(v, 5, base["bbox"], None, 6.0, "gif")


def test_render_version_is_part_of_the_key(tmp_path, monkeypatch):
    v = tmp_path / "c.mp4"
    v.write_bytes(b"1")
    args = (v, 1, 1, (0.1, 0.1, 0.1, 0.1), None, 6.0, False)
    k = ball_check.crop_disk_key(*args)
    monkeypatch.setattr(ball_check, "RENDER_VERSION", "2")
    assert ball_check.crop_disk_key(*args) != k
    monkeypatch.setattr(ball_check, "RENDER_VERSION", "1")
    c0 = ball_check.clip_disk_key(v, 1, args[3], None, 6.0, "mp4")
    monkeypatch.setattr(
        ball_check, "_CLIP_W", 999
    )  # a render constant alone changes a clip's key...
    assert ball_check.clip_disk_key(v, 1, args[3], None, 6.0, "mp4") != c0


def test_server_serves_a_prepared_card_from_disk_and_survives_a_restart(
    web, two, clip_file, fake_cv2, monkeypatch
):
    jid = _submit(web, [_card(5)]).json()["id"]
    plan = web.get(f"/api/jobs/{jid}/prepare_plan?dataset={DS}").json()
    card = plan["cards"][0]
    assert (
        plan["clips_dir"] == str(two[DS] / "clips")
        and card["video"]["name"] == clip_file.name
    )
    kinds = {v["kind"]: v for v in card["variants"]}
    assert kinds["crop"]["reticle"] is False and not kinds["crop"]["cached"]
    renders = []
    monkeypatch.setattr(
        ball_check, "render_crop_bytes", lambda *a, **k: renders.append(a) or b"JPEG"
    )
    monkeypatch.setattr(
        ball_check, "_render_clip", lambda *a, **k: (b"MP4", "video/mp4", 7, 300)
    )
    monkeypatch.setattr(ball_check, "_have_ffmpeg", lambda: True)
    plan = web.get(f"/api/jobs/{jid}/prepare_plan?dataset={DS}").json()
    out = prep.render_card(plan["cards"][0], str(clip_file), plan["renders_dir"])
    assert out["ok"] and out["made"] == 2 and len(renders) == 1
    # the server's key is the plan's key: a "restart" (empty memory caches) is served from disk, no render
    monkeypatch.setattr(ball_check, "_BC_CROP_CACHE", type(ball_check._BC_CROP_CACHE)())
    monkeypatch.setattr(ball_check, "_BC_CLIP_CACHE", type(ball_check._BC_CLIP_CACHE)())
    b = card["box_index"]
    r = web.get(
        f"/ball_check/crop/{CLIP}/5/{b}.jpg?dataset={DS}&src=list&pad=6&reticle=false"
    )
    assert (r.status_code, r.content, len(renders)) == (200, b"JPEG", 1)
    r = web.get(f"/ball_check/clip/{CLIP}/5/{b}.mp4?dataset={DS}&src=list&pad=6")
    assert (r.status_code, r.content, r.headers["x-clip-frames"]) == (200, b"MP4", "7")
    assert web.get(f"/api/jobs/{jid}/prepare_plan?dataset={DS}").json()["n_cached"] == 2


def test_prepare_plan_skips_judged_cards(web, two, clip_file):
    jid = _submit(web, [_card(5), _card(6)]).json()["id"]
    first = _queue(web, jid)["items"][0]
    assert _verdict(web, first)["ok"]
    plan = web.get(f"/api/jobs/{jid}/prepare_plan?dataset={DS}").json()
    assert [c["frame_index"] for c in plan["cards"]] == [6]
    assert (
        len(
            web.get(
                f"/api/jobs/{jid}/prepare_plan?dataset={DS}&skip_judged=false"
            ).json()["cards"]
        )
        == 2
    )


def test_prepare_plan_lists_unresolved_cards(web, two):
    jid = _submit(web, [_card(5)]).json()["id"]  # no clip file on disk
    plan = web.get(f"/api/jobs/{jid}/prepare_plan?dataset={DS}").json()
    assert plan["cards"] == [] and plan["unresolved"][0]["reason"] == "video not found"


def test_clip_with_a_different_size_on_the_worker_is_refused(two, tmp_path):
    clips = tmp_path / "wclips"
    clips.mkdir()
    (clips / "a.mp4").write_bytes(b"12")
    card = {"position": 0, "clip": "a", "frame_index": 1, "box_index": 1, "source": "list", "bbox": [0.1, 0.1, 0.1, 0.1],
            "other_bbox": None, "video": {"name": "a.mp4", "size": 99},
            "variants": [{"kind": "crop", "pad": 6.0, "reticle": False, "ext": "jpg", "key": "d" * 40, "cached": False}]}  # fmt: skip
    rc_dir = tmp_path / "out"
    code = prep.run_render(
        {"cards": [card]}, clips, rc_dir, job=None, workers=1, nice=0, floor=0
    )
    assert code == prep.EXIT_FAILED and not bcc.RenderCache(rc_dir).has("d" * 40, "jpg")


def test_ingest_merges_a_staged_tree_atomically(tmp_path):
    stage = bcc.RenderCache(tmp_path / "stage")
    stage.put("e" * 40, "jpg", b"img")
    stage.register("j-1", [("e" * 40, "jpg")])
    tar = tmp_path / "s.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(stage.blobs, arcname="blobs")
        tf.add(stage.refs, arcname="refs")
    live = tmp_path / "live"
    assert prep.run_ingest(tar, live) == 0
    lrc = bcc.RenderCache(live)
    assert lrc.get("e" * 40, "jpg")[0] == b"img" and lrc.read_ref("j-1")["keys"] == [
        ["e" * 40, "jpg"]
    ]


# ---------------------------------------------------------------------------
# Bounds: card cap, queued-card cap, disk floor
# ---------------------------------------------------------------------------


def test_card_cap_is_rejected_then_allowed_up_to_the_hard_ceiling(web, two):
    cards = [_card(i) for i in range(jobs.MAX_CARDS_DEFAULT + 1)]
    r = _submit(web, cards)
    assert (
        r.status_code == 422
        and "cap of 500" in r.json()["detail"]
        and "max_cards" in r.json()["detail"]
    )
    assert (
        not list(jobs.jobs_dir().glob("*.json")) if jobs.jobs_dir().exists() else True
    )
    assert _submit(web, cards, max_cards=jobs.MAX_CARDS_HARD).status_code == 201
    assert (
        _submit(
            web,
            [_card(i) for i in range(jobs.MAX_CARDS_HARD + 1)],
            max_cards=jobs.MAX_CARDS_HARD,
        ).status_code
        == 422
    )
    r = _submit(web, [_card(1)], max_cards=jobs.MAX_CARDS_HARD + 1)
    assert r.status_code == 422 and "hard ceiling" in r.json()["detail"]


def test_frame_jobs_have_their_own_larger_cap(web, two, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_FRAMES_DEFAULT", 100)
    ok = web.post("/api/jobs", json={"title": "f", "kind": "ball_place", "dataset": DS,
                                     "items": [{"clip": CLIP, "frame_start": 0, "frame_end": 99}]})  # fmt: skip
    big = web.post("/api/jobs", json={"title": "g", "kind": "ball_place", "dataset": DS,
                                      "items": [{"clip": CLIP, "frame_start": 0, "frame_end": 100}]})  # fmt: skip
    assert (
        ok.status_code == 201
        and big.status_code == 422
        and "cap of 100" in big.json()["detail"]
    )


def test_cli_refuses_over_the_cap_before_posting():
    cards = [_card(i) for i in range(501)]
    with pytest.raises(jobs_client.JobsError, match="--allow-large"):
        jobs_client.submit(
            "t", "ball_check", cards, url="http://127.0.0.1:9"
        )  # never reaches the network
    with pytest.raises(jobs_client.JobsError, match="hard ceiling"):
        jobs_client.submit(
            "t",
            "ball_check",
            [_card(i) for i in range(2001)],
            max_cards=2000,
            url="http://127.0.0.1:9",
        )


def test_queued_card_cap_refuses_a_prepare_that_would_pass_it(
    web, two, dispatched, monkeypatch
):
    monkeypatch.setattr(jobs, "MAX_QUEUED_CARDS", 5)
    a = _submit(web, [_card(i) for i in range(4)], prepare=True, title="A")
    assert a.status_code == 201 and a.json()["prepare"]["state"] == "queued"
    b = _submit(web, [_card(i) for i in range(3)], prepare=True, title="B")
    assert b.status_code == 429 and "cap of 5" in b.json()["detail"]
    assert [j["title"] for j in web.get("/api/jobs").json()["jobs"]] == [
        "A"
    ]  # B was never created
    # an unprepared job costs nothing and is allowed; preparing it later is refused the same way
    c = _submit(web, [_card(i) for i in range(3)], title="C")
    assert c.status_code == 201
    assert web.post(f"/api/jobs/{c.json()['id']}/prepare", json={}).status_code == 429
    # labelling A down frees the queue
    jid = a.json()["id"]
    jobs.prepare_event(jid, "finish", ok=True)
    for it in _queue(web, jid)["items"][:3]:
        assert _verdict(web, it)["ok"]
    assert web.post(f"/api/jobs/{c.json()['id']}/prepare", json={}).status_code == 200


def test_disk_floor_refuses_to_prepare(web, two, dispatched, monkeypatch):
    monkeypatch.setenv("FOOTY_BALLCHECK_MIN_FREE_GB", "20")
    monkeypatch.setattr(bcc, "free_bytes", lambda p: 5 * bcc.GB)
    r = _submit(web, [_card(1)], prepare=True)
    assert r.status_code == 507 and "floor" in r.json()["detail"]
    assert web.get("/api/jobs").json()["jobs"] == []
    plain = _submit(web, [_card(1)]).json()["id"]
    assert web.post(f"/api/jobs/{plain}/prepare", json={}).status_code == 507
    assert not dispatched
    monkeypatch.setattr(bcc, "free_bytes", lambda p: 500 * bcc.GB)
    assert web.post(f"/api/jobs/{plain}/prepare", json={}).status_code == 200


def test_render_run_stops_at_the_disk_floor(tmp_path, monkeypatch):
    card = {"position": 0, "clip": "a", "frame_index": 1, "box_index": 1, "source": "list", "bbox": [0.1, 0.1, 0.1, 0.1],
            "other_bbox": None, "video": {"name": "a.mp4", "size": 1},
            "variants": [{"kind": "crop", "pad": 6.0, "reticle": False, "ext": "jpg", "key": "f" * 40, "cached": False}]}  # fmt: skip
    clips = tmp_path / "c"
    clips.mkdir()
    (clips / "a.mp4").write_bytes(b"1")
    code = prep.run_render(
        {"cards": [card]},
        clips,
        tmp_path / "o",
        job=None,
        workers=1,
        nice=0,
        floor=10**15,
    )
    assert code == prep.EXIT_FLOOR
    assert not bcc.RenderCache(tmp_path / "o").has("f" * 40, "jpg")


# ---------------------------------------------------------------------------
# Job states and duplicate prepares
# ---------------------------------------------------------------------------


def test_submit_with_prepare_is_preparing_until_the_flow_finishes(
    web, two, clip_file, dispatched
):
    r = _submit(web, [_card(5), _card(6)], prepare=True)
    jid = r.json()["id"]
    assert r.json()["status"] == "preparing" and dispatched == [(jid, "mini")]
    assert web.get(f"/api/jobs/{jid}").json()["prepare"]["run_id"] == "run-1"
    # not offered: absent-as-a-link on the board (shown with its prep progress) and no cards on the phone
    (task,) = [t for t in web.get("/api/tasks").json()["tasks"] if t["id"] == jid]
    assert (
        task["url"] == ""
        and task["preparing"] == {"prepared": 0, "total": 2}
        and "count_url" not in task
    )
    q = _queue(web, jid)
    assert q["items"] == [] and q["preparing"] == {"prepared": 0, "total": 2}
    jobs.prepare_event(jid, "run", run_id="run-1")
    jobs.prepare_event(jid, "progress", prepared=1, total=2)
    assert _queue(web, jid)["preparing"]["prepared"] == 1
    jobs.prepare_event(jid, "finish", ok=True, prepared=2, total=2)
    j = web.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "open" and j["prepare"]["state"] == "ready"
    (task,) = [t for t in web.get("/api/tasks").json()["tasks"] if t["id"] == jid]
    assert task["url"] and "preparing" not in task
    assert len(_queue(web, jid)["items"]) == 2


def test_failed_prepare_releases_the_job_so_it_still_works_on_demand(
    web, two, clip_file, dispatched
):
    jid = _submit(web, [_card(5)], prepare=True).json()["id"]
    jobs.prepare_event(jid, "abort", error="boom")
    j = web.get(f"/api/jobs/{jid}").json()
    assert (
        j["status"] == "open"
        and j["prepare"]["state"] == "failed"
        and j["prepare"]["error"] == "boom"
    )
    assert len(_queue(web, jid)["items"]) == 1


def test_dispatch_failure_leaves_a_usable_job_and_says_why(web, two, monkeypatch):
    def broken(job, where):
        raise RuntimeError("prefect is down")

    monkeypatch.setattr(jobs, "DISPATCH", broken)
    r = _submit(web, [_card(5)], prepare=True)
    assert r.status_code == 201 and r.json()["status"] == "open"
    assert (
        r.json()["prepare"]["state"] == "failed"
        and "prefect is down" in r.json()["prepare"]["error"]
    )


def test_only_one_prepare_per_job_and_a_stale_one_can_be_redone(
    web, two, clip_file, dispatched
):
    jid = _submit(web, [_card(5)]).json()["id"]
    assert web.post(f"/api/jobs/{jid}/prepare", json={}).status_code == 200
    dup = web.post(f"/api/jobs/{jid}/prepare", json={})
    assert dup.status_code == 409 and "already" in dup.json()["detail"]
    assert len(dispatched) == 1
    # the run never reported: gc declares it failed, then a new prepare is accepted
    jobs.gc(now=time.time() + jobs.PREPARE_STALE_QUEUED_S + 10)
    assert web.get(f"/api/jobs/{jid}").json()["prepare"]["state"] == "failed"
    assert (
        web.post(f"/api/jobs/{jid}/prepare", json={}).status_code == 200
        and len(dispatched) == 2
    )


def test_prepare_does_not_hold_an_in_progress_job_and_rejects_non_ball_check(
    web, two, clip_file, dispatched
):
    jid = _submit(web, [_card(5), _card(6)]).json()["id"]
    assert _verdict(web, _queue(web, jid)["items"][0])["ok"]
    web.get(f"/api/jobs/{jid}")  # triggers open -> in_progress
    r = web.post(f"/api/jobs/{jid}/prepare", json={"hold": True})
    assert r.status_code == 200 and r.json()["status"] == "in_progress"
    other = web.post(
        "/api/jobs", json={"title": "x", "kind": "other", "dataset": DS}
    ).json()["id"]
    assert web.post(f"/api/jobs/{other}/prepare", json={}).status_code == 422


# ---------------------------------------------------------------------------
# Housekeeping: finished / idle renders, LRU cap, verdicts kept, auto re-prepare
# ---------------------------------------------------------------------------


def _prepared_job(web, two, rc_root, n=2, **over):
    """A job with ``n`` cards whose renders exist on disk and are registered to it."""
    jid = _submit(web, [_card(5 + i) for i in range(n)], **over).json()["id"]
    rc = bcc.RenderCache(rc_root)
    keys = []
    for i in range(n):
        k = f"{jid[-6:]}{i}".ljust(40, "0")
        rc.put(k, "jpg", b"x" * 100)
        keys.append((k, "jpg"))
    rc.register(jid, keys)
    return jid, rc, keys


def _arm(jid):
    """Give a job the prepare bookkeeping a finished run leaves."""
    job = jobs.read_job(jid)
    job["prepare"] = {
        **jobs._new_prepare("mini", hold=False, total=1),
        "state": "ready",
        "finished": time.time(),
    }
    jobs._write_atomic(jobs._job_path(jid), job)


def test_gc_deletes_renders_of_a_done_job_after_the_grace_but_keeps_its_verdicts(
    web, two, clip_file
):
    jid, rc, keys = _prepared_job(web, two, bcc.root())
    first = _queue(web, jid)["items"][0]
    assert _verdict(web, first)["ok"]
    log = (two[DS] / "marks") / "ball_checks" / f"{CLIP}.jsonl"
    before = log.read_text()
    web.post(f"/api/jobs/{jid}/done")
    now = time.time()
    r = jobs.gc(now=now + 12 * 3600)
    assert r["cache"]["dropped_refs"] == {} and rc.has(
        *keys[0]
    )  # inside the 24 h grace
    dry = jobs.gc(now=now + 25 * 3600, dry_run=True)
    assert dry["cache"]["dropped_refs"] == {jid: "finished"} and rc.has(
        *keys[0]
    )  # dry run deletes nothing
    jobs.gc(now=now + 25 * 3600)
    assert not rc.has(*keys[0]) and not rc.has(*keys[1]) and rc.read_ref(jid) == {}
    assert log.read_text() == before  # verdicts are never deleted
    assert web.get(f"/api/jobs/{jid}/results").json()["labels"][0]["verdict"] == "ball"
    assert jobs.read_job(jid) is not None  # and neither is the job


def test_gc_deletes_renders_of_a_cancelled_job(web, two, clip_file):
    jid, rc, keys = _prepared_job(web, two, bcc.root())
    web.post(f"/api/jobs/{jid}/cancel")
    jobs.gc(now=time.time() + 25 * 3600)
    assert not rc.has(*keys[0])


def test_idle_job_loses_renders_after_3_days_but_stays_open(web, two, clip_file):
    jid, rc, keys = _prepared_job(web, two, bcc.root())
    _arm(jid)
    assert jobs.gc(now=time.time() + 2 * DAY)["evicted_jobs"] == []
    assert rc.has(*keys[0])  # labelling-idle for 2 days: still there
    r = jobs.gc(now=time.time() + 3 * DAY + 60)
    assert r["evicted_jobs"] == [jid] and not rc.has(*keys[0])
    j = web.get(f"/api/jobs/{jid}").json()
    assert (
        j["status"] == "open" and j["prepare"]["state"] == "evicted"
    )  # the job never expires
    assert [
        t["id"] for t in web.get("/api/tasks").json()["tasks"] if t["id"] == jid
    ] == [jid]


def test_labelling_activity_keeps_the_renders(web, two, clip_file):
    jid, rc, keys = _prepared_job(web, two, bcc.root(), n=3)
    _arm(jid)
    assert _verdict(web, _queue(web, jid)["items"][0])["ok"]  # a verdict now
    jobs.gc(now=time.time() + 2 * DAY + 3600)
    assert rc.has(*keys[0])
    assert (
        jobs.gc(now=time.time() + 4 * DAY)["evicted_jobs"] != []
    )  # idle since that verdict


def test_labelling_resuming_on_an_evicted_job_reprepares_only_the_remaining_cards(
    web, two, clip_file, dispatched
):
    jid, rc, keys = _prepared_job(web, two, bcc.root(), n=3)
    _arm(jid)
    first = _queue(web, jid)["items"][0]
    assert _verdict(web, first)["ok"]
    jobs.gc(now=time.time() + 4 * DAY)
    assert (
        web.get(f"/api/jobs/{jid}").json()["prepare"]["state"] == "evicted"
        and not dispatched
    )
    q = _queue(
        web, jid
    )  # the phone opens the job: cards come at once (on-demand), a run is queued
    assert len(q["items"]) == 2 and dispatched == [(jid, "mini")]
    assert web.get(f"/api/jobs/{jid}").json()["prepare"]["state"] == "queued"
    _queue(web, jid)
    assert len(dispatched) == 1  # no duplicate run
    plan = web.get(f"/api/jobs/{jid}/prepare_plan?dataset={DS}").json()
    assert first["frame_index"] not in [
        c["frame_index"] for c in plan["cards"]
    ]  # judged: not re-rendered
    assert len(plan["cards"]) == 2


def test_first_card_request_of_an_evicted_job_also_reprepares(
    web, two, clip_file, dispatched, fake_cv2, monkeypatch
):
    jid, rc, keys = _prepared_job(web, two, bcc.root(), n=1)
    _arm(jid)
    item = _queue(web, jid)["items"][0]
    jobs.gc(now=time.time() + 4 * DAY)
    r = web.get(
        f"/ball_check/crop/{CLIP}/{item['frame_index']}/{item['box_index']}.jpg?dataset={DS}&src=list&reticle=false"
    )
    assert r.status_code == 200 and dispatched == [(jid, "mini")]


def test_failed_auto_reprepare_backs_off(web, two, clip_file, monkeypatch):
    jid, rc, keys = _prepared_job(web, two, bcc.root(), n=1)
    _arm(jid)
    jobs.gc(now=time.time() + 4 * DAY)
    n = {"calls": 0}

    def broken(job, where):
        n["calls"] += 1
        raise RuntimeError("down")

    monkeypatch.setattr(jobs, "DISPATCH", broken)
    assert jobs.maybe_reprepare(jid) is False and jobs.maybe_reprepare(jid) is False
    assert n["calls"] == 1  # paused for 10 minutes, the phone is not slowed by retries
    assert jobs.read_job(jid)["prepare"]["state"] == "evicted"


def _seed(rc, name, size, mtime):
    key = name.ljust(40, "0")
    rc.put(key, "jpg", b"x" * size)
    os.utime(rc.blob_path(key, "jpg"), (mtime, mtime))
    return (key, "jpg")


def test_lru_eviction_order_orphans_then_finished_jobs_then_open_jobs(rc):
    now = 1_000_000_000.0
    o_old, o_new = (
        _seed(rc, "orphan_old", 100, now - 500),
        _seed(rc, "orphan_new", 100, now - 100),
    )
    f_old, f_new = (
        _seed(rc, "fin_old", 100, now - 400),
        _seed(rc, "fin_new", 100, now - 50),
    )
    a_old, a_new = (
        _seed(rc, "open_old", 100, now - 600),
        _seed(rc, "open_new", 100, now - 10),
    )
    rc.register("fin", [f_old, f_new], now=now)
    rc.register("act", [a_old, a_new], now=now)
    jobs_map = {"fin": {"status": "done", "closed_ts": now - 60, "last_activity": now - 60},
                "act": {"status": "open", "closed_ts": None, "last_activity": now - 5}}  # fmt: skip
    kw = {
        "now": now,
        "grace_s": 86400,
        "idle_s": 3 * 86400,
        "orphan_ttl_s": 10**9,
        "dry_run": True,
    }
    # under the cap nothing goes, including renders of open jobs
    assert bcc.gc(rc, jobs_map, cap=10**6, **kw)["evicted_order"] == []
    # over it: orphans (oldest first), then finished jobs' renders (oldest first), open jobs last
    order = bcc.gc(rc, jobs_map, cap=150, **kw)["evicted_order"]
    assert order == [o_old, o_new, f_old, f_new, a_old]
    # the cap is met as soon as enough has gone: open-job renders untouched while finished ones remain
    assert bcc.gc(rc, jobs_map, cap=350, **kw)["evicted_order"] == [o_old, o_new, f_old]


def test_gc_enforces_the_size_cap_for_real(rc):
    now = time.time()
    keys = [
        _seed(rc, f"b{i}", 1000, now - 100 + i) for i in range(5)
    ]  # orphans, oldest first
    rep = bcc.gc(rc, {}, now, cap=2500, orphan_ttl_s=10**9)
    assert rep["evicted_for_cap"] == 3 and rep["bytes_after"] <= 2500
    assert [rc.has(*k) for k in keys] == [False, False, False, True, True]


def test_old_orphans_expire_and_blobs_of_a_dropped_ref_go_but_shared_ones_stay(rc):
    now = 1_000_000_000.0
    shared, only_a = _seed(rc, "shared", 10, now), _seed(rc, "only_a", 10, now)
    old_orphan = _seed(rc, "orph", 10, now - 8 * 86400)
    rc.register("A", [shared, only_a], now=now)
    rc.register("B", [shared], now=now)
    infos = {"A": {"status": "done", "closed_ts": now - 2 * 86400, "last_activity": now - 2 * 86400},
             "B": {"status": "open", "closed_ts": None, "last_activity": now}}  # fmt: skip
    rep = bcc.gc(rc, infos, now, cap=10**6)
    assert rep["dropped_refs"] == {"A": "finished"}
    assert not rc.has(*only_a) and not rc.has(*old_orphan) and rc.has(*shared)


def test_housekeeping_does_not_grow_the_clip_or_registry_listings(
    web, two, clip_file, dispatched
):
    reg = Path(os.environ["FOOTY_LABELLER_DATASETS"])
    before_reg = reg.read_bytes()
    clips_before = sorted(p.name for p in (two[DS] / "clips").iterdir())
    jid, rc, keys = _prepared_job(web, two, bcc.root())
    assert _verdict(web, _queue(web, jid)["items"][0])["ok"]
    jobs.gc(now=time.time() + 30 * DAY)
    assert (
        reg.read_bytes() == before_reg
        and sorted(p.name for p in (two[DS] / "clips").iterdir()) == clips_before
    )
    assert sorted(p.name for p in (two[DS] / "marks").iterdir()) == [
        "ball_checks"
    ]  # only the verdict log
    listing = web.get(f"/clips?dataset={DS}").json()["clips"]
    assert [c["name"] for c in listing] == [clip_file.name]
    assert (
        listing[0]["marked"] is False
    )  # a verdict log is not a sidecar: the clip does not turn "marked"


def test_a_run_started_by_hand_reports_without_a_prior_request(web, two, clip_file):
    """``$PF run ballcheck-prepare/... job=<id>`` has no server-side request behind it: its events still land."""
    jid = _submit(web, [_card(5)]).json()["id"]
    assert "prepare" not in web.get(f"/api/jobs/{jid}").json()
    r = web.post(
        f"/api/jobs/{jid}/prepare_event",
        json={"action": "run", "run_id": "r1", "where": "desktop"},
    )
    assert r.json()["state"] == "running" and r.json()["where"] == "desktop"
    web.post(
        f"/api/jobs/{jid}/prepare_event",
        json={"action": "finish", "ok": True, "prepared": 1, "total": 1},
    )
    j = web.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "open" and j["prepare"]["state"] == "ready"


def test_gc_endpoint_dry_run_reports_and_changes_nothing(web, two, clip_file):
    jid, rc, keys = _prepared_job(web, two, bcc.root())
    web.post(f"/api/jobs/{jid}/done")
    r = web.post("/api/jobs/gc?dry_run=true")
    assert r.status_code == 200 and r.json()["dry_run"] is True and rc.has(*keys[0])


def test_prefect_client_queues_a_run_with_basic_auth_read_from_the_env_file(
    tmp_path, monkeypatch
):
    seen: list[tuple[str, str, str | None, bytes]] = []

    class H(http.server.BaseHTTPRequestHandler):
        def _reply(self, doc):
            body = json.dumps(doc).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen.append(("GET", self.path, self.headers.get("Authorization"), b""))
            self._reply({"id": "dep-1"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen.append(
                (
                    "POST",
                    self.path,
                    self.headers.get("Authorization"),
                    self.rfile.read(n),
                )
            )
            self._reply({"id": "run-9"})

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = tmp_path / "prefect.env"
    env.write_text("OTHER=1\nPREFECT_API_AUTH_STRING=user:pw\n")
    monkeypatch.setenv("PREFECT_API_URL", f"http://127.0.0.1:{srv.server_port}/api")
    monkeypatch.setenv("FOOTY_PREFECT_ENV_FILE", str(env))
    monkeypatch.delenv("PREFECT_API_AUTH_STRING", raising=False)
    try:
        assert (
            prefect_client.create_flow_run({"job": "j-1", "where": "mini"}, name="bc")
            == "run-9"
        )
    finally:
        srv.shutdown()
    auth = "Basic " + base64.b64encode(b"user:pw").decode()
    assert seen[0][:3] == (
        "GET",
        "/api/deployments/name/ballcheck-prepare/ballcheck-prepare",
        auth,
    )
    assert seen[1][:3] == ("POST", "/api/deployments/dep-1/create_flow_run", auth)
    assert json.loads(seen[1][3]) == {
        "parameters": {"job": "j-1", "where": "mini"},
        "name": "bc",
    }
