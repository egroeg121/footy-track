"""Labelling hooks (README 19c): the label-write event point of every kind, per-job hook config and its
bounds, the reorder / append mutation API, pages picking up a new order, and ``broadcast_calibrate``.

Fixtures ``two`` / ``web`` (conftest): two datasets (calib default, nottinghamforest) in tmp dirs.
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
from pathlib import Path

import pytest

from footy_track.labeller import ball_check_cache as bcc
from footy_track.labeller import broadcast_review as br
from footy_track.labeller import hooks, job_edit, jobs, jobs_client, prefect_client
from footy_track.labeller import server as labeller_server

DS = "nottinghamforest"
MATCH = "che.v.mun.full.epl.31.10.2026"
CLIP = "arsenal_nottinghamforest_seg003"


@pytest.fixture(autouse=True)
def spy(monkeypatch):
    """A registered in-process action that records the event it ran for."""
    hooks._reset_for_tests()
    seen: list = []
    hooks.register_action("spy", lambda ctx: seen.append((ctx.job["id"], ctx.trigger, ctx.event)) or {"n": len(seen)})
    yield seen
    hooks.drain()
    hooks.ACTIONS.pop("spy", None)
    hooks._reset_for_tests()


def hook(name="h", every_n=1, interval=0, action="spy", **trig):
    return {"name": name, "trigger": {"every_n": every_n, "min_interval_s": interval, **trig},
            "action": {"type": "builtin", "name": action}}  # fmt: skip


def submit(web, kind, items, hooks_=None, **over):
    body = {"title": f"t-{kind}", "kind": kind, "dataset": DS, "items": items, **over}
    if hooks_ is not None:
        body["hooks"] = hooks_
    r = web.post("/api/jobs", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def bc_items(frames, p=0.5):
    return [{"clip": MATCH, "frame": f, "ts_s": f / 10, "p_yes": p if not isinstance(p, dict) else p[f], "pred": "Yes", "reason": "edge"} for f in frames]


def verdict(web, frame, v="yes", p=0.5):
    r = web.put(f"/broadcast_review/api/verdict/{MATCH}/{frame}?dataset={DS}", json={"verdict": v, "p_yes": p})
    assert r.status_code == 200, r.text


def stored(job_id):
    return jobs.read_job(job_id)


def frames_in_order(job_id):
    return [it["frame"] for it in stored(job_id)["items"]]


# ---------------------------------------------------------------------------
# LAB-1417: one event point per label write, every kind
# ---------------------------------------------------------------------------


def test_event_fires_for_a_non_broadcast_kind_ball_check(web, two, spy):
    cards = [{"clip": CLIP, "frame_index": f, "bbox": {"x": 0.4, "y": 0.2, "w": 0.01, "h": 0.02}} for f in (10, 20)]
    job = submit(web, "ball_check", cards, [hook()])
    stored_cards = stored(job["id"])["items"]
    c0 = stored_cards[0]
    r = web.post(f"/ball_check/verdict?dataset={DS}", json={"clip": CLIP, "frame_index": 10, "box_index": c0["box_index"],
                                                            "source": "list", "verdict": "ball"})  # fmt: skip
    assert r.json()["ok"]
    # a verdict on a card that is in no hooked job is not this job's event
    web.post(f"/ball_check/verdict?dataset={DS}", json={"clip": CLIP, "frame_index": 99, "box_index": 5, "source": "sidecar", "verdict": "ball"})
    assert hooks.drain()
    assert len(spy) == 1
    job_id, trigger, ev = spy[0]
    assert (job_id, trigger) == (job["id"], "every_n")
    assert (ev.kind, ev.dataset, ev.clip, ev.verdict) == ("ball_check", DS, CLIP, "ball")
    assert ev.unit == ("list", 10, c0["box_index"])
    web.post(f"/ball_check/undo?dataset={DS}", json={"clip": CLIP, "frame_index": 10, "box_index": c0["box_index"], "source": "list"})
    assert hooks.drain()
    assert [e.verdict for _, _, e in spy] == ["ball", None]  # an undo is an event too


def test_event_fires_for_broadcast_review_and_only_for_that_jobs_frames(web, two, spy):
    job = submit(web, "broadcast_review", bc_items([100, 200]), [hook()])
    verdict(web, 100, "no")
    verdict(web, 999, "yes")  # not in the job
    assert hooks.drain()
    assert [(e.kind, e.clip, e.unit, e.verdict) for _, _, e in spy] == [("broadcast_review", MATCH, 100, "no")]
    assert spy[0][0] == job["id"]


def test_event_fires_for_ball_place_marks(web, two, spy):
    sess = labeller_server.SESSION
    sess.video_path = Path(f"/x/{CLIP}.mp4")
    sess.total_frames = 30
    sess.timeline = [None] * 30
    sess.schedule_flush = lambda: None
    submit(web, "ball_place", [{"clip": CLIP, "frame_start": 10, "frame_end": 19, "step": 3}], [hook()], dataset="calib")
    assert web.post("/no-ball", json={"idx": 13}).status_code == 200  # 10,13,16,19 are the job's frames
    assert hooks.drain()  # one run in flight per hook: let it finish before the next write
    web.post("/no-ball", json={"idx": 14})  # in the window but off the step: not a unit of the job
    web.post("/not-broadcast", json={"idx": 16})
    assert hooks.drain()
    assert [(e.kind, e.unit, e.verdict) for _, _, e in spy] == [("ball_place", 13, "no_ball"), ("ball_place", 16, "not_broadcast")]


def test_event_fires_for_frame_type_flags_on_calibration_and_frame_type_jobs(web, two, spy):
    submit(web, "calibration", [{"clip": CLIP, "frame": 7}], [hook("c")], title="cal")
    submit(web, "frame_type", [{"clip": CLIP, "frame": 7}], [hook("f")], title="ft")
    r = web.put(f"/calibration/api/frame_type/{CLIP}/7?dataset={DS}", json={})
    assert r.status_code == 200, r.text
    assert hooks.drain()
    assert sorted(spy_job_kind(spy)) == ["calibration", "frame_type"]
    assert {e.kind for _, _, e in spy} == {"frame_type"}


def spy_job_kind(spy):
    return [jobs.read_job(j)["kind"] for j, _, _ in spy]


def test_event_fires_for_identity_pairs_and_reviews(web, two, spy):
    submit(web, "identity", [{"clip": "clipA"}], [hook()], title="ia")  # one job per clip: a hook has one run in flight
    submit(web, "identity", [{"clip": "clipB"}], [hook()], title="ib")
    a, b = {"clip": "clipA", "track_id": 1}, {"clip": "clipB", "track_id": 2}
    assert web.post(f"/identity/pair?dataset={DS}", json={"a": a, "b": b, "verdict": "same"}).json()["ok"]
    assert hooks.drain()
    assert sorted(e.clip for _, _, e in spy) == ["clipA", "clipB"]
    assert {e.kind for _, _, e in spy} == {"identity"}


def test_a_failing_or_full_hook_pipeline_never_breaks_the_write(web, two, monkeypatch):
    job = submit(web, "broadcast_review", bc_items([100, 200]), [hook()])
    def boom(_item):
        raise RuntimeError("boom")

    def full(_item):
        raise queue.Full

    hooks._Q.put_nowait = boom
    try:
        r = web.put(f"/broadcast_review/api/verdict/{MATCH}/100?dataset={DS}", json={"verdict": "yes", "p_yes": 0.5})
        assert r.status_code == 200 and web.get(f"/api/jobs/{job['id']}?progress=true").json()["progress"]["done"] == 1
        assert hooks.stats()["errors"] == 1
        # a full queue drops events and counts them, the write still succeeds
        hooks._Q.put_nowait = full
        verdict(web, 200)
    finally:
        del hooks._Q.put_nowait
    assert hooks.stats()["dropped"] == 1


def test_emit_is_off_the_request_thread(web, two, spy):
    gate = threading.Event()
    hooks.register_action("slow", lambda ctx: gate.wait(10) and {})
    try:
        submit(web, "broadcast_review", bc_items([100, 200]), [hook(action="slow")])
        t0 = time.monotonic()
        verdict(web, 100)
        assert time.monotonic() - t0 < 2  # the answer did not wait for the hook
        gate.set()
        assert hooks.drain()
    finally:
        gate.set()
        hooks.ACTIONS.pop("slow", None)


# ---------------------------------------------------------------------------
# LAB-1418: per-job hook config
# ---------------------------------------------------------------------------


def test_hooks_are_set_at_submit_and_updated_through_the_api(web, two):
    job = submit(web, "broadcast_review", bc_items([100]), [hook("a", every_n=5, interval=10, on_done=True)])
    h = stored(job["id"])["hooks"]
    assert h == [{"name": "a", "trigger": {"min_interval_s": 10.0, "on_done": True, "every_n": 5},
                  "action": {"type": "builtin", "name": "spy", "params": {}}}]  # fmt: skip
    r = web.put(f"/api/jobs/{job['id']}/hooks", json={"hooks": [hook("b"), hook("c")]})
    assert r.status_code == 200 and [x["name"] for x in r.json()["hooks"]] == ["b", "c"]
    g = web.get(f"/api/jobs/{job['id']}/hooks").json()
    assert [x["name"] for x in g["hooks"]] == ["b", "c"] and "spy" in g["actions"] and "broadcast_calibrate" in g["actions"]
    assert web.put(f"/api/jobs/{job['id']}/hooks", json={"hooks": []}).json()["hooks"] == []
    assert "hooks" not in stored(job["id"])


def test_hook_config_is_validated_and_capped(web, two):
    j = submit(web, "broadcast_review", bc_items([100]))["id"]

    def put(hs):
        return web.put(f"/api/jobs/{j}/hooks", json={"hooks": hs})

    assert put([hook(f"h{i}") for i in range(hooks.MAX_HOOKS_PER_JOB + 1)]).status_code == 422
    assert put([hook(f"h{i}") for i in range(hooks.MAX_HOOKS_PER_JOB)]).status_code == 200
    assert put([hook("x"), hook("x")]).status_code == 422  # duplicate name
    assert put([{"name": "Bad Name", "trigger": {"every_n": 1}, "action": {"type": "builtin", "name": "spy"}}]).status_code == 422
    assert put([{"name": "n", "trigger": {}, "action": {"type": "builtin", "name": "spy"}}]).status_code == 422  # no trigger
    assert put([hook("n", every_n=0)]).status_code == 422
    assert put([hook("n", action="nope")]).status_code == 422
    # broadcast_calibrate is for broadcast_review jobs only
    other = submit(web, "frame_type", [{"clip": CLIP, "frame": 1}], title="ft")["id"]
    bad = web.put(f"/api/jobs/{other}/hooks", json={"hooks": [hook("n", action="broadcast_calibrate")]})
    assert bad.status_code == 422 and "broadcast_review" in bad.json()["detail"]
    pf = {"name": "p", "trigger": {"every_n": 5, "min_interval_s": 5}, "action": {"type": "prefect", "deployment": "flow/dep"}}
    assert put([pf]).status_code == 422  # prefect actions have a debounce floor
    assert put([{**pf, "trigger": {"every_n": 5}, "action": {"type": "prefect", "deployment": "not a deployment"}}]).status_code == 422
    assert put([{**pf, "trigger": {"every_n": 5}}]).status_code == 200


def test_cli_hooks_roundtrip(web, two, tmp_path, monkeypatch):
    job = submit(web, "broadcast_review", bc_items([100]))
    monkeypatch.setattr(jobs_client, "_request", lambda method, path, **kw: _via_testclient(web, method, path, kw.get("body")))
    out = jobs_client.set_hooks(job["id"], [hook("a")])
    assert out["hooks"][0]["name"] == "a"
    assert jobs_client.get_hooks(job["id"])["hooks"][0]["name"] == "a"
    res = jobs_client.reorder(job["id"], [0])
    assert res["changed"] is False


def _via_testclient(web, method, path, body):
    r = web.request(method, path, json=body)
    if r.status_code >= 400:
        raise jobs_client_error(r)
    return r.json()


def jobs_client_error(r):
    return jobs_client.JobsError(r.status_code, r.json())


# ---------------------------------------------------------------------------
# LAB-1419: triggers and bounds
# ---------------------------------------------------------------------------


def test_every_n_triggers_after_n_events_and_debounce_holds_the_rest(web, two, spy):
    submit(web, "broadcast_review", bc_items(range(10)), [hook(every_n=3, interval=3600)])
    for f in range(7):
        verdict(web, f)
        hooks.drain()
    assert len(spy) == 1  # the 3rd event fired it; the 6th was inside the debounce window
    assert spy[0][2].unit == 2


def test_at_most_one_run_in_flight_per_job_and_hook(web, two):
    gate, started = threading.Event(), []
    hooks.register_action("block", lambda ctx: started.append(1) or gate.wait(10) or {})
    try:
        submit(web, "broadcast_review", bc_items(range(6)), [hook(action="block")])
        verdict(web, 0)
        deadline = time.monotonic() + 5
        while not started and time.monotonic() < deadline:
            time.sleep(0.01)
        for f in (1, 2, 3):
            verdict(web, f)
        time.sleep(0.2)
        assert len(started) == 1  # the first run is still going: nothing else was started
        gate.set()
        assert hooks.drain()
        verdict(web, 4)
        assert hooks.drain()
        assert len(started) == 2
    finally:
        gate.set()
        hooks.ACTIONS.pop("block", None)


def test_a_failed_run_backs_off_for_ten_minutes_and_is_recorded(web, two, spy):
    hooks.register_action("boom", lambda ctx: 1 / 0)
    try:
        job = submit(web, "broadcast_review", bc_items(range(6)), [hook(action="boom")])
        verdict(web, 0)
        assert hooks.drain()
        view = web.get(f"/api/jobs/{job['id']}/hooks").json()
        st = view["state"]["h"]
        assert "ZeroDivisionError" in st["last_error"] and 590 < st["backoff_until"] - time.time() <= 600
        assert view["history"][-1]["event"] == "run" and view["history"][-1]["ok"] is False
        runs = len(view["history"])
        verdict(web, 1)
        verdict(web, 2)
        assert hooks.drain()
        assert len(web.get(f"/api/jobs/{job['id']}/hooks").json()["history"]) == runs  # backing off: nothing ran
        # the backoff survives a restart (it is stored in the job)
        hooks._RT.clear()
        assert hooks.runtime_view(stored(job["id"]))["h"]["backoff_until"] == pytest.approx(st["backoff_until"], abs=1)
    finally:
        hooks.ACTIONS.pop("boom", None)


def test_on_done_runs_once_when_the_job_completes(web, two, spy):
    job = submit(web, "broadcast_review", bc_items([1, 2]), [{"name": "fin", "trigger": {"on_done": True}, "action": {"type": "builtin", "name": "spy"}}])
    verdict(web, 1)
    assert hooks.drain() and not spy
    verdict(web, 2)
    assert hooks.drain()
    assert [(t, e) for _, t, e in spy] == [("on_done", None)]
    assert stored(job["id"])["status"] == "done"


def test_manual_run_skips_the_debounce_but_not_the_backoff(web, two, spy):
    job = submit(web, "broadcast_review", bc_items(range(4)), [hook(every_n=100, interval=3600)])
    r = web.post(f"/api/jobs/{job['id']}/hooks/h/run")
    assert r.status_code == 200 and r.json()["history"][-1]["trigger"] == "manual"
    assert len(spy) == 1
    assert web.post(f"/api/jobs/{job['id']}/hooks/nope/run").status_code == 404


def test_prefect_action_queues_the_deployment_and_stays_in_flight_until_it_ends(web, two, monkeypatch):
    monkeypatch.setattr(hooks, "MIN_PREFECT_INTERVAL_S", 0)
    monkeypatch.setattr(hooks, "PREFECT_POLL_S", 0)
    queued, states = [], ["RUNNING", "COMPLETED"]
    monkeypatch.setattr(prefect_client, "create_flow_run", lambda params, deployment, name=None: queued.append((deployment, params, name)) or f"run-{len(queued)}")
    monkeypatch.setattr(prefect_client, "flow_run_state", lambda run_id: states.pop(0) if len(states) > 1 else states[0])
    pf = {"name": "rescore", "trigger": {"every_n": 1, "min_interval_s": 0},
          "action": {"type": "prefect", "deployment": "my-flow/my-dep", "params": {"top": 50}}}  # fmt: skip
    job = submit(web, "broadcast_review", bc_items(range(6)), [pf])
    verdict(web, 0)
    assert hooks.drain()
    dep, params, name = queued[0]
    assert dep == "my-flow/my-dep" and params["job_id"] == job["id"] and params["top"] == 50 and params["hook"] == "rescore"
    verdict(web, 1)  # run-1 is still RUNNING: polled, not duplicated
    assert hooks.drain()
    assert len(queued) == 1 and hooks.runtime_view(stored(job["id"]))["rescore"]["in_flight"]["run_id"] == "run-1"
    verdict(web, 2)  # COMPLETED now: in flight clears ...
    assert hooks.drain()
    verdict(web, 3)  # ... so the next event starts run 2
    assert hooks.drain()
    assert len(queued) == 2


def test_prefect_failure_backs_off_and_unreachable_prefect_is_recorded(web, two, monkeypatch):
    monkeypatch.setattr(hooks, "MIN_PREFECT_INTERVAL_S", 0)

    def down(*a, **k):
        raise prefect_client.PrefectError("connection refused")

    monkeypatch.setattr(prefect_client, "create_flow_run", down)
    pf = {"name": "p", "trigger": {"every_n": 1, "min_interval_s": 0}, "action": {"type": "prefect", "deployment": "f/d"}}
    job = submit(web, "broadcast_review", bc_items(range(4)), [pf])
    verdict(web, 0)
    assert hooks.drain()
    st = hooks.runtime_view(stored(job["id"]))["p"]
    assert "connection refused" in st["last_error"] and st["in_flight"] is None and st["backoff_until"] > time.time() + 500


def test_prefect_inflight_cap_across_jobs(web, two, monkeypatch):
    monkeypatch.setattr(hooks, "MIN_PREFECT_INTERVAL_S", 0)
    monkeypatch.setattr(hooks, "MAX_PREFECT_INFLIGHT", 1)
    monkeypatch.setattr(hooks, "PREFECT_POLL_S", 3600)
    queued = []
    monkeypatch.setattr(prefect_client, "create_flow_run", lambda params, deployment, name=None: queued.append(params["job_id"]) or "r")
    pf = {"name": "p", "trigger": {"every_n": 1, "min_interval_s": 0}, "action": {"type": "prefect", "deployment": "f/d"}}
    submit(web, "broadcast_review", bc_items([1]), [pf], title="a")
    submit(web, "broadcast_review", [{**bc_items([2])[0], "clip": "other.match"}], [pf], title="b")
    verdict(web, 1)
    assert hooks.drain()
    web.put(f"/broadcast_review/api/verdict/other.match/2?dataset={DS}", json={"verdict": "yes"})
    assert hooks.drain()
    assert len(queued) == 1


# ---------------------------------------------------------------------------
# LAB-1420: reorder the unanswered items only
# ---------------------------------------------------------------------------


def test_reorder_moves_only_unanswered_items_and_never_answered_ones(web, two):
    job = submit(web, "broadcast_review", bc_items(range(1, 9)))["id"]
    verdict(web, 2, "no")
    verdict(web, 5, "yes")
    before = stored(job)["items"]
    answered = {1: before[1], 4: before[4]}  # positions of frames 2 and 5
    r = web.post(f"/api/jobs/{job}/reorder", json={"order": [7, 6, 4, 1, 0, 99, 7, "x"], "by": "test", "note": "n"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] and body["skipped"] == {"answered": 2, "unknown": 2, "duplicate": 1}
    after = stored(job)["items"]
    assert [after[i] for i in answered] == list(answered.values())  # answered items did not move
    assert [it["frame"] for it in after] == [8, 2, 7, 1, 5, 3, 4, 6]  # unanswered: 8,7,1 first, rest keep order
    assert sorted(it["frame"] for it in after) == list(range(1, 9))  # nothing lost or duplicated
    assert body["rev"] == 1 == stored(job)["rev"]
    hist = stored(job)["hook_history"][-1]
    assert hist["event"] == "reorder" and hist["by"] == "test" and hist["note"] == "n"
    # verdicts untouched, progress unchanged
    g = web.get(f"/api/jobs/{job}?progress=true").json()["progress"]
    assert (g["done"], g["total"]) == (2, 8)


def test_reorder_is_a_noop_when_nothing_changes_and_honours_if_rev(web, two):
    job = submit(web, "broadcast_review", bc_items(range(1, 5)))["id"]
    assert web.post(f"/api/jobs/{job}/reorder", json={"order": [0, 1]}).json()["changed"] is False
    assert stored(job).get("rev", 0) == 0
    assert web.post(f"/api/jobs/{job}/reorder", json={"order": [3], "if_rev": 0}).json()["rev"] == 1
    stale = web.post(f"/api/jobs/{job}/reorder", json={"order": [2], "if_rev": 0})
    assert stale.status_code == 409 and stale.json()["rev"] == 1
    assert web.post(f"/api/jobs/{job}/reorder", json={"order": "nope"}).status_code == 422
    web.post(f"/api/jobs/{job}/cancel")
    assert web.post(f"/api/jobs/{job}/reorder", json={"order": [0]}).status_code == 409  # only live jobs


def test_reorder_works_for_window_items_of_a_frame_kind(web, two):
    items = [{"clip": CLIP, "frame_start": s, "frame_end": s + 2} for s in (0, 10, 20)]
    job = submit(web, "ball_place", items, dataset="calib")["id"]
    assert web.post(f"/api/jobs/{job}/reorder", json={"order": [2]}).json()["changed"]
    assert [it["frame_start"] for it in stored(job)["items"]] == [20, 0, 10]
    assert web.get(f"/api/jobs/{job}/frames").json()["frames"][0]["frame"] == 20


def test_generic_builtin_actions_shuffle_and_sort_any_kind(web, two):
    items = bc_items(range(1, 7), p={f: f / 10 for f in range(1, 7)})
    job = submit(web, "broadcast_review", items, [hook("s", action="sort_by_field", every_n=100)])["id"]
    web.put(f"/api/jobs/{job}/hooks", json={"hooks": [{"name": "s", "trigger": {"every_n": 100, "min_interval_s": 0},
                                                       "action": {"type": "builtin", "name": "sort_by_field",
                                                                  "params": {"key": "p_yes", "descending": True}}}]})  # fmt: skip
    assert web.post(f"/api/jobs/{job}/hooks/s/run").status_code == 200
    assert frames_in_order(job) == [6, 5, 4, 3, 2, 1]
    web.put(f"/api/jobs/{job}/hooks", json={"hooks": [hook("r", action="shuffle_unanswered", every_n=100)]})
    web.post(f"/api/jobs/{job}/hooks/r/run")
    assert sorted(frames_in_order(job)) == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# LAB-1421: append with provenance and the existing bounds
# ---------------------------------------------------------------------------


def test_append_adds_items_with_provenance_skips_duplicates_and_keeps_answers(web, two):
    job = submit(web, "broadcast_review", bc_items([1, 2, 3]))["id"]
    verdict(web, 1, "yes")
    r = web.post(f"/api/jobs/{job}/append", json={"items": bc_items([3, 4, 5]), "by": "hook:topup", "note": "near boundary"})
    assert r.status_code == 200, r.text
    assert (r.json()["added"], r.json()["skipped_duplicates"], r.json()["total"]) == (2, 1, 5)
    items = stored(job)["items"]
    assert [it["frame"] for it in items] == [1, 2, 3, 4, 5]
    assert "added_by" not in items[0] and items[3]["added_by"]["by"] == "hook:topup" and items[3]["added_by"]["note"] == "near boundary"
    assert stored(job)["hook_history"][-1] == {**stored(job)["hook_history"][-1], "event": "append", "added": 2}
    g = web.get(f"/api/jobs/{job}?progress=true").json()["progress"]
    assert (g["done"], g["total"]) == (1, 5)  # the answered item and its verdict are untouched
    res = web.get(f"/api/jobs/{job}/results").json()
    assert [(row["frame"], row["verdict"]) for row in res["labels"]] == [(1, "yes")]


def test_append_respects_the_per_job_cap_and_per_call_cap(web, two, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_FRAMES_DEFAULT", 5)
    job = submit(web, "broadcast_review", bc_items([1, 2, 3]))["id"]
    r = web.post(f"/api/jobs/{job}/append", json={"items": bc_items([4, 5, 6])})
    assert r.status_code == 422 and r.json()["cap"] == 5
    assert [it["frame"] for it in stored(job)["items"]] == [1, 2, 3]  # nothing was written
    assert web.post(f"/api/jobs/{job}/append", json={"items": bc_items([4, 5])}).status_code == 200
    monkeypatch.setattr(job_edit, "MAX_APPEND_PER_CALL", 1)
    assert web.post(f"/api/jobs/{job}/append", json={"items": bc_items([6, 7])}).status_code == 422


def test_append_to_a_prepared_ball_check_job_honours_queued_cap_and_disk_floor(web, two, monkeypatch):
    def card(f):
        return {"clip": CLIP, "frame_index": f, "bbox": {"x": 0.4, "y": 0.2, "w": 0.01, "h": 0.02}}

    job = submit(web, "ball_check", [card(f) for f in range(3)])["id"]
    j = stored(job)
    j["prepare"] = {"state": "ready"}
    jobs._write_atomic(jobs._job_path(job), j)
    monkeypatch.setattr(jobs, "MAX_QUEUED_CARDS", 4)
    r = web.post(f"/api/jobs/{job}/append", json={"items": [card(10), card(11)]})
    assert r.status_code == 429 and r.json()["cap"] == 4
    assert web.post(f"/api/jobs/{job}/append", json={"items": [card(10)]}).status_code == 200
    # appended inline cards get their own, distinct box indexes
    boxes = [it["box_index"] for it in stored(job)["items"]]
    assert len(set(boxes)) == 4 and all(b >= jobs.INLINE_BOX_BASE for b in boxes)
    def floor(*a, **k):
        raise bcc.DiskFloorError("below the floor")

    monkeypatch.setattr(jobs, "MAX_QUEUED_CARDS", 1500)
    monkeypatch.setattr(bcc, "check_floor", floor)
    monkeypatch.setattr(bcc, "root", lambda: Path("/"))
    assert web.post(f"/api/jobs/{job}/append", json={"items": [card(12)]}).status_code == 507


def test_append_is_refused_for_closed_jobs_and_validates_items(web, two):
    job = submit(web, "broadcast_review", bc_items([1]))["id"]
    assert web.post(f"/api/jobs/{job}/append", json={"items": [{"clip": MATCH}]}).status_code == 422  # no frame
    web.post(f"/api/jobs/{job}/done")
    assert web.post(f"/api/jobs/{job}/append", json={"items": bc_items([2])}).status_code == 409


# ---------------------------------------------------------------------------
# LAB-1422: pages that walk a job in order pick up the new order
# ---------------------------------------------------------------------------


def test_broadcast_page_api_serves_the_new_order_and_rev(web, two):
    job = submit(web, "broadcast_review", bc_items(range(1, 5)))["id"]
    v0 = web.get(f"/broadcast_review/api/job/{job}?dataset={DS}").json()
    assert v0["rev"] == 0 and [f["frame"] for f in v0["frames"]] == [1, 2, 3, 4]
    web.post(f"/api/jobs/{job}/reorder", json={"order": [3, 2]})
    v1 = web.get(f"/broadcast_review/api/job/{job}?dataset={DS}").json()
    assert v1["rev"] == 1 and [f["frame"] for f in v1["frames"]] == [4, 3, 1, 2]
    assert web.get(f"/api/jobs/{job}/rev").json() == {"id": job, "rev": 1, "status": "open"}


def test_ball_check_queue_serves_the_new_order_and_rev(web, two):
    cards = [{"clip": CLIP, "frame_index": f, "bbox": {"x": 0.4, "y": 0.2, "w": 0.01, "h": 0.02}} for f in (10, 20, 30)]
    job = submit(web, "ball_check", cards)["id"]
    q0 = web.get(f"/ball_check/queue?dataset={DS}&source=list&job={job}").json()
    assert q0["rev"] == 0 and [c["frame_index"] for c in q0["items"]] == [10, 20, 30]
    web.post(f"/api/jobs/{job}/reorder", json={"order": [2, 1]})
    q1 = web.get(f"/ball_check/queue?dataset={DS}&source=list&job={job}").json()
    assert q1["rev"] == 1 and [c["frame_index"] for c in q1["items"]] == [30, 20, 10]


def test_labeller_job_frames_carry_the_rev(web, two):
    job = submit(web, "ball_place", [{"clip": CLIP, "frame_start": s, "frame_end": s} for s in (1, 2, 3)], dataset="calib")["id"]
    assert web.get(f"/api/jobs/{job}/frames").json()["rev"] == 0
    web.post(f"/api/jobs/{job}/reorder", json={"order": [2]})
    f = web.get(f"/api/jobs/{job}/frames").json()
    assert f["rev"] == 1 and [x["frame"] for x in f["frames"]] == [3, 1, 2]


def test_pages_poll_the_rev_and_keep_the_users_place():
    web_dir = Path(__file__).parents[2] / "src" / "footy_track" / "labeller" / "web"
    bc = (web_dir / "broadcast_review.html").read_text()
    assert "job.rev" in bc and "hereKey" in bc and "p_cal" in bc and "setInterval(refresh" in bc
    assert "/rev`" in (web_dir / "ball_check.html").read_text() and "queue.slice(0, 2)" in (web_dir / "ball_check.html").read_text()
    assert "jobRefresh" in (web_dir / "index.html").read_text()


# ---------------------------------------------------------------------------
# LAB-1423..1425: broadcast_calibrate
# ---------------------------------------------------------------------------


def _miscalibrated(n, boundary=0.005, seed=1):
    """(p_yes, verdict) with the true 50/50 point at p_yes ~ ``boundary`` and a soft edge in logit space."""
    rng = random.Random(seed)
    lb = br.logit(boundary)
    out = []
    for _ in range(n):
        z = rng.uniform(-9, 3)  # logit(p_yes): the scores span 1e-4 .. 0.95
        p = br.sigmoid(z)
        yes = rng.random() < br.sigmoid(1.6 * (z - lb))
        out.append((p, "yes" if yes else "no"))
    return out


def test_platt_fit_recovers_the_known_scaling_and_boundary():
    rng = random.Random(3)
    xs = [rng.uniform(-9, 3) for _ in range(4000)]
    ys = [1 if rng.random() < br.sigmoid(1.6 * x + 8.5) else 0 for x in xs]
    a, b = br.fit_platt(xs, ys)
    assert a == pytest.approx(1.6, abs=0.15) and b == pytest.approx(8.5, abs=0.8)
    # separable data stays finite (Platt's smoothed targets)
    a2, b2 = br.fit_platt([-5, -4, -3, 3, 4, 5] * 3, [0, 0, 0, 1, 1, 1] * 3)
    assert 0 < a2 < 50 and abs(b2) < 5


def _seed_verdicts(web, rows, start=0):
    """Write verdicts (with p_yes) through the real route; returns {frame: p}."""
    ps = {}
    for k, (p, v) in enumerate(rows):
        f = start + k
        web.put(f"/broadcast_review/api/verdict/{MATCH}/{f}?dataset={DS}", json={"verdict": v, "p_yes": p})
        ps[f] = p
    return ps


def test_broadcast_calibrate_writes_a_versioned_file_and_reorders_by_uncertainty(web, two):
    rows = _miscalibrated(120)
    ps = _seed_verdicts(web, rows)  # frames 0..119 answered, p_yes stored with each
    todo = {1000 + k: br.sigmoid(z) for k, z in enumerate([-9, -8, -7, -6, -5.5, -5.3, -5.0, -4.5, -3, -1, 0, 2])}
    items = [{"clip": MATCH, "frame": f, "ts_s": float(f), "p_yes": p, "pred": "Yes", "reason": "edge"} for f, p in todo.items()]
    items = items[::-1]  # submitted most-confident-first, so any reorder is visible
    ans_items = [{"clip": MATCH, "frame": f, "ts_s": float(f), "p_yes": p} for f, p in list(ps.items())[:3]]
    job = submit(web, "broadcast_review", [*ans_items, *items], [hook("cal", every_n=1000, action="broadcast_calibrate")])["id"]
    web.put(f"/api/jobs/{job}/hooks", json={"hooks": [{"name": "cal", "trigger": {"every_n": 1000, "min_interval_s": 0},
        "action": {"type": "builtin", "name": "broadcast_calibrate", "params": {"jitter": 0.0, "penalty": 0.0}}}]})  # fmt: skip
    r = web.post(f"/api/jobs/{job}/hooks/cal/run")
    assert r.status_code == 200, r.text
    res = r.json()["history"][-1]
    assert res["ok"], res
    out = res["result"]
    assert out["version"] == 1 and out["n"] == 120 and out["changed"] is True

    path = two[DS] / "marks" / "broadcast_review" / "calibration" / "v001.json"
    cal = json.loads(path.read_text())
    assert cal["version"] == 1 and cal["n"] == 120 and cal["n_yes"] + cal["n_no"] == 120 and cal["a"] > 0
    assert cal["boundary_p_yes"] == pytest.approx(br.sigmoid(-cal["b"] / cal["a"]))
    assert 0.0005 < cal["boundary_p_yes"] < 0.05  # near the true 0.005, nowhere near 0.5
    assert br.calibrated_p(cal["boundary_p_yes"], cal) == pytest.approx(0.5)
    assert {"a", "b", "n", "ts", "boundary_p_yes", "version"} <= set(cal)

    # the answered items kept their positions; the rest are ordered by |a*logit(p)+b| ascending
    after = stored(job)["items"]
    assert [it["frame"] for it in after[:3]] == [it["frame"] for it in ans_items]
    rest = after[3:]
    d = [abs(cal["a"] * br.logit(it["p_yes"]) + cal["b"]) for it in rest]
    assert d == sorted(d) and sorted(it["frame"] for it in rest) == sorted(todo)
    assert stored(job)["hook_history"][-2]["event"] == "reorder"

    # the page shows calibrated p and the boundary next to the raw p_yes
    view = web.get(f"/broadcast_review/api/job/{job}?dataset={DS}").json()
    assert view["calibration"]["version"] == 1
    row = next(f for f in view["frames"] if f["frame"] == rest[0]["frame"])
    assert row["p_cal"] == pytest.approx(br.calibrated_p(row["p_yes"], cal))

    # same data again: no new version, and the order is stable
    again = web.post(f"/api/jobs/{job}/hooks/cal/run").json()["history"][-1]["result"]
    assert again["version"] == 1 and again["reused"] is True and not path.with_name("v002.json").exists()
    # new verdicts -> a new version
    _seed_verdicts(web, _miscalibrated(20, seed=9), start=500)
    assert web.post(f"/api/jobs/{job}/hooks/cal/run").json()["history"][-1]["result"]["version"] == 2


def test_broadcast_calibrate_is_a_noop_with_a_reason_when_there_are_too_few_verdicts(web, two):
    rows = [(0.9, "yes")] * 12 + [(0.001, "no")] * 5 + [(0.3, "unsure")] * 20
    _seed_verdicts(web, rows)
    job = submit(web, "broadcast_review", bc_items([5000, 5001]), [hook("cal", every_n=1000, action="broadcast_calibrate")])["id"]
    r = web.post(f"/api/jobs/{job}/hooks/cal/run").json()["history"][-1]
    assert r["ok"] and "need >= 10 yes and >= 10 no" in r["result"]["noop"] and "12 yes / 5 no" in r["result"]["noop"]
    assert not (two[DS] / "marks" / "broadcast_review" / "calibration").exists()
    assert frames_in_order(job) == [5000, 5001]


def test_broadcast_calibrate_fires_as_verdicts_arrive(web, two):
    _seed_verdicts(web, _miscalibrated(60))
    items = bc_items(range(2000, 2030), p={f: br.sigmoid(-9 + (f - 2000) * 0.3) for f in range(2000, 2030)})
    job = submit(web, "broadcast_review", items, [{"name": "cal", "trigger": {"every_n": 2, "min_interval_s": 0},
                                                   "action": {"type": "builtin", "name": "broadcast_calibrate"}}])["id"]  # fmt: skip
    verdict(web, 2000, "no", p=items[0]["p_yes"])
    verdict(web, 2001, "no", p=items[1]["p_yes"])
    assert hooks.drain(10)
    assert stored(job)["rev"] >= 1 and (two[DS] / "marks" / "broadcast_review" / "calibration" / "v001.json").is_file()
    assert [it["frame"] for it in stored(job)["items"][:2]] == [2000, 2001]  # answered ones did not move


def test_the_real_dataset_verdicts_fit_when_available():
    """Read-only sanity check against the live verdicts (skipped where they do not exist)."""
    root = Path("/mnt/storage/footy_data/broadcast_review_labels/broadcast_review")
    files = sorted(root.glob("*.json")) if root.is_dir() else []
    if not files:
        pytest.skip("no live broadcast-review verdicts here")
    xs, ys = [], []
    for f in files:
        for rec in (json.loads(f.read_text()).get("frames") or {}).values():
            if rec.get("verdict") in ("yes", "no") and isinstance(rec.get("p_yes"), (int, float)):
                xs.append(br.logit(rec["p_yes"]))
                ys.append(1 if rec["verdict"] == "yes" else 0)
    if sum(ys) < 10 or len(ys) - sum(ys) < 10:
        pytest.skip("too few live verdicts")
    a, b = br.fit_platt(xs, ys)
    assert a > 0 and 0 < br.sigmoid(-b / a) < 0.5
