"""The ``ball_mining_refit`` hook: hit rates, Platt on R+A only, LTT, versioned state, reorder of unanswered U / M cards."""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track.ball_mining import refit
from footy_track.ball_mining.round import ROUND_HOOKS
from footy_track.labeller import datasets, hooks, jobs

CLIP = "m1_seg001"


@pytest.fixture(autouse=True)
def _hooks_clean():
    hooks._reset_for_tests()
    yield
    hooks.drain()
    hooks._reset_for_tests()


def card(
    frame, bucket, p, *, pi=None, band=None, scorer="raw_conf", match="m1", clip=CLIP
):
    return {"clip": clip, "frame_index": frame, "bbox": {"x": 0.4, "y": 0.4, "w": 0.01, "h": 0.02}, "match": match, "p_real": p,
            "scorer_version": scorer, "sources": ["rtdetr"],
            "sampling": {"regime": "train", "bucket": bucket, "pi": pi, "band": band, "match": match, "scorer_version": scorer}}  # fmt: skip


def submit(items, hooks_=None, key="ball-round/2026-11-18/1"):
    body = {
        "title": "round",
        "kind": "ball_check",
        "dataset": "train",
        "items": items,
        "key": key,
        "max_cards": 2000,
    }
    if hooks_:
        body["hooks"] = hooks_
    job, created = jobs.submit(body, "train")
    assert created
    return job


def answer(marks, job_id, pos, verdict):
    job = jobs.read_job(job_id)
    it = job["items"][pos]
    d = marks / "ball_checks"
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{it['clip']}.jsonl").open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "clip": it["clip"],
                    "frame_index": it["frame_index"],
                    "box_index": it["box_index"],
                    "source": "list",
                    "verdict": verdict,
                }
            )
            + "\n"
        )


def decide(p, boundary=0.8):
    return "ball" if p > boundary else "not_ball"


def round_items():
    """60 answered R cards and 600 A cards (verdict ball iff p_real > 0.8), 50 answered U cards that are all wrong, then an
    unanswered tail: U cards at p 0.5 .. 0.95, one R, one A, one G, one M."""
    rng = np.random.default_rng(0)
    items, truth = [], {}
    f = 0
    for p in rng.uniform(0.3, 1.0, 60):
        items.append(card(f, "R", float(p), pi=0.1))
        truth[f] = decide(p)
        f += 100
    for p in np.linspace(0.5, 0.9999, 600):
        items.append(card(f, "A", float(p), pi=0.05, band="x"))
        truth[f] = decide(p)
        f += 100
    for _ in range(50):
        items.append(card(f, "U", 0.95))
        truth[f] = "not_ball"  # a perverse U result: must not leak into the calibration
        f += 100
    return items, truth, f


def tail_items(f):
    t = [card(f, "U", 0.5), card(f + 100, "R", 0.2, pi=0.1), card(f + 200, "U", 0.65), card(f + 300, "A", 0.9, pi=0.05),
         card(f + 400, "U", 0.8), card(f + 500, "G", None), card(f + 600, "U", 0.95), card(f + 700, "M", 0.4), card(f + 800, "M", 0.9), card(f + 900, "U", 0.3)]  # fmt: skip
    t[5]["p_real"] = None
    return t


@pytest.fixture
def round_job(ds):
    items, truth, f = round_items()
    tail = tail_items(f)
    job = submit(items + tail, ROUND_HOOKS)
    for pos, it in enumerate(items):
        answer(ds, job["id"], pos, truth[it["frame_index"]])
    return job


def test_hit_rates_are_per_bucket_and_unsure_is_not_counted_as_a_decision():
    rows = [
        {"bucket": "U", "verdict": v}
        for v in ["ball", "not_ball", "not_ball", "unsure"]
    ] + [{"bucket": "A", "verdict": "corrected"}]
    assert refit.hit_rates(rows) == {
        "A": {"n": 1, "correct": 1, "wrong": 0, "unsure": 0, "hit_rate": 1.0},
        "U": {"n": 4, "correct": 1, "wrong": 2, "unsure": 1, "hit_rate": 0.3333},
    }


def test_calibration_uses_only_random_and_audit_verdicts(round_job):
    state, cal = refit.refit(jobs.read_job(round_job["id"]))
    assert (
        cal["fitted"] and cal["buckets_used"] == ["R", "A"] and cal["n"] == 660
    )  # the 50 U verdicts are not in the fit
    assert abs(cal["boundary_p_real"] - 0.8) < 0.04
    assert (
        state["hit_rates"]["U"]["hit_rate"] == 0.0
        and state["hit_rates"]["A"]["n"] == 600
    )  # reported, not fitted
    assert state["scorer_version"] == "raw_conf" and state["n_verdicts"] == 710


def test_threshold_comes_from_the_audit_verdicts_alone(round_job):
    state, _ = refit.refit(jobs.read_job(round_job["id"]))
    g = state["threshold"]["global"]
    assert (
        g["status"] == "valid" and 0.80 <= g["threshold"] <= 0.85
    )  # audits above 0.8 are all right, those just below are wrong
    assert state["auto_accept"] == {
        "threshold": g["threshold"],
        "scale": "p_real",
        "fallback": False,
    }
    assert state["threshold"]["per_match"]["m1"]["threshold"] == g["threshold"]


def test_with_too_few_audits_the_fallback_is_recorded_as_such(ds):
    job = submit([card(i * 100, "A", 0.99, pi=0.1) for i in range(10)])
    for i in range(10):
        answer(ds, job["id"], i, "ball")
    state, cal = refit.refit(jobs.read_job(job["id"]))
    assert state["auto_accept"] == {
        "threshold": 0.95,
        "scale": "claude_verify_conf",
        "fallback": True,
    }
    assert (
        state["threshold"]["global"]["status"] == "no_valid_threshold"
        and cal["fitted"] is False
        and "need >= 10" in cal["noop"]
    )


def test_state_files_are_versioned_immutable_and_reused_for_the_same_verdicts(
    ds, round_job
):
    job = jobs.read_job(round_job["id"])
    s1, _ = refit.refit(job)
    s2, _ = refit.refit(job)
    assert (s1["version"], s1["reused"], s2["version"], s2["reused"]) == (
        1,
        False,
        1,
        True,
    )
    assert sorted(p.name for p in (ds / "ball_mining" / "state").iterdir()) == [
        "v001.json"
    ]
    answer(
        ds, job["id"], len(job["items"]) - 1, "ball"
    )  # one more verdict -> a new version
    s3, _ = refit.refit(jobs.read_job(job["id"]))
    assert s3["version"] == 2 and sorted(
        p.name for p in (ds / "ball_mining" / "state").iterdir()
    ) == ["v001.json", "v002.json"]
    assert refit.latest_state(ds)["version"] == 2


def test_verdicts_of_another_scorer_version_do_not_feed_the_fit(ds):
    items = [card(i * 100, "A", 0.99, pi=0.1, scorer="old") for i in range(150)] + [
        card(20000 + i * 100, "A", 0.99, pi=0.1, scorer="new") for i in range(5)
    ]
    job = submit(items)
    for i in range(155):
        answer(ds, job["id"], i, "ball")
    st, _ = refit.refit(jobs.read_job(job["id"]))  # the job's items are mostly "old"
    assert (
        st["scorer_version"] == "old"
        and st["n_rows_other_scorer"] == 5
        and st["threshold"]["global"]["status"] == "valid"
    )


def test_gold_accepted_too_often_flags_the_round(ds):
    job = submit([card(i * 100, "G", None) for i in range(8)])
    for i, v in enumerate(
        [
            "ball",
            "box_off",
            "corrected",
            "not_ball",
            "ball",
            "ball",
            "unsure",
            "box_off",
        ]
    ):
        answer(ds, job["id"], i, v)
    state, _ = refit.refit(jobs.read_job(job["id"]))
    assert state["gold"] == {
        "n": 7,
        "accepted_wrong": 4,
        "accept_rate": 0.5714,
        "flagged": True,
    }


def test_priority_order_redeals_u_and_m_cards_inside_their_own_slots():
    cal = {
        "fitted": True,
        "a": 4.0,
        "b": -4.0 * float(np.log(0.8 / 0.2)),
    }  # calibrated p = 0.5 at p_real 0.8
    un = [(10, card(0, "U", 0.5)), (11, card(1, "R", 0.2)), (12, card(2, "U", 0.65)), (13, card(3, "M", 0.4)), (14, card(4, "U", 0.8)),
          (15, card(5, "A", 0.9)), (16, card(6, "M", 0.9)), (17, card(7, "U", 0.95))]  # fmt: skip
    order = refit.priority_order(un, cal)
    assert sorted(order) == sorted(p for p, _ in un)  # a permutation
    kept = {11, 15}
    assert [order[i] for i, (p, _) in enumerate(un) if p in kept] == [
        11,
        15,
    ]  # R / A cards stay in their slots
    u_slots = [i for i, (_, it) in enumerate(un) if it["sampling"]["bucket"] == "U"]
    assert [order[i] for i in u_slots] == [
        14,
        12,
        10,
        17,
    ]  # nearest the fitted boundary first: .80, .65, .50, .95
    m_slots = [i for i, (_, it) in enumerate(un) if it["sampling"]["bucket"] == "M"]
    assert [order[i] for i in m_slots] == [16, 13]  # the more plausible gap first
    assert (
        refit.priority_order(un, None)[u_slots[0]] == 10
    )  # no calibration: raw |p - 0.5|, so the p=0.5 card leads


def test_the_hook_refits_stores_state_and_reorders_only_unanswered_u_and_m_cards(
    ds, round_job
):
    jid = round_job["id"]
    before = jobs.read_job(jid)
    res = hooks.run_now(jid, "refit")
    entry = res["history"][-1]
    assert entry["ok"] is True and entry["action"] == "ball_mining_refit", entry
    assert (
        entry["result"]["version"] == 1
        and entry["result"]["calibrated"] is True
        and entry["result"]["changed"] is True
    )
    after = jobs.read_job(jid)
    n_answered = 710
    assert (
        after["items"][:n_answered] == before["items"][:n_answered]
    )  # answered cards never move
    tb, ta = before["items"][n_answered:], after["items"][n_answered:]
    for i, (b, a) in enumerate(zip(tb, ta, strict=True)):
        if b["sampling"]["bucket"] in ("R", "A", "G"):
            assert a == b, (
                i
            )  # audits, random and gold keep their slots, so they stay interleaved
    u_after = [a["p_real"] for a in ta if a["sampling"]["bucket"] == "U"]
    assert u_after[0] == 0.8 and u_after[-1] in (
        0.3,
        0.5,
    )  # near the fitted boundary first
    assert sorted(u_after) == sorted(
        b["p_real"] for b in tb if b["sampling"]["bucket"] == "U"
    )
    assert after["rev"] == before.get("rev", 0) + 1
    assert (ds / "ball_mining" / "state" / "v001.json").is_file()


def test_the_close_hook_writes_a_final_state_and_does_not_reorder(ds, round_job):
    job = jobs.read_job(round_job["id"])
    hook = next(h for h in job["hooks"] if h["name"] == "close")
    assert hook["trigger"]["on_done"] is True and hook["action"]["params"] == {
        "close": True
    }
    ctx = hooks.HookContext(job, hook, "on_done", None, {"close": True})
    with datasets.using("train"):
        out = refit._action(ctx)
    assert "reordered" not in out and refit.latest_state(ds)["closed"] is True
    assert jobs.read_job(job["id"]).get("rev", 0) == job.get("rev", 0)


def test_hook_config_is_valid_for_ball_check_jobs_only():
    assert [h["name"] for h in hooks.validate_hooks("ball_check", ROUND_HOOKS)] == [
        "refit",
        "close",
    ]
    assert (
        hooks.validate_hooks("ball_check", ROUND_HOOKS)[0]["trigger"]["every_n"] == 50
    )
    with pytest.raises(jobs.JobError, match="ball_check"):
        hooks.validate_hooks("calibration", ROUND_HOOKS)
