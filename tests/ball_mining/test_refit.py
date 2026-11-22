"""Refit of a round (library + ``ball_round refit``, outside the labeller server): hit rates, Platt on R+A only, LTT, versioned
state, reorder of unanswered U / M cards through the API (with ``if_rev``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track.ball_mining import refit
from footy_track.ball_mining.round import ROUND_HOOKS
from footy_track.labeller import hooks, job_edit, jobs, jobs_client, prefect_client
from footy_track.scripts import ball_round

CLIP = "m1_seg001"


def run_refit(api, job_id, **kw):
    """``(summary, latest stored state)`` of a refit through the API double."""
    out = refit.refit_round(job_id, client=api, **kw)
    return out, refit.latest_state(refit.resolve_marks("train", None))


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


def test_calibration_uses_only_random_and_audit_verdicts(api, round_job):
    _, state = run_refit(api, round_job["id"], reorder=False)
    cal = state["calibration"]
    assert (
        cal["fitted"] and cal["buckets_used"] == ["R", "A"] and cal["n"] == 660
    )  # the 50 U verdicts are not in the fit
    assert abs(cal["boundary_p_real"] - 0.8) < 0.04
    assert (
        state["hit_rates"]["U"]["hit_rate"] == 0.0
        and state["hit_rates"]["A"]["n"] == 600
    )  # reported, not fitted
    assert state["scorer_version"] == "raw_conf" and state["n_verdicts"] == 710


def test_threshold_comes_from_the_audit_verdicts_alone(api, round_job):
    _, state = run_refit(api, round_job["id"], reorder=False)
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


def test_with_too_few_audits_the_fallback_is_recorded_as_such(api, ds):
    job = submit([card(i * 100, "A", 0.99, pi=0.1) for i in range(10)])
    for i in range(10):
        answer(ds, job["id"], i, "ball")
    _, state = run_refit(api, job["id"], reorder=False)
    assert state["auto_accept"] == {
        "threshold": 0.95,
        "scale": "claude_verify_conf",
        "fallback": True,
    }
    assert (
        state["threshold"]["global"]["status"] == "no_valid_threshold"
        and state["calibration"]["fitted"] is False
        and "need >= 10" in state["calibration"]["noop"]
    )


def test_state_files_are_versioned_immutable_and_reused_for_the_same_verdicts(
    api, ds, round_job
):
    jid = round_job["id"]
    s1, _ = run_refit(api, jid, reorder=False)
    s2, _ = run_refit(api, jid, reorder=False)
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
        ds, jid, len(jobs.read_job(jid)["items"]) - 1, "ball"
    )  # one more verdict -> a new version
    s3, latest = run_refit(api, jid, reorder=False)
    assert s3["version"] == 2 and sorted(
        p.name for p in (ds / "ball_mining" / "state").iterdir()
    ) == ["v001.json", "v002.json"]
    assert latest["version"] == 2


def test_verdicts_of_another_scorer_version_do_not_feed_the_fit(api, ds):
    items = [card(i * 100, "A", 0.99, pi=0.1, scorer="old") for i in range(150)] + [
        card(20000 + i * 100, "A", 0.99, pi=0.1, scorer="new") for i in range(5)
    ]
    job = submit(items)
    for i in range(155):
        answer(ds, job["id"], i, "ball")
    _, st = run_refit(api, job["id"], reorder=False)  # the job's items are mostly "old"
    assert (
        st["scorer_version"] == "old"
        and st["n_rows_other_scorer"] == 5
        and st["threshold"]["global"]["status"] == "valid"
    )


def test_gold_accepted_too_often_flags_the_round(api, ds):
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
    _, state = run_refit(api, job["id"], reorder=False)
    assert state["gold"] == {
        "n": 7,
        "accepted_wrong": 3,  # the three plain "ball"s; "corrected" fixed the planted box, which is right
        "accept_rate": 0.4286,
        "flagged": True,
    }


def test_verdicts_of_all_rounds_of_the_dataset_feed_the_fit_but_not_other_jobs(
    api, ds, round_job
):
    other = submit(
        [card(900000 + i * 100, "A", 0.99, pi=0.1) for i in range(5)],
        key="ball-round/2026-11-18/2",
    )
    for i in range(5):
        answer(ds, other["id"], i, "ball")
    plain = submit(
        [card(990000, "A", 0.99, pi=0.1)], key="someone-else/x"
    )  # not a round: never read
    answer(ds, plain["id"], 0, "not_ball")
    _, st = run_refit(api, round_job["id"], reorder=False)
    assert st["n_verdicts"] == 715 and st["hit_rates"]["A"]["n"] == 605
    assert not any(c == ("results", plain["id"]) for c in api.calls)


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


def test_refit_stores_state_and_reorders_only_unanswered_u_and_m_cards_through_the_api(
    api, ds, round_job
):
    jid = round_job["id"]
    before = jobs.read_job(jid)
    out, _ = run_refit(api, jid)
    assert (
        out["version"] == 1
        and out["calibrated"] is True
        and out["changed"] is True
        and out["attempts"] == 1
    ), out
    reorder_call = next(c for c in api.calls if c[0] == "reorder")
    assert reorder_call[3] == "ball_round_refit" and reorder_call[5] == (
        before.get("rev") or 0
    )  # if_rev from the same read
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


def test_a_rev_conflict_rereads_the_job_and_retries_with_the_fresh_rev(
    api, ds, round_job
):
    jid = round_job["id"]
    fired = []

    def racing_change(ref, order):  # someone reorders between our read and our write
        if not fired:
            fired.append(1)
            n = [i for i, _ in refit.unanswered_from_results(api.results(ref))]
            job_edit.reorder_unanswered(jid, n[::-1], by="someone")

    api.before_reorder = racing_change
    out, _ = run_refit(api, jid)
    assert out["attempts"] == 2 and "reordered" in out
    reorders = [c for c in api.calls if c[0] == "reorder"]
    assert (
        len(reorders) == 2 and reorders[1][5] == reorders[0][5] + 1
    )  # second try carries the new rev
    assert jobs.read_job(jid)["rev"] == 2


def test_a_persistent_rev_conflict_gives_up_after_the_retries_without_failing(
    api, ds, round_job
):
    jid = round_job["id"]

    def always(ref, order):
        n = [i for i, _ in refit.unanswered_from_results(api.results(ref))]
        job_edit.reorder_unanswered(jid, n[::-1], by="busy")

    api.before_reorder = always
    out, st = run_refit(api, jid)
    assert out["reorder"] == "rev_conflict" and out["attempts"] == refit.REORDER_TRIES
    assert st["version"] == 1  # the state was stored anyway


def test_other_api_errors_fail_closed_and_a_finished_job_is_not_reordered(
    api, ds, round_job
):
    jid = round_job["id"]

    def boom(*a, **k):
        raise jobs_client.JobsError(500, "labeller down")

    api.reorder = boom
    with pytest.raises(jobs_client.JobsError):
        run_refit(api, jid)
    assert (
        refit.latest_state(ds)["version"] == 1
    )  # the state was written before the failed reorder
    jobs._set_status(jid, "done", only_from=jobs.LIVE, by="test")
    api.reorder = lambda *a, **k: pytest.fail("a finished job must not be reordered")
    out, _ = run_refit(api, jid)
    assert out["reorder"] == "skipped, job is done"


def test_no_ball_mining_action_is_registered_in_the_labeller():
    hooks._load_builtins()
    assert (
        not any("ball_mining" in n for n in hooks.ACTIONS)
        and "broadcast_calibrate" in hooks.ACTIONS
    )


def test_round_hooks_are_prefect_actions_for_the_two_deployments():
    cleaned = hooks.validate_hooks("ball_check", ROUND_HOOKS)
    assert [(h["name"], h["trigger"].get("every_n"), h["trigger"]["on_done"], h["action"]["type"], h["action"]["deployment"]) for h in cleaned] == [
        ("refit", 50, False, "prefect", "ball-refit/ball-refit"),
        ("anchor-search", 10, False, "prefect", "ball-anchor-search/ball-anchor-search"),
        ("close", None, True, "prefect", "ball-round-close/ball-round-close"),
    ]  # fmt: skip
    assert all(
        h["trigger"]["min_interval_s"] >= hooks.MIN_PREFECT_INTERVAL_S for h in cleaned
    )
    assert [h["action"]["params"] for h in cleaned] == [
        {},
        {},
        {},
    ]  # the job ref arrives as job_id, added by the labeller


def test_the_prefect_run_the_hook_queues_carries_the_job_id(monkeypatch, round_job):
    """What ``hooks._queue_prefect`` sends: ``job_id`` is the flow's job reference (there is no ``job_ref``)."""
    sent = {}
    monkeypatch.setattr(
        prefect_client,
        "create_flow_run",
        lambda params, deployment, name=None: sent.update(
            params=params, deployment=deployment
        )
        or "run-1",
    )
    monkeypatch.setenv("FOOTY_LABELLER_URL", "http://127.0.0.1:8011")
    job = jobs.read_job(round_job["id"])
    hook = next(h for h in job["hooks"] if h["name"] == "refit")
    assert hooks._queue_prefect(job, hook, "every_n") == "run-1"
    assert sent["deployment"] == "ball-refit/ball-refit"
    assert sent["params"] == {
        "job_id": job["id"],
        "hook": "refit",
        "trigger": "every_n",
        "dataset": "train",
        "labeller_url": "http://127.0.0.1:8011",
    }


def test_cli_refit_prints_one_json_line_and_exits_2_on_an_api_error(
    monkeypatch, api, ds, round_job, capsys
):
    monkeypatch.setattr(refit.jobs_client, "results", api.results)
    monkeypatch.setattr(refit.jobs_client, "list_jobs", api.list_jobs)
    monkeypatch.setattr(refit.jobs_client, "reorder", api.reorder)
    assert (
        ball_round.main(
            ["refit", "--job", round_job["id"], "--labeller-url", "http://x:1"]
        )
        == 0
    )
    doc = json.loads(capsys.readouterr().out)
    assert (
        doc["status"] == "ok"
        and doc["version"] == 1
        and doc["changed"] is True
        and doc["job"] == round_job["id"]
    )

    def down(*a, **k):
        raise jobs_client.JobsError(503, "labeller down")

    monkeypatch.setattr(refit.jobs_client, "results", down)
    assert ball_round.main(["refit", "--job", round_job["id"]]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_gold_check_counts_only_a_plain_ball_as_accepting_the_planted_error():
    from footy_track.ball_mining.refit import gold_check

    rows = [{"bucket": "G", "verdict": v} for v in ("corrected", "corrected", "box_off", "box_off", "box_off")]
    out = gold_check(rows)
    assert out["accepted_wrong"] == 0 and not out["flagged"]  # fixing or rejecting a shifted gold box is the right answer
    out = gold_check(rows + [{"bucket": "G", "verdict": "ball"}])
    assert out["accepted_wrong"] == 1 and out["flagged"]  # 1/6 > 10 %


# ---- anchor-search (bucket S) cards and the per-job mutex (design task 5, 7.4) -----------------------


def _s_card(frame, p=0.95):
    return card(frame, "S", p, scorer=None)


def test_exclude_search_drops_only_s_rows():
    rows = [{"bucket": b} for b in ("R", "A", "S", "U", "S", "G")]
    assert [r["bucket"] for r in refit.exclude_search(rows)] == ["R", "A", "U", "G"]


def test_s_verdicts_never_reach_learn_then_test_or_the_platt_fit(api, ds, monkeypatch):
    items, truth, f = round_items()
    s_items = [_s_card(f + 100 * i, 0.95) for i in range(40)]  # S cards: all rejected, p_real high, scorer_version None
    job = submit(items + s_items, ROUND_HOOKS)
    for pos, it in enumerate(items):
        answer(ds, job["id"], pos, truth[it["frame_index"]])
    _, without_s = run_refit(api, job["id"], reorder=False)
    for pos in range(len(items), len(items) + 40):
        answer(ds, job["id"], pos, "not_ball")
    seen = []
    for name in ("learn_then_test", "learn_then_test_per_match", "fit_platt"):
        real = getattr(refit, name)

        def spy(*a, _real=real, _name=name, **k):
            rows = a[0]
            seen.append((_name, [r["bucket"] for r in rows] if _name != "fit_platt" else None, len(rows)))
            return _real(*a, **k)

        monkeypatch.setattr(refit, name, spy)
    _, with_s = run_refit(api, job["id"], reorder=False)
    assert with_s["version"] > without_s["version"]
    for name, buckets, _ in seen:
        if buckets is not None:
            assert "S" not in buckets, name
    assert any(n == "learn_then_test" for n, *_ in seen)
    assert with_s["calibration"] == without_s["calibration"]  # same n, a, b: S did not enter the fit
    assert with_s["threshold"] == without_s["threshold"]
    assert with_s["n_search_excluded"] == 40
    assert with_s["hit_rates"]["S"] == {"n": 40, "correct": 0, "wrong": 40, "unsure": 0, "hit_rate": 0.0}  # reported as its own bucket


def test_job_lock_second_holder_is_refused_then_released(tmp_path):
    with refit.job_lock(tmp_path, "j1"):
        with pytest.raises(refit.JobBusy), refit.job_lock(tmp_path, "j1"):
            pass
        with refit.job_lock(tmp_path, "j2"):  # another job is independent
            pass
    with refit.job_lock(tmp_path, "j1"):
        pass


def test_refit_skips_cleanly_while_the_job_is_locked(api, round_job, ds):
    with refit.job_lock(ds, round_job["id"]):
        out = refit.refit_round(round_job["id"], client=api)
    assert out["status"] == "skipped_busy"
    assert refit.latest_state(ds) is None
    assert not [c for c in api.calls if c[0] == "reorder"]
