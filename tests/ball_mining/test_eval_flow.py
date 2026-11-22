"""Eval family on the shared anchor-card flow (design ``ball_anchor_iteration.md`` task 10): the explicit teacher-vs-student
candidate-source rule, the fixed eval share, F logged before candidates, seed / no-candidate fallback, the CLI iteration and the
false-negative report. Synthetic candidates, in-process job store, in-memory feature store."""

from __future__ import annotations

import json

import pytest

from footy_track.ball_eval import eval_flow as EF  # noqa: N812
from footy_track.ball_eval import eval_source as ES  # noqa: N812
from footy_track.ball_eval import fn_report as FN  # noqa: N812
from footy_track.ball_eval.panel_score import BALL, NO_BALL, Human
from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.ball_mining import share
from footy_track.feature_store import FeatureStore
from footy_track.labeller import jobs
from footy_track.scripts import ball_anchor_search as cli

from .test_anchor_client import DAY
from .test_anchor_fallback import Api
from .test_anchor_search import HC, TC, register_splits, verdict  # noqa: F401

SIZE = (1920, 1080)
PANEL = {
    "panel_version": "ball-eval-v1", "match": "hm_a",
    "windows": [
        {"window_id": "w0", "clip": HC, "frames": [0, 5, 10], "pi_window": 0.5, "pi_frame": 0.5},
        {"window_id": "w1", "clip": HC, "frames": [100, 105, 110], "pi_window": 0.5, "pi_frame": 0.5},
    ],
}  # fmt: skip
KEY = "ball-eval/ball-eval-v1/hm_a"


def box(cx, cy, conf=0.8):
    return {"x": cx - 0.01, "y": cy - 0.02, "w": 0.02, "h": 0.04, "conf": conf}


def src(kind, sid, version, spec):
    return ES.Source(kind, sid, version, {(HC, f): v for f, v in spec.items()})


GOLD = {(HC, 0): Human(BALL, [(0.5, 0.5)]), (HC, 5): Human(BALL, [(0.6, 0.5)]), (HC, 10): Human(BALL, [(0.7, 0.5)]),
        (HC, 15): Human(BALL, [(0.2, 0.2)]), (HC, 20): Human(NO_BALL, [])}  # fmt: skip


def hits(n):
    """A source whose candidates hit the first ``n`` of the 4 gold ball frames."""
    pos = {0: (0.5, 0.5), 5: (0.6, 0.5), 10: (0.7, 0.5), 15: (0.2, 0.2)}
    return {f: [box(*p)] for f, p in list(pos.items())[:n]}


# ---- 1. candidate source selector --------------------------------------------------------------
def test_recall_is_within_20px_of_human_gold_any_candidate():
    s = src("teacher", "t", "offline_v1", {0: [box(0.5 + 19 / 1920, 0.5)], 5: [box(0.6 + 21 / 1920, 0.5)], 10: [box(0.1, 0.1), box(0.7, 0.5)]})
    sc = ES.score_source(s, ES.gold_ball_frames(GOLD), SIZE)
    assert (sc.n_gold, sc.n_hit) == (4, 2)  # 19 px yes, 21 px no, second candidate counts, no_ball is not gold


def test_teacher_wins_while_it_beats_the_best_student():
    sel, chosen = ES.choose([src("teacher", "t", "offline_v2", hits(3)), src("student", "s", "m1", hits(2))], GOLD, SIZE)
    assert chosen.kind == "teacher" and sel["reason"] == "teacher_recall>=best_student_recall"
    assert sel["chosen"]["version"] == "offline_v2" and not sel["selected_is_scored_model"]


def test_tie_goes_to_the_teacher():
    sel, chosen = ES.choose([src("student", "s", "m1", hits(3)), src("teacher", "t", "offline_v1", hits(3))], GOLD, SIZE)
    assert chosen.kind == "teacher" and sel["reason"] == "tie_goes_to_teacher"


def test_student_wins_when_it_scores_better_and_best_student_is_picked():
    srcs = [src("teacher", "t", "offline_v3", hits(2)), src("student", "s1", "m1", hits(3)), src("student", "s2", "m2", hits(4))]
    sel, chosen = ES.choose(srcs, GOLD, SIZE)
    assert chosen.id == "s2" and sel["reason"] == "best_student_recall>teacher_recall"
    assert [s["recall"] for s in sel["scores"]] == [0.75, 1.0, 0.5]  # every version reported, sorted by kind (student, teacher) / version
    assert sel["runner_up"]["id"] == "t"


def test_best_teacher_among_several_and_newest_wins_a_tie():
    sel, chosen = ES.choose([src("teacher", "a", "offline_v2", hits(3)), src("teacher", "b", "offline_v10", hits(3))], GOLD, SIZE)
    assert chosen.version == "offline_v10"  # numeric, not lexical (v10 > v2)


def test_scored_model_is_not_used_silently():
    # equal students: the one that is NOT being scored wins
    sel, chosen = ES.choose([src("student", "scored", "m1", hits(3)), src("student", "other", "m0", hits(3))], GOLD, SIZE, scored_model="scored")
    assert chosen.id == "other" and not sel["selected_is_scored_model"]
    # only the scored model is available / better: allowed, but flagged on the record and the card block
    sel, chosen = ES.choose([src("student", "scored", "m1", hits(4)), src("student", "other", "m0", hits(1))], GOLD, SIZE, scored_model="scored")
    assert chosen.id == "scored" and sel["selected_is_scored_model"] and "IS the model being scored" in sel["warning"]
    assert ES.candidate_source_block(sel)["selected_is_scored_model"] is True


def test_block_records_version_kind():
    sel, _ = ES.choose([src("teacher", "t", "offline_v2", hits(3)), src("student", "s", "m1", hits(1))], GOLD, SIZE)
    b = ES.candidate_source_block(sel)
    assert b["teacher_version"] == "offline_v2" and "model_version" not in b and b["gold_recall"] == 0.75 and b["n_gold"] == 4
    sel, _ = ES.choose([src("teacher", "t", "offline_v2", hits(1)), src("student", "s", "m9", hits(3))], GOLD, SIZE)
    b = ES.candidate_source_block(sel)
    assert b["model_version"] == "m9" and "teacher_version" not in b


def test_selector_refuses_unlike_or_empty_comparisons():
    with pytest.raises(ES.SourceSelectionError, match="no human gold"):
        ES.choose([src("teacher", "t", "offline_v1", {})], {}, SIZE)
    with pytest.raises(ES.SourceSelectionError, match="different gold"):
        ES.select_source([ES.SourceScore("teacher", "t", "v", 4, 2), ES.SourceScore("student", "s", "v", 5, 2)])
    with pytest.raises(ES.SourceSelectionError):
        ES.select_source([])


def test_suggest_file_roundtrips_into_the_generators_candidate_set(tmp_path):
    sel, chosen = ES.choose([src("teacher", "t", "offline_v2", hits(3)), src("student", "s", "m1", hits(1))], GOLD, SIZE)
    frames = [(HC, f) for f in (0, 5, 10, 15)]
    job = {"id": "j-x", "kind": "ball_check", "dataset": "train", "items": [{"clip": HC, "frame_index": f, "box_index": 0, "source": "list", "bbox": box(0.5, 0.5)} for f in (0, 5, 10, 15)]}
    p = ES.write_suggest_file("j-x", chosen, frames, sel, SIZE, root=tmp_path)
    cs = A.candidates_from_suggest_doc(json.loads(p.read_text()), A.job_scope(job))
    assert cs.detector == "t" and cs.source_info["teacher_version"] == "offline_v2"
    assert (HC, 0) in cs.by_frame and (HC, 15) not in cs.by_frame  # no candidate -> no entry (never a fake box)


# ---- 2. fixed eval share ----------------------------------------------------------------------
def card(clip, f):
    return {"clip": clip, "frame_index": f}


def test_default_share_is_30_percent_and_every_prefix_holds_it():
    assert share.EVAL_SHARE == 0.30
    ev = [card(HC, i) for i in range(100)]
    tr = [card(TC, i) for i in range(100)]
    q = share.merge(ev, tr)
    for k in range(1, 61):
        assert sum(1 for s in q[:k] if s.family == "eval") == int(k * 0.3 + 1e-9)  # exact floor, never starved, never bunched
    assert share.eval_fraction(q, 50) == pytest.approx(0.30)


def test_eval_exhausted_hands_all_slots_to_train_and_train_empty_to_eval():
    q = share.merge([card(HC, 0), card(HC, 1)], [card(TC, i) for i in range(10)])
    assert sum(s.family == "eval" for s in q) == 2 and len(q) == 12 and q[-1].family == "train"
    q = share.merge([card(HC, i) for i in range(5)], [])
    assert [s.family for s in q] == ["eval"] * 5  # no wasted slots


def test_share_never_lifts_a_cap_and_keeps_order_within_a_family():
    ev = [card(HC, i) for i in range(50)]
    tr = [card(TC, i) for i in range(500)]
    q = share.merge(ev, tr, eval_room=5, train_room=20)
    assert len(q) == 25 and sum(s.family == "eval" for s in q) == 5  # each side cut to its own room first
    assert [s.card["frame_index"] for s in q if s.family == "train"] == list(range(20))
    assert len(share.merge(ev, tr, limit=10)) == 10
    assert share.merge(ev, tr, eval_room=0, train_room=0) == []


def test_share_rejects_cards_of_the_wrong_family():
    with pytest.raises(A.SplitError, match="families never mix"):
        share.merge([card(TC, 0)], [card(TC, 1)])
    with pytest.raises(A.SplitError):
        share.merge([card(HC, 0)], [card(HC, 1)])
    with pytest.raises(ValueError):
        share.eval_slot_flags(10, 1.5)


# ---- 3. creation: F first, seeds, no-candidate fallback ------------------------------------------
@pytest.fixture
def store():
    s = FeatureStore.open(":memory:")
    yield s
    s.close()


def cset(spec, source_info=None):
    cs = A.CandidateSet("job_suggest", "t", source_info=source_info)
    for f, cs_ in spec.items():
        cs.by_frame[(HC, f)] = list(cs_)
    return cs


def cd(x, y, conf=0.8, tid=0):
    return A.Cand(x, y, 0.02, 0.04, conf, tid)


INFO = {"kind": "teacher", "id": "t", "version": "offline_v2", "teacher_version": "offline_v2", "rule": ES.RULE}


def test_cards_cannot_be_built_before_F_is_logged(store):
    with pytest.raises(EF.EvalFlowError, match="not in selected_frame"):
        EF.build_job_body(PANEL, cset({0: [cd(0.5, 0.5)]}), store=store)
    assert EF.record_selected(PANEL, store) == 6 and EF.record_selected(PANEL, store) == 6  # idempotent
    rows = store.read_selected_frames(job_key=KEY)
    assert len(rows) == 6 and set(rows.split) == {"eval_heldout_match"} and set(rows.window_id) == {"w0", "w1"}
    EF.build_job_body(PANEL, cset({0: [cd(0.5, 0.5)]}), store=store)


def test_a_train_clip_never_enters_the_eval_F(store):
    bad = {**PANEL, "windows": [{"window_id": "w", "clip": TC, "frames": [0], "pi_window": 1, "pi_frame": 1}]}
    with pytest.raises(A.SplitError, match="eval splits only"):
        EF.record_selected(bad, store)


def test_job_body_has_one_seed_per_window_with_source_versions_and_pi(store):
    EF.record_selected(PANEL, store)
    spec = {0: [cd(0.3, 0.3, 0.5)], 5: [cd(0.5, 0.5, 0.9), cd(0.7, 0.7, 0.4)], 100: [cd(0.4, 0.4, 0.6)]}
    body, plan = EF.build_job_body(PANEL, cset(spec, INFO), store=store)
    assert body["kind"] == "ball_check" and body["key"] == KEY and body["hooks"] == EF.EVAL_HOOKS
    assert [(i["clip"], i["frame_index"]) for i in body["items"]] == [(HC, 5), (HC, 100)]  # best candidate of each window
    for it in body["items"]:
        assert it["sampling"]["regime"] == "eval" and it["sampling"]["bucket"] == "E" and it["sampling"]["pi_frame"] == 0.5
        assert it["sampling"]["split"] == "eval_heldout_match" and it["sampling"]["candidate_source"]["teacher_version"] == "offline_v2"
        assert it["search"]["candidate_source"]["id"] == "t" and it["sampling"]["teacher_version"] == "offline_v2"
        assert it["sampling"]["source"] == "seed" and it["sampling"]["generator_version"].startswith("det:t|")
    assert plan.fallback == []


def test_window_without_any_candidate_goes_to_the_full_frame_check_not_no_ball(store):
    EF.record_selected(PANEL, store)
    body, plan = EF.build_job_body(PANEL, cset({0: [cd(0.3, 0.3)]}, INFO), store=store)
    assert [i["frame_index"] for i in body["items"]] == [0]
    assert plan.fallback == [{"clip": HC, "frame": f, "window_id": "w1", "reason": EF.REASON_NO_CANDIDATE} for f in (100, 105, 110)]
    assert store.read_ball_terminal_states(HC) == {}  # nothing was written: no candidate is not no_ball


def test_rejected_frame_gets_next_candidate_then_fallback():
    groups = {"w0": (HC, [0, 5])}
    cands = cset({0: [cd(0.3, 0.3, 0.9), cd(0.6, 0.6, 0.5)], 5: [cd(0.3, 0.3)]})
    an = A.AnchorSet()
    an.forbidden[(HC, 0)] = [A.Forbidden(HC, 0, 0.3, 0.3, 0.02, 0.04)]
    p = EF.plan_seeds(groups, cands, an, job_id="j")
    assert [(c["frame_index"], round(c["bbox"]["x"] + c["bbox"]["w"] / 2, 3)) for c in p.cards] == [(0, 0.6)]  # next candidate, not the rejected disc
    an.forbidden[(HC, 0)].append(A.Forbidden(HC, 0, 0.6, 0.6, 0.02, 0.04))
    p = EF.plan_seeds(groups, cands, an, job_id="j")
    assert {(f["frame"], f["reason"]) for f in p.fallback} == {(0, EF.REASON_EXHAUSTED)}
    assert [c["frame_index"] for c in p.cards] == [5]  # the window still gets its seed on a fresh frame


def test_plan_is_idempotent_and_leaves_windows_with_anchor_or_live_card_alone():
    groups = {"w0": (HC, [0, 5, 10])}
    cands = cset({0: [cd(0.5, 0.5)], 5: [cd(0.5, 0.5)]})
    assert len(EF.plan_seeds(groups, cands, A.AnchorSet(), job_id="j").cards) == 1
    assert EF.plan_seeds(groups, cands, A.AnchorSet(), job_id="j", pending=[(HC, 0)]).cards == []  # live card: wait for the answer
    an = A.AnchorSet()
    an.anchors[(HC, 10)] = A.Anchor(HC, 10, 0.5, 0.5, "human")
    assert EF.plan_seeds(groups, cands, an, job_id="j").cards == []  # the search takes it from the anchor
    assert EF.plan_seeds(groups, cands, an, job_id="j", terminal=[(HC, 0), (HC, 5), (HC, 10)]).fallback == []


# ---- 4. the shared flow end to end (CLI iteration) ------------------------------------------------
@pytest.fixture
def eval_job(ds, store, monkeypatch):
    EF.record_selected(PANEL, store)
    spec = {f: [cd(0.5 + 0.001 * i, 0.5, 0.8, 7)] for i, f in enumerate((0, 5, 10))} | {100: [cd(0.4, 0.4, 0.6)]}
    cs = cset(spec, INFO)
    monkeypatch.setattr(A, "load_candidates", lambda j, **k: cs)
    body, plan = EF.build_job_body(PANEL, cs, store=store, dataset="train")
    job, created = jobs.submit({k: v for k, v in body.items() if k != "hooks"}, "train")
    assert created and job["id"] == A.job_scope({**body, "id": job["id"]}).job_id
    return job["id"], Api()


def test_eval_iteration_appends_search_cards_with_source_and_stays_in_its_own_split(eval_job, ds, store):
    jid, cl = eval_job
    first = cli.run_job(jid, client=cl, marks_dir=ds, store=store, today=DAY)
    assert first["status"] == "ok" and first["push"]["added"] == 0  # no anchor yet: only the seeds exist
    verdict(ds, HC, 5, "ball", bbox={"x": 0.49, "y": 0.48, "w": 0.02, "h": 0.04})
    out = cli.run_job(jid, client=cl, marks_dir=ds, store=store, today=DAY)
    items = jobs.read_job(jid)["items"]
    seeds = {it["frame_index"]: it for it in items if it["sampling"]["bucket"] == "E"}
    new = [it for it in items if it["frame_index"] == 10]  # frame 0 holds the window's seed card; 100 the other window's
    assert sorted(seeds) == [0, 100] and out["push"]["added"] == 1 and {it["sampling"]["bucket"] for it in new} == {"S"}
    for it in new:
        assert it["search"]["candidate_source"]["teacher_version"] == "offline_v2" and it["sampling"]["teacher_version"] == "offline_v2"
        assert A.clip_split(it["clip"]) == "eval_heldout_match"
    # 105 has no candidate (unreachable) and 110 is beyond the hop cap: both go to the eval full-frame check, none is no_ball
    assert out["fallback"]["family"] == "eval" and out["fallback"]["frames"] == 2
    assert [g["key"] for g in out["fallback"]["groups"]] == ["ball-eval-place/ball-eval-v1/hm_a"]
    assert out["fallback"]["groups"][0]["added"] == 2
    assert store.read_ball_terminal_states(HC) == {}
    assert (ds / "search").is_dir()
    again = cli.run_job(jid, client=cl, marks_dir=ds, store=store, today=DAY)
    assert again["push"]["added"] == 0  # idempotent
    docs = [json.loads(p.read_text()) for p in (ds / "search").rglob("v*.json")]
    assert docs and all(d["teacher_version"] == "offline_v2" and d["candidate_source"]["kind"] == "teacher" for d in docs)


def test_eval_run_uses_its_own_open_cap_not_the_round_cap(eval_job, ds, store):
    jid, cl = eval_job
    verdict(ds, HC, 5, "ball", bbox={"x": 0.49, "y": 0.48, "w": 0.02, "h": 0.04})
    out = cli.run_job(jid, client=cl, marks_dir=ds, store=store, today=DAY, open_max=1)
    assert out["push"]["headroom"]["open_max"] == 1 and out["push"]["headroom"]["room"] == 0 and out["push"]["added"] == 0


def test_sweep_visits_eval_jobs(eval_job, ds, store):
    jid, cl = eval_job
    r = cli.sweep(client=cl, marks_dir=ds, store=store, today=DAY)
    assert [j["job"] for j in r["jobs"]] == [jid] and r["jobs"][0]["status"] == "ok"


# ---- 5. false-negative report ----------------------------------------------------------------------
JOB = {"id": "j1", "key": KEY, "items": [{"clip": HC, "frame_index": 0, "search": {"candidate_source": INFO}}]}
F6 = [(HC, f) for f in (0, 5, 10, 15, 20, 25)]
STATES = {(HC, 0): "ball", (HC, 5): "ball", (HC, 10): "ball", (HC, 15): "no_ball", (HC, 20): "unusable", (HC, 25): "deferred"}
CENT = {(HC, 0): (0.5, 0.5), (HC, 5): (0.6, 0.5), (HC, 10): (0.7, 0.5)}


def px(x, y):
    return (x * 1920, y * 1080)


def test_fn_report_counts_misses_per_version_and_terminal_states():
    models = {
        "v1": {(HC, 0): [px(0.5, 0.5)], (HC, 5): [px(0.9, 0.9)], (HC, 15): [px(0.1, 0.1)]},  # 5: off, 10: nothing; the no_ball frame is irrelevant
        "v2": {(HC, 0): [px(0.5, 0.5)], (HC, 5): [px(0.6, 0.5)], (HC, 10): [(px(0.7, 0.5)[0] + 19, px(0.7, 0.5)[1])]},
    }
    r = FN.build_report(JOB, selected=F6, states=STATES, centres=CENT, models=models, window_of={(HC, 5): "w0"})
    assert r["schema"] == "ball-eval-fn-report/v1" and r["complete"] and r["n_selected"] == 6 and r["n_ball_frames"] == 3
    assert r["by_state"] == {"ball": 3, "no_ball": 1, "unusable": 1, "deferred": 1, "outstanding": 0}
    v1, v2 = r["versions"]["v1"], r["versions"]["v2"]
    assert (v1["n_missed"], v1["missed_no_detection"], v1["missed_off"]) == (2, 1, 1) and v1["recall"] == pytest.approx(1 / 3)
    assert [m["frame"] for m in v1["missed_frames"]] == [5, 10] and v1["missed_frames"][0]["window_id"] == "w0"
    assert v2["n_missed"] == 0 and v2["recall"] == 1.0  # 19 px is inside 20
    assert r["candidate_sources"] == [INFO]
    json.dumps(r)


def test_fn_report_is_incomplete_when_frames_are_outstanding_and_never_counts_them():
    st = {k: v for k, v in STATES.items() if k != (HC, 25)}
    r = FN.build_report(JOB, selected=F6, states=st, centres=CENT, models={"v1": {}})
    assert not r["complete"] and r["by_state"]["outstanding"] == 1 and r["outstanding"] == [{"clip": HC, "frame": 25}]
    assert r["versions"]["v1"]["n_missed"] == 3  # the three ball frames, not the outstanding one


def test_fn_report_ball_frame_without_a_centre_is_not_a_hit_or_miss():
    r = FN.build_report(JOB, selected=F6, states=STATES, centres={(HC, 0): (0.5, 0.5)}, models={"v1": {(HC, 0): [px(0.5, 0.5)]}})
    assert r["ball_without_centre"] == 2 and r["versions"]["v1"]["n_ball_frames"] == 1 and r["versions"]["v1"]["n_missed"] == 0


def test_fn_report_from_the_store_end_to_end(ds, store, tmp_path):
    EF.record_selected(PANEL, store)
    jid = "j-eval"
    job = {"id": jid, "key": KEY, "kind": "ball_check", "dataset": "train",
           "items": [{"clip": HC, "frame_index": f, "box_index": 0, "source": "list", "bbox": box(0.5, 0.5)} for f in (0, 5, 10, 100, 105, 110)]}  # fmt: skip
    verdict(ds, HC, 0, "ball", bbox=box(0.5, 0.5))
    for f in (0, 5):
        store.write_ball_terminal_state(HC, f, "ball", method="verdict")
    store.write_ball_terminal_state(HC, 10, "no_ball", method="hand_frame", full_frame_check=True)
    p = tmp_path / "v1.jsonl"
    p.write_text(json.dumps({"clip": HC, "frame": 0, "x": 0.5 * 1920, "y": 0.5 * 1080}) + "\n")
    rep = FN.report_for_job(job, store, {"v1": FN.load_detections(p)}, marks_dir=ds)
    assert not rep["complete"] and rep["by_state"]["ball"] == 2 and rep["by_state"]["no_ball"] == 1 and rep["by_state"]["outstanding"] == 3
    assert rep["ball_without_centre"] == 1 and rep["versions"]["v1"]["n_ball_frames"] == 1 and rep["versions"]["v1"]["n_missed"] == 0
    assert rep["found_by"] == {"card": 2}
