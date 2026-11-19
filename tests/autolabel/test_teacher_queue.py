"""Teacher versioning, gold scoring (held-out only, per match) and the targeting queue."""

from __future__ import annotations

import pytest

from footy_track import splits
from footy_track.autolabel import queue as q
from footy_track.autolabel import teacher as t
from footy_track.autolabel import teacher_eval as te
from footy_track.autolabel.preverify import PreverifyResult

HELD, TRAIN = splits.EVAL_HELDOUT, splits.TRAIN


# ---- teacher version ----
def cfg(**kw):
    base = {"detector_weights_id": "abc", "verifier_model": "opus", "thresholds": {"a": 1.0, "b": 2.0}}
    return t.TeacherConfig(**{**base, **kw})


def test_hash_stable_and_sensitive():
    assert cfg().content_hash() == cfg(thresholds={"b": 2.0, "a": 1.0}).content_hash()
    assert cfg().content_hash() == t.TeacherConfig.from_json(cfg().to_json()).content_hash()
    for change in ({"detector_weights_id": "x"}, {"sam3": False}, {"verifier_prompt_version": "v3"},
                   {"thresholds": {"a": 1.1, "b": 2.0}}):
        assert cfg(**change).content_hash() != cfg().content_hash()


def test_registry_append_only(tmp_path):
    p = tmp_path / "teachers.json"
    assert t.register(cfg(), p) == "offline_v1"
    assert t.register(cfg(sam3=False), p) == "offline_v2"
    assert t.register(cfg(), p) == "offline_v1"
    assert [e["label"] for e in t.load_registry(p)] == ["offline_v1", "offline_v2"]


def test_preverify_json_carries_teacher_version():
    r = PreverifyResult("g_f000000", [], [], [], {}, {}, "offline_v3")
    assert r.to_json()["teacher_version"] == "offline_v3"
    assert PreverifyResult("g_f000000", [], [], [], {}).to_json()["teacher_version"] is None


def test_audit_pass_rates():
    docs = [{"teacher_version": "offline_v1", "audit": {"n": 10, "failed": 0}},
            {"teacher_version": "offline_v1", "audit_failed": [3]},
            {"audit": {"n": 5, "failed": 0}}]
    assert t.audit_pass_rates(docs) == {"offline_v1": {"n": 11, "passed": 10, "pass_rate": 10 / 11}}


# ---- gold scoring ----
def gf(match, fi, **kw):
    return te.GoldFrame(match, f"{match}_seg001", fi, **kw)


def split_held(mk, gid):
    return HELD


def test_ball_metrics_per_match_not_pooled():
    a = [gf("A", i, ball_method="hand_series", gold_ball=(100, 100), pred_ball=(105, 100)) for i in range(4)]
    b = [gf("B", 0, ball_method="hand_series", gold_ball=(100, 100), pred_ball=(120, 100)),
         gf("B", 1, ball_method="verdict", gold_ball=None, pred_ball=(5, 5)),
         gf("B", 2, ball_method="hand_frame", gold_ball=(10, 10), pred_ball=None)]
    s = te.score_teacher("offline_v1", a + b, split_held)
    assert s.per_match["A"]["ball_recall_10px"] == 1.0
    assert s.per_match["B"]["ball_recall_10px"] == 0.0
    assert s.per_match["B"]["ball_recall_25px"] == 0.5
    assert s.per_match["B"]["ball_precision_25px"] == 0.5  # 1 TP of 2 predictions
    assert s.headline["ball_recall_10px"] == 0.5  # mean of 1.0 and 0.0; pooled would be 4/6
    assert s.per_match["A"]["ball_precision_10px"] == 1.0


def test_person_buckets():
    gold = [(0, 0, 10, 20), (100, 0, 10, 20), (105, 0, 10, 20)]  # last two overlap
    pred = [(0, 0, 10, 20), (100, 0, 10, 20), (300, 0, 10, 20)]
    f = gf("A", 0, player_method="hand_series", gold_players=gold, pred_players=pred)
    m = te.score_match([f])
    assert m["person_recall_isolated"] == 1.0
    assert m["person_recall_overlapping"] == 0.5
    assert m["person_recall_all"] == 2 / 3
    assert m["person_precision_all"] == 2 / 3


def test_refuses_train_split_and_non_gold():
    f = gf("A", 0, ball_method="hand_series", gold_ball=(1, 1))
    with pytest.raises(te.ScoringRefused):
        te.score_teacher("v", [f], lambda mk, g: TRAIN)
    with pytest.raises(te.ScoringRefused):
        te.score_teacher("v", [gf("A", 0, ball_method="offline_audited")], split_held)


def test_result_rows():
    f = gf("A", 0, ball_method="hand_series", gold_ball=(1, 1), pred_ball=(1, 2))
    rows = te.to_result_rows(te.score_teacher("offline_v2", [f], split_held))
    assert {r.model_version for r in rows} == {"offline_v2"}
    assert {r.split for r in rows} == {HELD}
    metrics = {r.metric for r in rows}
    assert "ball_recall_10px" in metrics and "ball_recall_10px/A" in metrics


# ---- queue ----
SPLITS = {"tr": TRAIN, "held": HELD, "ev": splits.EVAL_IN_MATCH}


def run(cands, purpose, **kw):
    return q.build_queue(cands, purpose, kw.pop("top", 10), split_of=lambda g: SPLITS[g], **kw)


def test_ranking_and_purposes():
    cands = [
        q.Candidate("tr", 1, priority=0.4, flagged=True),
        q.Candidate("tr", 2, priority=0.4, flagged=True, audit_failure=True),
        q.Candidate("tr", 3, priority=0.4, flagged=True, small_ball=True),
        q.Candidate("tr", 4, priority=0.1),
        q.Candidate("tr", 5, priority=0.0, audit_frame=True),
        q.Candidate("tr", 6, priority=0.4, flagged=True,
                    boxes=[(0, 0, 10, 20), (5, 0, 10, 20), (10, 0, 10, 20), (15, 0, 10, 20)]),
    ]
    items, rej = run(cands, "improve_teacher")
    assert [i.frame_index for i in items] == [2, 3, 6, 1]
    assert "audit_failure" in items[0].reasons and "crowded" in items[2].reasons
    assert rej == {}
    assert [i.frame_index for i in run(cands, "student_data")[0]] == [4]
    assert [i.frame_index for i in run(cands, "measure_teacher")[0]] == [5]
    assert len(run(cands, "improve_teacher", top=2)[0]) == 2


def test_rejections():
    cands = [q.Candidate(g, f, priority=0.9, flagged=True)
             for g, f in [("tr", 1), ("tr", 2), ("tr", 3), ("held", 1), ("ev", 1)]]
    items, rej = run(cands, "improve_teacher", gold_frames=lambda g: {2}, complete_frames=lambda g: {3})
    assert [(i.game_id, i.frame_index) for i in items] == [("tr", 1)]
    assert rej == {"already_gold": 1, "complete_window": 1, "not_train": 2}


def test_eval_panel_heldout_only():
    cands = [q.Candidate(g, 1, priority=0.9) for g in ("tr", "held")]
    items, rej = run(cands, "measure_teacher", eval_panel=True)
    assert [i.game_id for i in items] == ["held"] and rej == {"not_heldout": 1}
    with pytest.raises(ValueError):
        run(cands, "improve_teacher", eval_panel=True)
    with pytest.raises(ValueError):
        run(cands, "random")


def test_candidates_from_preverify_and_audit_spread():
    doc = {"window_id": "tr_seg001_f000000", "audit": [1],
           "frames": [{"frame_index": 0, "verdict": "flagged", "priority": 0.8},
                      {"frame_index": 1, "verdict": "auto_ok", "priority": 0.0},
                      {"frame_index": 200, "verdict": "auto_ok", "priority": 0.0}]}
    cands = q.candidates_from_preverify(doc)
    assert cands[0].game_id == "tr_seg001" and cands[1].audit_frame and cands[0].flagged
    by = {(c.game_id, c.frame_index): c for c in cands}
    q.spread_audit_failures(by, [("tr_seg001", 1)])
    assert [c.audit_failure for c in cands] == [True, True, False]
