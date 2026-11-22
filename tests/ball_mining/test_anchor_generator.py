"""Anchor generator (design ``ball_anchor_iteration.md`` task 3): ranked frontier cards, judged frames skipped, versioned
immutable search files, unreachable frames, and the selected-set / split guard. Synthetic candidates only."""

from __future__ import annotations

import pandas as pd
import pytest

from footy_track.ball_mining import anchor_search as A  # noqa: N812

from .test_anchor_search import HC, TC, check_job, register, verdict  # noqa: F401

AX, AY = 0.41, 0.52  # anchor centre (the verdict BOX centre)


def cand(x, y, conf=0.8, tid=0):
    return A.Cand(x, y, 0.02, 0.04, conf, tid)


def cset(spec):
    cs = A.CandidateSet("job_suggest", "abcd1234")
    for f, cs_ in spec.items():
        cs.by_frame[(TC, f)] = list(cs_)
    return cs


@pytest.fixture
def run(ds, monkeypatch):
    """run(job, spec, **kw) -> search docs, candidates injected (no GPU / suggest file)."""

    def go(job, spec, **kw):
        monkeypatch.setattr(A, "load_candidates", lambda j, **k: cset(spec))
        return A.run_search(job, marks_dir=ds, **kw)

    return go


FRAMES = list(range(0, 100, 10))
NEAR = {f: [cand(AX + 0.001 * f / 10, AY, 0.8, 7)] for f in FRAMES}


def test_frontier_order_hop_cap_and_decay(ds, run):
    verdict(ds, TC, 0, "ball")
    (doc,) = run(check_job(TC, FRAMES), NEAR, teacher_version="offline_v1", run_id="r9")
    assert [c["frame_index"] for c in doc["cards"]] == [10, 20, 30]
    assert [c["hop"] for c in doc["cards"]] == [1, 2, 3]
    assert [b["frame"] for b in doc["beyond_cap"]] == [40, 50, 60, 70, 80, 90]  # needs a fresh seed
    c = doc["cards"][1]
    assert c["conf"] == pytest.approx(0.8 * A.HOP_DECAY**2)
    assert (c["source"], c["direction"], c["search_round"], c["run_id"], c["teacher_version"]) == ("search", "fwd", 1, "r9", "offline_v1")
    assert c["anchor_id"] == f"r1:{TC}:0" and c["p_cal"] is None and c["generator_version"].startswith("det:abcd1234|trk:")
    assert doc["version"] == 1 and doc["split"] == "train"


def test_tie_break_by_margin_then_confidence(ds, run):
    verdict(ds, TC, 0, "ball")
    verdict(ds, TC, 100, "ball")
    spec = {f: [cand(AX, AY, 0.8, 7)] for f in range(0, 101, 10)}
    spec[10] = [cand(AX, AY, 0.8), cand(AX + 0.004, AY, 0.79)]  # contestable -> small margin
    spec[90] = [cand(AX, AY, 0.8)]  # single candidate: infinite margin (None), after the contestable one
    spec[20] = [cand(AX, AY, 0.9)]  # hop 2 from either side: after every hop-1 card
    (doc,) = run(check_job(TC, list(range(0, 101, 10))), spec)
    order = [c["frame_index"] for c in doc["cards"]]
    assert order[:2] == [10, 90] and doc["cards"][0]["margin"] is not None and doc["cards"][1]["margin"] is None
    assert [c["hop"] for c in doc["cards"]] == sorted(c["hop"] for c in doc["cards"])
    assert doc["cards"][0]["direction"] == "fwd" and doc["cards"][1]["direction"] == "bwd"


def test_bracketed_frame_direction_is_both(ds, run):
    verdict(ds, TC, 0, "ball")
    verdict(ds, TC, 40, "ball")
    (doc,) = run(check_job(TC, FRAMES), NEAR)
    assert {c["frame_index"]: c["direction"] for c in doc["cards"]}[20] == "both"


def test_tid_bonus_breaks_ranking_ties(ds, run):
    verdict(ds, TC, 0, "ball")
    verdict(ds, TC, 100, "ball")
    spec = {f: [cand(AX, AY, 0.7, 7)] for f in range(0, 101, 10)}
    spec[0] = [cand(AX, AY, 0.9, 7)]
    spec[100] = [cand(AX, AY, 0.9, 8)]
    spec[10] = [cand(AX, AY, 0.7, 7)]  # same tracklet as anchor 0
    spec[90] = [cand(AX, AY, 0.7, 9)]  # other tracklet
    (doc,) = run(check_job(TC, list(range(0, 101, 10))), spec)
    cards = {c["frame_index"]: c for c in doc["cards"]}
    assert cards[10]["rank_conf"] > cards[90]["rank_conf"]
    assert [c["frame_index"] for c in doc["cards"]][:2] == [10, 90]


def test_judged_frames_are_skipped(ds, run):
    class Store:
        def read_selected_frames(self, job_key=None, **_):
            return pd.DataFrame(columns=["job_key", "game_id", "frame_index", "split"])

        def read_ball_terminal_states(self, game, frames=None):
            return {30: "no_ball"}

    verdict(ds, TC, 0, "ball")
    verdict(ds, TC, 10, "not_ball")  # live verdict
    (doc,) = run(check_job(TC, FRAMES), NEAR, store=Store(), pending={(TC, 20)})
    assert [c["frame_index"] for c in doc["cards"]] == []  # 10 verdicted, 20 pending, 30 terminal; rest beyond the cap
    assert doc["unreachable"] == [] and {b["frame"] for b in doc["beyond_cap"]} == set(range(40, 100, 10))


def test_forbidden_disc_removes_the_rejected_candidate(ds, run):
    verdict(ds, TC, 0, "ball")
    bad = {"x": AX - 0.01, "y": AY - 0.02, "w": 0.02, "h": 0.04}
    verdict(ds, TC, 10, "not_ball", bbox=bad)
    spec = {10: [cand(AX, AY, 0.9)], 20: [cand(AX, AY, 0.8)]}
    (doc,) = run(check_job(TC, FRAMES), spec)
    assert 10 not in [c["frame_index"] for c in doc["cards"]]


def test_no_candidate_in_gate_is_unreachable(ds, run):
    verdict(ds, TC, 0, "ball")
    spec = {10: [cand(AX + 0.5, AY, 0.9)], 20: [cand(AX + 0.5, AY, 0.9)], 30: [cand(AX, AY, 0.8)]}  # 960 px away: outside the gate
    (doc,) = run(check_job(TC, FRAMES), spec)
    un = {u["frame"]: u for u in doc["unreachable"]}
    assert set(un) == {10, 20} and un[10]["reason"] == A.REASON_NO_CANDIDATE and un[10]["window_id"] == doc["window_id"]
    assert [c["frame_index"] for c in doc["cards"]] == [30]


def test_same_digest_writes_no_new_file_and_new_input_makes_next_version(ds, run):
    verdict(ds, TC, 0, "ball")
    job = check_job(TC, FRAMES)
    (d1,) = run(job, NEAR)
    (d2,) = run(job, NEAR, run_id="other-run")
    wdir = ds / "search" / d1["window_id"]
    assert [p.name for p in sorted(wdir.iterdir())] == ["v001.json"] and d1["reused"] is False and d2["reused"] is True
    assert d2["version"] == 1
    verdict(ds, TC, 10, "ball")  # new anchor -> new input -> v002, search_round 2
    (d3,) = run(job, NEAR)
    assert [p.name for p in sorted(wdir.iterdir())] == ["v001.json", "v002.json"]
    assert d3["version"] == 2 and d3["search_round"] == 2 and d3["cards"][0]["search_round"] == 2
    assert (wdir / "v001.json").read_text()  # immutable
    # a changed candidate set changes the digest too
    (d4,) = run(job, {**NEAR, 40: [cand(AX, AY, 0.3)]})
    assert d4["version"] == 3


def test_windows_never_span_clips_or_exceed_250(ds, run):
    w = A.split_windows([(TC, 0), (TC, 249), (TC, 250), ("other_seg002", 5)])
    assert set(w) == {A.window_id_of(TC, 0), A.window_id_of(TC, 250), A.window_id_of("other_seg002", 0)}
    verdict(ds, TC, 240, "ball")
    spec = {f: [cand(AX, AY, 0.8)] for f in (240, 250, 260)}
    docs = run(check_job(TC, [240, 250, 260]), spec)
    assert len(docs) == 1 and [c["frame_index"] for c in docs[0]["cards"]] == []  # 250/260 are another window, no anchor there


def test_window_without_anchor_emits_nothing(ds, run):
    assert run(check_job(TC, FRAMES), NEAR) == []


# ---- guard -------------------------------------------------------------------------------------
class SelStore:
    def __init__(self, rows):
        self.rows = rows

    def read_selected_frames(self, job_key=None, **_):
        return pd.DataFrame([r for r in self.rows if r["job_key"] == job_key], columns=["job_key", "game_id", "frame_index", "split"])

    def read_ball_terminal_states(self, game, frames=None):
        return {}


def sel(job, clip, frames, split="train"):
    return [{"job_key": job, "game_id": clip, "frame_index": f, "split": split} for f in frames]


def test_selected_frame_rows_define_F(ds, run):
    verdict(ds, TC, 0, "ball")
    store = SelStore(sel("r1", TC, [0, 10, 20]))  # job items go to 90 but F stops at 20
    (doc,) = run(check_job(TC, FRAMES), NEAR, store=store)
    assert [c["frame_index"] for c in doc["cards"]] == [10, 20] and doc["beyond_cap"] == []


def test_selected_frames_of_another_job_are_ignored(ds, run):
    verdict(ds, TC, 0, "ball")
    store = SelStore(sel("other-job", TC, [0, 10]))  # nothing for r1 -> falls back to the job's item frames
    (doc,) = run(check_job(TC, FRAMES), NEAR, store=store)
    assert [c["frame_index"] for c in doc["cards"]] == [10, 20, 30]


def test_selected_row_of_another_split_is_refused(ds, run):
    verdict(ds, TC, 0, "ball")
    with pytest.raises(A.SplitError):
        run(check_job(TC, FRAMES), NEAR, store=SelStore(sel("r1", TC, [0, 10], split="eval_heldout_match")))
    with pytest.raises(A.SplitError):
        run(check_job(TC, FRAMES), NEAR, store=SelStore(sel("r1", HC, [0, 10])))  # heldout clip labelled train


def test_check_emittable_rejects_frames_outside_F_or_split(register):
    card = {"clip": TC, "frame_index": 10}
    A.check_emittable([card], frozenset({(TC, 10)}), "train")
    with pytest.raises(A.SplitError):
        A.check_emittable([card], frozenset({(TC, 20)}), "train")
    with pytest.raises(A.SplitError):
        A.check_emittable([{"clip": HC, "frame_index": 10}], frozenset({(HC, 10)}), "train")


def test_anchors_of_another_job_or_split_never_seed_the_search(ds, run):
    verdict(ds, TC, 500, "ball")  # another job's frame
    verdict(ds, HC, 0, "ball")  # another split
    assert run(check_job(TC, FRAMES), NEAR) == []
