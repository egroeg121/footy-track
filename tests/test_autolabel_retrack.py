"""Re-tracking after each verdict (footy_track.autolabel.retrack): constraints, follow-up
frames, queue arrangement, tier-3 labels of anchored chains."""

from __future__ import annotations

import json

import pytest

from footy_track.autolabel import loop
from footy_track.autolabel import retrack as rt
from footy_track.ball_eval.detection_replay import CandidateCache, write_cache
from footy_track.ball_trackers.detection_tracker import DETECTED, BallCandidate
from footy_track.labeller.constants import card_box_index
from footy_track.scripts.autolabel_loop import retrack_ingest

W, H = 1920, 1080
SIZE = (W, H)


def cand(cx, cy, conf):
    return BallCandidate((cx - 5) / W, (cy - 5) / H, 10 / W, 10 / H, conf)


def ball_xy(f):
    return 300 + 12 * f, 500 - 2 * f


def make_cache(n=60, ball_conf=0.5, spare_conf=0.5):
    """A moving ball and a parked spare ball of equal confidence: the path takes the spare one."""
    frames = [
        (f, [cand(*ball_xy(f), ball_conf), cand(1500, 900, spare_conf)])
        for f in range(n)
    ]
    return CandidateCache("A", SIZE, frames, {})


def card(frame, x, y, cid=None, clip="A"):
    return {
        "id": cid or f"c{frame}",
        "clip": "stemA",
        "run_clip": clip,
        "frame_index": frame,
        "frame_size": [W, H],
        "bbox": {"x": (x - 5) / W, "y": (y - 5) / H, "w": 10 / W, "h": 10 / H},
    }


def key(c):
    return ("list", *rt.card_id_key(c))


def test_verdicts_become_constraints():
    on_ball = card(10, *ball_xy(10))
    c = rt.card_constraint(on_ball, {"verdict": "ball"})
    assert (c.kind, c.frame) == ("anchor", 10)
    assert (c.x, c.y) == pytest.approx(ball_xy(10))
    assert rt.card_constraint(on_ball, {"verdict": "box_off"}).kind == "anchor"
    assert rt.card_constraint(on_ball, {"verdict": "not_ball"}).kind == "forbid"
    assert rt.card_constraint(on_ball, {"verdict": "unsure"}) is None
    drag = {
        "verdict": "corrected",
        "corrected_bbox": {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.01},
    }
    c = rt.card_constraint(on_ball, drag)
    assert c.kind == "anchor" and abs(c.x - 0.505 * W) < 1e-6
    place = {**on_ball, "bbox": None}
    assert rt.card_constraint(place, {"verdict": "not_ball"}) is None
    assert rt.card_constraint(place, drag).kind == "anchor"


def test_retrack_follows_verdicts_and_reports_what_changed():
    cache = make_cache()
    sol0, _ = rt.solve(cache, [])
    pts0 = {p.frame_index: p for p in sol0.points}
    # without a person the tracker sits on the parked spare ball (listed first)
    assert rt.point_centre_px(pts0[30], SIZE)[0] > 1400
    # "ball" on the moving ball in frame 30 pulls the path onto it
    cons = [rt.Constraint("anchor", 30, *ball_xy(30))]
    sol1, _ = rt.solve(cache, cons)
    pts1 = {p.frame_index: p for p in sol1.points}
    assert 30 in sol1.anchored
    x29 = rt.point_centre_px(pts1[29], SIZE)[0]
    assert abs(x29 - ball_xy(29)[0]) < 15
    changed = rt.changed_frames(pts0, sol1)
    assert 29 in changed and 31 in changed
    assert rt.changed_frames(None, sol1) == set()


def test_followups_most_uncertain_first_dedupe_and_blocked():
    cache = make_cache()
    sol, _ = rt.solve(cache, [rt.Constraint("anchor", 30, *ball_xy(30))])
    fs = rt.followup_frames(sol, set(), [], limit=3)
    assert len(fs) == 3 and 30 not in fs
    # never two follow-ups within DEDUPE_FRAMES of each other
    assert all(abs(a - b) > rt.DEDUPE_FRAMES for a in fs for b in fs if a != b)
    # frames near a blocked (asked / queued) frame are skipped
    fs2 = rt.followup_frames(sol, set(), [fs[0]], limit=3)
    assert all(abs(f - fs[0]) > rt.DEDUPE_FRAMES for f in fs2)
    # a frame the last re-track changed gets a discount and goes first
    margins = {
        f: m
        for f, m in sol.margins.items()
        if f in set(range(60)) and f not in sol.anchored
    }
    cand_f = [f for f in margins if margins[f] < rt.UNSURE_MARGIN and f not in fs]
    if cand_f:
        top = rt.followup_frames(sol, {cand_f[-1]}, [], limit=60)
        assert top.index(cand_f[-1]) <= fs.index(fs[-1]) + len(cand_f)


def test_arrange_interleaves_and_never_starves_the_base_order():
    base = [{"id": f"b{i}"} for i in range(10)]
    pins = [{"id": f"p{i}"} for i in range(5)] + [base[7]]  # one pin is already queued
    out = [c["id"] for c in rt.arrange(base, pins, every=2)]
    assert out[:3] == ["p0", "p1", "b0"]
    assert sorted(out) == sorted({c["id"] for c in base} | {c["id"] for c in pins})
    assert len(out) == len(set(out))  # a card is never queued twice
    rest = [i for i in out if i.startswith("b") and i != "b7"]
    assert rest == [f"b{i}" for i in range(10) if i != 7]  # base order kept
    # with endless pins, one base card still comes after every two
    many = [{"id": f"p{i}"} for i in range(100)]
    out = [c["id"] for c in rt.arrange(base, many, every=2)]
    assert [i for i in out[:30] if i.startswith("b")] == [f"b{i}" for i in range(10)]


def test_chain_labels_only_for_chains_a_person_anchored():
    cache = make_cache()
    lines = rt.chain_label_lines(
        rt.solve(cache, [])[0], SIZE, "in_play_ball", ["claude_auto", "retrack"]
    )
    assert (
        lines == []
    )  # nobody confirmed anything: the detector's own tracker labels nothing
    sol, _ = rt.solve(cache, [rt.Constraint("anchor", 30, *ball_xy(30))])
    lines = rt.chain_label_lines(sol, SIZE, "in_play_ball", ["claude_auto", "retrack"])
    frames = {ln["frame_index"] for ln in lines}
    assert 30 not in frames and len(frames) > 20
    assert all(ln["tags"] == ["in_play_ball", "claude_auto", "retrack"] for ln in lines)
    assert all("human_checked" not in ln["tags"] for ln in lines)
    pts = {p.frame_index: p for p in sol.points}
    assert all(
        pts[f].source == DETECTED for f in frames
    )  # never an interpolated position


def test_dynamic_card_is_id_keyed_blind_and_stable():
    cache = make_cache()
    sol, _ = rt.solve(cache, [])
    p = next(q for q in sol.points if q.frame_index == 12)
    a = rt.dynamic_card("round-9", "A", "stemA", SIZE, p)
    b = rt.dynamic_card("round-9", "A", "stemA", SIZE, p)
    assert a == b and a["bbox"] and a["provenance"] == "list"
    assert not {"note", "type", "role", "other_bbox"} & set(a)
    assert a["id"] != rt.dynamic_card("round-8", "A", "stemA", SIZE, p)["id"]


def _verdict_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_ingest_labels_from_the_final_track(tmp_path):
    cache_dir = tmp_path / "cache"
    write_cache(cache_dir / "A.jsonl", make_cache())
    gt = tmp_path / "gt"
    rdir = gt / "retrack"
    rdir.mkdir(parents=True)
    (rdir / "state.json").write_text("{}")
    c30 = card(30, *ball_xy(30), cid="main30")
    dyn = {**card(40, *ball_xy(40), cid="dyn40"), "dynamic": True}
    dyn_bad = {**card(45, 1500, 900, cid="dyn45"), "dynamic": True}
    (rdir / "cards.json").write_text(json.dumps([dyn, dyn_bad]))

    def rec(c, v, **kw):
        return {
            "clip": c["clip"],
            "frame_index": c["frame_index"],
            "source": "list",
            "box_index": card_box_index(c["id"]),
            "verdict": v,
            "ts": 1.0,
            **kw,
        }

    _verdict_file(
        gt / "ball_checks" / "stemA.jsonl",
        [rec(c30, "ball"), rec(dyn, "ball", ts=2.0), rec(dyn_bad, "not_ball", ts=3.0)],
    )
    plan = loop.IngestPlan()
    out = retrack_ingest(
        gt / "ball_checks", None, cache_dir, {"cards": [c30]}, plan, {"A": (W, H)}
    )
    assert out["clips_retracked"] == 1 and out["constraints"] == 3
    assert out["labels_dynamic_tier2"] == 1 and out["negatives_dynamic"] == 1
    lines = plan.labels["A"]
    t3 = [
        ln for ln in lines if ln["tags"] == ["in_play_ball", "claude_auto", "retrack"]
    ]
    t2 = [ln for ln in lines if "human_checked" in ln["tags"]]
    assert len(t3) > 20 and loop.tier(t3[0]["tags"]) == 3
    assert len(t2) == 1 and t2[0]["frame_index"] == 40 and loop.tier(t2[0]["tags"]) == 2
    assert 30 not in {
        ln["frame_index"] for ln in t3
    }  # the card rules own the verified frames
    assert plan.negatives[0]["reason"] == "retrack_not_ball"
    # idempotent, and a server that never re-tracked (no retrack/ folder) changes nothing
    plan2 = loop.IngestPlan()
    retrack_ingest(
        gt / "ball_checks", None, cache_dir, {"cards": [c30]}, plan2, {"A": (W, H)}
    )
    assert plan2.labels == plan.labels
    assert (
        retrack_ingest(
            tmp_path / "other" / "ball_checks",
            None,
            cache_dir,
            {"cards": []},
            loop.IngestPlan(),
            {},
        )
        is None
    )


def test_arrange_base_first_after_a_streak():
    base = [{"id": f"b{i}"} for i in range(4)]
    pins = [{"id": f"p{i}"} for i in range(4)]
    assert rt.arrange(base, pins, every=2, base_first=True)[0]["id"] == "b0"
    assert rt.arrange(base, pins, every=2, base_first=False)[0]["id"] == "p0"
    assert rt.arrange(base, [], every=2, base_first=True) == base
