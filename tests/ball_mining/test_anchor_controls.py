"""Wrong-anchor controls (design ``ball_anchor_iteration.md`` task 11): S-gold control cards (injection, indistinguishability,
agreement), the suspect-anchor flow (record, release, regenerate as a new version, impact), the calibration firewall, the
reconciler report. Synthetic candidates and an in-process job store only."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from footy_track.ball_mining import anchor_controls as ctl
from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.ball_mining import reconcile, refit
from footy_track.feature_store.store import FeatureStore
from footy_track.labeller import jobs
from footy_track.scripts import ball_anchor_search as cli

from .test_anchor_client import (  # noqa: F401
    DAY,
    KEY,
    AppendApi,
    doc,
    seed,
    spec,
    submit,
)
from .test_anchor_search import HC, TC, register_splits, verdict  # noqa: F401

AX, AY = 0.41, 0.52
FRAMES = list(range(0, 100, 10))
GC = "tm_c_seg002"  # a clip of the same (train) match as TC
GBOX = {"x": 0.30, "y": 0.40, "w": 0.02, "h": 0.04}


def gold_rows(n=6, clip=GC, kinds=("gt_shifted", "known_not_ball")):
    return pd.DataFrame([{"match": "tm_c", "clip": clip, "frame": 500 + i, "fps": 25.0, "x": 0.5, "y": 0.5, "w": 0.01, "h": 0.02,
                          "kind": kinds[i % len(kinds)]} for i in range(n)])  # fmt: skip


class Store:
    def read_selected_frames(self, job_key=None, **_):
        return pd.DataFrame({"job_key": job_key, "game_id": TC, "frame_index": FRAMES, "split": "train"})

    def read_ball_terminal_states(self, game, frames=None):
        return {}


def answer_card(ds, jid, frame, v):
    it = next(i for i in jobs.read_job(jid)["items"] if i["frame_index"] == frame)
    d = ds / "ball_checks"
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{it['clip']}.jsonl").open("a") as fh:
        fh.write(json.dumps({"clip": it["clip"], "frame_index": frame, "box_index": it["box_index"], "source": it["source"], "verdict": v,
                             "bbox": it["bbox"]}) + "\n")  # fmt: skip


# ---- pure pieces ------------------------------------------------------------------------------------
def test_constants_hop_cap_and_calibration_bucket_are_pinned():
    assert A.HOP_CAP == 3 and A.HOP_DECAY == 0.8 and ctl.CONTROL_RATE == 0.05
    assert ctl.SEARCH_BUCKET == refit.SEARCH_BUCKET == A.SEARCH_BUCKET


def test_controls_due_is_cumulative_and_never_negative():
    assert ctl.n_controls_due(19, 0) == 0 and ctl.n_controls_due(20, 0) == 1 and ctl.n_controls_due(40, 1) == 1
    assert ctl.n_controls_due(40, 2) == 0 and ctl.n_controls_due(40, 5) == 0


def test_pick_controls_deterministic_split_guarded_and_never_a_taken_frame(ds):
    g = pd.concat([gold_rows(8), gold_rows(4, clip=HC).assign(frame=lambda d: d["frame"] + 100)], ignore_index=True)
    a = ctl.pick_controls(g, job_id="j1", split="train", taken={(GC, 500), (GC, 501)}, n=4)
    assert a == ctl.pick_controls(g, job_id="j1", split="train", taken={(GC, 500), (GC, 501)}, n=4)
    assert len(a) == 4 and all(c["clip"] == GC and c["frame_index"] not in (500, 501) for c in a)  # heldout clip never drawn
    assert {c["kind"] for c in a} <= {"gt_exact", "gt_shifted", "known_not_ball"}
    for c in a:
        assert c["expected"] == ctl.EXPECTED[c["kind"]] and (c["shift_px"] is not None) == (c["kind"] == "gt_shifted")
        if c["shift_px"] is not None:
            assert 15.0 <= c["shift_px"] <= 30.0
    assert ctl.pick_controls(g, job_id="j1", split="eval_heldout_match", taken=(), n=3)[0]["clip"] == HC  # heldout job: heldout only
    assert ctl.pick_controls(g, job_id="j1", split=None, taken=(), n=3) == []
    assert ctl.pick_controls(g, job_id="j1", split="train", taken=(), n=3, playable=lambda c: None) == []


def test_agreement_per_window_and_unsure_is_neutral():
    def row(v, kind, w):
        return {"bucket": "S", "control": True, "verdict": v, "gold_kind": kind, "expected": ctl.EXPECTED[kind], "window_id": w,
                "anchor_id": f"j:{TC}:0", "clip": GC, "frame": 500}  # fmt: skip

    rows = [row("not_ball", "known_not_ball", "w1"), row("ball", "known_not_ball", "w1"), row("ball", "gt_exact", "w2"), row("unsure", "gt_exact", "w2"),
            {"bucket": "S", "control": False, "verdict": "ball"}]  # fmt: skip
    a = ctl.agreement(rows)
    assert (a["n"], a["agree"], a["disagree"], a["unsure"], a["rate"]) == (4, 2, 1, 1, round(2 / 3, 4))
    assert a["by_window"]["w1"]["rate"] == 0.5 and a["by_window"]["w2"]["rate"] == 1.0
    assert len(a["disagreements"]) == 1 and a["disagreements"][0]["verdict"] == "ball"


def test_detect_suspects_rules_and_idempotent_store(tmp_path):
    aid = f"j:{TC}:0"
    base = {"bucket": "S", "anchor_id": aid, "window_id": "w", "clip": TC}
    rows = [
        {**base, "verdict": "not_ball", "hop": 1, "frame": 10},  # contradicts the anchor
        {**base, "verdict": "not_ball", "hop": 2, "frame": 20},  # too far from the anchor to contradict it
        {**base, "verdict": "ball", "hop": 1, "frame": 11},
        {**base, "verdict": "ball", "control": True, "gold_kind": "known_not_ball", "expected": ["not_ball", "box_off"], "frame": 500},
        {**base, "bucket": "R", "verdict": "not_ball", "hop": 1, "frame": 12},  # not a search card
    ]
    sus = ctl.detect_suspects(rows)
    assert {(s["reason"], s["evidence"]["frame"]) for s in sus} == {("contradicted", 10), ("control_disagree", 500)}
    assert sus == ctl.detect_suspects(rows[::-1])
    assert len(ctl.record_suspects(tmp_path, "j", sus)) == 2 and ctl.record_suspects(tmp_path, "j", sus) == []
    assert len(ctl.read_suspects(tmp_path, "j")) == 2 and ctl.suspect_keys(ctl.read_suspects(tmp_path, "j"), "j") == {(TC, 0)}
    assert ctl.suspect_keys(ctl.read_suspects(tmp_path, "j"), "other-job") == frozenset()


# ---- injection ---------------------------------------------------------------------------------------
@pytest.fixture
def cl():
    return AppendApi()


def push(cl, jid, docs, **kw):
    kw.setdefault("today", DAY)
    return A.push_search(jid, docs, client=cl, **kw)


def many(n, start=1000, window="w1"):
    return doc([spec(start + i, i, window=window) for i in range(n)], window=window, digest=f"d{window}")


def items_of(jid):
    return jobs.read_job(jid)["items"]


def test_controls_are_injected_at_five_percent_cumulatively_and_never_twice(ds, cl):
    jid = submit([seed(0)])
    out = push(cl, jid, [many(40)], gold=gold_rows(8))
    assert out["added"] == 42 and out["controls_added"] == 2  # floor(0.05 * 40)
    c = [i for i in items_of(jid) if ctl.is_control_item(i)]
    assert len(c) == 2 and all(i["sampling"]["bucket"] == "S" for i in c)
    again = push(cl, jid, [many(40)], gold=gold_rows(8))
    assert again["added"] == 0 and again.get("controls_added", 0) == 0  # a retry adds nothing
    more = push(cl, jid, [many(20, start=2000, window="w2")], gold=gold_rows(8))  # 60 search cards -> 3 controls in all
    assert more["controls_added"] == 1 and sum(ctl.is_control_item(i) for i in items_of(jid)) == 3
    frames = [i["frame_index"] for i in items_of(jid)]
    assert len(frames) == len(set(frames))  # a control frame is never a frame already in the job


def test_a_control_card_is_indistinguishable_from_a_search_card(ds, cl):
    jid = submit([seed(0)])
    push(cl, jid, [many(20)], gold=gold_rows(4))
    its = items_of(jid)
    (c,) = [i for i in its if ctl.is_control_item(i)]
    s = next(i for i in its if (i.get("sampling") or {}).get("bucket") == "S" and not ctl.is_control_item(i))
    assert set(c) == set(s) and c["sources"] == s["sources"] and "p_real" in c
    assert set(c["sampling"]) - {"gold"} == set(s["sampling"]) and set(c["search"]) - {"control"} == set(s["search"])
    assert c["sampling"]["regime"] == "train" and c["sampling"]["window_id"] == "w1" and c["search"]["window_id"] == "w1"
    assert c["clip"] == GC and c["sampling"]["gold"]["expected"] == ctl.EXPECTED[c["sampling"]["gold"]["kind"]]
    assert not {"label", "provenance", "confidence", "note", "kind", "control"} & set(c)  # nothing visible beyond a normal card
    assert c["search"]["anchor_id"] == c["sampling"]["anchor_id"]  # the host's anchor: where a disagreement is charged


def test_controls_respect_split_selected_set_and_the_cap(ds, cl):
    jid = submit([seed(0)])
    held = gold_rows(6, clip=HC)  # heldout gold never enters a train job
    assert push(cl, jid, [many(40)], gold=held).get("controls_added", 0) == 0
    out = push(cl, jid, [many(40, start=3000, window="w3")], gold=gold_rows(6), selected={(GC, f) for f in range(500, 506)})
    assert out.get("controls_added", 0) == 0  # every gold frame is in the job's selected set F: not drawn


def test_controls_take_cap_headroom_from_the_search_cards(ds, cl):
    jid = submit([seed(i) for i in range(70)])
    out = push(cl, jid, [many(60)], gold=gold_rows(6), control_rate=0.2)  # room 30: 30 cards in all, controls included
    assert out["added"] == 30 and out["headroom"]["room"] == 30 and 1 <= out["controls_added"] <= 6
    assert sum(ctl.is_control_item(i) for i in items_of(jid)) == out["controls_added"]


def test_no_gold_no_controls(ds, cl):
    jid = submit([seed(0)])
    assert push(cl, jid, [many(40)]).get("controls_added", 0) == 0


# ---- calibration firewall ------------------------------------------------------------------------------
def test_control_verdicts_never_reach_platt_or_ltt_and_are_reported_apart():
    def row(bucket, v, control=False, **kw):
        return {"bucket": bucket, "verdict": v, "p_real": 0.9 if v == "ball" else 0.2, "control": control, "scorer_version": "s",
                "match": "m", "pi": 0.1, "clip": "c", "frame": 1, "gold_kind": "known_not_ball" if control else None, **kw}  # fmt: skip

    rows = [row("S", "ball", True)] * 30 + [row("S", "not_ball", True)] * 30 + [row("R", "ball")] * 12 + [row("R", "not_ball")] * 12
    kept = refit.exclude_search(rows)
    assert {r["bucket"] for r in kept} == {"R"} and len(kept) == 24
    assert refit.fit_calibration(rows)["n"] == 24  # the 60 controls (p_real included) are not in the fit
    assert "S" not in refit.hit_rates(rows) and refit.hit_rates(rows)["R"]["n"] == 24
    assert ctl.agreement(rows)["n"] == 60


def test_rows_from_results_carries_the_control_fields():
    samp = {"regime": "train", "bucket": "S", "anchor_id": "j:c:1", "hop": 2, "window_id": "w9",
            "gold": {"kind": "known_not_ball", "expected": ["not_ball", "box_off"]}}  # fmt: skip
    lab = {"clip": GC, "frame_index": 500, "verdict": "ball", "meta": {"sampling": samp}}
    (r,) = refit.rows_from_results({"job": {"id": "j"}, "labels": [lab]})
    assert r["control"] is True and r["anchor_id"] == "j:c:1" and r["window_id"] == "w9" and r["expected"] == ["not_ball", "box_off"]


# ---- suspect flow end to end (CLI run_job) ------------------------------------------------------------
NEAR = {f: [A.Cand(AX + 0.001 * f / 10, AY, 0.02, 0.04, 0.8, 7)] for f in FRAMES}
ABOX = {"x": 0.4, "y": 0.5, "w": 0.02, "h": 0.04}


@pytest.fixture
def two_anchor(ds, monkeypatch):
    cs = A.CandidateSet("job_suggest", "abcd1234")
    cs.by_frame.update({(TC, f): list(c) for f, c in NEAR.items()})
    monkeypatch.setattr(A, "load_candidates", lambda j, **k: cs)
    jid = submit([seed(0, "R"), seed(90, "R")])
    verdict(ds, TC, 0, "ball", bbox=ABOX)
    verdict(ds, TC, 90, "ball", bbox={**ABOX, "x": 0.4 + 0.009})
    return jid, AppendApi()


def run(jid, cl, ds, **kw):
    kw.setdefault("controls", False)
    return cli.run_job(jid, client=cl, marks_dir=ds, store=Store(), today=DAY, **kw)


def search_files(ds):
    return sorted(p.name for p in (ds / "search").rglob("v*.json"))


def test_rejection_next_to_an_anchor_marks_it_suspect_regenerates_and_reports_impact(two_anchor, ds):
    jid, cl = two_anchor
    first = run(jid, cl, ds)
    assert first["push"]["added"] == 6 and first["controls"]["suspect_anchors"] == 0
    v1 = A.latest_search(ds, A.window_id_of(TC, 0))
    assert {c["frame_index"] for c in v1["cards"]} == {10, 20, 30, 60, 70, 80} and v1["version"] == 1
    answer_card(ds, jid, 10, "not_ball")  # hop 1 from the anchor at frame 0
    second = run(jid, cl, ds)
    c = second["controls"]
    assert c["newly_suspect"] == [f"{jid}:{TC}:0"] and c["suspect_anchors"] == 1 and not second["noop"]
    imp = c["impact"][f"{jid}:{TC}:0"]
    assert [x["frame"] for x in imp["answered_cards"]] == [10] and [x["frame"] for x in imp["open_cards"]] == [20, 30]
    assert [x["frame"] for x in imp["in_previous_search"]] == [10, 20, 30]  # the cards the previous search file derived from it
    v2 = A.latest_search(ds, A.window_id_of(TC, 0))
    assert v2["version"] == 2 and v2["digest"] != v1["digest"] and v2["suspect_anchors"] == [0]  # a NEW immutable file
    assert all(x["anchor_id"] != f"{jid}:{TC}:0" for x in v2["cards"])  # (carded frames are pending, so a rerun adds none)
    assert search_files(ds) == ["v001.json", "v002.json"]
    third = run(jid, cl, ds)  # idempotent: nothing recorded, nothing written, nothing appended
    assert third["noop"] and third["controls"]["newly_suspect"] == [] and search_files(ds) == ["v001.json", "v002.json"]
    assert len(ctl.read_suspects(ds, jid)) == 1
    assert len(jobs.read_job(jid)["items"]) == 8


def test_window_left_without_anchor_gets_a_cardless_version(ds, monkeypatch):
    cs = A.CandidateSet("job_suggest", "abcd1234")
    cs.by_frame.update({(TC, f): list(c) for f, c in NEAR.items()})
    monkeypatch.setattr(A, "load_candidates", lambda j, **k: cs)
    jid = submit([seed(0, "R")])
    verdict(ds, TC, 0, "ball", bbox=ABOX)
    cl = AppendApi()
    assert run(jid, cl, ds)["push"]["added"] == 3
    answer_card(ds, jid, 10, "box_off")
    out = run(jid, cl, ds)
    v2 = A.latest_search(ds, A.window_id_of(TC, 0))
    assert v2["version"] == 2 and v2["cards"] == [] and out["controls"]["suspect_anchors"] == 1 and not out["noop"]
    assert len(jobs.read_job(jid)["items"]) == 4  # the suspect anchor's cards are not re-asked, nothing is appended


def test_control_disagreement_marks_the_hosts_anchor_suspect(two_anchor, ds):
    jid, cl = two_anchor
    first = run(jid, cl, ds, gold=gold_rows(4, kinds=("known_not_ball",)), playable=lambda c: c, controls=True, control_rate=0.5)
    assert first["controls"]["controls_added"] == 3  # floor(0.5 * 6) = 3, capped by the 4 gold frames
    cards = [i for i in jobs.read_job(jid)["items"] if ctl.is_control_item(i)]
    bad, good = cards[0], cards[1]
    answer_card(ds, jid, bad["frame_index"], "ball")  # a known not-ball accepted: the labeller was careless
    answer_card(ds, jid, good["frame_index"], "not_ball")
    out = run(jid, cl, ds, controls=False)
    a = out["controls"]["agreement"]
    assert (a["agree"], a["disagree"], a["rate"]) == (1, 1, 0.5)
    assert out["controls"]["newly_suspect"] == [bad["search"]["anchor_id"]]
    (e,) = ctl.read_suspects(ds, jid)
    assert e["reason"] == "control_disagree" and e["evidence"]["frame"] == bad["frame_index"]
    # the control frame itself never became an anchor / a measured frame of F
    assert all(not ctl.is_control_item(i) or i["frame_index"] >= 500 for i in jobs.read_job(jid)["items"])


def test_suspects_of_another_job_or_outside_f_are_ignored(ds, monkeypatch):
    cs = A.CandidateSet("job_suggest", "abcd1234")
    cs.by_frame.update({(TC, f): list(c) for f, c in NEAR.items()})
    monkeypatch.setattr(A, "load_candidates", lambda j, **k: cs)
    verdict(ds, TC, 0, "ball", bbox=ABOX)
    job = {"id": "r1", "kind": "ball_check", "dataset": "train",
           "items": [{"clip": TC, "frame_index": f, "box_index": 0, "source": "list", "bbox": ABOX} for f in FRAMES]}  # fmt: skip
    (d,) = A.run_search(job, marks_dir=ds, suspects={(TC, 5000)})
    assert d["cards"] and d["suspect_anchors"] == []


def test_regenerated_search_excludes_the_suspect_anchor_as_a_new_version(ds, monkeypatch):
    cs = A.CandidateSet("job_suggest", "abcd1234")
    cs.by_frame.update({(TC, f): list(c) for f, c in NEAR.items()})
    monkeypatch.setattr(A, "load_candidates", lambda j, **k: cs)
    verdict(ds, TC, 0, "ball", bbox=ABOX)
    verdict(ds, TC, 90, "ball", bbox={**ABOX, "x": 0.409})
    job = {"id": "r1", "kind": "ball_check", "dataset": "train",
           "items": [{"clip": TC, "frame_index": f, "box_index": 0, "source": "list", "bbox": ABOX} for f in FRAMES]}  # fmt: skip
    (v1,) = A.run_search(job, marks_dir=ds)
    assert {c["frame_index"] for c in v1["cards"]} == {10, 20, 30, 60, 70, 80}
    (same,) = A.run_search(job, marks_dir=ds)
    assert same["reused"] and search_files(ds) == ["v001.json"]
    (v2,) = A.run_search(job, marks_dir=ds, suspects={(TC, 0)})
    assert not v2["reused"] and v2["version"] == 2 and v2["digest"] != v1["digest"] and v2["suspect_anchors"] == [0]
    assert {c["frame_index"] for c in v2["cards"]} == {60, 70, 80} and v2["n_anchors"] == 1
    assert 0 not in {c["frame_index"] for c in v2["cards"]}  # the released anchor's own frame is judged, never re-asked
    (again,) = A.run_search(job, marks_dir=ds, suspects={(TC, 0)})
    assert again["reused"] and search_files(ds) == ["v001.json", "v002.json"]  # idempotent


# ---- reconciler ----------------------------------------------------------------------------------------
NOW = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)


class ControlClient:
    def __init__(self):
        self.key = "ball-round/2026-11-20/1"

    def list_jobs(self, **_):
        return [{"id": "id0", "key": self.key, "kind": "ball_check", "status": "open", "created": (NOW - timedelta(hours=1)).isoformat()}]

    def results(self, ref):
        box = {"x": 0.1, "y": 0.2, "w": 0.01, "h": 0.01}
        samp = {"regime": "train", "bucket": "S", "anchor_id": "id0:m1:1", "hop": 1, "window_id": "w",
                "gold": {"kind": "known_not_ball", "expected": ["not_ball", "box_off"]}}  # fmt: skip
        items = [{"index": 0, "item": {"clip": "m1", "frame_index": 1, "bbox": box}},
                 {"index": 1, "item": {"clip": "m1", "frame_index": 99, "bbox": box, "sampling": samp, "search": {"control": True}}}]  # fmt: skip
        labels = [{"clip": "m1", "frame_index": 99, "verdict": "ball", "meta": {"sampling": samp, "search": {"control": True}}}]
        return {"job": {"id": "id0"}, "items": items, "labels": labels}


def test_reconciler_does_not_flag_controls_as_orphans_and_reports_them(tmp_path):
    store = FeatureStore.open(":memory:")
    from footy_track.feature_store.schema import SelectedFrameRow  # noqa: PLC0415

    store.upsert_selected_frames([SelectedFrameRow(job_key="ball-round/2026-11-20/1", game_id="m1", frame_index=1, window_id="w", split="train",
                                                   selected_at=NOW - timedelta(hours=1))])  # fmt: skip
    ctl.record_suspects(tmp_path, "id0", [{"anchor_id": "id0:m1:1", "reason": "control_disagree", "window_id": "w", "evidence": {"clip": "m1", "frame": 99}}])
    rep = reconcile.reconcile(store, ControlClient(), now=NOW, split_fn=lambda c: "train", marks_dir=tmp_path)
    t = rep["families"]["train"]
    assert t["counts"]["orphans"] == 0 and t["counts"]["missing"] == 0  # the control frame (99) is not in F by design
    k = t["controls"]
    assert k["control_cards"] == 1 and k["agreement"]["disagree"] == 1 and k["suspect_anchors"] == 1
    assert k["jobs"][ControlClient().key]["impact"]["id0:m1:1"]["n_cards"] == 0
    assert "1 suspect anchor" in reconcile.summary(rep)
