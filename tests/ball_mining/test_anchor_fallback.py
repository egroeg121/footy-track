"""Fallback full-frame check job (design ``ball_anchor_iteration.md`` task 7): unreachable / beyond-cap / unsure-twice frames
produce one idempotent ``ball_place`` job per window (eval family: per panel and match), the human's placement or explicit
"No ball" becomes the terminal state (``no_ball`` only through the full-frame check) and a placed box re-seeds the search.
Real in-process job store + labeller sidecars, in-memory feature store, no GPU."""

from __future__ import annotations

import json

import pytest

from footy_track.ball_mining import anchor_fallback as F  # noqa: N812
from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.feature_store import FeatureStore
from footy_track.labeller import jobs, jobs_client

from .test_anchor_client import AppendApi
from .test_anchor_generator import AX, AY, cand
from .test_anchor_search import (  # noqa: F401
    HC,
    TC,
    check_job,
    register_splits,
    verdict,
)

FRAMES = list(range(0, 100, 10))
ROUND = "ball-round/2026-11-21/1"
# frames 10, 20 have only a far candidate (outside the gate) -> unreachable; 30 is reachable; 40+ are beyond the hop cap
SPEC = {0: [cand(AX, AY, 0.8, 7)], 10: [cand(AX + 0.5, AY)], 20: [cand(AX + 0.5, AY)], 30: [cand(AX, AY, 0.8, 7)]}


class Api(AppendApi):
    def submit(self, title, kind="other", items=None, *, dataset=None, why="", device="either", priority=2, requester=None,
               key=None, url=None, timeout=60, prepare=False, max_cards=None, hooks=None):  # noqa: ARG002
        self.calls.append(("submit", key, len(items or [])))
        body = {"title": title, "kind": kind, "items": items, "why": why, "device": device, "key": key, "dataset": dataset}
        try:
            job, created = jobs.submit(body, dataset or "train")
        except jobs.JobError as e:
            raise jobs_client.JobsError(e.status, {"detail": e.detail}) from None
        return {"id": job["id"], "created": created, "status": job["status"]}


def cset_for(spec, clip):
    cs = A.CandidateSet("job_suggest", "abcd1234")
    for f, c in spec.items():
        cs.by_frame[(clip, f)] = list(c)
    return cs


@pytest.fixture
def cl():
    return Api()


@pytest.fixture
def store():
    s = FeatureStore.open(":memory:")
    yield s
    s.close()


@pytest.fixture
def search(ds, monkeypatch):
    def go(job, spec=SPEC, clip=TC, **kw):
        monkeypatch.setattr(A, "load_candidates", lambda j, **k: cset_for(spec, clip))
        return A.run_search(job, marks_dir=ds, **kw)

    return go


def source_job(clip=TC, key=ROUND, frames=FRAMES):
    return {**check_job(clip, frames), "key": key}


def fallback_jobs():
    return [j for j in jobs.all_jobs() if j["kind"] == "ball_place"]


def mark(ds, clip, frame, *tags, center=None):
    rec = {"frame_index": frame, "tags": list(tags)}
    if center:
        rec["center"] = {"x": center[0], "y": center[1]}
    with (ds / f"{clip}.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def place(ds, clip, frame, x=0.5, y=0.5):
    mark(ds, clip, frame, "in_play_ball", "labeller", center=(x, y))


def no_ball(ds, clip, frame):
    mark(ds, clip, frame, "no_ball", "labeller")


def results_of(cl, job_id):
    return cl.results(job_id)


def pushed(cl, ds, search, store=None, job=None, **kw):
    job = job or source_job()
    docs = search(job)
    return job, docs, F.push_fallback(job, docs, client=cl, marks_dir=ds, store=store, **kw)


# ---- producing the job -------------------------------------------------------------------------
def test_unreachable_and_beyond_cap_make_exactly_one_idempotent_job(ds, cl, search):
    verdict(ds, TC, 0, "ball")
    job, docs, out = pushed(cl, ds, search)
    assert [u["frame"] for u in docs[0]["unreachable"]] == [10, 20]
    wid = docs[0]["window_id"]
    assert out["family"] == "train" and out["frames"] == 8  # 10, 20 unreachable + 40..90 beyond the cap
    (g,) = out["groups"]
    assert g["key"] == f"ball-place/{wid}" and g["created"] and g["added"] == 8
    (fj,) = fallback_jobs()
    assert fj["key"] == f"ball-place/{wid}" and fj["kind"] == "ball_place" and fj["device"] == "laptop"
    items = fj["items"]
    assert [it["frame"] for it in items] == [10, 20, 40, 50, 60, 70, 80, 90]
    it = items[0]
    assert (it["clip"], it["split"], it["window_id"], it["source"], it["source_job"]) == (TC, "train", wid, "fallback", "r1")
    assert it["reason"] == A.REASON_NO_CANDIDATE and items[2]["reason"] == F.REASON_BEYOND_CAP
    assert it["generator_version"] == docs[0]["generator_version"] and "bbox" not in it  # plain placement, no suggestion
    # rerun: nothing new, no second job
    again = F.push_fallback(job, docs, client=cl, marks_dir=ds)
    (g2,) = again["groups"]
    assert (g2["created"], g2["added"], g2["skipped_existing"]) == (False, 0, 8)
    assert len(fallback_jobs()) == 1 and len(fallback_jobs()[0]["items"]) == 8


def test_beyond_cap_can_be_left_out(ds, cl, search):
    verdict(ds, TC, 0, "ball")
    _, _, out = pushed(cl, ds, search, include_beyond_cap=False)
    assert out["frames"] == 2


def test_later_frames_are_appended_not_duplicated(ds, cl, search):
    verdict(ds, TC, 0, "ball")
    job = source_job()
    F.push_fallback(job, search(job), client=cl, marks_dir=ds)
    (d,) = search(job)
    more = {"clip": TC, "frame": 30, "window_id": d["window_id"], "reason": A.REASON_NO_PATH}
    docs = [{**d, "unreachable": [*d["unreachable"], more]}]  # a later run found frame 30 unreachable too
    out = F.push_fallback(job, docs, client=cl, marks_dir=ds)
    (g,) = out["groups"]
    assert (g["created"], g["added"], g["skipped_existing"]) == (False, 1, 8)
    (fj,) = fallback_jobs()
    assert sorted(it["frame"] for it in fj["items"]) == [10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert len(fj["items"]) == len({it["frame"] for it in fj["items"]})


def test_a_finished_fallback_job_reports_closed_frames(ds, cl, search):
    verdict(ds, TC, 0, "ball")
    job, docs, out = pushed(cl, ds, search, job=source_job(frames=[0, 10]))
    jobs.close(out["groups"][0]["job"], "done")
    docs2 = [{**docs[0], "unreachable": [*docs[0]["unreachable"], {"clip": TC, "frame": 20, "window_id": docs[0]["window_id"], "reason": "x"}]}]
    job2 = source_job(frames=[0, 10, 20])
    g = F.push_fallback(job2, docs2, client=cl, marks_dir=ds)["groups"][0]
    assert g["closed"] == 1 and g["added"] == 0 and len(fallback_jobs()[0]["items"]) == 1


def test_unsure_twice_goes_to_the_check(ds, cl, search):
    verdict(ds, TC, 0, "ball")
    verdict(ds, TC, 30, "unsure", ts=1.0)
    verdict(ds, TC, 30, "unsure", ts=2.0)
    verdict(ds, TC, 20, "unsure", ts=1.0)  # once only: a card is still due
    _, _, out = pushed(cl, ds, search, include_beyond_cap=False)
    assert out["frames"] == 2  # 20's single unsure leaves a card due; 10 unreachable; 30 unsure twice
    reasons = {it["frame"]: it["reason"] for it in fallback_jobs()[0]["items"]}
    assert reasons == {10: A.REASON_NO_CANDIDATE, 30: F.REASON_UNSURE_X2}


def test_terminal_and_anchored_frames_are_not_sent(ds, cl, search, store):
    verdict(ds, TC, 0, "ball")
    store.write_ball_terminal_state(TC, 10, "ball", method="verdict")
    place(ds, TC, 20)
    _, _, out = pushed(cl, ds, search, store=store, include_beyond_cap=False)
    # 10 is terminal, 20 is anchored; the anchor at 20 brings 40 / 50 into hop range with no candidate -> those two only
    assert [it["frame"] for it in fallback_jobs()[0]["items"]] == [40, 50] and out["frames"] == 2


def test_no_anchor_window_makes_no_fallback_job(ds, cl, search):
    job = source_job()
    out = F.push_fallback(job, search(job), client=cl, marks_dir=ds)  # no seed -> no search doc -> nothing claimed
    assert out["frames"] == 0 and fallback_jobs() == []


def test_dry_run_creates_nothing(ds, cl, search):
    verdict(ds, TC, 0, "ball")
    _, _, out = pushed(cl, ds, search, dry_run=True)
    assert out["groups"][0]["offered"] == 8 and fallback_jobs() == []


# ---- namespaces and the split guard ------------------------------------------------------------
def test_eval_family_has_its_own_namespace_one_job_per_panel_match(ds, cl, search):
    verdict(ds, HC, 0, "ball")
    job = source_job(HC, "ball-eval/p1/hm_a")
    docs = search(job, clip=HC)
    out = F.push_fallback(job, docs, client=cl, marks_dir=ds)
    assert out["family"] == "eval" and [g["key"] for g in out["groups"]] == ["ball-eval-place/p1/hm_a"]
    (fj,) = fallback_jobs()
    assert {it["split"] for it in fj["items"]} == {"eval_heldout_match"} and not fj["key"].startswith("ball-place/")


def test_family_must_agree_with_split():
    with pytest.raises(A.SplitError):
        F.family_of(source_job(HC, ROUND))
    with pytest.raises(A.SplitError):
        F.family_of(source_job(TC, "ball-eval/p1/hm_a"))
    assert F.family_of(source_job(TC)).key("w1") == "ball-place/w1"


def test_frames_outside_F_or_split_are_refused_and_no_job_is_made(ds, cl, search):
    verdict(ds, TC, 0, "ball")
    job = source_job()
    doc = search(job)[0]
    for bad in ({"clip": TC, "frame": 500}, {"clip": HC, "frame": 10}):
        evil = {**doc, "unreachable": [{**bad, "window_id": doc["window_id"], "reason": "x"}]}
        with pytest.raises(A.SplitError):
            F.push_fallback(job, [evil], client=cl, marks_dir=ds)
    assert fallback_jobs() == []


# ---- consuming the result ----------------------------------------------------------------------
def fallback_results(cl, ds, search):
    verdict(ds, TC, 0, "ball")
    job, docs, out = pushed(cl, ds, search, include_beyond_cap=False)
    return job, docs, out["groups"][0]["job"]


def test_placement_writes_terminal_ball_and_reseeds_the_search(ds, cl, search, store):
    job, docs, fid = fallback_results(cl, ds, search)
    d0 = docs[0]
    place(ds, TC, 20, 0.62, 0.3)  # the human placed the ball on an unreachable frame
    out = F.ingest_results(cl.results(fid), store)
    assert out["ball"] == [(TC, 20)] and out["undecided"] == 1 and out["written"] == 1
    assert store.read_ball_terminal_states(TC, [10, 20]) == {20: "ball"}
    # the box is a new anchor: the next search sees it (sidecar human label of a frame of F) and writes a new version
    assert A.load_anchors(job, marks_dir=ds).anchors[(TC, 20)].source == "human"
    (d1,) = search(job)
    assert d1["digest"] != d0["digest"] and d1["version"] == d0["version"] + 1 and d1["reused"] is False
    assert d1["n_anchors"] == 2
    assert 20 not in [c["frame_index"] for c in d1["cards"]]  # an anchor is judged, no card for it
    (d2,) = F.reseed(job, marks_dir=ds)  # same inputs again: converged
    assert d2["reused"] is True


def test_explicit_no_ball_is_the_only_way_to_no_ball(ds, cl, search, store):
    job, docs, fid = fallback_results(cl, ds, search)
    no_ball(ds, TC, 10)
    out = F.ingest_results(cl.results(fid), store)
    assert out["no_ball"] == [(TC, 10)] and out["ball"] == []
    row = store.read_frame_label_states(TC, "ball").set_index("frame_index").loc[10]
    assert (row["state"], row["method"]) == ("absent", "hand_frame") and row["reason"] is None
    assert store.read_ball_terminal_states(TC, [10, 20]) == {10: "no_ball"}  # 20 untouched: still unreviewed
    assert F.ingest_results(cl.results(fid), store)["written"] == 1  # idempotent re-ingest
    assert len(store.read_frame_label_states(TC, "ball")) == 1


def test_not_broadcast_is_unusable_and_precedence(ds, cl, search, store):
    _, _, fid = fallback_results(cl, ds, search)
    mark(ds, TC, 10, "not_broadcast")
    place(ds, TC, 20)
    no_ball(ds, TC, 20)  # a box and a no_ball mark on one frame: the box wins
    out = F.ingest_results(cl.results(fid), store)
    assert out["unusable"] == [(TC, 10)] and out["ball"] == [(TC, 20)] and out["no_ball"] == []
    r = store.read_frame_label_states(TC, "ball").set_index("frame_index")
    assert (r.loc[10, "state"], r.loc[10, "reason"]) == ("absent", "not_broadcast")


def test_no_candidate_frame_is_never_no_ball_by_any_path(ds, cl, search, store):
    job, docs, fid = fallback_results(cl, ds, search)
    out = F.ingest_results(cl.results(fid), store)  # the human has not looked yet
    assert out["written"] == 0 and out["undecided"] == 2
    assert store.read_ball_terminal_states(TC) == {} and len(store.read_frame_label_states(TC, "ball")) == 0
    # search / push never write the store, and a store given to them stays empty
    F.push_fallback(job, docs, client=cl, marks_dir=ds, store=store)
    assert len(store.read_frame_label_states(TC, "ball")) == 0
    # the store API refuses no_ball without the check, so no other caller can default to it either
    with pytest.raises(ValueError, match="full-frame"):
        store.write_ball_terminal_state(TC, 10, "no_ball", method="hand_frame")


def test_labels_of_frames_outside_the_fallback_items_are_ignored(ds, cl, search, store):
    _, _, fid = fallback_results(cl, ds, search)
    no_ball(ds, TC, 90)  # a frame of the clip that is NOT a fallback item here (beyond cap left out)
    place(ds, TC, 30)  # reachable: has a card, not in this job
    res = cl.results(fid)
    assert F.ingest_results(res, store)["written"] == 0
    assert F.ingest_results(res, store, selected=frozenset({(TC, 10)}))["written"] == 0


def test_ingest_refuses_a_non_fallback_job(ds, cl, store):
    job, _ = jobs.submit({"title": "x", "kind": "ball_place", "dataset": "train", "key": "some/other",
                          "items": [{"clip": TC, "frame": 1}]}, "train")
    with pytest.raises(A.SplitError, match="not a fallback"):
        F.ingest_results(cl.results(job["id"]), store)


def test_ingest_dry_run_writes_nothing(ds, cl, search, store):
    _, _, fid = fallback_results(cl, ds, search)
    no_ball(ds, TC, 10)
    out = F.ingest_results(cl.results(fid), store, dry_run=True)
    assert out["no_ball"] == [(TC, 10)] and out["written"] == 0 and store.read_ball_terminal_states(TC) == {}

