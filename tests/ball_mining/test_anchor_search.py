"""Anchor search inputs (design ``ball_anchor_iteration.md`` task 1): the per-job split guard (also on append), the job scope,
the uniform candidate reader (job_suggest file or pool) and anchors read from the marks dir, all confined to the job's own frames."""

from __future__ import annotations

import json

import pytest

from footy_track import match_splits as ms
from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.ball_mining import schema
from footy_track.labeller import job_edit, jobs
from footy_track.labeller.jobs import JobError

from .conftest import make_pool

HELD, TRAIN = "hm_a", "tm_c"
HC, TC = f"{HELD}_seg001", f"{TRAIN}_seg001"
BOX = {"x": 0.40, "y": 0.50, "w": 0.02, "h": 0.04}  # centre (0.41, 0.52)


@pytest.fixture(autouse=True)
def register_splits(tmp_path, monkeypatch):
    reg = tmp_path / "match_splits.json"
    reg.write_text(json.dumps({HELD: {"split": "heldout", "aliases": []}, TRAIN: {"split": "train", "aliases": []}}))
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(reg))
    monkeypatch.setenv("FOOTY_DATA", str(tmp_path))
    ms.clear_cache()
    yield
    ms.clear_cache()


def place_job(clip, jid="j1", starts=(0, 100), dataset="train"):
    return {"id": jid, "kind": "ball_place", "dataset": dataset,
            "items": [{"clip": clip, "frame_start": s, "frame_end": s + 20, "step": 10} for s in starts]}


def check_job(clip, frames, jid="r1"):
    return {"id": jid, "kind": "ball_check", "dataset": "train",
            "items": [{"clip": clip, "frame_index": f, "box_index": 0, "source": "list", "bbox": dict(BOX)} for f in frames]}


def verdict(marks, clip, frame, v, bbox=BOX, ts=1.0, **kw):
    d = marks / "ball_checks"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"clip": clip, "frame_index": frame, "box_index": 0, "source": "list", "verdict": v, "bbox": bbox, "ts": ts, **kw}
    with (d / f"{clip}.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ---- split guard ------------------------------------------------------------------------------
def test_job_split_names_the_one_split():
    assert A.job_split(place_job(HC)) == "eval_heldout_match"
    assert A.job_split(place_job(TC)) == "train"
    assert A.job_split({"id": "e", "kind": "ball_place", "items": []}) is None


def test_mixed_job_is_refused():
    job = place_job(HC)
    job["items"].append({"clip": TC, "frame_start": 0, "frame_end": 10, "step": 5})
    with pytest.raises(A.SplitError, match="more than one split"):
        A.job_split(job)
    with pytest.raises(A.SplitError):
        A.job_scope(job)


def test_check_append_split_rules():
    job = place_job(HC)
    assert A.check_append_split(job, [{"clip": HC}]) == "eval_heldout_match"
    with pytest.raises(A.SplitError, match="refusing to append"):
        A.check_append_split(job, [{"clip": TC}])
    with pytest.raises(A.SplitError, match="appended items span"):
        A.check_append_split(place_job(HC, starts=()), [{"clip": HC}, {"clip": TC}])
    assert A.check_append_split(place_job(HC, starts=()), [{"clip": TC}]) == "train"  # empty job takes the new split
    mixed = place_job(HC)
    mixed["items"].append({"clip": TC, "frame_start": 0, "frame_end": 10, "step": 5})
    with pytest.raises(A.SplitError, match="already spans"):
        A.check_append_split(mixed, [{"clip": HC}])


def test_append_items_rejects_cross_split_and_writes_nothing(ds):
    job, _ = jobs.submit({"title": "r", "kind": "ball_check", "dataset": "train", "key": "k1", "max_cards": 100,
                          "items": check_job(HC, [10])["items"]}, "train")
    bad = check_job(TC, [20])["items"]
    with pytest.raises(JobError) as e:
        job_edit.append_items(job["id"], bad)
    assert e.value.status == 422 and "split guard" in str(e.value)
    assert [it["frame_index"] for it in jobs.read_job(job["id"])["items"]] == [10]
    assert jobs.read_job(job["id"]).get("rev", 0) == 0  # untouched
    ok = job_edit.append_items(job["id"], check_job(HC, [20])["items"])  # same split still appends
    assert ok["added"] == 1


def test_append_items_rejects_mixed_batch(ds):
    job, _ = jobs.submit({"title": "r", "kind": "ball_check", "dataset": "train", "key": "k2", "max_cards": 100,
                          "items": check_job(TC, [10])["items"]}, "train")
    with pytest.raises(JobError):
        job_edit.append_items(job["id"], [*check_job(TC, [20])["items"], *check_job(HC, [30])["items"]])
    assert len(jobs.read_job(job["id"])["items"]) == 1


# ---- scope -------------------------------------------------------------------------------------
def test_job_scope_frames_for_both_kinds():
    s = A.job_scope(place_job(TC))
    assert s.split == "train" and s.clips == {TC}
    assert s.frames == {(TC, f) for st in (0, 100) for f in (st, st + 10, st + 20)}
    assert A.job_scope(check_job(HC, [5, 7])).frames == {(HC, 5), (HC, 7)}


# ---- candidates --------------------------------------------------------------------------------
def suggest_doc(job_id, clip=TC, frames=(0, 10), extra=None):
    boxes = {f"{clip}:{f}": {"x": 0.3, "y": 0.4, "w": 0.02, "h": 0.04, "conf": 0.9, "tid": 1,
                             "cands": [{"x": 0.3, "y": 0.4, "w": 0.02, "h": 0.04, "conf": 0.5, "tid": 2},
                                       {"x": 0.6, "y": 0.4, "w": 0.02, "h": 0.04, "conf": 0.9, "tid": 1}]} for f in frames}
    boxes[f"{clip}:20"] = None
    boxes.update(extra or {})
    return {"schema": "job-suggest/v1", "job_id": job_id, "detector": {"sha8": "abcd1234"}, "boxes": boxes}


def test_suggest_doc_candidates_are_centres_best_first_and_scoped():
    scope = A.job_scope(place_job(TC))
    doc = suggest_doc("j1", extra={f"{TC}:999": {"x": 0, "y": 0, "w": 0.1, "h": 0.1, "conf": 1.0}})  # frame outside F
    cs = A.candidates_from_suggest_doc(doc, scope)
    assert cs.source == "job_suggest" and cs.detector == "abcd1234"
    assert set(cs.by_frame) == {(TC, 0), (TC, 10)}  # null entry and out-of-F frame dropped
    first = cs.get(TC, 0)
    assert [c.conf for c in first] == [0.9, 0.5] and first[0].tid == 1
    assert first[0].cx == pytest.approx(0.61) and first[0].cy == pytest.approx(0.42)


def test_suggest_doc_of_another_job_is_refused():
    with pytest.raises(A.SplitError, match="belongs to job"):
        A.candidates_from_suggest_doc(suggest_doc("other"), A.job_scope(place_job(TC)))


def test_pool_candidates_use_the_same_interface_and_only_job_frames():
    pool = make_pool([
        {"clip": TC, "frame": 10, "x": 0.2, "y": 0.3, "p_real": 0.4, "track_id": 7},
        {"clip": TC, "frame": 10, "x": 0.7, "y": 0.3, "p_real": 0.8},
        {"clip": TC, "frame": 55, "x": 0.5, "y": 0.5},  # not a frame of the job
        {"clip": HC, "frame": 10, "x": 0.5, "y": 0.5, "match": HELD},  # another split's clip
    ])
    cs = A.candidates_from_pool(pool, A.job_scope(place_job(TC)))
    assert cs.source == "pool" and set(cs.by_frame) == {(TC, 10)}
    got = cs.get(TC, 10)
    assert [c.conf for c in got] == pytest.approx([0.8, 0.4]) and [c.tid for c in got] == [0, 7]
    assert (got[1].cx, got[1].cy) == pytest.approx((0.2, 0.3))


def test_load_candidates_reads_the_jobs_own_suggest_file(tmp_path):
    job = place_job(TC)
    assert A.load_candidates(job, root=tmp_path).by_frame == {}  # nothing generated yet
    d = tmp_path / "job_suggestions"
    d.mkdir()
    (d / "j1.suggest.abcd1234.json").write_text(json.dumps(suggest_doc("j1")))
    (d / "j2.suggest.abcd1234.json").write_text(json.dumps(suggest_doc("j2", frames=(0, 10, 20))))
    assert set(A.load_candidates(job, root=tmp_path).by_frame) == {(TC, 0), (TC, 10)}


def test_load_candidates_from_pool_file(tmp_path):
    p = schema.write_pool(make_pool([{"clip": TC, "frame": 0, "p_real": 0.3}]), tmp_path / "pool.parquet")
    cs = A.load_candidates(place_job(TC), pool_path=p)
    assert cs.source == "pool" and len(cs.get(TC, 0)) == 1


# ---- anchors -----------------------------------------------------------------------------------
def test_anchors_and_forbidden_from_verdicts(ds):
    job = check_job(TC, [10, 20, 30, 40, 50])
    verdict(ds, TC, 10, "ball")
    verdict(ds, TC, 20, "corrected", corrected_bbox={"x": 0.1, "y": 0.1, "w": 0.02, "h": 0.02})
    verdict(ds, TC, 30, "not_ball")
    verdict(ds, TC, 40, "unsure")
    verdict(ds, TC, 50, "ball")
    verdict(ds, TC, 50, None, ts=2.0)  # tombstone: undone
    got = A.load_anchors(job)
    assert set(got.anchors) == {(TC, 10), (TC, 20)}
    assert (got.anchors[(TC, 10)].cx, got.anchors[(TC, 10)].cy) == pytest.approx((0.41, 0.52))
    assert (got.anchors[(TC, 20)].cx, got.anchors[(TC, 20)].cy) == pytest.approx((0.11, 0.11))
    assert [f.frame for f in got.forbidden[(TC, 30)]] == [30] and got.forbidden[(TC, 30)][0].w == 0.02
    assert (TC, 40) not in got.anchors and (TC, 40) not in got.forbidden and (TC, 50) not in got.anchors


def test_anchor_box_falls_back_to_the_card_box(ds):
    verdict(ds, TC, 10, "ball", bbox=None)
    assert (A.load_anchors(check_job(TC, [10])).anchors[(TC, 10)].cx) == pytest.approx(0.41)


def test_anchors_from_human_sidecar_win_over_verdict(ds):
    (ds / f"{TC}.jsonl").write_text(json.dumps({"frame_index": 0, "center": {"x": 0.7, "y": 0.8}, "tags": ["in_play_ball", "labeller"]}) + "\n")
    verdict(ds, TC, 0, "ball")
    verdict(ds, TC, 10, "ball")
    got = A.load_anchors(place_job(TC))
    assert (got.anchors[(TC, 0)].cx, got.anchors[(TC, 0)].source) == (pytest.approx(0.7), "human")
    assert got.anchors[(TC, 10)].source == "verdict"


def test_anchors_never_cross_jobs_or_splits(ds):
    verdict(ds, TC, 10, "ball")  # in job A's frames
    verdict(ds, TC, 500, "ball")  # same clip, frame in another job's set
    verdict(ds, TC, 600, "not_ball")
    verdict(ds, HC, 10, "ball")  # other split's clip, same frame number
    (ds / f"{HC}.jsonl").write_text(json.dumps({"frame_index": 10, "center": {"x": 0.5, "y": 0.5}, "tags": ["in_play_ball", "labeller"]}) + "\n")
    job_a = place_job(TC, jid="A", starts=(0,))
    got = A.load_anchors(job_a)
    assert set(got.anchors) == {(TC, 10)} and got.forbidden == {}
    assert set(A.load_anchors(place_job(TC, jid="B", starts=(500, 600))).anchors) == {(TC, 500)}
    assert set(A.load_anchors(place_job(HC, jid="C", starts=(0,))).anchors) == {(HC, 10)}
