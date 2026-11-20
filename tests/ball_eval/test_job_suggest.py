"""Job-keyed detector suggestions for non-panel ball_place jobs (LAB-1479): no blind control, pre-labelled frames left
alone, heldout gate, report; and an accept on such a job saves the marker and survives a reload."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from footy_track.ball_eval import job_suggest as J
from footy_track.ball_eval import panel as P
from footy_track.ball_eval import panel_suggest as PS
from footy_track.labeller import datasets, jobs, suggestions

from .test_panel import HELD, TRAIN, fd  # noqa: F401
from .test_panel_suggest import FakeDet

CLIP = f"{HELD}_seg001"


def _job(clip=CLIP):
    items = [{"clip": clip, "frame_start": s, "frame_end": s + 20, "step": 10, "stratum": st}
             for s, st in ((0, "easy"), (100, "easy"), (200, "gappy"), (300, "gappy"))]
    return {"id": "j-test", "kind": "ball_place", "dataset": "ds", "items": items}


@pytest.fixture
def env(fd, monkeypatch, tmp_path):  # noqa: F811
    job = _job()
    marks = tmp_path / "marks"
    marks.mkdir()
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / f"{CLIP}.mp4").write_bytes(b"x")
    ds = SimpleNamespace(name="ds", marks_dir=marks, clips_dir=clips, extra_clips_dirs=())
    monkeypatch.setattr(datasets, "get", lambda name=None: ds)
    monkeypatch.setattr(jobs, "read_job", lambda jid: job if jid == job["id"] else None)
    monkeypatch.setattr(PS, "_read_frames", lambda video, wanted: ((f, object()) for f in wanted))
    # frame 0 labelled before the file exists (hand label), frame 10 labelled as an accept
    row = lambda f, tags: json.dumps({"frame_index": f, "center": {"x": 0.5, "y": 0.6}, "tags": ["in_play_ball", "labeller", *tags]})  # noqa: E731
    (marks / f"{CLIP}.jsonl").write_text(row(0, []) + "\n")
    w = fd / "w.pt"
    w.write_bytes(b"weights")
    return fd, job, marks, w


def test_suggest_job_covers_every_frame_and_serves_box_except_pre_labelled(env):
    fd, job, marks, w = env
    r = J.suggest_job("j-test", root=fd, weights=w, detector=FakeDet(), log=lambda m: None)
    doc = json.loads(J.find_suggest_file("j-test", fd).read_text())
    frames = {f"{CLIP}:{s + k}" for s in (0, 100, 200, 300) for k in (0, 10, 20)}
    assert set(doc["boxes"]) == frames and doc["pre_labelled"] == [f"{CLIP}:0"]
    assert r["n_pre_labelled"] == 1 and "blind_windows" not in doc
    served = [(f, suggestions.suggestion_for(job, CLIP, f, root=fd)) for f in (0, 10, 20, 110)]
    for f, res in served:
        box = doc["boxes"][f"{CLIP}:{f}"]
        if f == 0 or box is None:
            assert res["suggestion"] is None  # pre-labelled frame / no candidate: nothing offered
        else:
            assert res["suggest"] == "shown" and res["suggestion"]["conf"] == 0.9 and res["suggestion"]["detector"] == doc["detector"]["sha8"]
    assert any(res["suggestion"] for _, res in served)
    cov = J.coverage(job, doc)
    assert set(cov) == {"easy", "gappy"} and all(c["frames"] == 6 for c in cov.values())


def test_no_suggest_file_or_other_kind_answers_null(env):
    fd, job, *_ = env
    assert suggestions.suggestion_for(job, CLIP, 10, root=fd)["suggestion"] is None
    assert suggestions.suggestion_for({**job, "kind": "frame_type"}, CLIP, 10, root=fd)["suggestion"] is None
    assert suggestions.suggestion_for(job, CLIP, 11, root=fd)["suggestion"] is None  # frame not in the job


def test_non_heldout_job_is_refused(env, monkeypatch):
    fd, job, _, w = env
    job["items"][0]["clip"] = f"{TRAIN}_seg001"
    with pytest.raises(P.PanelError):
        J.suggest_job("j-test", root=fd, weights=w, detector=FakeDet(), log=lambda m: None)


def test_report_counts_accepts_per_stratum(env):
    fd, job, marks, w = env
    J.suggest_job("j-test", root=fd, weights=w, detector=FakeDet(), log=lambda m: None)
    rows = [{"frame_index": 10, "center": {"x": 0.5, "y": 0.6}, "bbox": {"x": 0.5, "y": 0.6, "w": 0, "h": 0}, "tags": ["in_play_ball", "labeller", "accepted_suggestion"]},
            {"frame_index": 0, "center": {"x": 0.1, "y": 0.1}, "tags": ["in_play_ball", "labeller"]}]
    (marks / f"{CLIP}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rep = J.report("j-test", root=fd)
    e = rep["strata"]["easy"]
    assert e["ball_labels"] == 2 and e["accepted"] == 1 and e["share_accepted"] == 0.5
    assert rep["strata"]["ALL"]["ball_labels"] == 2 and "blind" not in json.dumps(rep)


def test_panel_path_is_unchanged(fd):  # noqa: F811
    from .test_panel import build

    d = build(fd)
    job = {"id": "j-panel", "kind": "ball_place", "items": P.job_items(d)}
    w = d["windows"][0]
    res = suggestions.suggestion_for(job, w["clip"], w["frames"][0], root=fd)
    assert res["suggest"] in ("blind", "shown") and res["suggestion"] is None  # no panel suggest file: unchanged


def test_topk_entries_and_tracklets_and_job_payload(env):
    from footy_track.ball_eval import tracklets as T

    fd, job, *_ = env
    J.suggest_job("j-test", root=fd, weights=fd / "w.pt", detector=FakeDet(), log=lambda m: None)
    doc = json.loads(J.find_suggest_file("j-test", fd).read_text())
    e = next(v for v in doc["boxes"].values() if v)
    assert e["cands"][0]["conf"] == e["conf"] == 0.9 and len(e["cands"]) == 2 and e["tid"] == e["cands"][0]["tid"]  # top-1 first, old keys kept
    assert doc["size"] and all("tid" in c for v in doc["boxes"].values() if v for c in v["cands"])
    pay = suggestions.candidates_for_job(job, fd)
    assert pay["frames"] and f"{CLIP}:0" not in pay["frames"]  # pre-labelled frame is not offered
    row = next(iter(pay["frames"].values()))
    assert len(row) == 2 and len(row[0]) == 6 and pay["size"][CLIP] == doc["size"]
    # the same object keeps one tid across the window's consecutive frames (FakeDet's boxes do not move)
    tids = {frozenset(c["tid"] for c in doc["boxes"][f"{CLIP}:{f}"]["cands"]) for f in (10, 20) if doc["boxes"].get(f"{CLIP}:{f}")}
    assert tids  # smoke
    assert T.add_tracklets({}, [], (1920, 1080)) == 1


def test_tracklet_linking_speed_gate():
    from footy_track.ball_eval import tracklets as T

    c = lambda x, y: {"x": x, "y": y, "w": 0.01, "h": 0.01, "conf": 0.5}  # noqa: E731
    boxes = {"c:0": {"cands": [c(0.10, 0.5), c(0.80, 0.5)]}, "c:10": {"cands": [c(0.81, 0.5), c(0.11, 0.5), c(0.5, 0.1)]}}
    T.add_tracklets(boxes, [[("c", 0), ("c", 10)]], (1920, 1080))
    a, b = boxes["c:0"]["cands"], boxes["c:10"]["cands"]
    assert a[0]["tid"] == b[1]["tid"] and a[1]["tid"] == b[0]["tid"]  # nearest neighbour, not conf order
    assert b[2]["tid"] not in (a[0]["tid"], a[1]["tid"])  # 650 px gate at gap 10: a far jump starts a new tracklet


def test_panel_recompute_with_tag_writes_a_new_file_and_keeps_pre_labelled(fd, monkeypatch):  # noqa: F811
    from .test_panel_suggest import _suggest

    d, r, det = _suggest(fd, monkeypatch, labelled=lambda: {"a:1"})
    old = P.suggest_path(HELD, r["detector_sha8"], root=fd)
    before = old.read_bytes()
    r2 = PS.suggest_match(HELD, root=fd, weights=fd / "weights.pt", detector=FakeDet(), tag="k5", labelled_keys=lambda: {"zzz:9"}, log=lambda m: None)
    new = P.suggest_path(HELD, r["detector_sha8"] + "k5", root=fd)
    assert r2["status"] == "created" and new.is_file() and old.read_bytes() == before
    assert r2["pre_labelled"] == ["a:1"] and P.find_suggest_file(HELD, root=fd) == new
    assert next(v for v in json.loads(new.read_text()).values() if v)["cands"]


def test_payload_carries_existing_human_ball_boxes_as_anchors(env):
    fd, job, marks, w = env
    J.suggest_job("j-test", root=fd, weights=w, detector=FakeDet(), log=lambda m: None)
    pay = suggestions.candidates_for_job(job, fd)
    assert pay["human"] == {f"{CLIP}:0": [0.5, 0.6]}  # the hand label of the fixture sidecar, compact normalised centre
    (marks / f"{CLIP}.jsonl").write_text("")
    assert suggestions.candidates_for_job(job, fd)["human"] == {}
