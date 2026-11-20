"""Accepting a detector suggestion (LAB-1446): the page saves it as an ordinary human box with the audit marker
``accepted_suggestion`` (+ ``suggest_det:<sha8>``); the marker survives /edit -> flush -> reload (T-01 class of bug);
the endpoint never offers a suggestion on blind cards. Page wiring is checked by string (UNTESTED-FRONTEND)."""

from __future__ import annotations

import json
from pathlib import Path

from footy_track.feature_store.ingest_gt import _split_tags
from footy_track.labeller import jobs
from footy_track.labeller.server import Session

INDEX_HTML = Path(__file__).resolve().parents[2] / "src/footy_track/labeller/web/index.html"


class _Stem:
    def __init__(self, stem: str) -> None:
        self.stem = stem


def _session(s: Session, stem="clip_s", n=10):
    s.video_path = _Stem(stem)
    s.total_frames = n
    s.timeline = [None] * n
    s.no_ball_frames = set()
    s.not_broadcast_frames = set()
    s._box_markers = {}
    return s


def _obj(markers, x=0.3):
    return {"label": "in_play_ball", "model": "labeller", "conf": 1.0, "x": x, "y": 0.4, "w": 0.01, "h": 0.02, "markers": markers}


def _rows(d, stem="clip_s"):
    return [json.loads(x) for x in (d / f"{stem}.jsonl").read_text().splitlines()]


def test_accepted_marker_is_flushed_survives_reload_and_unrelated_edits(client, fresh_session, gt_marks_dir):
    s = _session(fresh_session)
    tags = ["accepted_suggestion", "suggest_det:deadbeef"]
    assert client.post("/edit", json={"idx": 4, "objects": [_obj(tags)]}).status_code == 200
    s._do_flush()
    assert _rows(gt_marks_dir)[0]["tags"] == ["in_play_ball", "labeller", *tags]  # tier-1 labeller GT + audit markers

    # reload the sidecar into a fresh session, edit another frame, flush again: the marker is still there
    s2 = _session(Session(), n=10)
    s2.dataset = s.dataset
    s2._load_existing_marks()
    s2.set_frame(6, [])
    s2._do_flush()
    assert _rows(gt_marks_dir)[0]["tags"] == ["in_play_ball", "labeller", *tags]
    # the page gets the markers back from /timeline and returns them unchanged on the next save
    boxes = client.get("/timeline/4").json()["boxes"]
    assert boxes[0]["source"] == "labeller" and boxes[0]["markers"] == tags
    # a moved box is a hand edit: no marker (the page clears it; here the geometry change alone drops it)
    assert client.post("/edit", json={"idx": 4, "objects": [_obj([], x=0.35)]}).status_code == 200
    s._do_flush()
    assert _rows(gt_marks_dir)[0]["tags"] == ["in_play_ball", "labeller"]


def test_edit_only_accepts_the_suggestion_marker_tags(client, fresh_session, gt_marks_dir):
    s = _session(fresh_session)
    bad = ["human_checked", "accepted_suggestion", "labeller", "evil", "suggest_det:" + "x" * 80]
    assert client.post("/edit", json={"idx": 2, "objects": [_obj(bad)]}).status_code == 200
    s._do_flush()
    assert _rows(gt_marks_dir)[0]["tags"] == ["in_play_ball", "labeller", "accepted_suggestion"]


def test_marker_tags_are_not_an_object_class_or_provenance_in_ingest():
    assert _split_tags(["in_play_ball", "labeller", "accepted_suggestion", "suggest_det:deadbeef"]) == ("in_play_ball", "labeller", False)
    assert _split_tags(["accepted_suggestion", "labeller"]) == (None, "labeller", False)


def test_job_counts_an_accepted_suggestion_as_a_human_label(gt_marks_dir):
    p = gt_marks_dir / "c.jsonl"
    row = {"frame_index": 3, "bbox": {"x": 0.1, "y": 0.1, "w": 0.01, "h": 0.01}, "center": {"x": 0.1, "y": 0.1}, "tags": ["in_play_ball", "labeller", "accepted_suggestion"]}
    p.write_text(json.dumps(row) + "\n")
    got = jobs._parse_sidecar(p)[3][0]
    assert got["status"] == "ball" and got["tier"] == 1 and "accepted_suggestion" in got["tags"]


def test_page_wiring():
    html = INDEX_HTML.read_text()
    for needle in (
        'id="jobAcceptBtn"',
        "Accept suggestion (Y)",
        '(e.key === "y" || e.key === "Y")',
        "const sugLayer = new Konva.Layer",
        "/suggestions`",
        'const SUG_TAG = "accepted_suggestion"',
        "markers: r.getAttr(\"markers\") || []",
    ):
        assert needle in html, needle
    # the overlay lives outside boxLayer, so a save can never turn it into a label
    assert "sugLayer.add(" in html and 'boxLayer.add(new Konva.Rect({ x, y, width: w' not in html


def test_accept_on_a_non_panel_job_frame_writes_marker_and_survives_reload(client, fresh_session, gt_marks_dir):
    """LAB-1479: the marker path does not depend on the job being an eval panel."""
    s = _session(fresh_session)
    tags = ["accepted_suggestion", "suggest_det:cafef00d"]
    assert client.post("/edit", json={"idx": 3, "objects": [_obj(tags)]}).status_code == 200
    s._do_flush()
    s2 = _session(Session(), n=10)
    s2.dataset = s.dataset
    s2._load_existing_marks()
    s2.set_frame(5, [])
    s2._do_flush()
    assert _rows(gt_marks_dir)[0]["tags"] == ["in_play_ball", "labeller", *tags]


def test_rank_marker_survives_and_numbered_key_wiring():
    from footy_track.labeller.constants import is_suggest_marker

    assert is_suggest_marker("suggest_rank:2") and not is_suggest_marker("suggest_rank:9") and not is_suggest_marker("suggest_rank:")
    assert _split_tags(["in_play_ball", "labeller", "accepted_suggestion", "suggest_rank:2"]) == ("in_play_ball", "labeller", False)
    html = INDEX_HTML.read_text()
    for needle in ("const SUG_SHOW = 3", "/^[1-3]$/.test(e.key)", '"suggest_rank:" + rank', 'job.kind === "ball_place"', "performance.measure(\"sug-latency\""):
        assert needle in html, needle
    assert html.index("/^[1-3]$/.test(e.key)") < html.index('const classKeys = {"1"')  # job-mode rebind wins; 4-6 stay class keys
