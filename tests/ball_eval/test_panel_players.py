"""Panel player pre-fill (footy_track.ball_eval.panel_players): machine rows merged into the sidecar without touching
George's ball boxes or markers, idempotent, skips frames that are his, and the ``player_check`` job body."""

from __future__ import annotations

import json

import pytest

from footy_track.ball_eval import panel as P
from footy_track.ball_eval import panel_players as PP

from .test_panel import HELD, build, fd  # noqa: F401  (fd is a fixture)


def _l(frame, tags, box=True):
    return json.dumps(
        {
            "frame_index": frame,
            "bbox": {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.02} if box else None,
            "center": {"x": 0.505, "y": 0.51} if box else None,
            "tags": tags,
        }
    )


def _b(label="player", x=0.2):
    return {"label": label, "x": x, "y": 0.3, "w": 0.03, "h": 0.1, "conf": 0.9}


BALL = _l(10, ["in_play_ball", "labeller", "accepted_suggestion", "suggest_det:abcd1234"])
NO_BALL = _l(20, ["no_ball"], box=False)
NOT_BC = _l(30, ["not_broadcast"], box=False)
HUMAN_PLAYER = _l(40, ["player", "labeller"])
COMPLETE = _l(50, ["players_complete"], box=False)
LINES = [BALL, NO_BALL, NOT_BC, HUMAN_PLAYER, COMPLETE]


def test_merge_appends_machine_rows_and_keeps_every_existing_line_byte_for_byte():
    new, rep = PP.merge_prefill(LINES, {10: [_b(), _b("referee", 0.4)], 20: [_b()]})
    assert new[: len(LINES)] == LINES
    added = [json.loads(x) for x in new[len(LINES) :]]
    assert len(added) == 3 and rep["added"] == 3 and rep["replaced"] == 0
    assert {tuple(a["tags"]) for a in added} == {("player", "yolo", "prefill:rtdetr"), ("referee", "yolo", "prefill:rtdetr")}
    assert rep["per_frame"] == {"10": {"player": 1, "referee": 1}, "20": {"player": 1}}


def test_frames_that_are_not_broadcast_hand_labelled_or_complete_are_skipped_untouched():
    new, rep = PP.merge_prefill(LINES, {30: [_b()], 40: [_b()], 50: [_b()], 60: [_b()]})
    assert rep["skipped"] == {"30": "not_broadcast", "40": "hand_player_box", "50": "players_complete"}
    assert len(new) == len(LINES) + 1 and json.loads(new[-1])["frame_index"] == 60


def test_rerun_replaces_only_its_own_rows_and_is_idempotent():
    once, _ = PP.merge_prefill(LINES, {10: [_b(), _b(x=0.6)]})
    again, rep = PP.merge_prefill(once, {10: [_b(x=0.7)]})
    assert rep["replaced"] == 2 and rep["added"] == 1
    assert again[: len(LINES)] == LINES
    assert PP.merge_prefill(again, {10: [_b(x=0.7)]})[0] == again  # same detections: byte-identical


def test_machine_rows_are_not_ground_truth_for_the_labeller_or_ingest():
    from footy_track.feature_store.ingest_gt import _split_tags  # noqa: PLC0415
    from footy_track.labeller.constants import PROVENANCE_TAGS  # noqa: PLC0415

    tags = json.loads(PP.box_line(5, "player", _b()))["tags"]
    assert _split_tags(tags) == ("player", "yolo", False)
    assert next(t for t in tags if t in PROVENANCE_TAGS) == "yolo"


def test_write_prefill_is_atomic_and_noop_when_nothing_changes(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(LINES) + "\n")
    PP.write_prefill(p, {10: [_b()]})
    assert p.read_text().splitlines()[: len(LINES)] == LINES and len(p.read_text().splitlines()) == len(LINES) + 1
    m = p.stat().st_mtime_ns
    PP.write_prefill(p, {10: [_b()]})
    assert p.stat().st_mtime_ns == m
    assert not list(tmp_path.glob("*.tmp"))


def test_prefill_match_covers_window_frames_of_main_clips_only_not_qa_aliases(fd, tmp_path):  # noqa: F811
    doc = build(fd)
    P.write_panel(doc, root=fd)
    marks, clips = tmp_path / "marks", tmp_path / "clips"
    marks.mkdir(), clips.mkdir()
    want = {w["clip"] for w in doc["windows"]}
    for c in want | {q["clip"] for q in doc["qa"]}:
        (marks / f"{c}.jsonl").write_text(BALL + "\n")
        (clips / f"{c}.mp4").write_bytes(b"x")
    seen = {}

    def detect(clip, video, frames):
        seen[clip] = frames
        return {f: [_b(), _b("referee")] for f in frames}

    rep = PP.prefill_match(HELD, marks_dir=marks, clips_dir=clips, detect=detect, root=fd, log=lambda *_: None)
    assert set(seen) == want and not any(c.endswith("__b") for c in seen)
    assert rep["frames"] == doc["n_frames"] and rep["players"] == rep["referees"] == doc["n_frames"]
    for c in want:
        assert BALL in (marks / f"{c}.jsonl").read_text()
    for q in doc["qa"]:
        assert (marks / f"{q['clip']}.jsonl").read_text() == BALL + "\n"


def test_prefill_refuses_a_non_heldout_match(fd):  # noqa: F811
    with pytest.raises(P.PanelError):
        PP.prefill_match("tm_c", marks_dir=fd, clips_dir=fd, detect=lambda *a: {}, root=fd)


def test_job_body_is_one_eval_item_per_window(fd):  # noqa: F811
    doc = build(fd)
    b = PP.job_body(doc)
    assert b["kind"] == "player_check" and b["dataset"] == "ball_eval_v1" and len(b["items"]) == doc["n_windows"]
    assert all(i["eval"] is True and i["frame_end"] >= i["frame_start"] for i in b["items"])
    assert b["key"] == f"player-check/{doc['panel_version']}/{HELD}"
