"""GET /progress/labelling: counts from a fixture feature store (no real data)."""

from __future__ import annotations

import json

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from footy_track import match_splits, splits
from footy_track.feature_store.schema import (
    ClsGroup,
    FrameLabelStateRow,
    LabelMethod,
    LabelState,
)
from footy_track.feature_store.store import FeatureStore
from footy_track.labeller import labelling_progress, progress_routes

B, P = ClsGroup.BALL, ClsGroup.PLAYER
C, A, PART = LabelState.COMPLETE, LabelState.ABSENT, LabelState.PARTIAL


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FOOTY_DATA", str(root))
    monkeypatch.setenv("FOOTY_FEATURE_STORE", str(root / "fs.duckdb"))
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(root / "match_splits.json"))
    monkeypatch.setenv("FOOTY_CLIP_SPLITS", str(root / "clip_splits.json"))
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED", raising=False)
    (root / "match_splits.json").write_text(json.dumps({"h1": {"split": "heldout"}, "t1": {"split": "train"}}))
    (root / "clip_splits.json").write_text(json.dumps({
        "version": 1, "rule": {"modulus": 5, "eval_remainder": 0},
        "clips": {"t1_seg001": {"split": "eval"}, "t1_seg002": {"split": "train"}},
    }))
    monkeypatch.setattr(labelling_progress, "_default_dirs", lambda: (None, None))
    match_splits.clear_cache()
    splits.clear_cache()
    return root


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(progress_routes.router)
    return TestClient(app)


def row(game, fi, group, state, *, method=LabelMethod.HAND_SERIES, tv=None):
    return FrameLabelStateRow(game_id=game, frame_index=fi, cls_group=group, state=state,
                              method=method, teacher_version=tv)


def seed(root, rows):
    with FeatureStore.open(root / "fs.duckdb") as s:
        s.upsert_frame_label_states(rows)


def test_missing_and_empty_store_are_not_a_500(env, client):
    r = client.get("/progress/labelling")
    assert r.status_code == 200
    assert r.json()["state"] == "missing" and r.json()["totals"]["frames"] == 0
    with FeatureStore.open(env / "fs.duckdb"):
        pass
    j = client.get("/progress/labelling").json()
    assert j["state"] == "empty" and j["clips"] == [] and j["by_split"] == {}


def test_store_without_the_table_is_empty(env, client):
    duckdb.connect(str(env / "fs.duckdb")).close()
    assert client.get("/progress/labelling").json()["state"] == "empty"


def test_counts_ball_only_clip_and_splits(env, client):
    seed(env, [
        # t1_seg002 (train): f0 fully labelled, f1 ball only, f2 ball complete + player partial
        row("t1_seg002", 0, B, C), row("t1_seg002", 0, P, A),
        row("t1_seg002", 1, B, A, method=LabelMethod.SIDECAR_MARKER),
        row("t1_seg002", 2, B, C), row("t1_seg002", 2, P, PART),
        # h1_seg001 (heldout): ball-only clip, never fully labelled
        row("h1_seg001", 0, B, C, method=LabelMethod.VERDICT), row("h1_seg001", 1, B, C, method=LabelMethod.VERDICT),
        # t1_seg001 (eval in-match): silver, two teacher versions
        row("t1_seg001", 0, B, C, method=LabelMethod.OFFLINE_AUDITED, tv="offline_v1"),
        row("t1_seg001", 0, P, C, method=LabelMethod.OFFLINE_AUDITED, tv="offline_v2"),
    ])
    j = client.get("/progress/labelling").json()
    assert j["state"] == "ok"
    by = {c["clip"]: c for c in j["clips"]}
    t = by["t1_seg002"]
    assert (t["frames"], t["ball_labelled"], t["player_labelled"], t["fully_labelled"], t["partial"]) == (3, 3, 1, 1, 2)
    ho = by["h1_seg001"]
    assert (ho["frames"], ho["ball_labelled"], ho["player_labelled"], ho["fully_labelled"]) == (2, 2, 0, 0)
    assert j["totals"] == {"frames": 6, "ball_labelled": 6, "player_labelled": 2,
                           "fully_labelled": 2, "partial": 4, "ball_unusable": 0, "ball_deferred": 0}
    assert t["split"] == splits.TRAIN
    assert ho["split"] == splits.EVAL_HELDOUT
    assert by["t1_seg001"]["split"] == splits.EVAL_IN_MATCH
    assert j["by_split"][splits.EVAL_HELDOUT]["frames"] == 2
    assert j["by_split"][splits.TRAIN]["fully_labelled"] == 1
    assert j["by_split"][splits.EVAL_IN_MATCH]["fully_labelled"] == 1
    assert j["by_method"] == {"hand_series": 3, "offline_audited": 2, "sidecar_marker": 1, "verdict": 2}
    assert j["tiers"] == {
        "gold": 5,
        "silver": {"total": 2, "by_teacher_version": {"offline_v1": 1, "offline_v2": 1}},
        "other": 1,
    }
    assert j["clips"][0]["fully_labelled"] == 0  # fewest fully labelled first


def test_direct_call_matches_route(env):
    seed(env, [row("t1_seg002", 0, B, C), row("t1_seg002", 0, P, C)])
    assert labelling_progress.get_labelling()["totals"]["fully_labelled"] == 1


def test_denominator_unlabelled_clips_and_sidecar_player_marks(env, tmp_path, monkeypatch):
    clips, marks = tmp_path / "clips", tmp_path / "marks"
    clips.mkdir()
    marks.mkdir()
    (clips / "t1_seg002.mp4").write_bytes(b"x")
    (clips / "t1_seg003.mp4").write_bytes(b"y")  # no state rows at all
    box = {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}
    lines = [
        {"frame_index": 0, "bbox": box, "tags": ["player"]},
        {"frame_index": 0, "bbox": box, "tags": ["referee"]},  # same frame: counted once
        {"frame_index": 4, "bbox": box, "tags": ["player_sub"]},
        {"frame_index": 5, "bbox": box, "tags": ["in_play_ball"]},  # ball: not a player box
        {"frame_index": 6, "bbox": None, "tags": ["no_ball"]},
    ]
    (marks / "t1_seg002.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    monkeypatch.setattr(labelling_progress, "_video_frames", lambda p: 100 if p.stem == "t1_seg002" else None)
    seed(env, [row("t1_seg002", 0, B, C), row("t1_seg002", 0, P, C), row("t1_seg002", 1, B, C)])
    j = labelling_progress.get_labelling(clips_dir=clips, marks_dir=marks)
    by = {c["clip"]: c for c in j["clips"]}
    a = by["t1_seg002"]
    assert (a["clip_frames"], a["frames"], a["fully_labelled"], a["partial"]) == (100, 2, 1, 1)
    assert a["player_marked_sidecar"] == 2 and a["player_labelled"] == 1  # store stays authoritative
    b = by["t1_seg003"]
    assert (b["clip_frames"], b["frames"], b["fully_labelled"], b["player_marked_sidecar"]) == (None, 0, 0, 0)


def test_partial_definition(env):
    seed(env, [
        row("t1_seg002", 0, B, LabelState.UNREVIEWED),  # nothing labelled: not partial
        row("t1_seg002", 1, P, C),  # player only: partial
        row("t1_seg002", 2, B, PART), row("t1_seg002", 2, P, LabelState.UNREVIEWED),  # explicit partial
    ])
    t = labelling_progress.get_labelling(clips_dir=env, marks_dir=env)["totals"]
    assert (t["frames"], t["partial"], t["fully_labelled"]) == (3, 2, 0)


def test_teacher_versions_block(env):
    seed(env, [row("t1_seg002", 0, B, C, method=LabelMethod.OFFLINE_AUDITED, tv="offline_v1")])
    rates = {"offline_v1": {"n": 10, "passed": 9, "pass_rate": 0.9}}
    out = labelling_progress.get_labelling(audit_rates=rates)
    assert out["teacher_versions"] == {"offline_v1": {"silver": 1, "audit": rates["offline_v1"]}}


# -- selected-frame progress (ball_anchor_iteration.md 3.2 / 3.3) -------------------------------


def _seed_f(root):
    from footy_track.feature_store.schema import SelectedFrameRow

    with FeatureStore.open(root / "fs.duckdb") as s:
        sel = [("ball-round/d/1", "w1", 0, 10), ("ball-round/d/1", "w2", 0, 3), ("ball-eval/m/v1", "p1", 0, 2)]
        s.upsert_selected_frames([
            SelectedFrameRow(job_key=k, game_id="t1_seg002", window_id=w, split="train", frame_index=base + i)
            for k, w, base, n in sel for i in range(n) for base in [{"w1": 0, "w2": 100, "p1": 200}[w]]
        ])
        t = lambda fi, term, **kw: s.write_ball_terminal_state(  # noqa: E731
            "t1_seg002", fi, term, method="hand_frame", full_frame_check=term == "no_ball", **kw)
        for fi in (0, 1, 2, 3):
            t(fi, "ball")
        t(4, "no_ball")
        t(5, "unusable")
        t(6, "deferred", reason="unsure_x2")
        # frames 7-9 of w1 unreviewed; w2 (100-102): all terminal; eval p1 (200,201): one ball
        for fi in (100, 101):
            t(fi, "ball")
        t(102, "no_ball")
        t(200, "ball")


def test_frame_progress_counts_and_families(env, client):
    _seed_f(env)
    fp = client.get("/progress/labelling").json()["frame_progress"]
    assert fp["state"] == "ok"
    train, ev = fp["families"]["train"], fp["families"]["eval"]
    assert train["totals"]["frames_total"] == 13 and train["totals"]["frames_terminal"] == 10
    assert train["totals"]["by_state"] == {"ball": 6, "no_ball": 2, "unusable": 1, "deferred": 1, "outstanding": 3}
    assert ev["totals"]["frames_total"] == 2 and ev["totals"]["frames_terminal"] == 1
    assert fp["families"]["other"]["jobs"] == []
    (job,) = train["jobs"]
    w1, w2 = job["windows"]
    assert (w1["window_id"], w1["done"]) == ("w1", False)
    assert (w2["window_id"], w2["done"]) == ("w2", True)
    assert w1["by_state"]["deferred"] == 1 and w1["flag_deferred"] is False  # exactly 10% is not flagged
    assert labelling_progress._bucket(10, 5, 2, 0, 2)["flag_deferred"] is True  # 20% deferred
    assert job["done"] is False


def test_reasoned_rows_are_not_plain_absent(env, client):
    _seed_f(env)
    t = client.get("/progress/labelling").json()["totals"]
    assert t["ball_unusable"] == 1 and t["ball_deferred"] == 1
    assert t["ball_labelled"] == 9  # 7 ball + 2 no_ball; unusable/deferred are not plain labels


def test_store_without_selected_frame_rows(env, client):
    seed(env, [row("t1_seg002", 0, B, C)])
    fp = client.get("/progress/labelling").json()["frame_progress"]
    assert fp["state"] == "empty" and fp["totals"]["frames_total"] == 0
    # a pre-T6 store: table dropped
    with duckdb.connect(str(env / "fs.duckdb")) as c:
        c.execute("DROP TABLE selected_frame")
    fp = labelling_progress.get_frame_progress()
    assert fp["state"] == "empty"


def test_job_frame_progress(env):
    _seed_f(env)
    p = labelling_progress.job_frame_progress("ball-eval/m/v1")
    assert (p["frames_total"], p["frames_terminal"]) == (2, 1)
    assert labelling_progress.job_frame_progress("nope") is None


def test_job_progress_gets_frame_fields(env):
    from footy_track.labeller import jobs

    _seed_f(env)
    base = {"supported": True, "done": 1, "total": 2}
    out = jobs._with_frames({"key": "ball-eval/m/v1"}, dict(base))
    assert out["frames_total"] == 2 and out["frames_terminal"] == 1 and out["by_state"]["ball"] == 1
    assert jobs._with_frames({"key": "other/x"}, dict(base)) == base
    assert jobs._with_frames({"key": "ball-round/none"}, dict(base)) == base
