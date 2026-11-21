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
                           "fully_labelled": 2, "partial": 4}
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


def test_groups_are_independent_and_never_pooled(env, client):
    seed(env, [
        # t1_seg002 (train): f0 both complete, f1 ball absent only, f2 ball complete + player partial,
        # f3 player unreviewed row + ball unreviewed (nothing labelled)
        row("t1_seg002", 0, B, C, method=LabelMethod.HAND_SERIES), row("t1_seg002", 0, P, A, method=LabelMethod.HAND_FRAME),
        row("t1_seg002", 1, B, A, method=LabelMethod.SIDECAR_MARKER),
        row("t1_seg002", 2, B, C, method=LabelMethod.VERDICT), row("t1_seg002", 2, P, PART),
        row("t1_seg002", 3, P, LabelState.UNREVIEWED),
        # h1_seg001 (heldout): player only
        row("h1_seg001", 0, P, C, method=LabelMethod.OFFLINE_AUDITED, tv="offline_v1"),
    ])
    j = client.get("/progress/labelling").json()
    g = j["groups"]
    assert set(g) == {"ball", "player"}
    assert "ball" in g["ball"]["classes"] and "referee" in g["player"]["classes"]
    bt, pt = g["ball"]["totals"], g["player"]["totals"]
    assert (bt["frames"], bt["complete"], bt["absent"], bt["partial"], bt["unreviewed"], bt["labelled"]) == (5, 2, 1, 0, 2, 3)
    assert (pt["frames"], pt["complete"], pt["absent"], pt["partial"], pt["unreviewed"], pt["labelled"]) == (5, 1, 1, 1, 2, 2)
    assert bt["pct_labelled"] == 60.0 and pt["pct_labelled"] == 40.0  # no video: % of frames with state
    # legacy keys unchanged and consistent with the groups
    assert j["totals"]["ball_labelled"] == bt["labelled"] and j["totals"]["player_labelled"] == pt["labelled"]
    assert j["totals"]["fully_labelled"] == 1
    # by_method is per group
    assert g["ball"]["by_method"] == {"hand_series": 1, "sidecar_marker": 1, "verdict": 1}
    assert g["player"]["by_method"] == {"hand_frame": 1, "offline_audited": 1}
    assert g["player"]["tiers"]["silver"]["by_teacher_version"] == {"offline_v1": 1}
    assert g["ball"]["tiers"]["silver"]["total"] == 0
    # by_split per group
    assert g["ball"]["by_split"][splits.TRAIN]["complete"] == 2
    assert g["ball"]["by_split"][splits.EVAL_HELDOUT]["frames"] == 1
    assert g["ball"]["by_split"][splits.EVAL_HELDOUT]["unreviewed"] == 1
    assert g["player"]["by_split"][splits.EVAL_HELDOUT]["complete"] == 1
    # per-clip rows, each group sorted by its own labelled count
    assert [c["clip"] for c in g["ball"]["clips"]] == ["h1_seg001", "t1_seg002"]
    assert g["player"]["clips"][0]["labelled"] == 1
    by = {c["clip"]: c for c in j["clips"]}
    assert by["t1_seg002"]["groups"]["ball"]["complete"] == 2
    assert by["h1_seg001"]["groups"]["ball"]["labelled"] == 0


def test_group_pct_uses_clip_frames_when_known(env, tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "t1_seg002.mp4").write_bytes(b"x")
    monkeypatch.setattr(labelling_progress, "_video_frames", lambda p: 200)
    seed(env, [row("t1_seg002", 0, B, C), row("t1_seg002", 1, B, A)])
    g = labelling_progress.get_labelling(clips_dir=clips, marks_dir=None)["groups"]
    assert g["ball"]["totals"]["clip_frames"] == 200 and g["ball"]["totals"]["pct_labelled"] == 1.0
    assert g["player"]["totals"]["labelled"] == 0 and g["player"]["totals"]["unreviewed"] == 2


def test_empty_result_has_groups(env, client):
    j = client.get("/progress/labelling").json()
    assert j["groups"]["ball"]["totals"]["frames"] == 0 and j["groups"]["player"]["clips"] == []
