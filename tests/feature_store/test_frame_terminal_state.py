"""Terminal ball states + reasons, selected_frame (expected set F), additive migration."""

from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pytest
from pydantic import ValidationError

from footy_track.feature_store import FeatureStore
from footy_track.feature_store.ingest_gt import ingest_gt_jsonl
from footy_track.feature_store.schema import (
    ClsGroup,
    FrameLabelStateRow,
    SelectedFrameRow,
)

G = "clip_a"


@pytest.fixture
def store():
    s = FeatureStore.open(":memory:")
    yield s
    s.close()


def test_mapping_and_roundtrip(store):
    w = store.write_ball_terminal_state
    w(G, 0, "ball", method="verdict", generator_version="g1")
    w(G, 1, "no_ball", method="hand_frame", full_frame_check=True)
    w(G, 2, "unusable", method="hand_frame")
    w(G, 3, "deferred", method="verdict", reason="unsure_x2")
    w(G, 4, "deferred", method="verdict", reason="window_rejected")
    df = store.read_frame_label_states(G, "ball").set_index("frame_index")
    assert df.loc[0, "state"] == "complete" and df.loc[0, "generator_version"] == "g1"
    assert df.loc[1, "state"] == "absent" and df.loc[1, "reason"] is None
    assert (df.loc[2, "state"], df.loc[2, "reason"]) == ("absent", "not_broadcast")
    assert (df.loc[3, "state"], df.loc[3, "reason"]) == ("partial", "unsure_x2")
    assert store.read_ball_terminal_states(G) == {
        0: "ball", 1: "no_ball", 2: "unusable", 3: "deferred", 4: "deferred",
    }  # fmt: skip
    assert store.read_ball_terminal_states(G, [1, 2]) == {1: "no_ball", 2: "unusable"}


def test_no_ball_requires_full_frame_check(store):
    with pytest.raises(ValueError, match="full-frame"):
        store.write_ball_terminal_state(G, 1, "no_ball", method="hand_frame")
    assert store.read_frame_label_states(G).empty


@pytest.mark.parametrize(
    ("terminal", "reason"),
    [("deferred", None), ("deferred", "not_broadcast"), ("ball", "unsure_x2"), ("unusable", "unsure_x2"),
     ("bogus", None), ("deferred", "bogus")],
)  # fmt: skip
def test_bad_combinations_rejected(store, terminal, reason):
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic ValidationError is a ValueError
        store.write_ball_terminal_state(G, 1, terminal, method="verdict", reason=reason)


def test_row_model_validates_reason():
    with pytest.raises(ValidationError):
        FrameLabelStateRow(game_id=G, frame_index=0, cls_group=ClsGroup.PLAYER, state="partial",
                           method="verdict", reason="unsure_x2")  # fmt: skip
    with pytest.raises(ValidationError):
        FrameLabelStateRow(game_id=G, frame_index=0, cls_group=ClsGroup.BALL, state="complete",
                           method="verdict", reason="unsure_x2")  # fmt: skip


def test_precedence_unchanged_and_reason_cleared(store):
    store.write_ball_terminal_state(G, 0, "deferred", method="verdict", reason="unsure_x2")
    # weaker writer cannot downgrade
    store.write_ball_terminal_state(G, 0, "unusable", method="sidecar_marker")
    assert store.read_ball_terminal_states(G) == {0: "deferred"}
    # stronger human decision replaces it and clears the reason
    store.write_ball_terminal_state(G, 0, "ball", method="hand_frame")
    row = store.read_frame_label_states(G, "ball").iloc[0]
    assert row["state"] == "complete" and row["reason"] is None
    # re-write is idempotent
    store.write_ball_terminal_state(G, 0, "ball", method="hand_frame")
    assert store.count("frame_label_state") == 1


def test_reasoned_rows_never_count_as_labelled_negatives(store):
    store.write_ball_terminal_state(G, 0, "unusable", method="hand_frame")
    store.write_ball_terminal_state(G, 1, "no_ball", method="hand_frame", full_frame_check=True)
    for fi in (0, 1):
        store.upsert_frame_label_states([
            FrameLabelStateRow(game_id=G, frame_index=fi, cls_group="player", state="complete", method="hand_frame")
        ])  # fmt: skip
    assert set(store.query("SELECT frame_index FROM frame_complete")["frame_index"]) == {1}


def test_selected_frames_idempotent_and_filters(store):
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    rows = [SelectedFrameRow(job_key="j1", game_id=G, frame_index=i, window_id="w0", split="train", selected_at=t0)
            for i in range(3)]  # fmt: skip
    assert store.upsert_selected_frames(rows) == 3
    # re-write (new window id, later time): no duplicates, selected_at kept
    again = [SelectedFrameRow(job_key="j1", game_id=G, frame_index=1, window_id="w9", split="train",
                              selected_at=datetime(2027, 1, 1))]  # fmt: skip
    store.upsert_selected_frames(again)
    store.upsert_selected_frames([SelectedFrameRow(job_key="j2", game_id=G, frame_index=1, split="eval_heldout_match")])
    df = store.read_selected_frames(job_key="j1")
    assert len(df) == 3
    r1 = df[df.frame_index == 1].iloc[0]
    assert r1["window_id"] == "w9" and r1["selected_at"] == t0
    assert len(store.read_selected_frames(game_id=G)) == 4
    assert len(store.read_selected_frames(window_id="w0")) == 2
    assert store.upsert_selected_frames([]) == 0
    with pytest.raises(ValidationError):
        SelectedFrameRow(job_key="j", game_id=G, frame_index=0, split="nope")


def test_existing_store_migrates_additively(tmp_path):
    db = tmp_path / "old.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute("""CREATE TABLE frame_label_state (game_id VARCHAR, frame_index INTEGER, cls_group VARCHAR,
        state VARCHAR, method VARCHAR, run_id VARCHAR, teacher_version VARCHAR, verified_by VARCHAR,
        verified_at TIMESTAMP, PRIMARY KEY (game_id, frame_index, cls_group),
        CHECK (cls_group IN ('ball','player')))""")
    conn.execute("INSERT INTO frame_label_state VALUES ('g', 1, 'ball', 'complete', 'verdict', NULL, NULL, NULL, NULL)")
    conn.close()
    for _ in range(2):  # twice: re-opening must be a no-op
        with FeatureStore.open(db) as s:
            df = s.read_frame_label_states("g")
            assert len(df) in (1, 2) and df.iloc[0]["reason"] is None
            s.write_ball_terminal_state("g", 2, "deferred", method="verdict", reason="video_unreadable")
            assert s.read_ball_terminal_states("g") == {1: "ball", 2: "deferred"}
            assert s.count("selected_frame") == 0


def test_gt_reingest_idempotent_with_terminal_rows(store, tmp_path):
    recs = [
        {"frame_index": 0, "bbox": {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03}, "tags": ["in_play_ball", "labeller"]},
        {"frame_index": 1, "bbox": None, "tags": ["no_ball"]},
    ]
    p = tmp_path / f"{G}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    store.write_ball_terminal_state(G, 5, "deferred", method="verdict", reason="unsure_x2")
    store.write_ball_terminal_state(G, 0, "ball", method="hand_frame", generator_version="g1")
    ingest_gt_jsonl(store, p, video_dir=tmp_path)
    first = store.read_frame_label_states(G, "ball")
    ingest_gt_jsonl(store, p, video_dir=tmp_path)
    second = store.read_frame_label_states(G, "ball")
    assert first.equals(second)
    assert store.read_ball_terminal_states(G) == {0: "ball", 1: "no_ball", 5: "deferred"}
    assert second.set_index("frame_index").loc[0, "method"] == "hand_frame"  # not downgraded


def test_parquet_roundtrip_with_new_columns(store, tmp_path):
    store.write_ball_terminal_state(G, 3, "deferred", method="verdict", reason="unsure_x2", generator_version="g")
    store.upsert_selected_frames([SelectedFrameRow(job_key="j", game_id=G, frame_index=3, split="train")])
    store.export_parquet(tmp_path / "pq")
    with FeatureStore.from_parquet(tmp_path / "pq") as s2:
        assert s2.read_ball_terminal_states(G) == {3: "deferred"}
        assert len(s2.read_selected_frames()) == 1
