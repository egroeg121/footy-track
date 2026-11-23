"""Interim ``inferred_gt`` rule (user decision 2026-11-23): human player box + ball decision => player complete."""

from __future__ import annotations

import json
from pathlib import Path

from footy_track.feature_store import FeatureStore
from footy_track.feature_store.ingest_gt import ingest_gt_jsonl
from footy_track.feature_store.schema import (
    ClsGroup,
    FrameLabelStateRow,
    LabelMethod,
    LabelState,
    METHOD_RANK,
)

BOX = {"x": 0.1, "y": 0.1, "w": 0.05, "h": 0.1}


def _line(fi, tags, bbox=BOX):
    return {"frame_index": fi, "bbox": bbox, "center": None, "tags": tags}


def _write(tmp_path: Path, recs, stem="clip_seg001") -> Path:
    p = tmp_path / f"{stem}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def _recs():
    return [
        # 0: human player + human ball -> marked
        _line(0, ["player", "labeller"]),
        _line(0, ["in_play_ball", "labeller"]),
        # 1: human referee + no_ball -> marked
        _line(1, ["referee", "labeller"]),
        _line(1, ["no_ball"], None),
        # 2: players only, no ball decision -> not marked
        _line(2, ["player", "labeller"]),
        # 3: ball only -> not marked
        _line(3, ["in_play_ball", "labeller"]),
        # 4: machine (yolo) players + human ball -> not marked
        _line(4, ["player", "yolo"]),
        _line(4, ["in_play_ball", "labeller"]),
        # 5: human player + ball but not_broadcast -> not marked
        _line(5, ["player", "labeller"]),
        _line(5, ["in_play_ball", "labeller"]),
        _line(5, ["not_broadcast"], None),
        # 6: human-checked machine player + no_ball -> marked
        _line(6, ["player", "yolo", "human_checked"]),
        _line(6, ["no_ball"], None),
        # 7: ball/no_ball conflict -> not marked
        _line(7, ["player", "labeller"]),
        _line(7, ["in_play_ball", "labeller"]),
        _line(7, ["no_ball"], None),
    ]


def _inferred(store, game="clip_seg001"):
    df = store.read_frame_label_states(game, "player")
    df = df[df["method"] == "inferred_gt"]
    return sorted(int(i) for i in df["frame_index"])


def _ingest(store, path, tmp_path, **kw):
    return ingest_gt_jsonl(store, path, video_dir=tmp_path, **kw)


def test_rank_is_lowest() -> None:
    assert METHOD_RANK[LabelMethod.INFERRED_GT] < min(
        v for k, v in METHOD_RANK.items() if k != LabelMethod.INFERRED_GT
    )


def test_rule_cases(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    p = _write(tmp_path, _recs())
    rep = _ingest(store, p, tmp_path, infer_player_complete=True)
    assert _inferred(store) == [0, 1, 6]
    assert rep.players_inferred_complete == 3
    df = store.read_frame_label_states("clip_seg001", "player")
    assert set(df["state"]) == {"complete"}
    assert set(df["verified_by"]) == {"inferred_gt"}


def test_off_by_default_and_flag_off_purges(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    p = _write(tmp_path, _recs())
    _ingest(store, p, tmp_path)
    assert _inferred(store) == []
    _ingest(store, p, tmp_path, infer_player_complete=True)
    assert _inferred(store) == [0, 1, 6]
    _ingest(store, p, tmp_path, infer_player_complete=False)
    assert _inferred(store) == []


def test_idempotent(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    p = _write(tmp_path, _recs())
    for _ in range(3):
        _ingest(store, p, tmp_path, infer_player_complete=True)
    assert _inferred(store) == [0, 1, 6]
    assert store.count("frame_label_state") == len(store.read_frame_label_states())
    assert len(store.read_frame_label_states("clip_seg001", "player")) == 3


def test_removed_player_boxes_drop_the_row(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    p = _write(tmp_path, _recs())
    _ingest(store, p, tmp_path, infer_player_complete=True)
    p = _write(tmp_path, [r for r in _recs() if not (r["frame_index"] == 0 and "player" in r["tags"])])
    _ingest(store, p, tmp_path, infer_player_complete=True)
    assert _inferred(store) == [1, 6]


def test_real_marker_overrides_inferred(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    recs = _recs() + [_line(0, ["players_complete"], None)]
    p = _write(tmp_path, recs)
    _ingest(store, p, tmp_path, infer_player_complete=True)
    assert _inferred(store) == [1, 6]
    row = store.read_frame_label_states("clip_seg001", "player")
    m = {int(r.frame_index): r.method for r in row.itertuples()}
    assert m[0] == "sidecar_marker"
    # marker removed again -> falls back to the inferred row
    _ingest(store, _write(tmp_path, _recs()), tmp_path, infer_player_complete=True)
    assert _inferred(store) == [0, 1, 6]


def test_stronger_rows_survive_and_win(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    store.upsert_frame_label_states(
        [
            FrameLabelStateRow(
                game_id="clip_seg001", frame_index=0, cls_group=ClsGroup.PLAYER,
                state=LabelState.PARTIAL, method=LabelMethod.HAND_FRAME,
            )
        ]
    )
    p = _write(tmp_path, _recs())
    _ingest(store, p, tmp_path, infer_player_complete=True)
    _ingest(store, p, tmp_path, infer_player_complete=True)
    df = store.read_frame_label_states("clip_seg001", "player")
    r0 = df[df["frame_index"] == 0].iloc[0]
    assert (r0["method"], r0["state"]) == ("hand_frame", "partial")
    assert _inferred(store) == [1, 6]


def test_old_store_check_constraint_is_migrated(tmp_path) -> None:
    db = tmp_path / "old.duckdb"
    store = FeatureStore.open(db)
    store._conn.execute("DROP VIEW IF EXISTS frame_complete")
    store._conn.execute("DROP TABLE frame_label_state")
    store._conn.execute(
        """CREATE TABLE frame_label_state (game_id VARCHAR, frame_index INTEGER, cls_group VARCHAR,
        state VARCHAR, method VARCHAR, run_id VARCHAR, teacher_version VARCHAR, verified_by VARCHAR,
        verified_at TIMESTAMP, reason VARCHAR, generator_version VARCHAR,
        PRIMARY KEY (game_id, frame_index, cls_group),
        CHECK (method IN ('hand_series','hand_frame','verdict','sidecar_marker','offline_audited')))"""
    )
    store._conn.execute(
        "INSERT INTO frame_label_state (game_id, frame_index, cls_group, state, method) "
        "VALUES ('g', 1, 'ball', 'complete', 'verdict')"
    )
    store.close()
    store = FeatureStore.open(db)  # migrates
    assert len(store.read_frame_label_states("g", "ball")) == 1
    store.upsert_frame_label_states(
        [FrameLabelStateRow(game_id="g", frame_index=2, cls_group=ClsGroup.PLAYER,
                            state=LabelState.COMPLETE, method=LabelMethod.INFERRED_GT)]
    )
    assert len(store.read_frame_label_states("g")) == 2
    store.query("SELECT * FROM frame_complete")  # view recreated
    store.close()
    FeatureStore.open(db).close()  # second open is a no-op
