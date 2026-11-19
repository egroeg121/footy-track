"""frame_label_state table, frame_complete view, precedence and sidecar/verdict writers."""

from __future__ import annotations

import json

import duckdb

from footy_track.feature_store import FeatureStore
from footy_track.feature_store.ball_checks import import_ball_checks
from footy_track.feature_store.ingest_gt import ingest_gt_jsonl
from footy_track.feature_store.schema import (
    ClsGroup,
    FrameLabelStateRow,
    LabelMethod,
    LabelState,
)

CLIP = "clip_a"
BOX = {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03}


def _records():
    return [
        {"frame_index": 0, "bbox": BOX, "tags": ["in_play_ball", "labeller"]},
        {"frame_index": 0, "bbox": BOX, "tags": ["player", "labeller"]},
        {"frame_index": 1, "bbox": None, "tags": ["no_ball"]},
        {"frame_index": 2, "bbox": BOX, "tags": ["in_play_ball", "vittrack"]},
        {"frame_index": 3, "bbox": BOX, "tags": ["player", "labeller"]},
    ]


def _ingest(store, tmp_path, records=None):
    p = tmp_path / f"{CLIP}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (records or _records())) + "\n")
    return ingest_gt_jsonl(store, p, video_dir=tmp_path)


def _states(store, group=None):
    df = store.read_frame_label_states(CLIP, group)
    return {(r.frame_index, r.cls_group): (r.state, r.method) for r in df.itertuples()}


def _row(fi, group, state, method=LabelMethod.HAND_SERIES, **kw):
    return FrameLabelStateRow(
        game_id=CLIP,
        frame_index=fi,
        cls_group=group,
        state=state,
        method=method,
        **kw,
    )


def test_no_ball_absent_and_human_ball_complete(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    rep = _ingest(store, tmp_path)
    assert _states(store) == {
        (0, "ball"): ("complete", "sidecar_marker"),
        (1, "ball"): ("absent", "sidecar_marker"),
    }
    assert (rep.ball_complete, rep.ball_absent) == (1, 1)


def test_no_player_backfill(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _ingest(store, tmp_path)
    assert _states(store, "player") == {}
    assert store.query("SELECT * FROM frame_complete").empty


def test_reingest_idempotent(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _ingest(store, tmp_path)
    first = store.read_frame_label_states().to_dict("records")
    _ingest(store, tmp_path)
    assert store.read_frame_label_states().to_dict("records") == first


def test_reingest_drops_stale_marker(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _ingest(store, tmp_path)
    _ingest(store, tmp_path, [r for r in _records() if r["tags"] != ["no_ball"]])
    assert (1, "ball") not in _states(store)


def test_reingest_never_downgrades_stronger_method(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _ingest(store, tmp_path)
    store.upsert_frame_label_states(
        [
            _row(1, ClsGroup.BALL, LabelState.PARTIAL, LabelMethod.HAND_SERIES),
            _row(0, ClsGroup.BALL, LabelState.ABSENT, LabelMethod.HAND_FRAME),
        ]
    )
    _ingest(store, tmp_path)
    got = _states(store)
    assert got[(1, "ball")] == ("partial", "hand_series")
    assert got[(0, "ball")] == ("absent", "hand_frame")
    # a weaker write is ignored directly, too
    store.upsert_frame_label_states(
        [
            _row(
                1,
                ClsGroup.BALL,
                LabelState.COMPLETE,
                LabelMethod.OFFLINE_AUDITED,
                teacher_version="offline_v1",
            )
        ]
    )
    assert _states(store)[(1, "ball")] == ("partial", "hand_series")


def test_contradictory_frame_left_unreviewed(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    recs = [*_records(), {"frame_index": 0, "bbox": None, "tags": ["no_ball"]}]
    rep = _ingest(store, tmp_path, recs)
    assert (0, "ball") not in _states(store)
    assert rep.ball_state_conflicts == 1


def test_frame_complete_view(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _ingest(store, tmp_path)
    store.upsert_frame_label_states(
        [
            _row(0, ClsGroup.PLAYER, LabelState.COMPLETE),  # ball complete too -> yes
            _row(
                1, ClsGroup.PLAYER, LabelState.PARTIAL
            ),  # ball absent, player partial -> no
            _row(5, ClsGroup.PLAYER, LabelState.ABSENT),  # player only -> no
        ]
    )
    df = store.query("SELECT frame_index FROM frame_complete")
    assert list(df.frame_index) == [0]


def _ball_check(tmp_path, verdict, frame=2):
    bc = tmp_path / "bc"
    bc.mkdir(exist_ok=True)
    rec = {
        "clip": CLIP,
        "frame_index": frame,
        "box_index": 0,
        "source": "sidecar",
        "verdict": verdict,
        "bbox": BOX,
        "label": "in_play_ball",
        "corrected_bbox": None,
        "ts": 1.0,
    }
    (bc / f"{CLIP}.jsonl").write_text(json.dumps(rec) + "\n")
    return bc


def test_ball_verdict_complete_and_unsure_stays_unreviewed(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _ingest(store, tmp_path)
    import_ball_checks(store, _ball_check(tmp_path, "ball"))
    assert _states(store)[(2, "ball")] == ("complete", "verdict")
    import_ball_checks(
        store, _ball_check(tmp_path, "unsure")
    )  # re-run replaces verdict rows
    assert (2, "ball") not in _states(store)
    import_ball_checks(store, _ball_check(tmp_path, "not_ball"))
    assert (2, "ball") not in _states(store)


def test_migrates_existing_store_file(tmp_path) -> None:
    db = tmp_path / "old.duckdb"
    store = FeatureStore.open(db)
    _ingest(store, tmp_path)
    store.close()
    con = duckdb.connect(str(db))  # simulate a store from before this change
    con.execute("DROP VIEW frame_complete")
    con.execute("DROP TABLE frame_label_state")
    con.close()
    store = FeatureStore.open(db)  # create_schema migrates in place
    assert store.count("frame_label_state") == 0
    assert store.count("detection") > 0  # existing data intact
    _ingest(store, tmp_path)
    assert (1, "ball") in _states(store)
    assert store.query("SELECT * FROM frame_complete").empty


# --- players_complete marker -> player state ("all players boxed", full_frame_labelling.md section 1) ---


def _player_records():
    return [
        {"frame_index": 0, "bbox": BOX, "tags": ["player", "labeller"]},
        {"frame_index": 0, "bbox": BOX, "tags": ["referee", "vittrack", "human_checked"]},
        {"frame_index": 0, "bbox": None, "tags": ["players_complete"]},
        {"frame_index": 1, "bbox": None, "tags": ["players_complete"]},  # zero-person frame
        {"frame_index": 2, "bbox": BOX, "tags": ["player", "labeller"]},  # boxed but NOT asserted
        {"frame_index": 3, "bbox": BOX, "tags": ["in_play_ball", "labeller"]},  # ball only
        {"frame_index": 3, "bbox": None, "tags": ["players_complete"]},  # marker, no person box -> absent
    ]


def test_players_complete_marker_gives_player_complete_or_absent(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    rep = _ingest(store, tmp_path, _player_records())
    assert _states(store, "player") == {
        (0, "player"): ("complete", "sidecar_marker"),
        (1, "player"): ("absent", "sidecar_marker"),
        (3, "player"): ("absent", "sidecar_marker"),
    }  # frame 2: boxes but no marker -> no state (no back-fill)
    assert (rep.players_complete, rep.players_absent) == (1, 2)
    # the marker line is not a detection and not an object class
    n = store.query("SELECT count(*) AS n FROM detection").n[0]
    assert n == 4


def test_removing_players_complete_marker_removes_state(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _ingest(store, tmp_path, _player_records())
    kept = [r for r in _player_records() if not (r["frame_index"] == 0 and r["tags"] == ["players_complete"])]
    _ingest(store, tmp_path, kept)
    assert (0, "player") not in _states(store, "player")
    assert (1, "player") in _states(store, "player")
    _ingest(store, tmp_path, [r for r in kept if r["tags"] != ["players_complete"]])
    assert _states(store, "player") == {}


def test_players_complete_reingest_idempotent(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _ingest(store, tmp_path, _player_records())
    first = store.read_frame_label_states().to_dict("records")
    _ingest(store, tmp_path, _player_records())
    assert store.read_frame_label_states().to_dict("records") == first


def test_players_complete_does_not_override_stronger_row(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _ingest(store, tmp_path, _player_records())
    store.upsert_frame_label_states([_row(0, ClsGroup.PLAYER, LabelState.PARTIAL, LabelMethod.HAND_FRAME)])
    _ingest(store, tmp_path, _player_records())
    assert _states(store, "player")[(0, "player")] == ("partial", "hand_frame")
    # and dropping the marker leaves the stronger row alone
    _ingest(store, tmp_path, [r for r in _player_records() if r["tags"] != ["players_complete"]])
    assert _states(store, "player") == {(0, "player"): ("partial", "hand_frame")}


def test_players_complete_feeds_frame_complete_view(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    recs = [
        *_records(),
        {"frame_index": 0, "bbox": None, "tags": ["players_complete"]},  # ball complete + player complete
        {"frame_index": 1, "bbox": None, "tags": ["players_complete"]},  # ball absent + player absent
    ]
    _ingest(store, tmp_path, recs)
    df = store.query("SELECT frame_index FROM frame_complete")
    assert sorted(df.frame_index) == [0, 1]


# --- legacy back-fill (--legacy-complete-clips): named clips only ---


def _legacy(store, tmp_path, clips, records=None):
    from footy_track.feature_store.ingest_gt import ingest_gt_dir  # noqa: PLC0415

    (tmp_path / f"{CLIP}.jsonl").write_text("\n".join(json.dumps(r) for r in (records or _records())) + "\n")
    return ingest_gt_dir(store, tmp_path, video_dir=tmp_path, legacy_complete_clips=set(clips))


def test_legacy_backfill_named_clip_only_labeller_player_frames(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    rep = _legacy(store, tmp_path, [CLIP])
    # frames 0 and 3 have a hand-label player box; 1 (no_ball) and 2 (vittrack ball) get no player state
    assert _states(store, "player") == {
        (0, "player"): ("complete", "sidecar_marker"),
        (3, "player"): ("complete", "sidecar_marker"),
    }
    assert rep.players_legacy_complete == 2
    row = store.read_frame_label_states(CLIP, "player").iloc[0]
    assert (row.verified_by, row.run_id) == ("legacy_backfill", "gt_import_legacy_backfill")


def test_legacy_backfill_ignores_machine_player_boxes(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _legacy(store, tmp_path, [CLIP], [{"frame_index": 5, "bbox": BOX, "tags": ["player", "yolo"]}])
    assert _states(store, "player") == {}


def test_legacy_backfill_unnamed_clip_untouched(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    rep = _legacy(store, tmp_path, ["some_other_clip"])
    assert _states(store, "player") == {}
    assert rep.players_legacy_complete == 0


def test_legacy_backfill_idempotent_and_removal_removes(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _legacy(store, tmp_path, [CLIP])
    first = store.read_frame_label_states().to_dict("records")
    _legacy(store, tmp_path, [CLIP])
    assert store.read_frame_label_states().to_dict("records") == first
    _legacy(store, tmp_path, [])
    assert _states(store, "player") == {}
    assert _states(store, "ball")  # ball rows untouched


def test_legacy_backfill_does_not_override_stronger_rows(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    store.upsert_frame_label_states(
        [
            _row(0, ClsGroup.PLAYER, LabelState.PARTIAL, LabelMethod.HAND_FRAME),
            _row(3, ClsGroup.PLAYER, LabelState.PARTIAL, LabelMethod.HAND_SERIES),
        ]
    )
    _legacy(store, tmp_path, [CLIP])
    assert _states(store, "player") == {
        (0, "player"): ("partial", "hand_frame"),
        (3, "player"): ("partial", "hand_series"),
    }


def test_legacy_backfill_feeds_frame_complete_view(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    _legacy(store, tmp_path, [CLIP])
    # frame 0: ball complete + backfilled player; frame 3: backfilled player but no ball state
    assert sorted(store.query("SELECT frame_index FROM frame_complete").frame_index) == [0]
