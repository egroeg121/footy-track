"""ingest_all: one idempotent, single-writer pass over every label source."""

from __future__ import annotations

import json
import multiprocessing as mp

import pytest

from footy_track.feature_store import ingest_all as ia
from footy_track.feature_store.lock import StoreLocked, store_lock
from footy_track.feature_store.store import FeatureStore
from tests.datasets.fixtures import box, write_broadcast


def run(w, **kw):
    return ia.run(w.db, data_root=w.root, marks_dirs=[w.marks], **kw)


def test_ingests_every_source(world):
    rep = run(world)
    assert rep.gt_clips == 4
    assert rep.ball_check_clips == 1 and rep.ball_check_verdicts == 1
    assert rep.broadcast_files == 1 and rep.broadcast_labels == 5
    assert rep.calibration_gt_frames == 4  # 1 + 1 + 2 homographies
    with FeatureStore.open(world.db) as s:
        assert s.count("broadcast_label") == 5
        markers = dict(
            s._conn.execute(
                "SELECT game_id || ':' || frame_index, marker FROM gt_frame WHERE marker IS NOT NULL"
            ).fetchall()
        )
        assert markers == {"hmatch_seg001:2": "no_ball", "tmatch_seg001:1": "not_broadcast"}
        # the existing ingesters did their part: GT boxes, a checked machine box, calibration + flag
        assert s.count("detection") == 8
        assert s.query("SELECT count(*) n FROM calibration WHERE review_status = 'not_broadcast'")["n"][0] == 1


def test_rerun_with_no_changes_changes_nothing(world):
    first = run(world)
    assert not first.unchanged  # empty store -> populated
    with FeatureStore.open(world.db) as s:
        before = ia.fingerprint(s)
        counts = {t: s.count(t) for t in ia.FINGERPRINT_TABLES}
    second = run(world)
    assert second.unchanged
    assert second.fingerprint_before == second.fingerprint_after == first.fingerprint_after == before
    with FeatureStore.open(world.db) as s:
        assert {t: s.count(t) for t in ia.FINGERPRINT_TABLES} == counts


def test_a_label_edit_is_picked_up_and_a_deleted_label_disappears(world):
    run(world)
    world.sidecar("tmatch_seg003", [box(5, "player"), box(6, "in_play_ball")])
    write_broadcast(world, "hmatch_full", {10: ("yes", "Yes", "edge")})
    rep = run(world)
    assert not rep.unchanged
    assert rep.broadcast_labels == 1
    with FeatureStore.open(world.db) as s:
        assert s.count("broadcast_label") == 1  # the cleared verdicts are gone, not ghosts
        assert s.query("SELECT count(*) n FROM detection WHERE game_id='tmatch_seg003'")["n"][0] == 2


def test_dry_run_never_touches_the_file(world):
    rep = run(world, dry_run=True)
    assert rep.gt_clips == 4
    assert not world.db.exists()


def test_single_writer_lock(world):
    with store_lock(world.db, exclusive=True), pytest.raises(StoreLocked):
        run(world)
    run(world)  # lock released: fine


def _hold(db, ready, release):
    with store_lock(db, exclusive=True):
        ready.set()
        release.wait(20)


def test_lock_is_held_across_processes(world):
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    p = ctx.Process(target=_hold, args=(world.db, ready, release))
    p.start()
    try:
        assert ready.wait(20)
        with pytest.raises(StoreLocked):
            run(world)
    finally:
        release.set()
        p.join(20)


def test_ball_check_logs_of_one_clip_in_two_dirs_are_merged(world, tmp_path):
    other = tmp_path / "marks2"
    (other / "ball_checks").mkdir(parents=True)
    world.sidecar("tmatch_seg002", [box(0, "in_play_ball"), box(1, "in_play_ball", prov="vittrack", extra=["human_checked"]),
                                    box(2, "in_play_ball", prov="vittrack")])
    # a later tombstone (verdict null) in the second dir retracts the first dir's verdict
    (other / "ball_checks" / "tmatch_seg002.jsonl").write_text(
        json.dumps({"clip": "tmatch_seg002", "frame_index": 0, "box_index": 0, "source": "sidecar",
                    "verdict": None, "ts": 200.0}) + "\n"
    )
    ia.run(world.db, data_root=world.root, marks_dirs=[world.marks, other])
    with FeatureStore.open(world.db) as s:
        assert s.count("detection_review") == 0
