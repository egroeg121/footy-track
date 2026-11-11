"""Importer for calibration_gt/ + calibration_proposals/ sidecars."""

from __future__ import annotations

import json
import logging

import pytest

from footy_track.feature_store import FeatureStore
from footy_track.feature_store.ingest_calibration import (
    GT_RUN_ID,
    ingest_calibration_dir,
    main,
    resolve_game_ids,
)

H = [[0.31, -0.002, -0.10], [-0.027, -0.79, 0.51], [-0.0007, 0.011, 0.004]]


def _frame(**kw):
    rec = {
        "correspondences": [
            {"image": [0.33, 0.63], "pitch_id": "CENTRE_SPOT", "pitch_xy": [0.0, 0.0]},
            {"image": [0.33, 0.52], "pitch_id": "CENTRE_CIRCLE_TOP", "pitch_xy": [0.0, 9.15]},
            {"image": [0.64, 0.63], "pitch_id": None, "pitch_xy": [9.15, 0.0]},
        ],
        "H": H,
        "H_convention": "normalised_image_to_pitch_m",
        "pitch_dims": [105.0, 68.0],
        "image_size": [1920, 1080],
        "method": "dlt",
        "inliers": [True, True, False],
        "mean_err_m": 0.25,
        "err_px": [6.0, 3.0, None],
        "ts": "2026-11-10T23:51:04+0000",
    }
    rec.update(kw)
    return rec


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _proposals(clip="arsenal_mancity_seg010", frames=None, run="claude_v1", source="claude"):
    return {
        "clip": clip,
        "runs": {
            run: {
                "source": source,
                "created": "2026-11-11T10:00:00+00:00",
                "params": {"model": "claude-x", "prompt_version": "p3"},
                "frames": frames
                if frames is not None
                else {"0": _frame(status="proposed", score=0.7, failure=None, view=None)},
            }
        },
    }


@pytest.fixture
def store():
    s = FeatureStore.open()
    yield s
    s.close()


def _best(store):
    return store.query(
        "SELECT game_id, frame_index, source, run_id, tier FROM frame_calibration_best "
        "ORDER BY game_id, frame_index"
    )


def test_gt_import_fields(store, tmp_path):
    clip = "arsenal_mancity_seg010"
    _write(tmp_path / "calibration_gt" / f"{clip}.json", {"clip": clip, "frames": {"0": _frame(), "349": _frame()}})
    r = ingest_calibration_dir(store, tmp_path)
    assert r.gt_frames == 2 and r.keypoints == 6
    row = store.query("SELECT * FROM calibration WHERE frame_index = 0").iloc[0]
    assert (row["game_id"], row["source"], row["run_id"]) == (clip, "hand_label", GT_RUN_ID)
    assert row["review_status"] == "hand" and row["reviewed"] and row["quality"] == 1.0
    assert row["n_points"] == 3 and row["n_inliers"] == 2
    assert row["mean_err_px"] == pytest.approx(4.5)
    assert row["pitch_length"] == 105.0 and row["image_width"] == 1920
    assert list(row["H"]) == [x for r_ in H for x in r_]
    assert store.query("SELECT stage FROM run WHERE run_id = ?", [GT_RUN_ID])["stage"][0] == "calibration"
    kp = store.query("SELECT * FROM pitch_keypoint WHERE frame_index=0 ORDER BY point_idx")
    assert list(kp["keypoint_id"])[:2] == ["CENTRE_SPOT", "CENTRE_CIRCLE_TOP"]
    assert kp["keypoint_id"].isna()[2]
    assert list(kp["is_inlier"]) == [True, True, False]


def test_alias_pair_lands_on_ball_gt_game_ids(store, tmp_path):
    # ball GT exists under the date-stamped alias only (and one other alias)
    for stem in ("arsenal_mancity_20250925_seg010", "mancity_seg010", "arsenal_mancity_seg020"):
        (tmp_path / f"{stem}.jsonl").write_text("")
    canon = "arsenal_mancity_seg010"
    _write(tmp_path / "calibration_gt" / f"{canon}.json", {"frames": {"0": _frame()}})
    assert resolve_game_ids(canon, tmp_path) == [
        "arsenal_mancity_20250925_seg010",
        "mancity_seg010",
    ]
    r = ingest_calibration_dir(store, tmp_path)
    assert r.game_map[canon] == ["arsenal_mancity_20250925_seg010", "mancity_seg010"]
    assert set(_best(store)["game_id"]) == {"arsenal_mancity_20250925_seg010", "mancity_seg010"}
    assert "arsenal_mancity_seg020" not in set(_best(store)["game_id"])  # other clip


def test_no_ball_gt_falls_back_to_canonical_and_existing_game_rows(store, tmp_path):
    from footy_track.feature_store import GameRow

    assert resolve_game_ids("mancity_seg010", tmp_path) == ["arsenal_mancity_seg010"]
    store.upsert_games([GameRow(game_id="arsenal_mancity_20250925_seg010")])
    assert resolve_game_ids("arsenal_mancity_seg010", tmp_path, store) == [
        "arsenal_mancity_20250925_seg010"
    ]


def test_detections_join_through_alias(store, tmp_path):
    """Calibration lands on the ball GT's game_id so detections_on_pitch projects."""
    from footy_track.feature_store import DetectionRow

    alias = "arsenal_mancity_20250925_seg010"
    (tmp_path / f"{alias}.jsonl").write_text("")
    _write(tmp_path / "calibration_gt" / "arsenal_mancity_seg010.json", {"frames": {"0": _frame()}})
    ingest_calibration_dir(store, tmp_path)
    store.upsert_detections([
        DetectionRow(game_id=alias, frame_index=0, continuous_time_s=0, detection_id=0,
                     source="hand_label", run_id="gt_import_labeller", label="in_play_ball",
                     bbox_x=0.4, bbox_y=0.5, bbox_w=0.01, bbox_h=0.02)
    ])
    r = store.query("SELECT * FROM detections_on_pitch").iloc[0]
    assert r["projection_status"] == "ok" and r["calibration_source"] == "hand_label"


def test_reimport_replaces_cleanly(store, tmp_path):
    clip = "arsenal_mancity_seg010"
    f = tmp_path / "calibration_gt" / f"{clip}.json"
    _write(f, {"frames": {"0": _frame(), "349": _frame()}})
    ingest_calibration_dir(store, tmp_path)
    ingest_calibration_dir(store, tmp_path)  # idempotent
    assert store.count("calibration") == 2 and store.count("pitch_keypoint") == 6
    # frame 349 deleted, frame 0 corrected to 2 points
    _write(f, {"frames": {"0": _frame(correspondences=_frame()["correspondences"][:2], inliers=[True, True], err_px=[1, 2])}})
    r = ingest_calibration_dir(store, tmp_path)
    assert r.rows_deleted == 2
    assert list(store.query("SELECT frame_index FROM calibration")["frame_index"]) == [0]
    assert store.count("pitch_keypoint") == 2


def test_reimport_leaves_other_writers_alone(store, tmp_path):
    from footy_track.feature_store import CalibrationRow

    clip = "arsenal_mancity_seg010"
    store.upsert_calibrations([CalibrationRow(game_id=clip, source="classical", run_id="m", frame_index=5, H=[1.0] * 9, quality=0.9, dataset_tag=None)])
    _write(tmp_path / "calibration_gt" / f"{clip}.json", {"frames": {"0": _frame()}})
    _write(tmp_path / "calibration_proposals" / f"{clip}.json", _proposals())
    ingest_calibration_dir(store, tmp_path)
    ingest_calibration_dir(store, tmp_path)
    assert store.count("calibration") == 3  # classical + gt + proposal
    # GT replace does not delete the proposal importer's rows and vice versa
    _write(tmp_path / "calibration_gt" / f"{clip}.json", {"frames": {}})
    ingest_calibration_dir(store, tmp_path)
    assert store.count("calibration") == 2


def test_proposals_precedence_and_run_rows(store, tmp_path):
    clip = "arsenal_mancity_seg010"
    props = _proposals(frames={
        "0": _frame(status="proposed", score=0.7),
        "1": _frame(status="accepted", score=0.4),
        "2": _frame(status="rejected", score=0.99),
        "3": {"status": "failed", "score": 0.0, "failure": "no lines visible", "correspondences": []},
    })
    props["runs"]["cl_v2"] = {
        "source": "classical", "created": "2026-11-11T11:00:00+00:00", "params": {},
        "frames": {"0": _frame(status="proposed", score=0.9), "1": _frame(status="proposed", score=0.95)},
    }
    _write(tmp_path / "calibration_proposals" / f"{clip}.json", props)
    _write(tmp_path / "calibration_gt" / f"{clip}.json", {"frames": {"1": _frame(origin={"source": "claude", "run_id": "claude_v1", "proposal_status": "edited"})}})
    r = ingest_calibration_dir(store, tmp_path)
    assert r.proposal_frames == 6 and dict(r.by_status)["failed"] == 1
    best = _best(store)
    got = {int(a): (b, c, d) for a, b, c, d in zip(best.frame_index, best.source, best.run_id, best.tier)}
    assert got[0] == ("classical", "cl_v2", "proposed")  # higher score
    assert got[1] == ("hand_label", GT_RUN_ID, "hand")  # accepted-copied-to-GT wins
    assert 2 not in got and 3 not in got  # rejected / failed never win
    origin = store.query("SELECT origin_source, origin_run_id, origin_status FROM calibration WHERE source='hand_label'").iloc[0]
    assert tuple(origin) == ("claude", "claude_v1", "edited")
    failed = store.query("SELECT H, failure, quality FROM calibration WHERE review_status='failed'").iloc[0]
    assert store.query("SELECT H IS NULL AS n FROM calibration WHERE review_status='failed'")["n"][0] and failed["failure"] == "no lines visible"
    run = store.query("SELECT model_name, model_version, stage FROM run WHERE run_id='claude_v1'").iloc[0]
    assert tuple(run) == ("claude-x", "p3", "calibration")


def test_malformed_frames_warn_not_crash(store, tmp_path, caplog):
    clip = "arsenal_mancity_seg010"
    gt = {"frames": {
        "0": _frame(),
        "bad": _frame(),
        "5": _frame(H=[[1, 2], [3, 4]]),
        "6": _frame(H=None),
        "7": _frame(H_convention="pixels->pitch"),
        "8": "nonsense",
        "9": _frame(correspondences=[{"image": ["x", 1], "pitch_xy": [0, 0]}, {"image": [0.1, 0.2], "pitch_xy": [1, 2]}]),
    }}
    _write(tmp_path / "calibration_gt" / f"{clip}.json", gt)
    (tmp_path / "calibration_gt" / "broken.json").write_text("{not json")
    _write(tmp_path / "calibration_proposals" / f"{clip}.json", _proposals(frames={"0": _frame(status="wat")}))
    with caplog.at_level(logging.WARNING):
        r = ingest_calibration_dir(store, tmp_path)
    assert sorted(store.query("SELECT frame_index FROM calibration")["frame_index"]) == [0, 9]
    assert store.query("SELECT n_points FROM calibration WHERE frame_index=9")["n_points"][0] == 1
    assert r.skipped >= 6 and r.warnings
    assert "broken.json" in " ".join(r.warnings)


def test_all_malformed_keeps_previous_import(store, tmp_path):
    clip = "arsenal_mancity_seg010"
    f = tmp_path / "calibration_gt" / f"{clip}.json"
    _write(f, {"frames": {"0": _frame()}})
    ingest_calibration_dir(store, tmp_path)
    _write(f, {"frames": {"0": _frame(H=None)}})
    ingest_calibration_dir(store, tmp_path)
    assert store.count("calibration") == 1


def test_dry_run_writes_nothing(store, tmp_path):
    clip = "arsenal_mancity_seg010"
    _write(tmp_path / "calibration_gt" / f"{clip}.json", {"frames": {"0": _frame()}})
    r = ingest_calibration_dir(store, tmp_path, dry_run=True)
    assert r.gt_frames == 1 and r.keypoints == 3
    assert store.count("calibration") == 0 and store.count("run") == 0


def test_cli(tmp_path, capsys):
    clip = "arsenal_mancity_seg010"
    _write(tmp_path / "calibration_gt" / f"{clip}.json", {"frames": {"0": _frame()}})
    db = tmp_path / "s.duckdb"
    main(["--gt-dir", str(tmp_path), "--db", str(db), "--dry-run"])
    assert not db.exists()
    main(["--gt-dir", str(tmp_path), "--db", str(db), "--clip", "mancity_seg010"])
    assert "calibration=1" in capsys.readouterr().out
