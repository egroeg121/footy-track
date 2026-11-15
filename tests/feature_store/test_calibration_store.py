"""calibration / pitch_keypoint tables, frame_calibration_best and
detections_on_pitch (SQL projection must match PitchProjector)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from footy_track.feature_store import (
    CalibrationRow,
    DetectionRow,
    FeatureStore,
    FrameRow,
    GameRow,
    PitchKeypointRow,
)
from footy_track.projection.projector import (
    H_QUALITY_THRESHOLD,
    PitchProjector,
    TrackedDetection,
)

G = "clip_a"
H_ID = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
H_CAM = [0.31, -0.002, -0.10, -0.027, -0.79, 0.51, -0.0007, 0.011, 0.004]


def _cal(source, run_id, fi=0, status="proposed", q=0.9, H=H_ID, **kw):
    return CalibrationRow(
        game_id=kw.pop("game_id", G),
        source=source,
        run_id=run_id,
        frame_index=fi,
        H=H,
        quality=q,
        review_status=status,
        **kw,
    )


def _best(store, fi=0):
    df = store.query(
        "SELECT * FROM frame_calibration_best WHERE game_id=? AND frame_index=?",
        [G, fi],
    )
    return df.iloc[0] if len(df) else None


@pytest.fixture
def store():
    s = FeatureStore.open()
    yield s
    s.close()


def test_h_must_be_nine_values():
    with pytest.raises(ValueError):
        CalibrationRow(game_id=G, source="claude", run_id="r", frame_index=0, H=[1.0])
    with pytest.raises(ValueError):
        _cal("claude", "r", status="bogus")


def test_upsert_idempotent_and_keypoints(store):
    row = _cal("claude", "r1")
    store.upsert_calibrations([row])
    store.upsert_calibrations([row.model_copy(update={"quality": 0.5})])
    assert store.count("calibration") == 1
    assert _best(store)["quality"] == 0.5
    kp = PitchKeypointRow(
        game_id=G, source="claude", run_id="r1", frame_index=0, point_idx=0,
        keypoint_id="CENTRE_SPOT", u=0.5, v=0.5, pitch_x=0.0, pitch_y=0.0,
        is_inlier=True, err_px=1.0,
    )
    store.upsert_pitch_keypoints([kp, kp])
    assert store.count("pitch_keypoint") == 1


def test_precedence_hand_beats_accepted_beats_proposed(store):
    store.upsert_calibrations(
        [
            _cal("claude", "r1", q=0.99),
            _cal("classical", "r2", q=0.2),
        ]
    )
    assert _best(store)["source"] == "claude"  # highest-scoring proposal
    store.upsert_calibrations([_cal("classical", "r2", q=0.2, status="accepted")])
    b = _best(store)
    assert (b["source"], b["run_id"], b["tier"]) == ("classical", "r2", "accepted")
    store.upsert_calibrations(
        [_cal("hand_label", "calibration_gt_import", status="hand", q=1.0, reviewed=True)]
    )
    b = _best(store)
    assert (b["source"], b["tier"]) == ("hand_label", "hand")


def test_failed_and_rejected_never_win(store):
    store.upsert_calibrations(
        [
            _cal("claude", "r1", q=0.99, status="rejected"),
            _cal("classical", "r2", q=0.99, status="failed", H=None),
        ]
    )
    assert _best(store) is None
    store.upsert_calibrations([_cal("claude", "r3", q=0.01)])
    assert _best(store)["run_id"] == "r3"


def test_one_row_per_frame_and_ties_deterministic(store):
    store.upsert_calibrations(
        [_cal("claude", "a", fi=1, q=0.5), _cal("claude", "b", fi=1, q=0.5),
         _cal("claude", "a", fi=2, q=0.5)]
    )
    df = store.query("SELECT * FROM frame_calibration_best ORDER BY frame_index")
    assert list(df["frame_index"]) == [1, 2]
    assert df.iloc[0]["run_id"] == "b"  # tie -> run_id DESC


def test_inline_frame_columns_ignored_by_view(store):
    store.upsert_games([GameRow(game_id=G)])
    store.upsert_frames(
        [FrameRow(game_id=G, frame_index=0, frame_uri="f", width=1, height=1,
                  continuous_time_s=0.0, homography=H_ID, calibration_quality=1.0)]
    )
    assert _best(store) is None  # deprecated columns are not a source


def _det(fi, label, did, x, y, w, h):
    return DetectionRow(
        game_id=G, frame_index=fi, continuous_time_s=fi / 25, detection_id=did,
        source="yolo", run_id="d", label=label, bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
    )


def test_detections_on_pitch_matches_projector(store):
    H = np.array(H_CAM).reshape(3, 3)
    store.upsert_calibrations(
        [_cal("hand_label", "calibration_gt_import", H=H_CAM, q=0.8,
              status="hand", pitch_length=100.0, pitch_width=60.0)]
    )
    dets = [
        _det(0, "player", 0, 0.30, 0.40, 0.05, 0.12),
        _det(0, "referee", 1, 0.55, 0.35, 0.04, 0.10),
        _det(0, "ball", 2, 0.45, 0.50, 0.01, 0.02),
    ]
    store.upsert_detections(dets)
    df = store.query(
        "SELECT * FROM detections_on_pitch ORDER BY detection_id"
    )
    proj = PitchProjector(pitch_dims=(100.0, 60.0))
    # bbox columns are FLOAT: feed the projector the stored (rounded) values.
    stored = store.query("SELECT bbox_x, bbox_y, bbox_w, bbox_h, label FROM detection "
                         "ORDER BY detection_id")
    exp = proj.project_frame(
        [TrackedDetection(i, 0.0, r.label, r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h)
         for i, r in stored.iterrows()],
        H,
        0.8,
    )
    assert len(df) == 3
    for (_, got), want in zip(df.iterrows(), exp, strict=True):
        assert got["projection_status"] == "ok"
        assert got["x_pitch"] == pytest.approx(want.x_pitch, abs=1e-9)
        assert got["y_pitch"] == pytest.approx(want.y_pitch, abs=1e-9)
        assert got["x_pitch_norm"] == pytest.approx(want.x_pitch_norm, abs=1e-9)
        assert got["y_pitch_norm"] == pytest.approx(want.y_pitch_norm, abs=1e-9)
        assert got["calibration_source"] == "hand_label"


@pytest.mark.parametrize("label", ["in_play_ball", "out_of_play_ball"])
def test_gt_ball_labels_use_centroid_anchor(store, label):
    store.upsert_calibrations([_cal("hand_label", "r", H=H_ID, q=1.0, status="hand")])
    store.upsert_detections([_det(0, label, 0, 0.5, 0.25, 0.1, 0.5)])
    r = store.query("SELECT * FROM detections_on_pitch").iloc[0]
    assert r["x_pitch"] == pytest.approx(0.55)
    assert r["y_pitch"] == pytest.approx(0.5)  # centroid, not bottom (0.75)


def test_detections_on_pitch_nan_null_rules(store):
    # f0: low quality, f1: no calibration, f2: exactly at threshold -> projects
    store.upsert_calibrations(
        [_cal("claude", "r", fi=0, q=H_QUALITY_THRESHOLD - 0.01),
         _cal("claude", "r", fi=2, q=H_QUALITY_THRESHOLD)]
    )
    store.upsert_detections([_det(i, "player", 0, 0.2, 0.2, 0.1, 0.1) for i in range(3)])
    df = store.query("SELECT * FROM detections_on_pitch ORDER BY frame_index")
    assert df.loc[0, "projection_status"] == "low_quality"
    assert math.isnan(df.loc[0, "x_pitch"])
    assert df.loc[1, "projection_status"] == "no_calibration"
    assert df["x_pitch"].isna()[1]
    assert not math.isnan(df.loc[2, "x_pitch"])
    # the NaN row agrees with PitchProjector
    p = PitchProjector().project_frame(
        [TrackedDetection(0, 0.0, "player", 0.2, 0.2, 0.1, 0.1)],
        np.eye(3), H_QUALITY_THRESHOLD - 0.01,
    )
    assert math.isnan(p[0].x_pitch)


def test_parquet_roundtrip_preserves_calibration(store, tmp_path):
    store.upsert_calibrations([_cal("claude", "r1", H=H_CAM)])
    store.upsert_pitch_keypoints(
        [PitchKeypointRow(game_id=G, source="claude", run_id="r1", frame_index=0,
                          point_idx=0, u=0.1, v=0.2, pitch_x=3.0, pitch_y=4.0)]
    )
    store.export_parquet(tmp_path)
    s2 = FeatureStore.from_parquet(tmp_path)
    assert s2.count("calibration") == 1 and s2.count("pitch_keypoint") == 1
    assert list(_best(s2)["H"]) == H_CAM


def test_detections_on_pitch_undistorts_with_k1_like_the_projector(store):
    from footy_track.calibration import geometry as geo

    H = np.array(H_CAM).reshape(3, 3)
    k1 = 0.03
    store.upsert_calibrations(
        [_cal("hand_label", "r", H=H_CAM, q=0.9, status="hand", distortion_k1=k1,
              pitch_length=105.0, pitch_width=68.0, image_width=1920, image_height=1080),
         _cal("hand_label", "r", fi=1, H=H_CAM, q=0.9, status="hand",  # same H, no lens
              pitch_length=105.0, pitch_width=68.0, image_width=1920, image_height=1080)]
    )
    store.upsert_detections([_det(0, "player", 0, 0.10, 0.20, 0.05, 0.30),
                             _det(1, "player", 1, 0.10, 0.20, 0.05, 0.30)])
    df = store.query("SELECT * FROM detections_on_pitch ORDER BY frame_index")
    stored = store.query("SELECT bbox_x, bbox_y, bbox_w, bbox_h FROM detection ORDER BY frame_index")
    r = stored.iloc[0].astype(float)  # FLOAT32 columns: do the projector's arithmetic in float64
    want = PitchProjector().project_frame(
        [TrackedDetection(0, 0.0, "player", r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h)], H, 0.9,
        distortion=geo.make_distortion(k1, 1920 / 1080),
    )[0]
    assert df.iloc[0]["x_pitch"] == pytest.approx(want.x_pitch, abs=1e-9)
    assert df.iloc[0]["y_pitch"] == pytest.approx(want.y_pitch, abs=1e-9)
    plain = PitchProjector().project_frame(
        [TrackedDetection(0, 0.0, "player", r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h)], H, 0.9)[0]
    assert df.iloc[1]["x_pitch"] == pytest.approx(plain.x_pitch, abs=1e-9)  # NULL k1 == old behaviour
    assert abs(df.iloc[0]["x_pitch"] - df.iloc[1]["x_pitch"]) > 1e-3  # and the lens does matter


def test_store_created_before_distortion_gets_the_k1_column(tmp_path):
    """Re-opening an existing store adds calibration.distortion_k1 (NULL = plain homography)."""
    import duckdb

    from footy_track.feature_store.schema import DDL, VIEWS

    path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(path))
    for stmt in DDL:
        if stmt.lstrip().startswith("ALTER TABLE calibration"):
            continue
        con.execute(stmt.replace("distortion_k1  DOUBLE,", ""))  # the pre-distortion table
    con.execute("CREATE OR REPLACE VIEW frame_calibration_best AS SELECT * FROM calibration")
    con.execute("INSERT INTO calibration (game_id, source, run_id, frame_index, H, review_status) "
                f"VALUES ('g', 'hand_label', 'r', 0, {H_ID}, 'hand')")
    con.close()
    with FeatureStore.open(path) as s:
        row = s.query("SELECT distortion_k1 FROM frame_calibration_best").iloc[0]
        assert row["distortion_k1"] is None or math.isnan(row["distortion_k1"])


def test_not_broadcast_frame_has_no_calibration_and_no_projection(store):
    store.upsert_calibrations([
        _cal("claude", "r1", fi=0, q=0.9),
        _cal("hand_label", "calibration_gt_import", fi=0, q=1.0, status="hand"),
        _cal("claude", "r1", fi=1, q=0.9),
        # a flag row (no H), from a different run than the proposal it vetoes
        _cal("hand_label", "frame_type_import", fi=1, status="not_broadcast", H=None, q=None),
        # a proposal itself marked not_broadcast (with an H) never wins either
        _cal("claude", "r2", fi=2, q=0.99, status="not_broadcast"),
    ])
    assert _best(store, 0)["tier"] == "hand"
    assert _best(store, 1) is None and _best(store, 2) is None
    store.upsert_detections([_det(0, "player", 0, 0.4, 0.3, 0.05, 0.1), _det(1, "player", 1, 0.4, 0.3, 0.05, 0.1)])
    d = store.query("SELECT frame_index, projection_status, x_pitch FROM detections_on_pitch ORDER BY frame_index")
    assert list(d["projection_status"]) == ["ok", "no_calibration"] and d["x_pitch"].isna()[1]
