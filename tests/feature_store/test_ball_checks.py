"""Ball Check verdicts -> feature store -> training export (T-02)."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from footy_track.feature_store import FeatureStore
from footy_track.feature_store.ball_checks import import_ball_checks
from footy_track.feature_store.ingest_gt import ingest_gt_jsonl
from footy_track.scripts.export_training_dataset import (
    _query_ball_detections,
    export,
)

CLIP = "clip_a"


def _box(x: float) -> dict:
    return {"x": x, "y": 0.5, "w": 0.02, "h": 0.03}


def _setup(tmp_path):
    """Sidecar: frames 0-4, one vittrack ball each at x=0.1*(f+1), plus a
    hand_label player on frame 0 (so ball box_index on f0 is 1)."""
    recs = [{"frame_index": 0, "bbox": _box(0.9), "tags": ["player", "labeller"]}]
    for f in range(5):
        recs.append(
            {
                "frame_index": f,
                "bbox": _box(0.1 * (f + 1)),
                "tags": ["in_play_ball", "vittrack"],
            }
        )
    gt = tmp_path / "gt"
    gt.mkdir()
    (gt / f"{CLIP}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    store = FeatureStore.open(":memory:")
    ingest_gt_jsonl(store, gt / f"{CLIP}.jsonl", video_dir=tmp_path)
    bc = gt / "ball_checks"
    bc.mkdir()
    return store, gt, bc


def _v(frame, box_index, verdict, x, **kw):
    rec = {
        "clip": CLIP,
        "frame_index": frame,
        "box_index": box_index,
        "source": "sidecar",
        "verdict": verdict,
        "bbox": _box(x),
        "label": "in_play_ball",
        "corrected_bbox": None,
        "ts": 1.0,
    }
    rec.update(kw)
    return rec


def _write(bc, recs):
    (bc / f"{CLIP}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def _state(store):
    df = store.query(
        "SELECT frame_index, reviewed FROM detection WHERE source='vittrack' "
        "ORDER BY frame_index"
    )
    rev = dict(zip(df.frame_index, df.reviewed, strict=False))
    rej = set(
        store.query(
            "SELECT frame_index FROM detection_review WHERE verdict='not_ball' "
            "AND matched_detection_id IS NOT NULL"
        ).frame_index
    )
    return rev, rej


def _recs():
    return [
        _v(0, 1, "ball", 0.1),
        _v(1, 0, "not_ball", 0.2),
        _v(2, 0, "unsure", 0.3),
        # corrected: original box is the vittrack one, human dragged a new one
        _v(3, 0, "corrected", 0.4, corrected_bbox=_box(0.41)),
        # judged not_ball, then undone by a tombstone -> no effect
        _v(4, 0, "not_ball", 0.5),
        {
            "clip": CLIP,
            "frame_index": 4,
            "box_index": 0,
            "source": "sidecar",
            "verdict": None,
            "ts": 2.0,
        },
        # verdict on a box that no longer exists
        _v(1, 7, "not_ball", 0.77),
        # candidate-source verdict with no stored box
        _v(2, 3, "not_ball", 0.05, source="cand"),
    ]


def test_import_applies_verdicts(tmp_path) -> None:
    store, gt, bc = _setup(tmp_path)
    _write(bc, _recs())
    rep = import_ball_checks(store, bc)
    rev, rej = _state(store)
    assert rev == {0: True, 1: False, 2: False, 3: True, 4: False}
    assert rej == {1}
    assert rep.verdicts == 6  # tombstoned one is gone
    assert rep.matched == 4 and rep.unmatched == 2
    assert rep.rejected == 1
    # unmatched hard negatives are still recorded
    assert store.count("detection_review") == 6


def test_import_idempotent(tmp_path) -> None:
    store, gt, bc = _setup(tmp_path)
    _write(bc, _recs())
    import_ball_checks(store, bc)
    first = (
        _state(store),
        store.query("SELECT * FROM detection_review ORDER BY ALL")
        .fillna(-1)
        .to_dict("records"),
        store.count("detection"),
    )
    import_ball_checks(store, bc)
    second = (
        _state(store),
        store.query("SELECT * FROM detection_review ORDER BY ALL")
        .fillna(-1)
        .to_dict("records"),
        store.count("detection"),
    )
    assert first == second


def test_later_tombstone_retracts_on_rerun(tmp_path) -> None:
    store, gt, bc = _setup(tmp_path)
    _write(bc, [_v(1, 0, "not_ball", 0.2)])
    import_ball_checks(store, bc)
    assert _state(store)[1] == {1}
    _write(
        bc,
        [
            _v(1, 0, "not_ball", 0.2),
            {
                "clip": CLIP,
                "frame_index": 1,
                "box_index": 0,
                "source": "sidecar",
                "verdict": None,
                "ts": 3.0,
            },
        ],
    )
    import_ball_checks(store, bc)
    assert _state(store)[1] == set()


def test_cand_not_ball_never_rejects_hand_label(tmp_path) -> None:
    store, gt, bc = _setup(tmp_path)
    p = gt / "hand.jsonl"
    p.write_text(
        json.dumps(
            {"frame_index": 0, "bbox": _box(0.3), "tags": ["in_play_ball", "labeller"]}
        )
        + "\n"
    )
    ingest_gt_jsonl(store, p, video_dir=tmp_path)
    (bc / "hand.jsonl").write_text(
        json.dumps({**_v(0, 0, "not_ball", 0.3, source="cand"), "clip": "hand"}) + "\n"
    )
    rep = import_ball_checks(store, bc)
    assert rep.hand_label_conflicts == 1 and rep.rejected == 0


def test_export_drops_not_ball_keeps_others(tmp_path) -> None:
    store, gt, bc = _setup(tmp_path)
    images = tmp_path / "img"
    images.mkdir()
    for f in range(5):
        img = images / f"{CLIP}_{f:06d}.jpg"
        Image.new("RGB", (64, 48)).save(img)
        store.query(
            "UPDATE frame SET frame_uri=? WHERE game_id=? AND frame_index=?",
            [str(img), CLIP, f],
        )
    _write(bc, _recs())
    import_ball_checks(store, bc)

    def run(**kw):
        return export(
            store,
            out_dir=tmp_path / "out",
            video_dir=tmp_path,
            eval_dir=tmp_path / "none",
            extra_exclude=set(),
            val_fraction=0.0,
            tag="t",
            **kw,
        )

    def boxes(m):
        return m["train"]["boxes"] + m["val"]["boxes"]

    assert boxes(run()) == 4  # 5 balls minus the one not_ball
    assert boxes(run(include_rejected=True)) == 5


def test_parquet_round_trip_carries_detection_review(tmp_path) -> None:
    store, gt, bc = _setup(tmp_path)
    _write(bc, _recs())
    import_ball_checks(store, bc)
    expected = store.query("SELECT * FROM detection_review ORDER BY ALL")
    assert len(expected) == 6
    store.export_parquet(tmp_path / "pq")
    reloaded = FeatureStore.from_parquet(tmp_path / "pq")
    got = reloaded.query("SELECT * FROM detection_review ORDER BY ALL")
    assert got.equals(expected)
    # the primary key survives the reload (duplicate identity is rejected)
    with pytest.raises(Exception, match="(?i)constraint|duplicate"):
        reloaded._conn.execute(  # noqa: SLF001
            "INSERT INTO detection_review (game_id, vsource, frame_index, box_index) "
            "VALUES (?, 'sidecar', 0, 1)",
            [CLIP],
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TODO: match verdicts by overlap at export time, not stored "
        "detection_id (stale ids after re-ingest)"
    ),
)
def test_export_survives_reingest_after_boxes_shift(tmp_path) -> None:
    """A not_ball verdict must keep following its *box*, not its position.

    Frame 0 has three ball boxes A, B, C; the user marks B not_ball and the
    importer runs. Later the sidecar is edited (A deleted) and re-ingested, so
    B and C move up one positional detection_id. Without re-running the
    importer the export should still drop B and keep C. Today the verdict is
    pinned to the stored detection_id, so it points at the wrong box.
    """
    gt = tmp_path / "gt"
    gt.mkdir()
    bc = gt / "ball_checks"
    bc.mkdir()
    path = gt / f"{CLIP}.jsonl"

    def sidecar(xs):
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "frame_index": 0,
                        "bbox": _box(x),
                        "tags": ["in_play_ball", "vittrack"],
                    }
                )
                for x in xs
            )
            + "\n"
        )

    store = FeatureStore.open(":memory:")
    a, b, c = 0.1, 0.4, 0.7
    sidecar([a, b, c])
    ingest_gt_jsonl(store, path, video_dir=tmp_path)
    _write(bc, [_v(0, 1, "not_ball", b)])
    import_ball_checks(store, bc)

    sidecar([b, c])  # A removed: B -> id 0, C -> id 1
    ingest_gt_jsonl(store, path, video_dir=tmp_path)

    xs = {round(x, 2) for x in _query_ball_detections(store).bbox_x}
    assert round(b, 2) not in xs
    assert round(c, 2) in xs
