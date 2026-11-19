"""Clustering geometry + crowded-frame ranking on a small in-file DuckDB store (no GPU, no video)."""

from __future__ import annotations

import pytest

from footy_track.autolabel.clusters import cluster_boxes, crowd_score
from footy_track.feature_store.schema import (
    DetectionRow,
    FrameLabelStateRow,
    FrameRow,
    GameRow,
    RunRow,
)
from footy_track.feature_store.store import FeatureStore
from footy_track.scripts import crowded_frames as cf

W, H = 1000, 1000


def box(x, y, w=50, h=100):
    return (x, y, w, h)


# ------------------------------ clustering ------------------------------ #


def test_overlapping_boxes_cluster():
    cs = cluster_boxes([box(0, 0), box(20, 0), box(40, 0), box(500, 500)])
    assert [c.indices for c in cs] == [(0, 1, 2), (3,)]
    assert cs[0].mean_iou > 0.2


def test_isolated_boxes_not_clustered():
    cs = cluster_boxes([box(0, 0), box(300, 0), box(0, 300)])
    assert [c.size for c in cs] == [1, 1, 1]
    assert crowd_score([box(0, 0), box(300, 0)]).largest_cluster == 1


def test_centre_distance_rule_links_without_iou():
    # Tall thin boxes stacked vertically: IoU 0 but centres 25px apart < 0.6 * 50.
    a, b = (0, 0, 50, 10), (0, 10, 50, 10)
    assert cluster_boxes([a, b])[0].size == 2
    far = (0, 40, 50, 10)  # centre dist 35 > 30, IoU 0
    assert [c.size for c in cluster_boxes([a, far])] == [1, 1]


def test_empty():
    assert cluster_boxes([]) == []
    assert crowd_score([]).score == 0.0


# ------------------------------ store fixture ------------------------------ #


def _det(game, frame, i, x, y, *, label="player", source="yolo", run="r1"):
    return DetectionRow(
        game_id=game, frame_index=frame, continuous_time_s=frame / 25, detection_id=i,
        source=source, run_id=run, label=label, bbox_x=x / W, bbox_y=y / H, bbox_w=0.05, bbox_h=0.1,
    )  # fmt: skip


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "fs.duckdb"
    store = FeatureStore.open(path)
    store.upsert_runs(
        [RunRow(run_id="r1", stage="detect", source="yolo", model_name="m")]
    )
    crowd = [(0, 0), (20, 0), (40, 0), (30, 20)]  # 4-cluster
    pair = [(0, 0), (20, 0), (500, 500)]
    frames = {
        # clip, frame -> boxes
        ("c1", 100): crowd,
        ("c1", 110): crowd,  # within min-gap of 100
        ("c1", 200): pair,
        ("c1", 300): crowd[:3],
        ("c1", 400): crowd,  # frame_complete -> excluded
        ("c1", 500): crowd,  # hand-labelled -> excluded
        ("c2", 100): [(0, 0), (300, 0)],  # isolated
        ("h1_c1", 100): crowd,  # heldout match
    }
    games = {g for g, _ in frames}
    store.upsert_games([GameRow(game_id=g) for g in games])
    store.upsert_frames([
        FrameRow(game_id=g, frame_index=f, frame_uri=f"{g}/{f}", width=W, height=H, continuous_time_s=f / 25)
        for g, f in frames
    ])  # fmt: skip
    dets = []
    for (g, f), bs in frames.items():
        for i, (x, y) in enumerate(bs):
            src = "hand_label" if (g, f) == ("c1", 500) and i == 0 else "yolo"
            dets.append(_det(g, f, i, x, y, source=src, run="r1"))
    # a non-person label must be ignored
    dets.append(_det("c2", 100, 9, 0, 0, label="ball"))
    store.upsert_detections(dets)
    store.upsert_frame_label_states([
        FrameLabelStateRow(game_id="c1", frame_index=400, cls_group=g, state="complete", method="hand_frame")
        for g in ("ball", "player")
    ])  # fmt: skip
    store.close()
    return path


def fake_split(game: str) -> str:
    return "eval_heldout_match" if game.startswith("h1") else "train"


def run(db, **kw):
    store = cf.open_readonly(str(db))
    try:
        return cf.rank_frames(store, split_fn=fake_split, **kw)
    finally:
        store.close()


def test_ranking_order_and_exclusions(db):
    rows = run(db, min_gap=25)
    keys = [(r["clip"], r["frame_index"]) for r in rows]
    assert keys[0] == ("c1", 100)  # biggest cluster first
    assert ("c1", 400) not in keys  # frame_complete
    assert ("c1", 500) not in keys  # hand-labelled
    scores = [r["crowd_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert rows[0]["largest_cluster"] == 4 and rows[0]["n_persons"] == 4
    assert keys.index(("c1", 300)) < keys.index(("c1", 200)) < keys.index(("c2", 100))


def test_min_gap_diversity(db):
    keys = [(r["clip"], r["frame_index"]) for r in run(db, min_gap=25)]
    assert ("c1", 110) not in keys
    keys0 = [(r["clip"], r["frame_index"]) for r in run(db, min_gap=1)]
    assert ("c1", 110) in keys0


def test_split_filtering(db):
    train = run(db, split="train")
    assert {r["split"] for r in train} == {"train"}
    assert all(not r["clip"].startswith("h1") for r in train)
    held = run(db, split="eval_heldout_match")
    assert [r["clip"] for r in held] == ["h1_c1"]
    assert {r["clip"] for r in run(db, split="all")} >= {"c1", "c2", "h1_c1"}


def test_top_and_clips(db):
    assert len(run(db, top=2)) == 2
    assert {r["clip"] for r in run(db, clips={"c2"})} == {"c2"}


def test_cli_csv_and_deep_link(db, tmp_path, capsys):
    out = tmp_path / "o.csv"
    rc = cf.main(
        ["--db", str(db), "--out", str(out), "--dataset", "calib", "--split", "all"]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "clip=c1&frame=100" in text and "dataset=calib" in text
    assert out.read_text().splitlines()[0].startswith("clip,frame_index,crowd_score")
