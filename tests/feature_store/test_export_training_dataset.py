"""Tests for the training-dataset exporter (export_training_dataset)."""

from __future__ import annotations

import json

import yaml
from PIL import Image

from footy_track.feature_store import FeatureStore, import_labeller_json
from footy_track.feature_store.ingest_gt import ingest_gt_jsonl
from footy_track.scripts.export_training_dataset import (
    _distinct_labels,
    _split_clips,
    _split_clips_three_way,
    export,
    main,
)


def _seed_clip(store, tmp_path, stem, images_dir, frames):
    """Write a GT jsonl with real backing images, ingest it, then point the
    frame_uri at the real images so the exporter can copy pixels."""
    records = [
        {
            "frame_index": fi,
            "bbox": {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03},
            "tags": ["in_play_ball", "labeller"],
        }
        for fi in frames
    ]
    path = tmp_path / f"{stem}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    ingest_gt_jsonl(store, path, video_dir=tmp_path)
    # create real images and rewrite frame_uri to point at them
    for fi in frames:
        img = images_dir / f"{stem}_{fi:06d}.jpg"
        Image.new("RGB", (64, 48)).save(img)
        store.query(
            "UPDATE frame SET frame_uri = ? WHERE game_id = ? AND frame_index = ?",
            [str(img), stem, fi],
        )


def test_split_clips_no_overlap() -> None:
    clips = [f"c{i}" for i in range(10)]
    train, val = _split_clips(clips, 0.2)
    assert not (train & val)
    assert train | val == set(clips)
    assert len(val) == 2


def test_split_clips_three_way_no_overlap_and_fractions() -> None:
    clips = [f"c{i}" for i in range(20)]
    train, val, test = _split_clips_three_way(clips, 0.2, 0.1)
    # No clip in more than one split, all pairs disjoint.
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == set(clips)
    assert len(val) == 4  # round(20 * 0.2)
    assert len(test) == 2  # round(20 * 0.1)
    assert len(train) == 14


def test_split_clips_three_way_test_fraction_zero_matches_two_way() -> None:
    """test_fraction=0 must reduce exactly to the two-way split (back-compat)."""
    clips = [f"c{i}" for i in range(11)]
    train2, val2 = _split_clips(clips, 0.2)
    train3, val3, test3 = _split_clips_three_way(clips, 0.2, 0.0)
    assert train3 == train2
    assert val3 == val2
    assert test3 == set()


def test_split_clips_three_way_small_and_zero_clips() -> None:
    assert _split_clips_three_way([], 0.2, 0.1) == (set(), set(), set())
    # single clip: no split possible (mirrors _split_clips's len(clips) > 1 guard)
    train, val, test = _split_clips_three_way(["only"], 0.2, 0.1)
    assert train == {"only"}
    assert val == set()
    assert test == set()


def test_export_three_way_split_end_to_end(tmp_path) -> None:
    """--test-fraction > 0 produces a genuine three-way by-clip split with no
    clip straddling any pair of splits, plus test dirs/data.yaml/manifest."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    for i in range(10):
        _seed_clip(store, tmp_path, f"clip_{i}", images_dir, frames=(0, 1))

    out = tmp_path / "ball_v1"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = export(
        store,
        out_dir=out,
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        val_fraction=0.2,
        test_fraction=0.1,
        tag="ball_v1",
    )

    train_clips = set(manifest["train_clips"])
    val_clips = set(manifest["val_clips"])
    test_clips = set(manifest["test_clips"])
    assert not (train_clips & val_clips)
    assert not (train_clips & test_clips)
    assert not (val_clips & test_clips)
    assert train_clips | val_clips | test_clips == {f"clip_{i}" for i in range(10)}
    assert len(test_clips) == 1  # round(10 * 0.1)
    assert len(val_clips) == 2  # round(10 * 0.2)

    # data.yaml has all three splits.
    dy = yaml.safe_load((out / "data.yaml").read_text())
    assert dy["train"] == "images/train"
    assert dy["val"] == "images/val"
    assert dy["test"] == "images/test"

    # Output dirs exist for all three splits.
    for split in ("train", "val", "test"):
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()

    # test split actually has images/boxes.
    assert manifest["test"]["images"] > 0
    assert manifest["test"]["boxes"] > 0

    # dataset_tag write-back covers the test split.
    tagged_test = store.query(
        "SELECT DISTINCT dataset_tag FROM detection WHERE dataset_tag = 'ball_v1:test'"
    )
    assert len(tagged_test) == 1


def test_export_test_fraction_zero_back_compat_default(tmp_path) -> None:
    """Default (test_fraction=0.0) leaves the manifest/data.yaml/dirs exactly
    as before: no 'test' key, no test dirs."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    for i in range(5):
        _seed_clip(store, tmp_path, f"clip_{i}", images_dir, frames=(0, 1, 2))

    out = tmp_path / "ball_v1"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = export(
        store,
        out_dir=out,
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        val_fraction=0.2,
        tag="ball_v1",
    )

    assert "test" not in manifest
    assert "test_clips" not in manifest
    dy = yaml.safe_load((out / "data.yaml").read_text())
    assert "test" not in dy
    assert not (out / "images" / "test").exists()
    assert not (out / "labels" / "test").exists()


def test_export_end_to_end_by_clip(tmp_path) -> None:
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    for i in range(5):
        _seed_clip(store, tmp_path, f"clip_{i}", images_dir, frames=(0, 1, 2))

    out = tmp_path / "ball_v1"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = export(
        store,
        out_dir=out,
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        val_fraction=0.2,
        tag="ball_v1",
    )

    # by-clip split, no clip in both
    assert not (set(manifest["train_clips"]) & set(manifest["val_clips"]))
    total_boxes = manifest["train"]["boxes"] + manifest["val"]["boxes"]
    assert total_boxes == 15  # 5 clips * 3 frames * 1 ball

    # YOLO layout present + data.yaml valid
    dy = yaml.safe_load((out / "data.yaml").read_text())
    assert dy["names"] == ["ball"]
    assert dy["nc"] == 1
    assert (out / "labels" / "train").is_dir()
    # a label file has YOLO center-xywh: 0.5,0.5 topleft w0.02 -> center 0.51
    any_label = next((out / "labels" / "train").glob("*.txt"))
    cls, cx, cy, w, h = any_label.read_text().split()
    assert cls == "0"
    assert abs(float(cx) - 0.51) < 1e-6


def test_export_excludes_eval_clips(tmp_path) -> None:
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    _seed_clip(store, tmp_path, "train_clip", images_dir, frames=(0, 1))
    _seed_clip(store, tmp_path, "eval_clip", images_dir, frames=(0, 1))

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "eval_clip.jsonl").write_text("{}\n")  # sidecar marks it as eval

    out = tmp_path / "ball_v1"
    manifest = export(
        store,
        out_dir=out,
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        val_fraction=0.5,
        tag="ball_v1",
    )

    all_split_clips = set(manifest["train_clips"]) | set(manifest["val_clips"])
    assert "eval_clip" not in all_split_clips
    assert "eval_clip" in manifest["excluded_eval_clips"]
    assert manifest["excluded_ball_boxes"] == 2  # eval_clip's two ball boxes


# --------------------------------------------------------------------------- #
# Multi-class / --sources tests                                               #
# --------------------------------------------------------------------------- #


def _seed_multiclass_clip(
    store, tmp_path, images_dir, stem, frames_spec, *, source="hand_label"
):
    """Seed one clip with arbitrary-label detections via the labeller import
    path (top-left boxes stored verbatim), then point every frame_uri at a
    real image so the exporter can copy pixels.

    ``frames_spec``: {frame_index: [(label, x, y, w, h), ...]} (top-left xywh).
    """
    records = [
        {
            "uri": f"{stem}_frame_{fi:06d}",
            "width": 64,
            "height": 48,
            "detections": [
                {"label": label, "confidence": 1.0, "x": x, "y": y, "w": w, "h": h}
                for label, x, y, w, h in dets
            ],
        }
        for fi, dets in frames_spec.items()
    ]
    path = tmp_path / f"{stem}_{source}.json"
    path.write_text(json.dumps(records))
    import_labeller_json(
        store, path, run_id=f"{source}_{stem}", game_id=stem, source=source
    )
    for fi in frames_spec:
        img = images_dir / f"{stem}_{fi:06d}.jpg"
        if not img.exists():
            Image.new("RGB", (64, 48)).save(img)
        store.query(
            "UPDATE frame SET frame_uri = ? WHERE game_id = ? AND frame_index = ?",
            [str(img), stem, fi],
        )


def _labels_by_frame(out_dir, names):
    """Read every exported label file -> {stem: [(class_name, cx, cy, w, h), ...]}."""
    result = {}
    for split in ("train", "val"):
        for txt in (out_dir / "labels" / split).glob("*.txt"):
            lines = txt.read_text().strip().splitlines()
            parsed = []
            for line in lines:
                cls, cx, cy, w, h = line.split()
                parsed.append(
                    (names[int(cls)], float(cx), float(cy), float(w), float(h))
                )
            result[txt.stem] = parsed
    return result


def test_export_multiclass_mapping_and_label_lines(tmp_path) -> None:
    """Multi-class mode: sorted class-id mapping in data.yaml, per-box class
    ids correct, frames with none of the requested classes skipped, and
    non-requested labels on mixed frames dropped."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    # Two clips so the by-clip split machinery is exercised.
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_a",
        {
            0: [("player", 0.1, 0.1, 0.2, 0.3), ("referee", 0.5, 0.5, 0.05, 0.1)],
            1: [("player", 0.3, 0.3, 0.2, 0.3), ("coach", 0.6, 0.6, 0.1, 0.2)],
            2: [("coach", 0.2, 0.2, 0.1, 0.2)],  # only non-requested -> frame skipped
        },
    )
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_b",
        {0: [("referee", 0.4, 0.4, 0.05, 0.1)], 1: [("player", 0.2, 0.5, 0.2, 0.3)]},
    )

    out = tmp_path / "mc_v1"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = export(
        store,
        out_dir=out,
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        val_fraction=0.5,
        tag="mc_v1",
        classes=["referee", "player"],  # deliberately unsorted input
    )

    dy = yaml.safe_load((out / "data.yaml").read_text())
    assert dy["names"] == ["player", "referee"]  # sorted, deterministic
    assert dy["nc"] == 2
    assert manifest["classes"] == ["player", "referee"]

    by_frame = _labels_by_frame(out, dy["names"])
    # clip_a frame 0: player + referee, in authored (detection_id) order
    f0 = by_frame["clip_a_000000"]
    assert [d[0] for d in f0] == ["player", "referee"]
    # centre conversion: player topleft (0.1,0.1,w0.2,h0.3) -> centre (0.2,0.25)
    assert abs(f0[0][1] - 0.2) < 1e-5 and abs(f0[0][2] - 0.25) < 1e-5
    # clip_a frame 1: coach dropped, only the player line remains
    assert [d[0] for d in by_frame["clip_a_000001"]] == ["player"]
    # clip_a frame 2 (coach-only) must not be exported at all
    assert "clip_a_000002" not in by_frame
    for split in ("train", "val"):
        assert not (out / "images" / split / "clip_a_000002.jpg").exists()
    # leakage: no clip in both splits
    assert not (set(manifest["train_clips"]) & set(manifest["val_clips"]))


def test_export_sources_filtering(tmp_path) -> None:
    """--sources restricts the export to the given detection sources; the
    default (None) keeps every canonical source."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_hand",
        {0: [("player", 0.1, 0.1, 0.2, 0.3)], 1: [("player", 0.2, 0.2, 0.2, 0.3)]},
        source="hand_label",
    )
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_vit",
        {0: [("player", 0.3, 0.3, 0.2, 0.3)]},
        source="vittrack",
    )

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    # Default: both sources exported.
    m_all = export(
        store,
        out_dir=tmp_path / "all",
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        val_fraction=0.5,
        tag="all",
        classes=["player"],
    )
    assert m_all["sources"] is None
    assert set(m_all["train_clips"]) | set(m_all["val_clips"]) == {
        "clip_hand",
        "clip_vit",
    }
    total = m_all["train"]["boxes"] + m_all["val"]["boxes"]
    assert total == 3

    # hand_label only: vittrack clip disappears entirely.
    m_hand = export(
        store,
        out_dir=tmp_path / "hand",
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        val_fraction=0.5,
        tag="hand",
        classes=["player"],
        sources=["hand_label"],
    )
    assert m_hand["sources"] == ["hand_label"]
    assert set(m_hand["train_clips"]) | set(m_hand["val_clips"]) == {"clip_hand"}
    by_source = dict(m_hand["train"]["by_source"]) | dict(m_hand["val"]["by_source"])
    assert set(by_source) == {"hand_label"}
    assert m_hand["train"]["boxes"] + m_hand["val"]["boxes"] == 2


def test_export_sources_filtering_ball_mode(tmp_path) -> None:
    """--sources also applies in the default ball mode without changing the
    single-class output contract."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_hand",
        {0: [("ball", 0.5, 0.5, 0.02, 0.03)]},
        source="hand_label",
    )
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_vit",
        {0: [("ball", 0.4, 0.4, 0.02, 0.03)]},
        source="vittrack",
    )

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = export(
        store,
        out_dir=tmp_path / "ball_hand",
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        val_fraction=0.5,
        tag="ball_hand",
        sources=["hand_label"],
    )
    dy = yaml.safe_load((tmp_path / "ball_hand" / "data.yaml").read_text())
    assert dy["names"] == ["ball"] and dy["nc"] == 1
    assert set(manifest["train_clips"]) | set(manifest["val_clips"]) == {"clip_hand"}


def test_export_default_back_compat_unchanged(tmp_path) -> None:
    """No --classes/--sources: identical contract to the original ball export
    (single 'ball' class id 0, ball-label collapse), even with non-ball labels
    present in the store."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    _seed_clip(store, tmp_path, "clip_ball", images_dir, frames=(0, 1))  # in_play_ball
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_people",
        {0: [("player", 0.1, 0.1, 0.2, 0.3)]},
    )

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    out = tmp_path / "ball_v1"
    manifest = export(
        store,
        out_dir=out,
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        val_fraction=0.5,
        tag="ball_v1",
    )
    dy = yaml.safe_load((out / "data.yaml").read_text())
    assert dy["names"] == ["ball"] and dy["nc"] == 1
    # only the ball clip exported; player-only clip has no ball labels
    assert set(manifest["train_clips"]) | set(manifest["val_clips"]) == {"clip_ball"}
    for parsed in _labels_by_frame(out, dy["names"]).values():
        assert all(d[0] == "ball" for d in parsed)


def test_cli_classes_all_and_sources(tmp_path) -> None:
    """`--classes all` resolves to every distinct canonical label; `--sources`
    parses as a comma list. Exercised through main() with a real db file."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    db_path = tmp_path / "fs.duckdb"
    store = FeatureStore.open(db_path)
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_a",
        {
            0: [("player", 0.1, 0.1, 0.2, 0.3), ("ball", 0.5, 0.5, 0.02, 0.03)],
            1: [("referee", 0.4, 0.4, 0.05, 0.1)],
        },
    )
    assert _distinct_labels(store) == ["ball", "player", "referee"]
    store.close()

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    out = tmp_path / "all_v1"
    main(
        [
            "--db",
            str(db_path),
            "--video-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--eval-dir",
            str(eval_dir),
            "--tag",
            "all_v1",
            "--classes",
            "all",
            "--sources",
            "hand_label,vittrack",
        ]
    )

    dy = yaml.safe_load((out / "data.yaml").read_text())
    assert dy["names"] == ["ball", "player", "referee"]
    assert dy["nc"] == 3
    manifest = yaml.safe_load((out / "manifest.yaml").read_text())
    assert manifest["sources"] == ["hand_label", "vittrack"]
    assert manifest["train"]["boxes"] + manifest["val"]["boxes"] == 3
