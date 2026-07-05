"""Tests for the ball training-dataset exporter (export_training_dataset)."""

from __future__ import annotations

import json

import yaml
from PIL import Image

from footy_track.feature_store import FeatureStore
from footy_track.feature_store.ingest_gt import ingest_gt_jsonl
from footy_track.scripts.export_training_dataset import _split_clips, export


def _seed_clip(store, tmp_path, stem, images_dir, frames):
    """Write a GT jsonl with real backing images, ingest it, then point the
    frame_uri at the real images so the exporter can copy pixels."""
    records = [
        {"frame_index": fi, "bbox": {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03},
         "tags": ["in_play_ball", "labeller"]}
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
