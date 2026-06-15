"""Tests for label-set importers (Roboflow YOLO + web-labeller/SAM3 JSON)."""

from __future__ import annotations

import json

import pytest
import yaml
from PIL import Image

from footy_track.feature_store import (
    FeatureStore,
    import_labeller_json,
    import_roboflow,
    source_overlap,
)
from footy_track.feature_store.importers import parse_labeller_uri, parse_roboflow_stem

# --------------------------------------------------------------------------- #
# filename parsing                                                            #
# --------------------------------------------------------------------------- #


def test_parse_roboflow_stem() -> None:
    stem, idx = parse_roboflow_stem(
        "arsenal_mancity_20250925_002143_png.rf.6b7c9d84.jpg"
    )
    assert stem == "arsenal_mancity_20250925"
    assert idx == 2143


def test_parse_roboflow_stem_plain() -> None:
    # a non-roboflow-mangled extract_frames name still parses
    assert parse_roboflow_stem("arsenal_mancity_000007.txt") == ("arsenal_mancity", 7)


def test_parse_labeller_uri() -> None:
    assert parse_labeller_uri("arsenal_mancity_example_video_frame_000000") == (
        "arsenal_mancity_example_video",
        0,
    )


# --------------------------------------------------------------------------- #
# fixtures: a tiny roboflow dataset and a labeller json on the SAME video      #
# --------------------------------------------------------------------------- #


def _make_roboflow(tmp_path, game="arsenal_demo", frames=(5, 6)):
    root = tmp_path / "roboflow_dataset_3"
    (root / "train" / "labels").mkdir(parents=True)
    (root / "train" / "images").mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "names": [
                    "ball",
                    "coach",
                    "in_play_ball",
                    "person",
                    "player",
                    "player_sub",
                    "referee",
                ],
                "nc": 7,
                "roboflow": {"version": 3},
            }
        )
    )
    for fi in frames:
        base = f"{game}_{fi:06d}_png.rf.deadbeef{fi}"
        # one player box, centre xywh -> should become top-left
        (root / "train" / "labels" / f"{base}.txt").write_text("4 0.5 0.5 0.2 0.4\n")
    return root


def _write_image_dims(root, game, frames, w=1920, h=1080):
    """Create real (tiny) images so the importer can read dims."""
    for fi in frames:
        base = f"{game}_{fi:06d}_png.rf.deadbeef{fi}"
        Image.new("RGB", (w, h)).save(root / "train" / "images" / f"{base}.jpg")


# --------------------------------------------------------------------------- #
# roboflow import                                                            #
# --------------------------------------------------------------------------- #


def test_import_roboflow_converts_centre_to_topleft(tmp_path) -> None:
    root = _make_roboflow(tmp_path, frames=(5,))
    store = FeatureStore.open(":memory:")
    report = import_roboflow(store, root, default_width=1920, default_height=1080)

    assert report.detections_written == 1
    assert report.sources == {"hand_label"}
    row = store.query(
        "SELECT label, bbox_x, bbox_y, bbox_w, bbox_h, source, confidence FROM detection"
    )
    assert row["label"][0] == "player"
    # centre (0.5,0.5) w0.2 h0.4 -> top-left (0.4, 0.3)
    assert row["bbox_x"][0] == pytest.approx(0.4)
    assert row["bbox_y"][0] == pytest.approx(0.3)
    assert row["source"][0] == "hand_label"
    assert row["confidence"][0] is None or row["confidence"].isna()[0]


def test_import_roboflow_reads_image_dims(tmp_path) -> None:
    root = _make_roboflow(tmp_path, game="arsenal_demo", frames=(5,))
    _write_image_dims(root, "arsenal_demo", (5,), w=1280, h=720)
    store = FeatureStore.open(":memory:")
    import_roboflow(store, root)
    dims = store.query("SELECT width, height FROM frame")
    assert int(dims["width"][0]) == 1280
    assert int(dims["height"][0]) == 720


def test_import_roboflow_game_id_and_frame_index(tmp_path) -> None:
    root = _make_roboflow(tmp_path, game="arsenal_demo", frames=(5, 6))
    store = FeatureStore.open(":memory:")
    import_roboflow(
        store, root, game_id="arsenal_demo", default_width=1920, default_height=1080
    )
    frames = store.query("SELECT frame_index FROM frame ORDER BY frame_index")
    assert list(frames["frame_index"]) == [5, 6]


def test_import_roboflow_is_idempotent(tmp_path) -> None:
    root = _make_roboflow(tmp_path, frames=(5, 6))
    store = FeatureStore.open(":memory:")
    import_roboflow(store, root, default_width=1920, default_height=1080)
    import_roboflow(store, root, default_width=1920, default_height=1080)
    assert store.count("detection") == 2
    assert store.count("frame") == 2


# --------------------------------------------------------------------------- #
# labeller/sam3 import                                                       #
# --------------------------------------------------------------------------- #


def _make_labeller_json(tmp_path, game="arsenal_demo", frames=(5, 6)):
    records = [
        {
            "uri": f"{game}_frame_{fi:06d}",
            "width": 1920,
            "height": 1080,
            "detections": [
                {
                    "label": "ball",
                    "confidence": 1.0,
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.02,
                    "h": 0.03,
                    "model": "sam3_video",
                }
            ],
        }
        for fi in frames
    ]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(records))
    return path


def test_import_labeller_json_default_source_sam3(tmp_path) -> None:
    path = _make_labeller_json(tmp_path, frames=(5,))
    store = FeatureStore.open(":memory:")
    report = import_labeller_json(
        store, path, run_id="sam3_sess1", game_id="arsenal_demo"
    )
    assert report.detections_written == 1
    src = store.query("SELECT source, label, bbox_x FROM detection")
    assert src["source"][0] == "sam3"
    assert src["label"][0] == "ball"
    assert src["bbox_x"][0] == pytest.approx(0.1)  # already top-left, no conversion


def test_import_labeller_json_as_ground_truth(tmp_path) -> None:
    path = _make_labeller_json(tmp_path, frames=(5,))
    store = FeatureStore.open(":memory:")
    import_labeller_json(
        store, path, run_id="verified_1", game_id="arsenal_demo", source="hand_label"
    )
    assert store.query("SELECT source FROM detection")["source"][0] == "hand_label"


# --------------------------------------------------------------------------- #
# combining the two + overlap report                                         #
# --------------------------------------------------------------------------- #


def test_roboflow_and_sam3_coexist_and_overlap_report(tmp_path) -> None:
    # both label sets target the SAME video + same frames -> they should align
    root = _make_roboflow(tmp_path, game="arsenal_demo", frames=(5, 6))
    json_path = _make_labeller_json(tmp_path, game="arsenal_demo", frames=(6, 7))

    store = FeatureStore.open(":memory:")
    import_roboflow(
        store, root, game_id="arsenal_demo", default_width=1920, default_height=1080
    )
    import_labeller_json(store, json_path, run_id="sam3_sess1", game_id="arsenal_demo")

    # frame 6 has BOTH hand_label and sam3; frames 5 and 7 have one each
    ov = source_overlap(store)
    by_frame = {int(r.frame_index): r for r in ov.itertuples()}
    assert by_frame[6].n_sources == 2
    assert by_frame[6].sources == "hand_label,sam3"
    assert by_frame[5].n_sources == 1
    assert by_frame[7].n_sources == 1

    # the two sources on frame 6 are distinct, comparable rows
    frame6 = store.query(
        "SELECT source FROM detection WHERE frame_index = 6 ORDER BY source"
    )
    assert list(frame6["source"]) == ["hand_label", "sam3"]
