"""Tests for the training-dataset exporter (export_training_dataset)."""

from __future__ import annotations

import json

import pytest
import yaml
from PIL import Image

from footy_track.feature_store import FeatureStore, import_labeller_json
from footy_track.feature_store.schema import (
    ClsGroup,
    FrameLabelStateRow,
    LabelMethod,
    LabelState,
)
from footy_track.feature_store.ingest_gt import ingest_gt_jsonl
from footy_track import splits as authority
from footy_track.constants import FROZEN_CLASS_IDS, FROZEN_CLASS_NAMES
from footy_track.scripts.export_training_dataset import (
    _distinct_labels,
    export,
    main,
)


@pytest.fixture(autouse=True)
def _half_eval_register(tmp_path, monkeypatch):
    """Clip register with a 1-in-2 hash rule so small fixtures see both train and val clips.
    (Production rate is 1 in 5; the rule is read from the register file.)"""
    reg = tmp_path / "clip_splits.json"
    reg.write_text(
        json.dumps(
            {"version": 1, "rule": {"modulus": 2, "eval_remainder": 0}, "clips": {}}
        )
    )
    monkeypatch.setenv("FOOTY_CLIP_SPLITS", str(reg))
    authority.clear_cache()
    yield
    authority.clear_cache()


def mark_complete(
    store, game_id=None, *, ball=LabelState.COMPLETE, player=LabelState.COMPLETE
):
    """Mark every frame (of ``game_id``, default all) fully labelled so the exporter's
    ``frame_complete`` filter lets it through (nothing writes player completeness yet)."""
    sql, params = "SELECT game_id, frame_index FROM frame", []
    if game_id:
        sql, params = sql + " WHERE game_id = ?", [game_id]
    rows = []
    for r in store.query(sql, params).itertuples():
        for grp, st in ((ClsGroup.BALL, ball), (ClsGroup.PLAYER, player)):
            rows.append(
                FrameLabelStateRow(
                    game_id=r.game_id,
                    frame_index=int(r.frame_index),
                    cls_group=grp,
                    state=st,
                    method=LabelMethod.HAND_FRAME,
                )
            )
    store.upsert_frame_label_states(rows)


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
    mark_complete(store, stem)


def _split_dir_of(manifest: dict, clip: str) -> str | None:
    for s in ("train", "val", "test"):
        if clip in manifest.get(f"{s}_clips", []):
            return s
    return None


def test_export_follows_the_authority(tmp_path) -> None:
    """val = in-match eval clips, train = the rest; nothing is positional."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    names = [f"clip_{i}" for i in range(12)]
    for n in names:
        _seed_clip(store, tmp_path, n, images_dir, frames=(0, 1))
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = export(
        store,
        out_dir=tmp_path / "o",
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        tag="t",
    )
    for n in names:
        want = (
            "val" if authority.split_of_name(n) == authority.EVAL_IN_MATCH else "train"
        )
        assert _split_dir_of(manifest, n) == want
    assert manifest["val_clips"] and manifest["train_clips"]
    assert manifest["split_authority"]["val"] == "eval_in_match"


def test_export_split_is_stable_when_clips_are_added(tmp_path) -> None:
    """The old position-based picker reshuffled everything when a clip appeared."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    store = FeatureStore.open(":memory:")
    for i in range(8):
        _seed_clip(store, tmp_path, f"clip_{i}", images_dir, frames=(0,))
    kw = dict(video_dir=tmp_path, eval_dir=eval_dir, extra_exclude=set(), tag="t")
    before = export(store, out_dir=tmp_path / "o1", **kw)
    for i in range(8, 14):
        _seed_clip(store, tmp_path, f"clip_{i}", images_dir, frames=(0,))
    after = export(store, out_dir=tmp_path / "o2", **kw)
    assert set(before["val_clips"]) <= set(after["val_clips"])
    assert set(before["train_clips"]) <= set(after["train_clips"])


def test_export_heldout_match_dropped_or_test(tmp_path) -> None:
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    store = FeatureStore.open(":memory:")
    _seed_clip(
        store, tmp_path, "arsenal_nottinghamforest_seg012", images_dir, frames=(0, 1)
    )
    _seed_clip(store, tmp_path, "clip_a", images_dir, frames=(0,))
    kw = dict(video_dir=tmp_path, eval_dir=eval_dir, extra_exclude=set(), tag="t")

    m = export(store, out_dir=tmp_path / "o1", **kw)
    exported = set(m["train_clips"]) | set(m["val_clips"])
    assert exported == {"clip_a"}
    assert "test" not in m and m["split_authority"]["heldout_excluded_boxes"] == 2
    assert not (tmp_path / "o1" / "images" / "test").exists()

    m = export(store, out_dir=tmp_path / "o2", heldout_as_test=True, **kw)
    assert m["test_clips"] == ["arsenal_nottinghamforest_seg012"]
    assert m["test"]["images"] == 2
    dy = yaml.safe_load((tmp_path / "o2" / "data.yaml").read_text())
    assert dy["test"] == "images/test"
    assert (
        store.query("SELECT count(*) AS n FROM detection WHERE dataset_tag = 't:test'")[
            "n"
        ][0]
        == 2
    )


def test_export_fails_closed_without_clip_register(tmp_path, monkeypatch) -> None:
    from footy_track.match_splits import RegisterError

    monkeypatch.setenv("FOOTY_CLIP_SPLITS", str(tmp_path / "absent.json"))
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED")
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text("{}")
    authority.clear_cache()
    with pytest.raises(RegisterError, match="clip split register"):
        export(
            None,
            out_dir=tmp_path / "o",
            video_dir=tmp_path,
            eval_dir=tmp_path,
            extra_exclude=set(),
            tag="t",
        )
    assert not (tmp_path / "o").exists()


def test_export_default_has_no_test_split(tmp_path) -> None:
    """Without --heldout-as-test: no 'test' key, no test dirs."""
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
        tag="ball_v1",
    )

    all_split_clips = set(manifest["train_clips"]) | set(manifest["val_clips"])
    assert "eval_clip" not in all_split_clips
    assert "eval_clip" in manifest["excluded_eval_clips"]
    assert manifest["excluded_ball_boxes"] == 2  # eval_clip's two ball boxes


def _split_of(manifest: dict, clip: str) -> str:
    return next(
        s for s in ("train", "val", "test") if clip in manifest.get(f"{s}_clips", [])
    )


def test_export_keeps_aliases_of_one_clip_on_one_side_and_excludes_eval_aliases(
    tmp_path,
) -> None:
    """T-03: aliases of one physical clip never straddle splits, and an eval
    clip stored under another alias is still excluded."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    # Real alias forms seen on /mnt/storage/footy_data (same video each pair).
    pairs = [
        ("arsenal_mancity_20250925_seg020", "arsenal_mancity_seg020"),
        ("arsenal_mancity_20250925_seg031", "mancity_seg031"),
        ("astonvilla_seg050", "Arsenal - Aston Villa_seg050"),
        ("bournemouth_seg009", "bournemouth_1st_seg009"),
        ("bournemouth_broadcast_seg015", "Bournemouth vs Arsenal 1_seg015"),
        ("arsenal_mancity_20250925_part050", "mancity_part050"),
    ]
    for a, b in pairs:
        _seed_clip(store, tmp_path, a, images_dir, frames=(0,))
        _seed_clip(store, tmp_path, b, images_dir, frames=(1,))
    # The eval clip, stored only under its raw (date-stamped) alias.
    _seed_clip(
        store, tmp_path, "arsenal_mancity_20250925_seg010", images_dir, frames=(0, 1)
    )

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "mancity_seg010.jsonl").write_text("{}\n")

    for heldout_as_test in (False, True):
        manifest = export(
            store,
            out_dir=tmp_path / f"out_{heldout_as_test}",
            video_dir=tmp_path,
            eval_dir=eval_dir,
            extra_exclude=set(),
            heldout_as_test=heldout_as_test,
            tag="ball_v1",
        )
        exported = (
            set(manifest["train_clips"])
            | set(manifest["val_clips"])
            | set(manifest.get("test_clips", []))
        )
        assert "arsenal_mancity_20250925_seg010" not in exported
        assert manifest["excluded_ball_boxes"] == 2
        assert exported == {c for pair in pairs for c in pair}
        assert manifest["val_clips"]  # the split really happened
        for a, b in pairs:
            assert _split_of(manifest, a) == _split_of(manifest, b), (a, b)
        assert manifest["clip_aliases"]["arsenal_mancity_seg020"] == sorted(pairs[0])


def test_export_alias_free_store_splits_by_the_authority(tmp_path) -> None:
    """Grouping by canonical id must not move clips when there are no aliases:
    groups are keyed by their smallest game_id, i.e. the game_id itself."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    stems = ["astonvilla_seg020", "arsenal_mancity_seg090", "bournemouth_seg035"]
    stems += [f"clip_{i}" for i in range(5)]
    for stem in stems:
        _seed_clip(store, tmp_path, stem, images_dir, frames=(0,))
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    manifest = export(
        store,
        out_dir=tmp_path / "out",
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        tag="ball_v1",
    )
    val = {n for n in stems if authority.split_of_name(n) == authority.EVAL_IN_MATCH}
    assert set(manifest["train_clips"]) == set(stems) - val
    assert set(manifest["val_clips"]) == val
    assert manifest["clip_aliases"] == {}


# --------------------------------------------------------------------------- #
# Multi-class / --sources tests                                               #
# --------------------------------------------------------------------------- #


def _seed_multiclass_clip(
    store,
    tmp_path,
    images_dir,
    stem,
    frames_spec,
    *,
    source="hand_label",
    complete=True,
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
    if complete:
        mark_complete(store, stem)


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
        tag="mc_v1",
        classes=["referee", "player"],  # deliberately unsorted input
    )

    dy = yaml.safe_load((out / "data.yaml").read_text())
    assert dy["names"] == list(FROZEN_CLASS_NAMES)  # frozen ids, not sorted-by-request
    assert dy["nc"] == len(FROZEN_CLASS_NAMES)
    assert manifest["classes"] == list(FROZEN_CLASS_NAMES)

    by_frame = _labels_by_frame(out, dy["names"])
    # clip_a frame 0: player + referee, in authored (detection_id) order
    f0 = by_frame["clip_a_000000"]
    assert [d[0] for d in f0] == ["player", "referee"]
    # centre conversion: player topleft (0.1,0.1,w0.2,h0.3) -> centre (0.2,0.25)
    assert abs(f0[0][1] - 0.2) < 1e-5 and abs(f0[0][2] - 0.25) < 1e-5
    # clip_a frame 1 carries an unrequested coach (an unlabelled object): skipped, not written short
    assert "clip_a_000001" not in by_frame
    assert manifest["frames"]["skipped_by_reason"]["unrequested_labels"] == 2
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

    # "all": both sources exported (explicit; the default is hand_label + reviewed).
    m_all = export(
        store,
        out_dir=tmp_path / "all",
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        tag="all",
        classes=["player"],
        sources=["all"],
    )
    assert m_all["sources"] == ["all"]
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
    mark_complete(
        store, "clip_people", ball=LabelState.ABSENT
    )  # players only: ball-absent negative

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    out = tmp_path / "ball_v1"
    manifest = export(
        store,
        out_dir=out,
        video_dir=tmp_path,
        eval_dir=eval_dir,
        extra_exclude=set(),
        tag="ball_v1",
    )
    dy = yaml.safe_load((out / "data.yaml").read_text())
    assert dy["names"] == ["ball"] and dy["nc"] == 1
    # the player-only clip is ball-absent: exported as an empty-label negative, never with player boxes
    assert set(manifest["train_clips"]) | set(manifest["val_clips"]) == {
        "clip_ball",
        "clip_people",
    }
    parsed_all = _labels_by_frame(out, dy["names"])
    assert parsed_all["clip_people_000000"] == []
    for parsed in parsed_all.values():
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
    assert dy["names"] == list(FROZEN_CLASS_NAMES)
    assert dy["nc"] == len(FROZEN_CLASS_NAMES)
    manifest = yaml.safe_load((out / "manifest.yaml").read_text())
    assert manifest["sources"] == ["hand_label", "vittrack"]
    assert manifest["train"]["boxes"] + manifest["val"]["boxes"] == 3


def test_heldout_match_is_never_exportable():
    from footy_track.scripts.export_training_dataset import is_heldout_clip

    assert is_heldout_clip("arsenal_nottinghamforest_seg012")
    assert is_heldout_clip("arsenal_nottinghamforest_seg012.mp4")
    assert is_heldout_clip(
        "arsenal_norwich_seg012"
    )  # the old, mistaken name of the same match
    assert not is_heldout_clip("arsenal_mancity_20250925_seg012")


def test_register_heldout_match_excluded_from_train_val(tmp_path, monkeypatch):
    """A held-out match from the register never reaches train/val; a train match does."""
    import json

    from footy_track import match_splits

    reg = tmp_path / "match_splits.json"
    reg.write_text(
        json.dumps(
            {
                "chelsea-vs-x-2026-27": {
                    "split": "heldout",
                    "aliases": ["chelsea_vs_x_2026_27"],
                },
                "arsenal_mancity": {"split": "train", "aliases": ["mancity"]},
            }
        )
    )
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(reg))
    match_splits.clear_cache()
    from footy_track.scripts.export_training_dataset import is_heldout_clip

    assert is_heldout_clip("chelsea_vs_x_2026_27_seg004")
    assert not is_heldout_clip("arsenal_mancity_20250925_seg004")
    match_splits.clear_cache()


def test_export_fails_closed_without_register(tmp_path, monkeypatch):
    import pytest

    from footy_track import match_splits
    from footy_track.scripts import export_training_dataset as etd

    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(tmp_path / "absent.json"))
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED")
    match_splits.clear_cache()
    out = tmp_path / "out"
    with pytest.raises(match_splits.RegisterError):
        etd.export(
            None,  # never touched: the register check runs first
            out_dir=out,
            video_dir=tmp_path,
            eval_dir=tmp_path,
            extra_exclude=set(),
            tag="t",
        )
    assert not out.exists()


# --------------------------------------------------------------------------- #
# Complete-frame filter, negatives, frozen ids, default sources               #
# --------------------------------------------------------------------------- #


def test_frozen_class_ids_never_change() -> None:
    """Class ids are baked into every trained model. Append-only: edit this test deliberately."""
    assert FROZEN_CLASS_NAMES == (
        "player", "ball", "referee", "coach", "player_sub", "person",
        "in_play_ball", "out_of_play_ball",
    )  # fmt: skip
    assert FROZEN_CLASS_IDS["player"] == 0 and FROZEN_CLASS_IDS["ball"] == 1
    assert FROZEN_CLASS_IDS["referee"] == 2


def _states(store, game_id, frame_index, ball, player, *, method=LabelMethod.HAND_FRAME):
    store.upsert_frame_label_states(
        [
            FrameLabelStateRow(
                game_id=game_id,
                frame_index=frame_index,
                cls_group=g,
                state=st,
                method=method,
            )
            for g, st in ((ClsGroup.BALL, ball), (ClsGroup.PLAYER, player))
            if st is not None
        ]
    )


def _kw(tmp_path, **extra):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(exist_ok=True)
    return dict(
        video_dir=tmp_path, eval_dir=eval_dir, extra_exclude=set(), tag="t", **extra
    )


def test_only_complete_frames_and_skip_reasons(tmp_path) -> None:
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    L = LabelState
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_a",
        {i: [("ball", 0.5, 0.5, 0.02, 0.03)] for i in range(6)},
        complete=False,
    )
    _states(store, "clip_a", 0, L.COMPLETE, L.COMPLETE)  # emitted
    _states(store, "clip_a", 1, L.COMPLETE, L.ABSENT)  # emitted
    _states(store, "clip_a", 2, L.COMPLETE, L.PARTIAL)  # partial
    _states(store, "clip_a", 3, L.COMPLETE, None)  # player unreviewed
    _states(store, "clip_a", 4, L.UNREVIEWED, L.COMPLETE)  # ball unreviewed
    # frame 5: box but no state row at all
    m = export(store, out_dir=tmp_path / "o", classes=["ball", "player"], **_kw(tmp_path))
    assert m["frames"]["emitted"] == 2
    assert m["frames"]["skipped_by_reason"] == {
        "partial": 1,
        "player_unreviewed": 1,
        "ball_unreviewed": 1,
        "no_state": 1,
    }
    assert m["frames"]["by_state"] == {
        "ball:complete": 2,
        "player:absent": 1,
        "player:complete": 1,
    }
    written = sorted(p.stem for p in (tmp_path / "o" / "images").rglob("*.jpg"))
    assert written == ["clip_a_000000", "clip_a_000001"]


def test_ball_absent_frame_is_an_empty_label_negative(tmp_path) -> None:
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_a",
        {0: [("player", 0.1, 0.1, 0.2, 0.3)], 1: [("player", 0.2, 0.2, 0.2, 0.3)]},
        complete=False,
    )
    _states(store, "clip_a", 0, LabelState.ABSENT, LabelState.COMPLETE)
    _states(store, "clip_a", 1, LabelState.ABSENT, LabelState.COMPLETE)
    out = tmp_path / "o"
    m = export(store, out_dir=out, **_kw(tmp_path))
    labels = list((out / "labels").rglob("*.txt"))
    assert len(labels) == 2 and all(p.read_text() == "" for p in labels)
    assert sum(m[s]["empty_frames"] for s in ("train", "val")) == 2
    assert m["frames"]["by_state"]["ball:absent"] == 2


def test_not_ball_goes_to_hard_negatives_not_a_class(tmp_path) -> None:
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_a",
        {0: [("ball", 0.5, 0.5, 0.02, 0.03)], 1: [("ball", 0.3, 0.3, 0.02, 0.03)]},
    )
    store.query(
        "INSERT INTO detection_review (game_id, vsource, frame_index, box_index, verdict, label, "
        "bbox_x, bbox_y, bbox_w, bbox_h) VALUES ('clip_a', 'cand', 1, 0, 'not_ball', 'ball', "
        "0.7, 0.7, 0.02, 0.03)"
    )
    out = tmp_path / "o"
    m = export(store, out_dir=out, **_kw(tmp_path))
    hn = json.loads((out / "hard_negatives.json").read_text())["negatives"]
    assert m["hard_negatives"] == 1 and len(hn) == 1
    assert hn[0]["frame_index"] == 1 and hn[0]["bbox_xywh"] == [0.7, 0.7, 0.02, 0.03]
    assert (out / hn[0]["image"]).is_file()
    assert m["train"]["by_class"] | m["val"]["by_class"] == {"ball": 2}
    assert yaml.safe_load((out / "data.yaml").read_text())["nc"] == 1


def test_default_sources_are_hand_plus_reviewed(tmp_path) -> None:
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    _seed_multiclass_clip(
        store, tmp_path, images_dir, "clip_h", {0: [("player", 0.1, 0.1, 0.2, 0.3)]}
    )
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_m",
        {0: [("player", 0.2, 0.2, 0.2, 0.3)]},
        source="vittrack",
    )
    _seed_multiclass_clip(
        store,
        tmp_path,
        images_dir,
        "clip_r",
        {0: [("player", 0.3, 0.3, 0.2, 0.3)]},
        source="vittrack",
    )
    store.query("UPDATE detection SET reviewed = TRUE WHERE game_id = 'clip_r'")

    def clips(**kw):
        m = export(
            store, out_dir=tmp_path / "o", classes=["player"], **_kw(tmp_path, **kw)
        )
        return set(m["train_clips"]) | set(m["val_clips"])

    assert clips() == {"clip_h", "clip_r"}  # unreviewed machine boxes stay out
    assert clips(sources=["all"]) == {"clip_h", "clip_m", "clip_r"}
    assert clips(sources=["vittrack"]) == {"clip_m", "clip_r"}


def test_unknown_class_label_rejected(tmp_path) -> None:
    store = FeatureStore.open(":memory:")
    with pytest.raises(ValueError, match="frozen class list"):
        export(store, out_dir=tmp_path / "o", classes=["goalkeeper"], **_kw(tmp_path))


def test_ball_only_needs_only_ball_state_multiclass_needs_both(tmp_path) -> None:
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    store = FeatureStore.open(":memory:")
    L = LabelState
    _seed_multiclass_clip(
        store, tmp_path, images_dir, "clip_a",
        {
            0: [("ball", 0.5, 0.5, 0.02, 0.03)],
            1: [("player", 0.1, 0.1, 0.2, 0.3)],
            2: [("ball", 0.5, 0.5, 0.02, 0.03)],
            3: [("ball", 0.5, 0.5, 0.02, 0.03)],
        },
        complete=False,
    )
    _states(store, "clip_a", 0, L.COMPLETE, None)  # ball only: no player state
    _states(store, "clip_a", 1, L.ABSENT, None)  # ball absent, no player state
    _states(store, "clip_a", 2, L.PARTIAL, None)
    _states(store, "clip_a", 3, L.UNREVIEWED, L.COMPLETE)
    m = export(store, out_dir=tmp_path / "o1", **_kw(tmp_path))
    assert m["frames"]["emitted"] == 2
    assert m["frames"]["skipped_by_reason"] == {"partial": 1, "ball_unreviewed": 1}
    absent = next((tmp_path / "o1" / "labels").rglob("clip_a_000001.txt"))
    assert absent.read_text() == ""

    m = export(store, out_dir=tmp_path / "o2", classes=["player", "ball"], **_kw(tmp_path))
    assert m["frames"]["emitted"] == 0  # player state missing/partial everywhere
    assert m["frames"]["skipped_by_reason"]["player_unreviewed"] == 2
