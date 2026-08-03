"""Round-trip fidelity tests for the feature store <-> Roboflow YOLO export path
(ft-drs).

The exporter shipped for ft-n2o.1 (``export_training_dataset.py``) is
deliberately ball-class-only. To prove the *general* store <-> YOLO round
trip is lossless across arbitrary classes (not just verify the ball-only
exporter), this module provides a small, general-purpose export helper
(:func:`export_source_to_yolo`) that mirrors the coordinate-conversion
pattern of that script's ``_yolo_line`` but keeps every distinct ``label``
as its own class, and is not itself part of the ball-specific pipeline.

The tests:

1. ``test_roboflow_roundtrip_fidelity`` — synthetic Roboflow dataset (multi
   class, multi frame) -> ``import_roboflow`` -> store -> our export helper
   -> compare every exported YOLO line back against the *original* YOLO
   lines authored in step 1 (not the store's intermediate top-left rows),
   so the assertion covers both conversion directions. No network.
2. ``test_export_is_structurally_roboflow_ready`` — validates the exported
   dataset is a well-formed, Roboflow-ingestible YOLO dataset (data.yaml,
   label/image pairing, field counts, ranges). No network.
3. ``test_live_roboflow_upload_roundtrip`` — full live round trip against
   the real Roboflow API. Marked ``@pytest.mark.live_roboflow`` AND
   self-skips unless ``RUN_LIVE_ROBOFLOW_TESTS=1`` is set, so it never runs
   by accident even under ``pytest -m live_roboflow`` alone (double
   opt-in). See the test's docstring for why: this workspace/project are
   real production Roboflow resources and project/version creation via the
   API is only partially reversible.
4. ``test_edge_overlapping_box_clamps_predictably`` — pins the *known,
   by-design lossy* case: YOLO boxes whose extent crosses the frame's
   left/top edge are clamped to [0, 1] on import (the store schema
   requires normalized top-left coords >= 0), which shifts the centre
   inward deterministically on export. Interior boxes (tests 1-3) round
   trip losslessly; edge-overlapping boxes shift by an exactly
   predictable amount, and this test asserts that exact amount so any
   future change to the clamping rule is caught. No network.
5. ``test_labeller_json_roundtrip_fidelity`` — the labeller/SAM3 path:
   boxes are *already* top-left in the JSON, so the round trip involves
   no centre<->topleft conversion on import, only float32 storage and
   one topleft->centre conversion on export. Verifies exported centres
   equal ``x + w/2`` of the original JSON values within tolerance. No
   network.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pytest
import yaml
from PIL import Image

from footy_track.feature_store import (
    FeatureStore,
    import_labeller_json,
    import_roboflow,
)
from footy_track.feature_store.importers import parse_roboflow_stem

# --------------------------------------------------------------------------- #
# Tolerance derivation (do not loosen without re-deriving)                    #
# --------------------------------------------------------------------------- #
#
# bbox_x/y/w/h are stored as DuckDB FLOAT = IEEE-754 binary32. Machine
# epsilon for float32 is 2**-23 ~= 1.1920929e-07 (the gap between 1.0 and
# the next representable float32, i.e. the worst-case *relative* rounding
# error introduced by a single float32 rounding at magnitude ~1).
#
# The round trip performs two float32-precision arithmetic ops on each
# coordinate: YOLO-centre -> top-left is one subtraction (import), and
# top-left -> YOLO-centre is one addition (export). Each op can introduce
# up to ~1 ULP of relative rounding error at the operand's magnitude (all
# our normalized coords are in [0, 1], so magnitude <= 1). Two ops
# compound to roughly 2 * epsilon ~= 2.4e-7 in the worst case, before any
# safety margin.
#
# We apply a ~50x safety factor over that theoretical bound to absorb
# implementation-detail rounding in DuckDB's FLOAT cast, Python's str
# formatting round-trip (the exporter writes with %.6f, i.e. 1e-6
# granularity, which is itself a source of error one order of magnitude
# above the float32 ULP), and pandas' numpy float32 -> python float
# promotion. 2.4e-7 * 50 ~= 1.2e-5, which we round down slightly to a
# clean 1e-5. This is *not* an arbitrary round number: it is float32
# epsilon at magnitude ~1, doubled for two lossy ops, times a ~50x margin
# for intermediate representation noise (esp. the exporter's %.6f text
# formatting), then rounded to 1e-5 for a tidy bound that still leaves
# genuine corruption (e.g. a dropped detection, a full off-by-one class
# remap) nowhere to hide.
FLOAT32_ROUNDTRIP_TOL = 1e-5


# --------------------------------------------------------------------------- #
# General-purpose (all-class) store -> YOLO export helper                    #
# --------------------------------------------------------------------------- #
#
# This intentionally does NOT touch export_training_dataset.py (which is
# ball-class-only by design for ft-n2o.1). It reuses the same top-left ->
# YOLO-centre conversion pattern as that script's `_yolo_line`.


def _yolo_line(class_id: int, x: float, y: float, w: float, h: float) -> str:
    """Top-left xywh (store format) -> one YOLO-format label line.

    Mirrors ``export_training_dataset._yolo_line``'s conversion math
    exactly (clamped centre, raw w/h), but takes an explicit class id
    instead of hardcoding ``BALL_CLASS_ID``.
    """
    cx = min(max(x + w / 2, 0.0), 1.0)
    cy = min(max(y + h / 2, 0.0), 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def export_source_to_yolo(store: FeatureStore, *, source: str, out_dir: Path) -> dict:
    """Export every ``detection`` row for a given ``source`` (no class/canonical
    filter) to a flat YOLOv8-format dataset under *out_dir*.

    Unlike ``export_training_dataset.export`` this is class-agnostic and has
    no leakage-guard / clip-split logic — it's a minimal, general-purpose
    helper for fidelity testing, not a training-dataset builder. Layout:

        out_dir/images/all/<game_id>_<frame_index:06d>.jpg
        out_dir/labels/all/<game_id>_<frame_index:06d>.txt
        out_dir/data.yaml

    Class ids are assigned by **sorting the distinct label strings
    alphabetically** and taking their index in that sorted list. This
    mapping is deterministic and is recorded in ``data.yaml["names"]`` (list
    index == class id), the same contract YOLOv8/Roboflow expect. Because
    class ids are reassigned from scratch here, they may legitimately differ
    from the original dataset's class ids for the same label string — only
    the *name* is guaranteed stable, which is why round-trip comparisons in
    these tests key on label name, not id.

    Only detections whose frame has real backing pixels at ``frame.frame_uri``
    (a file on disk) are exported; this helper does not do video-frame
    extraction (that's export_training_dataset's job for GT-mark clips).

    Returns a small report dict: {"images": N, "boxes": N, "classes": [...]}.
    """
    df = store.query(
        """
        SELECT d.game_id, d.frame_index, d.detection_id, d.label,
               d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, f.frame_uri
        FROM detection d
        JOIN frame f USING (game_id, frame_index)
        WHERE d.source = ?
        ORDER BY d.game_id, d.frame_index, d.detection_id
        """,
        [source],
    )

    class_names = sorted(df["label"].unique().tolist())
    class_id = {name: i for i, name in enumerate(class_names)}

    images_dir = out_dir / "images" / "all"
    labels_dir = out_dir / "labels" / "all"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    by_frame: dict[tuple[str, int], list] = defaultdict(list)
    frame_uri: dict[tuple[str, int], str] = {}
    for row in df.itertuples():
        key = (row.game_id, row.frame_index)
        by_frame[key].append(row)
        frame_uri[key] = row.frame_uri

    n_images = n_boxes = 0
    for (game_id, frame_index), rows in by_frame.items():
        uri = frame_uri[(game_id, frame_index)]
        src_img = Path(uri)
        if not src_img.is_file():
            continue  # this helper doesn't do video extraction
        base = f"{game_id}_{frame_index:06d}"
        img_out = images_dir / f"{base}.jpg"
        img_out.write_bytes(src_img.read_bytes())
        lines = [
            _yolo_line(class_id[r.label], r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h)
            for r in rows
        ]
        (labels_dir / f"{base}.txt").write_text("\n".join(lines) + "\n")
        n_images += 1
        n_boxes += len(lines)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/all",
        "val": "images/all",
        "nc": len(class_names),
        "names": class_names,
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    return {"images": n_images, "boxes": n_boxes, "classes": class_names}


# --------------------------------------------------------------------------- #
# Shared fixture: a multi-class, multi-frame synthetic Roboflow dataset       #
# --------------------------------------------------------------------------- #
#
# Mirrors test_importers.py's `_make_roboflow`, but stresses the fidelity
# check harder: >=2 classes, >=3 frames, some frames with multiple
# detections (to catch dropped/duplicated-detection bugs a single-box
# fixture couldn't).

IMG_W, IMG_H = 640, 480

# (frame_index, [(class_name, cx, cy, w, h), ...])
_FIXTURE_FRAMES: tuple[
    tuple[int, tuple[tuple[str, float, float, float, float], ...]], ...
] = (
    (10, (("player", 0.5, 0.5, 0.2, 0.4),)),
    (
        11,
        (
            ("ball", 0.100000, 0.200000, 0.020000, 0.030000),
            ("player", 0.75, 0.6, 0.15, 0.35),
        ),
    ),
    (
        12,
        (
            ("player", 0.25, 0.30, 0.10, 0.20),
            ("referee", 0.9, 0.1, 0.05, 0.09),
            ("ball", 0.4, 0.45, 0.018, 0.025),
        ),
    ),
)

_CLASS_NAMES = ["ball", "player", "referee"]  # data.yaml names list; index == class id


def _make_multiclass_roboflow(
    tmp_path: Path, game: str = "arsenal_demo"
) -> tuple[Path, dict]:
    """Build a synthetic Roboflow YOLO dataset with multiple classes/frames and
    real backing images (mirrors test_importers.py's ``_make_roboflow``).

    Returns (dataset_root, original_lines) where ``original_lines`` maps
    frame_index -> list of the literal (class_name, cx, cy, w, h) tuples
    written to each label file — the ground truth we compare the final
    export back against, so the test proves the *whole* round trip
    (YOLO-centre -> topleft -> YOLO-centre), not just the store's
    intermediate representation.
    """
    root = tmp_path / "roboflow_multiclass"
    (root / "train" / "labels").mkdir(parents=True)
    (root / "train" / "images").mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {"names": _CLASS_NAMES, "nc": len(_CLASS_NAMES), "roboflow": {"version": 7}}
        )
    )

    original_lines: dict[int, list[tuple[str, float, float, float, float]]] = {}
    for fi, dets in _FIXTURE_FRAMES:
        base = f"{game}_{fi:06d}_png.rf.deadbeef{fi}"
        lines = []
        recorded = []
        for label, cx, cy, w, h in dets:
            cls_idx = _CLASS_NAMES.index(label)
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            recorded.append((label, cx, cy, w, h))
        (root / "train" / "labels" / f"{base}.txt").write_text("\n".join(lines) + "\n")
        Image.new("RGB", (IMG_W, IMG_H)).save(root / "train" / "images" / f"{base}.jpg")
        original_lines[fi] = recorded

    return root, original_lines


def _assert_labels_match_originals(
    original_lines: dict[int, list[tuple[str, float, float, float, float]]],
    label_file_for_frame: dict[int, Path],
    names_list: list[str],
) -> None:
    """Shared fidelity comparison: every original (frame, detection) must
    appear in the corresponding label file with the same class *name* and
    centre-xywh values within ``FLOAT32_ROUNDTRIP_TOL``.

    ``label_file_for_frame`` maps original frame_index -> the label ``.txt``
    to compare against (callers build it for their layout: our flat export
    layout, or a Roboflow-downloaded split layout with mangled filenames).
    ``names_list`` is the data.yaml names list of the dataset under test
    (list index == class id). Comparison keys on class *name* because class
    ids may legitimately be remapped across a round trip.
    """
    for fi, expected_dets in original_lines.items():
        label_path = label_file_for_frame.get(fi)
        assert label_path is not None and label_path.is_file(), (
            f"frame {fi}: no label file found (dropped frame?); "
            f"mapping has {sorted(label_file_for_frame)}"
        )

        exported_lines = label_path.read_text().strip().splitlines()
        assert len(exported_lines) == len(expected_dets), (
            f"frame {fi}: expected {len(expected_dets)} detections, "
            f"got {len(exported_lines)} in {label_path}\n"
            f"  expected: {expected_dets}\n"
            f"  actual lines: {exported_lines}"
        )

        # Match exported lines to expected detections positionally: the
        # export preserves detection_id order, which was assigned in
        # fixture-authoring (label-file line) order on import.
        for i, (exp_label, exp_cx, exp_cy, exp_w, exp_h) in enumerate(expected_dets):
            actual_parts = exported_lines[i].split()
            assert len(actual_parts) == 5, (
                f"frame {fi} detection {i}: malformed exported line "
                f"{exported_lines[i]!r} (expected 5 whitespace-separated fields)"
            )
            act_cls_idx = int(actual_parts[0])
            act_cx, act_cy, act_w, act_h = (float(v) for v in actual_parts[1:])

            act_label = names_list[act_cls_idx]
            assert act_label == exp_label, (
                f"frame {fi} detection {i}: class mismatch — "
                f"expected label {exp_label!r}, got {act_label!r} "
                f"(class id {act_cls_idx}, names={names_list})"
            )

            for field_name, exp_val, act_val in (
                ("cx", exp_cx, act_cx),
                ("cy", exp_cy, act_cy),
                ("w", exp_w, act_w),
                ("h", exp_h, act_h),
            ):
                diff = abs(exp_val - act_val)
                assert diff < FLOAT32_ROUNDTRIP_TOL, (
                    f"frame {fi} detection {i} ({exp_label}): {field_name} mismatch "
                    f"beyond float32 round-trip tolerance ({FLOAT32_ROUNDTRIP_TOL}) — "
                    f"expected {exp_val!r}, got {act_val!r}, diff={diff!r}"
                )


# --------------------------------------------------------------------------- #
# Test 1 — full round-trip fidelity, no network                              #
# --------------------------------------------------------------------------- #


def test_roboflow_roundtrip_fidelity(tmp_path: Path) -> None:
    game = "arsenal_demo"
    root, original_lines = _make_multiclass_roboflow(tmp_path, game=game)

    store = FeatureStore.open(":memory:")
    report = import_roboflow(store, root, game_id=game)
    n_original_boxes = sum(len(v) for v in original_lines.values())
    assert report.detections_written == n_original_boxes

    out_dir = tmp_path / "export_all"
    export_report = export_source_to_yolo(store, source="hand_label", out_dir=out_dir)

    # -- no dropped/duplicated frames or detections ------------------------ #
    assert export_report["images"] == len(original_lines), (
        f"expected {len(original_lines)} exported images (one per original frame), "
        f"got {export_report['images']}"
    )
    assert export_report["boxes"] == n_original_boxes, (
        f"expected {n_original_boxes} exported boxes total, got {export_report['boxes']} "
        "(dropped or duplicated detections)"
    )

    exported_names = set(yaml.safe_load((out_dir / "data.yaml").read_text())["names"])
    assert exported_names == set(_CLASS_NAMES), (
        f"exported class-name set {exported_names} != original {set(_CLASS_NAMES)}"
    )

    # -- per-frame, per-detection comparison against the ORIGINAL YOLO lines.
    # Class ids may legitimately differ from the original dataset's ids;
    # only names are a stable comparison key (see export_source_to_yolo).
    exported_names_list = yaml.safe_load((out_dir / "data.yaml").read_text())["names"]
    label_file_for_frame = {
        fi: out_dir / "labels" / "all" / f"{game}_{fi:06d}.txt" for fi in original_lines
    }
    _assert_labels_match_originals(
        original_lines, label_file_for_frame, exported_names_list
    )


# --------------------------------------------------------------------------- #
# Test 2 — structural / Roboflow-ingestibility check, no network             #
# --------------------------------------------------------------------------- #


def test_export_is_structurally_roboflow_ready(tmp_path: Path) -> None:
    game = "arsenal_demo"
    root, original_lines = _make_multiclass_roboflow(tmp_path, game=game)

    store = FeatureStore.open(":memory:")
    import_roboflow(store, root, game_id=game)

    out_dir = tmp_path / "export_structural"
    export_source_to_yolo(store, source="hand_label", out_dir=out_dir)

    # -- data.yaml is present and well-formed ------------------------------ #
    data_yaml_path = out_dir / "data.yaml"
    assert data_yaml_path.is_file(), "data.yaml missing from export"
    data_yaml = yaml.safe_load(data_yaml_path.read_text())
    for key in ("path", "train", "val", "nc", "names"):
        assert key in data_yaml, f"data.yaml missing required key {key!r}: {data_yaml}"

    names = data_yaml["names"]
    assert isinstance(names, list) and names, (
        "data.yaml['names'] must be a non-empty list"
    )
    assert data_yaml["nc"] == len(names), (
        f"data.yaml['nc'] ({data_yaml['nc']}) != len(names) ({len(names)})"
    )
    distinct_labels_in_store = set(
        store.query("SELECT DISTINCT label FROM detection WHERE source = 'hand_label'")[
            "label"
        ]
    )
    assert set(names) == distinct_labels_in_store, (
        f"data.yaml names {set(names)} != distinct labels actually present in store "
        f"{distinct_labels_in_store}"
    )
    # Class-id assignment is documented as sorted-alphabetical; verify that
    # contract explicitly since a Roboflow re-ingest trusts list-index == id.
    assert names == sorted(names), (
        f"names list must be sorted alphabetically (documented class-id assignment "
        f"contract), got {names}"
    )

    images_dir = out_dir / "images" / "all"
    labels_dir = out_dir / "labels" / "all"
    assert images_dir.is_dir(), "images/all/ directory missing"
    assert labels_dir.is_dir(), "labels/all/ directory missing"

    image_stems = {p.stem for p in images_dir.glob("*.jpg")}
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}

    assert image_stems, "no images were exported"
    # -- no orphans in either direction ------------------------------------ #
    images_without_labels = image_stems - label_stems
    labels_without_images = label_stems - image_stems
    assert not images_without_labels, (
        f"images with no matching label file (orphan images): {sorted(images_without_labels)}"
    )
    assert not labels_without_images, (
        f"label files with no matching image (orphan labels): {sorted(labels_without_images)}"
    )

    n_classes = len(names)
    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = label_path.read_text().strip().splitlines()
        assert lines, (
            f"{label_path.name}: label file is empty (every exported frame must have >=1 box)"
        )
        for line_no, line in enumerate(lines):
            parts = line.split()
            assert len(parts) == 5, (
                f"{label_path.name} line {line_no}: expected 5 whitespace-separated fields "
                f"(class cx cy w h), got {len(parts)}: {line!r}"
            )
            cls_idx_str, cx_str, cy_str, w_str, h_str = parts
            assert cls_idx_str.isdigit(), (
                f"{label_path.name} line {line_no}: class index {cls_idx_str!r} is not a valid int"
            )
            cls_idx = int(cls_idx_str)
            assert 0 <= cls_idx < n_classes, (
                f"{label_path.name} line {line_no}: class index {cls_idx} out of range "
                f"[0, {n_classes})"
            )
            for field_name, val_str in (
                ("cx", cx_str),
                ("cy", cy_str),
                ("w", w_str),
                ("h", h_str),
            ):
                val = float(val_str)
                assert 0.0 <= val <= 1.0, (
                    f"{label_path.name} line {line_no}: {field_name}={val} out of range [0, 1]"
                )

    # -- total box count sanity: matches the number of original detections - #
    n_original_boxes = sum(len(v) for v in original_lines.values())
    total_exported_boxes = sum(
        len(p.read_text().strip().splitlines()) for p in labels_dir.glob("*.txt")
    )
    assert total_exported_boxes == n_original_boxes, (
        f"total exported box count {total_exported_boxes} != original {n_original_boxes}"
    )


# --------------------------------------------------------------------------- #
# Test 3 — live Roboflow API round trip (SKIPPED BY DEFAULT, double opt-in)   #
# --------------------------------------------------------------------------- #


def _download_generated_version(project, version_num: int, download_root: Path):
    """Poll ``.download()`` until the (asynchronously generated) Roboflow
    version materialises; the SDK raises while generation is pending."""
    deadline = time.monotonic() + 600  # generation usually takes < 10 min
    while True:
        try:
            return project.version(version_num).download(
                "yolov8", location=str(download_root)
            )
        except Exception:  # noqa: BLE001 - SDK raises assorted errors while pending
            if time.monotonic() > deadline:
                raise
            time.sleep(15)


@pytest.mark.live_roboflow
def test_live_roboflow_upload_roundtrip(tmp_path: Path) -> None:
    """Full live round trip: export a tiny dataset -> upload to a throwaway
    Roboflow project via the SDK -> download it back -> re-run the same
    fidelity comparison as ``test_roboflow_roundtrip_fidelity``.

    DELIBERATELY NOT EXECUTED as part of normal test runs. This workspace
    (``egroeg121``) is a real production Roboflow workspace; creating a
    project/version via the API is only partially reversible (uploads count
    against quota, and versions in particular are often not cleanly
    deletable via the API). This test requires a *double* opt-in beyond
    ``@pytest.mark.live_roboflow`` -- both the marker (for explicit
    ``-m live_roboflow`` selection) AND the ``RUN_LIVE_ROBOFLOW_TESTS=1``
    env var -- so it can never fire by accident, e.g. via a bare
    ``pytest -m live_roboflow`` someone runs without reading this
    docstring first.

    To actually run it (only do this deliberately, understanding it will
    create real, likely-permanent resources in the egroeg121 workspace):

        RUN_LIVE_ROBOFLOW_TESTS=1 uv run pytest \\
            tests/feature_store/test_roundtrip_fidelity.py -m live_roboflow -v
    """
    if not os.environ.get("RUN_LIVE_ROBOFLOW_TESTS"):
        pytest.skip(
            "live_roboflow tests require RUN_LIVE_ROBOFLOW_TESTS=1 in addition to "
            "-m live_roboflow (double opt-in; see test docstring) -- this creates "
            "real, likely-permanent resources in the production egroeg121 workspace"
        )

    from roboflow import Roboflow  # noqa: PLC0415

    def _load_api_key() -> str:
        env_key = os.environ.get("ROBOFLOW_API_KEY")
        if env_key:
            return env_key
        config_path = Path.home() / ".config" / "roboflow" / "config.json"
        config = json.loads(config_path.read_text())
        for ws in config.get("workspaces", {}).values():
            if ws.get("url") == "egroeg121":
                return ws["apiKey"]
        raise ValueError("Roboflow API key not found in config or environment")

    workspace_name = "egroeg121"
    project_name = "footy-track-roundtrip-test"  # clearly-named throwaway project

    game = "roundtrip_live_demo"
    root, original_lines = _make_multiclass_roboflow(tmp_path, game=game)
    # Trim to a tiny 2-3 image export for the live test, per instructions.
    store = FeatureStore.open(":memory:")
    import_roboflow(store, root, game_id=game)

    out_dir = tmp_path / "export_live"
    export_source_to_yolo(store, source="hand_label", out_dir=out_dir)

    api_key = _load_api_key()
    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(workspace_name)

    try:
        project = workspace.project(project_name)
    except Exception:  # noqa: BLE001 - SDK raises a bare RuntimeError on 404
        project = workspace.create_project(
            project_name=project_name,
            project_type="object-detection",
            project_license="MIT",
            annotation=project_name,
        )

    images_dir = out_dir / "images" / "all"
    labels_dir = out_dir / "labels" / "all"
    uploaded = 0
    for img_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{img_path.stem}.txt"
        project.upload(
            image_path=str(img_path),
            annotation_path=str(label_path) if label_path.is_file() else None,
            split="train",
            batch_name="roundtrip_fidelity_test",
        )
        uploaded += 1

    assert uploaded > 0, "expected at least one image to be uploaded"

    # -- generate a version and download it back --------------------------- #
    # Roboflow version generation is asynchronous; poll `.download()` until
    # the version materialises (the SDK raises while generation is pending).
    version_num = project.generate_version(
        settings={"preprocessing": {}, "augmentation": {}}
    )
    dataset = _download_generated_version(project, version_num, tmp_path / "downloaded")
    downloaded_root = Path(dataset.location)
    dl_yaml = yaml.safe_load((downloaded_root / "data.yaml").read_text())
    dl_names = list(dl_yaml["names"])

    # Build frame_index -> downloaded label file. Roboflow mangles filenames
    # (appends `_<ext>.rf.<hash>`) and may rebalance splits, so scan every
    # split and recover frame identity with the same parser the importer uses.
    label_file_for_frame: dict[int, Path] = {}
    for split in ("train", "valid", "test"):
        labels_dir = downloaded_root / split / "labels"
        if not labels_dir.is_dir():
            continue
        for label_path in labels_dir.glob("*.txt"):
            _stem, frame_index = parse_roboflow_stem(label_path.name)
            label_file_for_frame[frame_index] = label_path

    # Same fidelity comparison as test_roboflow_roundtrip_fidelity, against
    # the literal original YOLO lines authored in the fixture. Compared by
    # class *name*: Roboflow rebuilds the class list on ingest so ids may
    # be remapped, which is fine.
    _assert_labels_match_originals(original_lines, label_file_for_frame, dl_names)


# --------------------------------------------------------------------------- #
# Test 4 — edge-overlapping boxes clamp predictably (by-design lossy case)    #
# --------------------------------------------------------------------------- #


def test_edge_overlapping_box_clamps_predictably(tmp_path: Path) -> None:
    """Boxes whose YOLO extent crosses the frame's LEFT or TOP edge are the
    one *known, by-design lossy* case in the round trip, and this test pins
    the exact behaviour so it can never drift silently.

    Why lossy: the store schema requires normalized top-left coords in
    [0, 1], so ``_yolo_centre_to_topleft`` clamps ``x = max(0, cx - w/2)``
    (and same for y). For a box crossing the left edge the true top-left is
    negative, the clamp moves it to 0, and the re-derived centre on export
    shifts inward to exactly ``w/2`` (resp. ``h/2``). The shift equals the
    out-of-frame overhang — fully deterministic, not corruption.

    Right/bottom overhang is NOT clamped on import (top-left stays in
    range; only ``x + w`` exceeds 1, which the schema permits per-column),
    so right/bottom-edge boxes round trip losslessly. Asserted here too.
    """
    game = "edge_demo"
    root = tmp_path / "roboflow_edges"
    (root / "train" / "labels").mkdir(parents=True)
    (root / "train" / "images").mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": ["ball"], "nc": 1, "roboflow": {"version": 7}})
    )

    # (case_name, authored (cx, cy, w, h), predicted round-tripped (cx, cy, w, h))
    cases = [
        # crosses LEFT edge: true x = 0.01 - 0.02 = -0.01 -> clamped to 0
        # -> exported cx = 0 + w/2 = 0.02 (shift = the 0.01 overhang)
        ("left_edge", (0.01, 0.5, 0.04, 0.06), (0.02, 0.5, 0.04, 0.06)),
        # crosses TOP edge: true y = 0.02 - 0.04 = -0.02 -> clamped to 0
        # -> exported cy = 0 + h/2 = 0.04 (shift = the 0.02 overhang)
        ("top_edge", (0.5, 0.02, 0.1, 0.08), (0.5, 0.04, 0.1, 0.08)),
        # crosses RIGHT edge: x = 0.97 in range, x + w = 1.01 permitted
        # -> lossless round trip
        ("right_edge", (0.99, 0.5, 0.04, 0.06), (0.99, 0.5, 0.04, 0.06)),
        # crosses BOTTOM edge: same, lossless
        ("bottom_edge", (0.5, 0.98, 0.1, 0.08), (0.5, 0.98, 0.1, 0.08)),
    ]

    fi = 7
    base = f"{game}_{fi:06d}_png.rf.cafef00d{fi}"
    lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for _, (cx, cy, w, h), _p in cases]
    (root / "train" / "labels" / f"{base}.txt").write_text("\n".join(lines) + "\n")
    Image.new("RGB", (IMG_W, IMG_H)).save(root / "train" / "images" / f"{base}.jpg")

    store = FeatureStore.open(":memory:")
    import_roboflow(store, root, game_id=game)

    out_dir = tmp_path / "export_edges"
    export_source_to_yolo(store, source="hand_label", out_dir=out_dir)

    exported = (
        (out_dir / "labels" / "all" / f"{game}_{fi:06d}.txt")
        .read_text()
        .strip()
        .splitlines()
    )
    assert len(exported) == len(cases)

    for line, (case_name, authored, predicted) in zip(exported, cases, strict=True):
        act = tuple(float(v) for v in line.split()[1:])
        for field_name, pred_val, act_val, auth_val in zip(
            ("cx", "cy", "w", "h"), predicted, act, authored, strict=True
        ):
            diff = abs(pred_val - act_val)
            assert diff < FLOAT32_ROUNDTRIP_TOL, (
                f"{case_name}: {field_name} did not clamp as predicted — authored "
                f"{auth_val!r}, predicted round-trip {pred_val!r}, got {act_val!r} "
                f"(diff from prediction {diff!r}). The import-side clamp rule in "
                f"_yolo_centre_to_topleft (or the export-side centre re-derivation) "
                f"has changed behaviour."
            )


# --------------------------------------------------------------------------- #
# Test 5 — labeller/SAM3 (top-left JSON) round trip, no network              #
# --------------------------------------------------------------------------- #


def test_labeller_json_roundtrip_fidelity(tmp_path: Path) -> None:
    """The web-labeller/SAM3 import path stores boxes verbatim (already
    top-left, no conversion), so its round trip through the store to YOLO
    exercises only float32 storage plus the single topleft->centre
    conversion on export. Exported centres must equal ``x + w/2`` /
    ``y + h/2`` of the original JSON values within tolerance.
    """
    game = "labeller_demo"
    frames = {
        3: [
            ("ball", 0.1, 0.2, 0.02, 0.03),
            ("player", 0.6, 0.4, 0.05, 0.1),
        ],
        4: [
            ("player", 0.3, 0.5, 0.08, 0.12),
        ],
    }

    records = [
        {
            "uri": f"{game}_frame_{fi:06d}",
            "width": IMG_W,
            "height": IMG_H,
            "detections": [
                {"label": label, "confidence": 1.0, "x": x, "y": y, "w": w, "h": h}
                for label, x, y, w, h in dets
            ],
        }
        for fi, dets in frames.items()
    ]
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps(records))

    store = FeatureStore.open(":memory:")
    report = import_labeller_json(store, json_path, run_id="sam3_rt", game_id=game)
    assert report.detections_written == sum(len(d) for d in frames.values())

    # The labeller import records a synthetic frame_uri; point it at real
    # images so the exporter has pixels to copy (same pattern as
    # test_export_training_dataset._seed_clip).
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    for fi in frames:
        img = images_dir / f"{game}_{fi:06d}.jpg"
        Image.new("RGB", (IMG_W, IMG_H)).save(img)
        store.query(
            "UPDATE frame SET frame_uri = ? WHERE game_id = ? AND frame_index = ?",
            [str(img), game, fi],
        )

    out_dir = tmp_path / "export_labeller"
    export_source_to_yolo(store, source="sam3", out_dir=out_dir)

    # Expected YOLO lines derived from the authored top-left values.
    original_lines = {
        fi: [(label, x + w / 2, y + h / 2, w, h) for label, x, y, w, h in dets]
        for fi, dets in frames.items()
    }
    names_list = yaml.safe_load((out_dir / "data.yaml").read_text())["names"]
    label_file_for_frame = {
        fi: out_dir / "labels" / "all" / f"{game}_{fi:06d}.txt" for fi in frames
    }
    _assert_labels_match_originals(original_lines, label_file_for_frame, names_list)
