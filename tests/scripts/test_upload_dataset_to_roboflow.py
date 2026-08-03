"""Tests for upload_dataset_to_roboflow.py (the store -> Roboflow upload hop).

No live network calls in this file except the single ``live_roboflow``-marked
test, which is skipped by default and requires the double opt-in
(``-m live_roboflow`` AND ``RUN_LIVE_ROBOFLOW_TESTS=1``) documented on
``test_roundtrip_fidelity.test_live_roboflow_upload_roundtrip``. That test is
never executed as part of this task.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from PIL import Image

from footy_track import constants
from footy_track.scripts.upload_dataset_to_roboflow import (
    InvalidYoloDatasetError,
    UploadPlan,
    build_parser,
    compute_upload_plan,
    get_or_create_project,
    load_api_key,
    main,
    upload_plan,
    validate_yolo_dataset,
)

IMG_W, IMG_H = 64, 48


# --------------------------------------------------------------------------- #
# Helpers to build a valid / broken local YOLO export dir                     #
# --------------------------------------------------------------------------- #


def _write_yolo_dataset(
    root: Path,
    *,
    splits: dict[str, list[tuple[str, list[str]]]],
    class_names: list[str] = ("ball",),
) -> Path:
    """Build a minimal valid YOLOv8 export directory.

    ``splits`` maps split name -> list of (stem, [yolo_line, ...]) pairs.
    Each stem gets a real .jpg and a .txt label file with the given lines
    (label file omitted entirely if the line list is empty -> no file, to
    exercise the "image without label" case when desired).
    """
    class_names = list(class_names)
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(root), "nc": len(class_names), "names": class_names}
        )
    )
    for split, entries in splits.items():
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for stem, lines in entries:
            Image.new("RGB", (IMG_W, IMG_H)).save(images_dir / f"{stem}.jpg")
            if lines:
                (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    return root


def _valid_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ball_v1"
    return _write_yolo_dataset(
        root,
        splits={
            "train": [
                ("clip_a_000000", ["0 0.5 0.5 0.02 0.03"]),
                ("clip_a_000001", ["0 0.4 0.4 0.02 0.03", "0 0.6 0.6 0.02 0.03"]),
            ],
            "val": [
                ("clip_b_000000", ["0 0.5 0.5 0.02 0.03"]),
            ],
        },
    )


# --------------------------------------------------------------------------- #
# validate_yolo_dataset / compute_upload_plan (pure logic, no network)        #
# --------------------------------------------------------------------------- #


def test_validate_yolo_dataset_accepts_valid_dir(tmp_path: Path) -> None:
    root = _valid_dataset(tmp_path)
    validate_yolo_dataset(root)  # should not raise


def test_validate_yolo_dataset_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(InvalidYoloDatasetError, match="does not exist"):
        validate_yolo_dataset(tmp_path / "nope")


def test_validate_yolo_dataset_missing_data_yaml(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    with pytest.raises(InvalidYoloDatasetError, match="data.yaml"):
        validate_yolo_dataset(root)


def test_validate_yolo_dataset_empty_names_list(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 0, "names": []}))
    with pytest.raises(InvalidYoloDatasetError, match="names"):
        validate_yolo_dataset(root)


def test_validate_yolo_dataset_missing_images_dir(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 1, "names": ["ball"]}))
    (root / "labels" / "train").mkdir(parents=True)
    with pytest.raises(InvalidYoloDatasetError, match="images"):
        validate_yolo_dataset(root)


def test_validate_yolo_dataset_missing_labels_dir_for_split(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)  # labels/ exists, but not labels/train
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 1, "names": ["ball"]}))
    Image.new("RGB", (IMG_W, IMG_H)).save(root / "images" / "train" / "a.jpg")
    with pytest.raises(InvalidYoloDatasetError, match="labels/train"):
        validate_yolo_dataset(root)


def test_validate_yolo_dataset_no_split_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 1, "names": ["ball"]}))
    with pytest.raises(InvalidYoloDatasetError, match="split"):
        validate_yolo_dataset(root)


def test_validate_yolo_dataset_no_images(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 1, "names": ["ball"]}))
    with pytest.raises(InvalidYoloDatasetError, match="no image files"):
        validate_yolo_dataset(root)


def test_compute_upload_plan_counts_images_labels_boxes(tmp_path: Path) -> None:
    root = _valid_dataset(tmp_path)
    plan = compute_upload_plan(root, project_name="my-project")

    assert isinstance(plan, UploadPlan)
    assert plan.project_name == "my-project"
    assert plan.class_names == ["ball"]
    assert plan.project_exists is None  # not checked in a pure/local computation
    assert plan.total_images == 3
    assert plan.total_boxes == 4  # train: 1 + 2 boxes, val: 1 box

    by_split = {s.split: s for s in plan.splits}
    assert by_split["train"].n_images == 2
    assert by_split["train"].n_boxes == 3  # 1 box + 2 boxes
    assert by_split["val"].n_images == 1
    assert by_split["val"].n_boxes == 1


def test_compute_upload_plan_multiclass(tmp_path: Path) -> None:
    root = _write_yolo_dataset(
        tmp_path / "multi",
        splits={"train": [("a_000000", ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1"])]},
        class_names=["ball", "player"],
    )
    plan = compute_upload_plan(root, project_name="proj")
    assert plan.class_names == ["ball", "player"]
    assert plan.total_boxes == 2


def test_upload_plan_describe_mentions_key_facts(tmp_path: Path) -> None:
    root = _valid_dataset(tmp_path)
    plan = compute_upload_plan(root, project_name="my-project")
    text = plan.describe()
    assert "my-project" in text
    assert "ball" in text
    assert "never modified or deleted" in text


def test_compute_upload_plan_image_without_label_file(tmp_path: Path) -> None:
    """An image with no matching .txt label is counted in n_images but not
    n_labels/n_boxes -- this must not crash the plan computation."""
    root = _write_yolo_dataset(
        tmp_path / "unlabeled",
        splits={"train": [("has_label", ["0 0.5 0.5 0.1 0.1"]), ("no_label", [])]},
    )
    plan = compute_upload_plan(root, project_name="proj")
    train = plan.splits[0]
    assert train.n_images == 2
    assert train.n_labels == 1
    assert train.n_boxes == 1


# --------------------------------------------------------------------------- #
# CLI arg validation                                                          #
# --------------------------------------------------------------------------- #


def test_cli_requires_dataset_dir_or_from_store() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--project", "proj"])


def test_cli_dataset_dir_and_from_store_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--dataset-dir", "/tmp/x", "--from-store", "--project", "proj"]
        )


def test_cli_requires_project() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset-dir", "/tmp/x"])


def test_cli_dry_run_is_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["--dataset-dir", "/tmp/x", "--project", "proj"])
    assert args.yes is False


def test_cli_yes_flag_enables_upload() -> None:
    parser = build_parser()
    args = parser.parse_args(["--dataset-dir", "/tmp/x", "--project", "proj", "--yes"])
    assert args.yes is True


def test_cli_from_store_accepts_passthrough_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--from-store",
            "--project",
            "proj",
            "--db",
            "data/fs.duckdb",
            "--video-dir",
            "eval_data/clips",
            "--val-fraction",
            "0.3",
        ]
    )
    assert args.from_store is True
    assert args.db == Path("data/fs.duckdb")
    assert args.val_fraction == 0.3


def test_cli_default_workspace() -> None:
    parser = build_parser()
    args = parser.parse_args(["--dataset-dir", "/tmp/x", "--project", "proj"])
    assert args.workspace == constants.ROBOFLOW_WORKSPACE


# --------------------------------------------------------------------------- #
# load_api_key                                                                #
# --------------------------------------------------------------------------- #


def test_load_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ROBOFLOW_API_KEY", "env-key-123")
    assert load_api_key() == "env-key-123"


def test_load_api_key_from_config_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    fake_home = tmp_path / "home"
    config_dir = fake_home / ".config" / "roboflow"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(
        json.dumps(
            {"workspaces": {"default": {"url": "egroeg121", "apiKey": "file-key-456"}}}
        )
    )
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    assert load_api_key() == "file-key-456"


def test_load_api_key_missing_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    fake_home = tmp_path / "home"
    config_dir = fake_home / ".config" / "roboflow"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({"workspaces": {}}))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    with pytest.raises(ValueError, match="not found"):
        load_api_key()


# --------------------------------------------------------------------------- #
# Mock-based tests of the upload call sequence                                #
# --------------------------------------------------------------------------- #


def test_get_or_create_project_uses_existing_project() -> None:
    workspace = MagicMock()
    existing_project = MagicMock()
    workspace.project.return_value = existing_project

    project, created = get_or_create_project(workspace, "my-proj", ["ball"])

    assert project is existing_project
    assert created is False
    workspace.project.assert_called_once_with("my-proj")
    workspace.create_project.assert_not_called()


def test_get_or_create_project_creates_when_missing() -> None:
    workspace = MagicMock()
    workspace.project.side_effect = RuntimeError("404 not found")
    new_project = MagicMock()
    workspace.create_project.return_value = new_project

    project, created = get_or_create_project(workspace, "new-proj", ["ball"])

    assert project is new_project
    assert created is True
    workspace.create_project.assert_called_once_with(
        project_name="new-proj",
        project_type="object-detection",
        project_license="MIT",
        annotation="new-proj",
    )


def test_dry_run_makes_zero_sdk_calls(tmp_path: Path, monkeypatch) -> None:
    """The core safety guarantee: without --yes, main() must never import or
    touch the roboflow SDK at all."""
    root = _valid_dataset(tmp_path)

    # Ensure importing `roboflow` blows up loudly if attempted, so any code
    # path that reaches for the SDK during dry-run fails the test instead of
    # silently succeeding against a real/absent package.
    monkeypatch.setitem(sys.modules, "roboflow", None)

    main(["--dataset-dir", str(root), "--project", "proj"])  # no --yes


def test_upload_plan_calls_project_upload_per_image_then_generates_version(
    tmp_path: Path,
) -> None:
    root = _valid_dataset(tmp_path)
    plan = compute_upload_plan(root, project_name="my-proj")

    fake_project = MagicMock()
    fake_project.generate_version.return_value = 7
    fake_workspace = MagicMock()
    fake_workspace.project.return_value = fake_project

    fake_roboflow_instance = MagicMock()
    fake_roboflow_instance.workspace.return_value = fake_workspace
    fake_roboflow_cls = MagicMock(return_value=fake_roboflow_instance)

    fake_roboflow_module = MagicMock()
    fake_roboflow_module.Roboflow = fake_roboflow_cls

    import_saved = sys.modules.get("roboflow")
    sys.modules["roboflow"] = fake_roboflow_module
    try:
        result = upload_plan(plan, api_key="test-key", workspace_name="egroeg121")
    finally:
        if import_saved is not None:
            sys.modules["roboflow"] = import_saved
        else:
            sys.modules.pop("roboflow", None)

    # -- get-or-create project sequence -------------------------------- #
    fake_roboflow_cls.assert_called_once_with(api_key="test-key")
    fake_roboflow_instance.workspace.assert_called_once_with("egroeg121")
    fake_workspace.project.assert_called_once_with("my-proj")
    fake_workspace.create_project.assert_not_called()  # existing project: never create

    # -- one upload call per image, no delete/update calls at all ------- #
    assert fake_project.upload.call_count == plan.total_images
    assert not hasattr(fake_project, "delete") or not fake_project.delete.called
    called_methods = {c[0] for c in fake_project.method_calls}
    assert "delete" not in called_methods
    assert "update" not in called_methods

    # -- version generated exactly once, after all uploads -------------- #
    fake_project.generate_version.assert_called_once_with(
        settings={"preprocessing": {}, "augmentation": {}}
    )
    upload_call_indices = [
        i for i, c in enumerate(fake_project.method_calls) if c[0] == "upload"
    ]
    generate_call_indices = [
        i for i, c in enumerate(fake_project.method_calls) if c[0] == "generate_version"
    ]
    assert max(upload_call_indices) < min(generate_call_indices)

    assert result.project_created is False
    assert result.images_uploaded == plan.total_images
    assert result.images_failed == 0
    assert result.version_number == 7


def test_upload_plan_creates_project_when_missing(tmp_path: Path) -> None:
    root = _valid_dataset(tmp_path)
    plan = compute_upload_plan(root, project_name="brand-new")

    fake_project = MagicMock()
    fake_project.generate_version.return_value = 1
    fake_workspace = MagicMock()
    fake_workspace.project.side_effect = RuntimeError("not found")
    fake_workspace.create_project.return_value = fake_project

    fake_roboflow_instance = MagicMock()
    fake_roboflow_instance.workspace.return_value = fake_workspace
    fake_roboflow_module = MagicMock()
    fake_roboflow_module.Roboflow = MagicMock(return_value=fake_roboflow_instance)

    import_saved = sys.modules.get("roboflow")
    sys.modules["roboflow"] = fake_roboflow_module
    try:
        result = upload_plan(plan, api_key="test-key")
    finally:
        if import_saved is not None:
            sys.modules["roboflow"] = import_saved
        else:
            sys.modules.pop("roboflow", None)

    fake_workspace.create_project.assert_called_once_with(
        project_name="brand-new",
        project_type="object-detection",
        project_license="MIT",
        annotation="brand-new",
    )
    assert result.project_created is True


def test_upload_plan_passes_annotation_path_only_when_label_exists(
    tmp_path: Path,
) -> None:
    root = _write_yolo_dataset(
        tmp_path / "mixed",
        splits={"train": [("has_label", ["0 0.5 0.5 0.1 0.1"]), ("no_label", [])]},
    )
    plan = compute_upload_plan(root, project_name="proj")

    fake_project = MagicMock()
    fake_project.generate_version.return_value = 1
    fake_workspace = MagicMock()
    fake_workspace.project.return_value = fake_project
    fake_roboflow_instance = MagicMock()
    fake_roboflow_instance.workspace.return_value = fake_workspace
    fake_roboflow_module = MagicMock()
    fake_roboflow_module.Roboflow = MagicMock(return_value=fake_roboflow_instance)

    import_saved = sys.modules.get("roboflow")
    sys.modules["roboflow"] = fake_roboflow_module
    try:
        result = upload_plan(plan, api_key="test-key")
    finally:
        if import_saved is not None:
            sys.modules["roboflow"] = import_saved
        else:
            sys.modules.pop("roboflow", None)

    upload_calls = [c for c in fake_project.method_calls if c[0] == "upload"]
    assert len(upload_calls) == 2
    annotation_paths = {c.kwargs["annotation_path"] for c in upload_calls}
    assert None in annotation_paths
    assert any(p is not None for p in annotation_paths)
    assert result.labels_uploaded == 1
    assert result.images_uploaded == 2


# --------------------------------------------------------------------------- #
# Live test -- SKIPPED BY DEFAULT, double opt-in, NEVER RUN during this task  #
# --------------------------------------------------------------------------- #


@pytest.mark.live_roboflow
def test_live_upload_dataset_to_roboflow_dry_run_then_real_upload(
    tmp_path: Path,
) -> None:
    """Full live round trip through the CLI's --yes upload path against a
    throwaway Roboflow project.

    DELIBERATELY NOT EXECUTED as part of normal test runs or during this
    task's development. Same double opt-in as
    ``test_roundtrip_fidelity.test_live_roboflow_upload_roundtrip``: the
    marker AND ``RUN_LIVE_ROBOFLOW_TESTS=1`` are both required.
    """
    if not os.environ.get("RUN_LIVE_ROBOFLOW_TESTS"):
        pytest.skip(
            "live_roboflow tests require RUN_LIVE_ROBOFLOW_TESTS=1 in addition to "
            "-m live_roboflow (double opt-in) -- this creates real, likely-permanent "
            "resources in the production egroeg121 workspace"
        )

    root = _valid_dataset(tmp_path)
    plan = compute_upload_plan(root, project_name="footy-track-roundtrip-test")
    api_key = load_api_key()
    result = upload_plan(
        plan, api_key=api_key, batch_name="upload_dataset_to_roboflow_live_test"
    )

    assert result.images_uploaded == plan.total_images
    assert result.version_number is not None
