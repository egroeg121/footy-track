"""Upload a local YOLO export directory to Roboflow as a new dataset version
(the store -> Roboflow hop, ft-drs follow-on).

This script takes the output of ``export_training_dataset.py`` (or any
YOLOv8-format directory with the same ``images/{split}``, ``labels/{split}``,
``data.yaml`` layout) and uploads it to a Roboflow project, either an
existing project (by name) or a brand-new one.

**Safety model**:

- Dry-run is the default. Without ``--yes`` the script only inspects the
  local dataset and prints the upload plan (project, image/label counts,
  class list, new-vs-existing project) -- it makes *zero* Roboflow API
  calls.
- Uploads are strictly additive: images are uploaded via
  ``Project.upload(...)``, which adds new images to the project. Nothing in
  this script ever calls a delete/overwrite endpoint, and existing project
  data is never touched.
- ``--yes`` is required to actually talk to the Roboflow API.

Two ways to get a source dataset:

1. Point ``--dataset-dir`` at an existing local YOLO export (e.g. the output
   of ``export_training_dataset.py``).
2. Pass ``--from-store`` plus the same store-selection args
   ``export_training_dataset.py`` takes (``--db``, ``--video-dir``, etc.) to
   run that export fresh and upload its result in one step.

Auth: reuses the same ``ROBOFLOW_API_KEY`` env var / ``~/.config/roboflow/
config.json`` pattern as ``create_roboflow_v11.py`` / ``labelling.py``.

CLI:
    # Dry run against an existing local export (default; makes no network calls)
    uv run python -m footy_track.scripts.upload_dataset_to_roboflow \\
        --dataset-dir data/training_datasets/ball_v1 \\
        --project footy-track-ball-detection

    # Actually upload
    uv run python -m footy_track.scripts.upload_dataset_to_roboflow \\
        --dataset-dir data/training_datasets/ball_v1 \\
        --project footy-track-ball-detection \\
        --yes

    # End-to-end: export from the feature store, then upload
    uv run python -m footy_track.scripts.upload_dataset_to_roboflow \\
        --from-store \\
        --db data/feature_store.duckdb --video-dir eval_data/clips \\
        --out data/training_datasets/ball_v1 \\
        --project footy-track-ball-detection --yes
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from footy_track import constants
from footy_track.feature_store.store import FeatureStore
from footy_track.scripts.export_training_dataset import export as export_from_store

# --------------------------------------------------------------------------- #
# Auth (mirrors create_roboflow_v11.load_api_key exactly)                     #
# --------------------------------------------------------------------------- #


def load_api_key() -> str:
    """Load a Roboflow API key from ``ROBOFLOW_API_KEY`` or the local config
    file. Mirrors ``create_roboflow_v11.load_api_key`` -- do not fork this
    logic, keep both in sync if the auth pattern ever changes."""
    env_key = os.environ.get("ROBOFLOW_API_KEY")
    if env_key:
        return env_key
    config_path = Path.home() / ".config" / "roboflow" / "config.json"
    config = json.loads(config_path.read_text())
    workspaces = config.get("workspaces", {})
    for ws in workspaces.values():
        if ws.get("url") == "egroeg121":
            return ws["apiKey"]
    raise ValueError("Roboflow API key not found in config or environment")


# --------------------------------------------------------------------------- #
# YOLO dataset-dir validation + plan computation (pure, no network)           #
# --------------------------------------------------------------------------- #


class InvalidYoloDatasetError(ValueError):
    """Raised when ``--dataset-dir`` is not a well-formed YOLOv8 export."""


# Roboflow's canonical split names (accepted by Project.upload's ``split``
# param) are "train", "valid", "test" -- our local YOLO export directories
# use "train", "val", "test" (matching Ultralytics' data.yaml convention).
# This maps our local split dir name -> the name we must send to Roboflow.
ROBOFLOW_SPLIT_NAME = {"train": "train", "val": "valid", "test": "test"}


def _roboflow_split_name(local_split: str) -> str:
    """Map a local split directory name ("train"/"val"/"test") to the split
    name Roboflow's API expects. Unknown split dir names pass through
    unchanged (defensive; every split this codebase produces is covered
    above)."""
    return ROBOFLOW_SPLIT_NAME.get(local_split, local_split)


@dataclass
class SplitPlan:
    """Per-split image/label counts for one split (e.g. "train", "val")."""

    split: str
    images: list[Path] = field(default_factory=list)
    labels: list[Path] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def n_labels(self) -> int:
        return sum(1 for p in self.labels if p.stat().st_size > 0)

    @property
    def n_boxes(self) -> int:
        total = 0
        for label_path in self.labels:
            text = label_path.read_text().strip()
            if not text:
                continue
            total += len(text.splitlines())
        return total


@dataclass
class UploadPlan:
    """The full, purely-local upload plan: what would be uploaded and where."""

    dataset_dir: Path
    project_name: str
    project_exists: bool | None  # None until checked against the live API
    class_names: list[str]
    splits: list[SplitPlan]

    @property
    def total_images(self) -> int:
        return sum(s.n_images for s in self.splits)

    @property
    def total_boxes(self) -> int:
        return sum(s.n_boxes for s in self.splits)

    def describe(self) -> str:
        lines = [
            "=== Roboflow upload plan ===",
            f"dataset dir: {self.dataset_dir}",
            f"project: {self.project_name} "
            f"({'existing' if self.project_exists else 'NEW' if self.project_exists is False else 'unknown (dry-run: not checked)'})",
            f"classes ({len(self.class_names)}): {self.class_names}",
        ]
        for s in self.splits:
            lines.append(
                f"  split={s.split} (roboflow split={_roboflow_split_name(s.split)}): "
                f"images={s.n_images} labels={s.n_labels} boxes={s.n_boxes}"
            )
        lines.append(f"total: images={self.total_images} boxes={self.total_boxes}")
        lines.append(
            "split assignment: pinned per-image at upload time from the local "
            "train/val/test dirs (val -> roboflow 'valid'). generate_version is "
            "called with no rebalance/resplit setting, so these pinned "
            "assignments are preserved, not recomputed by Roboflow."
        )
        lines.append(
            "version strategy: upload adds images to the project; a NEW version "
            "is generated afterwards. Existing project data/images/versions are "
            "never modified or deleted."
        )
        return "\n".join(lines)


def _iter_split_dirs(dataset_dir: Path) -> list[str]:
    images_root = dataset_dir / "images"
    if not images_root.is_dir():
        return []
    return sorted(p.name for p in images_root.iterdir() if p.is_dir())


def validate_yolo_dataset(dataset_dir: Path) -> None:
    """Raise :class:`InvalidYoloDatasetError` if *dataset_dir* is not a
    well-formed YOLOv8 export (as produced by ``export_training_dataset.py``).

    Checks:
      - directory exists
      - ``data.yaml`` exists and is parseable, with a non-empty ``names`` list
      - at least one ``images/<split>`` directory exists
      - every ``images/<split>`` directory has a corresponding
        ``labels/<split>`` directory
      - at least one image file exists somewhere under ``images/``
    """
    if not dataset_dir.is_dir():
        raise InvalidYoloDatasetError(f"dataset dir does not exist: {dataset_dir}")

    data_yaml_path = dataset_dir / "data.yaml"
    if not data_yaml_path.is_file():
        raise InvalidYoloDatasetError(f"missing data.yaml in {dataset_dir}")

    try:
        data_yaml = yaml.safe_load(data_yaml_path.read_text())
    except yaml.YAMLError as exc:
        raise InvalidYoloDatasetError(f"data.yaml is not valid YAML: {exc}") from exc

    if not isinstance(data_yaml, dict) or not data_yaml.get("names"):
        raise InvalidYoloDatasetError(
            f"data.yaml missing a non-empty 'names' list: {data_yaml_path}"
        )

    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"
    if not images_root.is_dir():
        raise InvalidYoloDatasetError(f"missing images/ directory in {dataset_dir}")
    if not labels_root.is_dir():
        raise InvalidYoloDatasetError(f"missing labels/ directory in {dataset_dir}")

    splits = _iter_split_dirs(dataset_dir)
    if not splits:
        raise InvalidYoloDatasetError(
            f"images/ has no split subdirectories in {dataset_dir}"
        )

    n_images_total = 0
    for split in splits:
        split_images_dir = images_root / split
        split_labels_dir = labels_root / split
        if not split_labels_dir.is_dir():
            raise InvalidYoloDatasetError(
                f"images/{split} has no matching labels/{split} directory"
            )
        n_images_total += sum(1 for _ in split_images_dir.glob("*.jpg")) + sum(
            1 for _ in split_images_dir.glob("*.png")
        )

    if n_images_total == 0:
        raise InvalidYoloDatasetError(
            f"no image files (.jpg/.png) found under {images_root}"
        )


def compute_upload_plan(dataset_dir: Path, project_name: str) -> UploadPlan:
    """Validate *dataset_dir* and compute the (purely local) upload plan.

    Does not make any network calls; ``project_exists`` is left ``None``
    (caller may fill it in after checking the live API, e.g. right before an
    actual upload).
    """
    validate_yolo_dataset(dataset_dir)
    data_yaml = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    class_names = list(data_yaml["names"])

    splits: list[SplitPlan] = []
    for split in _iter_split_dirs(dataset_dir):
        images = sorted((dataset_dir / "images" / split).glob("*.jpg")) + sorted(
            (dataset_dir / "images" / split).glob("*.png")
        )
        labels_dir = dataset_dir / "labels" / split
        labels = [
            labels_dir / f"{img.stem}.txt"
            for img in images
            if (labels_dir / f"{img.stem}.txt").is_file()
        ]
        splits.append(SplitPlan(split=split, images=images, labels=labels))

    return UploadPlan(
        dataset_dir=dataset_dir,
        project_name=project_name,
        project_exists=None,
        class_names=class_names,
        splits=splits,
    )


# --------------------------------------------------------------------------- #
# Live upload (network calls only happen here, never during dry-run)          #
# --------------------------------------------------------------------------- #


@dataclass
class UploadResult:
    project_name: str
    project_created: bool
    images_uploaded: int
    images_failed: int
    labels_uploaded: int
    version_number: int | None


def get_or_create_project(
    workspace, project_name: str, class_names: list[str]
) -> tuple[object, bool]:
    """Fetch an existing project by name, or create a new one if it doesn't
    exist. Mirrors the try/except pattern used in
    ``test_roundtrip_fidelity.test_live_roboflow_upload_roundtrip`` --
    the Roboflow SDK raises a bare exception on a 404 lookup, there's no
    typed not-found error to catch more narrowly.

    Returns (project, created) where ``created`` is True iff a brand-new
    project was created (never touches/deletes an existing one).
    """
    try:
        project = workspace.project(project_name)
        return project, False
    except Exception:  # noqa: BLE001 - SDK raises assorted errors on 404
        project = workspace.create_project(
            project_name=project_name,
            project_type="object-detection",
            project_license="MIT",
            annotation=project_name,
        )
        return project, True


def upload_plan(
    plan: UploadPlan,
    *,
    api_key: str,
    workspace_name: str = constants.ROBOFLOW_WORKSPACE,
    batch_name: str | None = None,
    generate_version: bool = True,
    progress_every: int = 10,
) -> UploadResult:
    """Execute *plan* against the live Roboflow API: get-or-create the
    project, upload every image+label pair (additive only), and generate a
    new version.

    Only ever ADDS data: uses ``Project.upload`` per image (no delete/update
    calls at all). Never touches pre-existing images or versions.

    **Split pinning**: each image's split is explicitly pinned to the split
    Roboflow expects (mapped from the local train/val/test dir via
    ``_roboflow_split_name`` -- our local ``val`` -> Roboflow's ``valid``).
    This is deliberate: without an explicit, correctly-named ``split`` on
    every upload, Roboflow can rebalance/reassign splits itself at version
    generation time. ``generate_version`` below is called with an empty
    ``preprocessing``/``augmentation`` settings dict -- no rebalance/resplit
    key is set -- so the pinned per-image split assignments from upload time
    are preserved rather than recomputed.

    **Class-name labelmap**: every annotated upload also passes
    ``annotation_labelmap`` (the dataset's ``data.yaml`` path) so Roboflow
    resolves each YOLO ``.txt`` class id to its real class name (e.g.
    ``ball``, ``player``) instead of leaving classes as bare numeric ids.
    """
    from roboflow import Roboflow  # noqa: PLC0415 - keep SDK import lazy/optional

    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(workspace_name)
    project, created = get_or_create_project(
        workspace, plan.project_name, plan.class_names
    )

    batch_name = (
        batch_name or f"upload_dataset_to_roboflow_{time.strftime('%Y%m%d%H%M%S')}"
    )

    labelmap_path = str(plan.dataset_dir / "data.yaml")

    import threading  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from tqdm import tqdm  # noqa: PLC0415

    images_uploaded = 0
    images_failed = 0
    labels_uploaded = 0
    total = plan.total_images
    counter_lock = threading.Lock()

    jobs = []  # (img_path, label_path, roboflow_split)
    for split in plan.splits:
        label_by_stem = {p.stem: p for p in split.labels}
        roboflow_split = _roboflow_split_name(split.split)
        for img_path in split.images:
            jobs.append((img_path, label_by_stem.get(img_path.stem), roboflow_split))

    with tqdm(total=total, desc="uploading", unit="img") as bar:

        def _upload_one(job):
            nonlocal images_uploaded, images_failed, labels_uploaded
            img_path, label_path, roboflow_split = job
            try:
                project.upload(
                    image_path=str(img_path),
                    annotation_path=str(label_path) if label_path else None,
                    annotation_labelmap=labelmap_path if label_path else None,
                    split=roboflow_split,
                    batch_name=batch_name,
                    num_retry_uploads=2,
                )
                with counter_lock:
                    images_uploaded += 1
                    if label_path is not None:
                        labels_uploaded += 1
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                with counter_lock:
                    images_failed += 1
                tqdm.write(f"  FAILED upload {img_path}: {exc}")
            bar.update(1)

        # Uploads are I/O-bound HTTP calls; parallel workers give ~10x
        # throughput. Split/labelmap are per-call parameters so pinning is
        # unaffected by ordering.
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(_upload_one, jobs))

    print(f"  uploaded {images_uploaded + images_failed}/{total} images (final)")

    version_number = None
    if generate_version:
        # No rebalance/resplit key here -- keeps the per-image split
        # assignments pinned at upload time above.
        version_number = project.generate_version(
            settings={"preprocessing": {}, "augmentation": {}}
        )
        print(f"  requested new version: {version_number}")

    return UploadResult(
        project_name=plan.project_name,
        project_created=created,
        images_uploaded=images_uploaded,
        images_failed=images_failed,
        labels_uploaded=labels_uploaded,
        version_number=version_number,
    )


# --------------------------------------------------------------------------- #
# --from-store convenience mode                                               #
# --------------------------------------------------------------------------- #


def run_store_export(args: argparse.Namespace) -> Path:
    """Run ``export_training_dataset.export`` with the args passed through
    from this script's CLI, reusing that module's logic (not forked)."""
    store = FeatureStore.open(args.db)
    export_from_store(
        store,
        out_dir=args.out,
        video_dir=args.video_dir,
        eval_dir=args.eval_dir,
        extra_exclude=set(args.exclude_clip),
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        data_root=args.data_root,
        tag=args.tag,
    )
    return args.out


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-dir",
        type=Path,
        help="Path to an existing local YOLOv8 export directory to upload.",
    )
    source.add_argument(
        "--from-store",
        action="store_true",
        help="Run export_training_dataset's exporter against the feature store first, "
        "then upload its output. Requires --db, --video-dir, --out (and accepts the "
        "same optional args as export_training_dataset.py).",
    )

    parser.add_argument(
        "--project", type=str, required=True, help="Roboflow project name."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=constants.ROBOFLOW_WORKSPACE,
        help=f"Roboflow workspace name (default: {constants.ROBOFLOW_WORKSPACE!r}).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform the upload. Without this flag the script only prints "
        "the upload plan and makes zero Roboflow API calls (default: dry-run).",
    )
    parser.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="Roboflow upload batch name (default: auto-generated timestamped name).",
    )
    parser.add_argument(
        "--no-generate-version",
        action="store_true",
        help="Skip generating a new dataset version after upload (images are still added).",
    )

    # --from-store passthrough args (mirrors export_training_dataset.py's CLI).
    store_group = parser.add_argument_group(
        "--from-store options (passed through to export_training_dataset)"
    )
    store_group.add_argument(
        "--db",
        type=Path,
        help="Feature store DuckDB path (required with --from-store).",
    )
    store_group.add_argument(
        "--video-dir", type=Path, help="Video dir (required with --from-store)."
    )
    store_group.add_argument(
        "--out", type=Path, default=Path("data/training_datasets/ball_v1")
    )
    store_group.add_argument("--eval-dir", type=Path, default=Path("eval_data/clips"))
    store_group.add_argument(
        "--exclude-clip", action="append", default=[], dest="exclude_clip"
    )
    store_group.add_argument("--val-fraction", type=float, default=0.2)
    store_group.add_argument("--test-fraction", type=float, default=0.0)
    store_group.add_argument("--tag", type=str, default="ball_v1")
    store_group.add_argument("--data-root", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.from_store:
        if args.db is None or args.video_dir is None:
            parser.error("--from-store requires --db and --video-dir")
        print("=== running feature-store export first (--from-store) ===")
        dataset_dir = run_store_export(args)
    else:
        dataset_dir = args.dataset_dir

    plan = compute_upload_plan(dataset_dir, args.project)
    print(plan.describe())

    if not args.yes:
        print(
            "\nDry run: no Roboflow API calls were made. Pass --yes to actually upload."
        )
        return

    api_key = load_api_key()
    result = upload_plan(
        plan,
        api_key=api_key,
        workspace_name=args.workspace,
        batch_name=args.batch_name,
        generate_version=not args.no_generate_version,
    )

    print("\n=== upload summary ===")
    print(
        f"project: {result.project_name} ({'created new' if result.project_created else 'existing'})"
    )
    print(f"images uploaded: {result.images_uploaded} (failed: {result.images_failed})")
    print(f"labels uploaded: {result.labels_uploaded}")
    print(
        f"version: {result.version_number if result.version_number is not None else '(not generated)'}"
    )


if __name__ == "__main__":
    main()
