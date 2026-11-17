"""Versioned, content-hashed datasets (tracking, broadcast, calibration) and model results.

See ``docs/design/datasets_and_progress.md``. CLI: ``python -m footy_track.datasets``.
"""

from footy_track.datasets.items import DATASETS, DatasetBuildError, Item, build_items
from footy_track.datasets.results import (
    ModelRef,
    ResultRow,
    TrainingData,
    contamination_flags,
    list_results,
    record_results,
)
from footy_track.datasets.snapshot import (
    BumpResult,
    DatasetVersion,
    SnapshotError,
    SnapshotExists,
    SnapshotTampered,
    build,
    bump,
    content_hash,
    diff_versions,
    latest_version,
    list_versions,
    load_version,
)

__all__ = [
    "DATASETS",
    "BumpResult",
    "DatasetBuildError",
    "DatasetVersion",
    "Item",
    "ModelRef",
    "ResultRow",
    "SnapshotError",
    "SnapshotExists",
    "SnapshotTampered",
    "TrainingData",
    "build",
    "build_items",
    "bump",
    "content_hash",
    "contamination_flags",
    "diff_versions",
    "latest_version",
    "list_results",
    "list_versions",
    "load_version",
    "record_results",
]
