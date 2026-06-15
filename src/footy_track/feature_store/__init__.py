"""footy-track feature store.

A single DuckDB database (over partitioned Parquet) consolidating per-frame
metadata, broadcast classification, pitch segmentation, calibration, object
detections from multiple sources, and tracks. See
``docs/design/feature_store.md``.
"""

from footy_track.feature_store.ingest import (
    classifier_run,
    detector_run,
    ingest_frame,
    to_detection_rows,
    to_frame_row,
)
from footy_track.feature_store.schema import (
    SCHEMA_VERSION,
    DetectionRow,
    FrameRow,
    GameRow,
    Point,
    RunRow,
    Source,
    Stage,
    TrackMetaRow,
)
from footy_track.feature_store.store import FeatureStore

__all__ = [
    "SCHEMA_VERSION",
    "DetectionRow",
    "FeatureStore",
    "FrameRow",
    "GameRow",
    "Point",
    "RunRow",
    "Source",
    "Stage",
    "TrackMetaRow",
    "classifier_run",
    "detector_run",
    "ingest_frame",
    "to_detection_rows",
    "to_frame_row",
]
