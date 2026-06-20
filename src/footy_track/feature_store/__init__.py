"""footy-track feature store.

A single DuckDB database (over partitioned Parquet) consolidating per-frame
metadata, broadcast classification, pitch segmentation, calibration, object
detections from multiple sources, and tracks. See
``docs/design/feature_store.md``.
"""

from footy_track.feature_store.difficulty import (
    DifficultyReport,
    flag_for_review,
    score_detections,
)
from footy_track.feature_store.importers import (
    ImportReport,
    import_labeller_json,
    import_roboflow,
    source_overlap,
)
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
    "DifficultyReport",
    "FeatureStore",
    "FrameRow",
    "GameRow",
    "ImportReport",
    "Point",
    "RunRow",
    "Source",
    "Stage",
    "TrackMetaRow",
    "classifier_run",
    "detector_run",
    "flag_for_review",
    "import_labeller_json",
    "import_roboflow",
    "ingest_frame",
    "score_detections",
    "source_overlap",
    "to_detection_rows",
    "to_frame_row",
]
