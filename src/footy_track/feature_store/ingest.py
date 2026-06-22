"""Adapters from the existing pipeline's Pydantic outputs into feature-store rows.

The detector (`ObjectDetector.predict_from_path` -> `FrameDetections`) and the
broadcast classifier (`Classifier.predict_from_path` -> `FrameClassifications`)
already produce typed per-frame results. These helpers map those results onto
the feature-store grain (``FrameRow`` / ``DetectionRow``) and write them.

`frame_index` and `continuous_time_s` are NOT carried on the detector/classifier
outputs — they come from the Input stage (see ``docs/timings.md``) and are
passed in explicitly. ``ContinuousTime`` is the canonical timestamp.

These functions are pure mappers (no model loading), so they are fast and
offline-testable. Run the models upstream, hand the results here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from footy_track.feature_store.schema import DetectionRow, FrameRow, RunRow, Stage
from footy_track.schema import (
    BroadcastClassification,
    EnumBroadcastClassification,
    FrameDetections,
)

if TYPE_CHECKING:
    from footy_track.feature_store.store import FeatureStore


def to_detection_rows(
    frame_detections: FrameDetections,
    *,
    clip_id: str,
    game_id: str,
    frame_index: int,
    continuous_time_s: float,
    source: str,
    run_id: str,
    track_ids: list[int | None] | None = None,
) -> list[DetectionRow]:
    """Map a ``FrameDetections`` to ``DetectionRow``s.

    ``detection_id`` is the per-frame object index (0..N-1), matching the
    feature-store primary key. ``track_ids`` (if given) must be aligned with
    ``frame_detections.detections`` and assigns a track to each box.
    """
    dets = frame_detections.detections
    if track_ids is not None and len(track_ids) != len(dets):
        raise ValueError(
            f"track_ids length {len(track_ids)} != detections length {len(dets)}"
        )

    rows: list[DetectionRow] = []
    for i, det in enumerate(dets):
        rows.append(
            DetectionRow(
                clip_id=clip_id,
                game_id=game_id,
                frame_index=frame_index,
                continuous_time_s=continuous_time_s,
                detection_id=i,
                source=source,
                run_id=run_id,
                label=det.label,
                confidence=det.confidence,
                bbox_x=det.x,
                bbox_y=det.y,
                bbox_w=det.w,
                bbox_h=det.h,
                track_id=None if track_ids is None else track_ids[i],
            )
        )
    return rows


def to_frame_row(
    *,
    clip_id: str,
    game_id: str,
    frame_index: int,
    frame_uri: str,
    width: int,
    height: int,
    continuous_time_s: float,
    half: int | None = None,
    game_time_s: float | None = None,
    classification: BroadcastClassification | None = None,
    broadcast_run_id: str | None = None,
) -> FrameRow:
    """Build a ``FrameRow`` from frame metadata + an optional broadcast
    classification. Maps ``YES``/``NO`` to ``is_broadcast`` true/false.
    """
    is_broadcast: bool | None = None
    broadcast_confidence: float | None = None
    if classification is not None:
        is_broadcast = classification.label == EnumBroadcastClassification.YES
        broadcast_confidence = classification.confidence

    return FrameRow(
        clip_id=clip_id,
        game_id=game_id,
        frame_index=frame_index,
        frame_uri=frame_uri,
        width=width,
        height=height,
        continuous_time_s=continuous_time_s,
        half=half,
        game_time_s=game_time_s,
        is_broadcast=is_broadcast,
        broadcast_confidence=broadcast_confidence,
        broadcast_model_version=broadcast_run_id,
    )


def detector_run(
    run_id: str, model_name: str, *, source: str, model_version: str | None = None
) -> RunRow:
    """Convenience: a detection-stage ``RunRow`` (e.g. from ``detector.model_tag``)."""
    return RunRow(
        run_id=run_id,
        stage=Stage.DETECTION,
        source=source,
        model_name=model_name,
        model_version=model_version,
    )


def classifier_run(
    run_id: str, model_name: str, *, model_version: str | None = None
) -> RunRow:
    """Convenience: a broadcast-stage ``RunRow`` for the frame classifier."""
    return RunRow(
        run_id=run_id,
        stage=Stage.BROADCAST,
        source="broadcast_classifier",
        model_name=model_name,
        model_version=model_version,
    )


def ingest_frame(
    store: FeatureStore,
    *,
    clip_id: str,
    game_id: str,
    frame_index: int,
    frame_uri: str,
    width: int,
    height: int,
    continuous_time_s: float,
    half: int | None = None,
    game_time_s: float | None = None,
    classification: BroadcastClassification | None = None,
    broadcast_run_id: str | None = None,
    detections: FrameDetections | None = None,
    detection_source: str | None = None,
    detection_run_id: str | None = None,
    track_ids: list[int | None] | None = None,
) -> int:
    """Write one frame's ``FrameRow`` and (optionally) its ``DetectionRow``s.

    Idempotent: re-ingesting the same frame/run upserts in place. Returns the
    number of detection rows written. The frame spine is written first so the
    detections satisfy the (clip_id, frame_index) relationship.
    """
    store.upsert_frames(
        [
            to_frame_row(
                clip_id=clip_id,
                game_id=game_id,
                frame_index=frame_index,
                frame_uri=frame_uri,
                width=width,
                height=height,
                continuous_time_s=continuous_time_s,
                half=half,
                game_time_s=game_time_s,
                classification=classification,
                broadcast_run_id=broadcast_run_id,
            )
        ]
    )

    if detections is None:
        return 0
    if detection_source is None or detection_run_id is None:
        raise ValueError(
            "detection_source and detection_run_id are required when detections are given"
        )

    rows = to_detection_rows(
        detections,
        clip_id=clip_id,
        game_id=game_id,
        frame_index=frame_index,
        continuous_time_s=continuous_time_s,
        source=detection_source,
        run_id=detection_run_id,
        track_ids=track_ids,
    )
    store.upsert_detections(rows)
    return len(rows)
