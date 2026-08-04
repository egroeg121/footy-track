"""Row models and DuckDB DDL for the footy-track feature store.

The feature store is a single DuckDB database (over partitioned Parquet) that
consolidates every per-frame fact and every object detection for a match. See
``docs/design/feature_store.md`` for the full design.

Grains:
  - ``game``       — one row per match (metadata + GameTime->ContinuousTime map)
  - ``frame``      — one row per (game, frame): path, resolution, clock,
                     broadcast flag, pitch segmentation, calibration
  - ``detection``  — one row per (game, frame, source, run, object)
  - ``track_meta`` — one row per track (post-hoc identity / span)
  - ``run``        — provenance for every produced artifact

All timestamps are ContinuousTime (seconds from kickoff); all boxes are
normalised top-left xywh in ``[0, 1]`` — matching ``footy_track.schema`` and
the cross-stage invariants in ``docs/system_design.md`` §4.

Canonical-detection semantics (``detections_enriched.canonical``)
-------------------------------------------------------------------
``canonical`` picks, for every ``(game_id, frame_index)``, which *batch* of
detection rows an export/training query should use. The unit of arbitration
is the **batch** (a run or run-family from one coherent import/flush), not
the individual row and not the row's source tier.

Why: the labeller (``ingest_gt.py``) enforces label precedence *per box*
before it ever flushes a sidecar — a hand-corrected object's machine box is
dropped or superseded, but machine boxes for *other* objects on that same
frame are deliberately left in place and are just as valid. A flushed frame
therefore already is one coherent, internally-resolved snapshot (e.g. 2
``hand_label`` player rows + 13 ``vittrack`` rows). Re-adjudicating rows
individually by source tier — the old behaviour — silently discards the
coexisting machine rows on every hand-corrected frame, because "any hand row
present on this frame" used to demote every machine row on that frame to
non-canonical regardless of which import produced them.

The rule:

1. Every detection row belongs to a **run_group**:
   - Rows from a GT-marks sidecar flush (``ingest_gt.py``) all share the
     ``run_id`` family ``gt_import_<provenance_tag>`` (one run per
     provenance tag actually present in that flush, e.g.
     ``gt_import_labeller``, ``gt_import_vittrack``). All ``gt_import_*``
     runs for the same ``game_id`` are treated as **one run_group**
     (``gt_import:<game_id>``), because they came from one sidecar file /
     one labeller pass and the labeller already resolved per-box
     precedence across them.
   - Every other ``run_id`` (e.g. a Roboflow dataset import
     ``roboflow_v<N>``, or a future non-sidecar import) is its own,
     independent run_group.
2. Per ``(game_id, frame_index)``, run_groups are ranked:
   a. a group containing at least one ``hand_label`` row outranks one that
      doesn't (human ground truth wins);
   b. among groups that tie on (a), the GT-marks sidecar's run_group
      (``gt_import:<game_id>``) outranks a Roboflow/other import — the
      sidecar is the labeller's live, per-box-resolved snapshot and is
      treated as the more current source of truth than a static dataset
      export;
   c. remaining ties broken by ``max(run_id)`` descending — a deliberately
      simple, deterministic tiebreaker given ``run.created_at`` is not
      populated by either importer today.
3. **Every row in the winning run_group is canonical** — not just the
   hand-labelled ones. Rows in losing run_groups are all non-canonical.

This means: a frame touched only by one import (Roboflow-only, or
sidecar-only with no hand labels) stays fully canonical, unchanged. A frame
covered by *both* a Roboflow import and a sidecar flush containing a hand
label picks the sidecar's entire run_group. A hand-corrected sidecar frame
(hand_label + coexisting machine-source rows from the same flush) keeps
*all* of those rows canonical, matching what the labeller already decided.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


class _Row(BaseModel):
    """Base for feature-store row models (frozen, extra forbidden)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Stage(StrEnum):
    """The pipeline stage a run belongs to."""

    BROADCAST = "broadcast"
    DETECTION = "detection"
    PITCH_SEG = "pitch_seg"
    CALIBRATION = "calibration"
    TRACKING = "tracking"


class Source(StrEnum):
    """Known detection/track sources. Free strings are accepted at the DB layer;
    this enum documents the canonical values."""

    HAND_LABEL = "hand_label"
    YOLO = "yolo"
    SAM3 = "sam3"
    VITTRACK = "vittrack"
    BYTETRACK = "bytetrack"
    BOTSORT = "botsort"


class Point(BaseModel):
    """A normalised 2D point (pitch polygon vertex)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)


class GameRow(_Row):
    """One match. Subsumes the old ``GameMetadata`` sidecar block."""

    game_id: str
    home_team: str | None = None
    away_team: str | None = None
    match_date: date | None = None
    venue: str | None = None
    source_video_uri: str | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    half1_start_continuous_s: float = 0.0
    half2_start_continuous_s: float | None = None
    game_start_wallclock: datetime | None = None
    schema_version: str = SCHEMA_VERSION


class FrameRow(_Row):
    """The per-frame spine: one row per (game, frame), broadcast or not."""

    game_id: str
    frame_index: int = Field(..., ge=0)
    frame_uri: str
    width: int
    height: int
    continuous_time_s: float
    half: int | None = Field(None, ge=1, le=2)
    game_time_s: float | None = None
    # broadcast classification
    is_broadcast: bool | None = None
    broadcast_confidence: float | None = Field(None, ge=0.0, le=1.0)
    broadcast_model_version: str | None = None
    # pitch segmentation (final polygon + allowance threshold)
    pitch_polygon: list[Point] | None = None
    pitch_seg_threshold: float | None = None
    pitch_seg_confidence: float | None = Field(None, ge=0.0, le=1.0)
    pitch_seg_model_version: str | None = None
    # calibration (image -> pitch homography, flattened 3x3)
    homography: list[float] | None = None
    calibration_quality: float | None = None
    calibration_model_version: str | None = None


class DetectionRow(_Row):
    """One detected object from one source/run. The high-volume table."""

    game_id: str
    frame_index: int = Field(..., ge=0)
    continuous_time_s: float
    detection_id: int = Field(..., ge=0, description="Per-frame object index (0..N-1)")
    source: str
    run_id: str
    label: str
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    bbox_x: float = Field(..., ge=0.0, le=1.0)
    bbox_y: float = Field(..., ge=0.0, le=1.0)
    bbox_w: float = Field(..., ge=0.0, le=1.0)
    bbox_h: float = Field(..., ge=0.0, le=1.0)
    mask_ref: str | None = None
    track_id: int | None = None
    is_interpolated: bool = False
    needs_review: bool = False
    reviewed: bool = False
    dataset_tag: str | None = None


class TrackMetaRow(_Row):
    """Per-track summary + identity. Updated post-hoc without touching rows."""

    game_id: str
    source: str
    run_id: str
    track_id: int
    label: str
    start_frame: int
    end_frame: int
    start_continuous_time_s: float
    end_continuous_time_s: float
    team_id: str | None = None
    jersey_number: int | None = None
    player_id: str | None = None
    reid_parent_track_id: int | None = None


class RunRow(_Row):
    """Provenance for every produced artifact. Referenced by *_model_version /
    run_id columns elsewhere."""

    run_id: str
    stage: str
    source: str
    model_name: str
    model_version: str | None = None
    params_json: str | None = None
    created_at: datetime | None = None
    code_version: str | None = None
    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# DuckDB DDL                                                                   #
# --------------------------------------------------------------------------- #

# Note: DuckDB enforces PRIMARY KEY uniqueness, which we rely on for the
# ON CONFLICT upsert path (see store.py). Polygon stored as a native
# LIST<STRUCT(x,y)>; homography as a flat DOUBLE[9].

DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS game (
        game_id                   VARCHAR PRIMARY KEY,
        home_team                 VARCHAR,
        away_team                 VARCHAR,
        match_date                DATE,
        venue                     VARCHAR,
        source_video_uri          VARCHAR,
        fps                       DOUBLE,
        width                     INTEGER,
        height                    INTEGER,
        half1_start_continuous_s  DOUBLE,
        half2_start_continuous_s  DOUBLE,
        game_start_wallclock      TIMESTAMP,
        schema_version            VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS run (
        run_id          VARCHAR PRIMARY KEY,
        stage           VARCHAR,
        source          VARCHAR,
        model_name      VARCHAR,
        model_version   VARCHAR,
        params_json     VARCHAR,
        created_at      TIMESTAMP,
        code_version    VARCHAR,
        schema_version  VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS frame (
        game_id                     VARCHAR,
        frame_index                 INTEGER,
        frame_uri                   VARCHAR,
        width                       INTEGER,
        height                      INTEGER,
        continuous_time_s           DOUBLE,
        half                        TINYINT,
        game_time_s                 DOUBLE,
        is_broadcast                BOOLEAN,
        broadcast_confidence        FLOAT,
        broadcast_model_version     VARCHAR,
        pitch_polygon               STRUCT(x DOUBLE, y DOUBLE)[],
        pitch_seg_threshold         FLOAT,
        pitch_seg_confidence        FLOAT,
        pitch_seg_model_version     VARCHAR,
        homography                  DOUBLE[],
        calibration_quality         FLOAT,
        calibration_model_version   VARCHAR,
        PRIMARY KEY (game_id, frame_index)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS detection (
        game_id            VARCHAR,
        frame_index        INTEGER,
        continuous_time_s  DOUBLE,
        detection_id       BIGINT,
        source             VARCHAR,
        run_id             VARCHAR,
        label              VARCHAR,
        confidence         FLOAT,
        bbox_x             FLOAT,
        bbox_y             FLOAT,
        bbox_w             FLOAT,
        bbox_h             FLOAT,
        mask_ref           VARCHAR,
        track_id           INTEGER,
        is_interpolated    BOOLEAN,
        needs_review       BOOLEAN DEFAULT FALSE,
        reviewed           BOOLEAN DEFAULT FALSE,
        dataset_tag        VARCHAR,
        PRIMARY KEY (game_id, source, run_id, frame_index, detection_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS track_meta (
        game_id                   VARCHAR,
        source                    VARCHAR,
        run_id                    VARCHAR,
        track_id                  INTEGER,
        label                     VARCHAR,
        start_frame               INTEGER,
        end_frame                 INTEGER,
        start_continuous_time_s   DOUBLE,
        end_continuous_time_s     DOUBLE,
        team_id                   VARCHAR,
        jersey_number             INTEGER,
        player_id                 VARCHAR,
        reid_parent_track_id      INTEGER,
        PRIMARY KEY (game_id, source, run_id, track_id)
    );
    """,
)

# Convenience views — the "one wide table" ergonomics the design promises.
VIEWS: tuple[str, ...] = (
    """
    CREATE OR REPLACE VIEW frame_features AS
        SELECT f.*,
               g.home_team, g.away_team, g.match_date, g.venue,
               g.fps AS game_fps,
               g.half2_start_continuous_s
        FROM frame f
        JOIN game g USING (game_id);
    """,
    """
    CREATE OR REPLACE VIEW detections_enriched AS
        WITH scored AS (
            SELECT d.*,
                   -- Rows imported together from one GT-marks sidecar flush
                   -- (ingest_gt.py) share the run-id family "gt_import_*" --
                   -- one run per provenance tag seen in that flush -- but
                   -- are one coherent per-frame snapshot: the labeller has
                   -- already resolved hand-vs-machine precedence per box
                   -- before writing the sidecar, so hand_label rows and the
                   -- machine rows they coexist with on the same frame must
                   -- win or lose canonical status *together*. Any other
                   -- run_id (e.g. a Roboflow dataset import "roboflow_v*")
                   -- is its own independent, competing batch.
                   CASE
                       WHEN d.run_id LIKE 'gt_import_%' THEN 'gt_import:' || d.game_id
                       ELSE d.run_id
                   END AS run_group,
                   (d.run_id LIKE 'gt_import_%') AS is_gt_import
            FROM detection d
        ),
        group_rank AS (
            SELECT game_id, frame_index, run_group,
                   RANK() OVER (
                       PARTITION BY game_id, frame_index
                       ORDER BY
                           -- A batch that contains any hand-labelled row
                           -- (human ground truth) always outranks a batch
                           -- that does not -- this is what makes a sidecar
                           -- flush win over a plain Roboflow import when
                           -- both cover the same frame.
                           bool_or(source = 'hand_label') DESC,
                           -- Among hand-label-containing batches, the GT-marks
                           -- sidecar flush (the labeller's live, per-box-
                           -- resolved snapshot) outranks a static Roboflow
                           -- dataset import -- the sidecar is the more
                           -- current, more granular source of truth.
                           bool_or(is_gt_import) DESC,
                           max(run_id) DESC
                   ) AS rnk
            FROM scored
            GROUP BY game_id, frame_index, run_group
        )
        SELECT s.game_id, s.frame_index, s.continuous_time_s, s.detection_id,
               s.source, s.run_id, s.label, s.confidence,
               s.bbox_x, s.bbox_y, s.bbox_w, s.bbox_h, s.mask_ref, s.track_id,
               s.is_interpolated, s.needs_review, s.reviewed, s.dataset_tag,
               f.frame_uri, f.is_broadcast, f.half, f.game_time_s,
               r.stage, r.model_name, r.model_version,
               (gr.rnk = 1) AS canonical
        FROM scored s
        JOIN frame f USING (game_id, frame_index)
        LEFT JOIN run r USING (run_id)
        JOIN group_rank gr USING (game_id, frame_index, run_group);
    """,
    """
    CREATE OR REPLACE VIEW tracks_enriched AS
        SELECT d.*,
               t.label AS track_label,
               t.team_id, t.jersey_number, t.player_id,
               t.start_frame, t.end_frame,
               t.start_continuous_time_s, t.end_continuous_time_s,
               t.reid_parent_track_id
        FROM detection d
        JOIN track_meta t USING (game_id, source, run_id, track_id)
        WHERE d.track_id IS NOT NULL;
    """,
)

# Logical order in which tables are exported/imported as Parquet partitions.
TABLES: tuple[str, ...] = ("game", "run", "frame", "detection", "track_meta")
