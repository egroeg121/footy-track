"""Store tables added for the versioned-dataset system (see ``docs/design/datasets_and_progress.md``).

Kept apart from :mod:`footy_track.feature_store.schema` (the pydantic-backed core tables that
``export_parquet`` / ``from_parquet`` round-trip) because these are plain SQL tables owned by
``ingest_all`` and :mod:`footy_track.datasets`:

``broadcast_label``   human broadcast-review verdicts (``broadcast_review_labels/**/<video>.json``)
``gt_frame``          every frame a ball-GT sidecar mentions, with its skip marker (``no_ball`` /
                      ``not_broadcast``), which the detection spine drops
``dataset_version``   index of the immutable dataset snapshots (the files are authoritative)
``eval_result``       model metrics per (dataset version x model x split / cohort); append-only
"""

from __future__ import annotations

EXTRA_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS broadcast_label (
        video_stem   VARCHAR NOT NULL,   -- the review's match id: the full-match video file stem
        frame_index  BIGINT  NOT NULL,
        verdict      VARCHAR NOT NULL,   -- yes | no | unsure
        p_yes        DOUBLE,             -- the classifier's p(Yes) when the frame was queued
        pred         VARCHAR,            -- the classifier's call then (Yes | No)
        reason       VARCHAR,            -- why the frame was sampled (edge | random | ...)
        ts_s         DOUBLE,             -- position in the video, seconds
        labelled_at  VARCHAR,            -- ISO timestamp of the verdict (not part of dataset hashes)
        source_file  VARCHAR,
        PRIMARY KEY (video_stem, frame_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gt_frame (
        game_id      VARCHAR NOT NULL,
        frame_index  BIGINT  NOT NULL,
        marker       VARCHAR,            -- NULL | no_ball | not_broadcast
        PRIMARY KEY (game_id, frame_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_version (
        dataset         VARCHAR NOT NULL,
        version         INTEGER NOT NULL,
        content_hash    VARCHAR NOT NULL,
        created_at      VARCHAR NOT NULL,
        parent_version  INTEGER,
        n_items         BIGINT,
        git_sha         VARCHAR,
        note            VARCHAR,
        path            VARCHAR,
        manifest_json   VARCHAR,
        PRIMARY KEY (dataset, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_result (
        task                  VARCHAR NOT NULL,   -- tracking | broadcast | calibration
        dataset               VARCHAR NOT NULL,
        dataset_version       INTEGER NOT NULL,
        split                 VARCHAR NOT NULL,   -- train | eval_heldout_match | eval_in_match | eval_all
        cohort_version        INTEGER,            -- NULL = the full current split; N = items with first_version <= N
        model_name            VARCHAR NOT NULL,
        model_version         VARCHAR NOT NULL,   -- weights sha256 / Roboflow version / proposals run id
        metric                VARCHAR NOT NULL,
        value                 DOUBLE,
        n_items               BIGINT,
        run_at                VARCHAR NOT NULL,
        code_sha              VARCHAR,
        contaminated          BOOLEAN,            -- NULL = not checked
        contamination_note    VARCHAR
    )
    """,
    # Append-only history; this view is "the" number: the latest run per key.
    """
    CREATE OR REPLACE VIEW eval_result_latest AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY task, dataset, dataset_version, split,
                             coalesce(cohort_version, -1), model_name, model_version, metric
                ORDER BY run_at DESC, rowid DESC) AS rn
            FROM eval_result)
        WHERE rn = 1
    """,
)

EXTRA_TABLES: tuple[str, ...] = (
    "broadcast_label",
    "gt_frame",
    "dataset_version",
    "eval_result",
)
