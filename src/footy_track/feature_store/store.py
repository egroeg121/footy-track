"""The FeatureStore: a single DuckDB database over the footy-track schema.

Typical use
-----------
    from footy_track.feature_store import FeatureStore, GameRow, FrameRow, DetectionRow

    store = FeatureStore.open("data/feature_store.duckdb")
    store.upsert_games([GameRow(game_id="arsenal_mancity", home_team="Arsenal", ...)])
    store.upsert_frames(frame_rows)
    store.upsert_runs([run_row])
    store.upsert_detections(detection_rows)   # idempotent; re-running replaces

    df = store.query("SELECT * FROM frame_features WHERE is_broadcast")
    traj = store.player_trajectory("arsenal_mancity", track_id=7, source="bytetrack")

Idempotency
-----------
Every ``upsert_*`` is ``INSERT ... ON CONFLICT DO UPDATE`` keyed on the table's
primary key, so re-ingesting the same run never duplicates rows (the
footy-stats invariant). Detections are keyed on
``(game_id, source, run_id, frame_index, detection_id)`` (``detection_id`` is
the per-frame object index) so independent sources (hand_label / yolo / sam3 /
trackers) coexist without overwriting each other.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from footy_track.feature_store.schema import (
    DDL,
    TABLES,
    VIEWS,
    DetectionRow,
    FrameRow,
    GameRow,
    RunRow,
    TrackMetaRow,
)

if TYPE_CHECKING:
    import pandas as pd

# Column order per table, used to build positional INSERT statements that match
# the DDL exactly. Kept here (not derived from pydantic) so the SQL contract is
# explicit and stable across model refactors.
_COLUMNS: dict[str, tuple[str, ...]] = {
    "game": (
        "game_id",
        "home_team",
        "away_team",
        "match_date",
        "venue",
        "source_video_uri",
        "fps",
        "width",
        "height",
        "half1_start_continuous_s",
        "half2_start_continuous_s",
        "game_start_wallclock",
        "schema_version",
    ),
    "run": (
        "run_id",
        "stage",
        "source",
        "model_name",
        "model_version",
        "params_json",
        "created_at",
        "code_version",
        "schema_version",
    ),
    "frame": (
        "game_id",
        "frame_index",
        "frame_uri",
        "width",
        "height",
        "continuous_time_s",
        "half",
        "game_time_s",
        "is_broadcast",
        "broadcast_confidence",
        "broadcast_model_version",
        "pitch_polygon",
        "pitch_seg_threshold",
        "pitch_seg_confidence",
        "pitch_seg_model_version",
        "homography",
        "calibration_quality",
        "calibration_model_version",
    ),
    "detection": (
        "game_id",
        "frame_index",
        "continuous_time_s",
        "detection_id",
        "source",
        "run_id",
        "label",
        "confidence",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "mask_ref",
        "track_id",
        "is_interpolated",
        "needs_review",
    ),
    "track_meta": (
        "game_id",
        "source",
        "run_id",
        "track_id",
        "label",
        "start_frame",
        "end_frame",
        "start_continuous_time_s",
        "end_continuous_time_s",
        "team_id",
        "jersey_number",
        "player_id",
        "reid_parent_track_id",
    ),
}

# Primary-key columns per table — the ON CONFLICT target.
_PK: dict[str, tuple[str, ...]] = {
    "game": ("game_id",),
    "run": ("run_id",),
    "frame": ("game_id", "frame_index"),
    "detection": ("game_id", "source", "run_id", "frame_index", "detection_id"),
    "track_meta": ("game_id", "source", "run_id", "track_id"),
}


def _row_to_value(table: str, column: str, value: object) -> object:
    """Convert a pydantic field value to a DuckDB-bindable value."""
    if value is None:
        return None
    if table == "frame" and column == "pitch_polygon":
        # list[Point] -> list[dict] for the STRUCT(x,y)[] column.
        return [{"x": p.x, "y": p.y} for p in value]  # type: ignore[union-attr]
    return value


class FeatureStore:
    """A handle to one feature-store DuckDB database."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    # -- lifecycle ---------------------------------------------------------- #

    @classmethod
    def open(
        cls, path: str | Path = ":memory:", *, create: bool = True
    ) -> FeatureStore:
        """Open (and by default create) a feature store at *path*.

        ``:memory:`` gives an ephemeral in-process store (used by tests).
        """
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(path))
        store = cls(conn)
        if create:
            store.create_schema()
        return store

    def create_schema(self) -> None:
        """Create all tables and views if they do not already exist."""
        for stmt in DDL:
            self._conn.execute(stmt)
        for stmt in VIEWS:
            self._conn.execute(stmt)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> FeatureStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- ingest (idempotent upserts) --------------------------------------- #

    def _upsert(self, table: str, rows: Iterable[object]) -> int:
        cols = _COLUMNS[table]
        rows = list(rows)
        if not rows:
            return 0

        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(cols)
        pk = _PK[table]
        non_pk = [c for c in cols if c not in pk]
        # ON CONFLICT (pk) DO UPDATE SET col = excluded.col, ...
        if non_pk:
            set_clause = ", ".join(f"{c} = excluded.{c}" for c in non_pk)
            conflict = f"ON CONFLICT ({', '.join(pk)}) DO UPDATE SET {set_clause}"
        else:
            conflict = f"ON CONFLICT ({', '.join(pk)}) DO NOTHING"

        sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) {conflict}"

        params = [
            [_row_to_value(table, c, getattr(row, c)) for c in cols] for row in rows
        ]
        self._conn.executemany(sql, params)
        return len(params)

    def upsert_games(self, rows: Sequence[GameRow]) -> int:
        return self._upsert("game", rows)

    def upsert_runs(self, rows: Sequence[RunRow]) -> int:
        return self._upsert("run", rows)

    def upsert_frames(self, rows: Sequence[FrameRow]) -> int:
        return self._upsert("frame", rows)

    def upsert_detections(self, rows: Sequence[DetectionRow]) -> int:
        return self._upsert("detection", rows)

    def upsert_track_meta(self, rows: Sequence[TrackMetaRow]) -> int:
        return self._upsert("track_meta", rows)

    # -- query -------------------------------------------------------------- #

    def query(self, sql: str, params: Sequence[object] | None = None) -> pd.DataFrame:
        """Run a SQL query and return a pandas DataFrame."""
        rel = self._conn.execute(sql, list(params) if params else None)
        return rel.df()

    def count(self, table: str) -> int:
        return int(self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])

    def player_trajectory(
        self, game_id: str, track_id: int, *, source: str, run_id: str | None = None
    ) -> pd.DataFrame:
        """Return one track's detections across the game, ordered by time.

        This is the cross-game tracking query: every frame the player (track)
        appears in, with box and time, plus their resolved identity.
        """
        sql = (
            "SELECT continuous_time_s, frame_index, bbox_x, bbox_y, bbox_w, bbox_h, "
            "confidence, is_interpolated, team_id, jersey_number, player_id "
            "FROM tracks_enriched "
            "WHERE game_id = ? AND source = ? AND track_id = ?"
        )
        params: list[object] = [game_id, source, track_id]
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY continuous_time_s"
        return self.query(sql, params)

    # Ball-specific label set used by ball_trajectory (matches constants.py).
    _BALL_LABELS: tuple[str, ...] = ("ball", "in_play_ball", "out_of_play_ball")

    def ball_trajectory(
        self,
        game_id: str,
        *,
        source: str,
        run_id: str | None = None,
        labels: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Return all ball detections for a game, ordered by time.

        Queries the detection table for rows whose label is one of the canonical
        ball labels (``ball``, ``in_play_ball``, ``out_of_play_ball``).  Pass
        ``labels`` to override; useful when a tracker uses a different label.

        Returns columns: ``frame_index``, ``continuous_time_s``, ``bbox_x``,
        ``bbox_y``, ``bbox_w``, ``bbox_h``, ``confidence``, ``label``,
        ``is_interpolated``.
        """
        ball_labels = labels if labels is not None else self._BALL_LABELS
        placeholders = ", ".join("?" for _ in ball_labels)
        sql = (
            "SELECT frame_index, continuous_time_s, bbox_x, bbox_y, bbox_w, bbox_h, "
            "confidence, label, is_interpolated "
            "FROM detection "
            f"WHERE game_id = ? AND source = ? AND label IN ({placeholders})"
        )
        params: list[object] = [game_id, source, *ball_labels]
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY continuous_time_s"
        return self.query(sql, params)

    # -- parquet export / import ------------------------------------------- #

    def export_parquet(self, out_dir: str | Path) -> dict[str, Path]:
        """Materialise every table as a partitioned Parquet dataset under
        *out_dir*. ``frame``/``detection``/``track_meta`` partition by game_id
        (and source for the detection-grained tables); ``game``/``run`` are
        single files.

        Returns a mapping of table name -> written path/root.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        for table in TABLES:
            if self.count(table) == 0:
                continue
            if table in ("game", "run"):
                target = out_dir / table / "part.parquet"
                target.parent.mkdir(parents=True, exist_ok=True)
                self._conn.execute(
                    f"COPY (SELECT * FROM {table}) TO '{target}' (FORMAT PARQUET)"
                )
                written[table] = target
            else:
                root = out_dir / table
                partition = (
                    "game_id, source"
                    if table in ("detection", "track_meta")
                    else "game_id"
                )
                self._conn.execute(
                    f"COPY (SELECT * FROM {table}) TO '{root}' "
                    f"(FORMAT PARQUET, PARTITION_BY ({partition}), OVERWRITE_OR_IGNORE)"
                )
                written[table] = root
        return written

    @classmethod
    def from_parquet(
        cls, parquet_dir: str | Path, db_path: str | Path = ":memory:"
    ) -> FeatureStore:
        """Rebuild a store (the DuckDB index) from a Parquet export directory.

        Parquet is the source of truth on disk; the DuckDB file is a
        rebuildable, constraint-enforcing index over it.
        """
        parquet_dir = Path(parquet_dir)
        store = cls.open(db_path, create=True)
        for table in TABLES:
            root = parquet_dir / table
            if not root.exists():
                continue
            cols = ", ".join(_COLUMNS[table])
            store._conn.execute(
                f"INSERT INTO {table} ({cols}) "
                f"SELECT {cols} FROM read_parquet('{root}/**/*.parquet', hive_partitioning=true)"
            )
        return store
