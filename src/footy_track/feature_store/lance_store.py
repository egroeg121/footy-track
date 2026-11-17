"""SPIKE: a LanceDB-backed feature store (storage = Lance, query engine = DuckDB).

Design: ``docs/design/lance_feature_store.md``. This is a small prototype that
proves the load-bearing ideas, not a replacement for ``FeatureStore``:

* one Lance table per DuckDB table; the Arrow schema is *derived from the real
  DDL* in ``schema.py`` (no second schema definition) plus a ``row_hash`` column;
* idempotent upserts via ``merge_insert`` keyed on the table's PK, with a
  pre-diff on ``row_hash`` so re-ingesting identical rows writes nothing and
  creates no new Lance version;
* named dataset versions = Lance tags (``<dataset>-v<N>``) on the dataset's
  tables, plus a row in ``dataset_manifest``;
* SQL (including every view in ``schema.VIEWS``) runs unchanged in DuckDB over
  the Lance tables at the latest version or at any tagged version.

Requires the optional ``lance`` extra (``lancedb``). NOTE: the lancedb native
wheels abort with SIGILL on CPUs without AVX2 (the Mac mini's i7-2635QM) -- see
the design doc. Writers are serialised with an advisory file lock because
concurrent ``merge_insert`` on the same key inserts duplicates (measured).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import lancedb
import pyarrow as pa

from footy_track.feature_store.schema import DDL, TABLES, VIEWS
from footy_track.feature_store.store import _PK, _row_to_value

#: Which Lance tables make up each independently versioned dataset.
DATASET_TABLES: dict[str, tuple[str, ...]] = {
    "tracking": ("game", "run", "frame", "detection", "track_meta", "detection_review"),
    "calibration": ("game", "run", "frame", "calibration", "pitch_keypoint"),
    "broadcast": ("game", "frame", "broadcast_label"),
}

MANIFEST = "dataset_manifest"
RESULTS = "eval_result"

# Tables that exist only in the Lance store (not in the DuckDB DDL).
_EXTRA_SCHEMAS: dict[str, pa.Schema] = {
    # one row per human broadcast-view verdict (today: broadcast_review_labels/*.json)
    "broadcast_label": pa.schema(
        [
            ("game_id", pa.string()),
            ("frame_index", pa.int32()),
            ("verdict", pa.string()),  # yes | no | unsure
            ("p_yes", pa.float32()),
            ("reason", pa.string()),
            ("ts", pa.string()),
        ]
    ),
    MANIFEST: pa.schema(
        [
            ("dataset", pa.string()),
            ("version", pa.int32()),
            ("tag", pa.string()),
            ("created_at", pa.timestamp("us")),
            ("note", pa.string()),
            ("content_hash", pa.string()),
            (
                "split_register_hash",
                pa.string(),
            ),  # hash of match_splits.json + clip register
            ("counts_json", pa.string()),  # {table: rows} and per-label/split counts
            ("table_versions_json", pa.string()),  # {table: lance version}
        ]
    ),
    RESULTS: pa.schema(
        [
            ("task", pa.string()),
            ("dataset", pa.string()),
            ("dataset_version", pa.int32()),
            # split authority value: train | eval_heldout_match | eval_in_match | eval_all
            ("split", pa.string()),
            (
                "cohort_version",
                pa.int32(),
            ),  # eval items first present at this dataset version
            ("model_name", pa.string()),
            ("model_version", pa.string()),
            ("metric", pa.string()),
            ("value", pa.float64()),
            ("run_at", pa.timestamp("us")),
            ("code_sha", pa.string()),
        ]
    ),
}
_EXTRA_PK: dict[str, tuple[str, ...]] = {
    "broadcast_label": ("game_id", "frame_index"),
    MANIFEST: ("dataset", "version"),
    RESULTS: (
        "task",
        "dataset",
        "dataset_version",
        "split",
        "cohort_version",
        "model_name",
        "model_version",
        "metric",
    ),
}


def _ddl_arrow_schemas() -> dict[str, pa.Schema]:
    """Derive each table's Arrow schema from the real DuckDB DDL."""
    con = duckdb.connect(":memory:")
    try:
        for stmt in DDL:
            con.execute(stmt)
        out = {}
        for t in TABLES:
            out[t] = con.execute(f"SELECT * FROM {t} LIMIT 0").to_arrow_table().schema
        return out
    finally:
        con.close()


def _row_hash(values: Sequence[object]) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


@dataclass
class UpsertResult:
    table: str
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    version: int | None = None


class LanceFeatureStore:
    """A directory of Lance tables mirroring the DuckDB feature-store schema."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.root))
        base = _ddl_arrow_schemas()
        self._schemas: dict[str, pa.Schema] = {
            t: base[t].append(pa.field("row_hash", pa.string())) for t in TABLES
        }
        self._schemas.update(_EXTRA_SCHEMAS)
        self._pk: dict[str, tuple[str, ...]] = {**_PK, **_EXTRA_PK}
        self.create_schema()

    @classmethod
    def open(cls, root: str | Path) -> LanceFeatureStore:
        return cls(root)

    def create_schema(self) -> None:
        existing = set(self._db.list_tables().tables)
        for name, schema in self._schemas.items():
            if name not in existing:
                self._db.create_table(name, schema=schema)

    def table(self, name: str):
        return self._db.open_table(name)

    # -- writes (serialised) ------------------------------------------------ #

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        # Concurrent merge_insert on the same key duplicates rows, so all
        # writers take this lock. flock is advisory and unreliable on NFS.
        with open(self.root / ".write.lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _to_arrow(self, name: str, rows: Iterable[object]) -> pa.Table:
        pk = self._pk[name]
        schema = self._schemas[name]
        cols = [f.name for f in schema if f.name != "row_hash"]
        dicts: dict[tuple, dict[str, Any]] = {}
        for row in rows:
            if isinstance(row, dict):
                rec = {c: row.get(c) for c in cols}
            else:  # pydantic row model from schema.py
                rec = {c: _row_to_value(name, c, getattr(row, c)) for c in cols}
            rec["row_hash"] = _row_hash([rec[c] for c in cols if c not in pk])
            dicts[tuple(rec[k] for k in pk)] = rec  # last wins within a batch
        return pa.Table.from_pylist(list(dicts.values()), schema=schema)

    def upsert(self, name: str, rows: Iterable[object]) -> UpsertResult:
        """Idempotent upsert keyed on the table PK. Unchanged rows are skipped,
        so re-ingesting the same data creates no new Lance version."""
        src = self._to_arrow(name, rows)
        res = UpsertResult(name)
        if src.num_rows == 0:
            return res
        pk = self._pk[name]
        with self._write_lock():
            tbl = self.table(name)
            existing = tbl.to_arrow().select([*pk, "row_hash"])  # noqa: F841 - arrow scan
            on = " AND ".join(f's."{k}" IS NOT DISTINCT FROM e."{k}"' for k in pk)
            changed = duckdb.sql(  # noqa: F841 - arrow replacement scan
                f"SELECT s.* FROM src s LEFT JOIN existing e ON {on} "
                "WHERE e.row_hash IS NULL OR e.row_hash <> s.row_hash"
            ).to_arrow_table()
            res.skipped = src.num_rows - changed.num_rows
            if changed.num_rows:
                changed = changed.cast(self._schemas[name])
                r = (
                    tbl.merge_insert(list(pk))
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(changed)
                )
                res.inserted, res.updated = r.num_inserted_rows, r.num_updated_rows
            res.version = self.table(name).version
        return res

    def delete_where(self, name: str, where: str) -> int:
        with self._write_lock():
            tbl = self.table(name)
            before = tbl.count_rows()
            tbl.delete(where)
            return before - self.table(name).count_rows()

    def count(self, name: str, *, tag: str | None = None) -> int:
        return self._open_at(name, tag).count_rows()

    def _open_at(self, name: str, tag: str | None):
        tbl = self.table(name)
        if tag is not None and tag in tbl.tags.list():
            tbl.checkout(tag)
        elif tag is not None:
            raise KeyError(f"table {name!r} has no tag {tag!r}")
        return tbl

    # -- dataset versions ---------------------------------------------------- #

    def dataset_tag(self, dataset: str, version: int) -> str:
        return f"{dataset}-v{version}"

    def tag_dataset(
        self, dataset: str, version: int, note: str = "", split_register_hash: str = ""
    ) -> dict[str, Any]:
        """Freeze the current state of *dataset*'s tables as ``<dataset>-v<N>``."""
        tag = self.dataset_tag(dataset, version)
        tables = DATASET_TABLES[dataset]
        versions, counts, h = {}, {}, hashlib.sha256()
        with self._write_lock():
            for name in tables:
                tbl = self.table(name)
                tbl.tags.create(tag, tbl.version)  # raises if the tag already exists
                versions[name] = tbl.version
                counts[name] = tbl.count_rows()
                pk = self._pk[name]
                data = tbl.to_arrow().select([*pk, "row_hash"])
                data = data.sort_by([(k, "ascending") for k in pk])
                h.update(name.encode())
                h.update(json.dumps(data.to_pylist(), default=str).encode())
            row = {
                "dataset": dataset,
                "version": version,
                "tag": tag,
                "created_at": datetime.now(UTC).replace(tzinfo=None),
                "note": note,
                "content_hash": h.hexdigest(),
                "split_register_hash": split_register_hash,
                "counts_json": json.dumps(counts),
                "table_versions_json": json.dumps(versions),
            }
            self.table(MANIFEST).add(
                pa.Table.from_pylist([row], schema=self._schemas[MANIFEST])
            )
        return row

    def record_result(self, **kw: Any) -> None:
        kw.setdefault("run_at", datetime.now(UTC).replace(tzinfo=None))
        with self._write_lock():
            self.table(RESULTS).add(
                pa.Table.from_pylist([kw], schema=self._schemas[RESULTS])
            )

    # -- query: DuckDB over Lance --------------------------------------------- #

    def duckdb_connection(self, tag: str | None = None) -> duckdb.DuckDBPyConnection:
        """An in-memory DuckDB with every Lance table registered (latest, or as
        of *tag*) plus all ``schema.VIEWS`` -- existing SQL runs unchanged."""
        con = duckdb.connect(":memory:")
        for name in self._schemas:
            tbl = self.table(name)
            if tag is not None and tag in tbl.tags.list():
                tbl.checkout(tag)
            con.register(name, tbl.to_arrow())
        for stmt in VIEWS:
            con.execute(
                stmt.replace("CREATE OR REPLACE VIEW", "CREATE OR REPLACE TEMP VIEW")
            )
        return con

    def query(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        tag: str | None = None,
    ):
        con = self.duckdb_connection(tag)
        try:
            return con.execute(sql, list(params) if params else None).df()
        finally:
            con.close()
