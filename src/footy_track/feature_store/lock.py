"""Single-writer file lock for the feature-store DuckDB file.

DuckDB lets one process hold a file read-write, so two ingesters racing would fail with an
opaque IO error (or, worse, queue forever). Every writer (``ingest_all``, ``datasets bump``,
``datasets results record``) takes the exclusive lock ``<db>.lock`` first; readers that open the
file read-only take the shared lock so they never collide with a writer mid-run.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB = "/mnt/storage/footy_data/feature_store/feature_store.duckdb"
ENV_DB = "FOOTY_FEATURE_STORE"


class StoreLocked(RuntimeError):
    """Another process holds the store's write lock."""


def default_db_path() -> Path:
    return Path(os.environ.get(ENV_DB) or DEFAULT_DB)


@contextmanager
def store_lock(
    db: Path, *, exclusive: bool = True, timeout: float = 0.0
) -> Iterator[None]:
    """Hold ``<db>.lock`` (exclusive for writers, shared for readers).

    ``timeout=0`` fails immediately with :class:`StoreLocked`; a positive value polls that long.
    """
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    lock_path = db.with_name(db.name + ".lock")
    flag = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
    with open(lock_path, "a") as lf:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lf, flag)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    kind = "write" if exclusive else "read"
                    raise StoreLocked(
                        f"feature store {db} is locked by another process ({kind} lock "
                        f"{lock_path}); single writer only, retry shortly"
                    ) from None
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
