"""Model results per (dataset version x model x split / cohort), and contamination flags.

``eval_result`` (see :mod:`footy_track.feature_store.extra_schema`) is append-only: each call to
:func:`record_results` adds rows, and the ``eval_result_latest`` view returns the newest row per
(task, dataset, dataset_version, split, cohort, model, model_version, metric). A result stays tied
to the dataset version it was measured on and is never recomputed against "current labels".

Split values: ``train`` | ``eval_heldout_match`` | ``eval_in_match`` | ``eval_all`` (both eval
tiers). ``cohort_version`` NULL means the full current split; ``N`` means only items whose
``first_version <= N`` (the eval set only grows, so a fixed cohort compares models that were
evaluated on different versions on the same items).

Contamination: ``contaminated`` is true when the model's *training* data contains items of the
evaluated split (so the number is not an honest estimate), false when it was checked and is
clean, and NULL when the training data was not declared. See :func:`contamination_flags`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from footy_track import splits
from footy_track.datasets.snapshot import DatasetVersion, code_sha, datasets_root, load_version
from footy_track.feature_store.lock import default_db_path, store_lock
from footy_track.feature_store.store import FeatureStore

RESULT_SPLITS = ("train", splits.EVAL_HELDOUT, splits.EVAL_IN_MATCH, "eval_all")
TASKS = ("tracking", "broadcast", "calibration")

_COLS = (
    "task", "dataset", "dataset_version", "split", "cohort_version", "model_name", "model_version",
    "metric", "value", "n_items", "run_at", "code_sha", "contaminated", "contamination_note",
)


@dataclass(frozen=True)
class ResultRow:
    task: str
    dataset: str
    dataset_version: int
    split: str
    model_name: str
    model_version: str
    metric: str
    value: float | None
    n_items: int | None = None
    cohort_version: int | None = None
    run_at: str | None = None
    code_sha: str | None = None
    contaminated: bool | None = None
    contamination_note: str | None = None

    def tuple(self, now: str, sha: str | None) -> tuple:
        return (
            self.task, self.dataset, self.dataset_version, self.split, self.cohort_version,
            self.model_name, self.model_version, self.metric,
            None if self.value is None else float(self.value), self.n_items,
            self.run_at or now, self.code_sha or sha, self.contaminated, self.contamination_note,
        )


def _validate(r: ResultRow) -> None:
    if r.split not in RESULT_SPLITS:
        raise ValueError(f"split {r.split!r} not in {RESULT_SPLITS}")
    if r.task not in TASKS:
        raise ValueError(f"task {r.task!r} not in {TASKS}")


def record_results(
    target: FeatureStore | Path | str | None,
    rows: Iterable[ResultRow],
    *,
    now: datetime | None = None,
) -> int:
    """Append ``rows`` to ``eval_result``. ``target`` is an open store (the caller holds any lock),
    a store path, or ``None`` for the default store (these two take the write lock)."""
    rows = list(rows)
    for r in rows:
        _validate(r)
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = code_sha()
    data = [r.tuple(stamp, sha) for r in rows]

    def write(store: FeatureStore) -> None:
        if data:
            store._conn.executemany(  # noqa: SLF001
                f"INSERT INTO eval_result ({', '.join(_COLS)}) VALUES ({', '.join('?' * len(_COLS))})", data
            )

    if isinstance(target, FeatureStore):
        write(target)
        return len(data)
    path = Path(target) if target else default_db_path()
    with store_lock(path, exclusive=True):
        store = FeatureStore.open(path)
        try:
            write(store)
        finally:
            store.close()
    return len(data)


def list_results(
    target: FeatureStore | Path | str | None,
    *,
    dataset: str | None = None,
    task: str | None = None,
    model: str | None = None,
    dataset_version: int | None = None,
    split: str | None = None,
    latest_only: bool = True,
) -> list[dict[str, Any]]:
    """Result rows (newest run per key by default), oldest dataset version first."""
    where, params = [], []
    for col, val in (("dataset", dataset), ("task", task), ("model_name", model),
                     ("dataset_version", dataset_version), ("split", split)):
        if val is not None:
            where.append(f"{col} = ?")
            params.append(val)
    sql = f"SELECT {', '.join(_COLS)} FROM {'eval_result_latest' if latest_only else 'eval_result'}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY dataset, dataset_version, split, coalesce(cohort_version, -1), model_name, model_version, metric, run_at"

    def read(conn: Any) -> list[dict[str, Any]]:
        cur = conn.execute(sql, params)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]

    if isinstance(target, FeatureStore):
        return read(target._conn)  # noqa: SLF001
    path = Path(target) if target else default_db_path()
    if not path.exists():
        return []
    with store_lock(path, exclusive=False):
        conn = duckdb.connect(str(path), read_only=True)
        try:
            return read(conn)
        finally:
            conn.close()


# --------------------------------------------------------------------------------------------
# contamination
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Flag:
    contaminated: bool | None
    note: str


def _combine(flags: dict[str, Flag]) -> Flag:
    hit = [s for s in (splits.EVAL_HELDOUT, splits.EVAL_IN_MATCH) if flags[s].contaminated]
    if hit:
        return Flag(True, "; ".join(f"{s}: {flags[s].note}" for s in hit))
    if all(flags[s].contaminated is None for s in (splits.EVAL_HELDOUT, splits.EVAL_IN_MATCH)):
        return Flag(None, flags[splits.EVAL_HELDOUT].note)
    return Flag(False, "; ".join(f"{s}: {flags[s].note}" for s in (splits.EVAL_HELDOUT, splits.EVAL_IN_MATCH)))


def _with_train_and_all(flags: dict[str, Flag]) -> dict[str, Flag]:
    out = dict(flags)
    out["train"] = Flag(None, "train split: contamination does not apply")
    out["eval_all"] = _combine(flags)
    return out


def contamination_from_roboflow(dataset_dir: Path, *, train_subsets: Sequence[str] = ("train",)) -> dict[str, Flag]:
    """Per result split, whether a Roboflow / YOLO export's training images include eval items
    (via :func:`footy_track.splits.contamination`, which parses image names)."""
    rep = splits.contamination(Path(dataset_dir), train_splits=train_subsets)
    c = rep["train_contaminated"]
    unknown = rep["train_clip_unknown"]
    caveat = (
        f" ({unknown} of {rep['train_images']} training images name a match but not a clip, so "
        "in-match contamination cannot be fully ruled out)"
        if unknown
        else ""
    )
    held = Flag(
        c[splits.EVAL_HELDOUT] > 0,
        f"{c[splits.EVAL_HELDOUT]} of {rep['train_images']} training images are from held-out matches "
        f"{sorted(rep['contaminating_clips'].get(splits.EVAL_HELDOUT, {}))[:5]}"
        if c[splits.EVAL_HELDOUT]
        else f"no held-out-match images among {rep['train_images']} training images",
    )
    inm = Flag(
        c[splits.EVAL_IN_MATCH] > 0,
        f"{c[splits.EVAL_IN_MATCH]} training images are from eval clips "
        f"{sorted(rep['contaminating_clips'].get(splits.EVAL_IN_MATCH, {}))[:5]}{caveat}"
        if c[splits.EVAL_IN_MATCH]
        else f"no eval-clip images among the training images{caveat}",
    )
    return _with_train_and_all({splits.EVAL_HELDOUT: held, splits.EVAL_IN_MATCH: inm})


def contamination_from_dataset(train: DatasetVersion, evaluated: DatasetVersion) -> dict[str, Flag]:
    """A model trained on ``train``'s ``train`` split, evaluated on ``evaluated``'s eval items:
    contaminated where an eval item's clip (in-match) or match (held-out) is in the training
    items. Sticky split assignment makes this empty unless a register was edited."""
    tr = train.select(splits.TRAIN)
    clips = {r["clip"] for r in tr}
    matches = {r["match_key"] for r in tr if r["match_key"]}
    held = [r for r in evaluated.select(splits.EVAL_HELDOUT) if r["match_key"] in matches]
    inm = [r for r in evaluated.select(splits.EVAL_IN_MATCH) if r["clip"] in clips]
    n_h = len(evaluated.select(splits.EVAL_HELDOUT))
    n_i = len(evaluated.select(splits.EVAL_IN_MATCH))
    src = f"{train.dataset} v{train.version} train"
    return _with_train_and_all(
        {
            splits.EVAL_HELDOUT: Flag(bool(held), f"{len(held)}/{n_h} held-out eval items share a match with {src}"),
            splits.EVAL_IN_MATCH: Flag(bool(inm), f"{len(inm)}/{n_i} in-match eval items share a clip with {src}"),
        }
    )


@dataclass(frozen=True)
class TrainingData:
    """What a model was trained on: a Roboflow/YOLO export dir *or* a dataset version."""

    roboflow_dir: Path | None = None
    dataset: str | None = None
    version: int | None = None
    train_subsets: tuple[str, ...] = ("train",)
    root: Path | None = None


def contamination_flags(
    training: TrainingData | None, evaluated: DatasetVersion
) -> dict[str, Flag]:
    """``{result split: Flag}`` for every :data:`RESULT_SPLITS` entry. ``training=None`` means the
    training data is undeclared: every flag is NULL (unchecked), never an implicit clean."""
    if training is None:
        unk = Flag(None, "training data not declared; contamination unchecked")
        return {s: unk for s in RESULT_SPLITS}
    if training.roboflow_dir is not None:
        return contamination_from_roboflow(training.roboflow_dir, train_subsets=training.train_subsets)
    if training.dataset and training.version is not None:
        tr = load_version(training.dataset, training.version, training.root)
        return contamination_from_dataset(tr, evaluated)
    raise ValueError("TrainingData needs roboflow_dir or dataset + version")


@dataclass
class ModelRef:
    name: str
    version: str
    training: TrainingData | None = None
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "RESULT_SPLITS", "TASKS", "Flag", "ModelRef", "ResultRow", "TrainingData",
    "contamination_flags", "contamination_from_dataset", "contamination_from_roboflow",
    "datasets_root", "list_results", "record_results",
]
