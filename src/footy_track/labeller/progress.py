"""Project progress for the ``/progress`` page: dataset growth, model results, split health.

Everything is read-only and cheap:

* dataset growth: every ``datasets/<dataset>/vN/manifest.json`` under the datasets root (never
  ``items.parquet``);
* model performance: ``eval_result_latest`` in the feature store, opened ``read_only`` for the
  duration of one query, under the store's shared lock (a writer mid-ingest just makes the
  results section report "store busy" instead of blocking or failing the page);
* split health: ``match_splits.json`` and ``clip_splits.json``, read as plain JSON.

Nothing here raises for a missing file: each section reports its own ``state`` (``ok`` /
``empty`` / ``missing`` / ``error`` / ``busy``) plus the CLI command that creates it. The payload
is cached for :data:`CACHE_TTL_S` seconds, keyed on the mtimes of the files it was built from.
See ``docs/design/datasets_and_progress.md``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from footy_track import match_splits as ms
from footy_track import splits
from footy_track.datasets import snapshot as snap
from footy_track.datasets.items import DATASETS
from footy_track.feature_store.lock import StoreLocked, default_db_path, store_lock

CACHE_TTL_S = 60.0
SPLITS = (splits.TRAIN, splits.EVAL_IN_MATCH, splits.EVAL_HELDOUT)

CMD_INGEST = "python -m footy_track.feature_store.ingest_all"
CMD_BUMP = "python -m footy_track.datasets bump all"
CMD_RESULTS = "python -m footy_track.datasets results record --help"
CMD_ASSIGN = "python -m footy_track.splits assign --apply"

#: Where to go to create heldout eval labels, per dataset (a call to action, not a command).
_HELDOUT_TODO = {
    "broadcast": (
        "No held-out broadcast eval items: every broadcast-review label comes from a train match. "
        "Label broadcast-review frames of a held-out match at /broadcast_review"
    ),
    "tracking": "No held-out tracking eval items: label ball marks / Ball Checks in a held-out match.",
    "calibration": "No held-out calibration eval items: calibrate frames of a held-out match at /calibration.",
}

_COLS = (
    "task", "dataset", "dataset_version", "split", "cohort_version", "model_name", "model_version",
    "metric", "value", "n_items", "run_at", "contaminated", "contamination_note",
)

_cache: dict[str, Any] = {"key": None, "t": 0.0, "data": None}
_cache_lock = threading.Lock()


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _mtime(p: Path) -> tuple[int, int] | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read_json(p: Path) -> tuple[Any, str | None]:
    """``(doc, None)`` or ``(None, reason)``; reason is ``"missing"`` or an error string."""
    try:
        return json.loads(p.read_text()), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def _version_manifest_paths(root: Path) -> dict[str, list[tuple[int, Path]]]:
    out: dict[str, list[tuple[int, Path]]] = {}
    for ds in DATASETS:
        try:
            versions = snap.list_versions(ds, root)
        except OSError:
            versions = []
        out[ds] = [(v, snap.version_dir(ds, v, root) / "manifest.json") for v in versions]
    return out


def _cache_key(root: Path, db: Path) -> tuple:
    mans = _version_manifest_paths(root)
    return (
        str(root),
        str(db),
        tuple((ds, v, _mtime(p)) for ds, vs in mans.items() for v, p in vs),
        _mtime(db),
        _mtime(db.with_name(db.name + ".wal")),
        _mtime(ms.register_path()),
        _mtime(splits.clip_register_path()),
    )


# --------------------------------------------------------------------------------------------
# dataset growth
# --------------------------------------------------------------------------------------------
def _version_row(manifest: dict) -> dict:
    counts = manifest.get("counts") or {}
    by_split = {s: int((counts.get("by_split") or {}).get(s, 0)) for s in SPLITS}
    by_split_kind = counts.get("by_split_kind") or {}
    by_kind: dict[str, int] = defaultdict(int)
    for kinds in by_split_kind.values():
        for k, n in kinds.items():
            by_kind[k] += int(n)
    diff = manifest.get("diff_vs_parent") or {}
    return {
        "version": manifest.get("version"),
        "created_at": manifest.get("created_at"),
        "note": manifest.get("note"),
        "n_items": int(counts.get("n_items", sum(by_split.values()))),
        "by_split": by_split,
        "by_kind": dict(sorted(by_kind.items())),
        "by_split_kind": by_split_kind,
        "clips_by_split": counts.get("clips_by_split") or {},
        "parent_version": manifest.get("parent_version"),
        "diff": {
            "added": int(diff.get("added", 0)),
            "removed": int(diff.get("removed", 0)),
            "changed": int(diff.get("changed", 0)),
            "added_by_split": diff.get("added_by_split") or {},
            "removed_by_split": diff.get("removed_by_split") or {},
            "changed_by_split": diff.get("changed_by_split") or {},
        },
        "n_items_unregistered_match": int(manifest.get("n_items_unregistered_match") or 0),
        "split_registers": manifest.get("split_registers") or {},
        "preview": bool((manifest.get("split_registers") or {}).get("preview")),
        "content_hash": (manifest.get("content_hash") or "")[:12],
        "git_sha": (manifest.get("git_sha") or "")[:10] or None,
        "git_dirty": manifest.get("git_dirty"),
    }


def _growth(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for ds, vs in _version_manifest_paths(root).items():
        rows, errors = [], []
        for v, path in vs:
            doc, err = _read_json(path)
            if err or not isinstance(doc, dict):
                errors.append(f"v{v}: {err or 'not a JSON object'}")
                continue
            doc.setdefault("version", v)
            rows.append(_version_row(doc))
        rows.sort(key=lambda r: r["version"])
        out[ds] = {
            "state": "ok" if rows else "empty",
            "versions": rows,
            "latest": rows[-1]["version"] if rows else None,
            "errors": errors,
            "hint": None if rows else CMD_BUMP,
        }
    return out


# --------------------------------------------------------------------------------------------
# model performance
# --------------------------------------------------------------------------------------------
def _read_results(db: Path) -> dict[str, Any]:
    """``{state, rows, error, hint}``; never raises."""
    if not db.is_file():
        return {"state": "missing", "rows": [], "error": None, "hint": CMD_INGEST, "n": 0}
    try:
        with store_lock(db, exclusive=False):
            conn = duckdb.connect(str(db), read_only=True)
            try:
                cur = conn.execute(
                    f"SELECT {', '.join(_COLS)} FROM eval_result_latest "
                    "ORDER BY dataset, dataset_version, split, coalesce(cohort_version, -1), "
                    "model_name, model_version, metric"
                )
                rows = [dict(zip(_COLS, r, strict=True)) for r in cur.fetchall()]
            finally:
                conn.close()
    except StoreLocked:
        return {"state": "busy", "rows": [], "error": "store is being written; retry shortly",
                "hint": None, "n": 0}
    except duckdb.Error as e:
        msg = str(e)
        if "lock" in msg.lower():  # another process holds the file read-write
            return {"state": "busy", "rows": [], "error": "store is being written; retry shortly",
                    "hint": None, "n": 0}
        # an old / empty store without the eval tables: same as "no results yet"
        if "eval_result" in msg and ("does not exist" in msg or "not found" in msg.lower()):
            return {"state": "empty", "rows": [], "error": None, "hint": CMD_RESULTS, "n": 0}
        return {"state": "error", "rows": [], "error": msg.splitlines()[0][:300], "hint": None, "n": 0}
    except OSError as e:
        return {"state": "error", "rows": [], "error": str(e)[:300], "hint": None, "n": 0}
    if not rows:
        return {"state": "empty", "rows": [], "error": None, "hint": CMD_RESULTS, "n": 0}
    return {"state": "ok", "rows": rows, "error": None, "hint": None, "n": len(rows)}


def _tier(split: str, cohort: int | None) -> str:
    return f"{split} / " + ("full current" if cohort is None else f"cohort v{cohort}")


def _group_results(rows: list[dict]) -> dict[str, list[dict]]:
    """``{dataset: [model]}``; a model holds ``metrics: {metric: [series]}`` and a series is one
    (split, cohort) line of points over dataset versions."""
    by_ds: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for r in rows:
        models = by_ds[r["dataset"]]
        mk = (r["model_name"], r["model_version"])
        model = models.setdefault(
            mk,
            {"model_name": r["model_name"], "model_version": r["model_version"], "task": r["task"],
             "metrics": {}, "n_contaminated": 0, "n_unknown": 0},
        )
        series_list = model["metrics"].setdefault(r["metric"], [])
        ck = r["cohort_version"]
        series = next((s for s in series_list if s["split"] == r["split"] and s["cohort_version"] == ck), None)
        if series is None:
            series = {
                "split": r["split"],
                "cohort_version": ck,
                "kind": "full" if ck is None else "cohort",
                "tier": _tier(r["split"], ck),
                "points": [],
            }
            series_list.append(series)
        series["points"].append(
            {
                "dataset_version": r["dataset_version"],
                "value": r["value"],
                "n_items": r["n_items"],
                "run_at": r["run_at"],
                "contaminated": r["contaminated"],  # True | False | None (unknown)
                "note": r["contamination_note"],
            }
        )
        if r["split"] != splits.TRAIN:  # contamination is meaningless for the train split
            if r["contaminated"] is True:
                model["n_contaminated"] += 1
            elif r["contaminated"] is None:
                model["n_unknown"] += 1
    return {ds: list(models.values()) for ds, models in by_ds.items()}


# --------------------------------------------------------------------------------------------
# split health
# --------------------------------------------------------------------------------------------
def _register_health() -> dict[str, Any]:
    mp, cp = ms.register_path(), splits.clip_register_path()
    mdoc, merr = _read_json(mp)
    cdoc, cerr = _read_json(cp)
    out: dict[str, Any] = {"match_splits_path": str(mp), "clip_splits_path": str(cp)}

    if isinstance(mdoc, dict):
        held = sorted(k for k, v in mdoc.items() if isinstance(v, dict) and v.get("split") == ms.HELDOUT)
        train = sorted(k for k, v in mdoc.items() if isinstance(v, dict) and v.get("split") == ms.TRAIN)
        out["matches"] = {"state": "ok", "heldout": len(held), "train": len(train),
                          "heldout_names": held, "hint": None}
    else:
        out["matches"] = {"state": "missing" if merr == "missing" else "error", "error": merr,
                          "heldout": 0, "train": 0, "heldout_names": [],
                          "hint": "create match_splits.json (see docs/data.md)"}

    clips = cdoc.get("clips") if isinstance(cdoc, dict) else None
    if isinstance(clips, dict):
        ev = sum(1 for v in clips.values() if isinstance(v, dict) and v.get("split") == splits.CLIP_EVAL)
        tr = sum(1 for v in clips.values() if isinstance(v, dict) and v.get("split") == splits.CLIP_TRAIN)
        out["clips"] = {"state": "ok", "eval": ev, "train": tr, "hint": None}
    else:
        out["clips"] = {"state": "missing" if cerr == "missing" else "error", "error": cerr,
                        "eval": 0, "train": 0, "hint": CMD_ASSIGN}
    return out


def _health(growth: dict[str, dict], registers: dict[str, Any]) -> dict[str, Any]:
    cur = {
        "match_splits_sha256": snap._file_sha(ms.register_path()),  # noqa: SLF001
        "clip_splits_sha256": snap._file_sha(splits.clip_register_path()),  # noqa: SLF001
    }
    datasets: dict[str, dict] = {}
    for ds, g in growth.items():
        warnings: list[dict] = []
        latest = g["versions"][-1] if g["versions"] else None
        info: dict[str, Any] = {"version": None, "eval_in_match": 0, "eval_heldout_match": 0,
                                "train": 0, "warnings": warnings}
        if latest:
            bs = latest["by_split"]
            info.update(version=latest["version"], train=bs[splits.TRAIN],
                        eval_in_match=bs[splits.EVAL_IN_MATCH], eval_heldout_match=bs[splits.EVAL_HELDOUT])
            if bs[splits.EVAL_HELDOUT] == 0:
                warnings.append({"level": "action", "code": "no_heldout", "message": _HELDOUT_TODO[ds]})
            if bs[splits.EVAL_IN_MATCH] == 0:
                warnings.append({"level": "warn", "code": "no_in_match",
                                 "message": f"No in-match eval items in {ds} v{latest['version']}."})
            if latest["n_items_unregistered_match"]:
                warnings.append({"level": "warn", "code": "unregistered",
                                 "message": f"{latest['n_items_unregistered_match']} items belong to a match that is "
                                            "in neither split register (treated as train)."})
            if latest["preview"]:
                warnings.append({"level": "warn", "code": "preview",
                                 "message": "Built from a preview split (a register file was missing); "
                                            "do not compare against it."})
            sr = latest["split_registers"]
            changed = [k for k in ("match_splits_sha256", "clip_splits_sha256")
                       if sr.get(k) and cur[k] and sr[k] != cur[k]]
            if changed:
                warnings.append({"level": "warn", "code": "registers_changed",
                                 "message": f"Split registers changed since v{latest['version']}; "
                                            f"run `{CMD_BUMP}` to snapshot the new split."})
        datasets[ds] = info
    return {"registers": registers, "datasets": datasets}


# --------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------
def _build(root: Path, db: Path) -> dict[str, Any]:
    growth = _growth(root)
    results = _read_results(db)
    grouped = _group_results(results["rows"])
    registers = _register_health()
    health = _health(growth, registers)
    datasets = {}
    for ds in DATASETS:
        models = grouped.get(ds, [])
        datasets[ds] = {
            "growth": growth[ds],
            "models": models,
            "results_state": results["state"] if not models else "ok",
            "health": health["datasets"][ds],
        }
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paths": {"datasets_root": str(root), "store": str(db)},
        "commands": {"ingest": CMD_INGEST, "bump": CMD_BUMP, "results": CMD_RESULTS, "assign": CMD_ASSIGN},
        "store": {
            "state": results["state"],
            "exists": db.is_file(),
            "error": results["error"],
            "hint": results["hint"],
            "n_results": results["n"],
        },
        "registers": registers,
        "datasets": datasets,
    }


def get_progress(*, force: bool = False, root: Path | None = None, db: Path | None = None) -> dict[str, Any]:
    """The ``/api/progress`` payload (cached ~60 s, invalidated early when a source file changes)."""
    root = snap.datasets_root(root)
    db = Path(db) if db else default_db_path()
    key = _cache_key(root, db)
    now = time.monotonic()
    with _cache_lock:
        if not force and _cache["key"] == key and now - _cache["t"] < CACHE_TTL_S:
            return _cache["data"]
    data = _build(root, db)
    with _cache_lock:
        _cache.update(key=key, t=time.monotonic(), data=data)
    return data


def clear_cache() -> None:
    with _cache_lock:
        _cache.update(key=None, t=0.0, data=None)
