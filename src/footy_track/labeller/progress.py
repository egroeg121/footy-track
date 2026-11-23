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

The page itself is two small line charts per stage (dataset size, one headline metric per model);
:data:`PERFORMANCE_SERIES` below says which metric each line plots. Everything else (tables,
diagnostics, warnings) is still in the payload and sits behind the page's "Details" section.
"""

from __future__ import annotations

import fnmatch
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


# --------------------------------------------------------------------------------------------
# what the two charts per stage plot  (edit here to change a chart)
# --------------------------------------------------------------------------------------------
#: Dataset-size lines, the same for every stage: ``(split, legend label)``.
SIZE_SERIES = (
    (splits.TRAIN, "train"),
    (splits.EVAL_IN_MATCH, "eval (in-match)"),
    (splits.EVAL_HELDOUT, "eval (held-out match)"),
)

#: Performance lines per stage, one headline metric per model (keep it to about 3). Each entry:
#:
#: * ``label``: legend text;
#: * ``metric`` / ``split``: the ``eval_result_latest`` metric and split to plot (always the full
#:   current eval, ``cohort_version IS NULL``);
#: * ``model`` / ``exclude_model``: fnmatch patterns on ``model_name`` choosing which model(s) feed the
#:   line (``exclude_model`` is optional);
#: * ``pick`` (optional): when several ``model_version`` rows share one dataset version, ``"latest"``
#:   (default) takes the most recent run and ``"max_n"`` the one scored on most items (ties: latest).
#:
#: Metric names are the ones the adapters write (``datasets/eval_adapters.py``,
#: ``datasets/detector_eval.py``): ``<cls>_<recall|f1>[_10px]`` for the detector, ``ball_recall_10px``
#: for the ball tracker (the same name, told apart by ``model_name = ball_tracker_<kind>``),
#: ``accuracy`` (random stratum only) for broadcast, ``success_at_5px`` for calibration.
PERFORMANCE_SERIES: dict[str, tuple[dict[str, Any], ...]] = {
    "tracking": (
        # Ball found within 10 px: the question tracking needs answered (IoU 0.5 only measures box
        # tightness on an 11 px ball). Held-out match = the clean number.
        {"label": "Ball F1 @10px (held-out)", "metric": "ball_f1_10px", "split": splits.EVAL_HELDOUT,
         "model": "*", "exclude_model": "ball_tracker_*"},
        {"label": "Player recall (in-match)", "metric": "player_recall", "split": splits.EVAL_IN_MATCH,
         "model": "*", "exclude_model": "ball_tracker_*"},
        {"label": "Ball tracker recall @10px (held-out)", "metric": "ball_recall_10px",
         "split": splits.EVAL_HELDOUT, "model": "ball_tracker_*"},
    ),
    "broadcast": (
        # `accuracy` is the random stratum only; the edge / all-biased variants are diagnostics.
        {"label": "Classifier, re-scored (accuracy)", "metric": "accuracy", "split": splits.EVAL_IN_MATCH,
         "model": "broadcast_classifier"},
        {"label": "Calls at review time (accuracy)", "metric": "accuracy", "split": splits.EVAL_IN_MATCH,
         "model": "broadcast_classifier_at_review"},
    ),
    "calibration": (
        # Several proposal runs are scored on one version: show the best-evidenced one.
        {"label": "Reprojection success @5px", "metric": "success_at_5px", "split": splits.EVAL_IN_MATCH,
         "model": "calibration_proposals", "pick": "max_n"},
    ),
}

#: The two detectors (decision 2026-11-21, ``docs/design/full_frame_labelling.md``). The ``tracking``
#: stage is split into these two groups on the page; their series are never pooled. ``perf`` entries
#: have the same shape as :data:`PERFORMANCE_SERIES` lines. ``map50`` (pooled over ball and
#: player/referee) is deliberately in neither group. Metric names are class-prefixed by
#: ``datasets/detector_eval.py`` (``ball_*``, ``player_*``, ``referee_*``); ``ball_tracker_*`` models are
#: ball-only. No track-level metrics (HOTA / IDF1) are recorded by anything yet.
GROUPS: dict[str, dict[str, Any]] = {
    "ball": {
        "label": "Ball detection / tracking",
        "classes": ("ball", "in_play_ball", "out_of_play_ball"),
        "perf": (
            {"label": "Detector F1 @10px (held-out)", "metric": "ball_f1_10px", "split": splits.EVAL_HELDOUT,
             "model": "*", "exclude_model": "ball_tracker_*"},
            {"label": "Detector recall @10px (held-out)", "metric": "ball_recall_10px",
             "split": splits.EVAL_HELDOUT, "model": "*", "exclude_model": "ball_tracker_*"},
            {"label": "Detector recall @10px (in-match)", "metric": "ball_recall_10px",
             "split": splits.EVAL_IN_MATCH, "model": "*", "exclude_model": "ball_tracker_*"},
            {"label": "Tracker recall @10px (held-out)", "metric": "ball_recall_10px",
             "split": splits.EVAL_HELDOUT, "model": "ball_tracker_*"},
        ),
        "record_metric": "ball_recall_10px",
    },
    "player": {
        "label": "Player-side detection / tracking",
        "classes": ("player", "referee", "coach", "player_sub", "person"),
        "perf": (
            {"label": "Player recall (held-out)", "metric": "player_recall", "split": splits.EVAL_HELDOUT,
             "model": "*", "exclude_model": "ball_tracker_*"},
            {"label": "Player recall (in-match)", "metric": "player_recall", "split": splits.EVAL_IN_MATCH,
             "model": "*", "exclude_model": "ball_tracker_*"},
            {"label": "Player F1 (in-match)", "metric": "player_f1", "split": splits.EVAL_IN_MATCH,
             "model": "*", "exclude_model": "ball_tracker_*"},
            {"label": "Referee recall (in-match)", "metric": "referee_recall", "split": splits.EVAL_IN_MATCH,
             "model": "*", "exclude_model": "ball_tracker_*"},
        ),
        "record_metric": "player_recall",
    },
}
_BALL_KINDS = ("ball", "no_ball")  # item kinds that carry a ball verdict (a ball box or a no-ball mark)
_cls_cache: dict[tuple, dict[str, int] | None] = {}

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
    out = {ds: list(models.values()) for ds, models in by_ds.items()}
    for model in out.get("broadcast", []):
        _annotate_broadcast(model)
    return out


# --------------------------------------------------------------------------------------------
# broadcast: random-stratum headline vs diagnostic edge / all-frames metrics
# --------------------------------------------------------------------------------------------
#: A headline computed on fewer random frames than this is flagged "small eval sample".
MIN_HEADLINE_N = 50

_EDGE_NOTE = (
    "Diagnostic only. Edge frames are the ones the classifier was least sure about "
    "(stored p_yes about 0.48-0.52), so any preprocessing difference flips the call and "
    "accuracy there is noise (broadcast v1: stored 0.33, JPEG re-score 0.76, video-seek 0.59)."
)
_BIASED_NOTE = (
    "Diagnostic only. Counts every scored frame, about two thirds of which are edge frames "
    "picked because the classifier was unsure, so this is edge-weighted and not the accuracy on a "
    "random frame. The headline is the random stratum."
)
_HEADLINE_NOTE = "Headline: random-stratum frames only, the unbiased sample."
_UNKNOWN_NOTE = (
    "Frames with no recorded sampling reason (legacy review data). They are never assumed random, "
    "so they are excluded from the headline."
)


def _broadcast_metric_meta(metric: str) -> dict[str, Any]:
    """``{role, note}`` of a broadcast metric: ``headline`` / ``diagnostic`` / ``count``."""
    if metric.endswith("_edge"):
        return {"role": "diagnostic", "note": _EDGE_NOTE}
    if metric.endswith("_all_biased"):
        return {"role": "diagnostic", "note": _BIASED_NOTE}
    if metric == "n_unknown":
        return {"role": "count", "note": _UNKNOWN_NOTE}
    if metric.startswith("n_"):
        return {"role": "count", "note": None}
    return {"role": "headline", "note": _HEADLINE_NOTE}


def _annotate_broadcast(model: dict) -> None:
    """Flag the edge / all-biased metrics as diagnostic, show the headline with its n, and warn
    on a small headline sample. Mutates ``model`` (adds ``metric_meta`` and ``headline``)."""
    metrics = model["metrics"]
    model["metric_meta"] = {m: _broadcast_metric_meta(m) for m in metrics}

    def point_at(metric: str, series: dict, version: int) -> dict | None:
        for s in metrics.get(metric, []):
            if s["split"] == series["split"] and s["cohort_version"] == series["cohort_version"]:
                return next((p for p in s["points"] if p["dataset_version"] == version), None)
        return None

    # every metric's n is the n of its own stratum, not the all-frames n_items of the run
    for metric, series_list in metrics.items():
        n_metric = (
            "n_edge" if metric.endswith("_edge")
            else None if metric.endswith("_all_biased") or metric.startswith("n_")
            else "n_random"
        )
        for s in series_list:
            for p in s["points"]:
                if n_metric:
                    npt = point_at(n_metric, s, p["dataset_version"])
                    if npt and npt["value"] is not None:
                        p["n_items"] = int(npt["value"])
                if metric.endswith("_all_biased"):
                    p["note"] = " | ".join(x for x in (p["note"], "edge-weighted, diagnostic") if x)

    headline = []
    for s in metrics.get("accuracy", []):
        p = max(s["points"], key=lambda q: q["dataset_version"])
        f1 = point_at("f1_yes", s, p["dataset_version"])
        npt = point_at("n_random", s, p["dataset_version"])
        n = int(npt["value"]) if npt and npt["value"] is not None else None
        entry = {
            "split": s["split"], "cohort_version": s["cohort_version"], "tier": s["tier"],
            "dataset_version": p["dataset_version"], "accuracy": p["value"],
            "f1_yes": f1["value"] if f1 else None, "n": n,
            "contaminated": p["contaminated"], "warning": None,
        }
        if n is None:
            entry["warning"] = ("recorded before the random/edge split: this accuracy covers all frames "
                                "(edge-weighted); re-run the eval")
        elif n < MIN_HEADLINE_N:
            entry["warning"] = f"small eval sample: only {n} random frames (< {MIN_HEADLINE_N})"
        headline.append(entry)
    model["headline"] = sorted(headline, key=lambda h: (h["split"], h["cohort_version"] or 0))


# --------------------------------------------------------------------------------------------
# the two charts per stage
# --------------------------------------------------------------------------------------------
def _matches(model_name: str, spec: dict[str, Any]) -> bool:
    if not fnmatch.fnmatchcase(model_name, spec.get("model", "*")):
        return False
    excl = spec.get("exclude_model")
    return not (excl and fnmatch.fnmatchcase(model_name, excl))


def _pick(rows: list[dict], how: str) -> dict:
    if how == "max_n":
        return max(rows, key=lambda r: (r["n_items"] or 0, r["run_at"] or ""))
    return max(rows, key=lambda r: r["run_at"] or "")


def performance_series(dataset: str, rows: list[dict], specs: tuple | None = None) -> list[dict]:
    """One entry per :data:`PERFORMANCE_SERIES` line: ``{key, label, metric, split, points}`` with at
    most one point per dataset version (see the config for how a version's model is chosen)."""
    out = []
    for i, spec in enumerate(PERFORMANCE_SERIES.get(dataset, ()) if specs is None else specs):
        by_version: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            if (r["dataset"] == dataset and r["metric"] == spec["metric"] and r["split"] == spec["split"]
                    and r["cohort_version"] is None and r["value"] is not None and _matches(r["model_name"], spec)):
                by_version[r["dataset_version"]].append(r)
        points = []
        for v in sorted(by_version):
            r = _pick(by_version[v], spec.get("pick", "latest"))
            points.append({
                "version": v, "value": r["value"], "n_items": r["n_items"], "model_name": r["model_name"],
                "model_version": r["model_version"], "run_at": r["run_at"],
                "contaminated": r["contaminated"], "note": r["contamination_note"],
            })
        out.append({"key": f"{dataset}-{i}", "label": spec["label"], "metric": spec["metric"],
                    "split": spec["split"], "points": points})
    return out


def _player_frames(root: Path, dataset: str, version: int) -> dict[str, int] | None:
    """``{split: frames with >= 1 player-side box}`` of one immutable snapshot, or ``None`` if its
    ``items.parquet`` cannot be read. Cached per file (a version never changes)."""
    path = snap.version_dir(dataset, version, root) / "items.parquet"
    key = (str(path), _mtime(path))
    if key in _cls_cache:
        return _cls_cache[key]
    out: dict[str, int] | None
    try:
        import pyarrow.parquet as pq

        tbl = pq.read_table(path, columns=["split", "payload"]).to_pydict()
        out = dict.fromkeys(SPLITS, 0)
        side = set(GROUPS["player"]["classes"])
        for sp, payload in zip(tbl["split"], tbl["payload"], strict=True):
            try:
                boxes = json.loads(payload).get("boxes") or []
            except ValueError:
                continue
            if sp in out and any(b.get("label") in side for b in boxes):
                out[sp] += 1
    except Exception:  # noqa: BLE001 - a missing/odd snapshot only means "no per-group size"
        out = None
    _cls_cache[key] = out
    return out


def _group_charts(root: Path, growth: dict, rows: list[dict]) -> dict[str, Any]:
    """``{ball, player}`` for the tracking stage: each group has its own size chart (frames labelled
    for that group, per split) and performance chart. See :data:`GROUPS` for the attribution."""
    gv = {v["version"]: v for v in growth["versions"]}
    out: dict[str, Any] = {}
    for g, cfg in GROUPS.items():
        perf = []
        for i, spec in enumerate(cfg["perf"]):
            ser = performance_series("tracking", rows, specs=(spec,))[0]
            ser["key"] = f"tracking-{g}-{i}"
            perf.append(ser)
        sizes: dict[int, dict[str, int] | None] = {}
        for v in gv:
            if g == "ball":
                bsk = gv[v]["by_split_kind"]
                sizes[v] = {sp: sum(int((bsk.get(sp) or {}).get(k, 0)) for k in _BALL_KINDS) for sp in SPLITS}
            else:
                sizes[v] = _player_frames(root, "tracking", v)
        size = [{"split": sp, "label": label,
                 "points": [{"version": v, "value": sizes[v][sp]} for v in sorted(gv) if sizes[v] is not None]}
                for sp, label in SIZE_SERIES]
        versions = sorted(set(gv) | {p["version"] for s in perf for p in s["points"]})
        notes = []
        for s in perf:
            if s["split"] == splits.EVAL_HELDOUT and not s["points"]:
                notes.append(f"No held-out results for '{s['metric']}': the held-out eval has no "
                             f"{'ball' if g == 'ball' else 'player-side'} ground truth scored yet.")
        out[g] = {
            "label": cfg["label"], "classes": list(cfg["classes"]),
            "size_unit": "frames with a ball verdict (ball box or no-ball)" if g == "ball"
            else "frames with at least one player-side box",
            "charts": {"versions": [{"version": v, "date": ((gv.get(v) or {}).get("created_at") or "")[:10] or None}
                                    for v in versions],
                       "size": size, "performance": perf},
            "performance_empty": not any(s["points"] for s in perf),
            "notes": list(dict.fromkeys(notes)),
            "record_cmd": ("python -m footy_track.datasets results record --task tracking --dataset tracking "
                           f"--dataset-version <N> --split eval_heldout_match --model <name> --model-version <v> "
                           f"--metric {cfg['record_metric']} --value <x> --n-items <n>"),
        }
    return out


def _charts(dataset: str, growth: dict, rows: list[dict]) -> dict[str, Any]:
    """``{versions, size, performance}`` for one stage: what the page's two charts draw."""
    gv = {v["version"]: v for v in growth["versions"]}
    perf = performance_series(dataset, rows)
    versions = sorted(set(gv) | {p["version"] for s in perf for p in s["points"]})
    size = [{"split": sp, "label": label,
             "points": [{"version": v, "value": gv[v]["by_split"][sp]} for v in sorted(gv)]}
            for sp, label in SIZE_SERIES]
    return {
        "versions": [{"version": v, "date": ((gv.get(v) or {}).get("created_at") or "")[:10] or None}
                     for v in versions],
        "size": size,
        "performance": perf,
    }


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
            "charts": _charts(ds, growth[ds], results["rows"]),
        }
        if ds == "tracking":  # legacy keys above stay; ball and player-side are never pooled below
            datasets[ds]["groups"] = _group_charts(root, growth[ds], results["rows"])
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
