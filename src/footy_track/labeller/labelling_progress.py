"""Labelling coverage for ``GET /progress/labelling`` (docs/design/full_frame_labelling.md section 3).

Reads ``frame_label_state`` from the DuckDB feature store (never the JSONL sidecars), opened
``read_only`` under the shared store lock, so it can run on the mini. A missing, empty, locked
or pre-Task-1 store gives ``state != "ok"`` and empty counts, never an exception.

Per clip (``game_id`` == clip stem) and in ``totals``:

* ``frames``: distinct frames that have any ``frame_label_state`` row;
* ``clip_frames``: the clip video's total frame count (cv2, cached by mtime; ``null`` when the
  video is missing/unreadable). Clips with a video but no state rows are listed with zero counts;
* ``ball_labelled`` / ``player_labelled``: that group is ``complete`` or ``absent``;
* ``fully_labelled``: both groups are (the ``frame_complete`` view);
* ``partial``: frames not fully labelled where some group is explicitly ``partial`` OR exactly
  one group is labelled (ball-only or player-only). Frames whose rows are all ``unreviewed``
  are in none of these;
* ``player_marked_sidecar``: distinct frames with at least one player-class box in the labeller
  sidecar JSONL (``PLAYER_LABELS``). Boxes drawn, NOT verified complete; informational only,
  the store counts above stay authoritative.

``by_method`` and ``tiers`` count labelled (``complete``/``absent``) group rows, not frames.
Gold = ``hand_*`` / ``verdict``; silver = ``offline_audited``, broken down by ``teacher_version``;
the sidecar-derived markers are ``other``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb

from footy_track import splits
from footy_track.feature_store.lock import StoreLocked, default_db_path, store_lock
from footy_track.labeller.constants import PLAYER_LABELS

GOLD_METHODS = frozenset({"hand_series", "hand_frame", "verdict"})
SILVER_METHODS = frozenset({"offline_audited"})
_LABELLED = "('complete', 'absent')"
# A ball row with a ``reason`` is not a plain label: absent+not_broadcast is ``unusable`` and
# partial+reason is ``deferred`` (ball_anchor_iteration.md 3.1); neither is "ball labelled".

_FRAMES_SQL = f"""
    SELECT game_id,
           count(*) AS frames,
           count(*) FILTER (WHERE ball IN {_LABELLED} AND ball_reason IS NULL) AS ball_labelled,
           count(*) FILTER (WHERE player IN {_LABELLED}) AS player_labelled,
           count(*) FILTER (WHERE ball IN {_LABELLED} AND ball_reason IS NULL
                            AND player IN {_LABELLED}) AS fully_labelled,
           count(*) FILTER (WHERE NOT (ball IN {_LABELLED} AND player IN {_LABELLED})
                            AND (ball = 'partial' OR player = 'partial'
                                 OR (ball IN {_LABELLED} AND ball_reason IS NULL) != (player IN {_LABELLED}))) AS partial,
           count(*) FILTER (WHERE ball = 'absent' AND ball_reason = 'not_broadcast') AS ball_unusable,
           count(*) FILTER (WHERE ball = 'partial' AND ball_reason IS NOT NULL) AS ball_deferred
    FROM (
        SELECT game_id, frame_index,
               coalesce(max(state) FILTER (WHERE cls_group = 'ball'), 'unreviewed') AS ball,
               coalesce(max(state) FILTER (WHERE cls_group = 'player'), 'unreviewed') AS player,
               max(reason) FILTER (WHERE cls_group = 'ball') AS ball_reason
        FROM frame_label_state GROUP BY game_id, frame_index
    ) GROUP BY game_id
"""
_METHOD_SQL = f"""
    SELECT game_id, method, coalesce(teacher_version, ''), count(*)
    FROM frame_label_state WHERE state IN {_LABELLED} AND reason IS NULL
    GROUP BY ALL
"""


_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
_COUNT_KEYS = (
    "frames", "ball_labelled", "player_labelled", "fully_labelled", "partial", "ball_unusable", "ball_deferred",
)
_file_cache: dict[tuple, Any] = {}


def _empty_counts() -> dict[str, int]:
    return dict.fromkeys(_COUNT_KEYS, 0)


def _cached(kind: str, path: Path, fn) -> Any:  # noqa: ANN001
    try:
        st = path.stat()
    except OSError:
        return None
    key = (kind, str(path), st.st_mtime_ns, st.st_size)
    if key not in _file_cache:
        _file_cache.clear() if len(_file_cache) > 20000 else None
        _file_cache[key] = fn(path)
    return _file_cache[key]


def _video_frames(path: Path) -> int | None:
    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(str(path))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
    except Exception:
        return None
    return n if n > 0 else None


def _sidecar_player_frames(path: Path) -> int:
    frames: set[int] = set()
    try:
        with path.open() as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue
                if d.get("bbox") and PLAYER_LABELS & set(d.get("tags") or ()):
                    frames.add(int(d.get("frame_index", 0)))
    except (OSError, TypeError, ValueError):
        return 0
    return len(frames)


def _default_dirs() -> tuple[Path | None, Path | None]:
    try:
        from footy_track.labeller import datasets  # noqa: PLC0415

        ds = datasets.current()
        return ds.clips_dir, ds.marks_dir
    except Exception:
        return None, None


def _video_index(clips_dir: Path | None) -> dict[str, Path]:
    if clips_dir is None:
        return {}
    try:
        return {
            p.stem: p for p in sorted(clips_dir.iterdir())
            if p.suffix.lower() in _VIDEO_SUFFIXES and not p.name.startswith(".")
        }
    except OSError:
        return {}


def _split_of(game_id: str) -> str:
    try:
        return str(splits.split_of(splits.match_key_of(game_id), game_id))
    except Exception:  # a missing/invalid register must not break the progress page
        return "unknown"


# -- expected set F per job (ball_anchor_iteration.md 3.2 / 3.3) -------------------------------------

STATES = ("ball", "no_ball", "unusable", "deferred")
DEFERRED_FLAG_FRACTION = 0.10
FAMILIES = (("eval", "ball-eval/"), ("train", "ball-round/"))  # separate job_key namespaces

_FP_SQL = """
    SELECT s.job_key, coalesce(s.window_id, ''), count(*) AS n,
           count(*) FILTER (WHERE b.state = 'complete') AS n_ball,
           count(*) FILTER (WHERE b.state = 'absent' AND b.reason IS NULL) AS n_no_ball,
           count(*) FILTER (WHERE b.state = 'absent' AND b.reason = 'not_broadcast') AS n_unusable,
           count(*) FILTER (WHERE b.state = 'absent' AND b.reason IS NOT NULL
                            AND b.reason != 'not_broadcast') AS n_odd,
           count(*) FILTER (WHERE b.state = 'partial' AND b.reason IS NOT NULL) AS n_deferred
    FROM selected_frame s
    LEFT JOIN (SELECT game_id, frame_index, state, reason FROM frame_label_state
               WHERE cls_group = 'ball') b
      ON b.game_id = s.game_id AND b.frame_index = s.frame_index
    GROUP BY s.job_key, coalesce(s.window_id, '')
"""


def family_of(job_key: str) -> str:
    for name, prefix in FAMILIES:
        if job_key.startswith(prefix):
            return name
    return "other"


def _bucket(n: int, ball: int, no_ball: int, unusable: int, deferred: int) -> dict[str, Any]:
    terminal = ball + no_ball + unusable + deferred
    return {
        "frames_total": n,
        "frames_terminal": terminal,
        "by_state": {
            "ball": ball, "no_ball": no_ball, "unusable": unusable, "deferred": deferred,
            "outstanding": n - terminal,
        },
        # design 3.2: done only when every selected frame is terminal; a window with >10% deferred is flagged
        "done": n > 0 and terminal == n,
        "deferred_fraction": round(deferred / n, 4) if n else 0.0,
        "flag_deferred": bool(n) and deferred / n > DEFERRED_FLAG_FRACTION,
    }


def _add(a: dict[str, Any], b: dict[str, Any]) -> None:
    a["frames_total"] += b["frames_total"]
    a["frames_terminal"] += b["frames_terminal"]
    for k, v in b["by_state"].items():
        a["by_state"][k] += v


def _empty_frame_progress(state: str, error: str | None = None) -> dict[str, Any]:
    zero = _bucket(0, 0, 0, 0, 0)
    fams = {name: {"totals": _bucket(0, 0, 0, 0, 0), "jobs": []} for name, _ in (*FAMILIES, ("other", ""))}
    return {"state": state, "error": error, "families": fams, "totals": zero}


def frame_progress_from_rows(rows: list[tuple]) -> dict[str, Any]:
    """Aggregate ``_FP_SQL`` rows into ``{families: {eval|train|other: {totals, jobs: [...]}}, totals}``.
    Each job (and its ``windows``) carries ``frames_total / frames_terminal / by_state / done``."""
    out = _empty_frame_progress("ok" if rows else "empty")
    jobs: dict[str, dict[str, Any]] = {}
    for key, window, n, ball, no_ball, unusable, odd, deferred in rows:
        # absent+other reason is not producible by terminal writers; count it as outstanding (not terminal)
        _ = odd
        w = {"window_id": window or None, **_bucket(n, ball, no_ball, unusable, deferred)}
        j = jobs.setdefault(key, {"job_key": key, "windows": [], **_bucket(0, 0, 0, 0, 0)})
        j["windows"].append(w)
        _add(j, w)
    for j in jobs.values():
        j.update({k: v for k, v in _bucket(
            j["frames_total"], j["by_state"]["ball"], j["by_state"]["no_ball"],
            j["by_state"]["unusable"], j["by_state"]["deferred"]).items()
            if k in ("done", "deferred_fraction", "flag_deferred")})
        j["done"] = bool(j["windows"]) and all(w["done"] for w in j["windows"])
        j["windows"].sort(key=lambda w: w["window_id"] or "")
        fam = out["families"][family_of(j["job_key"])]
        fam["jobs"].append(j)
        _add(fam["totals"], j)
        _add(out["totals"], j)
    for fam in out["families"].values():
        fam["jobs"].sort(key=lambda j: j["job_key"])
        fam["totals"]["done"] = bool(fam["jobs"]) and all(j["done"] for j in fam["jobs"])
    return out


def get_frame_progress(db: Path | None = None) -> dict[str, Any]:
    """Frame-level progress of every job's expected set F (``selected_frame``) against the ball
    terminal states in ``frame_label_state``. Tolerant: a store without ``selected_frame`` (or
    without the ``reason`` column) gives ``state="empty"`` and zero counts, never an exception."""
    db = Path(db) if db else default_db_path()
    if not db.is_file():
        return _empty_frame_progress("missing")
    try:
        with store_lock(db, exclusive=False):
            conn = duckdb.connect(str(db), read_only=True)
            try:
                rows = conn.execute(_FP_SQL).fetchall()
            finally:
                conn.close()
    except StoreLocked:
        return _empty_frame_progress("busy", "store is being written; retry shortly")
    except duckdb.Error as e:
        msg = str(e)
        if "does not exist" in msg or "not found" in msg.lower() or "not have a column" in msg:
            return _empty_frame_progress("empty")
        return _empty_frame_progress("error", msg.splitlines()[0][:300])
    except OSError as e:
        return _empty_frame_progress("error", str(e)[:300])
    return frame_progress_from_rows(rows)


def job_frame_progress(job_key: str, db: Path | None = None) -> dict[str, Any] | None:
    """``{frames_total, frames_terminal, by_state}`` for one job_key, or ``None`` when it has no F rows."""
    fp = get_frame_progress(db)
    for fam in fp["families"].values():
        for j in fam["jobs"]:
            if j["job_key"] == job_key:
                return {k: j[k] for k in ("frames_total", "frames_terminal", "by_state", "done")}
    return None


def _empty(state: str, error: str | None = None) -> dict[str, Any]:
    return {
        "state": state,
        "error": error,
        "clips": [],
        "totals": _empty_counts(),
        "by_method": {},
        "by_split": {},
        "tiers": {"gold": 0, "silver": {"total": 0, "by_teacher_version": {}}, "other": 0},
        "teacher_versions": {},
        "frame_progress": _empty_frame_progress("empty"),
    }


def _read(db: Path) -> tuple[list[tuple], list[tuple]]:
    with store_lock(db, exclusive=False):
        conn = duckdb.connect(str(db), read_only=True)
        try:
            return conn.execute(_FRAMES_SQL).fetchall(), conn.execute(_METHOD_SQL).fetchall()
        finally:
            conn.close()


def _read_store(db: Path) -> tuple[list[tuple], list[tuple], str, str | None]:
    """``(frame_rows, method_rows, state, error)``; never raises."""
    if not db.is_file():
        return [], [], "missing", None
    try:
        frame_rows, method_rows = _read(db)
    except StoreLocked:
        return [], [], "busy", "store is being written; retry shortly"
    except duckdb.Error as e:
        msg = str(e)
        if "frame_label_state" in msg and ("does not exist" in msg or "not found" in msg.lower()):
            return [], [], "empty", None
        return [], [], ("busy" if "lock" in msg.lower() else "error"), msg.splitlines()[0][:300]
    except OSError as e:
        return [], [], "error", str(e)[:300]
    return frame_rows, method_rows, ("ok" if frame_rows else "empty"), None


def get_labelling(
    db: Path | None = None, *, clips_dir: Path | None = None, marks_dir: Path | None = None,
    audit_rates: dict[str, dict] | None = None,
) -> dict[str, Any]:
    db = Path(db) if db else default_db_path()
    if clips_dir is None and marks_dir is None:
        clips_dir, marks_dir = _default_dirs()
    frame_rows, method_rows, state, error = _read_store(db)
    frame_progress = get_frame_progress(db)
    videos = _video_index(clips_dir)
    if state != "ok" and not videos and frame_progress["state"] != "ok":
        return _empty(state, error)

    out = _empty("ok" if state == "ok" else state, error)
    out["frame_progress"] = frame_progress
    totals = out["totals"]
    by_split: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    clips = []
    stored = [tuple(r) for r in frame_rows]
    stored += [(g, 0, 0, 0, 0, 0, 0, 0) for g in videos if g not in {r[0] for r in frame_rows}]
    for game_id, frames, ball, player, full, partial, unusable, deferred in stored:
        video = videos.get(game_id)
        side = marks_dir / f"{game_id}.jsonl" if marks_dir else None
        row = {
            "clip": game_id, "frames": frames, "clip_frames": _cached("v", video, _video_frames) if video else None,
            "ball_labelled": ball, "player_labelled": player, "fully_labelled": full,
            "partial": partial, "ball_unusable": unusable, "ball_deferred": deferred, "split": _split_of(game_id),
            "player_marked_sidecar": (_cached("s", side, _sidecar_player_frames) or 0) if side else 0,
        }
        clips.append(row)
        for agg in (totals, by_split[row["split"]]):
            for k in agg:
                agg[k] += row[k]
    # fewest fully labelled first, then clip name (stable)
    out["clips"] = sorted(clips, key=lambda r: (r["fully_labelled"], r["clip"]))
    out["by_split"] = dict(sorted(by_split.items()))

    by_method: dict[str, int] = defaultdict(int)
    tiers = out["tiers"]
    for _game, method, teacher, n in method_rows:
        by_method[method] += n
        if method in GOLD_METHODS:
            tiers["gold"] += n
        elif method in SILVER_METHODS:
            tiers["silver"]["total"] += n
            tv = tiers["silver"]["by_teacher_version"]
            key = teacher or "unknown"
            tv[key] = tv.get(key, 0) + n
        else:
            tiers["other"] += n
    out["by_method"] = dict(sorted(by_method.items()))
    # per teacher version: silver count + blind-audit pass rate (None when no audit is on record)
    rates = audit_rates or {}
    tvs = tiers["silver"]["by_teacher_version"]
    out["teacher_versions"] = {
        tv: {"silver": tvs.get(tv, 0), "audit": rates.get(tv)} for tv in sorted(set(tvs) | set(rates))
    }
    return out
