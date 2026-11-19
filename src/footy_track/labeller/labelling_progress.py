"""Labelling coverage for ``GET /progress/labelling`` (docs/design/full_frame_labelling.md section 3).

Reads ``frame_label_state`` from the DuckDB feature store (never the JSONL sidecars), opened
``read_only`` under the shared store lock, so it can run on the mini. A missing, empty, locked
or pre-Task-1 store gives ``state != "ok"`` and empty counts, never an exception.

Per clip (``game_id`` == clip stem) and in ``totals``:

* ``frames``: distinct frames that have any ``frame_label_state`` row (frames somebody has
  judged at least one class group of; the store does not know a clip's real frame count);
* ``ball_labelled`` / ``player_labelled``: that group is ``complete`` or ``absent``;
* ``fully_labelled``: both groups are (the ``frame_complete`` view);
* ``partial``: frames with some row but not fully labelled.

``by_method`` and ``tiers`` count labelled (``complete``/``absent``) group rows, not frames.
Gold = ``hand_*`` / ``verdict``; silver = ``offline_audited``, broken down by ``teacher_version``;
the sidecar-derived markers are ``other``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb

from footy_track import splits
from footy_track.feature_store.lock import StoreLocked, default_db_path, store_lock

GOLD_METHODS = frozenset({"hand_series", "hand_frame", "verdict"})
SILVER_METHODS = frozenset({"offline_audited"})
_LABELLED = "('complete', 'absent')"

_FRAMES_SQL = f"""
    SELECT game_id,
           count(*) AS frames,
           count(*) FILTER (WHERE ball IN {_LABELLED}) AS ball_labelled,
           count(*) FILTER (WHERE player IN {_LABELLED}) AS player_labelled,
           count(*) FILTER (WHERE ball IN {_LABELLED} AND player IN {_LABELLED}) AS fully_labelled
    FROM (
        SELECT game_id, frame_index,
               max(state) FILTER (WHERE cls_group = 'ball') AS ball,
               max(state) FILTER (WHERE cls_group = 'player') AS player
        FROM frame_label_state GROUP BY game_id, frame_index
    ) GROUP BY game_id
"""
_METHOD_SQL = f"""
    SELECT game_id, method, coalesce(teacher_version, ''), count(*)
    FROM frame_label_state WHERE state IN {_LABELLED}
    GROUP BY ALL
"""


def _empty_counts() -> dict[str, int]:
    return {"frames": 0, "ball_labelled": 0, "player_labelled": 0, "fully_labelled": 0, "partial": 0}


def _split_of(game_id: str) -> str:
    try:
        return str(splits.split_of(splits.match_key_of(game_id), game_id))
    except Exception:  # a missing/invalid register must not break the progress page
        return "unknown"


def _empty(state: str, error: str | None = None) -> dict[str, Any]:
    return {
        "state": state,
        "error": error,
        "clips": [],
        "totals": _empty_counts(),
        "by_method": {},
        "by_split": {},
        "tiers": {"gold": 0, "silver": {"total": 0, "by_teacher_version": {}}, "other": 0},
    }


def _read(db: Path) -> tuple[list[tuple], list[tuple]]:
    with store_lock(db, exclusive=False):
        conn = duckdb.connect(str(db), read_only=True)
        try:
            return conn.execute(_FRAMES_SQL).fetchall(), conn.execute(_METHOD_SQL).fetchall()
        finally:
            conn.close()


def get_labelling(db: Path | None = None) -> dict[str, Any]:  # noqa: PLR0911, PLR0912
    db = Path(db) if db else default_db_path()
    if not db.is_file():
        return _empty("missing")
    try:
        frame_rows, method_rows = _read(db)
    except StoreLocked:
        return _empty("busy", "store is being written; retry shortly")
    except duckdb.Error as e:
        msg = str(e)
        if "frame_label_state" in msg and ("does not exist" in msg or "not found" in msg.lower()):
            return _empty("empty")
        state = "busy" if "lock" in msg.lower() else "error"
        return _empty(state, msg.splitlines()[0][:300])
    except OSError as e:
        return _empty("error", str(e)[:300])
    if not frame_rows:
        return _empty("empty")

    out = _empty("ok")
    totals = out["totals"]
    by_split: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    clips = []
    for game_id, frames, ball, player, full in frame_rows:
        row = {
            "clip": game_id, "frames": frames, "ball_labelled": ball, "player_labelled": player,
            "fully_labelled": full, "partial": frames - full, "split": _split_of(game_id),
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
    return out
