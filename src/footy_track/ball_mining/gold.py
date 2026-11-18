"""Gold (bucket G) sources: boxes whose right answer is already known.

* ``gt_shifted``: a tier-1 human ball box. The card shows it moved 15-30 px, so the correct verdict is ``box_off``
  (or ``not_ball``); a labeller who accepts it is over-accepting.
* ``known_not_ball``: a box a human already judged ``not_ball`` (a spare ball, a sock). The correct verdict is
  ``not_ball``.

Both are read from the marks directory the labeller already writes: ``<marks>/<clip>.jsonl`` GT sidecars (rows tagged
``labeller`` / ``human_checked`` with a ball tag) and ``<marks>/ball_checks/<clip>.jsonl`` verdicts (live ``not_ball``
with a stored ``bbox``). The result is a frame in the same shape as the pool plus ``kind``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from footy_track.ball_mining.schema import BALL_TAGS
from footy_track.clip_id import canonical_clip_id

HUMAN_TAGS = ("labeller", "human_checked")
GOLD_COLUMNS = ("match", "clip", "frame", "fps", "x", "y", "w", "h", "kind")
GT_SHIFTED, KNOWN_NOT_BALL = "gt_shifted", "known_not_ball"


def _box(bb: object) -> tuple[float, float, float, float] | None:
    """(cx, cy, w, h) normalised from a ``{x,y,w,h}`` dict (top-left) or ``[x,y,w,h]`` list."""
    try:
        if isinstance(bb, dict):
            x, y, w, h = (float(bb[k]) for k in "xywh")
        elif isinstance(bb, (list, tuple)) and len(bb) == 4:
            x, y, w, h = (float(v) for v in bb)
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x + w / 2, y + h / 2, w, h


def load_gold(marks_dir: str | Path, *, match_of, fps: float = 25.0) -> pd.DataFrame:  # noqa: PLR0912
    """Gold candidates from a marks dir. ``match_of(clip) -> match key``."""
    marks = Path(marks_dir)
    rows: list[dict] = []
    for path in sorted(marks.glob("*.jsonl")):
        clip = path.stem
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            tags = r.get("tags") or []
            if not (
                any(t in BALL_TAGS for t in tags) and any(t in HUMAN_TAGS for t in tags)
            ):
                continue
            b = _box(r.get("bbox"))
            if b is None or "frame_index" not in r:
                continue
            rows.append(
                {
                    "clip": clip,
                    "frame": int(r["frame_index"]),
                    "x": b[0],
                    "y": b[1],
                    "w": b[2],
                    "h": b[3],
                    "kind": GT_SHIFTED,
                }
            )
    vdir = marks / "ball_checks"
    for path in sorted(vdir.glob("*.jsonl")) if vdir.is_dir() else []:
        live: dict[tuple, dict] = {}
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                key = (
                    r.get("source") or "sidecar",
                    int(r["frame_index"]),
                    int(r["box_index"]),
                )
            except (ValueError, KeyError, TypeError):
                continue
            if r.get("verdict") is None:
                live.pop(key, None)
            else:
                live[key] = r
        for (_, frame, _), r in live.items():
            b = _box(r.get("bbox"))
            if r.get("verdict") == "not_ball" and b is not None:
                rows.append(
                    {
                        "clip": r.get("clip") or path.stem,
                        "frame": frame,
                        "x": b[0],
                        "y": b[1],
                        "w": b[2],
                        "h": b[3],
                        "kind": KNOWN_NOT_BALL,
                    }
                )
    df = pd.DataFrame(rows, columns=["clip", "frame", "x", "y", "w", "h", "kind"])
    if df.empty:
        return pd.DataFrame(columns=list(GOLD_COLUMNS))
    df["match"] = [match_of(c) for c in df["clip"]]
    df["fps"] = float(fps)
    df["clip"] = df["clip"].map(lambda c: c)  # keep the stem the labeller knows
    df["_cid"] = df["clip"].map(canonical_clip_id)
    df = df.drop_duplicates(["_cid", "frame", "kind"]).drop(columns="_cid")
    return df[list(GOLD_COLUMNS)].reset_index(drop=True)
