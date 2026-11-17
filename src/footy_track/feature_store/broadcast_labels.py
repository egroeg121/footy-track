"""Broadcast-review verdicts -> the ``broadcast_label`` table.

The broadcast review (``labeller/broadcast_review.py``) keeps one file per match,
``<marks dir>/broadcast_review/<video stem>.json``::

    {"frames": {"12345": {"verdict": "yes"|"no"|"unsure", "ts": "...", "p_yes": 0.43,
                          "pred": "No", "reason": "edge", "ts_s": 513.2}}}

``p_yes`` / ``pred`` are the classifier's call when the frame was queued, so they are the
*stored predictions* the ``broadcast`` eval adapter scores. Re-ingest replaces a file's rows
(a verdict cleared in the UI disappears), so the table always mirrors the files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from footy_track.feature_store.store import FeatureStore

VERDICTS = ("yes", "no", "unsure")


@dataclass
class BroadcastImportReport:
    files: int = 0
    labels: int = 0
    by_verdict: dict[str, int] = field(default_factory=dict)
    by_file: dict[str, int] = field(default_factory=dict)
    skipped: int = 0


def _f(v: object) -> float | None:
    try:
        return None if v is None else float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_review_file(path: Path) -> list[tuple]:
    """Rows ``(video_stem, frame, verdict, p_yes, pred, reason, ts_s, labelled_at, file)``."""
    doc = json.loads(path.read_text())
    rows: list[tuple] = []
    for k, v in ((doc or {}).get("frames") or {}).items():
        if not (isinstance(v, dict) and str(k).isdigit() and v.get("verdict") in VERDICTS):
            continue
        rows.append(
            (
                path.stem,
                int(k),
                v["verdict"],
                _f(v.get("p_yes")),
                v.get("pred"),
                v.get("reason"),
                _f(v.get("ts_s")),
                v.get("ts"),
                path.name,
            )
        )
    return rows


def import_broadcast_reviews(
    store: FeatureStore, review_dirs: list[Path]
) -> BroadcastImportReport:
    """Replace each review file's rows in ``broadcast_label``. ``review_dirs`` are the
    ``<marks dir>/broadcast_review`` directories."""
    rep = BroadcastImportReport()
    seen: set[str] = set()
    for d in review_dirs:
        for path in sorted(d.glob("*.json")) if d.is_dir() else []:
            if path.stem in seen:  # same match in two marks dirs: first wins
                rep.skipped += 1
                continue
            seen.add(path.stem)
            try:
                rows = parse_review_file(path)
            except (OSError, ValueError):
                rep.skipped += 1
                continue
            with store.transaction():
                store._conn.execute(  # noqa: SLF001 - same package
                    "DELETE FROM broadcast_label WHERE video_stem = ?", [path.stem]
                )
                if rows:
                    store._conn.executemany(  # noqa: SLF001
                        "INSERT INTO broadcast_label VALUES (?,?,?,?,?,?,?,?,?)", rows
                    )
            rep.files += 1
            rep.labels += len(rows)
            rep.by_file[path.stem] = len(rows)
            for r in rows:
                rep.by_verdict[r[2]] = rep.by_verdict.get(r[2], 0) + 1
    return rep
