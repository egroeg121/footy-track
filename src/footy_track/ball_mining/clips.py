"""Footage-time spans of clips (for the +-10 s eval exclusion) and verdict-file readers.

``<footy_data>/<match>/split_video_broadcast_frames/segments.json`` lists every broadcast segment of a match with
``start_s`` / ``end_s`` in *match* footage time. Consecutive ``_segNNN`` clips are not contiguous, so time is the
only honest way to say "within 10 s of an eval clip".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from footy_track.clip_id import canonical_clip_id

SEGMENTS_DIR = "split_video_broadcast_frames"
_NUM_RE = re.compile(r"^(?P<stem>.+_(?:seg|part))(?P<n>\d+)$")


def load_clip_spans(root: str | Path) -> dict[str, tuple[float, float]]:
    """``{canonical clip id: (start_s, end_s)}`` from every ``<root>/<match>/split_video_broadcast_frames/segments.json``."""
    out: dict[str, tuple[float, float]] = {}
    for p in sorted(Path(root).glob(f"*/{SEGMENTS_DIR}/segments.json")):
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        for seg in doc.get("segments") or []:
            f = seg.get("file")
            if f and "start_s" in seg and "end_s" in seg:
                out[canonical_clip_id(f)] = (float(seg["start_s"]), float(seg["end_s"]))
    return out


def neighbour_clips(clip: str) -> list[str]:
    """Canonical ids of the segments numbered immediately before / after ``clip`` (used as the conservative
    stand-in for "+-10 s" when a clip's footage span is unknown)."""
    m = _NUM_RE.match(canonical_clip_id(clip))
    if not m:
        return []
    n, width = int(m["n"]), len(m["n"])
    return [f"{m['stem']}{k:0{width}d}" for k in (n - 1, n + 1) if k >= 0]


def read_live_verdicts(marks_dir: str | Path) -> list[dict]:
    """Live Ball Check verdict records of ``<marks>/ball_checks/*.jsonl`` (full record, ``clip`` canonicalised).

    Per ``(source, frame_index, box_index)`` the last line wins and a ``verdict: null`` line is a tombstone
    (undo) that retracts it, exactly as ``ball_check._read_verdicts``. A tombstoned card is not returned."""
    out: list[dict] = []
    vdir = Path(marks_dir) / "ball_checks"
    if not vdir.is_dir():
        return out
    for path in sorted(vdir.glob("*.jsonl")):
        live: dict[tuple, dict] = {}
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for raw in lines:
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
                key = (
                    rec.get("source") or "sidecar",
                    int(rec["frame_index"]),
                    int(rec["box_index"]),
                )
                clip = rec.get("clip") or path.stem
            except (ValueError, KeyError, TypeError):
                continue
            if rec.get("verdict") is None:
                live.pop(key, None)
            else:
                live[key] = {**rec, "clip": canonical_clip_id(clip)}
        out.extend(live.values())
    return out


def read_verdict_frames(marks_dir: str | Path) -> set[tuple[str, int]]:
    """``{(canonical clip id, frame)}`` that hold a *live* Ball Check verdict in ``<marks>/ball_checks/*.jsonl``
    (see :func:`read_live_verdicts`). A tombstoned card does not count as labelled."""
    return {(r["clip"], int(r["frame_index"])) for r in read_live_verdicts(marks_dir)}
