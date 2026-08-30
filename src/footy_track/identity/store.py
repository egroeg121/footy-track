"""Append-only storage for identity labels.

**Append-only is a deliberate safety choice, not a stylistic one.**

The box labeller writes its sidecars by rewriting the whole file from in-memory
state every couple of seconds (``session.py::_do_flush``). That design destroyed
a real ground-truth file: 3,348 rows became 0 because the in-memory timeline was
empty when the flush fired. Any whole-file rewrite has that failure mode, and
identity labels are more expensive to collect than boxes — a human answering
"same or different" cannot re-derive their answer from the video the way a box
can be redrawn.

So this module only ever appends. There is no code path that truncates or
rewrites a label log. Correcting a label means appending a newer verdict for the
same pair; the reader keeps the last one. That makes the log an audit trail
(you can see that a human changed their mind, and when), and makes the whole
class of truncation bugs structurally impossible.

Each append is flushed and fsynced, so a crash loses at most the record being
written — never earlier ones.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from footy_track.identity.labels import (
    CheckedInterval,
    PairLabel,
    TrackletRef,
    TrackletReview,
    Verdict,
)

PAIRS_FILENAME = "identity_pairs.jsonl"
REVIEWS_FILENAME = "identity_reviews.jsonl"


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _ref_to_dict(t: TrackletRef) -> dict:
    return {
        "clip": t.clip,
        "track_id": t.track_id,
        "source": t.source,
        "run_id": t.run_id,
    }


def _ref_from_dict(d: dict) -> TrackletRef:
    return TrackletRef(
        clip=d["clip"],
        track_id=int(d["track_id"]),
        source=d.get("source", "bytetrack"),
        run_id=d.get("run_id"),
    )


def append_pair_label(dir_path: Path, label: PairLabel, *, ts: float | None = None) -> None:
    """Append one same/different/unknown verdict."""
    a, b = label.ordered()
    _append(
        Path(dir_path) / PAIRS_FILENAME,
        {
            "kind": "pair",
            "a": _ref_to_dict(a),
            "b": _ref_to_dict(b),
            "verdict": label.verdict.value,
            "annotator": label.annotator,
            "tier": label.tier,
            "ts": ts if ts is not None else time.time(),
        },
    )


def load_pair_labels(dir_path: Path) -> list[PairLabel]:
    """Read verdicts, keeping only the LATEST for each pair.

    Later records supersede earlier ones so a human can correct themselves
    without any file being rewritten. Order within the file is authoritative;
    the timestamp is recorded for auditing but not used for ordering, because
    clock skew should never change which label wins.
    """
    path = Path(dir_path) / PAIRS_FILENAME
    if not path.exists():
        return []
    latest: dict[tuple, PairLabel] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line must not poison the whole log
        if d.get("kind") != "pair":
            continue
        try:
            a, b = _ref_from_dict(d["a"]), _ref_from_dict(d["b"])
            verdict = Verdict(d["verdict"])
        except (KeyError, ValueError):
            continue
        lbl = PairLabel(
            a=a,
            b=b,
            verdict=verdict,
            annotator=d.get("annotator", "human"),
            tier=int(d.get("tier", 1)),
        )
        oa, ob = lbl.ordered()
        latest[(oa.key(), ob.key())] = lbl
    return list(latest.values())


def append_tracklet_review(
    dir_path: Path, review: TrackletReview, *, ts: float | None = None
) -> None:
    """Append the result of an 'is this one player throughout?' pass."""
    _append(
        Path(dir_path) / REVIEWS_FILENAME,
        {
            "kind": "review",
            "tracklet": _ref_to_dict(review.tracklet),
            "checked_intervals": [
                [iv.start_frame, iv.end_frame] for iv in review.checked_intervals
            ],
            "split_at": list(review.split_at),
            "unsure": bool(review.unsure),
            "jersey_number": review.jersey_number,
            "annotator": review.annotator,
            "ts": ts if ts is not None else time.time(),
        },
    )


def load_tracklet_reviews(dir_path: Path) -> list[TrackletReview]:
    """Read reviews, keeping the LATEST per tracklet.

    Checked intervals are NOT merged across records: a re-review replaces the
    earlier one wholesale. Accumulating them would let a tracklet drift toward
    "fully checked" through repeated partial passes, overstating human coverage
    — exactly the claim the interval model exists to prevent.
    """
    path = Path(dir_path) / REVIEWS_FILENAME
    if not path.exists():
        return []
    latest: dict[tuple, TrackletReview] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("kind") != "review":
            continue
        try:
            tracklet = _ref_from_dict(d["tracklet"])
        except (KeyError, ValueError):
            continue
        review = TrackletReview(
            tracklet=tracklet,
            checked_intervals=[
                CheckedInterval(int(s), int(e))
                for s, e in d.get("checked_intervals", [])
                if int(e) >= int(s)
            ],
            split_at=[int(f) for f in d.get("split_at", [])],
            unsure=bool(d.get("unsure", False)),
            jersey_number=(d.get("jersey_number") or None),
            annotator=d.get("annotator", "human"),
        )
        latest[tracklet.key()] = review
    return list(latest.values())
