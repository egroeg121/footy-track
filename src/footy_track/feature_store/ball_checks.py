"""Import Ball Check verdicts into the feature store (T-02).

The Ball Check UI (``labeller/ball_check.py``) appends one JSON line per card
decision to ``<gt-dir>/ball_checks/<clip>.jsonl``::

    {clip, frame_index, box_index, source, confidence, verdict, bbox, label,
     provenance, corrected_bbox, ts}

``source`` is ``"sidecar"`` (a box already in the clip's GT sidecar; records
written before candidates existed have no ``source`` and mean ``sidecar``) or
``"cand"`` (a raw detector candidate that is usually NOT in any sidecar).
An undo appends a tombstone (``verdict: null``); the latest line per
``(source, frame_index, box_index)`` wins and a tombstone retracts it.

What this module does with the latest live verdict per box:

* every one is recorded in ``detection_review`` (so hard negatives on
  candidate boxes that were never ingested are not lost);
* it is *matched* to a stored ``detection`` on the same ``(game, frame)`` by
  IoU of its bbox (``corrected_bbox`` is tried too). ``detection_id`` is
  positional per (frame, source) and ``box_index`` counts every box on the
  frame incl. other sources/classes, so index equality is not reliable; boxes
  are. Only ball-class detections are candidates;
* ``ball`` / ``corrected`` set ``detection.reviewed = TRUE`` on the match;
* ``not_ball`` is stored as a rejected negative (exporters drop it from
  positives); a ``cand`` not_ball never rejects a ``hand_label`` row (a human
  drew that box, the verdict was about a different detector box) -- counted
  as a conflict instead;
* ``unsure`` / ``box_off`` change nothing in ``detection``.

Idempotent: a clip's ``detection_review`` rows are replaced on each run.
Note ``ingest_gt`` re-upserts ``reviewed`` from the sidecar, so run this
*after* ingesting the sidecars.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from footy_track.feature_store.store import FeatureStore

LOGGER = logging.getLogger(__name__)

BALL_LABELS = ("ball", "in_play_ball", "out_of_play_ball")
#: Minimum IoU for a verdict box to count as "the same box" as a detection.
MATCH_IOU = 0.5
REVIEWED_VERDICTS = ("ball", "corrected")


@dataclass
class BallCheckReport:
    clips: int = 0
    verdicts: int = 0  # live (non-tombstoned) verdicts
    by_verdict: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    matched: int = 0
    unmatched: int = 0
    reviewed_marked: int = 0
    rejected: int = 0  # not_ball matched to a detection
    hand_label_conflicts: int = 0
    unmatched_keys: list[tuple] = field(default_factory=list)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _xywh(b: object) -> tuple[float, float, float, float] | None:
    if isinstance(b, dict):
        try:
            return (float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]))
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(b, (list, tuple)) and len(b) == 4:
        try:
            return tuple(float(v) for v in b)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def read_latest_verdicts(path: Path) -> dict[tuple[str, int, int], dict]:
    """Latest live verdict per ``(source, frame_index, box_index)``; tombstones
    retract. Same resolution rule as ``ball_check._read_verdicts``."""
    out: dict[tuple[str, int, int], dict] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            key = (
                rec.get("source") or "sidecar",
                int(rec["frame_index"]),
                int(rec["box_index"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if rec.get("verdict") is None:
            out.pop(key, None)
        else:
            out[key] = rec
    return out


def _import_clip(
    store: FeatureStore, clip: str, verdicts: dict, report: BallCheckReport, dry_run: bool
) -> None:
    conn = store._conn  # noqa: SLF001 - same package; needs UPDATE/DELETE
    in_clause = ", ".join("?" for _ in BALL_LABELS)
    dets_by_frame: dict[int, list[tuple]] = defaultdict(list)
    for row in conn.execute(
        f"SELECT frame_index, source, run_id, detection_id, bbox_x, bbox_y, bbox_w, bbox_h "
        f"FROM detection WHERE game_id = ? AND lower(label) IN ({in_clause})",
        [clip, *BALL_LABELS],
    ).fetchall():
        dets_by_frame[row[0]].append(row[1:])

    review_rows: list[list] = []
    reviewed_keys: set[tuple] = set()
    for (vsource, frame_index, box_index), rec in sorted(verdicts.items()):
        verdict = rec["verdict"]
        report.verdicts += 1
        report.by_verdict[verdict] += 1
        bbox = _xywh(rec.get("bbox"))
        probes = [b for b in (_xywh(rec.get("corrected_bbox")), bbox) if b is not None]

        best: tuple[float, tuple] | None = None
        for det in dets_by_frame.get(frame_index, ()):
            dbox = (det[3], det[4], det[5], det[6])
            iou = max((_iou(p, dbox) for p in probes), default=0.0)
            if iou >= MATCH_IOU and (best is None or iou > best[0]):
                best = (iou, det)

        m_source = m_run = m_id = m_iou = None
        if best is None:
            report.unmatched += 1
            report.unmatched_keys.append((clip, vsource, frame_index, box_index, verdict))
            LOGGER.info(
                "ball_check verdict %s has no matching detection: %s f%s %s#%s",
                verdict, clip, frame_index, vsource, box_index,
            )
        else:
            m_iou, det = best
            m_source, m_run, m_id = det[0], det[1], det[2]
            if verdict == "not_ball" and vsource == "cand" and m_source == "hand_label":
                # Verdict was on a detector candidate; the matching box is
                # human-drawn GT. Don't let one demote the other.
                report.hand_label_conflicts += 1
                m_source = m_run = m_id = None
                m_iou = None
                report.unmatched += 1
            else:
                report.matched += 1
                if verdict == "not_ball":
                    report.rejected += 1
                elif verdict in REVIEWED_VERDICTS:
                    reviewed_keys.add((m_source, m_run, m_id, frame_index))

        bx = bbox or (None, None, None, None)
        review_rows.append(
            [clip, vsource, frame_index, box_index, verdict, rec.get("ts"),
             rec.get("label"), *bx, m_source, m_run, m_id, m_iou]
        )

    report.reviewed_marked += len(reviewed_keys)
    if dry_run:
        return
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM detection_review WHERE game_id = ?", [clip])
        if review_rows:
            conn.executemany(
                "INSERT INTO detection_review VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                review_rows,
            )
        for src, run, did, fi in reviewed_keys:
            conn.execute(
                "UPDATE detection SET reviewed = TRUE WHERE game_id = ? AND frame_index = ? "
                "AND source = ? AND run_id = ? AND detection_id = ?",
                [clip, fi, src, run, did],
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def import_ball_checks(
    store: FeatureStore,
    ball_checks_dir: Path,
    *,
    clip: str | None = None,
    dry_run: bool = False,
) -> BallCheckReport:
    """Apply the latest Ball Check verdict per box to *store*. See module doc."""
    report = BallCheckReport()
    if not ball_checks_dir.is_dir():
        return report
    for path in sorted(ball_checks_dir.glob("*.jsonl")):
        if clip and path.stem != clip:
            continue
        verdicts = read_latest_verdicts(path)
        if not verdicts and dry_run:
            continue
        report.clips += 1
        _import_clip(store, path.stem, verdicts, report, dry_run)
    return report
