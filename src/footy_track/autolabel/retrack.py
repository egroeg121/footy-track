"""Re-tracking inside the review: constraints from verdicts, follow-up cards, labels.

Pure logic (no web framework, no GPU) shared by the Ball Check server
(``labeller/retrack_service.py``) and the ingest step (``scripts/autolabel_loop.py``), so
the track the reviewer saw and the labels written at ingest come from the same function
of (detector candidate cache, verdicts).

Each verdict on a list card becomes a constraint of the clip's whole-clip constrained
Viterbi tracker (:mod:`footy_track.ball_trackers.constrained`): ``ball`` / ``box_off``
anchor the path at the card's box, ``corrected`` at the dragged box, ``not_ball`` forbids
the card's box in that frame. ``unsure`` and a "no ball" answer on a tap-to-place card
constrain nothing. Evidence and the simulation: ``docs/ball-tracking-eval.md`` (re-tracking
after each verdict) and ``docs/design/claude_autolabel_loop.md`` section 12.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from footy_track.ball_eval.detection_replay import CandidateCache
from footy_track.ball_trackers.constrained import ConstrainedViterbiTracker, Solution
from footy_track.ball_trackers.detection_tracker import DETECTED, BallPoint
from footy_track.labeller.constants import card_box_index

ANCHOR_RADIUS_PX = 25.0  # "ball": candidates within this of the box survive
FORBID_TOL_PX = 3.0  # "not a ball": candidates within this of the box are removed
PRESENT = ("ball", "box_off")  # anchor at the card's own box
#: A follow-up card is only worth a tap while the path is unsure about the frame.
UNSURE_MARGIN = 3.0
#: Two cards of one clip closer than this many frames are the same question.
DEDUPE_FRAMES = 2
CHANGED_BONUS = 2.0  # margin discount of a frame whose answer the last re-track changed
TAG_RETRACK = "retrack"  # marker tag on every line this module makes


@dataclass(frozen=True)
class Constraint:
    kind: str  # "anchor" | "forbid"
    frame: int
    x: float  # pixels of the native frame
    y: float


def _centre(bbox: dict, size: Sequence[int]) -> tuple[float, float]:
    return (bbox["x"] + bbox["w"] / 2) * size[0], (bbox["y"] + bbox["h"] / 2) * size[1]


def card_constraint(card: dict, rec: dict) -> Constraint | None:
    """The constraint a verdict record puts on its card's clip (None: none)."""
    verdict = rec.get("verdict")
    size = card["frame_size"]
    frame = int(card["frame_index"])
    cb = rec.get("corrected_bbox")
    if verdict == "corrected" and isinstance(cb, dict):
        return Constraint("anchor", frame, *_centre(cb, size))
    box = card.get("bbox")
    if not isinstance(box, dict):
        return None  # tap-to-place card without a tap
    if verdict in PRESENT:
        return Constraint("anchor", frame, *_centre(box, size))
    if verdict == "not_ball":
        return Constraint("forbid", frame, *_centre(box, size))
    return None


def card_id_key(card: dict) -> tuple[str, int, int]:
    return card["clip"], int(card["frame_index"]), card_box_index(card["id"])


def collect_constraints(
    cards: Iterable[dict], verdicts: dict[tuple, dict]
) -> dict[str, list[Constraint]]:
    """``{run_clip: constraints}`` from the live verdicts (``("list", clip stem, frame,
    box_index)`` -> record), oldest first so a later anchor on a frame wins."""
    found: list[tuple[float, str, Constraint]] = []
    for card in cards:
        rec = verdicts.get(("list", *card_id_key(card)))
        if rec is None:
            continue
        c = card_constraint(card, rec)
        if c is not None:
            found.append((float(rec.get("ts") or 0), card["run_clip"], c))
    out: dict[str, list[Constraint]] = {}
    for _, clip, c in sorted(found, key=lambda t: t[0]):
        out.setdefault(clip, []).append(c)
    return out


def solve(
    cache: CandidateCache, constraints: Sequence[Constraint]
) -> tuple[Solution, ConstrainedViterbiTracker]:
    """One whole-clip constrained solve (a fresh tracker: constraints may have been undone)."""
    trk = ConstrainedViterbiTracker(cache.frame_size)
    for f, cands in cache.frames:
        trk.update(f, cands)
    for c in constraints:
        if c.kind == "anchor":
            trk.anchor(c.frame, c.x, c.y, ANCHOR_RADIUS_PX)
        else:
            trk.forbid(c.frame, c.x, c.y, FORBID_TOL_PX)
    return trk.solve(), trk


def point_centre_px(p: BallPoint, size: Sequence[int]) -> tuple[float, float] | None:
    if p.x is None:
        return None
    return (p.x + p.w / 2) * size[0], (p.y + p.h / 2) * size[1]


def differs(a: BallPoint, b: BallPoint) -> bool:
    if a.source != b.source:
        return True
    if a.x is None:
        return False
    return abs(a.x - b.x) > 1e-6 or abs(a.y - b.y) > 1e-6


def changed_frames(before: dict[int, BallPoint] | None, after: Solution) -> set[int]:
    """Frames whose answer differs from the previous solve (none known: none)."""
    if before is None:
        return set()
    return {
        p.frame_index
        for p in after.points
        if p.frame_index in before and differs(before[p.frame_index], p)
    }


def followup_frames(
    sol: Solution,
    changed: set[int],
    blocked: Iterable[int],
    *,
    limit: int,
    margin_max: float = UNSURE_MARGIN,
) -> list[int]:
    """The frames of one clip to ask next, most uncertain first.

    Candidates: frames where the path sits on a detected candidate (a box to show) and
    whose margin is below ``margin_max``; frames the last re-track changed are first
    (``CHANGED_BONUS``). Anchored frames, frames within ``DEDUPE_FRAMES`` of a ``blocked``
    frame (asked, queued or already chosen) are skipped.
    """
    taken = sorted(blocked)
    pts = {p.frame_index: p for p in sol.points}

    def near(f: int) -> bool:
        return any(abs(f - b) <= DEDUPE_FRAMES for b in taken)

    ranked = sorted(
        (
            (m - (CHANGED_BONUS if f in changed else 0.0), f)
            for f, m in sol.margins.items()
            if f not in sol.anchored
            and math.isfinite(m)
            and m < margin_max
            and pts[f].source == DETECTED
        )
    )
    out: list[int] = []
    for _, f in ranked:
        if len(out) >= limit:
            break
        if near(f):
            continue
        out.append(f)
        taken.append(f)
    return out


def point_box(p: BallPoint, size: Sequence[int]) -> dict:
    """The point's box as a card box (normalised, at least 10 px, inside the frame)."""
    fw, fh = size
    cx, cy = (p.x + p.w / 2) * fw, (p.y + p.h / 2) * fh
    w, h = max(p.w * fw, 10.0), max(p.h * fh, 10.0)
    x = max(0.0, min(1.0, (cx - w / 2) / fw))
    y = max(0.0, min(1.0, (cy - h / 2) / fh))
    return {
        "x": round(x, 5),
        "y": round(y, 5),
        "w": round(min(w / fw, 1 - x), 5),
        "h": round(min(h / fh, 1 - y), 5),
    }


def dynamic_card(
    round_id: str, run_clip: str, stem: str, size: Sequence[int], p: BallPoint
) -> dict:
    """An id-keyed blind card for the path's box at ``p`` (same shape as a list card)."""
    box = point_box(p, size)
    cx, cy = point_centre_px(p, size) or (0.0, 0.0)
    cid = hashlib.sha1(
        f"{round_id}|retrack|{run_clip}|{p.frame_index}|{round(cx)}|{round(cy)}".encode()
    ).hexdigest()[:16]
    return {
        "id": cid,
        "clip": stem,
        "run_clip": run_clip,
        "frame_index": p.frame_index,
        "frame_size": list(size),
        "bbox": box,
        "label": "in_play_ball",
        "provenance": "list",
        "confidence": None,
        "dynamic": True,
    }


def box_matches(
    card: dict, p: BallPoint, size: Sequence[int], tol_px: float = 12.0
) -> bool:
    """The card's box sits on the path's point (same object, within ``tol_px``)."""
    if p.x is None or not isinstance(card.get("bbox"), dict):
        return False
    a = _centre(card["bbox"], size)
    b = point_centre_px(p, size)
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol_px


def arrange(
    base: list[dict], pinned: list[dict], *, every: int = 2, base_first: bool = False
) -> list[dict]:
    """``every`` pinned cards, then one card of the base order, repeated.

    The base order (the round's list: priority classes, blind audits spread through it) is
    never reordered, only delayed. ``base_first`` puts a base card first: the server sets
    it once ``every`` pinned cards were judged in a row, so a reviewer who always takes
    the top card still meets a base card (audits, controls and random frames among them)
    after every ``every`` follow-ups, however many follow-ups the re-tracks keep making. A
    pinned card already in the base order is moved (not shown twice).
    """
    pin_ids = {c["id"] for c in pinned}
    rest = [c for c in base if c["id"] not in pin_ids]
    out: list[dict] = []
    p = b = 0
    if base_first and rest:
        out.append(rest[0])
        b = 1
    while p < len(pinned) or b < len(rest):
        for _ in range(every):
            if p < len(pinned):
                out.append(pinned[p])
                p += 1
        if b < len(rest):
            out.append(rest[b])
            b += 1
    return out


def chain_label_lines(
    sol: Solution, size: Sequence[int], label: str, tags: Sequence[str]
) -> list[dict]:
    """Sidecar lines (tier 3 by the caller's ``tags``) for the path's detected boxes on
    chains a person anchored.

    A chain is the run of consecutive detections the tracker links into one object
    (``track_id``); one a person confirmed at some frame (an anchor) is that ball followed
    through the clip, so its other detected boxes are machine labels worth keeping. A
    chain nobody touched is just the detector's own tracker: nothing is written for it.
    Anchored frames themselves are labelled by the card rules (tier 1 / 2), and
    interpolated (invented) positions never.
    """
    ids = {
        p.track_id
        for p in sol.points
        if p.frame_index in sol.anchored and p.track_id is not None
    }
    lines = []
    for p in sol.points:
        if (
            p.source == DETECTED
            and p.track_id in ids
            and p.frame_index not in sol.anchored
        ):
            b = point_box(p, size)
            lines.append(
                {
                    "frame_index": p.frame_index,
                    "bbox": b,
                    "center": {"x": b["x"] + b["w"] / 2, "y": b["y"] + b["h"] / 2},
                    "tags": [label, *tags],
                }
            )
    return lines


__all__ = [
    "Constraint",
    "arrange",
    "card_constraint",
    "chain_label_lines",
    "collect_constraints",
    "dynamic_card",
    "followup_frames",
    "solve",
]
