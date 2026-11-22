"""Wrong-anchor controls (design ``docs/design/ball_anchor_iteration.md`` section 5 and task 11).

A confirmed anchor is trusted as ground truth, so one wrong tap can spread through every card the search derives from it.
This module holds the three controls that catch that. The hop cap (``anchor_search.HOP_CAP`` = 3, decay 0.8) is the
fourth and stays where it is; there is no bulk "accept the run" card.

1. **S-gold control cards.** Known-answer frames (the gold pool of :mod:`footy_track.ball_mining.gold`) are mixed into the
   search card stream at about :data:`CONTROL_RATE` of the S cards. A control is a normal ``ball_check`` item of bucket
   ``S``: same keys, same ``search`` block (copied from the S card it is slotted next to, so it shares that card's
   ``anchor_id`` / ``window_id`` / rank), same ``sources``. Only two hidden markers differ, ``sampling.gold`` (the expected
   answers) and ``search.control``, and neither is shown by any page. Nothing in the UI tells a labeller which card is a
   control. The verdicts ride with the labels like every other card's, so the **agreement rate** (per window and per
   job) is computed from the job results alone (:func:`agreement`).
2. **Suspect anchors.** An anchor is marked suspect when (a) a control card slotted next to it is answered against its
   expected verdict, or (b) a later human ``not_ball`` / ``box_off`` lands on a search card adjacent to it
   (``hop <= ADJACENT_HOP``). The marks live in an append-only ``<marks>/suspects/<job>.jsonl`` (:func:`record_suspects`,
   dedupe on ``(anchor, reason, evidence frame)``: idempotent). ``anchor_search.run_search(suspects=...)`` releases them,
   so the window's search is regenerated without them as a new immutable ``vNNN`` file (the input digest changes), and
   :func:`suspect_impact` lists the cards / frames that had been derived from them.
3. **Calibration firewall.** Control cards are bucket ``S``; ``refit.exclude_search`` already keeps every S verdict out of
   Platt and learn-then-test, controls included, and ``refit.hit_rates`` leaves controls out of the S hit rate.

Split guard: a control is only ever drawn from a clip of the job's own split, never from the job's selected frame set or a
frame already carded in the job, so no frame is measured twice and no label crosses a split. No imports from
``anchor_search`` (it imports this module).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from footy_track import splits
from footy_track.ball_mining.gold import GT_SHIFTED, KNOWN_NOT_BALL
from footy_track.clip_id import canonical_clip_id

SEARCH_BUCKET = "S"  # == refit.SEARCH_BUCKET (refit imports this module, so it cannot be imported from there; a test pins it)
CONTROL_RATE = 0.05  # share of the S cards of a job that are controls (cumulative, deterministic)
SHIFT_PX = (15.0, 30.0)  # same shift range as the G bucket's shifted gold
FRAME_WH = (1920, 1080)
ADJACENT_HOP = 1  # a rejection on a card this close to its anchor contradicts the anchor
GT_EXACT = "gt_exact"  # a human ball box shown as it is: the right answer is "ball"
#: Verdicts that count as the right answer, per control kind (``unsure`` is neither right nor wrong).
EXPECTED: dict[str, list[str]] = {
    GT_EXACT: ["ball", "corrected"],
    GT_SHIFTED: ["box_off", "not_ball", "corrected"],
    KNOWN_NOT_BALL: ["not_ball", "box_off"],
}
DECIDED = ("ball", "corrected", "not_ball", "box_off")
REJECTS = ("not_ball", "box_off")
REASON_CONTROL = "control_disagree"
REASON_CONTRADICTED = "contradicted"

Key = tuple[str, int]


def _h(*parts: object) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def is_control_item(item: dict) -> bool:
    """A card the search appended as a control (hidden marker; also true of the label's ``sampling`` echo)."""
    return bool((item.get("search") or {}).get("control")) or bool((item.get("sampling") or {}).get("gold") and (item.get("sampling") or {}).get("bucket") == SEARCH_BUCKET)


# ----------------------------------------------------------------------------------------------------------
# Selection and injection
# ----------------------------------------------------------------------------------------------------------
def n_controls_due(n_search: int, n_controls: int, rate: float = CONTROL_RATE) -> int:
    """Controls to add so that ``controls ~ rate * search cards`` cumulatively: ``floor(rate * n_search) - n_controls`` (>= 0).
    A retried run sees its own controls in ``n_controls`` and adds 0."""
    return max(0, math.floor(rate * n_search + 1e-9) - n_controls)


def _records(gold) -> list[dict]:
    if gold is None:
        return []
    if hasattr(gold, "to_dict"):
        return list(gold.to_dict("records"))
    return [dict(r) for r in gold]


def pick_controls(
    gold,
    *,
    job_id: str,
    split: str | None,
    taken: Iterable[Key],
    n: int,
    frame_wh: tuple[int, int] = FRAME_WH,
    playable: Callable[[str], str | None] | None = None,
) -> list[dict]:
    """``n`` control specs ``{clip, frame_index, bbox, kind, expected, shift_px}`` drawn deterministically (hash order) from the
    ``gold`` rows (``gold.load_gold`` frame or dicts with ``clip, frame, x, y, w, h, kind``; ``x, y`` are centres).

    Only rows of the job's split, whose frame is not in ``taken`` (the job's cards, its selected set, controls already used),
    one per frame, and whose clip ``playable`` resolves (it returns the clip name the job's dataset serves, or None to skip).
    A ``gt_shifted`` row becomes either an exact human box (kind ``gt_exact``) or a 15-30 px shifted one, by hash."""
    if n <= 0 or not split:
        return []
    taken_k = {(canonical_clip_id(c), int(f)) for c, f in taken}
    rows = sorted(_records(gold), key=lambda r: _h(job_id, canonical_clip_id(r["clip"]), r["frame"], r["kind"]))
    out: list[dict] = []
    for r in rows:
        if len(out) >= n:
            break
        cid, frame = canonical_clip_id(r["clip"]), int(r["frame"])
        if (cid, frame) in taken_k or str(splits.split_of(splits.match_key_of(cid), cid)) != split:
            continue
        name = playable(r["clip"]) if playable else r["clip"]
        if not name:
            continue
        cx, cy, w, h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
        kind, shift = r["kind"], None
        if kind == GT_SHIFTED:
            hv = _h("variant", job_id, cid, frame)
            if hv % 2 == 0:
                kind = GT_EXACT
            else:
                d = SHIFT_PX[0] + (SHIFT_PX[1] - SHIFT_PX[0]) * ((hv >> 1) % 1000) / 1000.0
                th = 2 * math.pi * ((hv >> 11) % 1000) / 1000.0
                cx = min(max(cx + d * math.cos(th) / frame_wh[0], 0.0), 1.0)
                cy = min(max(cy + d * math.sin(th) / frame_wh[1], 0.0), 1.0)
                shift = round(d, 2)
        if kind not in EXPECTED:
            continue
        taken_k.add((cid, frame))
        out.append({"clip": name, "frame_index": frame, "kind": kind, "shift_px": shift, "expected": list(EXPECTED[kind]),
                    "bbox": {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h}})  # fmt: skip
    return out


def mark_item(item: dict, ctl: dict) -> dict:
    """Add the two hidden control markers to an item built like a normal S card (``anchor_search.item_from_card`` on a card
    spec cloned from its host S card). Returns the item."""
    item["sampling"]["gold"] = {"kind": ctl["kind"], "shift_px": ctl.get("shift_px"), "expected": list(ctl["expected"])}
    item["search"]["control"] = True
    return item


# ----------------------------------------------------------------------------------------------------------
# Agreement
# ----------------------------------------------------------------------------------------------------------
def control_rows(rows: Iterable[dict]) -> list[dict]:
    return [r for r in rows if r.get("control")]


def _verdict_ok(row: dict) -> bool | None:
    v = row.get("verdict")
    if v not in DECIDED:
        return None  # unsure / no verdict: neither agrees nor disagrees
    exp = row.get("expected") or EXPECTED.get(row.get("gold_kind") or "", [])
    return v in exp


def agreement(rows: Iterable[dict]) -> dict:
    """Agreement of the control verdicts with their expected answers, overall and per window.

    ``rows`` are ``refit.rows_from_results`` rows. ``rate`` = agree / (agree + disagree), None with no decided control."""
    tot = {"n": 0, "agree": 0, "disagree": 0, "unsure": 0}
    by_w: dict[str, dict] = defaultdict(lambda: {"n": 0, "agree": 0, "disagree": 0, "unsure": 0})
    bad: list[dict] = []
    for r in control_rows(rows):
        ok = _verdict_ok(r)
        k = "unsure" if ok is None else ("agree" if ok else "disagree")
        w = r.get("window_id") or "?"
        for d in (tot, by_w[w]):
            d["n"] += 1
            d[k] += 1
        if ok is False:
            bad.append({"clip": r.get("clip"), "frame": r.get("frame"), "kind": r.get("gold_kind"), "verdict": r.get("verdict"),
                        "expected": r.get("expected"), "window_id": r.get("window_id"), "anchor_id": r.get("anchor_id")})  # fmt: skip

    def rate(d: dict) -> float | None:
        dec = d["agree"] + d["disagree"]
        return round(d["agree"] / dec, 4) if dec else None

    return {**tot, "rate": rate(tot), "by_window": {w: {**d, "rate": rate(d)} for w, d in sorted(by_w.items())}, "disagreements": bad}


# ----------------------------------------------------------------------------------------------------------
# Suspect anchors
# ----------------------------------------------------------------------------------------------------------
def detect_suspects(rows: Iterable[dict]) -> list[dict]:
    """Anchors the verdicts in ``rows`` put in doubt: ``{anchor_id, reason, window_id, evidence}`` (one per anchor and reason
    and evidence frame, sorted; the same rows give the same list)."""
    out: dict[tuple, dict] = {}
    for r in rows:
        aid = r.get("anchor_id")
        if not aid or r.get("bucket") != SEARCH_BUCKET:
            continue
        ev = {"clip": r.get("clip"), "frame": r.get("frame"), "verdict": r.get("verdict")}
        if r.get("control"):
            if _verdict_ok(r) is False:
                reason, ev = REASON_CONTROL, {**ev, "kind": r.get("gold_kind")}
            else:
                continue
        elif r.get("verdict") in REJECTS and r.get("hop") is not None and int(r["hop"]) <= ADJACENT_HOP:
            reason = REASON_CONTRADICTED
        else:
            continue
        out.setdefault((aid, reason, ev["clip"], ev["frame"]), {"anchor_id": aid, "reason": reason, "window_id": r.get("window_id"), "evidence": ev})
    return [out[k] for k in sorted(out, key=lambda k: tuple(map(str, k)))]


def anchor_key(anchor_id: str, job_id: str | None = None) -> Key | None:
    """``(clip, frame)`` of an ``anchor_id`` (``<job>:<clip>:<frame>``); None for a malformed id or another job's anchor."""
    try:
        job, clip, frame = str(anchor_id).rsplit(":", 2)
        if job_id is not None and job != str(job_id):
            return None
        return canonical_clip_id(clip), int(frame)
    except (ValueError, IndexError):
        return None


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def suspects_path(marks_dir: str | Path, job_id: str) -> Path:
    return Path(marks_dir) / "suspects" / f"{_SAFE.sub('_', str(job_id))}.jsonl"


def read_suspects(marks_dir: str | Path, job_id: str) -> list[dict]:
    p = suspects_path(marks_dir, job_id)
    if not p.is_file():
        return []
    out, seen = [], set()
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
            k = _entry_key(e)
        except (ValueError, KeyError, TypeError):
            continue
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def _entry_key(e: dict) -> tuple:
    ev = e.get("evidence") or {}
    return (e["anchor_id"], e["reason"], ev.get("clip"), ev.get("frame"))


def record_suspects(marks_dir: str | Path, job_id: str, entries: Iterable[dict], *, ts: str | None = None) -> list[dict]:
    """Append the entries not already recorded (dedupe on anchor / reason / evidence frame); returns the NEW ones. Call under
    the per-job mutex (``refit.job_lock``). A second call with the same entries writes nothing."""
    have = {_entry_key(e) for e in read_suspects(marks_dir, job_id)}
    new = []
    for e in entries:
        k = _entry_key(e)
        if k not in have:
            have.add(k)
            new.append({**e, **({"ts": ts} if ts else {})})
    if new:
        p = suspects_path(marks_dir, job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write("".join(json.dumps(e, sort_keys=True) + "\n" for e in new))
    return new


def suspect_keys(entries: Iterable[dict], job_id: str | None = None) -> frozenset[Key]:
    return frozenset(k for e in entries if (k := anchor_key(e["anchor_id"], job_id)) is not None)


def suspect_impact(job_doc: dict, entries: Iterable[dict], *, prev_docs: Iterable[dict] = ()) -> dict:
    """What the suspect anchors had touched, per suspect anchor id: ``{anchor_id: {...}}``.

    * ``open_cards`` / ``answered_cards``: frames of cards in the job derived from it (``sampling.anchor_id``), answered or not
      (an answered one is a verdict that may rest on a wrong anchor; an open one is stale and can still be answered);
    * ``in_previous_search``: frames of the search files (as they were before the regeneration, ``prev_docs``) whose cards were
      derived from it, including cards never appended to the job.
    """
    ids = {e["anchor_id"] for e in entries}
    answered = {(lab.get("clip"), lab.get("frame_index")) for lab in job_doc.get("labels") or [] if lab.get("verdict") is not None}
    out: dict[str, dict] = {a: {"open_cards": [], "answered_cards": [], "in_previous_search": []} for a in sorted(ids)}
    for p in job_doc.get("items") or []:
        it = p.get("item", p)
        aid = (it.get("sampling") or {}).get("anchor_id") or (it.get("search") or {}).get("anchor_id")
        if aid in ids and not is_control_item(it):
            slot = "answered_cards" if (it["clip"], it.get("frame_index")) in answered else "open_cards"
            out[aid][slot].append({"clip": it["clip"], "frame": it.get("frame_index")})
    for d in prev_docs:
        for c in d.get("cards") or []:
            if c.get("anchor_id") in ids:
                out[c["anchor_id"]]["in_previous_search"].append({"clip": c["clip"], "frame": c["frame_index"]})
    for v in out.values():
        for k in v:
            v[k].sort(key=lambda x: (str(x["clip"]), x["frame"] if x["frame"] is not None else -1))
        v["n_cards"] = len(v["open_cards"]) + len(v["answered_cards"])
    return out
