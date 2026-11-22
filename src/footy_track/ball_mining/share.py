"""Fixed eval share of labelling time (design ``ball_anchor_iteration.md`` section 7.0, open question 9; George 2026-11-22).

Eval and train jobs run side by side. The eval family gets a FIXED share of the cards a labeller sees (phone and laptop queues)
until its panel is done, so a busy train round cannot starve it. The rule is an ordering / quota rule on card lists, nothing else:

* :data:`EVAL_SHARE` (default 0.30): in the merged queue, every prefix of ``k`` cards holds ``floor(k * EVAL_SHARE)`` eval cards
  (``ceil`` is never needed: the first eval card comes at position 4 for 30 %) for as long as eval cards are left. When the eval
  list is exhausted (its panel is done) the rest is train; when train is empty the eval cards continue alone, so no slot is
  wasted. The share is of card SLOTS (equal time per card is the working assumption; make it a share of time later by weighting).
* The rule never mixes families inside a job and never lifts a cap. Each side is first cut to its own room (``eval_room`` /
  ``train_room``: the eval family's own open-card cap and the train round's open < 100 / today < 600 headroom,
  ``anchor_search.headroom``), then merged. Every card must belong to its family's split: a train card whose clip is not
  ``train`` or an eval card whose clip is ``train`` raises ``SplitError`` (the per-job split guard, applied again at the queue).
* Pure and deterministic: the same inputs give the same order.

``python -m footy_track.ball_mining.share --eval-job REF --train-job REF [--limit N]`` prints the merged order of the two jobs'
unanswered cards (read-only; the labeller serves one job at a time, so the page that deals the queue calls :func:`merge`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from footy_track.ball_mining import anchor_search as A  # noqa: N812

#: Share of labelling slots reserved for the eval family while it has cards left (a config constant: change it here).
EVAL_SHARE = 0.30
EVAL, TRAIN = "eval", "train"


@dataclass(frozen=True)
class Slot:
    family: str  # eval | train
    card: dict


def eval_slot_flags(n: int, share: float = EVAL_SHARE) -> list[bool]:
    """``True`` where position ``i`` (0-based) is an eval slot: ``floor((i+1)*share) > floor(i*share)``. Every prefix of ``k``
    positions holds ``floor(k * share)`` eval slots (exact for 0 <= share <= 1; evenly spread, never bunched)."""
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must be in [0, 1], got {share}")
    # integer arithmetic on thousandths avoids float drift at multiples (0.3 * 10 = 3.0000000000000004)
    pm = round(share * 1000)
    return [((i + 1) * pm) // 1000 > (i * pm) // 1000 for i in range(n)]


def _check_split(cards: Iterable[dict], family: str) -> None:
    for c in cards:
        split = A.clip_split(A.canonical_clip_id(c["clip"]))
        if (split == "train") != (family == TRAIN):
            raise A.SplitError(f"{family} queue got a card of split {split!r} ({c['clip']}:{c.get('frame_index')}); families never mix")


def merge(
    eval_cards: Sequence[dict],
    train_cards: Sequence[dict],
    *,
    share: float = EVAL_SHARE,
    eval_room: int | None = None,
    train_room: int | None = None,
    limit: int | None = None,
) -> list[Slot]:
    """Interleave the two families' unanswered cards (each list in its own priority order) under the fixed eval share.

    ``eval_room`` / ``train_room`` cut each list first (their own caps); ``limit`` cuts the merged queue. While both lists have
    cards, positions follow :func:`eval_slot_flags`; a family that runs out hands its slots to the other."""
    _check_split(eval_cards, EVAL)
    _check_split(train_cards, TRAIN)
    ev = list(eval_cards if eval_room is None else eval_cards[: max(0, eval_room)])
    tr = list(train_cards if train_room is None else train_cards[: max(0, train_room)])
    total = len(ev) + len(tr)
    flags = eval_slot_flags(total, share)
    out: list[Slot] = []
    ie = it = 0
    for want_eval in flags:
        if ie >= len(ev) and it >= len(tr):
            break
        take_eval = (want_eval and ie < len(ev)) or it >= len(tr)
        if take_eval:
            out.append(Slot(EVAL, ev[ie]))
            ie += 1
        else:
            out.append(Slot(TRAIN, tr[it]))
            it += 1
    return out if limit is None else out[: max(0, limit)]


def eval_fraction(queue: Sequence[Slot], k: int | None = None) -> float:
    """Share of eval slots in the first ``k`` positions (all by default)."""
    head = queue if k is None else queue[:k]
    return sum(1 for s in head if s.family == EVAL) / len(head) if head else 0.0


def _unanswered(doc: dict) -> list[dict]:
    return [p["item"] for p in doc.get("items") or [] if not (p["total"] and p["done"] >= p["total"])]


def main(argv: list[str] | None = None) -> int:
    from footy_track.labeller import jobs_client  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-job", required=True)
    ap.add_argument("--train-job", required=True)
    ap.add_argument("--share", type=float, default=EVAL_SHARE)
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args(argv)
    ev, tr = jobs_client.results(a.eval_job), jobs_client.results(a.train_job)
    q = merge(_unanswered(ev), _unanswered(tr), share=a.share, limit=a.limit)
    print(json.dumps({"share": a.share, "eval_job": ev["job"]["id"], "train_job": tr["job"]["id"],
                      "queue": [{"family": s.family, "clip": s.card["clip"], "frame": s.card.get("frame_index")} for s in q],
                      "eval_fraction": round(eval_fraction(q), 4)}, indent=2))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
