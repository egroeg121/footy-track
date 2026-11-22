"""The eval job family on the shared anchor-card flow (design ``ball_anchor_iteration.md`` task 10, sections 4 and 7.0).

The frozen panel (``panel.py``: windows of one heldout match, exact ``pi_window`` / ``pi_frame``) becomes a ``ball_check`` job
``ball-eval/<panel>/<match>`` whose cards are pre-filled by ONE recorded candidate source (:mod:`eval_source`: the offline
teacher while it beats the best student on human gold, else that student). It runs the same anchor search, the same append /
reorder client and the same full-frame fallback (``ball-eval-place/<panel>/<match>``) as the train family, and differs only in
its set F, its caps and its namespace.

Order of creation (``create``): 1. F and the split of every frame go to ``selected_frame`` (job key = the job's key) BEFORE any
candidate is looked at; 2. the candidate source is chosen and its suggest file written; 3. one seed card per window is built
from it. :func:`build_job_body` refuses to build cards when F is not in the store, so candidates can never decide which frames
are measured. QA repeat / gold cards stay in the blind ``ball-eval-panel`` ``ball_place`` job (they are not part of F).

Termination, per frame of F (:func:`plan_seeds`): a window with no confirmed ball gets ONE seed card (its best candidate that is
not inside a rejected disc); a frame whose card was rejected (``not_ball`` / ``box_off``) is offered its next candidate; when no
eligible candidate is left the frame goes to the full-frame check (``reason`` ``no_candidate`` / ``candidates_exhausted``).
No candidate is never ``no_ball``: only the human check may write that. Frames the anchor search cannot reach
(``unreachable`` / ``beyond_cap``) take the existing fallback path. ``unsure`` twice is routed by ``anchor_fallback``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from footy_track.ball_eval import panel as P  # noqa: N812
from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import jobs

KEY_PREFIX = "ball-eval"
SEED_BUCKET = "E"  # seed cards of the eval family (search cards stay bucket S)
REASON_NO_CANDIDATE = "no_candidate"
REASON_EXHAUSTED = "candidates_exhausted"
EVAL_OPEN_CARDS_MAX = 100  # the eval family's own open-card cap for appended cards (train's is round.OPEN_CARDS_MAX; never one shared count)
JOB_MAX_CARDS = 2000  # total cards a job may hold (jobs.MAX_CARDS ceiling applies)
EVAL_HOOKS = [
    {"name": "anchor-search", "trigger": {"every_n": 10, "min_interval_s": 60},
     "action": {"type": "prefect", "deployment": "ball-anchor-search/ball-anchor-search", "params": {}}},
]  # fmt: skip

Key = tuple[str, int]


class EvalFlowError(RuntimeError):
    pass


def job_key(panel_version: str, match: str) -> str:
    return f"{KEY_PREFIX}/{panel_version}/{match}"


def is_eval_job(job: Mapping) -> bool:
    return str(job.get("key") or "").startswith(KEY_PREFIX + "/")


def panel_frames(panel: Mapping) -> list[tuple[str, int, str, float]]:
    """``[(clip, frame, window_id, pi_frame)]`` of the panel's windows (QA cards excluded: they are not estimated)."""
    return [(w["clip"], int(f), w["window_id"], float(w["pi_frame"])) for w in panel["windows"] for f in w["frames"]]


def selected_rows(panel: Mapping, *, game_of_clip=lambda c: c):
    """``SelectedFrameRow`` of every panel frame, keyed by the eval job key, with the frame's split (``splits.split_of``)."""
    from footy_track.feature_store.schema import SelectedFrameRow  # noqa: PLC0415

    key = job_key(panel["panel_version"], panel["match"])
    return [
        SelectedFrameRow(job_key=key, game_id=game_of_clip(canonical_clip_id(c)), frame_index=f, window_id=w, split=A.clip_split(canonical_clip_id(c)))
        for c, f, w, _ in panel_frames(panel)
    ]  # fmt: skip


def record_selected(panel: Mapping, store, *, game_of_clip=lambda c: c) -> int:
    """Write F (and its splits) to ``selected_frame``. Idempotent. Must run before any candidate card is built."""
    rows = selected_rows(panel, game_of_clip=game_of_clip)
    bad = {r.split for r in rows} - {"eval_heldout_match", "eval_in_match"}
    if bad:
        raise A.SplitError(f"panel {panel['match']!r} has frames of split {sorted(bad)}; the eval family takes eval splits only")
    return store.upsert_selected_frames(rows)


def require_selected(panel: Mapping, store, *, game_of_clip=lambda c: c) -> None:
    """Refuse to continue unless every panel frame is already in ``selected_frame`` (F is logged before candidates)."""
    key = job_key(panel["panel_version"], panel["match"])
    df = store.read_selected_frames(job_key=key)
    have = {(r.game_id, int(r.frame_index)) for r in df.itertuples(index=False)} if df is not None and len(df) else set()
    want = {(game_of_clip(canonical_clip_id(c)), f) for c, f, _, _ in panel_frames(panel)}
    if want - have:
        raise EvalFlowError(f"{len(want - have)} panel frames of {key} are not in selected_frame: record F before building candidate cards")


def window_groups(panel: Mapping) -> dict[str, tuple[str, list[int]]]:
    return {w["window_id"]: (canonical_clip_id(w["clip"]), sorted(int(f) for f in w["frames"])) for w in panel["windows"]}


def groups_from_store(store, key: str) -> dict[str, tuple[str, list[int]]]:
    """Window groups of a job's F from ``selected_frame.window_id`` (clip = game id)."""
    out: dict[str, tuple[str, list[int]]] = {}
    df = store.read_selected_frames(job_key=key)
    for r in df.itertuples(index=False):
        out.setdefault(r.window_id or f"{r.game_id}:{r.frame_index}", (canonical_clip_id(r.game_id), []))[1].append(int(r.frame_index))
    return {w: (c, sorted(fs)) for w, (c, fs) in out.items()}


# ---------------------------------------------------------------------------
# seeds and the no-candidate fallback
# ---------------------------------------------------------------------------


@dataclass
class SeedPlan:
    cards: list[dict] = field(default_factory=list)  # card specs (anchor_search format), hop 0, source "seed"
    fallback: list[dict] = field(default_factory=list)  # {clip, frame, window_id, reason}: needs the full-frame check


def _inside(c: A.Cand, discs: Iterable[A.Forbidden]) -> bool:
    fw, fh = A.FRAME_SIZE
    return any(math.hypot((c.cx - d.cx) * fw, (c.cy - d.cy) * fh) <= max(d.w * fw, d.h * fh) / 2 + A.FORBIDDEN_MARGIN_PX for d in discs)


def _eligible(cands: A.CandidateSet, anchorset: A.AnchorSet, clip: str, f: int) -> list[A.Cand]:
    discs = anchorset.forbidden.get((clip, f), [])
    return [c for c in cands.get(clip, f) if not _inside(c, discs)]


def _seed_card(clip: str, f: int, wid: str, c: A.Cand, *, job_id: str, gen: str, source: Mapping | None, run_id: str | None, rank: int, pi: float | None) -> dict:
    return {
        "clip": clip, "frame_index": f, "bbox": {"x": c.cx - c.w / 2, "y": c.cy - c.h / 2, "w": c.w, "h": c.h},
        "conf": c.conf, "raw_conf": c.conf, "p_cal": None, "tid": c.tid or None, "cand_source": "detected", "hop": 0, "margin": None,
        "direction": "both", "anchor_id": None, "source": "seed", "bucket": SEED_BUCKET, "generator_version": gen,
        "teacher_version": (source or {}).get("teacher_version"), "candidate_source": dict(source) if source else None, "run_id": run_id,
        "search_round": 0, "window_id": wid, "rank_conf": c.conf, "rank": rank, "job_id": job_id, "pi_frame": pi,
    }  # fmt: skip


def plan_seeds(
    groups: Mapping[str, tuple[str, list[int]]],
    cands: A.CandidateSet,
    anchorset: A.AnchorSet,
    *,
    job_id: str,
    terminal: Iterable[Key] = (),
    pending: Iterable[Key] = (),
    offered: Iterable[Key] = (),
    skip: Iterable[Key] = (),
    run_id: str | None = None,
    pi: Mapping[Key, float] | None = None,
) -> SeedPlan:
    """Seed cards and no-candidate fallback entries of the open frames of F (see the module docstring).

    ``terminal`` = frames with a terminal state in the store, ``pending`` = frames with an unanswered card in the job, ``offered``
    = frames a search document already offers a card for, ``skip`` = frames routed elsewhere (unsure twice -> fallback).
    A frame with a human anchor, a terminal state, a live card or a card on offer is not open. Deterministic and idempotent: the
    same inputs give the same plan, and a rerun after the cards are appended plans nothing for those frames."""
    done = {(canonical_clip_id(c), int(f)) for c, f in (*terminal, *pending, *offered, *skip)} | set(anchorset.anchors)
    gen = A.generator_version(cands.detector)
    src = cands.source_info
    plan = SeedPlan()
    n = 0
    for wid in sorted(groups):
        clip, frames = groups[wid]
        opn = [f for f in frames if (clip, f) not in done]
        if not opn:
            continue
        has_anchor = any((clip, f) in anchorset.anchors for f in frames)
        has_card = any((clip, f) in done and (clip, f) not in anchorset.anchors for f in frames)
        carded: set[int] = set()
        for f in opn:  # a rejected frame is offered its next candidate (or goes to the full-frame check)
            if not anchorset.forbidden.get((clip, f)):
                continue
            el = _eligible(cands, anchorset, clip, f)
            if el:
                plan.cards.append(_seed_card(clip, f, wid, el[0], job_id=job_id, gen=gen, source=src, run_id=run_id, rank=n, pi=(pi or {}).get((clip, f))))
                n += 1
                carded.add(f)
            else:
                plan.fallback.append({"clip": clip, "frame": f, "window_id": wid, "reason": REASON_EXHAUSTED})
        if has_anchor or has_card or carded:
            continue  # the search (or the card already out) takes this window from here
        fresh = [f for f in opn if not anchorset.forbidden.get((clip, f))]
        best = max(((f, c) for f in fresh for c in _eligible(cands, anchorset, clip, f)[:1]), key=lambda fc: (fc[1].conf, -fc[0]), default=None)
        if best is None:
            plan.fallback.extend({"clip": clip, "frame": f, "window_id": wid, "reason": REASON_NO_CANDIDATE} for f in fresh)
            continue
        f, c = best
        plan.cards.append(_seed_card(clip, f, wid, c, job_id=job_id, gen=gen, source=src, run_id=run_id, rank=n, pi=(pi or {}).get((clip, f))))
        n += 1
    return plan


def seed_doc(job_id: str, split: str | None, cards: list[dict], dataset: str | None = None) -> dict:
    """The seed cards wrapped as a search document so ``anchor_search.push_search`` appends them within the caps (digest = their keys)."""
    digest = hashlib.sha1("|".join(f"{c['clip']}:{c['frame_index']}:{A.card_key(c)[2]}" for c in cards).encode()).hexdigest()
    return {"schema": A.SEARCH_SCHEMA, "version": 0, "window_id": "seed", "job_id": job_id, "dataset": dataset, "split": split,
            "digest": digest, "cards": cards, "unreachable": [], "beyond_cap": [], "kind": "eval_seed"}  # fmt: skip


def augment(job: dict, docs: list[dict], *, marks_dir, store=None, pending: Iterable[Key] = (), run_id: str | None = None) -> tuple[list[dict], SeedPlan]:
    """One eval iteration after ``anchor_search.run_search``: the search documents plus a seed document, and the plan (whose
    ``fallback`` entries the caller hands to ``anchor_fallback.push_fallback``). Reads F from the store when given."""
    scope = A.job_scope(job)
    selected = A.resolve_selected(job, store)
    jobf = A.job_with_selected(job, selected)
    cands = A.load_candidates(jobf)
    cands.by_frame = {k: v for k, v in cands.by_frame.items() if k in selected}
    anchorset = A.load_anchors(jobf, marks_dir=marks_dir)
    anchorset.anchors = {k: v for k, v in anchorset.anchors.items() if k in selected}
    anchorset.forbidden = {k: v for k, v in anchorset.forbidden.items() if k in selected}
    term = A.terminal_frames(store, selected) if store is not None else ()
    groups: dict[str, tuple[str, list[int]]] = {}
    if store is not None and len(store.read_selected_frames(job_key=job.get("key"))):
        groups = {w: (c, [f for f in fs if (c, f) in selected]) for w, (c, fs) in groups_from_store(store, job["key"]).items()}
    else:
        for wid, (clip, _s, fs) in A.split_windows(selected).items():
            groups[wid] = (clip, fs)
    offered = {(c["clip"], int(c["frame_index"])) for d in docs for c in d.get("cards") or []}
    from footy_track.ball_mining import anchor_fallback as F  # noqa: N812, PLC0415

    x2 = F.unsure_twice(marks_dir, selected)
    plan = plan_seeds(groups, cands, anchorset, job_id=scope.job_id, terminal=term, pending=pending, offered=offered, skip=x2, run_id=run_id)
    check = [{"clip": c["clip"], "frame_index": c["frame_index"]} for c in plan.cards]
    A.check_emittable(check, selected, scope.split)
    out = list(docs)
    if plan.cards:
        out.append(seed_doc(scope.job_id, scope.split, plan.cards, scope.dataset))
    return out, plan


def fallback_docs(docs: Iterable[dict], plan: SeedPlan, generator_version: str | None = None) -> list[dict]:
    """Search documents plus one synthetic document carrying the plan's no-candidate frames as ``unreachable`` rows, the
    shape ``anchor_fallback.push_fallback`` consumes."""
    extra = {"window_id": "seed", "generator_version": generator_version, "unreachable": list(plan.fallback), "beyond_cap": [], "cards": []}
    return [*docs, extra] if plan.fallback else list(docs)


# ---------------------------------------------------------------------------
# creation
# ---------------------------------------------------------------------------


def build_job_body(panel: Mapping, cands: A.CandidateSet, *, store, dataset: str | None = None, game_of_clip=lambda c: c) -> tuple[dict, SeedPlan]:
    """The ``ball_check`` job body of a panel: one seed card per window from the recorded source, the eval hooks, key
    ``ball-eval/<panel>/<match>``. Needs F in the store (:func:`require_selected`); windows with no candidate at all are NOT
    cards: they come back in ``plan.fallback`` for the full-frame check (never ``no_ball``)."""
    require_selected(panel, store, game_of_clip=game_of_clip)
    key = job_key(panel["panel_version"], panel["match"])
    jid = jobs._sha_id(key)  # noqa: SLF001 (the job id is a function of its key, so it is known before the job exists)
    pi = {(canonical_clip_id(c), f): p for c, f, _, p in panel_frames(panel)}
    wanted = set(pi)
    cands = A.CandidateSet(cands.source, cands.detector, {k: v for k, v in cands.by_frame.items() if k in wanted}, cands.source_info)
    plan = plan_seeds(window_groups(panel), cands, A.AnchorSet(), job_id=jid, pi=pi)
    items = []
    for c in plan.cards:
        it = A.item_from_card(c, digest=hashlib.sha1(key.encode()).hexdigest(), split=A.clip_split(c["clip"]), file_version=0)
        it["sampling"].update({"regime": "eval", "panel": panel["panel_version"], "match": panel["match"], "window_id": c["window_id"],
                               "pi_window": c["pi_frame"], "pi_frame": c["pi_frame"]})  # fmt: skip
        items.append(it)
    body = {
        "title": f"Ball eval cards {panel['panel_version']}: {panel['match']}", "kind": "ball_check",
        "dataset": dataset or P.dataset_name(panel["panel_version"]), "items": items, "key": key, "device": "phone", "priority": 2,
        "why": "Ball Check on the frozen eval panel: confirm or reject the suggested ball. Frames with no confirmed ball go to a full-frame check.",
        "hooks": EVAL_HOOKS, "max_cards": JOB_MAX_CARDS,
    }  # fmt: skip
    return body, plan
