# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408
"""Evaluation driver: clean evaluable set, dev/test split, in-process method scoring."""

from __future__ import annotations

import dataclasses

from footy_track.ball_offline.data import ClipData, load_clip
from footy_track.ball_offline.metrics import Counts, ceiling, score_clip
from footy_track.ball_offline.snapshot import Snapshot
from footy_track.ball_offline.sources import EvalSetSpec


@dataclasses.dataclass
class ClipEval:
    clip: ClipData
    contaminated: set[int]  # GT/no_ball frames overlapping detector training
    clean_scored: set[int]  # GT ball + no_ball frames not contaminated
    all_scored: set[int]
    role: str  # "dev" | "test"

    @property
    def n_gt_clean(self) -> int:
        return sum(1 for t in self.clean_scored if t in self.clip.gt)

    @property
    def n_gt_contam(self) -> int:
        return sum(1 for t in self.contaminated if t in self.clip.gt)

    @property
    def scored(self) -> set[int]:
        """Frames this clip is scored on: only clean frames for test clips, every
        labelled frame for dev clips (tuning only, never reported as the result)."""
        return self.clean_scored if self.role == "test" else self.all_scored


def build_sets(
    snap: Snapshot,
    spec: EvalSetSpec,
    *,
    clips: list[str] | None = None,
) -> list[ClipEval]:
    """Scorable clips of a frozen snapshot: GT, roles and contamination come from
    the snapshot, candidates from ``spec``."""
    out: list[ClipEval] = []
    for cid, sc in sorted(snap.clips.items()):
        if clips is not None and cid not in clips:
            continue
        clip = load_clip(cid, spec, gt=sc.gt_load())
        scored_all = set(clip.gt) | clip.no_ball
        contam = scored_all & sc.contaminated
        out.append(ClipEval(clip, contam, scored_all - contam, scored_all, sc.role))
    return out


# --------------------------------------------------------------------------


def run_method(ev: ClipEval, fn, scored: set[int] | None = None) -> Counts:
    pred = fn(ev.clip)
    sc = ev.scored if scored is None else scored
    gt = {t: g.xy for t, g in ev.clip.gt.items()}
    return score_clip(pred, gt, ev.clip.no_ball, ev.clip.n_frames, scored=sc)


def pooled(items: list[Counts]) -> Counts:
    tot = Counts()
    for c in items:
        tot += c
    return tot


def eval_method(evs: list[ClipEval], fn) -> tuple[dict[str, dict], dict]:
    per = {ev.clip.clip_id: run_method(ev, fn) for ev in evs}
    return {k: v.summary() for k, v in per.items()}, pooled(
        list(per.values())
    ).summary()


def theoretical_ceiling(evs: list[ClipEval], r: float = 10.0, min_conf: float = 0.1):
    hit = n = 0
    per = {}
    for ev in evs:
        h, m = ceiling(ev.clip, ev.scored, r, min_conf)
        per[ev.clip.clip_id] = (h, m)
        hit += h
        n += m
    return hit, n, per
