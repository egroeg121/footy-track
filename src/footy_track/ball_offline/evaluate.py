# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408
"""Evaluation driver: clean evaluable set, dev/test split, tuning, runs."""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np

from footy_track.ball_offline import baselines as bl
from footy_track.ball_offline.data import (
    ClipData,
    contaminated_frames,
    data_root,
    load_clip,
    load_gt,
    machine_stem_map,
    roboflow_frames,
)
from footy_track.ball_offline.metrics import Counts, ceiling, score_clip
from footy_track.ball_offline.solver import SolverParams
from footy_track.clip_id import canonical_clip_id

NEAR = 5  # frames either side of a detector-training image that count as contaminated
DEV_MAJORITY = (
    0.5  # clip is "dev" when >= this share of its GT ball frames is contaminated
)


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


def gt_clip_ids(root: Path) -> list[str]:
    ids = set()
    for p in (root / "ball_gt_marks").glob("*.jsonl"):
        ids.add(canonical_clip_id(p.name))
    return sorted(ids)


def build_sets(root: Path | None = None, *, with_camera: bool = True) -> list[ClipEval]:
    """All clips that have both human ball GT and machine candidates."""
    root = root or data_root()
    ml = machine_stem_map(root)
    rf = roboflow_frames(root)
    out: list[ClipEval] = []
    for cid in gt_clip_ids(root):
        if cid not in ml:
            continue
        gt, nb, _, _ = load_gt(root, cid)
        if not gt and not nb:
            continue
        clip = load_clip(cid, root, with_camera=with_camera)
        bad = contaminated_frames(cid, rf, near=NEAR)
        scored_all = set(clip.gt) | clip.no_ball
        contam = scored_all & bad
        clean = scored_all - bad
        n_ball = len(clip.gt)
        n_ball_bad = sum(1 for t in contam if t in clip.gt)
        role = "dev" if n_ball and n_ball_bad / n_ball >= DEV_MAJORITY else "test"
        out.append(ClipEval(clip, contam, clean, scored_all, role))
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


def sweep_b0(evs: list[ClipEval], taus) -> list[dict]:
    rows = []
    for tau in taus:
        _, pool = eval_method(evs, lambda c, tau=tau: bl.b0_per_frame(c, tau))
        rows.append({"tau": tau, **pool})
    return rows


def best_row(rows: list[dict], key: str = "f1@10") -> dict:
    return max(rows, key=lambda r: r[key] if r[key] == r[key] else -1)


# --------------------------------------------------------------------------
# tuning (dev clips only)
# --------------------------------------------------------------------------

SEARCH_SPACE = {
    "min_conf": [0.1, 0.2, 0.3, 0.4, 0.5],
    "bias": [-1.0, 0.0, 1.0, 2.0],
    "kappa": [0.5, 1.5, 3.0, 5.0],
    "lam_pos": [0.5, 2.0, 4.0],
    "vmax": [20.0, 40.0, 70.0],
    "r0": [8.0, 12.0, 20.0],
    "mu": [0.0, 0.02, 0.1],
    "max_gap": [10, 25],
    "fill_gap": [0, 10, 25],
    "fill_min_p": [0.5, 0.8],
    "fill_mode": ["linear", "kalman"],
    "lam_vel": [0.0, 0.5, 1.5],
    "static_px": [0.0, 3.0, 8.0],
}


_G: dict = {}


def _work(args):
    idx, p = args
    ev = _G["dev"][idx]
    pred = bl.b1_solver(ev.clip, p)[0]
    gt = {t: g.xy for t, g in ev.clip.gt.items()}
    return score_clip(pred, gt, ev.clip.no_ball, ev.clip.n_frames, scored=ev.scored)


def random_search(
    dev: list[ClipEval],
    base: SolverParams,
    n_iter: int,
    seed: int = 0,
    space: dict | None = None,
    log=None,
    workers: int = 1,
) -> tuple[SolverParams, list[tuple[float, dict]]]:
    """Random search on dev clips maximising pooled F1@10px.  Starts from ``base``."""
    space = space or SEARCH_SPACE
    rng = random.Random(seed)

    mp_pool = None
    if workers > 1:
        import multiprocessing as mp  # noqa: PLC0415

        _G["dev"] = dev
        mp_pool = mp.get_context("fork").Pool(workers)

    def objective(p: SolverParams) -> float:
        if mp_pool is not None:
            pool = pooled(
                mp_pool.map(_work, [(i, p) for i in range(len(dev))])
            ).summary()
        else:
            _, pool = eval_method(dev, lambda c: bl.b1_solver(c, p)[0])
        f1 = pool["f1@10"]
        return f1 if f1 == f1 else 0.0

    best, best_f = base, objective(base)
    hist = [(best_f, dataclasses.asdict(best))]
    for i in range(n_iter):
        kw = {k: rng.choice(v) for k, v in space.items()}
        if "fill_gap" in kw and kw["fill_gap"] == 0:
            kw["fill_mode"] = "none"
        cand = base.replace(**kw)
        f = objective(cand)
        hist.append((f, kw))
        if f > best_f:
            best, best_f = cand, f
            if log:
                log(f"  iter {i}: F1@10={f:.4f} {kw}")
    # coordinate polish
    for k, vals in space.items():
        for v in vals:
            kw = {k: v}
            if k == "fill_gap":
                kw["fill_mode"] = (
                    "none"
                    if v == 0
                    else (best.fill_mode if best.fill_mode != "none" else "linear")
                )
            cand = best.replace(**kw)
            f = objective(cand)
            if f > best_f + 1e-9:
                best, best_f = cand, f
                if log:
                    log(f"  polish {k}={v}: F1@10={f:.4f}")
    if mp_pool is not None:
        mp_pool.close()
    return best, hist


# --------------------------------------------------------------------------
# failure analysis
# --------------------------------------------------------------------------


def failure_breakdown(
    evs: list[ClipEval], pred_fn, nodes_fn=None, r: float = 10.0, pan_px: float = 12.0
) -> dict:
    """Classify every scored GT-ball frame of ``evs`` for one method."""
    cats: dict[str, int] = {}

    def bump(k):
        cats[k] = cats.get(k, 0) + 1

    n_total = 0
    wrong_on_player = 0
    wrong_total = 0
    for ev in evs:
        clip = ev.clip
        pred = pred_fn(clip)
        nodes = nodes_fn(clip) if nodes_fn else None
        acc_frames = (
            np.array(sorted(n.t for n in nodes)) if nodes else np.array([], int)
        )
        cum = np.hypot(*clip.cam_shift.T)
        for t in sorted(ev.scored):
            if t not in clip.gt:
                continue
            n_total += 1
            gx, gy = clip.gt[t].xy
            xy, conf = clip.cands(t)
            avail = bool(
                len(xy)
                and np.any(
                    (np.hypot(xy[:, 0] - gx, xy[:, 1] - gy) <= r) & (conf >= 0.1)
                )
            )
            p = pred.get(t)
            hit = p is not None and np.hypot(p[0] - gx, p[1] - gy) <= r
            if hit:
                bump("hit")
                continue
            ctx = []
            if t < 25:
                ctx.append("clip_start")
            lo, hi = max(0, t - 5), min(clip.n_frames, t + 6)
            if np.median(cum[lo:hi]) > pan_px:
                ctx.append("fast_pan")
            if len(acc_frames):
                gap = int(np.min(np.abs(acc_frames - t)))
                if gap > 25:
                    ctx.append("far_from_any_solved_point(>25f)")
            ctx_s = "+".join(ctx) if ctx else "mid_clip"
            if not avail:
                bump("miss:no_candidate>=0.1_within_10px")
                bump(f"  [ctx] undetectable/{ctx_s}")
            elif p is None:
                bump("miss:candidate_available_but_nothing_emitted")
                bump(f"  [ctx] unemitted/{ctx_s}")
            else:
                bump("miss:candidate_available_but_wrong_point_emitted")
                bump(f"  [ctx] wrong/{ctx_s}")
                wrong_total += 1
                pl = clip.players[t]
                if len(pl):
                    inside = (np.abs(pl[:, 0] - p[0]) <= pl[:, 2] / 2) & (
                        np.abs(pl[:, 1] - p[1]) <= pl[:, 3] / 2
                    )
                    if inside.any():
                        wrong_on_player += 1
    cats["_n_gt_frames"] = n_total
    cats["_wrong_point_inside_a_player_box"] = wrong_on_player
    cats["_wrong_point_total"] = wrong_total
    return cats


def theoretical_ceiling(evs: list[ClipEval], r: float = 10.0, min_conf: float = 0.1):
    hit = n = 0
    per = {}
    for ev in evs:
        h, m = ceiling(ev.clip, ev.scored, r, min_conf)
        per[ev.clip.clip_id] = (h, m)
        hit += h
        n += m
    return hit, n, per
