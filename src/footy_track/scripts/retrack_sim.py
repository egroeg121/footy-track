"""Offline simulation: does re-tracking after each human verdict cut review taps?

A simulated reviewer (the oracle = human GT marks) answers one card per tap: the
card is a frame plus the tracker's current answer there. Strategies:

  A  no re-tracking, cards in spot-check order (first / last / biggest jump /
     middle frame of each track, interleaved over tracks)
  B  no re-tracking, random order (mean of several seeds)
  C  after each verdict re-run the whole-clip constrained Viterbi (anchor on
     "ball", forbidden candidate on "not a ball"); next card = lowest margin,
     frames that changed in the last re-track first
  D  C + a "not a ball" also forbids that image position in every frame
  E  D + chained consecutive segments (only for clip pairs that are adjacent)

Cards can only be asked on frames that have GT, because that is all the oracle can
answer. Accuracy is per GT frame: a ball frame is right when the track is within R
px of a mark, a no-ball frame when the track says nothing. A frame the reviewer was
asked about is right by construction (the verdict, or the dragged position).

    python -m footy_track.scripts.retrack_sim sim   --cache C --gt G --split S --out OUT
    python -m footy_track.scripts.retrack_sim joins --clips DIR --out OUT
    python -m footy_track.scripts.retrack_sim plot  --results OUT/results.json --out OUT
"""

from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from footy_track.ball_eval.detection_replay import (
    CandidateCache,
    load_gt,
    read_cache,
)
from footy_track.ball_trackers.constrained import ConstrainedViterbiTracker, Solution
from footy_track.ball_trackers.detection_tracker import DETECTED, NONE, BallPoint

R_PX = 25.0
DONE_MARGINS = (1.0, 2.0, 3.0)  # margin at which a detected frame counts as done (C/D)
DONE_CONFS = (0.5, 0.6, 0.7, 0.8)  # confidence at which a detected frame counts as done
CHANGED_BONUS = 2.0  # margin discount for frames the last re-track changed
FORBID_TOL = 3.0  # px, "not a ball" in this frame
PARK_TOL = 5.0  # px, "not a ball" at this image position in every frame
MIN_GT_FRAMES = 5
STRATEGIES = ("A", "B", "C", "D", "E")
N_SEEDS = 8

# ------------------------------------------------------------------------ data


@dataclass
class Unit:
    """One clip (or a chain of consecutive clips) with its GT."""

    name: str
    group: str
    cache: CandidateCache
    gt: dict[int, list[tuple[float, float]]]  # frame -> ball marks in px ([] = no ball)

    @property
    def frames(self) -> list[int]:
        return sorted(self.gt)


def _to_px(marks, size):
    return [(x * size[0], y * size[1]) for x, y in marks]


def load_units(
    cache_dir: Path, gt_dir: Path, split: dict[str, str], truth_path: Path | None
) -> list[Unit]:
    gts = load_gt(gt_dir)
    truth = json.loads(truth_path.read_text()) if truth_path else {}
    units = []
    for clip, group in split.items():
        path = cache_dir / f"{clip}.jsonl"
        if not path.exists():
            continue
        cache = read_cache(path)
        gt: dict[int, list[tuple[float, float]]] = {}
        if group == "held_out" and truth:
            for key, v in truth.items():
                c, f = key.rsplit("|", 1)
                if c == clip:
                    gt[int(f)] = (
                        [tuple(m) for m in v["gt"]] if v["kind"] == "ball" else []
                    )
        else:
            g = gts.get(clip)
            if g is None:
                continue
            ball, no_ball = g.scored(("human",))
            for f, cs in ball.items():
                gt[f] = _to_px(cs, cache.frame_size)
            for f in no_ball:
                gt[f] = []
        units.append(Unit(clip, group, cache, gt))
    return units


def chain_cache(a: CandidateCache, b: CandidateCache, gap: int) -> CandidateCache:
    """``b`` appended to ``a`` with frame indices continuing after ``gap`` source frames."""
    last = a.frames[-1][0]
    off = last + gap
    frames = list(a.frames) + [(f + off, cs) for f, cs in b.frames]
    return CandidateCache(f"{a.clip}+{b.clip}", a.frame_size, frames, {})


# ---------------------------------------------------------------- oracle / state


def point_xy(p: BallPoint, size) -> tuple[float, float] | None:
    if p.x is None:
        return None
    return (p.x + p.w / 2) * size[0], (p.y + p.h / 2) * size[1]


def dist_to_marks(xy, marks) -> float:
    return min(math.hypot(xy[0] - mx, xy[1] - my) for mx, my in marks)


def classify(p: BallPoint, marks, size) -> str:
    """hit | wrong | miss on a ball frame; ok | false on a no-ball frame."""
    xy = point_xy(p, size) if p.source != NONE else None
    if marks:
        if xy is None:
            return "miss"
        return "hit" if dist_to_marks(xy, marks) <= R_PX else "wrong"
    return "ok" if xy is None else "false"


def tally(points: dict[int, BallPoint], unit: Unit, asked: set[int]) -> dict[str, int]:
    size = unit.cache.frame_size
    out = {"correct": 0, "wrong": 0, "miss": 0, "false": 0}
    for f, marks in unit.gt.items():
        if f in asked:
            out["correct"] += 1
            continue
        k = classify(points[f], marks, size)
        if k in ("hit", "ok"):
            out["correct"] += 1
        else:
            out[k] += 1
    return out


def apply_verdict(
    trk: ConstrainedViterbiTracker, unit: Unit, f: int, p: BallPoint, *, park: bool
) -> None:
    """The oracle answers the card (frame ``f``, the tracker's answer ``p``)."""
    size = unit.cache.frame_size
    marks = unit.gt[f]
    xy = point_xy(p, size) if p.source != NONE else None
    if marks:
        if xy is not None and dist_to_marks(xy, marks) <= R_PX:
            trk.anchor(f, *xy, radius=R_PX)  # "ball"
            return
        # wrong or missing: drag to the nearest mark; a detected box is "not a ball"
        target = (
            min(marks, key=lambda m: math.hypot(m[0] - xy[0], m[1] - xy[1]))
            if xy
            else marks[0]
        )
        trk.anchor(f, *target, radius=R_PX)
        if xy is not None and p.source == DETECTED and park:
            trk.forbid_everywhere(*xy, tol=PARK_TOL)
    elif xy is not None and p.source == DETECTED:  # no ball here: "not a ball"
        trk.forbid(f, *xy, tol=FORBID_TOL)
        if park:
            trk.forbid_everywhere(*xy, tol=PARK_TOL)


# ------------------------------------------------------------------- strategies


def spot_check_order(points: list[BallPoint], gt_frames: list[int], size) -> list[int]:
    """GT frames in spot-check order: per track first, last, biggest jump, middle."""
    by_f = {p.frame_index: p for p in points}
    tracks: dict[int, list[int]] = {}
    for p in points:
        if p.source != NONE and p.track_id is not None:
            tracks.setdefault(p.track_id, []).append(p.frame_index)
    gset = set(gt_frames)

    def step(f: int) -> float:
        a, b = by_f.get(f - 1), by_f.get(f)
        if a is None or b is None or a.source == NONE or a.track_id != b.track_id:
            return 0.0
        pa, pb = point_xy(a, size), point_xy(b, size)
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    lists = []
    for fs in tracks.values():
        g = [f for f in sorted(fs) if f in gset]
        if not g:
            continue
        picks = [g[0], g[-1], max(g, key=step)]
        if len(g) > 4:
            picks.append(g[len(g) // 2])
        lists.append(list(dict.fromkeys(picks)))
    order: list[int] = []
    for rank in range(4):  # interleave over tracks
        for lst in lists:
            if rank < len(lst) and lst[rank] not in order:
                order.append(lst[rank])
    in_track = sorted(f for fs in tracks.values() for f in fs if f in gset)
    order += [f for f in in_track if f not in order]  # phase 2: densify
    order += [f for f in sorted(gt_frames) if f not in order]  # frames with no track
    return order


def _curve_rows(n: int) -> dict[str, list]:
    keys = ["correct", "wrong", "miss", "false"]
    for pred in _preds():
        keys += [f"done_{pred}", f"dw_{pred}"]
    return {k: [0] * (n + 1) for k in keys}


def _preds() -> list[str]:
    return [f"c{c}" for c in DONE_CONFS] + [f"m{m}" for m in DONE_MARGINS]


def _snap(rows, k, unit, pts, asked, margins) -> None:
    """Record accuracy and, per done-rule, how many frames it would call done and how many of those are wrong."""
    for key, v in tally(pts, unit, asked).items():
        rows[key][k] = v
    size = unit.cache.frame_size
    open_det = [f for f in unit.frames if f not in asked and pts[f].source == DETECTED]
    wrong = {
        f: classify(pts[f], unit.gt[f], size) not in ("hit", "ok") for f in open_det
    }
    for pred in _preds():
        if pred[0] == "c":
            thr = float(pred[1:])
            done = [f for f in open_det if (pts[f].confidence or 0) >= thr]
        elif margins is not None:
            thr = float(pred[1:])
            done = [f for f in open_det if margins[f] >= thr]
        else:
            done = []
        rows[f"done_{pred}"][k] = len(done)
        rows[f"dw_{pred}"][k] = sum(wrong[f] for f in done)


def run_static(unit: Unit, order: list[int], base: list[BallPoint]) -> dict:
    """A / B: no re-tracking; asked frames are right, the rest keep the base track."""
    pts = {p.frame_index: p for p in base}
    frames = unit.frames
    n = len(frames)
    rows = _curve_rows(n)
    asked: set[int] = set()

    def snap(k: int) -> None:
        _snap(rows, k, unit, pts, asked, None)

    snap(0)
    for k, f in enumerate(order[:n], 1):
        asked.add(f)
        snap(k)
    return rows


def run_retrack(
    unit: Unit, *, park: bool, params: dict, extra_cache: CandidateCache | None = None
) -> dict:
    """C / D / E: re-run the constrained tracker after every verdict."""
    cache = extra_cache or unit.cache
    trk = ConstrainedViterbiTracker(cache.frame_size, **params)
    for f, cands in cache.frames:
        trk.update(f, cands)
    frames = unit.frames
    n = len(frames)
    rows = _curve_rows(n)
    asked: set[int] = set()
    sol: Solution = trk.solve()
    pts = {p.frame_index: p for p in sol.points}
    changed: set[int] = set()

    def snap(k: int) -> None:
        _snap(rows, k, unit, pts, asked, sol.margins)

    snap(0)
    for k in range(1, n + 1):
        cand = [f for f in frames if f not in asked]
        f = min(
            cand,
            key=lambda x: (
                sol.margins[x] - (CHANGED_BONUS if x in changed else 0.0),
                x,
            ),
        )
        apply_verdict(trk, unit, f, pts[f], park=park)
        asked.add(f)
        new = trk.solve()
        npts = {p.frame_index: p for p in new.points}
        changed = {g for g in frames if g not in asked and _differs(pts[g], npts[g])}
        sol, pts = new, npts
        snap(k)
    rows["park"] = [len(trk._forbid_all)]
    return rows


def _differs(a: BallPoint, b: BallPoint) -> bool:
    if a.source != b.source:
        return True
    if a.x is None:
        return False
    return abs(a.x - b.x) > 1e-6 or abs(a.y - b.y) > 1e-6


def simulate_unit(args: tuple) -> dict:
    """Run every strategy on one unit (picklable for the process pool)."""
    unit, params, what = args
    base_trk = ConstrainedViterbiTracker(unit.cache.frame_size, **params)
    for f, cands in unit.cache.frames:
        base_trk.update(f, cands)
    base = base_trk.points()
    frames = unit.frames
    out: dict[str, dict] = {}
    if "A" in what:
        order = spot_check_order(base, frames, unit.cache.frame_size)
        out["A"] = run_static(unit, order, base)
    if "B" in what:
        runs = []
        for seed in range(N_SEEDS):
            order = frames[:]
            random.Random(seed).shuffle(order)
            runs.append(run_static(unit, order, base))
        out["B"] = {k: list(np.mean([r[k] for r in runs], axis=0)) for k in runs[0]}
    if "M" in what:  # control: margin order computed once, never re-tracked
        sol0 = base_trk.solve()
        order = sorted(frames, key=lambda x: (sol0.margins[x], x))
        out["M"] = run_static(unit, order, base)
    if "C" in what:
        out["C"] = run_retrack(unit, park=False, params=params)
    if "D" in what:
        out["D"] = run_retrack(unit, park=True, params=params)
    return {"unit": unit.name, "group": unit.group, "n": len(frames), **out}


# ----------------------------------------------------------------------- report


def pooled(results: list[dict], strat: str, key: str, kmax: int) -> np.ndarray:
    """Sum over clips of ``key`` at k taps per clip (clamped to each clip's GT count)."""
    tot = np.zeros(kmax + 1)
    for r in results:
        row = r[strat][key]
        for k in range(kmax + 1):
            tot[k] += row[min(k, len(row) - 1)]
    return tot


def taps_to(acc: np.ndarray, target: float) -> int | None:
    hit = np.nonzero(acc >= target)[0]
    return int(hit[0]) if len(hit) else None


def stop_stats(results: list[dict], strat: str, pred: str) -> dict:
    """Stop when every un-asked frame is "done" under ``pred``. Pooled over clips.

    Static strategies (no re-tracking) stop after asking every frame that is not done
    at the start; re-tracking strategies stop at the first tap count where no un-asked
    frame is left that is not done.
    """
    taps = correct = frames = done = done_wrong = 0
    for r in results:
        row, n = r[strat], r["n"]
        d, dw = row[f"done_{pred}"], row[f"dw_{pred}"]
        if strat in ("A", "B", "M"):
            k = n - int(round(d[0]))
        else:
            k = next((i for i in range(n + 1) if n - i - d[i] == 0), n)
        k = min(k, n)
        taps += k
        done += d[k] if strat not in ("A", "B", "M") else d[0]
        done_wrong += dw[k] if strat not in ("A", "B", "M") else dw[0]
        # asked frames are right; the done ones keep the tracker's answer
        correct += n - (dw[0] if strat in ("A", "B", "M") else dw[k])
        frames += n
    return {
        "taps_per_clip": taps / len(results),
        "tap_fraction": taps / frames,
        "accuracy": correct / frames,
        "done_frames": done,
        "done_wrong": done_wrong,
        "done_error_rate": done_wrong / done if done else None,
    }


def summarise(results: list[dict]) -> dict:
    out = {}
    for group in ("held_out", "seen"):
        rs = [r for r in results if r["group"] == group and r["n"] >= MIN_GT_FRAMES]
        if not rs:
            continue
        n_frames = sum(r["n"] for r in rs)
        kmax = max(r["n"] for r in rs)
        g = {"clips": len(rs), "gt_frames": n_frames, "strategies": {}}
        for s in ("A", "B", "M", "C", "D"):
            acc = pooled(rs, s, "correct", kmax) / n_frames
            taps = np.array([sum(min(k, r["n"]) for r in rs) for k in range(kmax + 1)])
            g["strategies"][s] = {
                "acc_at_0": acc[0],
                "acc_at_5": acc[min(5, kmax)],
                "acc_at_10": acc[min(10, kmax)],
                "acc_at_20": acc[min(20, kmax)],
                "taps_to_90_per_clip": _per_clip(taps_to(acc, 0.90), taps, len(rs)),
                "taps_to_95_per_clip": _per_clip(taps_to(acc, 0.95), taps, len(rs)),
                "taps_to_90_frac": _frac(taps_to(acc, 0.90), taps, n_frames),
                "taps_to_95_frac": _frac(taps_to(acc, 0.95), taps, n_frames),
                "curve": acc.tolist(),
            }
        g["park_events_D"] = sum(r["D"]["park"][0] for r in rs)
        g["stop_rule"] = {
            s: {p: stop_stats(rs, s, p) for p in _preds()} for s in ("A", "C", "D")
        }
        g["errors_at_10"] = {
            s: {
                key: float(pooled(rs, s, key, 10)[10])
                for key in ("correct", "wrong", "miss", "false")
            }
            for s in ("A", "B", "M", "C", "D")
        }
        out[group] = g
    return out


def _per_clip(k, taps, n_clips):
    return None if k is None else float(taps[k] / n_clips)


def _frac(k, taps, n_frames):
    return None if k is None else float(taps[k] / n_frames)


# ------------------------------------------------------------------------ joins


def cmd_joins(args) -> None:
    import cv2  # noqa: PLC0415

    meta = json.loads((Path(args.clips) / "segments.json").read_text())
    segs = meta["segments"]
    out = Path(args.out)
    (out / "joins").mkdir(parents=True, exist_ok=True)

    def read(path: Path, idx: int):
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx if idx >= 0 else n + idx)
        ok, img = cap.read()
        cap.release()
        return img if ok else None

    def feats(img):
        g = cv2.cvtColor(cv2.resize(img, (64, 36)), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
        hsv = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(h, h, 1, 0, cv2.NORM_L1)
        return g, h

    def compare(a, b):
        ga, ha = feats(a)
        gb, hb = feats(b)
        return (
            float(np.abs(ga - gb).mean()),
            float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)),
        )

    rows = []
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        pa, pb = Path(args.clips) / a["file"], Path(args.clips) / b["file"]
        ia, ib = read(pa, -1), read(pb, 0)
        ref = read(
            pa, -26
        )  # one second earlier in the same segment: what "same shot" looks like
        if ia is None or ib is None or ref is None:
            continue
        mad, corr = compare(ia, ib)
        mad_ref, corr_ref = compare(ia, ref)
        rows.append(
            {
                "a": a["file"],
                "b": b["file"],
                "gap_frames": b["start_frame"] - a["end_frame"],
                "mad": mad,
                "hist_corr": corr,
                "ref_mad_1s": mad_ref,
                "ref_hist_corr_1s": corr_ref,
                "img": (ia, ib),
            }
        )
    for r in rows:
        r["continuous"] = bool(
            r["gap_frames"] <= args.max_gap
            and r["hist_corr"] >= args.corr
            and r["mad"] <= args.mad
        )
    pick = (
        sorted(rows, key=lambda r: -r["hist_corr"])[:3]
        + sorted(rows, key=lambda r: r["hist_corr"])[:2]
    )
    pick += [r for r in rows if r["a"].endswith("seg030.mp4")]
    for r in pick:
        ia, ib = r["img"]
        h = 360
        side = np.hstack([cv2.resize(ia, (640, h)), cv2.resize(ib, (640, h))])
        name = (
            r["a"].replace(".mp4", "").split("_")[-1]
            + "_"
            + r["b"].replace(".mp4", "").split("_")[-1]
        )
        cv2.imwrite(str(out / "joins" / f"{name}_corr{r['hist_corr']:.2f}.jpg"), side)
    for r in rows:
        r.pop("img")
    summary = {
        "joins": len(rows),
        "min_gap_frames": min(r["gap_frames"] for r in rows),
        "median_gap_frames": float(np.median([r["gap_frames"] for r in rows])),
        "judged_continuous": sum(r["continuous"] for r in rows),
        "hist_corr_ge_0.9": sum(r["hist_corr"] >= 0.9 for r in rows),
        "median_hist_corr": float(np.median([r["hist_corr"] for r in rows])),
        "median_ref_hist_corr_1s": float(
            np.median([r["ref_hist_corr_1s"] for r in rows])
        ),
        "rule": f"continuous = gap <= {args.max_gap} source frames and hist corr >= {args.corr} and mean abs diff <= {args.mad}",
    }
    (out / "joins.json").write_text(
        json.dumps({"summary": summary, "joins": rows}, indent=1)
    )
    print(json.dumps(summary, indent=1))


# ------------------------------------------------------------------------- plot


def cmd_plot(args) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    data = json.loads(Path(args.results).read_text())
    summ = data["summary"]
    colors = {
        "A": "#1f77b4",
        "B": "#7f7f7f",
        "M": "#9467bd",
        "C": "#ff7f0e",
        "D": "#2ca02c",
    }
    names = {
        "A": "A spot-check order, no re-track",
        "M": "M margin order, no re-track (control)",
        "B": "B random order, no re-track",
        "C": "C re-track + margin order",
        "D": "D C + forbid parked spot",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, group in zip(axes, ("held_out", "seen"), strict=True):
        g = summ.get(group)
        if not g:
            continue
        for s in ("A", "B", "M", "C", "D"):
            curve = g["strategies"][s]["curve"][: args.kmax + 1]
            ax.plot(
                range(len(curve)),
                curve,
                label=names[s],
                color=colors[s],
                lw=4 if s == "C" else 2,
                ls="--" if s == "D" else "-",
            )
        ax.axhline(0.9, color="k", lw=0.5, ls=":")
        ax.axhline(0.95, color="k", lw=0.5, ls=":")
        ax.set_title(
            f"{group.replace('_', '-')}: {g['clips']} clips, {g['gt_frames']} GT frames"
        )
        ax.set_xlabel("taps per clip")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("per-frame accuracy on GT frames")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out / "accuracy_vs_taps.png", dpi=130)
    print(out / "accuracy_vs_taps.png")


# -------------------------------------------------------------------------- sim


def cmd_sim(args) -> None:
    params = dict(kv.split("=") for kv in args.param)
    params = {k: float(v) for k, v in params.items()}
    split = json.loads(Path(args.split).read_text())
    units = load_units(
        Path(args.cache),
        Path(args.gt),
        split,
        Path(args.truth) if args.truth else None,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(u, params, ("A", "B", "M", "C", "D")) for u in units if len(u.gt) >= 1]
    tasks.sort(key=lambda t: -len(t[0].cache.frames) * len(t[0].gt))
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(simulate_unit, tasks))
    results.sort(key=lambda r: (r["group"], r["unit"]))
    summary = summarise(results)
    # chained pair (E): adjacent segments, with and without pretending the join is continuous
    chain = {}
    by = {u.name: u for u in units}
    for a_name, b_name, gap in args.chain:
        if a_name not in by or b_name not in by:
            continue
        a, b = by[a_name], by[b_name]
        for label, g in (
            ("joint_no_link_D", 10_000),  # both clips solved together, nothing chained
            ("real_gap_E", gap),  # the real gap between the segments
            ("pretend_contiguous_E", 1),  # upper bound: as if the join had no gap
        ):
            cc = chain_cache(a.cache, b.cache, g)
            ua = Unit(
                f"{a_name}+{b_name}",
                "pair",
                cc,
                {**a.gt, **{f + a.cache.frames[-1][0] + g: m for f, m in b.gt.items()}},
            )
            chain[label] = run_retrack(ua, park=True, params=params, extra_cache=cc)
        chain["n"] = len(ua.gt)
    (out / "results.json").write_text(
        json.dumps(
            {"params": params, "summary": summary, "per_clip": results, "chain": chain},
            default=float,
        )
    )
    print(
        json.dumps(
            {
                g: {
                    s: {k: v for k, v in d.items() if k != "curve"}
                    for s, d in v["strategies"].items()
                }
                for g, v in summary.items()
            },
            indent=1,
            default=float,
        )
    )
    print(
        json.dumps(
            {g: v["stop_rule"] for g, v in summary.items()}, indent=1, default=float
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--cache", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--truth", help="corrected truth json for the held-out clips")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=10)
    s.add_argument("--param", action="append", default=[], help="viterbi.key=value")
    s.add_argument(
        "--chain",
        nargs=3,
        action="append",
        default=[],
        metavar=("A", "B", "GAP"),
        type=lambda x: int(x) if x.isdigit() else x,
    )
    s.set_defaults(fn=cmd_sim)
    j = sub.add_parser("joins")
    j.add_argument("--clips", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--max-gap", type=int, default=2)
    j.add_argument("--corr", type=float, default=0.85)
    j.add_argument("--mad", type=float, default=25.0)
    j.set_defaults(fn=cmd_joins)
    p = sub.add_parser("plot")
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kmax", type=int, default=60)
    p.set_defaults(fn=cmd_plot)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
