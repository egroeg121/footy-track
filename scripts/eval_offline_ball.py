#!/usr/bin/env python
# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408
"""Offline (non-causal) ball-tracking evaluation: B0 / B1 / B2 / B3.

    uv run python scripts/eval_offline_ball.py [--tune-iter 120] [--quick]

Builds the clean evaluable set (human ball GT x machine candidates, minus GT
frames that overlap the detector's Roboflow training images), tunes on the
``dev`` clips, reports on the ``test`` clips, and writes
``docs/experiments/offline_ball_baseline.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from footy_track.ball_offline import baselines as bl
from footy_track.ball_offline import evaluate as E
from footy_track.ball_offline.metrics import Counts, score_clip, wilson
from footy_track.ball_offline.solver import SolverParams

TAUS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
ANCHOR_EVERY = (10, 25, 50)
MIN_GT_FOR_ANCHORS = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt(x, nd=3):
    return "n/a" if x != x else f"{x:.{nd}f}"


def row(name: str, s: dict) -> str:
    lo, hi = wilson(s["tp@10"], s["n_gt_ball"])
    return (
        f"| {name} | {s['n_gt_ball']} | {fmt(s['recall@10'])} [{fmt(lo, 2)}-{fmt(hi, 2)}] "
        f"| {fmt(s['precision@10'])} ({s['tp@10']}/{s['tp@10'] + s['fp@10']}) | {fmt(s['f1@10'])} "
        f"| {fmt(s['recall@5'])} / {fmt(s['recall@20'])} | {fmt(s['coverage'])} "
        f"| {fmt(s['median_err_px'], 1)} |"
    )


HEAD = (
    "| method | GT ball frames | recall@10px [95% Wilson] | precision@10px (TP/emitted on scored) "
    "| F1@10 | recall@5 / @20 | coverage | median err px |\n"
    "|---|---|---|---|---|---|---|---|"
)


def per_clip_spread(per: dict[str, dict], min_gt: int = 10) -> str:
    rs = [v["recall@10"] for v in per.values() if v["n_gt_ball"] >= min_gt]
    if not rs:
        return "n/a"
    return f"min {min(rs):.2f} / median {float(np.median(rs)):.2f} / max {max(rs):.2f} over {len(rs)} clips with >={min_gt} GT"


def run(args) -> dict:
    t0 = time.time()
    sets = E.build_sets()
    dev = [e for e in sets if e.role == "dev"]
    test = [e for e in sets if e.role == "test"]
    result: dict = {"split": {}, "notes": []}

    # ---- clean evaluable set -------------------------------------------------
    clips_tbl = []
    for e in sets:
        c = e.clip
        clips_tbl.append(
            {
                "clip": c.clip_id,
                "role": e.role,
                "frames": c.n_frames,
                "gt_ball_total": len(c.gt),
                "gt_ball_contaminated": e.n_gt_contam,
                "gt_ball_clean": e.n_gt_clean,
                "no_ball_frames": len(c.no_ball),
                "gt_aliases": c.gt_aliases,
                "gt_alias_conflicts_gt10px": c.gt_conflicts,
                "camera_cuts_detected": int(c.cam_cut.sum()),
            }
        )
    result["clips"] = clips_tbl
    result["split"] = {
        "dev": [e.clip.clip_id for e in dev],
        "test": [e.clip.clip_id for e in test],
    }
    n_test_ball = sum(e.n_gt_clean for e in test)
    n_test_nb = sum(1 for e in test for t in e.clean_scored if t in e.clip.no_ball)
    n_dev_ball = sum(len(e.clip.gt) for e in dev)
    log(
        f"test: {len(test)} clips, {n_test_ball} clean ball instances, {n_test_nb} no_ball frames"
    )
    log(
        f"dev: {len(dev)} clips, {n_dev_ball} ball instances (contaminated; tuning only)"
    )

    # ---- ceiling --------------------------------------------------------------
    result["ceiling"] = {}
    for name, evs in (("test", test), ("dev", dev)):
        for mc in (0.1, 0.3, 0.6):
            h, n, per = E.theoretical_ceiling(evs, 10.0, mc)
            result["ceiling"][f"{name}@conf>={mc}"] = {
                "hit": h,
                "n": n,
                "frac": h / n if n else None,
            }
        for r in (5.0, 20.0):
            h, n, _ = E.theoretical_ceiling(evs, r, 0.1)
            result["ceiling"][f"{name}@{int(r)}px,conf>=0.1"] = {
                "hit": h,
                "n": n,
                "frac": h / n if n else None,
            }
        _, _, per = E.theoretical_ceiling(evs, 10.0, 0.1)
        result["ceiling"][f"{name}_per_clip@10px,conf>=0.1"] = {
            k: list(v) for k, v in per.items()
        }

    # ---- B0 -------------------------------------------------------------------
    dev_rows = E.sweep_b0(dev, TAUS)
    test_rows = E.sweep_b0(test, TAUS)
    tau_dev = E.best_row(dev_rows)["tau"]
    result["B0"] = {
        "tau_selected_on_dev": tau_dev,
        "dev_sweep": dev_rows,
        "test_sweep": test_rows,
        "test_at_dev_tau": next(r for r in test_rows if r["tau"] == tau_dev),
        "test_oracle_best_tau": E.best_row(test_rows),
    }
    per_b0, pool_b0 = E.eval_method(test, lambda c: bl.b0_per_frame(c, tau_dev))
    result["B0"]["per_clip"] = per_b0
    log(f"B0 tau(dev)={tau_dev}: test F1@10={pool_b0['f1@10']:.3f}")

    # ---- B1 tuning on dev -----------------------------------------------------
    base = SolverParams()
    tuned = {}
    for cam in (False, True):
        log(f"tuning B1 (camera={cam}) on {len(dev)} dev clips, {args.tune_iter} iters")
        p, hist = E.random_search(
            dev,
            base.replace(camera=cam),
            args.tune_iter,
            seed=args.seed,
            log=log,
            workers=args.workers,
        )
        _, dev_pool = E.eval_method(dev, lambda c, p=p: bl.b1_solver(c, p)[0])
        tuned[cam] = (p, dev_pool)
        log(f"  dev F1@10 = {dev_pool['f1@10']:.4f}")
    best_cam = max(
        tuned,
        key=lambda k: (
            tuned[k][1]["f1@10"] if tuned[k][1]["f1@10"] == tuned[k][1]["f1@10"] else 0
        ),
    )
    result["B1"] = {
        "tuned_params": {
            ("camera_on" if k else "camera_off"): dataclasses.asdict(v[0])
            for k, v in tuned.items()
        },
        "dev_pooled": {
            ("camera_on" if k else "camera_off"): v[1] for k, v in tuned.items()
        },
        "camera_selected_on_dev": best_cam,
    }
    # B2 gate on dev
    p_best = tuned[best_cam][0]
    gates = {}
    for g in (6.0, 12.0, 20.0):
        _, dp = E.eval_method(dev, lambda c, g=g: bl.b2_rescue(c, p_best, gate_px=g)[0])
        gates[g] = dp["f1@10"]
    gate = max(gates, key=lambda g: gates[g] if gates[g] == gates[g] else 0)
    result["B2"] = {"gate_px_selected_on_dev": gate, "dev_f1_by_gate": gates}

    # ---- test evaluation --------------------------------------------------------
    methods = {}
    for cam in (False, True):
        p = tuned[cam][0]
        key = f"B1 camera={'on' if cam else 'off'}"
        methods[key] = lambda c, p=p: bl.b1_solver(c, p)[0]
    methods[f"B2 (B1 + rescue, camera={'on' if best_cam else 'off'})"] = lambda c: (
        bl.b2_rescue(c, p_best, gate_px=gate)[0]
    )
    methods[f"B0 per-frame tau={tau_dev} (dev-selected)"] = lambda c: bl.b0_per_frame(
        c, tau_dev
    )
    oracle_tau = E.best_row(test_rows)["tau"]
    methods[f"B0 per-frame tau={oracle_tau} (test-oracle, optimistic)"] = lambda c: (
        bl.b0_per_frame(c, oracle_tau)
    )
    # ablations at the selected camera setting
    methods["B1 ablation: no gap fill"] = lambda c: bl.b1_solver(
        c, p_best.replace(fill_mode="none")
    )[0]
    methods["B1 ablation: no velocity term"] = lambda c: bl.b1_solver(
        c, p_best.replace(lam_vel=0.0)
    )[0]
    methods["B1 ablation: kalman fill"] = lambda c: bl.b1_solver(
        c, p_best.replace(fill_mode="kalman")
    )[0]
    methods["B1 ablation: linear fill"] = lambda c: bl.b1_solver(
        c, p_best.replace(fill_mode="linear")
    )[0]
    result["test"] = {}
    for name, fn in methods.items():
        per, pool = E.eval_method(test, fn)
        result["test"][name] = {
            "pooled": pool,
            "per_clip": per,
            "spread": per_clip_spread(per),
        }
        log(
            f"{name}: R@10={fmt(pool['recall@10'])} P@10={fmt(pool['precision@10'])} F1={fmt(pool['f1@10'])} (n={pool['n_gt_ball']})"
        )
    # same methods on dev (flagged contaminated)
    result["dev"] = {}
    for name in ("B1 camera=off", "B1 camera=on"):
        _, pool = E.eval_method(dev, methods[name])
        result["dev"][name] = pool
    _, pool = E.eval_method(dev, lambda c: bl.b0_per_frame(c, tau_dev))
    result["dev"][f"B0 tau={tau_dev}"] = pool

    # ---- B3: anchors ------------------------------------------------------------
    result["B3"] = {}
    anchor_clips = [e for e in test if e.n_gt_clean >= MIN_GT_FOR_ANCHORS]
    result["B3"]["clips"] = [e.clip.clip_id for e in anchor_clips]
    for N in ANCHOR_EVERY:
        agg = {
            "B1+anchors": Counts(),
            "B1 (no anchors, same frames)": Counts(),
            "anchors-only interpolation": Counts(),
            "B0 (same frames)": Counts(),
        }
        per_clip = {}
        spacing = []
        n_anchor = 0
        for e in anchor_clips:
            clip = e.clip
            anchors = bl.anchors_from_gt(clip, N)
            n_anchor += len(anchors)
            ts = sorted(anchors)
            spacing += list(np.diff(ts))
            held = {t for t in e.clean_scored if t not in anchors}
            gt = {t: g.xy for t, g in clip.gt.items()}
            preds = {
                "B1+anchors": bl.b1_solver(clip, p_best, anchors)[0],
                "B1 (no anchors, same frames)": bl.b1_solver(clip, p_best)[0],
                "anchors-only interpolation": bl.anchor_interp(clip, anchors),
                "B0 (same frames)": bl.b0_per_frame(clip, tau_dev),
            }
            for k, pr in preds.items():
                c = score_clip(pr, gt, clip.no_ball, clip.n_frames, scored=held)
                agg[k] += c
                per_clip.setdefault(clip.clip_id, {})[k] = c.summary()
        result["B3"][f"N={N}"] = {
            "n_anchors": n_anchor,
            "median_anchor_spacing_frames": float(np.median(spacing))
            if spacing
            else None,
            "pooled": {k: v.summary() for k, v in agg.items()},
            "per_clip": per_clip,
        }
        log(
            f"B3 N={N}: "
            + ", ".join(
                f"{k}: R@10={fmt(v.summary()['recall@10'])}" for k, v in agg.items()
            )
        )

    # ---- failure analysis ---------------------------------------------------------
    result["failures"] = {
        "B0": E.failure_breakdown(test, lambda c: bl.b0_per_frame(c, tau_dev)),
        "B1": E.failure_breakdown(
            test,
            lambda c: bl.b1_solver(c, p_best)[0],
            lambda c: bl.b1_solver(c, p_best)[1],
        ),
    }
    result["failures_dev"] = {
        "B1": E.failure_breakdown(
            dev,
            lambda c: bl.b1_solver(c, p_best)[0],
            lambda c: bl.b1_solver(c, p_best)[1],
        ),
    }
    # camera diagnostics
    result["camera"] = {
        e.clip.clip_id: {
            "median_pan_px_per_frame": float(np.median(np.hypot(*e.clip.cam_shift.T))),
            "p95_pan_px_per_frame": float(
                np.percentile(np.hypot(*e.clip.cam_shift.T), 95)
            ),
            "frames_pan_gt_12": int((np.hypot(*e.clip.cam_shift.T) > 12).sum()),
            "cuts": int(e.clip.cam_cut.sum()),
        }
        for e in sets
    }
    result["seconds"] = time.time() - t0
    return result


def write_md(res: dict, path: Path) -> None:
    L = []
    A = L.append
    A("# Offline ball-tracking baseline (B0 / B1 / B2 / B3)\n")
    A(
        "Generated by `scripts/eval_offline_ball.py` (code: `src/footy_track/ball_offline/`). "
        "All distances in pixels at 1920x1080. Detector: RT-DETR-L @1280, conf>=0.1 (`machine_labels/`).\n"
    )
    A("## 1. Clean evaluable set\n")
    A(
        "GT = human-placed or vittrack+human_checked `in_play_ball` marks (machine-only marks excluded), "
        "all alias files of a clip merged through `clip_id.canonical_clip_id`. A GT frame is **contaminated** if it "
        "lies within 5 frames of an image in the Roboflow v11/v3 *train* or *valid* split (the detector "
        "`rtdetr_1920_v11` was trained on train and selected on valid), i.e. the detector has effectively "
        "seen it. Contaminated frames are never part of the reported (test) numbers.\n"
    )
    A(
        "| clip | role | frames | GT ball total | contaminated | **clean** | no_ball frames |"
    )
    A("|---|---|---|---|---|---|---|")
    for c in res["clips"]:
        A(
            f"| {c['clip']} | {c['role']} | {c['frames']} | {c['gt_ball_total']} | {c['gt_ball_contaminated']} | {c['gt_ball_clean']} | {c['no_ball_frames']} |"
        )
    tc = [c for c in res["clips"] if c["role"] == "test"]
    A(
        f"\n**Test (reported) set: {len(tc)} clips, {sum(c['gt_ball_clean'] for c in tc)} clean ball instances.** "
        f"Dev (tuning only; majority-contaminated clips, scored on all their labelled frames): "
        f"{sum(1 for c in res['clips'] if c['role'] == 'dev')} clips, "
        f"{sum(c['gt_ball_total'] for c in res['clips'] if c['role'] == 'dev')} ball instances. "
        "The split is clip-disjoint and rule-based (dev = at least half of the clip's GT ball frames contaminated).\n"
    )
    A(
        "GT clips with no machine labels (arsenal_example, Aston Villa, Bournemouth, `mancity_part*`) cannot be scored "
        "without re-running the detector and are excluded.\n"
    )
    A("## 2. Theoretical ceiling (any candidate >=0.1 within R px of GT)\n")
    A("| set | R | conf floor | frames hit / GT frames | fraction |")
    A("|---|---|---|---|---|")
    for k, v in res["ceiling"].items():
        if "per_clip" in k:
            continue
        A(
            f"| {k.split('@')[0]} | {k.split('@')[1]} | | {v['hit']} / {v['n']} | {fmt(v['frac'])} |"
        )
    A(
        "\nAnything that only re-ranks or links existing candidates (B0, B1, B2) is bounded by this. "
        "Interpolation can exceed it only on frames the detector missed but the trajectory brackets.\n"
    )
    A("## 3. Results on the test set\n")
    A(
        "Hyperparameters tuned on the dev clips only; selected values below. Precision counts every emitted "
        "prediction on a scored frame (GT ball frame or explicit no_ball frame); unlabelled frames are not scored.\n"
    )
    A(HEAD)
    for name, v in res["test"].items():
        A(row(name, v["pooled"]))
    A("\nPer-clip recall@10 spread (clips with >=10 GT):\n")
    for name, v in res["test"].items():
        if (
            name.startswith(("B1 camera", "B0 per-frame tau=0"))
            or "dev-selected" in name
            or name.startswith("B2")
        ):
            A(f"- {name}: {v['spread']}")
    A("\n### Per-clip (selected B1 vs B0)\n")
    b1k = [k for k in res["test"] if k.startswith("B1 camera")]
    best_b1 = max(
        b1k,
        key=lambda k: (
            res["test"][k]["pooled"]["f1@10"]
            if res["test"][k]["pooled"]["f1@10"] == res["test"][k]["pooled"]["f1@10"]
            else 0
        ),
    )
    b0k = [k for k in res["test"] if "dev-selected" in k][0]
    A(f"| clip | GT | B0 recall | B0 prec | {best_b1} recall | prec |")
    A("|---|---|---|---|---|---|")
    for clip in res["test"][b0k]["per_clip"]:
        a = res["test"][b0k]["per_clip"][clip]
        b = res["test"][best_b1]["per_clip"][clip]
        A(
            f"| {clip} | {a['n_gt_ball']} | {fmt(a['recall@10'], 2)} | {fmt(a['precision@10'], 2)} | {fmt(b['recall@10'], 2)} | {fmt(b['precision@10'], 2)} |"
        )
    A("\n### Dev (contaminated, tuning set; optimistic)\n")
    A(HEAD)
    for name, s in res["dev"].items():
        A(row(name, s))
    A("\n## 4. B3: solver + human anchors (leave-anchors-out)\n")
    A(
        f"Anchors = every N-th labelled GT frame of the clip (hard constraints); scored only on the remaining "
        f"clean GT frames of clips with >= {MIN_GT_FOR_ANCHORS} clean GT frames: {', '.join(res['B3']['clips'])}.\n"
    )
    A(HEAD)
    for N in ANCHOR_EVERY:
        b = res["B3"][f"N={N}"]
        A(
            f"| **N={N}**: {b['n_anchors']} anchors, median spacing {b['median_anchor_spacing_frames']} frames | | | | | | | |"
        )
        for k, s in b["pooled"].items():
            A(row("  " + k, s))
    A("\n## 5. Failure analysis (test set, selected B1 and B0)\n")
    for m in ("B0", "B1"):
        A(f"**{m}**\n")
        for k, v in res["failures"][m].items():
            A(f"- {k}: {v}")
        A("")
    A("**B1 on dev (contaminated):**\n")
    for k, v in res["failures_dev"]["B1"].items():
        A(f"- {k}: {v}")
    A("\n## 6. Selected hyperparameters (dev)\n")
    A("```json\n" + json.dumps(res["B1"]["tuned_params"], indent=1) + "\n```\n")
    A(
        f"B0 tau (dev): {res['B0']['tau_selected_on_dev']}; B2 gate px: {res['B2']['gate_px_selected_on_dev']}.\n"
    )
    A("## 7. Caveats\n")
    for s in (
        "GT is sparse (a few percent of frames are labelled), so precision is measured only on labelled frames and a "
        "prediction on an unlabelled frame is never penalised; absolute precision is therefore optimistic for every method, "
        "including B1's interpolated points.",
        "Dev clips are detector-contaminated, so confidence calibration/thresholds tuned there are likely too trusting "
        "of confidence. Test numbers are the honest ones; test is small, hence the Wilson intervals.",
        "No homography; camera motion is a global translation estimated from player-box centres (see camera diagnostics in the JSON). "
        "Shot cuts are flagged heuristically (low support of the dominant player displacement) and unverified against the video.",
        "Calibration knots between the measured Ball Check bands are interpolated assumptions.",
    ):
        A(f"- {s}")
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-iter", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-json", default="docs/experiments/offline_ball_baseline.json")
    ap.add_argument("--out-md", default="docs/experiments/offline_ball_baseline.md")
    args = ap.parse_args()
    res = run(args)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(res, indent=1, default=float))
    write_md(res, Path(args.out_md))
    log(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
