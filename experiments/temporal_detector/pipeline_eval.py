"""Does the 5-channel detector's candidate set help once a trajectory solver does the ranking?  (CPU, numpy only)

Feeds per-frame candidate sets of RT-DETR-L (production), YOLO11s arm A (3ch) and arm B (5ch), seeds 0/1, into the offline
ball-tracking harness (`footy_track.ball_offline`, branch exp/offline-ball-tracking @ fc05732, imported read-only) with the
harness's selected solver parameters, and scores B0 (per-frame top-1, tau 0.4) and B1 (Viterbi + gap fill, camera on).

    uv run --no-project --python /usr/bin/python3 --with numpy python pipeline_eval.py \
        --harness-src <worktree>/src --clips-out /mnt/storage/exp/temporal/clips_out/v1 \
        --runs /mnt/storage/exp/temporal/runs/v1 --out docs/experiments/temporal_pipeline_eval

Calibration variants for the YOLO detectors: ``raw`` = the harness's RT-DETR calibration knots applied to the YOLO confidence
as is; ``iso`` = a per-model isotonic conf -> candidate-precision map fitted on the v11 *train* split only (candidates within
10 px of a labelled ball), with solver ``min_conf`` and the B0 threshold moved to the YOLO confidence that has the same
calibrated precision as the harness's 0.3 / 0.4 on RT-DETR. RT-DETR always uses the harness knots.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

HIT_PX = 10.0
N_BOOT = 10_000
# harness-selected parameters (docs/experiments/offline_ball_baseline.md section 6, "camera_on"), B0 tau (dev) = 0.4
CAMERA_ON = dict(min_conf=0.3, bias=1.0, emit_scale=1.0, kappa=1.5, lam_pos=0.5, lam_vel=0.0, vmax=40.0, r0=12.0, dmax=2.0,
                 mu=0.02, static_px=3.0, max_gap=25, camera=True, fill_gap=10, fill_min_p=0.8, fill_mode="kalman")
B0_TAU = 0.4
ARM_TAGS = {"A": ["A_s0", "A_s1"], "B": ["B_s0", "B_s1"]}  # round 2: Ap_s0 ... via --a-tags / --b-tags


# ------------------------------------------------------------------------------------------------ isotonic map
def pav_increasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: non-decreasing least squares fit of y (already ordered by x)."""
    vals, wts, idx = [], [], []
    for i, v in enumerate(y.astype(float)):
        vals.append(v)
        wts.append(1.0)
        idx.append([i])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w]
            wts[-2:] = [w]
            idx[-2:] = [idx[-2] + idx[-1]]
    out = np.empty(len(y))
    for v, ii in zip(vals, idx, strict=True):
        out[ii] = v
    return out


def fit_isotonic(conf: np.ndarray, correct: np.ndarray):
    """Returns f(conf) -> precision, monotone non-decreasing, piecewise-linear through the block means."""
    o = np.argsort(conf)
    c, y = conf[o], correct[o]
    fit = pav_increasing(y)
    # unique knots: mean conf of every constant block
    xs, ps = [], []
    i = 0
    while i < len(fit):
        j = i
        while j + 1 < len(fit) and fit[j + 1] == fit[i]:
            j += 1
        xs.append(float(c[i:j + 1].mean()))
        ps.append(float(fit[i]))
        i = j + 1
    xs, ps = np.array(xs), np.clip(np.array(ps), 1e-3, 0.999)
    return lambda q: np.interp(q, xs, ps), xs, ps


def conf_at_p(f, p_target: float) -> float:
    grid = np.linspace(0.1, 1.0, 901)
    ok = grid[f(grid) >= p_target]
    return float(ok[0]) if len(ok) else 1.01


def train_candidates(preds_json: Path):
    d = json.loads(preds_json.read_text())
    conf, correct = [], []
    for im in d["images"]:
        if not im["gt"]:
            continue
        g = np.array([[b[0], b[1]] for b in im["gt"]])
        for x, y, _, _, c in im["dets"]:
            if c < 0.1:
                continue
            conf.append(c)
            correct.append(float(np.min(np.hypot(g[:, 0] - x, g[:, 1] - y)) <= HIT_PX))
    return np.array(conf), np.array(correct)


# ------------------------------------------------------------------------------------------------ harness glue
def load_harness(src: Path):
    sys.path.insert(0, str(src))
    from footy_track.ball_offline import baselines as bl
    from footy_track.ball_offline import data as hd
    from footy_track.ball_offline import evaluate as he
    from footy_track.ball_offline import solver as hs

    return bl, hd, he, hs


def cands_from_npz(z, key: str):
    ptr, det = z[f"{key}_ptr"], z[f"{key}_det"].astype(float)
    n = int(z["n_frames"])
    rows = []
    for t in range(n):
        d = det[ptr[t]:ptr[t + 1]]
        d = d[np.argsort(-d[:, 4])] if len(d) else d
        rows.append(d)
    cnt = np.array([len(r) for r in rows])
    new_ptr = np.zeros(n + 1, np.int64)
    new_ptr[1:] = np.cumsum(cnt)
    arr = np.concatenate(rows, axis=0) if cnt.sum() else np.zeros((0, 5))
    return new_ptr, arr[:, 0:2].copy(), arr[:, 4].copy(), arr[:, 2:4].copy()


def with_cands(clip, z, key):
    ptr, xy, conf, wh = cands_from_npz(z, key)
    return dataclasses.replace(clip, ptr=ptr, xy=xy, conf=conf, wh=wh)


def frame_records(clip, scored: set[int], pred: dict, r: float = HIT_PX):
    """Per scored frame (GT ball frames then no_ball frames): (tp, fp, hit_cand) arrays in sorted frame order."""
    rows = []
    for t in sorted(scored):
        if t in clip.gt:
            gx, gy = clip.gt[t].xy
            p = pred.get(t)
            hit = p is not None and math.hypot(p[0] - gx, p[1] - gy) <= r
            xy, conf = clip.cands(t)
            ceil = bool(len(xy) and np.any((np.hypot(xy[:, 0] - gx, xy[:, 1] - gy) <= r) & (conf >= 0.1)))
            rows.append((t, "gt", int(hit), int(p is not None and not hit), int(ceil)))
        elif t in clip.no_ball:
            rows.append((t, "no_ball", 0, int(t in pred), 0))
    return rows


def boot_diff(fn_a, fn_b, n: int, rng) -> tuple[float, float, float]:
    """Paired bootstrap of stat_b - stat_a over frames; fn(idx) -> statistic from index array."""
    idx0 = np.arange(n)
    d0 = fn_b(idx0) - fn_a(idx0)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    ds = np.array([fn_b(i) - fn_a(i) for i in idx])
    ds = ds[~np.isnan(ds)]
    if not len(ds):
        return float(d0), float("nan"), float("nan")
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-src", type=Path, required=True)
    ap.add_argument("--clips-out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--a-tags", nargs="+", default=ARM_TAGS["A"])
    ap.add_argument("--b-tags", nargs="+", default=ARM_TAGS["B"])
    ap.add_argument("--title", default="# Temporal candidates through the trajectory solver: does 5-channel input help once a solver ranks?\n")
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/storage/footy_data"))
    a = ap.parse_args()
    ARM_TAGS["A"], ARM_TAGS["B"] = a.a_tags, a.b_tags
    bl, hd, he, hs = load_harness(a.harness_src)
    SolverParams = hs.SolverParams
    orig_cal = hs.calibrated_p

    # ---- clips: harness test clips + arsenal_example
    evs = he.build_sets(a.data_root)
    test = [ev for ev in evs if ev.role == "test"]
    assert len(test) == 9, [ev.clip.clip_id for ev in test]
    zs = {ev.clip.clip_id: np.load(a.clips_out / f"{ev.clip.clip_id}.npz") for ev in test}
    ex = "arsenal_example"
    zx = np.load(a.clips_out / f"{ex}.npz")
    n_ex = int(zx["n_frames"])
    pp, parr = zx["players_ptr"], zx["players_xywh"].astype(float)
    players = [parr[pp[t]:pp[t + 1]].reshape(-1, 4) for t in range(n_ex)]
    shift, cut = hd.estimate_camera_shift(players)
    gt, nb, aliases, conflicts = hd.load_gt(a.data_root, ex)
    z0 = cands_from_npz(zx, "ref")
    ex_clip = hd.ClipData(clip_id=ex, n_frames=n_ex, ptr=z0[0], xy=z0[1], conf=z0[2], wh=z0[3],
                          gt={t: g for t, g in gt.items() if t < n_ex}, no_ball={t for t in nb if t < n_ex},
                          cam_shift=shift, cam_cut=cut, players=players, gt_aliases=aliases, gt_conflicts=conflicts)
    rf = hd.roboflow_frames(a.data_root)
    bad = hd.contaminated_frames(ex, rf, near=he.NEAR)
    scored_all = set(ex_clip.gt) | ex_clip.no_ball
    ex_clean = scored_all - bad

    for ev in test:  # clip lengths (ffmpeg passthrough decode) agree with the harness' frame index space
        assert int(zs[ev.clip.clip_id]["n_frames"]) == ev.clip.n_frames, ev.clip.clip_id
    assert n_ex == 250

    # groups: (name, [(clip, scored set, candidate-source npz or None)])
    groups = {
        "test (9 clips, harness exclusions)": [(ev.clip, ev.clean_scored) for ev in test],
        "arsenal_example, harness-clean frames": [(ex_clip, ex_clean)],
        "arsenal_example, all labelled frames (NOT harness-clean)": [(ex_clip, scored_all)],
    }

    # ---- detectors

    def clip_for(det: str, clip):
        cid = clip.clip_id
        z = zx if cid == ex else zs[cid]
        if det == "RT-DETR (production)":
            return clip if cid != ex else with_cands(clip, zx, "ref")  # test clips: the harness's own machine labels
        if det == "RT-DETR (rerun)":
            return with_cands(clip, z, "ref")
        return with_cands(clip, z, det)

    # ---- isotonic maps (v11 train only)
    iso = {}
    for key in ARM_TAGS["A"] + ARM_TAGS["B"]:
        conf, corr = train_candidates(a.runs / f"preds_train_{key}.json")
        f, xs, ps = fit_isotonic(conf, corr)
        iso[key] = {"f": f, "n_cand": int(len(conf)), "n_correct": int(corr.sum()), "knots_conf": xs.tolist(), "knots_p": ps.tolist(),
                    "min_conf_equiv": conf_at_p(f, float(orig_cal(np.array([CAMERA_ON["min_conf"]]))[0])),
                    "b0_tau_equiv": conf_at_p(f, float(orig_cal(np.array([B0_TAU]))[0]))}

    def run(det: str, variant: str, clip, scored, method: str):
        c = clip_for(det, clip)
        if method == "B0":
            tau = B0_TAU
            if variant == "iso":
                tau = iso[det]["b0_tau_equiv"]
            return bl.b0_per_frame(c, tau)
        params = dict(CAMERA_ON)
        if variant == "iso":
            params["min_conf"] = iso[det]["min_conf_equiv"]
            hs.calibrated_p = lambda conf, f=iso[det]["f"]: f(conf)
        try:
            return bl.b1_solver(c, SolverParams(**params))[0]
        finally:
            hs.calibrated_p = orig_cal

    # ---- evaluate
    results: dict = {}
    recs: dict = {}  # (group, det, variant, method) -> array (n_frames, 3) tp, fp, ceil and kind
    detlist = ["RT-DETR (production)", "RT-DETR (rerun)", *ARM_TAGS["A"], *ARM_TAGS["B"]]
    for gname, parts in groups.items():
        for det in detlist:
            variants = ["raw", "iso"] if det[0] in "AB" else ["raw"]
            for variant in variants:
                for method in ("B0", "B1"):
                    rows = []
                    counts = he.Counts()
                    for clip, scored in parts:
                        pred = run(det, variant, clip, scored, method)
                        rows += [(clip.clip_id, *r) for r in frame_records(clip_for(det, clip), scored, pred)]
                        counts += he.score_clip(pred, {t: g.xy for t, g in clip.gt.items()}, clip.no_ball, clip.n_frames, scored=scored)
                    arr = np.array([[r[3], r[4], r[5]] for r in rows], float)
                    recs[(gname, det, variant, method)] = {"rows": [(r[0], r[1], r[2]) for r in rows], "arr": arr}
    # keys sanity: every detector has the same frame list in a group
    for gname in groups:
        base = recs[(gname, ARM_TAGS["A"][0], "raw", "B1")]["rows"]
        for k, v in recs.items():
            if k[0] == gname:
                assert v["rows"] == base, (k, "frame list differs")

    def metrics(arr, kinds):
        gt_m = kinds == "gt"
        tp, fp = arr[:, 0].sum(), arr[:, 1].sum()
        return {"n_gt": int(gt_m.sum()), "tp": int(tp), "fp": int(fp), "recall10": float(arr[gt_m, 0].mean()) if gt_m.any() else float("nan"),
                "precision10": float(tp / (tp + fp)) if tp + fp else float("nan"),
                "ceiling10": float(arr[gt_m, 2].mean()) if gt_m.any() else float("nan")}

    table: dict = {}
    for gname in groups:
        kinds = np.array([r[2] for r in recs[(gname, ARM_TAGS["A"][0], "raw", "B1")]["rows"]])
        table[gname] = {}
        for det in detlist:
            for variant in (["raw", "iso"] if det[0] in "AB" else ["raw"]):
                for method in ("B0", "B1"):
                    arr = recs[(gname, det, variant, method)]["arr"]
                    table[gname][f"{det}|{variant}|{method}"] = metrics(arr, kinds)
        # seed means per arm
        for arm in "AB":
            for variant in ("raw", "iso"):
                for method in ("B0", "B1"):
                    ms = [table[gname][f"{t}|{variant}|{method}"] for t in ARM_TAGS[arm]]
                    table[gname][f"{arm} (mean of seeds)|{variant}|{method}"] = {
                        k: float(np.mean([m[k] for m in ms])) for k in ("recall10", "precision10", "ceiling10")} | {"n_gt": ms[0]["n_gt"]}

    # ---- paired bootstrap
    rng = np.random.default_rng(0)
    paired: dict = {}

    def arm_arr(gname, arm, variant, method):
        return np.mean([recs[(gname, t, variant, method)]["arr"] for t in ARM_TAGS[arm]], axis=0)

    for gname in groups:
        kinds = np.array([r[2] for r in recs[(gname, ARM_TAGS["A"][0], "raw", "B1")]["rows"]])
        n = len(kinds)
        gt_m = kinds == "gt"
        paired[gname] = {}
        for variant in ("raw", "iso"):
            for method in ("B1", "B0"):
                A3 = arm_arr(gname, "A", variant, method)
                B5 = arm_arr(gname, "B", variant, method)
                R = recs[(gname, "RT-DETR (production)", "raw", method)]["arr"]

                def stat(arr, which):
                    def f(idx):
                        x, k = arr[idx], gt_m[idx]
                        if which == "recall10":
                            return x[k, 0].mean() if k.any() else np.nan
                        if which == "ceiling10":
                            return x[k, 2].mean() if k.any() else np.nan
                        tp, fp = x[:, 0].sum(), x[:, 1].sum()
                        return tp / (tp + fp) if tp + fp else np.nan
                    return f
                for which in ("recall10", "precision10", "ceiling10"):
                    paired[gname][f"{method}|{variant}|5ch-3ch|{which}"] = boot_diff(stat(A3, which), stat(B5, which), n, rng)
                    paired[gname][f"{method}|{variant}|5ch-RT-DETR|{which}"] = boot_diff(stat(R, which), stat(B5, which), n, rng)
                # per seed 5ch-3ch recall
                paired[gname][f"{method}|{variant}|5ch-3ch|recall10|per_seed"] = {
                    str(i): float(recs[(gname, tb_, variant, method)]["arr"][gt_m, 0].mean()
                                  - recs[(gname, ta_, variant, method)]["arr"][gt_m, 0].mean())
                    for i, (ta_, tb_) in enumerate(zip(ARM_TAGS["A"], ARM_TAGS["B"], strict=True))}

    # ---- sanity: reproduce the harness numbers; RT-DETR rerun vs machine labels
    chk = {}
    gname = "test (9 clips, harness exclusions)"
    chk["harness_repro_B1_camera_on"] = table[gname]["RT-DETR (production)|raw|B1"]
    chk["harness_repro_B0_tau0.4"] = table[gname]["RT-DETR (production)|raw|B0"]
    chk["rtdetr_rerun_B1"] = table[gname]["RT-DETR (rerun)|raw|B1"]
    agree = []
    for ev in test:
        z = zs[ev.clip.clip_id]
        c0, c1 = ev.clip, with_cands(ev.clip, z, "ref")
        both = n_match = 0
        for t in range(c0.n_frames):
            x0, f0 = c0.cands(t)
            x1, f1 = c1.cands(t)
            if len(f0) and f0.max() >= 0.3:
                both += 1
                k1 = np.argmax(f1) if len(f1) else None
                if k1 is not None and math.hypot(*(x0[np.argmax(f0)] - x1[k1])) <= 3 and abs(f0.max() - f1.max()) < 0.05:
                    n_match += 1
        agree.append({"clip": ev.clip.clip_id, "frames_with_conf>=0.3_top1_in_machine_labels": both, "rerun_top1_same_pos_and_conf": n_match})
    chk["rtdetr_rerun_vs_machine_labels"] = agree
    chk["isotonic"] = {k: {kk: vv for kk, vv in v.items() if kk != "f"} for k, v in iso.items()}
    chk["status_counts"] = {c: {s: int((zz["status"] == i).sum()) for s, i in (("ok", 0), ("edge", 1), ("cut", 2), ("align_fail", 3))}
                            for c, zz in {**zs, ex: zx}.items()}
    out = {"table": table, "paired": paired, "checks": chk, "params_camera_on": CAMERA_ON, "b0_tau": B0_TAU,
           "n_frames_per_group": {g: len(next(v["rows"] for k, v in recs.items() if k[0] == g)) for g in groups}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".json").write_text(json.dumps(out, indent=1, default=lambda o: None if o != o else float(o)))
    write_md(out, a, groups)
    print("wrote", str(a.out) + ".md")


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def write_md(out, a, groups):
    L = [a.title]
    if a.meta:
        L.append(a.meta + "\n")
    labels = [("RT-DETR (production)", "RT-DETR-L production (frame t)"), ("RT-DETR (rerun)", "RT-DETR-L rerun by this pipeline"),
              *[(t, f"A 3ch {t}") for t in ARM_TAGS["A"]], ("A (mean of seeds)", "A 3ch mean of seeds"),
              *[(t, f"B 5ch {t}") for t in ARM_TAGS["B"]], ("B (mean of seeds)", "B 5ch mean of seeds")]
    for gname in groups:
        n = out["n_frames_per_group"][gname]
        tb = out["table"][gname]
        L.append(f"## {gname}\n")
        L.append(f"{n} scored frames; n_gt = {tb[ARM_TAGS['A'][0] + '|raw|B1']['n_gt']}. recall@10 = emitted point within 10 px of GT (denominator: GT ball frames); "
                 "precision = TP / (TP + FP) over scored frames (FP = emitted point > 10 px away, or emitted on a `no_ball` frame); "
                 "ceiling = any candidate with conf >= 0.1 within 10 px (candidate-set recall, identical for B0 and B1).\n")
        L.append("| detector | calibration | B0 recall / prec | B1 recall / prec | ceiling |")
        L.append("|---|---|---|---|---|")
        for key, name in labels:
            for variant in (["raw", "iso"] if key[0] in "AB" and not key.startswith("RT") else ["raw"]):
                b0, b1 = tb[f"{key}|{variant}|B0"], tb[f"{key}|{variant}|B1"]
                L.append(f"| {name} | {variant} | {fmt(b0['recall10'])} / {fmt(b0['precision10'])} | {fmt(b1['recall10'])} / {fmt(b1['precision10'])} | {fmt(b1['ceiling10'])} |")
        L.append("")
        L.append("Paired bootstrap (10,000 resamples of scored frames; seed-mean per-frame outcomes for the arms), B1 camera on:\n")
        L.append("| comparison | calibration | recall@10 diff [95% CI] | precision diff [95% CI] | ceiling diff [95% CI] | per-seed recall diff |")
        L.append("|---|---|---|---|---|---|")
        pr = out["paired"][gname]
        for cmp_, nm in (("5ch-3ch", "B1(5ch) - B1(3ch)"), ("5ch-RT-DETR", "B1(5ch) - B1(RT-DETR)")):
            for variant in ("raw", "iso"):
                cells = []
                for w in ("recall10", "precision10", "ceiling10"):
                    d = pr[f"B1|{variant}|{cmp_}|{w}"]
                    cells.append(f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
                ps = pr.get(f"B1|{variant}|{cmp_}|recall10|per_seed") if cmp_ == "5ch-3ch" else None
                L.append(f"| {nm} | {variant} | " + " | ".join(cells) + f" | {', '.join(f's{k}: {v:+.3f}' for k, v in ps.items()) if ps else ''} |")
        L.append("")
    L.append("## Checks\n")
    c = out["checks"]
    L.append(f"- Harness reproduction on the 9 test clips (RT-DETR machine labels, camera-on B1): recall "
             f"{c['harness_repro_B1_camera_on']['recall10']:.3f} ({c['harness_repro_B1_camera_on']['tp']} TP of {c['harness_repro_B1_camera_on']['n_gt']}), precision "
             f"{c['harness_repro_B1_camera_on']['precision10']:.3f}; B0 tau 0.4 recall {c['harness_repro_B0_tau0.4']['recall10']:.3f}, precision "
             f"{c['harness_repro_B0_tau0.4']['precision10']:.3f} (harness doc: 0.759 / 0.985 and 0.741 / 0.947).")
    L.append("- RT-DETR rerun by this pipeline vs the harness's machine labels (top-1 per frame with conf >= 0.3, same position within 3 px and conf within 0.05): "
             + "; ".join(f"{x['clip'].replace('arsenal_mancity_', '')} {x['rerun_top1_same_pos_and_conf']}/{x['frames_with_conf>=0.3_top1_in_machine_labels']}" for x in c["rtdetr_rerun_vs_machine_labels"]) + ".")
    L.append("- Neighbour status per clip (frames, [prev, next] slots): " + "; ".join(
        f"{k.replace('arsenal_mancity_', '')} " + "/".join(f"{s}:{v}" for s, v in d.items() if v) for k, d in c["status_counts"].items()) + ".")
    L.append("- Isotonic maps (fitted on v11 train candidates within 10 px of a labelled ball): " + "; ".join(
        f"{k}: {v['n_correct']}/{v['n_cand']} correct, min_conf {v['min_conf_equiv']:.2f}, B0 tau {v['b0_tau_equiv']:.2f}" for k, v in c["isotonic"].items()) + ".")
    L.append("\nSolver parameters (harness `camera_on`): `" + json.dumps(out["params_camera_on"]) + f"`, B0 tau {out['b0_tau']}.")
    Path(str(a.out) + ".md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
