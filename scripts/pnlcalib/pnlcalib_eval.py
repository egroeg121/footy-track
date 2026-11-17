"""Evaluate PnLCalib zero-shot output against the calibration GT, and as a keyframe source for ``calibrate_clip``.

    PYTHONPATH=src:scripts/pnlcalib python scripts/pnlcalib/pnlcalib_eval.py --results RUN/results.jsonl --out OUT \
        [--cache-dir DIR]   # masks/motion npz of seg010 / seg020 (projection_eval layout) are reused when present

Stages (all CPU, run on the mini under nice, one process):
  1. per-GT-frame error of every PnLCalib variant (metres; grass lattice -> pitch through both H, near/far halves, same as
     ``projection_eval``) + the sign convention (``fit_flip``) + the existing Claude proposals on the same frames;
  2. PnLCalib keyframes (gated by ``line_cost``) -> ``calibrate_clip`` on seg010 / seg020, held-out error on the GT frames;
  3. ``refine_homography`` started from PnLCalib's H.
Everything is written to ``OUT`` (``stage1.json``, ``stage2.json``, ``stage3.json``, ``report.md``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration import temporal as tp
from footy_track.calibration.clip_calibration import MaskStore, Policy, calibrate_clip, line_cost
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.pipeline import find_video
from footy_track.calibration.prelabel.proposals import read_proposals
from footy_track.calibration.refine import model_samples
from footy_track.clip_id import canonical_clip_id
from footy_track.scripts.calibrate_clip import CLIPS_DIR, GT_DIR, load_gt_keyframes
from footy_track.scripts.projection_eval import err_stats, grass_points, halves
from pnlcalib_lib import FLIPS, load_results, mean_err_m, variant_H

VARIANTS = ("cam", "cam_pnl", "gh", "gh_pnl")
SWEEP_CLIPS = ("arsenal_mancity_seg010", "arsenal_mancity_seg020")
PITCH = PitchModel()


def get_masks(clip: str, frames, cache_dir: Path | None, out: Path) -> MaskStore:
    """Masks of ``frames`` of ``clip`` (cached under ``out`` and read from ``cache_dir`` when present)."""
    p = out / f"{clip}.masks.npz"
    pol = Policy()
    for cand in (p, (cache_dir / f"{clip}.masks.npz") if cache_dir else None):
        if cand and cand.exists():
            ms = MaskStore.load(cand)
            break
    else:
        ms = MaskStore(pol.work_width)
    n0 = len(ms)
    ms.ensure(find_video(CLIPS_DIR, clip), set(frames))
    if len(ms) != n0 or not p.exists():
        ms.save(p)
    return ms


def fit_flip(res: dict, gt_all: dict, var: str) -> tuple[str, dict]:
    """Which (x, y) sign convention maps PnLCalib's centred pitch to ours: lowest median error on the GT frames."""
    sc = {}
    for name, flip in FLIPS.items():
        es = []
        for (clip, f), r in res.items():
            if (clip, f) not in gt_all:
                continue
            H = variant_H(r["variants"].get(var), r["W"], r["H"], flip)
            if H is None:
                continue
            Hg = gt_all[(clip, f)]
            g = np.linspace(0.1, 0.9, 7)
            pts = np.array([[a, b] for a in g for b in np.linspace(0.3, 0.9, 5)])
            es.append(mean_err_m(H, Hg, pts)[0])
        sc[name] = {"median": float(np.nanmedian(es)) if es else None, "n": len(es)}
    best = min(sc, key=lambda k: sc[k]["median"] if sc[k]["median"] is not None else 1e9)
    return best, sc


def claude_proposals(clip: str, frame: int) -> list[dict]:
    out = []
    for run_id, run in read_proposals(Path(GT_DIR), clip).get("runs", {}).items():
        fr = run.get("frames", {}).get(str(frame))
        if fr and fr.get("H") is not None:
            out.append({"run": run_id, "status": fr.get("status"), "H": np.array(fr["H"], float), "dist": fr.get("distortion")})
    return out


def stage1(res, gt_all, gt_dist, masks_by_clip, flip, out: Path) -> dict:
    rows = []
    pooled = {v: {"e": [], "y": []} for v in VARIANTS}
    pooled["claude_edited"] = {"e": [], "y": []}
    for (clip, f), Hg in sorted(gt_all.items()):
        r = res.get((clip, f))
        pts = grass_points(masks_by_clip[clip], f)
        row = {"clip": clip, "frame": f, "n_pts": len(pts)}
        if r is not None:
            row["t_net"] = r["t_net"]
            row["n_kp"] = r["n_kp"]
            row["n_lines"] = r["n_lines"]
        for v in VARIANTS:
            H = variant_H(r["variants"].get(v), r["W"], r["H"], flip) if r else None
            m, e = mean_err_m(H, Hg, pts)
            row[v] = {"mean_err_m": m if np.isfinite(m) else None, "rep_err_px": (r["variants"].get(v) or {}).get("rep_err") if r else None,
                      "t": (r["variants"].get(v) or {}).get("t") if r else None}
            if H is not None:
                pooled[v]["e"] += list(e)
                pooled[v]["y"] += list(pts[:, 1])
        cp = []
        for p in claude_proposals(clip, f):
            m, _ = mean_err_m(p["H"], Hg, pts, p["dist"])
            cp.append({"run": p["run"], "status": p["status"], "mean_err_m": m})
        row["claude"] = cp
        rows.append(row)
    summary = {}
    for v in VARIANTS:
        errs = np.array([row[v]["mean_err_m"] if row[v]["mean_err_m"] is not None else np.nan for row in rows])
        fails = np.isnan(errs)
        n = len(errs)
        e_ok = errs[~fails]
        pe, py = np.array(pooled[v]["e"]), np.array(pooled[v]["y"])
        summary[v] = {"n_frames": n, "failures": int(fails.sum()), "failure_rate": float(fails.mean()),
                      "frames_lt1m": int((errs < 1).sum()), "frac_lt1m": float((errs < 1).sum() / n),
                      "frames_lt2m": int((errs < 2).sum()), "frac_lt2m": float((errs < 2).sum() / n),
                      "frames_gt5m": int((errs > 5).sum()),
                      "frame_mean_of_means_m": float(e_ok.mean()) if len(e_ok) else None, "frame_median_m": float(np.median(e_ok)) if len(e_ok) else None,
                      "frame_p90_m": float(np.percentile(e_ok, 90)) if len(e_ok) else None,
                      "pooled_points": halves(pe, py)}
    # Claude proposals on the same frames, by status
    by_status: dict[str, list[float]] = {}
    for row in rows:
        for c in row["claude"]:
            if c["mean_err_m"] is not None and np.isfinite(c["mean_err_m"]):
                by_status.setdefault(c["status"], []).append(c["mean_err_m"])
    summary["claude_by_status"] = {s: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)), "frac_lt1m": float(np.mean(np.array(v) < 1)),
                                       "frac_lt2m": float(np.mean(np.array(v) < 2))} for s, v in by_status.items()}
    (out / "stage1.json").write_text(json.dumps({"flip": flip, "frames": rows, "summary": summary}, indent=1, default=float))
    return {"rows": rows, "summary": summary}


def held_out(cal, gt_all, clip, masks, skip=()) -> dict:
    """calibrate_clip result vs the GT frames of ``clip`` (all of them: PnLCalib keyframes are not GT). Same lattice / halves as projection_eval."""
    e_all, y_all, e_ok, y_ok, per = [], [], [], [], []
    for (c, f), Hg in sorted(gt_all.items()):
        if c != clip or f in skip:
            continue
        pts = grass_points(masks, f)
        ref = geo.image_to_pitch(Hg, pts)[0]
        He, row = cal.H_any(f), cal.row(f)
        d = {"frame": f, "ok": bool(row["ok"]), "source": row["source"], "line_cost": float(row["line_cost"]), "key_frame": int(row["key_frame"])}
        if He is not None:
            e = np.linalg.norm(geo.image_to_pitch(He, pts, cal.distortion(f))[0] - ref, axis=1)
            e_all += list(e); y_all += list(pts[:, 1])
            d["mean_err_m"] = float(e.mean())
            if row["ok"]:
                e_ok += list(e); y_ok += list(pts[:, 1])
        per.append(d)
    A = np.asarray
    return {"n_gt": len(per), "calib_all": halves(A(e_all, float), A(y_all, float)), "calib_ok": halves(A(e_ok, float), A(y_ok, float)), "per_frame": per}


def load_motion(clip: str, cache_dir: Path | None):
    for d in (cache_dir,):
        if d and (d / f"{clip}.motion.npz").exists():
            return tp.ClipMotion.load(d / f"{clip}.motion.npz")
    return tp.clip_motion(find_video(CLIPS_DIR, clip), boxes=None, mask_mode="all")


def pnl_candidates(res, clip, var, flip, masks, samples) -> list[dict]:
    """Every swept frame of ``clip`` with an H from PnLCalib and its line cost on the frame's own masks."""
    out = []
    for (c, f), r in sorted(res.items()):
        if c != clip or f not in masks:
            continue
        H = variant_H(r["variants"].get(var), r["W"], r["H"], flip)
        cost = line_cost(H, masks.get(f), samples, 0.0) if H is not None else float("nan")
        out.append({"frame": f, "H": H, "cost": cost, "rep_err": (r["variants"].get(var) or {}).get("rep_err")})
    return out


def scenarios(cands: list[dict], shot_of) -> dict[str, list[int]]:
    """name -> keyframe list. Gate: line cost (px at 1280 wide) <= tau, finite."""
    ok = lambda c, tau: c["H"] is not None and np.isfinite(c["cost"]) and c["cost"] <= tau  # noqa: E731
    out: dict[str, list[int]] = {}
    for tau in (8.0, 12.0):
        good = [c for c in cands if ok(c, tau)]
        best: dict[int, dict] = {}
        for c in good:
            s = shot_of(c["frame"])
            if s not in best or c["cost"] < best[s]["cost"]:
                best[s] = c
        if best:
            out[f"best1_per_shot_cost<={tau:g}"] = sorted(c["frame"] for c in best.values())
        for n in (25, 50, 100):
            ks = [c["frame"] for c in good if c["frame"] % n == 0]
            if ks:
                out[f"every{n}_cost<={tau:g}"] = ks
    return out


def stage2(res, gt_all, var, flip, cache_dir, out: Path, masks_by_clip) -> dict:
    s2 = {}
    samples = model_samples(PITCH)
    for clip in SWEEP_CLIPS:
        video = find_video(CLIPS_DIR, clip)
        motion = load_motion(clip, cache_dir)
        masks = masks_by_clip[clip]
        cands = pnl_candidates(res, clip, var, flip, masks, samples)
        costs = np.array([c["cost"] for c in cands if np.isfinite(c["cost"])])
        shot = motion.shot
        sc = scenarios(cands, lambda f: int(shot[f - motion.start]))
        s2[clip] = {"n_swept": len(cands), "n_with_H": sum(c["H"] is not None for c in cands), "n_cost_finite": int(len(costs)),
                    "cost_quantiles": [float(np.percentile(costs, q)) for q in (0, 10, 50, 90, 100)] if len(costs) else None,
                    "n_shots": int(len(set(int(s) for s in shot))), "scenarios": {}}
        byf = {c["frame"]: c for c in cands}
        for name, keys in sc.items():
            for refine in (True,):
                t0 = time.perf_counter()
                cal = calibrate_clip(video, {k: (byf[k]["H"], None) for k in keys}, None, Policy(refine=refine), motion=motion, masks=masks)
                ho = held_out(cal, gt_all, clip, masks)
                s2[clip]["scenarios"][name] = {"keys": keys, "n_keys": len(keys), "refine": refine, "s": time.perf_counter() - t0,
                                               "ok_frac": float(cal.table["ok"].mean()), "held_out": ho,
                                               "key_costs": [byf[k]["cost"] for k in keys]}
                a = ho["calib_all"]["all"]
                print(f"{clip} {name}: keys={len(keys)} n_gt={ho['n_gt']} mean={a.get('mean', float('nan')):.2f} lt1={a.get('lt1', float('nan')):.2f} ok={cal.table['ok'].mean():.2f}", flush=True)
        (out / "stage2.json").write_text(json.dumps(s2, indent=1, default=float))
    return s2


def stage3(res, gt_all, var, flip, masks_by_clip, out: Path) -> dict:
    rows = []
    for (clip, f), Hg in sorted(gt_all.items()):
        r = res.get((clip, f))
        H = variant_H(r["variants"].get(var), r["W"], r["H"], flip) if r else None
        if H is None:
            continue
        pts = grass_points(masks_by_clip[clip], f)
        m = masks_by_clip[clip].get(f)
        samples = model_samples(PITCH)
        e0 = mean_err_m(H, Hg, pts)[0]
        c0 = line_cost(H, m, samples, 0.0)
        t0 = time.perf_counter()
        rr = tp.refine_frame(H, m, PITCH)
        dt = time.perf_counter() - t0
        H1 = geo.orient_homography(rr.H, np.array([[0.5, 0.5]]))
        e1 = mean_err_m(H1, Hg, pts)[0]
        c1 = line_cost(H1, m, samples, 0.0)
        rows.append({"clip": clip, "frame": f, "err_before": e0, "err_after": e1, "cost_before": c0, "cost_after": c1, "applied": bool(rr.applied), "s": dt})
        print(f"refine {clip}:{f} {e0:.2f} -> {e1:.2f} m (cost {c0:.1f} -> {c1:.1f}, applied={rr.applied})", flush=True)
    (out / "stage3.json").write_text(json.dumps(rows, indent=1, default=float))
    return {"rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--stage", default="1,2,3")
    ap.add_argument("--variant", default=None, help="variant used for stages 2/3 (default: best mean error of stage 1)")
    ap.add_argument("--flip", default=None, help="override the fitted sign convention, e.g. +x-y")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = load_results(a.results)
    gt_all = {}
    for p in sorted((GT_DIR / "calibration_gt").glob("*.json")):
        for f, (H, d) in load_gt_keyframes(p.stem).items():
            gt_all[(p.stem, f)] = H
    # masks at the GT frames of every clip; the sweep clips also need the check frames (cached)
    masks_by_clip = {}
    for clip in sorted({c for c, _ in gt_all}):
        frames = {f for (c, f) in gt_all if c == clip}
        if clip in SWEEP_CLIPS:
            frames |= {f for (c, f) in res if c == clip}
        masks_by_clip[clip] = get_masks(clip, frames, a.cache_dir, a.out)
    ref_var = a.variant or "cam_pnl"
    flip_name, scores = fit_flip(res, gt_all, ref_var)
    if a.flip:
        flip_name = a.flip
    flip = FLIPS[flip_name]
    print("flip", flip_name, scores, flush=True)
    (a.out / "flip.json").write_text(json.dumps({"chosen": flip_name, "scores": scores}))
    s1 = stage1(res, gt_all, None, masks_by_clip, flip, a.out)
    for v in VARIANTS:
        s = s1["summary"][v]
        print(v, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items() if k != "pooled_points"}, flush=True)
    var = a.variant or min(VARIANTS, key=lambda v: s1["summary"][v]["frame_mean_of_means_m"] if s1["summary"][v]["frame_mean_of_means_m"] is not None else 1e9)
    print("variant for stage 2/3:", var, flush=True)
    if "2" in a.stage:
        stage2(res, gt_all, var, flip, a.cache_dir, a.out, masks_by_clip)
    if "3" in a.stage:
        stage3(res, gt_all, var, flip, masks_by_clip, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
