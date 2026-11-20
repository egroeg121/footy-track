"""PTZ V3 temporal calibration of ONE broadcast segment mp4 (production run: no ground truth, no eval).

    python -m footy_track.scripts.ptz_temporal_segment \
        --video /.../arsenal_mancity_20250925_seg010.mp4 --clip arsenal_mancity_seg010 \
        --rig-json src/footy_track/calibration/camera_rigs/emirates.json --out-root /scratch/temporal

Thin wrapper over the research driver :mod:`footy_track.scripts.ptz_temporal` (``prep`` + ``solve --variants full`` with
``--video`` / ``--rig``): the same ``load_clip`` -> ``solve_variant("full")`` -> ``export_calib`` chain, in-process, plus

* no GT lookups (``clip_gt`` returns nothing: the GT directory does not exist on the desktop);
* the anchor step shrinks for short segments so every segment gets a few anchor attempts;
* ``line_f1_full.json``: line-evidence F1 of the SOLVED camera on every evidence frame, and the lines-only search F1 /
  acceptance of every anchor, so a consumer can tell confident frames from uncertain ones without GT;
* ``meta.json`` and ``solve_full.json``; every file is written atomically.

**Recovery pass** (``--no-recover`` skips it): when the standard pass accepts < 50% of the frames, the run is repeated on a
segment with anchors from a pose-band, multi-start lines-only search (and, in a second stage,
the centre-circle / penalty-D finders) that other candidates must confirm along the motion chain
(:mod:`footy_track.calibration.ptz_recover`). The recovered result replaces the standard one only if it accepts more
frames, and then an evidence frame must also sit on the rig's pose band; ``meta.json["recovery"]`` records both passes
and ``anchors_rec.json`` the candidates. Segments the standard pass accepts are untouched.

Work directory: ``<out-root>/<clip>@<rig stem>/`` (the driver's convention: anchors and motion depend on the rig).
Output (small): ``calib_full.npz`` (frame, pan, tilt, logf, H (N,3,3) normalised undistorted image -> pitch m, rig),
``states_full.npy``, ``line_f1_full.json``, ``solve_full.json``, ``meta.json``, ``anchors_lines.json``,
``anchors_refit.json``, ``pairs_g1_5.pkl``. **``calib_full.npz`` is written LAST: its presence means the segment is done.**
Caches (grey frames, masks, anchors, pairs) make a re-run resume. With ``--estimate-rig`` (default off; the outputs above are then
bit-identical to a run without the flag) the segment also writes ``rig_estimate.json`` BEFORE the done marker: the level-1 rig
estimate (C, roll) with its 4x4 covariance, see :func:`solve_rig_estimate` and ``docs/design/calibration_match_flow.md`` section 2.1. k1 stays fixed at the rig value (it is poorly
constrained by the lines; see reports/ptz_v3.md).
"""

# ruff: noqa: PLR0915
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from footy_track.calibration import ptz_recover as pr
from footy_track.calibration import ptz_temporal as pt
from footy_track.scripts import ptz_temporal as pte

MIN_ANCHOR_STEP = 25
RECOVER_BELOW = 0.5  # run the recovery pass when the standard pass accepts less than this fraction of the frames
REC_PROBE = 4  # probe frames per camera position when choosing it
REC_FRAMES = 12  # candidate frames (lines-only search + finders) per segment in recovery stage A; stage B doubles them
F1_ACCEPT = 0.5  # the driver's acceptance threshold for an evidence frame (cmd_eval coverage)


def anchor_step_for(n_frames: int, wanted: int) -> int:
    """``wanted`` (100 in the report), but small enough for >= ~3 anchor attempts in a short segment (a multiple of EV_STEP)."""
    third = int(round(n_frames / 3 / pte.EV_STEP)) * pte.EV_STEP
    return int(max(pte.EV_STEP, min(wanted, max(MIN_ANCHOR_STEP, third))))


def _git_sha() -> str | None:
    if os.environ.get("PTZ_CODE_SHA"):
        return os.environ["PTZ_CODE_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 (a git archive export has no .git)
        return None


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _ranges(mask: np.ndarray) -> list[list[int]]:
    """Runs of True as [start, end] inclusive."""
    out, a = [], None
    for t, m in enumerate(mask):
        if m and a is None:
            a = t
        if not m and a is not None:
            out.append([a, t - 1])
            a = None
    if a is not None:
        out.append([a, len(mask) - 1])
    return out


def line_f1_report(cache, states, shared, anchors, segs, prior: dict | None = None) -> dict:
    """F1 of the solved camera against the line evidence of every evidence frame (every 5th frame), the anchors' search F1,
    and ``accepted_ranges``: frames whose two neighbouring evidence frames both reach ``F1_ACCEPT`` (the driver's coverage rule).
    With a ``prior`` (recovery pass) an evidence frame also has to sit on the rig's pose band (:func:`pr.in_prior`)."""
    ev = list(cache.ev_frames)
    f1 = pte.line_f1(cache, states, shared, ev)
    good = {t: f1.get(t, -1.0) >= F1_ACCEPT and (prior is None or pr.in_prior(prior, *states[t])) for t in ev}
    acc = np.zeros(cache.n, bool)
    for t in range(cache.n):
        prev = max((e for e in ev if e <= t), default=None)
        nxt = min((e for e in ev if e >= t), default=None)
        acc[t] = all(good[e] for e in (prev, nxt) if e is not None) and bool(np.all(np.isfinite(states[t])))
    anc = {t: {"ok": bool(a.get("ok")), "search_f1": a.get("f1"), "solved_f1": f1.get(int(t))} for t, a in anchors.items()}
    vals = list(f1.values())
    return {"threshold": F1_ACCEPT, "evidence_f1": {str(t): round(v, 4) for t, v in f1.items()}, "anchors": anc,
            "accepted_ranges": _ranges(acc), "accepted_frac_of_frames": float(acc.mean()) if cache.n else 0.0,
            "median_evidence_f1": float(np.median(vals)) if vals else None,
            "main_camera_runs": [list(map(int, s)) for s in segs]}


def candidate_frames(cache, runs: list[tuple[int, int]], n: int) -> list[int]:
    """``n`` evidence frames spread evenly over the runs (all of them when there are fewer)."""
    ev = [t for t in cache.ev_frames if any(a <= t <= b for a, b in runs)]
    if len(ev) <= n:
        return ev
    return [ev[i] for i in np.unique(np.linspace(0, len(ev) - 1, n).round().astype(int))]


def usable_candidates(cands: dict, frames: list[int]) -> dict[int, dict]:
    """Best usable record per frame: on the pose band and a relaxed F1 with C at the rig (what the joint solve uses)."""
    out = {}
    for t in frames:
        recs = [r for r in (cands.get(str(t)) or {}).values() if r and r["prior_ok"] and r["f1_fixed"] >= pr.F1_CAND]
        if recs:
            out[t] = max(recs, key=lambda r: r["f1_fixed"])
    return out


def _ensure_candidates(cache, rig_v, prior, pitch, cands: dict, frames: list[int], cp: Path) -> None:
    """Lines-only (multi-start, prior band) and finder candidates of ``frames`` for one rig variant, cached in ``cands``."""
    for t in frames:
        rec = cands.setdefault(str(t), {})
        ev = cache.evidence(t)
        if ev is None:
            continue
        if "lines" not in rec:
            rec["lines"] = pr.lines_candidate(ev, rig_v, pitch, prior)
            cp.write_text(json.dumps(cands, default=float))
        if "finder" not in rec:
            rec["finder"] = pr.finder_anchor(cache, t, rig_v, pitch, prior)
            cp.write_text(json.dumps(cands, default=float))


def recover_pass(cache, rig, shot_info, out: Path, trim: bool = False) -> dict:
    """Recovery of a segment the standard pass left uncalibrated (see :mod:`ptz_recover`). Trim non-pitch tails; choose the
    camera position (``pr.rig_variants``: the stand's centre position or one of the two penalty-area-line positions) from a
    few probe frames; then stage A: multi-start prior-band lines-only and finder candidates on ``REC_FRAMES`` frames,
    motion confirmation, joint solve; stage B (only when stage A still accepts < ``RECOVER_BELOW``): twice the frames.
    Returns the better stage: ``{"states", "infos", "segs", "anchors", "f1rep", "stages", "runs", "rig", "variant"}``."""
    prior = pr.pose_prior(rig)
    pitch = pte.PITCH
    pm = {t: pt.pmain_of(cache, t) for t in cache.ev_frames}
    base_runs = [(x["start"], x["end"]) for x in shot_info if x["p_main"] >= 0.5]
    # trimming is opt-in: cutting where the main-camera probability drops also cut valid zoomed goal-area views (Man City seg108
    # f130-174: 100% accepted untrimmed, 75% trimmed; seg073 80% vs 72%), and the F1 acceptance already rejects real tails
    runs = pr.trim_runs(base_runs, cache, pm) if trim else base_runs
    pte.log(f"recovery: main shots {base_runs} -> runs {runs}")
    cp = out / "anchors_rec.json"
    allc = json.loads(cp.read_text()) if cp.exists() else {}
    # --- which camera position: best mean candidate F1 over a few probe frames (the default wins ties)
    variants = pr.rig_variants(rig)
    probe = candidate_frames(cache, runs, REC_PROBE)
    vscore = {}
    for name, rig_v in variants:
        if len(variants) > 1:
            _ensure_candidates(cache, rig_v, prior, pitch, allc.setdefault(name, {}), probe, cp)
            use = usable_candidates(allc[name], probe)
            vscore[name] = sum(r["f1_fixed"] for r in use.values()) / max(len(probe), 1)
    vname, rig_v = variants[0]
    if len(variants) > 1:
        vname = max(vscore, key=lambda k: (vscore[k], k == variants[0][0]))
        rig_v = dict(variants)[vname]
        pte.log(f"recovery: camera position scores {({k: round(v, 3) for k, v in vscore.items()})} -> {vname}")
    sh = pt.Shared.from_rig(rig_v)
    cands = allc.setdefault(vname, {})
    best, stages = None, []
    for stage, n_fr in enumerate((REC_FRAMES, 2 * REC_FRAMES)):
        frames = candidate_frames(cache, runs, n_fr)
        _ensure_candidates(cache, rig_v, prior, pitch, cands, frames, cp)
        use = usable_candidates(cands, frames)
        pte.log(f"recovery stage {'AB'[stage]}: {len(use)}/{len(frames)} usable candidates "
                f"{ {t: (round(r['f1'], 2), round(r['f1_fixed'], 2), r['src']) for t, r in use.items()} }")
        info = {"stage": "AB"[stage], "variant": vname, "variant_scores": vscore, "frames": len(frames), "usable": len(use)}
        if not use:
            stages.append(info | {"accepted_frac": 0.0})
            continue
        # motion on the trimmed runs with the candidates' zoom as the nominal focal length
        fnom = pte.f_nominal(cache, {str(t): {"ok": True, "logf": r["logf"]} for t, r in use.items()})
        pairs = []
        for a, b in runs:
            ps, st = pt.compute_motion(cache, a, b, fnom[a:b + 1], rig_v.k1, gaps=(1, 5), log=pte.log)
            pairs += ps
        pairs, weak = pt.reject_weak_pairs(pairs)
        segs = pte.segments([{"start": a, "end": b, "p_main": 1.0} for a, b in runs], pairs)
        pairs_by = {(p.i, p.j): p for p in pairs}
        states = {t: np.array(r["refit"]["fixed"]) for t, r in use.items()}
        strong = {t: r["f1"] >= pr.F1_STRONG and r["f1_fixed"] >= 0.6 for t, r in use.items()}
        kept, partners = pr.confirm_anchors(states, pairs_by, sh, strong=strong)
        anchors = {str(t): {**r, "ok": t in kept} for t, r in use.items()}
        pte.log(f"recovery stage {'AB'[stage]}: confirmed {sorted(kept)} of {sorted(use)}; partners {partners}")
        st_arr, infos = pte.solve_variant(cache, segs, pairs, anchors, rig_v, [], None, "recover")
        f1rep = line_f1_report(cache, st_arr, sh, anchors, segs, prior)
        info |= {"confirmed": sorted(kept), "weak_pairs": weak, "accepted_frac": f1rep["accepted_frac_of_frames"]}
        stages.append(info)
        pte.log(f"recovery stage {'AB'[stage]}: accepted {100 * info['accepted_frac']:.0f}%")
        if best is None or info["accepted_frac"] > best["f1rep"]["accepted_frac_of_frames"]:
            best = {"states": st_arr, "infos": infos, "segs": segs, "anchors": anchors, "f1rep": f1rep, "runs": runs,
                    "rig": rig_v, "variant": vname}
        if info["accepted_frac"] >= RECOVER_BELOW:
            break
    if best is not None:
        best["stages"] = stages
    return best or {"stages": stages, "runs": runs}


RIG_FREE = ("Cx", "Cy", "Cz", "roll")  # k1 stays fixed at the given value
RIG_PRIOR_SIGMA = (5.0, 10.0, 3.0, float(np.deg2rad(2.0)), 0.05)  # fit_shared defaults (m, m, m, rad, k1 unused)
RIG_N_ITER = 30


def solve_rig_estimate(cache, states, segs, pairs, anchors, rig, *, n_iter: int = RIG_N_ITER, log=None,
                       bootstrap: int = 0, seed: int = 0, multistart: bool = False) -> dict:
    """Level-1 rig estimate of one segment: free (C, roll) with a loose prior centred on ``rig`` (the ``--rig`` input; no
    GT-derived prior), jointly with the (pan, tilt, log f) of every calibrated frame of every motion-connected run
    (``segs``), starting from the normal solve's ``states``; k1 stays at ``rig.k1``. Returns a json-able record.

    ``cov`` is the 4x4 covariance of (Cx, Cy, Cz [m], roll [rad]) from the final Gauss-Newton normal matrix, poses
    marginalised (Schur complement), robust IRLS weights included, scaled by max(1, chi2/dof)
    times the empirical ridge inflation (:func:`footy_track.calibration.ptz_temporal.shared_covariance`, which also
    explains why); it includes the loose prior, so a weak segment's
    sigma saturates at the prior sigma rather than blowing up."""
    units, anc_all, n_pairs = [], {}, 0
    for a, b in segs:
        fr = [t for t in range(a, b + 1) if np.all(np.isfinite(states[t]))]
        if len(fr) < 2:
            continue
        evs = {t: e for t in cache.ev_frames if a <= t <= b and np.all(np.isfinite(states[t])) and (e := cache.evidence(t)) is not None}
        anc = pte.anchors_state(anchors, [t for t in evs])
        keep = set(fr)
        prs = [p for p in pairs if a <= p.i and p.j <= b and p.i in keep and p.j in keep]
        n_pairs += sum(1 for p in prs if p.ok and len(p.a) >= 8)
        anc_all[a] = anc
        units.append(pt.Unit(a, fr, states[fr], evs, prs, anc))
    prior = {"centre_C": [float(x) for x in rig.C], "centre_roll": float(rig.roll), "k1_fixed": float(rig.k1),
             "sigma_C_m": list(RIG_PRIOR_SIGMA[:3]), "sigma_roll_rad": RIG_PRIOR_SIGMA[3], "source": "rig_json"}
    n_anc = sum(len(v) for v in anc_all.values())
    rec = {"names": list(RIG_FREE), "units": ["m", "m", "m", "rad"], "prior": prior, "n_anchors": n_anc,
           "n_evidence_frames": sum(len(u.evidence) for u in units), "n_motion_pairs": n_pairs, "code_sha": _git_sha(),
           "converged": False}
    if not units or not rec["n_evidence_frames"]:
        return rec | {"reason": "no calibrated frame with line evidence"}
    if multistart:  # P3 ablation: 3 x 3 grid of (C_y, C_z) seeds (C_x fixed at the prior), screened on every 3rd unit, 2 finalists
        shn, poses, info = pt.fit_shared_multistart(units, rig, pte.PITCH, offsets=pt.seed_offsets(step=(3.0, 4.0, 3.0), n=(0, 1, 1)), top_k=2,
                                                    screen_iter=6, screen_stride=1, log=log, free=RIG_FREE, sigma=RIG_PRIOR_SIGMA,
                                                    n_iter=n_iter, want_cov=True)
    else:
        shn, poses, info = pt.fit_shared(units, rig, pte.PITCH, free=RIG_FREE, sigma=RIG_PRIOR_SIGMA, n_iter=n_iter, log=log,
                                         want_cov=True)
    cv = info["cov"]
    allp = np.concatenate(poses)
    ok = np.all(np.isfinite(allp), axis=1)
    f = np.exp(allp[ok, 2]) if ok.any() else np.array([np.nan])
    sd = np.sqrt(np.clip(np.diag(cv["cov"]), 0, None))
    rec.update(
        rig=[float(x) for x in (*shn.C, shn.roll)], rig_roll_deg=float(np.rad2deg(shn.roll)),
        cov=cv["cov"].tolist(), cov_model=cv["cov_model"].tolist(), cov_nominal=cv["cov_nominal"].tolist(),
        cov_scale=float(cv["scale"]), cov_inflation=float(cv["inflation"]),
        cov_method="Schur complement of the IRLS-weighted Gauss-Newton normal matrix J^T W J at the solution "
                   "(poses marginalised, loose prior included), times max(1, chi2/dof) = cov_model, times cov_inflation (variance) = cov",
        chi2=cv["chi2"], dof=cv["dof"], sigma=[float(x) for x in sd], corr=(cv["cov"] / np.outer(sd, sd)).tolist(),
        focal_min=float(np.nanmin(f)), focal_max=float(np.nanmax(f)),
        pan_min_deg=float(np.rad2deg(np.nanmin(allp[ok, 0]))), pan_max_deg=float(np.rad2deg(np.nanmax(allp[ok, 0]))),
        final_cost=float(info["stage2"]["cost"]), stage2_iters=int(info["stage2"]["iters"]),
        converged=bool(info["stage2"]["iters"] < n_iter and np.all(np.isfinite(cv["cov"]))),
        n_frames=int(info["n_frames"]), multistart=info.get("starts"))
    if bootstrap > 0:
        rec["bootstrap"] = rig_bootstrap(units, poses, shn, rig, rec, bootstrap, seed, log=log)
    return rec


BOOT_BLOCK = 4  # consecutive evidence frames (20 frames at EV_STEP 5) drawn together: evidence is correlated along the clip
BOOT_FRAC = 0.5  # share of evidence blocks / anchors kept per resample
BOOT_N_ITER = 12


def rig_bootstrap(units, poses, shn, rig, rec, n_boot: int, seed: int, *, log=None) -> dict:
    """Half-sample resampling check of the level-1 covariance (P3, ``reports/ptz_v4_two_level.md``).

    Each resample keeps a random ``BOOT_FRAC`` of the evidence BLOCKS (``BOOT_BLOCK`` consecutive evidence frames) and of the
    anchors of every unit, keeps all motion pairs and frames, and refits (C, roll) warm-started at the full solution (poses
    and rig), so only the data sampling varies (local variability, not other basins). Sub-sampling at fraction m/n:
    ``Var_full ~= (m/n) Var_half``, so ``d = sqrt(BOOT_FRAC) (rig_b - rig_full)`` is the deviation to compare with the model
    covariance. Returns raw rigs, the scaled bootstrap covariance and the z^2 statistics against ``cov_model`` (no ridge
    inflation) and ``cov`` (inflated): a calibrated covariance has E[z^2] = 4 (df)."""
    rng = np.random.default_rng(seed)
    full = np.array([*shn.C, shn.roll], float)
    rigs, fails = [], 0
    for b in range(n_boot):
        ub = []
        for u, ps in zip(units, poses):
            evf = sorted(u.evidence)
            blocks = [evf[i:i + BOOT_BLOCK] for i in range(0, len(evf), BOOT_BLOCK)]
            keep_b = rng.random(len(blocks)) < BOOT_FRAC
            if not keep_b.any() and blocks:
                keep_b[rng.integers(len(blocks))] = True
            kept = {t for k, blk in enumerate(blocks) if keep_b[k] for t in blk}
            anc_t = sorted(u.anchors)
            keep_a = {t for t in anc_t if rng.random() < BOOT_FRAC}
            ub.append(pt.Unit(u.uid, u.frames, np.asarray(ps), {t: e for t, e in u.evidence.items() if t in kept},
                              u.pairs, {t: a for t, a in u.anchors.items() if t in keep_a}))
        try:
            sb, _, inf = pt.fit_shared(ub, rig, pte.PITCH, free=RIG_FREE, sigma=RIG_PRIOR_SIGMA, n_iter=BOOT_N_ITER,
                                       log=None, start=shn)
            rigs.append([*sb.C, sb.roll])
        except Exception as e:  # noqa: BLE001
            fails += 1
            if log:
                log("bootstrap fit failed", repr(e)[:100])
        if log and (b + 1) % 5 == 0:
            log(f"bootstrap {b + 1}/{n_boot}")
    R = np.array(rigs, float).reshape(-1, 4)
    out = {"n": int(len(R)), "n_failed": fails, "frac": BOOT_FRAC, "block": BOOT_BLOCK, "n_iter": BOOT_N_ITER, "seed": seed,
           "full_rig": full.tolist(), "rigs": R.tolist()}
    if len(R) >= 4 and rec.get("cov_model") is not None:
        D = np.sqrt(BOOT_FRAC) * (R - full)  # deviation scaled to full-sample size
        Sb = (D.T @ D) / len(D)  # about the full estimate (not the resample mean: also catches bias)
        out["cov_boot"] = Sb.tolist()
        out["sd_boot"] = np.sqrt(np.diag(Sb)).tolist()
        for name in ("cov_nominal", "cov_model", "cov"):
            C0 = np.asarray(rec[name], float)
            try:
                z2 = np.einsum("bi,ij,bj->b", D, np.linalg.inv(C0), D)
                out[f"z2_{name}"] = {"mean": float(z2.mean()), "median": float(np.median(z2)), "values": z2.tolist()}
            except np.linalg.LinAlgError:
                pass
        # variance factor lambda s.t. Sb ~= lambda * cov_model, in the whitened sense: mean z^2 / df
        if "z2_cov_model" in out:
            out["lambda_vs_cov_model"] = out["z2_cov_model"]["mean"] / 4.0
    return out


def run(video: Path, clip: str, rig_json: Path, out_root: Path, anchor_step: int = 100, recover: bool = True,
        trim: bool = False, force_recover: bool = False, estimate_rig: bool = False, bootstrap_rig: int = 0,
        bootstrap_seed: int = 0, rig_multistart: bool = False) -> dict:
    t0, cpu0 = time.time(), time.process_time()
    pte.ROOT = out_root
    pte.VIDEO_PATH = video
    pte.RIG_PATH = rig_json
    pte.WORK_SUFFIX = f"@{rig_json.stem}"
    pte.clip_gt = lambda _clip: []  # noqa: E731 (production: no labels, and no GT dir on the desktop)
    n = pte.n_frames(video)
    pte.ANCHOR_STEP = anchor_step_for(n, anchor_step)
    out = pte.wd(clip)
    out.mkdir(parents=True, exist_ok=True)
    pte.log(f"{clip}: {n} frames (container), anchor step {pte.ANCHOR_STEP}, rig {rig_json}, work dir {out}")

    rig, cache, shot_info, anchors, pairs, segs = pte.load_clip(clip)  # prep, refit anchors, motion, weak-pair guard
    pte.log("motion-connected main-camera runs:", segs)
    states, infos = pte.solve_variant(cache, segs, pairs, anchors, rig, [], None, "full")
    shared = pt.Shared.from_rig(rig)
    est = None
    if estimate_rig:  # P3: from the STANDARD pass (before any recovery variant replaces rig / states), prior centred on --rig-json
        est = solve_rig_estimate(cache, states, segs, pairs, anchors, rig, log=pte.log, bootstrap=bootstrap_rig,
                                 seed=bootstrap_seed, multistart=rig_multistart)

    f1rep = line_f1_report(cache, states, shared, anchors, segs)
    recovery = None
    if recover and (force_recover or f1rep["accepted_frac_of_frames"] < RECOVER_BELOW):
        pte.log(f"standard pass accepts {100 * f1rep['accepted_frac_of_frames']:.0f}% of the frames: recovery pass")
        rec = recover_pass(cache, rig, shot_info, out, trim)
        recovery = {"standard_accepted_frac": f1rep["accepted_frac_of_frames"], "stages": rec["stages"],
                    "runs": [list(map(int, r)) for r in rec["runs"]], "used": False}
        if rec.get("f1rep") and (force_recover or rec["f1rep"]["accepted_frac_of_frames"] > f1rep["accepted_frac_of_frames"]):
            recovery["used"] = True
            recovery["standard_anchors"] = f1rep["anchors"]
            states, infos, segs, anchors = rec["states"], rec["infos"], rec["segs"], rec["anchors"]
            f1rep, rig, shared = rec["f1rep"], rec["rig"], pt.Shared.from_rig(rec["rig"])
            recovery["variant"] = rec["variant"]
    valid = np.isfinite(states).all(axis=1)
    meta = {"clip": clip, "video": str(video), "frames_decoded": int(cache.n), "frames_container": n,
            "anchor_step": pte.ANCHOR_STEP, "rig_json": str(rig_json), "rig_C": [float(x) for x in rig.C],
            "rig_k1": float(rig.k1), "rig_variant": (recovery or {}).get("variant"), "code_sha": _git_sha(), "shots": shot_info, "runs": f1rep["main_camera_runs"],
            "calibrated_frames": int(valid.sum()), "calibrated_frac": float(valid.mean()) if cache.n else 0.0,
            "accepted_frac": f1rep["accepted_frac_of_frames"], "median_evidence_f1": f1rep["median_evidence_f1"],
            "anchors_ok": int(sum(bool(a.get("ok")) for a in anchors.values())), "anchors_tried": len(anchors),
            "recovery": recovery, "wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - cpu0, 1)}
    _atomic(out / "solve_full.json", lambda p: p.write_text(json.dumps(infos, default=float)))
    _atomic(out / "line_f1_full.json", lambda p: p.write_text(json.dumps(f1rep)))
    _atomic(out / "meta.json", lambda p: p.write_text(json.dumps(meta, indent=1, default=float)))

    def _states(p: Path) -> None:
        with open(p, "wb") as f:
            np.save(f, states)

    _atomic(out / "states_full.npy", _states)
    if est is not None:  # computed after the standard solve (does not change its outputs), written before the done marker
        bs = est.pop("bootstrap", None)
        _atomic(out / "rig_estimate.json", lambda p: p.write_text(json.dumps(est, indent=1, default=float)))
        pte.log("rig estimate:", est.get("rig"), "sigma", est.get("sigma"), "converged", est["converged"])
        if bs is not None:
            _atomic(out / "rig_bootstrap.json", lambda p: p.write_text(json.dumps(bs, default=float)))
            pte.log("bootstrap: n", bs["n"], "sd", bs.get("sd_boot"), "lambda", bs.get("lambda_vs_cov_model"))
    tmp = out / ".export"
    tmp.mkdir(exist_ok=True)
    pte.export_calib(tmp, "full", states, shared)
    os.replace(tmp / "calib_full.npz", out / "calib_full.npz")  # last: the completion marker
    tmp.rmdir()
    pte.log(f"done {clip}: {meta['calibrated_frames']}/{meta['frames_decoded']} frames calibrated "
            f"({100 * meta['accepted_frac']:.0f}% with F1>={F1_ACCEPT} evidence on both sides), "
            f"{meta['wall_s']:.0f}s wall, {meta['cpu_s']:.0f}s cpu")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--clip", required=True, help="e.g. arsenal_mancity_seg010; work dir is <out-root>/<clip>@<rig stem>")
    ap.add_argument("--rig-json", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=pte.ROOT)
    ap.add_argument("--anchor-step", type=int, default=100)
    ap.add_argument("--no-recover", action="store_true", help="skip the recovery pass for segments the standard pass leaves weak")
    ap.add_argument("--force-recover", action="store_true",
                    help="evaluation only: run the recovery pass on any segment and keep its result (accuracy of the recovery path on segments the standard pass solves)")
    ap.add_argument("--trim", action="store_true",
                    help="recovery pass: cut stretches whose main-camera probability is low before solving (off by default, see recover_pass)")
    ap.add_argument("--estimate-rig", action="store_true",
                    help="level-1 rig estimate: free (C, roll) with a loose prior around --rig-json, write rig_estimate.json "
                         "(rig, 4x4 covariance, evidence counts) before the done marker; the other outputs are unchanged")
    ap.add_argument("--bootstrap-rig", type=int, default=0, metavar="N",
                    help="with --estimate-rig: N half-sample resamples of the evidence/anchors, written to rig_bootstrap.json (P3 calibration of the covariance)")
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--rig-multistart", action="store_true", help="with --estimate-rig: 9 (C_y, C_z) seeds around the prior, keep the lowest cost (P3 ablation)")
    a = ap.parse_args()
    run(a.video, a.clip, a.rig_json, a.out_root, a.anchor_step, recover=not a.no_recover, trim=a.trim,
        force_recover=a.force_recover, estimate_rig=a.estimate_rig, bootstrap_rig=a.bootstrap_rig,
        bootstrap_seed=a.bootstrap_seed, rig_multistart=a.rig_multistart)


if __name__ == "__main__":
    main()
