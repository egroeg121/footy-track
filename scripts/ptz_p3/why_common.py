"""Shared harness for the 'why two-level failed' experiments (reports/ptz_v4_two_level.md appendix).

Rebuilds the per-segment inputs of the L1 solve (cache, anchors, motion pairs, standard-pass states) from the kept L1
caches in a scratch ROOT and builds the Units exactly as ``solve_rig_estimate`` does. Run with
  PYTHONPATH=src OMP_NUM_THREADS=1 nice -n 19 uv run --no-project --python /usr/bin/python3 \
    --with numpy --with scipy --with opencv-python-headless python scripts/ptz_p3/why_*.py ...
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from footy_track.calibration import ptz
from footy_track.calibration import ptz_temporal as pt
from footy_track.scripts import ptz_temporal as pte
from footy_track.scripts import ptz_temporal_segment as seg

TEMPORAL = Path("/mnt/storage/footy_data/scratch/temporal")
BOOT = TEMPORAL / "p3" / "boot"
SCRATCH = Path(os.environ.get("WHY_ROOT", "/tmp/claude-1000/-home-george-footy-track/dcd27824-b487-4c6e-9e22-d876363d461e/scratchpad/why"))
PRIOR = TEMPORAL / "p3" / "generic_gantry_prior_v2.json"
STEM = "generic_gantry_prior_v2-1be31b96"
GT_C = np.array([0.100, -66.576, 15.743])  # GT-points rig: REFERENCE only
GT_ROLL = -0.002277934861682635  # emirates.json (-0.13 deg)
VIDEO = "/mnt/storage/footy_data/arsenal_mancity/split_video_broadcast_frames/arsenal_mancity_20250925_seg{i:03d}.mp4"


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[cpu {time.process_time():.0f}s]", *a, flush=True)


def setup(i: int, rig_json: Path = PRIOR, stem: str = STEM, src: Path | None = None):
    """Returns (rig, cache, anchors, pairs, segs, states) for Man City segment ``i`` at the prior rig."""
    clip = f"arsenal_mancity_seg{i:03d}"
    src = src or (BOOT / f"{clip}@{STEM}")
    pte.ROOT = SCRATCH
    pte.VIDEO_PATH = Path(VIDEO.format(i=i))
    pte.RIG_PATH = rig_json
    pte.WORK_SUFFIX = f"@{stem}"
    pte.clip_gt = lambda _c: []
    n = pte.n_frames(pte.VIDEO_PATH)
    pte.ANCHOR_STEP = seg.anchor_step_for(n, 100)
    d = pte.wd(clip)
    d.mkdir(parents=True, exist_ok=True)
    if src is not None and src.exists():
        for f in ("anchors_lines.json", "anchors_refit.json", "pairs_g1_5.pkl"):
            if (src / f).exists() and not (d / f).exists():
                shutil.copy(src / f, d / f)
    rig, cache, shot_info, anchors, pairs, segs = pte.load_clip(clip)
    st = np.load(src / "states_full.npy") if (src / "states_full.npy").exists() else None
    return rig, cache, anchors, pairs, segs, st


def build_units(cache, states, segs, pairs, anchors, use_anchors=True):
    units = []
    for a, b in segs:
        fr = [t for t in range(a, b + 1) if np.all(np.isfinite(states[t]))]
        if len(fr) < 2:
            continue
        evs = {t: e for t in cache.ev_frames if a <= t <= b and np.all(np.isfinite(states[t])) and (e := cache.evidence(t)) is not None}
        anc = pte.anchors_state(anchors, list(evs)) if use_anchors else {}
        keep = set(fr)
        prs = [p for p in pairs if a <= p.i and p.j <= b and p.i in keep and p.j in keep]
        units.append(pt.Unit(a, fr, states[fr], evs, prs, anc))
    return units


def rig_at(rig, C, roll):
    return replace(rig, C=np.asarray(C, float), roll=float(roll))


def lm_solve_tol(rs, x0, n_iter=25, log=None, tag="", tol=1e-4, max_try=8):
    """Copy of ``pt.lm_solve`` with the relative-decrease stopping tolerance exposed (original: 1e-4 * cost)."""
    from scipy.linalg import cho_factor, cho_solve

    quad = np.zeros(rs.n_res, bool)
    quad[rs.n_res - rs.nsh - rs.nacc: rs.n_res] = True
    c = rs.pb.c_robust
    x = x0.copy()

    def cost(r):
        return float(np.sum(np.where(quad, 0.5 * r * r, 0.5 * c * c * np.log1p((r / c) ** 2))))

    r = rs.residual(x); f0 = f = cost(r); lam = 1e-3; hist = [f]
    for it in range(n_iter):
        J = rs.jacobian(x); w = pt._robust_w(r, c, quad); sw = np.sqrt(w)
        Jw = J.multiply(sw[:, None]).tocsr(); A = (Jw.T @ Jw).toarray(); g = Jw.T @ (sw * r)
        d = np.diag(A).copy(); d[d < 1e-9] = 1e-9
        improved = False
        for _ in range(max_try):
            try:
                cf = cho_factor(A + lam * np.diag(d), lower=True, check_finite=False); step = -cho_solve(cf, g, check_finite=False)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            xn = x + step; rn = rs.residual(xn); fn = cost(rn)
            if np.isfinite(fn) and fn < f:
                x, r, f, lam, improved = xn, rn, fn, max(lam / 5, 1e-7), True; break
            lam *= 5
        hist.append(f)
        if not improved or (hist[-2] - f) < tol * max(hist[-2], 1.0):
            break
    return x, {"cost0": f0, "cost": f, "iters": len(hist) - 1, "hit_cap": len(hist) - 1 >= n_iter, "improved_last": improved}


def with_lm(tol, max_try=8):
    """Context-free patch: make ``pt.joint_solve`` use ``lm_solve_tol`` with the given tolerance."""
    pt.lm_solve = lambda rs, x0, n_iter=25, log=None, tag="": lm_solve_tol(rs, x0, n_iter, log, tag, tol, max_try)


ORIG_LM = pt.lm_solve


def free_fit(units, rig, start_C, start_roll, src_sh, *, n_iter, tol, reseed=True):
    """The L1 free fit (Cx, Cy, Cz, roll free, prior centred on ``rig``) started at ``start_C``."""
    with_lm(tol) if tol is not None else setattr(pt, "lm_solve", ORIG_LM)
    start = replace(pt.Shared.from_rig(rig), C=np.asarray(start_C, float), roll=float(start_roll))
    us = [replace(u, states=pt.reseed_states(u.states, src_sh, start)) for u in units] if reseed else units
    shn, poses, info = pt.fit_shared(us, rig, pte.PITCH, free=seg.RIG_FREE, sigma=seg.RIG_PRIOR_SIGMA, n_iter=n_iter, start=start, want_cov=True)
    return shn, poses, info
