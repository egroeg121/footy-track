"""H5: fit_shared on frames of ONE segment vs frames pooled over several segments, same evidence-frame count, same prior start, tight LM.
    python scripts/ptz_p3/why_pool.py -> $WHY_ROOT/pool.json"""
from __future__ import annotations

import json
import sys
from dataclasses import replace

sys.path.insert(0, "scripts/ptz_p3")
from why_common import *  # noqa: F403

SEGS = (10, 20, 80, 104, 150)
N_MATCH = 60


def subset(units, n, rng):
    """Keep ``n`` evidence frames drawn at random over all units (poses / motion pairs of every frame stay)."""
    keys = [(k, t) for k, u in enumerate(units) for t in u.evidence]
    pick = {keys[j] for j in rng.choice(len(keys), size=min(n, len(keys)), replace=False)}
    return [replace(u, evidence={t: e for t, e in u.evidence.items() if (k, t) in pick}) for k, u in enumerate(units)]


def main():
    per = {}
    for i in SEGS:
        rig, cache, anchors, pairs, segs, st = setup(i)
        us = build_units(cache, st, segs, pairs, anchors)
        for u in us:
            u.uid = i
        per[i] = us
    rig = setup(10)[0]
    src = pt.Shared.from_rig(rig)
    allu = [u for i in SEGS for u in per[i]]
    out = []

    def run(name, units):
        sh, poses, info = free_fit(units, rig, rig.C, rig.roll, src, n_iter=150, tol=1e-6)
        allp = np.concatenate(poses); f = np.exp(allp[:, 2])
        rec = {"name": name, "n_ev": sum(len(u.evidence) for u in units), "C": sh.C.tolist(), "err_m": float(np.linalg.norm(sh.C - GT_C)),
               "err_y": float(sh.C[1] - GT_C[1]), "cost": info["stage2"]["cost"], "iters": info["stage2"]["iters"],
               "pan_range_deg": float(np.rad2deg(np.ptp(allp[:, 0]))) if len(units) == 1 else None,
               "f_ratio": float(f.max() / f.min()), "sd_model": [float(x) for x in np.sqrt(np.diag(info["cov"]["cov_model"]))[:3]]}
        out.append(rec); log(rec)
        (SCRATCH / "pool.json").write_text(json.dumps(out))

    for i in SEGS:
        run(f"single seg{i:03d} all evidence", per[i])
    run("pooled 5 segments, all evidence", allu)
    run("pooled 4 short segments (no 010), all evidence", [u for i in SEGS if i != 10 for u in per[i]])
    for seed in range(3):
        rng = np.random.default_rng(seed)
        run(f"seg010 only, {N_MATCH} random evidence frames, seed {seed}", subset(per[10], N_MATCH, rng))
    for seed in range(3):
        rng = np.random.default_rng(seed)
        run(f"pooled 5 segments, {N_MATCH} random evidence frames, seed {seed}", subset(allu, N_MATCH, rng))


main()
