"""H2: optimiser. Free-C L1 fit from the prior and from the GT reference with the production stopping rule vs tight tolerance / long budget.
    python scripts/ptz_p3/why_optimiser.py SEG...  -> $WHY_ROOT/optimiser_segNNN.json"""
from __future__ import annotations

import json
import sys
from dataclasses import replace

sys.path.insert(0, "scripts/ptz_p3")
from why_common import *  # noqa: F403


def main():
    for i in map(int, sys.argv[1:]):
        rig, cache, anchors, pairs, segs, st = setup(i)
        units = build_units(cache, st, segs, pairs, anchors)
        src = pt.Shared.from_rig(rig)
        res = []
        for name, C0, r0, n_iter, tol in (
            ("prior, prod rule (30 it, tol 1e-4)", rig.C, rig.roll, 30, None),
            ("prior, tol 1e-6, 150 it", rig.C, rig.roll, 150, 1e-6),
            ("prior, tol 1e-8, 400 it", rig.C, rig.roll, 400, 1e-8),
            ("GT start, prod rule", GT_C, GT_ROLL, 30, None),
            ("GT start, tol 1e-6, 150 it", GT_C, GT_ROLL, 150, 1e-6),
        ):
            t0 = time.process_time()
            sh, poses, info = free_fit(units, rig, C0, r0, src, n_iter=n_iter, tol=tol)
            s2 = info["stage2"]
            rec = {"name": name, "C": sh.C.tolist(), "roll_deg": float(np.rad2deg(sh.roll)), "err_m": float(np.linalg.norm(sh.C - GT_C)),
                   "cost": s2["cost"], "iters": s2["iters"], "cpu_s": time.process_time() - t0, "stage1_iters": info["stage1"]["iters"],
                   "sd_model": [float(x) for x in np.sqrt(np.diag(info["cov"]["cov_model"]))[:3]], "scale": info["cov"]["scale"]}
            res.append(rec); log(i, rec)
        # tight-tolerance fixed-C profile along E -> G (poses re-optimised; compares with the production-rule profile)
        E = np.array(res[0]["C"]); rE = np.deg2rad(res[0]["roll_deg"]); prof = []
        for t in (0, 0.25, 0.5, 0.75, 1.0, 1.25):
            C = E + t * (GT_C - E); roll = rE + t * (GT_ROLL - rE)
            for tol, nit in ((None, 30), (1e-7, 300)):
                with_lm(tol) if tol is not None else setattr(pt, "lm_solve", ORIG_LM)
                r2 = rig_at(rig, C, roll); start = pt.Shared.from_rig(r2)
                us = [replace(u, states=pt.reseed_states(u.states, src, start)) for u in units]
                _, _, inf = pt.fit_shared(us, r2, pte.PITCH, free=(), n_iter=nit)
                prof.append({"t": t, "tol": tol, "cost": inf["stage2"]["cost"], "iters": inf["stage2"]["iters"]})
            log(i, "fixed-C profile", prof[-2:])
        (SCRATCH / f"optimiser_prof_seg{i:03d}.json").write_text(json.dumps(prof))
        (SCRATCH / f"optimiser_seg{i:03d}.json").write_text(json.dumps(res))


main()
