"""H1: profile the per-segment cost (poses re-optimised, k1 fixed 0.03, no rig prior rows) along E->G, and over a (C_y, C_z) grid.
    python scripts/ptz_p3/why_profile.py SEG [SEG...]   -> $WHY_ROOT/profile_segNNN.json"""
from __future__ import annotations

import json
import sys
from dataclasses import replace

sys.path.insert(0, "scripts/ptz_p3")
from why_common import *  # noqa: F403


def cost_at(units, rig, src_sh, C, roll, n_iter=30, reseed=True):
    r2 = rig_at(rig, C, roll)
    start = pt.Shared.from_rig(r2)
    us = [replace(u, states=pt.reseed_states(u.states, src_sh, start)) for u in units] if reseed else units
    sh, poses, info = pt.fit_shared(us, r2, pte.PITCH, free=(), n_iter=n_iter)
    return info["stage2"]["cost"], poses


def main():
    for i in map(int, sys.argv[1:]):
        rig, cache, anchors, pairs, segs, st = setup(i)
        units = build_units(cache, st, segs, pairs, anchors)
        est = json.loads((BOOT / f"arsenal_mancity_seg{i:03d}@{STEM}" / "rig_estimate.json").read_text())
        E = np.array(est["rig"][:3]); rE = est["rig"][3]
        src = pt.Shared.from_rig(rig)
        out = {"seg": i, "est": est["rig"], "est_cost": est["final_cost"], "scale": est["cov_scale"], "chi2": est["chi2"], "dof": est["dof"],
               "n_frames": est["n_frames"], "pan_range_deg": est["pan_max_deg"] - est["pan_min_deg"],
               "focal_range": [est["focal_min"], est["focal_max"]], "line": [], "grid": []}
        # line E -> G (roll interpolated), t = 0 at E, 1 at G, plus beyond G and back toward the prior
        for t in (-0.5, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
            C = E + t * (GT_C - E); roll = rE + t * (GT_ROLL - rE)
            c, _ = cost_at(units, rig, src, C, roll)
            out["line"].append({"t": t, "C": C.tolist(), "cost": c}); log(i, "line", t, np.round(C, 2).tolist(), round(c, 1))
        for name, C, roll in (("prior", rig.C, rig.roll), ("G", GT_C, GT_ROLL), ("E", E, rE)):
            c, _ = cost_at(units, rig, src, C, roll); out[f"cost_{name}"] = c; log(i, name, round(c, 1))
        # grid in (C_y, C_z) at C_x = GT, roll = GT
        for dy in (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0):
            for dz in (-3.0, -1.5, 0.0, 1.5, 3.0):
                C = GT_C + np.array([0, dy, dz])
                c, _ = cost_at(units, rig, src, C, GT_ROLL)
                out["grid"].append({"dy": dy, "dz": dz, "cost": c})
        # axis scans around G for Cx and roll
        out["cx"] = [{"dx": dx, "cost": cost_at(units, rig, src, GT_C + [dx, 0, 0], GT_ROLL)[0]} for dx in (-2, -1, 1, 2)]
        out["roll"] = [{"d": d, "cost": cost_at(units, rig, src, GT_C, GT_ROLL + np.deg2rad(d))[0]} for d in (-0.5, -0.25, 0.25, 0.5)]
        (SCRATCH / f"profile_seg{i:03d}.json").write_text(json.dumps(out))


main()
