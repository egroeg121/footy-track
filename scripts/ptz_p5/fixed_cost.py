"""Data cost (poses re-optimised, rig fixed) of each fitted rig and of the reference rig on the label-free unit set."""
import glob, json, sys
from dataclasses import replace
import numpy as np
sys.path.insert(0, "scripts/ptz_p5")
import rig_search as R
from footy_track.calibration import ptz, ptz_temporal as pt

gen = ptz.Rig.from_json(R.GENERIC)
sa = pt.Shared.from_rig(gen)
units = R.units_generic()
rigs = {"REF(GT points, eval only)": ptz.Rig(R.REF, -0.13 * np.pi / 180, 0.0147, gen.sigma_C, gen.pitch, {})}
rigs["REF_k1_0.03"] = ptz.Rig(R.REF, -0.13 * np.pi / 180, 0.03, gen.sigma_C, gen.pitch, {})
for f in sorted(glob.glob("reports/ptz_v5_assets/rigs/*.json")):
    if not f.endswith(".fit.json"):
        rigs[f.split("/")[-1][:-5]] = ptz.Rig.from_json(f)
out = {}
for n, r in rigs.items():
    us = [replace(u, states=pt.reseed_states(u.states, sa, pt.Shared.from_rig(r))) for u in units]
    _, _, inf = pt.fit_shared(us, r, R.PITCH, free=(), n_iter=15)
    out[n] = {"C": r.C.tolist(), "k1": float(r.k1), "data_cost": inf["stage2"]["cost"], "dist_ref": float(np.linalg.norm(r.C - R.REF))}
    print(f"{n:28s} dist {out[n]['dist_ref']:.2f} cost {out[n]['data_cost']:.0f}", flush=True)
json.dump(out, open("reports/ptz_v5_assets/fixed_cost.json", "w"), indent=1)
