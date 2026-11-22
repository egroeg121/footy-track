"""Experiment D: label-free rig fit (C, roll, [k1]) of Man City from one line-evidence frame per segment.

Units = the ok frames of ``anchors_mancity_generic.json`` (``anchors_generic.py``: searched at the GENERIC prior, no GT rig).
Arms: e1 = fit_shared_multistart, production stopping rule; e2 = same, tight rule on the finalists; e3 = coarse (C_y, C_z)
grid of fixed-C pose solves around the start, then converged local refine (tight rule) of the best ``--top`` grid points.
Reference rig (GT points, evaluation only): (0.100, -66.576, 15.743).

  PYTHONPATH=src nice -n 19 uv run --no-project --with numpy --with scipy --with opencv-python-headless --with scikit-image \
     python scripts/ptz_p5/rig_search.py --arm e3 --start generic --k1 fixed
"""
import argparse, json, time
from dataclasses import replace
from pathlib import Path

import numpy as np

from footy_track.calibration import ptz
from footy_track.calibration import ptz_temporal as pt
from footy_track.calibration.pitch_model import PitchModel
from footy_track.scripts import ptz_temporal as T

REF = np.array([0.100, -66.576, 15.743])
STARTS = {"generic": (0.0, -63.0, 14.0), "s1": (0.0, -60.0, 12.0), "s2": (2.0, -71.0, 10.0)}
GENERIC = Path("reports/ptz_v4_assets/generic_gantry_prior_v2.json")
OUT = Path("reports/ptz_v5_assets/rigs")
PITCH = PitchModel()


def units_generic():
    root = T.ROOT / "shared"
    store = json.loads((root / "anchors_mancity_generic.json").read_text())
    return T.shared_units("mancity", T.match_segments("mancity"), store, root, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["e1", "e2", "e3"])
    ap.add_argument("--start", default="generic", choices=list(STARTS))
    ap.add_argument("--k1", default="fixed", choices=["fixed", "free"])
    ap.add_argument("--tol", type=float, default=1e-8, help="tight stopping rule (e2 / e3 refine)")
    ap.add_argument("--iters", type=int, default=100, help="LM iterations per stage for the tight rule")
    ap.add_argument("--gy", default="-10,10,2.5", help="e3 grid C_y offsets lo,hi,step")
    ap.add_argument("--gz", default="-6,6,2", help="e3 grid C_z offsets lo,hi,step")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1, help="e3: grid pose solves on every n-th unit")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    gen = ptz.Rig.from_json(GENERIC)
    rig = ptz.Rig(np.array(STARTS[a.start], float), 0.0, 0.03, gen.sigma_C, gen.pitch, gen.info)
    states_at = pt.Shared.from_rig(gen)
    units = units_generic()
    free = ("Cx", "Cy", "Cz", "roll") + (("k1",) if a.k1 == "free" else ())
    name = a.name or f"{a.arm}_{a.start}_k1{a.k1}"
    print(name, len(units), "units; start", rig.C, "dist to ref", np.linalg.norm(rig.C - REF), flush=True)
    t0 = time.process_time()
    log = lambda *x: print(f"[cpu {time.process_time() - t0:.0f}s]", *x, flush=True)
    extra = {}
    if a.arm in ("e1", "e2"):
        kw = dict(rel_tol=a.tol, n_iter=a.iters) if a.arm == "e2" else {}
        shn, _, info = pt.fit_shared_multistart(units, rig, PITCH, states_at=states_at, offsets=pt.seed_offsets(), top_k=3,
                                                screen_stride=2, free=free, log=log, **kw)
        extra["starts"] = info["starts"]
    else:
        lo, hi, st = map(float, a.gy.split(","))
        oy = np.arange(lo, hi + 1e-9, st)
        lo, hi, st = map(float, a.gz.split(","))
        oz = np.arange(lo, hi + 1e-9, st)
        us_g = units[:: a.stride]
        grid = []
        for dy in oy:
            for dz in oz:
                c = rig.C + np.array([0.0, dy, dz])
                seed = ptz.Rig(c, rig.roll, rig.k1, rig.sigma_C, rig.pitch, rig.info)
                us = [replace(u, states=pt.reseed_states(u.states, states_at, pt.Shared.from_rig(seed))) for u in us_g]
                _, _, inf = pt.fit_shared(us, seed, PITCH, free=(), n_iter=15)
                grid.append({"C": c.tolist(), "cost": inf["stage2"]["cost"], "n_line_frames": inf["n_line_frames"]})
                log(f"grid C {np.round(c, 1).tolist()} cost {grid[-1]['cost']:.0f}")
        order = sorted(range(len(grid)), key=lambda i: grid[i]["cost"])
        best = None
        for i in order[: a.top]:
            start = replace(pt.Shared.from_rig(rig), C=np.array(grid[i]["C"]))
            us = [replace(u, states=pt.reseed_states(u.states, states_at, start)) for u in units]
            shn_i, _, inf = pt.fit_shared(us, rig, PITCH, start=start, free=free, n_iter=a.iters, rel_tol=a.tol, log=log)
            grid[i].update(final_C=shn_i.C.tolist(), final_cost=inf["stage2"]["cost"], final_k1=float(shn_i.k1),
                           iters=inf["stage2"]["iters"])
            log(f"refine from {np.round(grid[i]['C'], 1).tolist()}: cost {inf['stage2']['cost']:.0f} C {np.round(shn_i.C, 2).tolist()}")
            if best is None or inf["stage2"]["cost"] < best[0]:
                best = (inf["stage2"]["cost"], shn_i, inf)
        _, shn, info = best
        extra["grid"] = grid
    cpu = time.process_time() - t0
    res = {"name": name, "arm": a.arm, "start": a.start, "start_C": rig.C.tolist(), "start_dist_ref": float(np.linalg.norm(rig.C - REF)),
           "k1_mode": a.k1, "C": shn.C.tolist(), "roll_deg": float(np.degrees(shn.roll)), "k1": float(shn.k1),
           "cost": info["stage2"]["cost"], "iters": info["stage2"]["iters"], "n_units": len(units), "cpu_s": cpu,
           "dist_ref_m": float(np.linalg.norm(shn.C - REF)), "dC_ref": (shn.C - REF).tolist(), **extra}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.fit.json").write_text(json.dumps(res, indent=1))
    out_rig = ptz.Rig(shn.C, shn.roll, float(shn.k1), (1.0, 3.0, 1.0), gen.pitch,
                      {"ground": name, "source": f"ptz_v5 rig search {a.arm}, label-free, start {a.start}; reference rig NOT used"})
    out_rig.to_json(OUT / f"{name}.json")
    print("RESULT", {k: v for k, v in res.items() if k not in ("starts", "grid")}, flush=True)


if __name__ == "__main__":
    main()
