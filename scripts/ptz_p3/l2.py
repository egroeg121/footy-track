"""P3 level 2: consensus over the L1 rig_estimate.json files with several covariance inflations + report numbers.

    python scripts/ptz_p3/l2.py --glob '/mnt/.../arsenal_mancity_seg*@generic_gantry_prior_v2-1be31b96' --out-dir DIR --infl 1,36,CAL
numpy/scipy only (light; fine on merlin via an ephemeral uv env)."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
from scipy.stats import chi2, kstest

from footy_track.calibration.rig_consensus import ConsensusParams, compute_consensus, load_estimate

GT_REF = [0.100, -66.576, 15.743, math.radians(-0.13)]  # GT-points rig (emirates.json): EVAL REFERENCE ONLY
FIT95 = [-0.2068, -67.0326, 15.9769, math.radians(-0.0825)]  # existing 95-frame multi-start fit_shared (ms_mancity_ms_p1.json, k1 free)
GANTRY = (46, 71, 73, 108, 137, 154)


def seg_no(sid):
    return int(re.search(r"_seg(\d+)@", sid).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--infl", default="1,36")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.glob, "rig_estimate.json")))
    noest = [os.path.basename(os.path.dirname(f)) for f in files if not json.load(open(f)).get("rig")]
    est = [load_estimate(f) for f in files if json.load(open(f)).get("rig")]
    n_total = len(files)
    n_noest = len(noest)
    out = {"n_files": n_total, "n_with_estimate": n_total - n_noest, "runs": {}}
    for tok in a.infl.split(","):
        infl = float(tok)
        prm = ConsensusParams(cov_inflation=infl)
        use = [e for e in est if np.all(np.isfinite(e.rig))]
        res = compute_consensus(use, prm, reference=GT_REF)
        res.save(os.path.join(a.out_dir, f"consensus_infl{tok}.json"))
        r = np.array(res.rig)
        rows = res.segments
        d2 = np.array([s["d"] ** 2 for s in rows if s["d"] is not None])
        flags = {}
        for s in rows:
            flags[s["flag"].split(":")[0]] = flags.get(s["flag"].split(":")[0], 0) + 1
        inl = d2[np.array([s["flag"] == "inlier" and s["d"] is not None for s in rows if s["d"] is not None])] if len(d2) else d2
        rec = {"rig": r.tolist(), "dC_vs_gt_m": float(np.linalg.norm(r[:3] - GT_REF[:3])), "dC_vs_fit95_m": float(np.linalg.norm(r[:3] - FIT95[:3])),
               "droll_vs_gt_deg": float(math.degrees(r[3] - GT_REF[3])), "sd_C": np.sqrt(np.diag(res.cov))[:3].tolist(),
               "flags": flags, "n_fit": res.n_fit, "main_weight_share": res.main_weight_share, "variants": len(res.variants),
               "chi2_df4": {"mean_d2": float(d2.mean()) if len(d2) else None, "median_d2": float(np.median(d2)) if len(d2) else None,
                            "expected_mean": 4.0, "expected_median": float(chi2.ppf(0.5, 4)),
                            "frac_le_0.999": float(np.mean(d2 <= chi2.ppf(0.999, 4))) if len(d2) else None,
                            "ks_p": float(kstest(d2, chi2(4).cdf).pvalue) if len(d2) > 3 else None},
               "gantry": {str(g): next((s["flag"] + " d=%s" % (None if s["d"] is None else round(s["d"], 1)) for s in rows if seg_no(s["seg_id"]) == g), "no estimate file / uncalibrated") for g in GANTRY}}
        non_inl = [s["seg_id"] for s in rows if not s["flag"].startswith("inlier")]
        rec["non_inlier_segments"] = [seg_no(x) for x in non_inl]
        out["runs"][tok] = rec
        print(f"--- inflation {tok}\n{res.summary()}\n  dC vs GT-points rig {rec['dC_vs_gt_m']:.2f} m, vs 95-frame fit {rec['dC_vs_fit95_m']:.2f} m; chi2 mean d2 {rec['chi2_df4']['mean_d2']:.1f} (df 4), "
              f"{100 * rec['chi2_df4']['frac_le_0.999']:.0f}% <= chi2.999; gantry: {rec['gantry']}")
    out["segments_without_estimate"] = n_noest
    out["segments_without_estimate_ids"] = [seg_no(x) for x in noest]
    json.dump(out, open(os.path.join(a.out_dir, "l2_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
