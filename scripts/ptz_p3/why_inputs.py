"""H3: input bias. Free-C fit (tight LM) from the prior start with the inputs (anchors, initial poses, motion nominal f) computed at
the prior rig vs at the GT reference rig (k1 0.03 in both), with and without the pose anchors.
    python scripts/ptz_p3/why_inputs.py SEG... -> $WHY_ROOT/inputs_segNNN.json"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "scripts/ptz_p3")
from why_common import *  # noqa: F403

GT_JSON = SCRATCH / "gt_ref_k03.json"


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    ptz.Rig(GT_C, GT_ROLL, 0.03, (1.0, 3.0, 1.0)).to_json(GT_JSON)
    for i in map(int, sys.argv[1:]):
        rigP, cacheP, ancP, pairsP, segsP, stP = setup(i)
        rigG, cacheG, ancG, pairsG, segsG, _ = setup(i, GT_JSON, "gt_ref_k03", src=None)
        stG, _ = pte.solve_variant(cacheG, segsG, pairsG, ancG, rigG, [], None, "full")
        log(i, "G pipeline anchors ok", sum(bool(a.get("ok")) for a in ancG.values()), "of", len(ancG), "| P:", sum(bool(a.get("ok")) for a in ancP.values()))
        srcP, srcG = pt.Shared.from_rig(rigP), pt.Shared.from_rig(rigG)
        res = []
        for name, args, use_anc, src in (
            ("prior inputs, anchors", (cacheP, stP, segsP, pairsP, ancP), True, srcP),
            ("prior inputs, no anchors", (cacheP, stP, segsP, pairsP, ancP), False, srcP),
            ("GT-rig inputs, anchors", (cacheG, stG, segsG, pairsG, ancG), True, srcG),
            ("GT-rig inputs, no anchors", (cacheG, stG, segsG, pairsG, ancG), False, srcG),
        ):
            units = build_units(*args, use_anchors=use_anc)
            t0 = time.process_time()
            sh, poses, info = free_fit(units, rigP, rigP.C, rigP.roll, src, n_iter=150, tol=1e-6)
            rec = {"name": name, "C": sh.C.tolist(), "roll_deg": float(np.rad2deg(sh.roll)), "err_m": float(np.linalg.norm(sh.C - GT_C)),
                   "cost": info["stage2"]["cost"], "iters": info["stage2"]["iters"], "n_anchors": sum(len(u.anchors) for u in units),
                   "cpu_s": time.process_time() - t0}
            res.append(rec); log(i, rec)
        (SCRATCH / f"inputs_seg{i:03d}.json").write_text(json.dumps(res))


main()
