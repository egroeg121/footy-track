"""H2b: does the free-C fit from the prior reach the GT valley with a very long LM budget (n_iter 1500, tol 1e-11)? Also pooled-5 from GT start.
    python scripts/ptz_p3/why_long.py SEG... -> $WHY_ROOT/long_segNNN.json"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "scripts/ptz_p3")
from why_common import *  # noqa: F403

for i in map(int, sys.argv[1:]):
    rig, cache, anchors, pairs, segs, st = setup(i)
    units = build_units(cache, st, segs, pairs, anchors)
    src = pt.Shared.from_rig(rig)
    t0 = time.process_time()
    sh, poses, info = free_fit(units, rig, rig.C, rig.roll, src, n_iter=1500, tol=1e-11)
    rec = {"seg": i, "C": sh.C.tolist(), "roll_deg": float(np.rad2deg(sh.roll)), "err_m": float(np.linalg.norm(sh.C - GT_C)),
           "cost": info["stage2"]["cost"], "iters": info["stage2"]["iters"], "hit_cap": info["stage2"]["hit_cap"], "cpu_s": time.process_time() - t0}
    log(rec)
    (SCRATCH / f"long_seg{i:03d}.json").write_text(json.dumps(rec))
