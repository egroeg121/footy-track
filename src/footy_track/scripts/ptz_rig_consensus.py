"""Level-2 rig consensus over per-segment ``rig_estimate.json`` (``calibration.rig_consensus``).

    python -m footy_track.scripts.ptz_rig_consensus ROOT_OR_FILES... [--out consensus.json] [--reference-rig rig.json]
        [--outlier-quantile 0.999] [--max-pos-sd-m 10] [--prev prev_consensus.json]

``--reference-rig`` (JSON ``[Cx,Cy,Cz,roll_rad]``, ``{"rig": [...]}`` or ``{"C": [...], "roll_deg": ..}``) is only used
to REPORT the distance to it; it never enters the fit. ``--prev`` prints the section 2.4 stop-rule verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from footy_track.calibration.rig_consensus import (
    ConsensusParams,
    ConsensusResult,
    compute_consensus,
    load_estimates,
    stop_rule,
)


def _ref(path: str):
    d = json.loads(Path(path).read_text())
    if isinstance(d, list):
        return d
    if "rig" in d:
        return d["rig"]
    C = d.get("C") or [d["Cx"], d["Cy"], d["Cz"]]
    roll = d["roll_rad"] if "roll_rad" in d else math.radians(d.get("roll_deg", 0.0))
    return [*C, roll]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--reference-rig")
    ap.add_argument("--prev")
    d = ConsensusParams()
    ap.add_argument("--outlier-quantile", type=float, default=d.outlier_quantile)
    ap.add_argument("--cauchy-quantile", type=float, default=d.cauchy_quantile)
    ap.add_argument("--max-pos-sd-m", type=float, default=d.max_pos_sd_m)
    ap.add_argument("--max-roll-sd-rad", type=float, default=d.max_roll_sd_rad)
    ap.add_argument("--min-variant-size", type=int, default=d.min_variant_size)
    a = ap.parse_args(argv)
    prm = ConsensusParams(
        cauchy_quantile=a.cauchy_quantile,
        outlier_quantile=a.outlier_quantile,
        max_pos_sd_m=a.max_pos_sd_m,
        max_roll_sd_rad=a.max_roll_sd_rad,
        min_variant_size=a.min_variant_size,
    )
    paths = [Path(p) for p in a.inputs]
    est = (
        load_estimates(paths[0])
        if len(paths) == 1 and paths[0].is_dir()
        else load_estimates(paths)
    )
    res = compute_consensus(
        est, prm, reference=_ref(a.reference_rig) if a.reference_rig else None
    )
    if a.out:
        res.save(a.out)
    print(res.summary())
    if a.prev:
        v = stop_rule(ConsensusResult.load(a.prev), res)
        print(
            "stop-rule:", "STOP" if v["stop"] else "continue", "; ".join(v["reasons"])
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
