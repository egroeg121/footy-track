"""Held-out recall split by source match (and arsenal_example), for any set of model groups (CPU, numpy + cv2 via report.py).

    python heldout_by_match.py --runs v1=/.../runs/v1 r2=/.../runs/r2 [r3=...] --dataset <k2> --out heldout_by_match.json \
        --groups "RT-DETR=ref" "A=A_s0,A_s1" "A+=Ap_s0,Ap_s1" "B+=Bp_s0,Bp_s1" ["A++=App_s0,App_s1"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import report as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=dir")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--groups", nargs="+", required=True, help="name=tag1,tag2")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = {}
    for kv in a.runs:
        runs.update(R.load_runs(Path(kv.split("=", 1)[1]), a.dataset))
    groups = {g.split("=", 1)[0]: g.split("=", 1)[1].split(",") for g in a.groups}
    groups = {g: t for g, t in groups.items() if all(x in runs for x in t)}
    any_tag = next(iter(groups.values()))[0]
    base = {k: v for k, v in runs[any_tag].items() if k != "__meta__"}
    held = {k: v for k, v in base.items() if v["split"] == "heldout" and v["gt"]}

    def is_bm(v):
        return "bournemouth" in (v["clip_id"] or "")

    def is_mc(v):
        return "mancity" in (v["clip_id"] or "") or v["clip_id"] == "arsenal_example"

    subsets = {
        "heldout all": list(held),
        "Bournemouth": [k for k, v in held.items() if is_bm(v)],
        "Bournemouth strict": [k for k, v in held.items() if is_bm(v) and v["strict_heldout"]],
        "Man City": [k for k, v in held.items() if is_mc(v)],
        "Man City without arsenal_example": [k for k, v in held.items() if is_mc(v) and v["clip_id"] != "arsenal_example"],
        "arsenal_example (heldout frames)": [k for k, v in held.items() if v["clip_id"] == "arsenal_example"],
    }
    per_img = {t: {k: R.per_image(v) for k, v in recs.items() if k != "__meta__"} for t, recs in runs.items()}
    out = {"n": {s: len(k) for s, k in subsets.items()}, "table": {}}
    for s, keys in subsets.items():
        out["table"][s] = {}
        for g, tags in groups.items():
            out["table"][s][g] = {
                "hit10": float(np.mean([np.mean([per_img[t][k]["hit"] for k in keys]) for t in tags])),
                "any10": float(np.mean([np.mean([per_img[t][k]["any"] for k in keys]) for t in tags])),
                "per_seed_hit10": {t: float(np.mean([per_img[t][k]["hit"] for k in keys])) for t in tags}}
    a.out.write_text(json.dumps(out, indent=1))
    print("subset (n) | " + " | ".join(groups))
    for s in subsets:
        print(f"{s} ({len(subsets[s])}) | " + " | ".join(f"{out['table'][s][g]['hit10']:.3f}/{out['table'][s][g]['any10']:.3f}" for g in groups))


if __name__ == "__main__":
    main()
