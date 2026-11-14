"""Extended hit@10-by-source audit on the GT-bearing clips of the previous experiments.

The pseudo-label pool excludes (by construction) nearly every clip with human GT, so the pool itself has
only ~100 GT-covered frames. The 18 harness clips (9 test + 9 dev, plus arsenal_example) were already
run through B1 and S1 (conf >= 0.8, spacing 10, same gate) in docs/experiments/sam2_machine_anchors*;
this reuses those decisions to measure the pseudo-label error by source on every human GT frame that
the pipeline would label. Dev clips are detector-contaminated: B1 detection confidence there is
inflated, so those rows are an optimistic bound on B1 and are reported separately.

usage: pseudo_audit_ext.py --prep-dir <mprep> --dec-dir <ms1 with machine_decisions_*.json> --out audit_ext.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def band(src: str, conf, rule) -> str:
    if src == "B1_det":
        return "B1_det conf<0.6" if conf < 0.6 else ("B1_det conf 0.6-0.8" if conf < 0.8 else "B1_det conf>=0.8")
    if src == "S1_fill":
        return f"S1_fill {rule}"
    return "B1_interp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--dec-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="thr0.8_sp10")
    a = ap.parse_args()
    pd, dd = Path(a.prep_dir), Path(a.dec_dir)
    sets = [("test (clean)", json.loads((pd / "prep.json").read_text()), "small_crop512_test", "test"),
            ("dev (contaminated)", json.loads((pd / "prep.json").read_text()), "small_crop512_dev", "dev"),
            ("arsenal_example (independent)", json.loads((pd / "prep_example.json").read_text()), "small_crop512_indep", "indep")]
    out = {}
    for name, prep, tag, role in sets:
        dec = json.loads((dd / f"machine_decisions_{tag}.json").read_text())[a.key]
        agg = defaultdict(lambda: {"n": 0, "hit5": 0, "hit10": 0, "hit20": 0, "errs": []})
        for cid, c in prep["clips"].items():
            if c["role"] != role:
                continue
            nodes = {n["t"]: n for n in c["nodes"]}
            b1 = {int(t): v for t, v in c["b1"].items()}
            scored = set(c["scored"])
            for t, g in c["gt"].items():
                t = int(t)
                if t not in scored:
                    continue
                if t in nodes:
                    pos, key = nodes[t]["xy"], band("B1_det", nodes[t]["conf"], None)
                elif t in b1:
                    pos, key = b1[t], "B1_interp"
                elif str(t) in dec.get(cid, {}) and not dec[cid][str(t)]["crosses_cut"]:
                    r = dec[cid][str(t)]
                    pos, key = r["pos"], band("S1_fill", None, r["rule"])
                else:
                    agg["no pseudo point (untracked GT frame)"]["n"] += 1
                    continue
                e = math.hypot(pos[0] - g[0], pos[1] - g[1])
                for k in (key, "ALL pseudo points"):
                    s = agg[k]
                    s["n"] += 1
                    s["hit5"] += e <= 5
                    s["hit10"] += e <= 10
                    s["hit20"] += e <= 20
                    s["errs"].append(e)
        res = {}
        for k, v in sorted(agg.items()):
            if "errs" in v and v["errs"]:
                es = sorted(v["errs"])
                res[k] = {"n": v["n"], "hit@5": v["hit5"] / v["n"], "hit@10": v["hit10"] / v["n"], "hit@20": v["hit20"] / v["n"],
                          "median_err_px": es[len(es) // 2], "wrong_gt10": v["n"] - v["hit10"]}
            else:
                res[k] = {"n": v["n"]}
        out[name] = res
    Path(a.out).write_text(json.dumps(out, indent=1))
    for name, res in out.items():
        print(name)
        for k, v in res.items():
            print("  ", k, v)


if __name__ == "__main__":
    main()
