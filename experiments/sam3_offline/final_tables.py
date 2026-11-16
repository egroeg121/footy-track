"""All final-set tables for the write-up (mini CPU).  Reads the track files of the final runs and writes one JSON + prints markdown.

    python final_tables.py --spec spec_v2.json --oracle DIR --rtdetr DIR --sam2 TAG --sam3 TAG [--sam31 TAG] --out results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S  # noqa: E402
from analyze import load_all, perf_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--oracle", nargs="+", required=True)
ap.add_argument("--rtdetr", nargs="*", default=[])
ap.add_argument("--sam2", required=True)
ap.add_argument("--sam3", required=True)
ap.add_argument("--sam31", default="")
ap.add_argument("--extra", nargs="*", default=[], help="more oracle tags to tabulate (ablations), name=tag")
ap.add_argument("--max-reach", type=int, default=400)
ap.add_argument("--out", required=True)
ap.add_argument("--boot", type=int, default=3000)
a = ap.parse_args()
spec = json.loads(Path(a.spec).read_text())
FINAL = [n for n, c in spec["clips"].items() if c["set"] == "final"]
EXAMPLE = ["arsenal_example"]
methods = {"SAM2.1-small crop512": a.sam2, "SAM3 box crop512": a.sam3}
if a.sam31:
    methods["SAM3.1 box crop512"] = a.sam31
for kv in a.extra:
    k, v = kv.split("=", 1)
    methods[k] = v
res = {"oracle": {}, "rtdetr": {}}


def table(name, protocol, dirs, clips, max_reach):
    out, rows_by = {}, {}
    for label, tag in methods.items():
        docs = load_all(dirs, tag, protocol)
        docs = {k: v for k, v in docs.items() if k in clips}
        if not docs:
            continue
        rows = S.samples(spec, docs, clips, max_reach)
        rows_by[label] = rows
        s = S.summarize(rows, None, a.boot)
        s["perf"] = perf_of(docs)
        s["by_clip"] = S.by_clip(rows)
        s["gate_curve"] = S.gate_curve(rows)
        out[label] = s
    # paired differences on identical (clip, seed, frame)
    paired = {}
    labs = list(rows_by)
    for i, la in enumerate(labs):
        for lb in labs[:i]:
            base = {S.key_of(r): r for r in rows_by[lb] if r["kind"] == "ball"}
            com = [(r, base[S.key_of(r)]) for r in rows_by[la] if r["kind"] == "ball" and S.key_of(r) in base]
            if com:
                cl = np.array([c[0]["clip"] for c in com])
                paired[f"{la} - {lb}"] = {"n": len(com), **S.paired_diff(S.hits([c[0] for c in com]), S.hits([c[1] for c in com]), cl, a.boot)}
    res[protocol][name] = {"methods": out, "paired": paired}
    return rows_by


rows_o = table("held-out + Aston Villa dense (no arsenal_example)", "oracle", a.oracle, FINAL, a.max_reach)
table("arsenal_example", "oracle", a.oracle, EXAMPLE, a.max_reach)
table("held-out only, no astonvilla (sparse GT)", "oracle", a.oracle, [c for c in FINAL if not c.startswith("astonvilla")], a.max_reach)
table("astonvilla_seg080 (dense run)", "oracle", a.oracle, ["astonvilla_seg080"], a.max_reach)
# gates on the oracle run: S1 (pairs of seeds), SAM3-SAM2 agreement
gates = {}
for label, tag in methods.items():
    docs = {k: v for k, v in load_all(a.oracle, tag, "oracle").items() if k in FINAL}
    if docs:
        gates[label] = {"s1_5px": S.s1_summary(S.s1_rows(spec, docs, 5.0, a.max_reach)), "s1_10px": S.s1_summary(S.s1_rows(spec, docs, 10.0, a.max_reach))}
if "SAM3 box crop512" in rows_o and "SAM2.1-small crop512" in rows_o:
    ag = S.agree_rows(rows_o["SAM3 box crop512"], rows_o["SAM2.1-small crop512"], 5.0)
    gates["SAM3 gated by SAM2 agreement (5 px)"] = S.summarize(ag, None, a.boot)
res["oracle"]["gates"] = gates
if a.rtdetr:
    table("held-out realistic (RT-DETR conf>=0.8 seeds), no example", "rtdetr", a.rtdetr, FINAL, 250)
    table("arsenal_example realistic", "rtdetr", a.rtdetr, EXAMPLE, 250)
    # seed check separation
    sc = []
    for label, tag in methods.items():
        if "SAM3" not in label:
            continue
        for clip, d in load_all(a.rtdetr, tag, "rtdetr").items():
            for t in d["tracks"]:
                if t.get("seed_check"):
                    sc.append((clip, t["seed"], t["dir"], t["seed_check"]["iou"], t["seed_check"]["score"]))
    res["rtdetr"]["seed_check_rows"] = sc
Path(a.out).write_text(json.dumps(res, indent=1))
for proto in ("oracle", "rtdetr"):
    for name, blk in res[proto].items():
        if name in ("gates", "seed_check_rows"):
            continue
        print(f"\n### {proto}: {name}")
        print("| method | n (clips/seeds) | recall@10 [95% CI frames] | " + " | ".join(S.BIN_NAMES) + " | coverage | prec(emitted) | no-ball silent | first loss mean/median (censored) | fps |")
        print("|" + "---|" * 11)
        for lab, s in blk["methods"].items():
            bd = " | ".join(("%.3f (n=%d)" % (s["by_dist"][b]["recall10"], s["by_dist"][b]["n"])) if s["by_dist"][b]["n"] else "-" for b in S.BIN_NAMES)
            nb = s.get("no_ball")
            fl = s["first_loss"]
            r = s["recall10"]
            print(f"| {lab} | {s['n_samples']} ({s['n_clips']}/{s['n_seeds']}) | {r['mean']:.3f} [{r['ci_frame'][0]:.3f}, {r['ci_frame'][1]:.3f}] | {bd} | {s['coverage']:.3f} | "
                  f"{'-' if s['precision_emitted'] is None else '%.3f' % s['precision_emitted']} | {('%.2f (n=%d)' % (nb['silent_rate'], nb['n'])) if nb else '-'} | "
                  f"{fl.get('mean', float('nan')):.0f} / {fl.get('median', float('nan')):.0f} ({fl.get('censored_frac', float('nan')):.2f}) | {s['perf']['fps'] and round(s['perf']['fps'], 2)} |")
        for k, v in blk["paired"].items():
            print(f"  paired {k}: {v['diff']:+.3f} frame CI [{v['ci_frame'][0]:+.3f}, {v['ci_frame'][1]:+.3f}]" + (f" clip CI [{v['ci_clip'][0]:+.3f}, {v['ci_clip'][1]:+.3f}]" if "ci_clip" in v else "") + f" n={v['n']}")
print("\n### gates", json.dumps(res["oracle"]["gates"], indent=1)[:3000])
