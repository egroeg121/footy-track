"""Mini-side analysis of sam3_offline track files: one table row per config (markdown + JSON).

    python analyze.py --spec spec.json --tracks DIR [DIR..] --tags tagA tagB --clips a,b,c --max-reach 120 --ref tagA --out result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S  # noqa: E402


def perf_of(docs: dict) -> dict:
    tf = sum(d.get("perf", {}).get("frames_propagated", 0) for d in docs.values())
    ts = sum(d.get("perf", {}).get("seconds", 0.0) for d in docs.values())
    vr = max([d.get("perf", {}).get("peak_vram_alloc_mb", 0) for d in docs.values()] or [0])
    return {"frames": tf, "sec": ts, "fps": (tf / ts) if ts else None, "peak_vram_mb": vr}


def load_all(dirs: list[str], tag: str, protocol: str) -> dict:
    docs = {}
    for d in dirs:
        docs.update(S.load_tracks(d, tag, protocol))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--tracks", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--protocol", default="oracle")
    ap.add_argument("--clips", default="")
    ap.add_argument("--max-reach", type=int, default=0)
    ap.add_argument("--ref", default="")
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    clips = [c for c in a.clips.split(",") if c] or None
    res = {"tags": {}, "paired": {}}
    allrows = {}
    for tag in a.tags:
        docs = load_all(a.tracks, tag, a.protocol)
        if clips:
            docs = {k: v for k, v in docs.items() if k in clips}
        if not docs:
            print("no tracks for", tag)
            continue
        rows = S.samples(spec, docs, clips, a.max_reach or None)
        allrows[tag] = rows
        summ = S.summarize(rows, a.gate, a.boot)
        summ["perf"] = perf_of(docs)
        summ["by_clip"] = S.by_clip(rows, a.gate)
        summ["gate_curve"] = S.gate_curve(rows)
        res["tags"][tag] = summ
    hdr = "| config | n (clips/seeds) | recall@10 [CI frame] | " + " | ".join(f"r@10 {b}" for b in S.BIN_NAMES) + " | coverage | prec(emitted) | no-ball silent | first loss (mean/med, cens.) | fps | VRAM MB |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for tag, s in res["tags"].items():
        bd = " | ".join(("%.3f (n=%d)" % (s["by_dist"][b]["recall10"], s["by_dist"][b]["n"])) if s["by_dist"][b]["n"] else "-" for b in S.BIN_NAMES)
        nb = s.get("no_ball")
        fl = s["first_loss"]
        r = s["recall10"]
        print(f"| {tag} | {s['n_samples']} ({s['n_clips']}/{s['n_seeds']}) | {r['mean']:.3f} [{r['ci_frame'][0]:.3f},{r['ci_frame'][1]:.3f}] | {bd} | {s['coverage']:.3f} | "
              f"{s['precision_emitted'] if s['precision_emitted'] is None else round(s['precision_emitted'], 3)} | {('%.2f (n=%d)' % (nb['silent_rate'], nb['n'])) if nb else '-'} | "
              f"{fl.get('mean', float('nan')):.0f}/{fl.get('median', float('nan')):.0f}, {fl.get('censored_frac', float('nan')):.2f} | {s['perf']['fps'] and round(s['perf']['fps'], 2)} | {s['perf']['peak_vram_mb']} |")
    if a.ref and a.ref in allrows:
        base = {S.key_of(r): r for r in allrows[a.ref] if r["kind"] == "ball"}
        for tag, rows in allrows.items():
            if tag == a.ref:
                continue
            common = [(r, base[S.key_of(r)]) for r in rows if r["kind"] == "ball" and S.key_of(r) in base]
            if not common:
                continue
            ha = S.hits([c[0] for c in common], a.gate)
            hb = S.hits([c[1] for c in common], a.gate)
            cl = np.array([c[0]["clip"] for c in common])
            res["paired"][f"{tag} - {a.ref}"] = {"n": len(common), **S.paired_diff(ha, hb, cl, a.boot)}
        print("\npaired recall@10 differences (same seeds & GT frames):")
        for k, v in res["paired"].items():
            print(f"  {k}: {v['diff']:+.3f}  frame CI [{v['ci_frame'][0]:+.3f},{v['ci_frame'][1]:+.3f}]  clip CI {['%+.3f' % x for x in v.get('ci_clip', [])]}  n={v['n']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
