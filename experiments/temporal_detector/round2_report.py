"""Round 2: v1 detectors (A 3ch, B 5ch) vs the same recipe with filtered pseudo labels added (A+, B+). CPU, numpy only.

    uv run --no-project --python /usr/bin/python3 --with numpy --with opencv-python-headless python round2_report.py \
        --v1 /mnt/storage/exp/temporal/runs/v1 --r2 /mnt/storage/exp/temporal/runs/r2 \
        --dataset /mnt/storage/footy_data/multiframe_v1/k2 --out docs/experiments/temporal_detector_round2 --meta "$(cat narrative)"

Same evaluation as v1 (`report.py`: hit@10 top-1, any@10 candidate recall, AP50; val / test / heldout / strict / val clip
arsenal_example / test+heldout motion terciles) plus FP rate of the top-1 detection on ball-free images, seed spread, and
paired bootstrap CIs (images) for A+ - A, B+ - B, B+ - A+, B - A and the difference of differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import report as R

MODELS = {
    "RT-DETR-L v11 (production)": ["ref"],
    "A  3ch, v1": ["A_s0", "A_s1"],
    "A+ 3ch, + pseudo": ["Ap_s0", "Ap_s1"],
    "B  5ch, v1": ["B_s0", "B_s1"],
    "B+ 5ch, + pseudo": ["Bp_s0", "Bp_s1"],
}
SUBS = ["val", "val / clip arsenal_example (74% of val balls)", "val / all other clips", "test", "heldout", "heldout_strict", "test+heldout",
        "test+heldout / ball motion low", "test+heldout / ball motion mid", "test+heldout / ball motion high"]
PAIRS = [("A+ - A (v1)", "A  3ch, v1", "A+ 3ch, + pseudo"), ("B+ - B (v1)", "B  5ch, v1", "B+ 5ch, + pseudo"),
         ("B+ - A+ (round 2)", "A+ 3ch, + pseudo", "B+ 5ch, + pseudo"), ("B - A (v1)", "A  3ch, v1", "B  5ch, v1")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--r2", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", default="")
    a = ap.parse_args()
    runs = {**R.load_runs(a.v1, a.dataset), **R.load_runs(a.r2, a.dataset)}
    models = {m: t for m, t in MODELS.items() if all(x in runs for x in t)}
    first = "A_s0"
    base = [v for k, v in runs[first].items() if k != "__meta__"]
    ms = np.array([max(i["motion_prev"], i["motion_next"]) for i in base if i["split"] in ("test", "heldout") and i["gt"]
                   and i["prev_status"] == "ok" and i["next_status"] == "ok"])
    S = R.subsets(base, (float(np.percentile(ms, 100 / 3)), float(np.percentile(ms, 200 / 3))))
    by_key = {k: v for k, v in runs[first].items() if k != "__meta__"}
    S["ball-free (val+test+heldout)"] = [k for k in S["val+test+heldout"] if not by_key[k]["gt"]]
    pim = {t: {k: R.per_image(v) for k, v in recs.items() if k != "__meta__"} for t, recs in runs.items()}
    rng = np.random.default_rng(0)

    def block(tags, keys):
        gk = [k for k in keys if by_key[k]["gt"]]
        out = {"n_gt_img": len(gk)}
        if not gk:
            return out
        per_seed = {}
        for t in tags:
            per_seed[t] = {"hit10": float(np.mean([pim[t][k]["hit"] for k in gk])), "any10": float(np.mean([pim[t][k]["any"] for k in gk])),
                           "ap50": R.ap50([runs[t][k] for k in keys])}
        out["per_seed"] = per_seed
        for m in ("hit10", "any10", "ap50"):
            v = [per_seed[t][m] for t in tags]
            out[m] = float(np.mean(v))
            out[m + "_seed_range"] = float(max(v) - min(v))
        return out

    table = {s: {m: block(t, S[s]) for m, t in models.items()} for s in SUBS if s in S}
    # FP on ball-free images
    free = S["ball-free (val+test+heldout)"]
    fp = {}
    for m, tags in models.items():
        fp[m] = {f"top1_conf>={thr}": float(np.mean([np.mean([pim[t][k]["top_conf"] >= thr for k in free]) for t in tags])) for thr in (0.1, 0.25, 0.5, 0.7)}
    # wrong top-1 on ball frames at conf >= 0.5 (confident false picks), val+test+heldout
    gtk = [k for k in S["val+test+heldout"] if by_key[k]["gt"]]
    wrong = {m: float(np.mean([np.mean([(pim[t][k]["top_conf"] >= 0.5) and pim[t][k]["hit"] == 0 for k in gtk]) for t in tags])) for m, tags in models.items()}

    def hits(tags, keys, what="hit"):
        return np.array([np.mean([pim[t][k][what] for t in tags]) for k in keys])

    paired = {}
    for s in SUBS:
        if s not in S:
            continue
        gk = [k for k in S[s] if by_key[k]["gt"]]
        if not gk:
            continue
        clips = np.array([by_key[k]["clip_id"] or "none" for k in gk])
        paired[s] = {"n": len(gk), "n_clips": int(len(set(clips)))}
        for name, ma, mb in PAIRS:
            if ma in models and mb in models:
                ha, hb = hits(models[ma], gk), hits(models[mb], gk)
                aa, ab = hits(models[ma], gk, "any"), hits(models[mb], gk, "any")
                paired[s][name] = {"hit10": R.bootstrap_diff(ha, hb, None, rng), "hit10_clip": R.bootstrap_diff(ha, hb, clips, rng),
                                   "any10": R.bootstrap_diff(aa, ab, None, rng)}
        # difference of differences (B+ - A+) - (B - A)
        if all(m in models for m in ("A  3ch, v1", "B  5ch, v1", "A+ 3ch, + pseudo", "B+ 5ch, + pseudo")):
            d1 = hits(models["B  5ch, v1"], gk) - hits(models["A  3ch, v1"], gk)
            d2 = hits(models["B+ 5ch, + pseudo"], gk) - hits(models["A+ 3ch, + pseudo"], gk)
            paired[s]["DiD (B+ - A+) - (B - A)"] = {"hit10": R.bootstrap_diff(d1, d2, None, rng), "hit10_clip": R.bootstrap_diff(d1, d2, clips, rng)}
    # training curves
    curves = {}
    for d in (a.v1, a.r2):
        for p in sorted(d.glob("curve_*.jsonl")):
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            first_row = {}
            for r in rows:
                if "ball_ap50" in r:
                    first_row.setdefault(r["epoch"], r)
            ap_ = [r["ball_ap50"] for r in first_row.values()]
            curves[p.stem.removeprefix("curve_")] = {"epochs": len(ap_), "last": ap_[-1], "mean_last10": float(np.mean(ap_[-10:])),
                                                     "best": max(ap_)}
    res = {"models": models, "table": table, "paired": paired, "fp_ball_free": fp, "n_ball_free": len(free),
           "confident_wrong_top1_on_ball_frames_conf>=0.5": wrong, "curves": curves}
    Path(str(a.out) + ".json").write_text(json.dumps(res, indent=1, default=lambda o: None if o != o else float(o)))
    write_md(res, a)
    print("wrote", str(a.out) + ".md")


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def write_md(res, a):
    L = ["# Round 2: does pseudo-labelled data (v11 train + filtered pseudo) change the 3ch / 5ch comparison?\n"]
    if a.meta:
        L.append(a.meta + "\n")
    for metric, title in (("hit10", "Top-1 recall@10 px (hit@10), mean of 2 seeds [seed range]"), ("any10", "Candidate recall any@10"),
                          ("ap50", "Ball AP50")):
        L.append(f"## {title}\n")
        subs = list(res["table"])
        first_m = next(iter(res["models"]))
        L.append("| model | " + " | ".join(f"{s} (n={res['table'][s][first_m]['n_gt_img']})" for s in subs) + " |")
        L.append("|---|" + "---|" * len(subs))
        for m in res["models"]:
            cells = []
            for s in subs:
                b = res["table"][s][m]
                rng_ = b.get(metric + "_seed_range")
                cells.append(f"{fmt(b.get(metric))}" + (f" [{rng_:.3f}]" if rng_ and len(res['models'][m]) > 1 else ""))
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("## Paired differences in hit@10 (bootstrap over images; seed-mean outcomes; clip bootstrap in brackets)\n")
    names = [p[0] for p in PAIRS] + ["DiD (B+ - A+) - (B - A)"]
    L.append("| subset | n | " + " | ".join(names) + " |")
    L.append("|---|---|" + "---|" * len(names))
    for s, pr in res["paired"].items():
        cells = []
        for n_ in names:
            d = pr.get(n_)
            if not d:
                cells.append("")
                continue
            h, c = d["hit10"], d["hit10_clip"]
            cells.append(f"{h[0]:+.3f} [{h[1]:+.3f}, {h[2]:+.3f}] ({c[1]:+.3f}, {c[2]:+.3f})")
        L.append(f"| {s} | {pr['n']} | " + " | ".join(cells) + " |")
    L.append("\n## Paired differences in candidate recall any@10\n")
    names2 = [p[0] for p in PAIRS]
    L.append("| subset | " + " | ".join(names2) + " |")
    L.append("|---|" + "---|" * len(names2))
    for s, pr in res["paired"].items():
        L.append(f"| {s} | " + " | ".join(f"{pr[n_]['any10'][0]:+.3f} [{pr[n_]['any10'][1]:+.3f}, {pr[n_]['any10'][2]:+.3f}]" if n_ in pr else "" for n_ in names2) + " |")
    L.append(f"\n## False positives on ball-free images (n = {res['n_ball_free']}, val+test+heldout)\n")
    L.append("| model | top-1 conf >= 0.1 | >= 0.25 | >= 0.5 | >= 0.7 | confident wrong top-1 on ball frames (conf >= 0.5, fraction of ball frames) |")
    L.append("|---|---|---|---|---|---|")
    for m, v in res["fp_ball_free"].items():
        L.append(f"| {m} | " + " | ".join(fmt(x) for x in v.values()) + f" | {fmt(res['confident_wrong_top1_on_ball_frames_conf>=0.5'][m])} |")
    L.append("\n## Training curves (val ball AP50)\n")
    L.append("| run | epochs | last | mean last 10 | best |")
    L.append("|---|---|---|---|---|")
    for t, c in res["curves"].items():
        L.append(f"| {t} | {c['epochs']} | {c['last']:.3f} | {c['mean_last10']:.3f} | {c['best']:.3f} |")
    Path(str(a.out) + ".md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
