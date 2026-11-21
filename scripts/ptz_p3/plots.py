"""P3 plots: L1 rig_i scatter (C_y vs C_z) and the depth-split GT error by arm. matplotlib + numpy."""
import glob, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = "/mnt/storage/footy_data/scratch/temporal"
OUT = T + "/p3/plots"
import os; os.makedirs(OUT, exist_ok=True)
GT = (-66.576, 15.743); PRIOR = (-63.0, 14.0); FIT95 = (-67.033, 15.977)
rs = json.load(open(T + "/p3/l2/consensus_infl30.json"))["rig"]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for pat, lab, col in ((T + "/arsenal_mancity_seg*@generic_gantry_prior_v2-1be31b96", "L1 single start (162 segs)", "#8da0cb"),
                      (T + "/p3/msL1/*", "L1 multistart (24 segs)", "#fc8d62"), (T + "/p3/ms/*", "L1 multistart (4 segs)", "#fc8d62")):
    pts = []
    for f in glob.glob(pat + "/rig_estimate.json"):
        e = json.load(open(f))
        if e.get("rig"): pts.append((e["rig"][1], e["rig"][2], e["converged"]))
    if not pts: continue
    p = np.array(pts, float)
    ax[0].scatter(p[:, 0], p[:, 1], s=14, c=col, alpha=.8, label=lab if "(4" not in lab else None, edgecolors=np.where(p[:, 2] > 0, "k", "none"), linewidths=.5)
ax[0].scatter(*GT, marker="*", s=200, c="g", label="GT-points rig (eval ref only)")
ax[0].scatter(*PRIOR, marker="s", s=70, c="k", label="generic prior (start)")
ax[0].scatter(*FIT95[:2], marker="^", s=70, c="m", label="existing 95-frame fit")
ax[0].scatter(rs[1], rs[2], marker="D", s=70, c="r", label="rig* (consensus, infl 30)")
ax[0].set_xlabel("C_y (m)"); ax[0].set_ylabel("C_z (m)"); ax[0].legend(fontsize=7); ax[0].set_title("Per-segment rig estimates (black edge = converged)")
arms = {"A emirates GT rig (contaminated)": "A10", "B existing 95-frame rig": "B", "C two-level rig* (single-start L1)": "C", "C2 two-level rig* (multistart L1)": "C2"}
w = .2
for k, (lab, f) in enumerate(arms.items()):
    d = json.load(open(f"{T}/p3/score/{f}.json"))["summary"]
    v = [d[n]["median"] for n in ("near", "mid", "far")]; q = [d[n]["p95"] for n in ("near", "mid", "far")]
    ax[1].bar(np.arange(3) + k * w, v, w, label=lab); ax[1].scatter(np.arange(3) + k * w, q, c="k", s=12, zorder=3)
ax[1].set_xticks(np.arange(3) + w); ax[1].set_xticklabels(["near", "mid", "far"]); ax[1].set_ylabel("metres (bar = median, dot = p95)")
ax[1].set_title("GT error by pitch depth (40 frames, 10 segments)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT + "/p3_overview.png", dpi=110); print("ok")
