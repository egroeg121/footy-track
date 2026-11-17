"""Four-tracker agreement breakdown + VitTrack behaviour (mini; numpy for the player boxes).

    python report_stats4.py --fused fused4_arm2.json.gz --npz CLIPS_OUT --out stats4_arm2.json
"""
import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np

import fusion as F
import fusion4 as G

ap = argparse.ArgumentParser()
ap.add_argument("--fused", type=Path, required=True)
ap.add_argument("--npz", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
fused = F.load(a.fused)
tab = G.table(fused)
N = sum(1 for rows in tab.values() for r in rows if r is not None)
kinds, roles, cov = Counter(), Counter(), Counter()
odd_by = Counter()
pw = {}
for c, rows in tab.items():
    cl = fused["clips"][c]
    for t, r in enumerate(rows):
        if r is None:
            continue
        k, info = r
        key = k if k not in ("agree_all",) else f"agree_all_{info['n_present']}of4"
        kinds[key] += 1
        roles[G.vit_role(k, info)] += 1
        pos = G.pos_at(cl, t)
        for nm, p in zip(G.NAMES, pos):
            if p is not None:
                cov[nm] += 1
        if k == "odd_one":
            for o in (info["odd"] if isinstance(info["odd"], list) else [info["odd"]]):
                odd_by[o] += 1
        for i in range(4):
            for j in range(i + 1, 4):
                if pos[i] and pos[j]:
                    e = pw.setdefault(f"{G.NAMES[i]}-{G.NAMES[j]}", [0, 0])
                    e[0] += 1
                    e[1] += G.dist(pos[i], pos[j]) <= G.TOL
out = {"frames": N, "kinds": dict(kinds), "vit_role": dict(roles), "coverage": dict(cov), "odd_one_out_when_exactly_one": dict(odd_by),
       "pairwise_agree": {k: {"both": v[0], "agree_8px": v[1]} for k, v in pw.items()}, "vit": fused.get("vit")}
# VitTrack score profile, coverage at thresholds, loss events, would_stop, drift onto player
sc = []
thr = {0.0: 0, 0.2: 0, 0.3: 0, 0.5: 0}
stops = Counter()
tot_raw = 0
for c, cl in fused["clips"].items():
    t4 = cl.get("t4")
    if not t4:
        continue
    for t in range(cl["n"]):
        if t4["raw_x"][t] is None or F.excluded(fused, c, t):
            continue
        tot_raw += 1
        s = t4["score"][t]
        sc.append(s)
        for th in thr:
            thr[th] += s >= th
        if t4["would_stop"][t]:
            stops[t4["would_stop"][t]] += 1
sc.sort()
out["vit_score"] = {"frames_with_box": tot_raw, "quantiles": {q: round(sc[int(len(sc) * q)], 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if sc else None, "share_ge": {str(k): round(v / max(tot_raw, 1), 3) for k, v in thr.items()},
                    "would_stop_frames": dict(stops)}
# loss: VitTrack had a position (score >= min) and loses it (>= 3 frames) while another tracker still has the ball
loss = 0
for c, cl in fused["clips"].items():
    t4 = cl.get("t4")
    if not t4:
        continue
    n = cl["n"]
    for t in range(n - 3):
        if t4["x"][t] is not None and t4["x"][t + 1] is None and all(t4["x"][min(t + k, n - 1)] is None for k in (2, 3)) and any(cl[k]["x"][t + 1] is not None for k in ("t1", "t2", "t3") if cl[k]):
            loss += 1
out["vit_loss_events"] = loss
# drift: VitTrack is the odd one out (>= 2 others agree) -- is its position inside a player box, vs the consensus position?
inside_vit = inside_cons = n_odd = feet_vit = feet_cons = 0
players = {}
for c, cl in fused["clips"].items():
    z = np.load(a.npz / f"{c}.npz")
    pp, parr = z["players_ptr"], z["players_xywh"].astype(float)
    def inside(t, p, feet=False):
        pl = parr[pp[t]:pp[t + 1]].reshape(-1, 4)
        if not len(pl):
            return False
        m = (abs(pl[:, 0] - p[0]) < pl[:, 2] / 2 + 3) & (abs(pl[:, 1] - p[1]) < pl[:, 3] / 2 + 3)
        if feet:       # lower 30% of the box: boots / lower legs
            m &= p[1] > pl[:, 1] + pl[:, 3] * 0.2
        return bool(np.any(m))
    for t, r in enumerate(tab[c]):
        if r is None or r[0] != "odd_one" or r[1]["odd"] != "t4":
            continue
        pos = G.pos_at(cl, t)
        if cl["t4"] and cl["t4"]["is_seed"][t]:
            continue
        n_odd += 1
        inside_vit += inside(t, pos[3])
        agree = [pos[G.NAMES.index(k)] for k in r[1]["agree"]]
        inside_cons += inside(t, agree[0])
        feet_vit += inside(t, pos[3], True)
        feet_cons += inside(t, agree[0], True)
out["vit_odd_drift"] = {"vit_odd_one_out_frames": n_odd, "vit_inside_player_box": inside_vit, "consensus_inside_player_box": inside_cons, "vit_in_lower_30pct_of_player_box": feet_vit, "consensus_in_lower_30pct_of_player_box": feet_cons}
# agreed-frame baseline: how often does the 3-tracker consensus (t1+t2+t3 agree) sit inside a player box
Path(a.out).write_text(json.dumps(out, indent=1))
print(f"frames {N}")
for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
    print(f"| {k} | {v} | {100 * v / N:.1f}% |")
print("vit role:", {k: f"{v} ({100 * v / N:.1f}%)" for k, v in roles.most_common()})
print("coverage:", {G.LABEL[k]: f"{cov[k]} ({100 * cov[k] / N:.1f}%)" for k in G.NAMES})
print("odd one out (exactly one):", {G.LABEL[k]: v for k, v in odd_by.items()})
print("pairwise:", out["pairwise_agree"])
print("vit score:", out["vit_score"])
print("loss events:", loss, "drift:", out["vit_odd_drift"])
