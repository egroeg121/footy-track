"""Agreement breakdown / coverage / odd-one-out of the fused tracker positions -> json + markdown tables (mini; stdlib)."""
import json, sys
from pathlib import Path
import fusion as F

fused = F.load(Path(sys.argv[1]))
tab = F.frame_table(fused)
b = F.breakdown(fused, tab)
n = b["frames"]
out = {"breakdown": b, "per_clip": {}}
L = [f"frames {n} in {len(tab)} clips"]
for k in F.KINDS:
    L.append(f"| {k} | {b['kinds'].get(k, 0)} | {100 * b['kinds'].get(k, 0) / n:.1f}% |")
L.append("coverage: " + ", ".join(f"{F.LABEL[k]} {b['coverage'][k]} ({100 * b['coverage'][k] / n:.1f}%)" for k in F.NAMES))
tot3 = b["kinds"].get("two_agree", 0)
L.append("odd one out in two_agree (n=%d): " % tot3 + ", ".join(f"{F.LABEL[k]} {b['odd_one_out'].get(k, 0)}" for k in F.NAMES))
L.append("only_one: " + json.dumps(b["only_one"]))
L.append("pair_by_absent: " + json.dumps(b["pair_by_absent"]))
# per-tracker coverage restricted to frames where >=1 other tracker present; pairwise agreement among frames both present
pw = {}
for i, a in enumerate(F.NAMES):
    for j in range(i + 1, 3):
        c = ag = 0
        dists = []
        for cl_id, cl in fused["clips"].items():
            for t in range(cl["n"]):
                if tab[cl_id][t] is None: continue
                p = F.pos_at(cl, t)
                if p[i] and p[j]:
                    c += 1; d = F.dist(p[i], p[j]); ag += d <= F.TOL; dists.append(d)
        dists.sort()
        pw[f"{a}-{F.NAMES[j]}"] = {"both": c, "agree_8px": ag, "median_px_when_differ": (sorted(d for d in dists if d > F.TOL)[len([d for d in dists if d > F.TOL]) // 2] if any(d > F.TOL for d in dists) else None)}
out["pairwise"] = pw
L.append("pairwise: " + json.dumps(pw))
# seed-source split for SAM presence
src = {"t2": {}, "t3": {}}
for cl in fused["clips"].values():
    for k in ("t2", "t3"):
        tr = cl[k]
        if tr is None: continue
        for t in range(cl["n"]):
            if tr["x"][t] is not None:
                s = tr["seed_src"][t]; src[k][s] = src[k].get(s, 0) + 1
out["sam_seed_src"] = src
L.append("sam seed source of present frames: " + json.dumps(src))
# clips where a SAM tracker is present on (almost) every frame but A++/B1 never: possible false-seed clips
fl = []
for c, cl in fused["clips"].items():
    n_ = cl["n"]; pres = {k: sum(x is not None for x in cl[k]["x"]) / n_ if cl[k] else None for k in F.NAMES}
    out["per_clip"][c] = {"n": n_, **{k: round(v, 3) if v is not None else None for k, v in pres.items()}}
    segs = [s for s in F.disagreement_segments({"clips": {c: cl}, "exclude": {}}, {c: tab[c]}) ]
    out["per_clip"][c]["dis_segments"] = len(segs); out["per_clip"][c]["dis_frames"] = sum(s["n_dis"] for s in segs)
segs = F.disagreement_segments(fused, tab)
out["segments"] = {"n": len(segs), "frames": sum(s["n_dis"] for s in segs), "length_median": sorted(s["n_dis"] for s in segs)[len(segs) // 2] if segs else 0, "length_max": max((s["n_dis"] for s in segs), default=0)}
L.append("disagreement segments: " + json.dumps(out["segments"]))
ev = F.presence_events(fused, tab)
out["loss_events"] = {k: sum(e["tracker"] == k for e in ev) for k in F.NAMES}
L.append("loss events: " + json.dumps(out["loss_events"]))
print("\n".join(L))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
