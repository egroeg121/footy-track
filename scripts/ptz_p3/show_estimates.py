"""Print rig_estimate.json of result dirs vs the GT-points reference rig (eval reference only). stdlib only."""
import glob, json, math, os, sys
REF = (0.100, -66.576, 15.743)
for pat in sys.argv[1:]:
    for d in sorted(glob.glob(pat)):
        f = os.path.join(d, "rig_estimate.json")
        m = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
        if not os.path.exists(f):
            print(os.path.basename(d), "no rig_estimate"); continue
        e = json.load(open(f))
        if not e.get("rig"):
            print(os.path.basename(d), "NO ESTIMATE:", e.get("reason"), "accepted", m.get("accepted_frac")); continue
        r = e["rig"]; dc = [a - b for a, b in zip(r[:3], REF)]
        print(os.path.basename(d), "C=(%.2f,%.2f,%.2f) err=%.2f m (%.2f,%.2f,%.2f) sd=(%.2f,%.2f,%.2f) conv=%s anch=%s ev=%s acc=%.2f" % (
            *r[:3], math.sqrt(sum(x * x for x in dc)), *dc, *e["sigma"][:3], e["converged"], e["n_anchors"], e["n_evidence_frames"], m.get("accepted_frac", -1)))
