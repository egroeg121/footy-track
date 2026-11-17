"""Sub-experiment B: does a per-match ball reference descriptor improve true/false candidate discrimination?"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

D = os.path.join(os.path.dirname(__file__), "data"); O = os.path.dirname(__file__)
df = pd.read_pickle(f"{D}/ball_cands_labelled.pkl")
VC = [f"v{i}" for i in range(72)]
rng = np.random.default_rng(0)

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def feats(d, ref, sd):
    V = d[VC].values
    out = pd.DataFrame(index=d.index)
    out["lconf"] = logit(d.conf.values)
    z = (V - ref) / sd
    out["d_all"] = np.sqrt((z ** 2).mean(1))
    out["d_patch"] = np.sqrt((z[:, :64] ** 2).mean(1))
    out["d_hue"] = np.sqrt((z[:, 64:] ** 2).mean(1))
    out["d_euc"] = np.linalg.norm(V - ref, axis=1)
    return out

def ref_from(pos):
    V = pos[VC].values
    return V.mean(0), V.std(0) + 0.02   # floor std: patch dims are L2-normalised (~0.1 scale), hue bins are fractions

def prat(y, s, rec):
    p, r, _ = precision_recall_curve(y, s)
    return float(p[r >= rec].max()) if (r >= rec).any() else np.nan

def fpr_at_tpr(y, s, tpr):
    f, t, _ = roc_curve(y, s)
    return float(f[np.searchsorted(t, tpr)])

def metrics(y, s):
    return dict(n=len(y), pos=int(y.sum()), auroc=roc_auc_score(y, s), ap=average_precision_score(y, s),
                p_at_r50=prat(y, s, .5), p_at_r80=prat(y, s, .8), fpr_at_tpr80=fpr_at_tpr(y, s, .8))

def fit_predict(tr, te, cols, label="lab_gt"):
    ref, sd = ref_from(tr[tr[label] == 1])
    Xtr, Xte = feats(tr, ref, sd)[cols], feats(te, ref, sd)[cols]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, class_weight=None)).fit(Xtr, tr[label].values)
    return m.decision_function(Xte), m

VARIANTS = {"conf only": ["lconf"], "conf + d_all": ["lconf", "d_all"], "conf + d_patch + d_hue": ["lconf", "d_patch", "d_hue"],
            "d_all only (no detector)": ["d_all"]}
rows = []
def add(proto, subset, y, preds):
    for name, s in preds.items():
        rows.append(dict(protocol=proto, subset=subset, variant=name, **metrics(y, s)))

g = df[df.lab_gt.notna()].copy(); g["y"] = g.lab_gt.astype(int)
print("GT-labelled", len(g), g.y.sum())

# ---------- Protocol T: temporal. reference + LR from early segments, test on later segments
for split_seg, name in [(40, "T: fit seg<40, test seg>=40")]:
    tr, te = g[g.seg < split_seg], g[g.seg >= split_seg]
    # drop clips where no pos
    preds = {}
    for v, cols in VARIANTS.items():
        s, m = fit_predict(tr, te, cols, "lab_gt"); preds[v] = s
    preds["detector conf (raw)"] = te.conf.values
    for sub, mask in [("all", np.ones(len(te), bool)), ("detector-unseen frames (not in roboflow train)", (te.rf != "train").values)]:
        yy = te.y.values[mask]
        if yy.sum() >= 5 and (1 - yy).sum() >= 5:
            add(name, sub, yy, {k: v[mask] for k, v in preds.items()})
    print(name, "train", len(tr), int(tr.y.sum()), "test", len(te), int(te.y.sum()))

# ---------- Protocol L: leave-one-clip-out
clips = sorted(g["clip"].unique()); pos_clips = [c for c in clips if g[g["clip"] == c].y.sum() > 0]
print("clips with GT", len(clips), "with positives", len(pos_clips))
oof = {v: np.zeros(len(g)) for v in VARIANTS}; oof["detector conf (raw)"] = g.conf.values
for c in clips:
    te_m = (g["clip"] == c).values
    for v, cols in VARIANTS.items():
        s, _ = fit_predict(g[~te_m], g[te_m], cols); oof[v][te_m] = s
for sub, mask in [("all", np.ones(len(g), bool)), ("detector-unseen frames (not in roboflow train)", (g.rf != "train").values)]:
    add("L: leave-one-clip-out", sub, g.y.values[mask], {k: v[mask] for k, v in oof.items()})

# bootstrap over clips for delta AP / AUROC (LOCO, all)
def boot(variant, base="conf only", nb=500):
    out = {"ap": [], "auroc": []}
    cl = np.array(clips); idx = {c: np.where(g["clip"].values == c)[0] for c in clips}
    for _ in range(nb):
        ii = np.concatenate([idx[c] for c in rng.choice(cl, len(cl))])
        y = g.y.values[ii]
        if y.sum() < 5: continue
        out["ap"].append(average_precision_score(y, oof[variant][ii]) - average_precision_score(y, oof[base][ii]))
        out["auroc"].append(roc_auc_score(y, oof[variant][ii]) - roc_auc_score(y, oof[base][ii]))
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}
boots = {v: boot(v) for v in ["conf + d_all", "conf + d_patch + d_hue", "d_all only (no detector)"]}

# ---------- Protocol C: train on GT labels -> test on independent Ball Check verdicts (clips w/o GT overlap)
ck = df[df.lab_ck.notna()].copy(); ck["y"] = ck.lab_ck.astype(int)
gt_clips = set(g["clip"])
ck_ind = ck[~ck["clip"].isin(gt_clips)]
print("Ball Check independent rows", len(ck_ind), int(ck_ind.y.sum()))
preds = {}
for v, cols in VARIANTS.items():
    s, _ = fit_predict(g, ck_ind, cols); preds[v] = s
preds["detector conf (raw)"] = ck_ind.conf.values
add("C: fit on GT marks, test on Ball Check verdicts (other clips)", "all", ck_ind.y.values, preds)
# and the in-GT-clip verdict rows are excluded; also temporal split of the verdicts alone
# ---------- "Early reference from a few confirmed balls" sensitivity: how many ref balls needed
sens = []
tr_all, te_all = g[g.seg < 40], g[g.seg >= 40]
pos_tr = tr_all[tr_all.y == 1]
_, m_full = fit_predict(tr_all, te_all, ["lconf", "d_all"])
for k in [3, 5, 10, 20, 50, 100, len(pos_tr)]:
    aps, aucs = [], []
    for rep in range(30 if k < len(pos_tr) else 1):
        sub = pos_tr.sample(k, random_state=rep) if k < len(pos_tr) else pos_tr
        ref, sd = ref_from(sub)
        # LR fitted on early rows with the k-ball reference, test on later
        Xtr, Xte = feats(tr_all, ref, sd)[["lconf", "d_all"]], feats(te_all, ref, sd)[["lconf", "d_all"]]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, tr_all.y.values)
        s = m.decision_function(Xte)
        aps.append(average_precision_score(te_all.y, s)); aucs.append(roc_auc_score(te_all.y, s))
    sens.append(dict(n_ref_balls=k, ap_mean=np.mean(aps), ap_sd=np.std(aps), auroc_mean=np.mean(aucs)))
sens = pd.DataFrame(sens)
base_te = dict(ap=average_precision_score(te_all.y, te_all.conf), auroc=roc_auc_score(te_all.y, te_all.conf))

res = pd.DataFrame(rows); res.to_csv(f"{O}/b_results.csv", index=False)
sens.to_csv(f"{O}/b_ref_size_sensitivity.csv", index=False)
json.dump(dict(boot_LOCO_delta_vs_conf_only=boots, base_temporal_raw_conf=base_te), open(f"{O}/b_bootstrap.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(res.round(3).to_string())
print(json.dumps(boots, indent=1)); print(sens.round(3).to_string()); print(base_te)

# per-clip LOCO view + FP analysis: false candidates with high conf: does d_all reject them?
g["oof_conf"] = oof["conf only"]; g["oof_comb"] = oof["conf + d_all"]
# at threshold giving 80% recall for conf-only (pooled): how many FPs removed by combined at the same recall
def fp_at_recall(s, y, r=.8):
    thr = np.sort(s[y == 1])[int(np.floor((1 - r) * (y == 1).sum()))]
    return int(((s >= thr) & (y == 0)).sum()), thr
for nm in ["oof_conf", "oof_comb"]:
    print(nm, "FPs accepted at 80% recall:", fp_at_recall(g[nm].values, g.y.values))
ref, sd = ref_from(g[g.y == 1])
G = feats(g, ref, sd); g["d_all"] = G.d_all
print("median d_all  true:", g[g.y == 1].d_all.median().round(3), " false:", g[g.y == 0].d_all.median().round(3))
hc = g[(g.y == 0) & (g.conf >= 0.5)]
print("false cands with conf>=0.5:", len(hc), " of which d_all above true-median:", int((hc.d_all > g[g.y == 1].d_all.median()).sum()))
g.drop(columns=VC).to_csv(f"{O}/b_scored_candidates.csv", index=False)
