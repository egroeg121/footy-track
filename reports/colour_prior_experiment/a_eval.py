"""Sub-experiment A: per-match colour prior as post-hoc rescoring of referee vs player (no retraining).

Inputs : data/dets_cpu.jsonl (STORED machine_labels RT-DETR-L 1920 detections mapped onto the GT frames by a_map_dets.py; 7 class scores per box, floor 0.10), roboflow v11 GT labels,
         data/frames_meta.tsv (frame index, split, early|late). Palette frames = 'early' (match frames < 1500);
         evaluation frames = 'late' TEST-split frames (>= 1500; the detector never trained on them) -> time-disjoint from palette frames.
"""
import json, os, sys, collections, re
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(2)
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/data"
FD = "/mnt/storage/footy_data/detection_dataset/roboflow_dataset_11"
NAMES = {0: "ball", 1: "coach", 2: "in_play_ball", 3: "person", 4: "player", 5: "player_sub", 6: "referee"}
PERSON = [1, 3, 4, 5, 6]
SEED = 0
NDRAWS = int(os.environ.get("NDRAWS", 20))
BETAS = [0.25, 0.5, 1.0, 2.0]
FEATVAR = sys.argv[1] if len(sys.argv) > 1 else "hsv4"     # hsv4 (design doc encoding) | sat3 (saturation-weighted hue)
OUTTAG = "" if FEATVAR == "hsv4" else f"_{FEATVAR}"

meta = pd.read_csv(f"{D}/frames_meta.tsv", sep="\t", header=None, names=["image", "frame", "split", "phase"])
meta = meta.set_index("image")

def iou(a, b):  # xywh top-left arrays
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[0] + a[2], b[:, 0] + b[:, 2]); y2 = np.minimum(a[1] + a[3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (a[2] * a[3] + b[:, 2] * b[:, 3] - inter + 1e-12)

def torso_feat(img, box):
    h_px, w_px = img.shape[:2]
    x, y, w, h = box
    x1, x2 = int((x + .25 * w) * w_px), int((x + .75 * w) * w_px)
    y1, y2 = int((y + .20 * h) * h_px), int((y + .50 * h) * h_px)
    c = img[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
    if c.size == 0: return None
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    # exclude near-pitch-green pixels? kept raw to match the design-doc M6 recipe
    hh = np.deg2rad(hsv[:, 0] * 2.0); s = hsv[:, 1] / 255; v = hsv[:, 2] / 255
    if FEATVAR == "hsv4":
        return np.array([np.cos(hh).mean(), np.sin(hh).mean(), s.mean(), v.mean()])  # mean of encoded hue, circular-safe
    return np.array([(s * np.cos(hh)).mean(), (s * np.sin(hh)).mean(), v.mean()])

# ---------------------------------------------------------------- load detections + GT
frames = {}
for l in open(f"{D}/dets_cpu.jsonl"):
    r = json.loads(l); b = r["image"]
    if b not in meta.index: continue
    q = np.array(r["q"], dtype=np.float32).reshape(-1, 11)
    frames[b] = dict(q=q, split=meta.loc[b, "split"], phase=meta.loc[b, "phase"], frame=int(meta.loc[b, "frame"]))
print("frames with detections:", len(frames), collections.Counter(v["phase"] for v in frames.values()))

def load_gt(b):
    split = frames[b]["split"]
    p = f"{FD}/{split}/labels/{b.rsplit('.', 1)[0]}.txt"
    g = []
    for l in open(p):
        c, xc, yc, w, h = l.split(); c = int(c); xc, yc, w, h = map(float, (xc, yc, w, h))
        g.append((c, xc - w / 2, yc - h / 2, w, h))
    return g

def person_candidates(q):
    """q rows: cx,cy,w,h, 7 scores. Returns dict of arrays after class-agnostic NMS over person-type queries."""
    sc = q[:, 4:]
    top = sc.argmax(1)
    pers = sc[:, PERSON].max(1)
    ok = np.isin(top, PERSON) & (pers >= 0.05)
    q, sc, pers = q[ok], sc[ok], pers[ok]
    box = np.stack([q[:, 0] - q[:, 2] / 2, q[:, 1] - q[:, 3] / 2, q[:, 2], q[:, 3]], 1)
    order = np.argsort(-pers); keep = []
    for i in order:
        if all(iou(box[i], box[[j]])[0] < 0.7 for j in keep): keep.append(i)
    keep = np.array(keep, int)
    return dict(box=box[keep], sc=sc[keep], pers=pers[keep])

cand = {}; gt = {}; imgs_feat = {}
for b, f in frames.items():
    cand[b] = person_candidates(f["q"]); gt[b] = load_gt(b)
    img = cv2.imread(f"{FD}/{f['split']}/images/{b}")
    cand[b]["x"] = np.array([torso_feat(img, bx) if torso_feat(img, bx) is not None else np.full(4 if FEATVAR == "hsv4" else 3, np.nan) for bx in cand[b]["box"]])
    cand[b]["gtfeat"] = [(g[0], torso_feat(img, g[1:])) for g in gt[b] if g[0] in (4, 6, 1, 5, 3)]
    # GT match for every candidate: best-IoU GT person-type box >=0.5
    gb = np.array([g[1:] for g in gt[b] if g[0] in PERSON]).reshape(-1, 4); gc = np.array([g[0] for g in gt[b] if g[0] in PERSON])
    m = np.full(len(cand[b]["box"]), -1); mi = np.zeros(len(cand[b]["box"]))
    for i, bx in enumerate(cand[b]["box"]):
        if len(gb):
            v = iou(bx, gb); j = v.argmax()
            if v[j] >= 0.5: m[i] = gc[j]; mi[i] = v[j]
    cand[b]["gtcls"] = m
    cand[b]["gtbox_ref"] = np.array([g[1:] for g in gt[b] if g[0] == 6]).reshape(-1, 4)
    cand[b]["gtbox_pl"] = np.array([g[1:] for g in gt[b] if g[0] == 4]).reshape(-1, 4)
    cand[b]["gtbox_all"] = gb; cand[b]["gtcls_all"] = gc
nlate = [b for b in frames if frames[b]["phase"] == "late" and frames[b]["split"] == "test"]; nearly = [b for b in frames if frames[b]["phase"] == "early"]
print("early frames", len(nearly), "late (eval) frames", len(nlate))

# ---------------------------------------------------------------- palettes
def mvn_logpdf(x, mu, cov):
    d = len(mu); cov = cov + 1e-6 * np.eye(d)
    L = np.linalg.cholesky(cov); z = np.linalg.solve(L, (x - mu).T)
    return -0.5 * (z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)

class Palette:
    """player mixture (>=2 comps: teams (+GKs)) and a referee Gaussian."""
    def __init__(self, name, player_X, ref_X, kmax=4, ref_cov_from_players=True, seed=0):
        self.name = name
        X = player_X
        best = None
        for k in range(2, kmax + 1):
            if len(X) < 4 * k: break
            g = GaussianMixture(k, covariance_type="full", reg_covar=2e-3, n_init=3, random_state=seed).fit(X)
            bic = g.bic(X)
            if best is None or bic < best[0]: best = (bic, g)
        if best is None:
            g = GaussianMixture(2, covariance_type="tied", reg_covar=5e-3, random_state=seed).fit(X); best = (0, g)
        self.gm = best[1]
        self.k = self.gm.n_components
        self.w = self.gm.weights_
        # referee gaussian: mean from exemplars; covariance = shrinkage of exemplar cov toward average player-component cov
        mu = np.median(ref_X, 0)
        pc = np.mean([self.gm.covariances_[i] if self.gm.covariance_type == "full" else self.gm.covariances_ for i in range(self.k)], 0)
        if len(ref_X) >= 8:
            S = np.cov(ref_X.T); lam = 0.3
        else:
            S = pc; lam = 0.0
        self.ref_mu = mu; self.ref_cov = (1 - lam) * S + lam * pc if len(ref_X) >= 8 else pc * 1.5
        self.n_ref = len(ref_X); self.n_pl = len(X)

    def llr(self, x):
        lp = self.gm.score_samples(x)
        lr = mvn_logpdf(x, self.ref_mu, self.ref_cov)
        return np.clip(lr - lp, -8, 8)

    def team(self, x):
        return self.gm.predict(x)

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def sig(z): return 1 / (1 + np.exp(-z))

def all_cands(phase="early"):
    out = []
    for b in frames:
        if frames[b]["phase"] != phase: continue
        c = cand[b]
        for i in range(len(c["box"])):
            out.append((b, i))
    return out

# (i) unsupervised: confident detector players + confident detector referees from early frames
def palette_unsup():
    P, R = [], []
    for b in nearly:
        c = cand[b]
        for i in range(len(c["box"])):
            if np.isnan(c["x"][i]).any(): continue
            s = c["sc"][i]; top = s.argmax()
            if top == 4 and s[4] >= 0.8: P.append(c["x"][i])
            if top == 6 and s[6] >= 0.5: R.append(c["x"][i])
    return np.array(P), np.array(R)

def exemplars_gt(rng, n_ref=5, n_pl=10):
    refs, pls = [], []
    for b in nearly:
        for cl, f in cand[b]["gtfeat"]:
            if f is None or np.isnan(f).any(): continue
            if cl == 6: refs.append(f)
            if cl == 4: pls.append(f)
    refs, pls = np.array(refs), np.array(pls)
    ri = rng.choice(len(refs), min(n_ref, len(refs)), replace=False); pi = rng.choice(len(pls), min(n_pl, len(pls)), replace=False)
    return pls[pi], refs[ri], len(refs), len(pls)

# ---------------------------------------------------------------- evaluation helpers
def referee_eval(score_fn, beta, thr_list=(0.1, 0.2, 0.3, 0.5, 0.7)):
    """Returns dict with AP and PRF at thresholds. score_fn(b)-> referee score array per candidate."""
    recs = []   # (score, tp)
    ngt = 0
    for b in nlate:
        c = cand[b]; s = score_fn(b); ngt += len(c["gtbox_ref"])
        order = np.argsort(-s); used = set()
        for i in order:
            tp = 0
            if len(c["gtbox_ref"]):
                v = iou(c["box"][i], c["gtbox_ref"])
                for j in np.argsort(-v):
                    if v[j] < 0.5: break
                    if j not in used: used.add(j); tp = 1; break
            recs.append((s[i], tp, b))
    return recs, ngt

def ap_from(recs, ngt):
    r = sorted(recs, key=lambda z: -z[0]); tp = np.cumsum([z[1] for z in r]); fp = np.cumsum([1 - z[1] for z in r])
    rec = tp / max(ngt, 1); prec = tp / np.maximum(tp + fp, 1)
    # all-point interpolated AP
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())

def prf(recs, ngt, t):
    sel = [z for z in recs if z[0] >= t]; tp = sum(z[1] for z in sel); n = len(sel)
    p = tp / n if n else float("nan"); r = tp / max(ngt, 1)
    f = 2 * p * r / (p + r) if (n and tp) else 0.0
    return p, r, f, n, tp

def role_confusions(rescore_fn=None):
    """Among late candidates that match a GT player/referee (IoU>=.5) and whose top person role is player or referee:
       called = referee if s_ref > s_player else player (after optional rescoring). Returns counts."""
    cm = collections.Counter()
    rows = []
    for b in nlate:
        c = cand[b]
        sref, spl = (c["sc"][:, 6], c["sc"][:, 4]) if rescore_fn is None else rescore_fn(b)
        for i in range(len(c["box"])):
            g = c["gtcls"][i]
            if g not in (4, 6): continue
            if max(c["sc"][i][[4, 6]]) < 0.1: continue          # neither role has any signal: not a role decision
            called = 6 if sref[i] > spl[i] else 4
            cm[(int(g), called)] += 1; rows.append((b, i, int(g), called))
    return cm, rows

# detection-level facts
nref_gt = sum(len(cand[b]["gtbox_ref"]) for b in nlate); npl_gt = sum(len(cand[b]["gtbox_pl"]) for b in nlate)
print("late GT referees", nref_gt, "players", npl_gt)
# referee GT boxes that have NO person candidate at all (IoU>=.5)
miss = 0
for b in nlate:
    c = cand[b]
    for g in c["gtbox_ref"]:
        if not len(c["box"]) or iou(g, c["box"]).max() < 0.5: miss += 1
print("late GT referees with no candidate box >=0.05 (cannot be rescued by rescoring):", miss)

def run_palette(pal, label, rows_out, conf_rows):
    res = {}
    llr_cache = {}
    for b in nlate:
        x = cand[b]["x"]; ll = np.zeros(len(x)); ok = ~np.isnan(x).any(1)
        if ok.any(): ll[ok] = pal.llr(x[ok])
        llr_cache[b] = ll
    def base(b): return cand[b]["sc"][:, 6]
    recs, ngt = referee_eval(base, 0)
    for beta in [0] + BETAS:
        if beta == 0: fn = base
        else: fn = (lambda bb, beta=beta: sig(logit(cand[bb]["sc"][:, 6]) + beta * llr_cache[bb]))
        recs, ngt = referee_eval(fn, beta)
        row = dict(palette=label, beta=beta, AP=ap_from(recs, ngt), n_gt_ref=ngt, n_cand=len(recs))
        for t in (0.1, 0.2, 0.3, 0.5, 0.7):
            p, r, f, n, tp = prf(recs, ngt, t)
            row.update({f"P@{t}": p, f"R@{t}": r, f"F1@{t}": f, f"n@{t}": n})
        # best-F1 over a threshold sweep (oracle, optimistic)
        ts = np.linspace(0.05, 0.95, 91); fs = [prf(recs, ngt, t)[2] for t in ts]; row["bestF1"] = max(fs); row["bestF1_thr"] = ts[int(np.argmax(fs))]
        # player AP-like effect: role confusions at argmax
        if beta == 0: cm, rws = role_confusions(None)
        else:
            def rs(bb, beta=beta):
                z = beta * llr_cache[bb]
                return sig(logit(cand[bb]["sc"][:, 6]) + z), sig(logit(cand[bb]["sc"][:, 4]) - z)
            cm, rws = role_confusions(rs)
        row.update(ref_called_player=cm[(6, 4)], ref_called_ref=cm[(6, 6)], player_called_ref=cm[(4, 6)], player_called_player=cm[(4, 4)])
        row["_rws"] = rws
        rows_out.append(row)
    return llr_cache

rows = []
# ---- (i) unsupervised
P, R = palette_unsup()
print("unsup palette: confident player torsos", len(P), "confident referee torsos", len(R))
palU = Palette("U", P, R, seed=SEED)
print("  player mixture components:", palU.k, "weights", np.round(palU.w, 3))
llrU = run_palette(palU, "U: unsupervised (detector-confident early boxes)", rows, None)

# ---- (ii) human exemplars, repeated over random draws
rng = np.random.default_rng(SEED)
H_rows = []
for rep in range(NDRAWS):
    pl, rf_, nrefavail, nplavail = exemplars_gt(rng)
    palH = Palette("H", pl, rf_, kmax=2, seed=rep)
    tmp = []
    run_palette(palH, f"H: 5 GT referees + 10 GT players (draw {rep})", tmp, None)
    H_rows += tmp
print("GT exemplar pool in early frames: referees", nrefavail, "players", nplavail)
# ---- (iii) unsup players + human 5 referees
UH_rows = []
for rep in range(NDRAWS):
    pl, rf_, _, _ = exemplars_gt(rng)
    palUH = Palette("UH", P, rf_, seed=SEED)
    tmp = []
    run_palette(palUH, f"UH: unsup players + 5 GT referees (draw {rep})", tmp, None)
    UH_rows += tmp

def summarize(rs, name):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_rws"} for r in rs]); df["group"] = name; return df
dfU = summarize(rows, "U"); dfH = summarize(H_rows, "H"); dfUH = summarize(UH_rows, "UH")
allrows = pd.concat([dfU, dfH, dfUH])
allrows.to_csv(f"{HERE}/a_results_raw{OUTTAG}.csv", index=False)
num = [c for c in allrows.columns if c not in ("palette", "group")]
agg = pd.concat([dfU.assign(draw=0).groupby(["group", "beta"])[num].mean(),
                 dfH.groupby(["group", "beta"])[num].mean(), dfUH.groupby(["group", "beta"])[num].mean()])
sd = pd.concat([dfH.groupby(["group", "beta"])[["AP", "bestF1"]].std(), dfUH.groupby(["group", "beta"])[["AP", "bestF1"]].std()]).rename(columns=lambda c: c + "_sd")
agg = agg.join(sd)
agg.to_csv(f"{HERE}/a_results_summary{OUTTAG}.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(agg[["AP", "AP_sd", "P@0.3", "R@0.3", "F1@0.3", "P@0.5", "R@0.5", "F1@0.5", "bestF1", "bestF1_thr", "ref_called_player", "player_called_ref", "ref_called_ref", "player_called_player"]].round(3).to_string())

# ---- fixed/introduced confusions for beta=1, palette U and (mean of H draws)
def transitions(rws_base, rws_new):
    base = {(b, i): (g, c) for b, i, g, c in rws_base}; new = {(b, i): (g, c) for b, i, g, c in rws_new}
    t = collections.Counter()
    for k, (g, c0) in base.items():
        c1 = new[k][1]
        t[(g, c0, c1)] += 1
    return t
tr = collections.Counter()
for r in rows:
    if r["beta"] == 0: base = r["_rws"]
for r in rows:
    if r["beta"] == 1.0:
        t = transitions(base, r["_rws"])
        print("U beta=1 transitions (GT, baseline-called, rescored-called):", dict(t))
        json.dump({str(k): v for k, v in t.items()}, open(f"{HERE}/a_transitions_U{OUTTAG}.json", "w"))
# palette team accuracy -> no team GT. dump palette info
json.dump(dict(unsup_n_players=len(P), unsup_n_ref=len(R), k=int(palU.k), weights=palU.w.tolist(), ref_mu=palU.ref_mu.tolist(),
               comp_means=palU.gm.means_.tolist(), n_early_frames=len(nearly), n_late_frames=len(nlate), late_gt_ref=nref_gt,
               late_gt_players=npl_gt, late_ref_no_candidate=miss), open(f"{HERE}/a_palette_info{OUTTAG}.json", "w"), indent=1)

# baseline per-class sanity on late frames: referee score stats among GT refs
sref = np.concatenate([cand[b]["sc"][cand[b]["gtcls"] == 6][:, 6] for b in nlate])
print("late: detector referee-class score among matched GT referees: n", len(sref), "median", np.median(sref).round(3), "frac>=0.5", (sref >= 0.5).mean().round(3))

# bootstrap by frame: delta AP (beta=1, palette U)
fn_b = lambda b: cand[b]["sc"][:, 6]
fn_u = lambda b: sig(logit(cand[b]["sc"][:, 6]) + 1.0 * llrU[b])
rb, ng = referee_eval(fn_b, 0); ru, _ = referee_eval(fn_u, 1)
bb = np.array([z[2] for z in rb])
brng = np.random.default_rng(1); dl = []
fr = np.array(nlate)
for _ in range(300):
    pick = brng.choice(fr, len(fr)); cnt = collections.Counter(pick)
    def sub(rr): return [(s, t) for s, t, b in rr for _ in range(cnt.get(b, 0))]
    ngb = sum(cnt[b] * len(cand[b]["gtbox_ref"]) for b in cnt)
    dl.append(ap_from(sub(ru), ngb) - ap_from(sub(rb), ngb))
print("U beta=1 dAP bootstrap (by frame): mean %.3f  95%% CI [%.3f, %.3f]" % (np.mean(dl), *np.percentile(dl, [2.5, 97.5])))
json.dump(dict(dAP_mean=float(np.mean(dl)), lo=float(np.percentile(dl, 2.5)), hi=float(np.percentile(dl, 97.5))), open(f"{HERE}/a_bootstrap_U{OUTTAG}.json", "w"))

# stash assignments for plots
import pickle
pickle.dump(dict(llrU=llrU, late=nlate), open(f"{D}/a_llrU{OUTTAG}.pkl", "wb"))

# ============================================================ diagnostics
from sklearn.metrics import roc_auc_score
diag = {}
L, Y, REFSC, PLSC, TOP = [], [], [], [], []
for b in nlate:
    c = cand[b]
    for i in range(len(c["box"])):
        if c["gtcls"][i] in (4, 6) and not np.isnan(c["x"][i]).any():
            L.append(llrU[b][i]); Y.append(int(c["gtcls"][i] == 6))
L, Y = np.array(L), np.array(Y)
diag["llr_U_auroc_ref_vs_player"] = float(roc_auc_score(Y, L)); diag["n_ref"] = int(Y.sum()); diag["n_player"] = int((1 - Y).sum())
diag["llr_U_median_ref"] = float(np.median(L[Y == 1])); diag["llr_U_median_player"] = float(np.median(L[Y == 0]))
diag["llr_U_frac_ref_llr_gt0"] = float((L[Y == 1] > 0).mean()); diag["llr_U_frac_player_llr_gt0"] = float((L[Y == 0] > 0).mean())
# detector referee-class score reliability (reconciles M8): all referee-class detections on late frames
rs = []
for b in nlate:
    c = cand[b]
    for i in range(len(c["box"])):
        if c["sc"][i, 6] >= 0.1: rs.append((c["sc"][i, 6], c["gtcls"][i]))
rs = np.array(rs)
diag["refclass_dets_n"] = len(rs); diag["refclass_median_score_all"] = float(np.median(rs[:, 0])); diag["refclass_frac_ge_0.5_all"] = float((rs[:, 0] >= .5).mean())
for lo, hi in [(0.1, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]:
    m = (rs[:, 0] >= lo) & (rs[:, 0] < hi)
    diag[f"refclass_bin_{lo}_{hi}"] = dict(n=int(m.sum()), frac_matching_GT_referee=float((rs[m, 1] == 6).mean()) if m.any() else None,
                                           frac_matching_GT_player=float((rs[m, 1] == 4).mean()) if m.any() else None,
                                           frac_no_GT_match=float((rs[m, 1] == -1).mean()) if m.any() else None)
# per-GT-class median colour: where do rescored losses come from
lost = []
for r in rows:
    if r["beta"] == 1.0:
        for b_, i_, g_, c_ in r["_rws"]: pass
base_d = {(b_, i_): c_ for b_, i_, g_, c_ in [x for r in rows if r["beta"] == 0 for x in r["_rws"]]}
for r in rows:
    if r["beta"] == 1.0:
        for b_, i_, g_, c_ in r["_rws"]:
            if base_d[(b_, i_)] != c_: lost.append(dict(image=b_, cand=i_, gt=NAMES[g_], base=NAMES[base_d[(b_, i_)]], rescored=NAMES[c_], llr=float(llrU[b_][i_]),
                                                         s_ref=float(cand[b_]["sc"][i_, 6]), s_player=float(cand[b_]["sc"][i_, 4]), feat=np.round(cand[b_]["x"][i_], 3).tolist()))
pd.DataFrame(lost).to_csv(f"{HERE}/a_flipped_roles_U_beta1{OUTTAG}.csv", index=False)
# team-assignment audit: confident late players, component assignment + ambiguity
T, Pm, crops = [], [], []
rng2 = np.random.default_rng(5)
for b in nlate:
    c = cand[b]
    for i in range(len(c["box"])):
        if c["gtcls"][i] == 4 and c["sc"][i, 4] >= .8 and not np.isnan(c["x"][i]).any():
            T.append((b, i))
X = np.array([cand[b]["x"][i] for b, i in T]); post = palU.gm.predict_proba(X); comp = post.argmax(1)
diag["team_audit_n_confident_late_players"] = len(T); diag["team_comp_share"] = np.round(np.bincount(comp, minlength=palU.k) / len(comp), 3).tolist()
diag["team_frac_posterior_lt_0.9"] = float((post.max(1) < .9).mean())
diag["comp_means_(cos,sin,s,v)"] = np.round(palU.gm.means_, 2).tolist(); diag["ref_mu"] = np.round(palU.ref_mu, 2).tolist()
json.dump(diag, open(f"{HERE}/a_diagnostics{OUTTAG}.json", "w"), indent=1)
print(json.dumps(diag, indent=1))
# montage: 8 random confident players per component + 12 GT referees, torso crops
def tile(b, i, lab):
    f = frames[b]; img = cv2.imread(f"{FD}/{f['split']}/images/{b}"); x, y, w, h = cand[b]["box"][i]
    H_, W_ = img.shape[:2]; c = img[int(y * H_):int((y + h) * H_), int(x * W_):int((x + w) * W_)]
    c = cv2.resize(c, (60, 120)); cv2.putText(c, lab, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1); return c
rows_img = []
for k in range(palU.k):
    idx = np.where(comp == k)[0]; pick = rng2.choice(idx, min(10, len(idx)), replace=False) if len(idx) else []
    t = [tile(*T[j], f"C{k}") for j in pick]; t += [np.zeros((120, 60, 3), np.uint8)] * (10 - len(t)); rows_img.append(np.hstack(t))
refs = [(b, i) for b in nlate for i in range(len(cand[b]["box"])) if cand[b]["gtcls"][i] == 6]
pick = rng2.choice(len(refs), 10, replace=False); rows_img.append(np.hstack([tile(*refs[j], "REF") for j in pick]))
cv2.imwrite(f"{HERE}/a_palette_audit{OUTTAG}.png", np.vstack(rows_img))

# ============================================================ stacked (learned weight) variant
# Fit logistic regression on the EARLY frames (palette period, GT available) over candidates matched to GT player/referee:
#   y = GT referee ; features [logit s_ref, logit s_player, llr(palette U)] vs control without llr. Apply to late frames.
from sklearn.linear_model import LogisticRegression
def stack_feats(b, llr):
    c = cand[b]; return np.column_stack([logit(np.maximum(c["sc"][:, 6], .05)), logit(np.maximum(c["sc"][:, 4], .05)), llr / 4.0])
def early_llr(pal):
    out = {}
    for b in nearly:
        x = cand[b]["x"]; ll = np.zeros(len(x)); ok = ~np.isnan(x).any(1)
        if ok.any(): ll[ok] = pal.llr(x[ok])
        out[b] = ll
    return out
ell = early_llr(palU)
Xtr, ytr = [], []
for b in nearly:
    m = np.isin(cand[b]["gtcls"], (4, 6))
    if m.any(): Xtr.append(stack_feats(b, ell[b])[m]); ytr.append((cand[b]["gtcls"][m] == 6).astype(int))
Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
print("stacker training rows (early, matched to GT player/referee):", len(ytr), "referees", int(ytr.sum()))
stack_res = []
for name, cols in [("recalibrated detector only (no colour)", [0, 1]), ("detector + colour llr (palette U)", [0, 1, 2])]:
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr[:, cols], ytr)
    fn = lambda b, lr=lr, cols=cols: lr.predict_proba(stack_feats(b, llrU[b])[:, cols])[:, 1]
    recs, ngt = referee_eval(fn, 0)
    row = dict(variant=name, coef=np.round(lr.coef_[0], 2).tolist(), AP=ap_from(recs, ngt))
    for t in (0.3, 0.5, 0.7):
        p, r, f, n, tp = prf(recs, ngt, t); row.update({f"P@{t}": p, f"R@{t}": r, f"F1@{t}": f})
    cm, rws = role_confusions(lambda b, fn=fn: (fn(b), 1 - fn(b)))
    row.update(ref_called_player=cm[(6, 4)], player_called_ref=cm[(4, 6)])
    # bootstrap by frame, delta AP vs baseline raw detector
    rb2, _ = referee_eval(lambda b: cand[b]["sc"][:, 6], 0)
    brng = np.random.default_rng(2); dl2 = []
    for _ in range(300):
        pick = brng.choice(np.array(nlate), len(nlate)); cnt = collections.Counter(pick)
        sub = lambda rr: [(s, t, b) for s, t, b in rr for _ in range(cnt.get(b, 0))]
        ngb = sum(cnt[b] * len(cand[b]["gtbox_ref"]) for b in cnt)
        dl2.append(ap_from(sub(recs), ngb) - ap_from(sub(rb2), ngb))
    row.update(dAP_mean=float(np.mean(dl2)), dAP_lo=float(np.percentile(dl2, 2.5)), dAP_hi=float(np.percentile(dl2, 97.5)))
    stack_res.append(row)
pd.DataFrame(stack_res).to_csv(f"{HERE}/a_stacked{OUTTAG}.csv", index=False)
print(pd.DataFrame(stack_res).round(3).to_string())
