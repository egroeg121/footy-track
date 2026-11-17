"""B extras: (1) learned ceiling: LR on conf + full 72-d descriptor (LOCO); (2) montage of failure cases."""
import os, json
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(2)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__)); FD = "/mnt/storage/footy_data"
df = pd.read_pickle(f"{HERE}/data/ball_cands_labelled.pkl"); g = df[df.lab_gt.notna()].copy(); g["y"] = g.lab_gt.astype(int)
VC = [f"v{i}" for i in range(72)]
lc = np.log(np.clip(g.conf, 1e-4, 1 - 1e-4) / (1 - np.clip(g.conf, 1e-4, 1 - 1e-4))).values
X = np.column_stack([lc, g[VC].values]); oof = np.zeros(len(g)); oof_h = np.zeros(len(g))
Xh = np.column_stack([lc, g[VC[64:]].values])
for c in g["clip"].unique():
    m = (g["clip"] == c).values
    for XX, o, C in ((X, oof, 0.05), (Xh, oof_h, 1.0)):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(XX[~m], g.y.values[~m]); o[m] = mdl.decision_function(XX[m])
for nm, s in (("conf only", g.conf.values), ("conf + full 72-d descriptor (LR, C=0.05)", oof), ("conf + hue-hist only (LR)", oof_h)):
    print(f"{nm:45s} LOCO AUROC {roc_auc_score(g.y, s):.3f} AP {average_precision_score(g.y, s):.3f}")
    un = (g.rf != "train").values
    print(f"{'':45s} unseen-frames AUROC {roc_auc_score(g.y[un], s[un]):.3f} AP {average_precision_score(g.y[un], s[un]):.3f}")

# failure montage: (a) false, conf>=0.5, smallest d_all (look like the ball but aren't)  (b) true, largest d_all (ball but unlike reference)
sc = pd.read_csv(f"{HERE}/b_scored_candidates.csv")
A = sc[(sc.y == 0) & (sc.conf >= 0.4)].nsmallest(12, "d_all"); B = sc[(sc.y == 1)].nlargest(12, "d_all")
def crop(r, half=40):
    cap = cv2.VideoCapture(f"{FD}/clips/{r['clip']}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame)); ok, fr = cap.read(); cap.release()
    if not ok: return np.zeros((2 * half * 3, 2 * half * 3, 3), np.uint8)
    x, y = int(r.cx), int(r.cy); pad = cv2.copyMakeBorder(fr, half, half, half, half, cv2.BORDER_CONSTANT)
    c = pad[y:y + 2 * half, x:x + 2 * half]; c = cv2.resize(c, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cv2.circle(c, (half * 3, half * 3), int(max(r.bw, r.bh) * 3 / 2 + 6), (0, 255, 255), 1)
    cv2.putText(c, f"c{r.conf:.2f} d{r.d_all:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
    return c
for nm, S in (("false_looks_like_ref", A), ("true_far_from_ref", B)):
    tiles = [crop(r) for _, r in S.iterrows()]
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(f"{HERE}/b_montage_{nm}.png", np.vstack(rows))
    print(nm, S[["clip", "frame", "conf", "d_all"]].round(2).to_string())
