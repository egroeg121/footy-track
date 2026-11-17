"""Build the labelled ball-candidate table (detector conf + 72-d descriptor + label) for sub-experiment B.

Labels (candidate -> true/false), two independent sources, both on arsenal_mancity segment clips that
have stored candidates (ball_candidates/) and descriptors (ball_embeddings/):
  gt   : ball_gt_marks_MERGED human/labeller in_play_ball marks. Frame is "ball-complete" if it has an
         in_play_ball GT box or a no_ball row. Candidate within POS_PX of a GT ball = true; farther than
         NEG_PX from all GT balls = false; in between dropped. (vittrack/'yolo' machine-propagated rows excluded
         except human_checked ones.)
  check: Ball Check verdicts (ball_gt_marks/ball_checks): ball -> true, not_ball -> false (matched on bbox).
Also records whether the frame is in the detector's roboflow v11 TRAIN split (in-sample for the detector).
"""
import json, glob, os, re, collections
import numpy as np, pandas as pd

FD = "/mnt/storage/footy_data"
W, H = 1920.0, 1080.0
POS_PX, NEG_PX = 15.0, 40.0
OUT = os.path.join(os.path.dirname(__file__), "data")

def canon(name):
    m = re.fullmatch(r"(?:arsenal_)?mancity_(?:20250925_)?seg(\d+)", name)
    return f"arsenal_mancity_20250925_seg{int(m.group(1)):03d}" if m else None

# roboflow train membership (clip canon, frame)
rf = {}
for s in ("train", "valid", "test"):
    for f in os.listdir(f"{FD}/detection_dataset/roboflow_dataset_11/{s}/images"):
        m = re.match(r"(.+?)_(\d{3,6})_(png|jpg)\.rf", f)
        if m and "seg" in m.group(1):
            c = canon(m.group(1))
            if c: rf[(c, int(m.group(2)))] = s

gt = collections.defaultdict(lambda: {"balls": collections.defaultdict(list), "nob": set()})
for f in glob.glob(f"{FD}/ball_gt_marks_MERGED/*.jsonl"):
    c = canon(os.path.basename(f)[:-6])
    if not c: continue
    for l in open(f):
        d = json.loads(l); t = d["tags"]
        if "no_ball" in t: gt[c]["nob"].add(d["frame_index"])
        if "in_play_ball" in t and ("yolo" not in t) and (("vittrack" not in t) or ("human_checked" in t)):
            b = d["bbox"]; gt[c]["balls"][d["frame_index"]].append(((b["x"] + b["w"] / 2) * W, (b["y"] + b["h"] / 2) * H))

checks = collections.defaultdict(list)
for f in glob.glob(f"{FD}/ball_gt_marks/ball_checks/*.jsonl"):
    c = canon(os.path.basename(f)[:-6]) or ("arsenal_mancity_20250925_" + os.path.basename(f)[:-6].split("_")[-1] if False else None)
    if not c: continue
    for l in open(f):
        d = json.loads(l)
        if d.get("verdict") in ("ball", "not_ball"): checks[c].append(d)

clips = sorted(set(gt) | set(checks))
rows = []
for c in clips:
    cp = f"{FD}/ball_candidates/{c}.jsonl"; ep = f"{FD}/ball_embeddings/{c}.npz"
    if not (os.path.exists(cp) and os.path.exists(ep)): continue
    z = np.load(ep); emb = {(int(f), int(i)): v for f, i, v in zip(z["frame_index"], z["cand_index"], z["vec"])}
    perf = collections.Counter(); cands = []
    for l in open(cp):
        d = json.loads(l)
        if not any(t in ("ball", "in_play_ball", "out_of_play_ball") for t in d["tags"]) or not d.get("bbox"): continue
        fi = d["frame_index"]; i = perf[fi]; perf[fi] += 1
        b = d["bbox"]; cx, cy = (b["x"] + b["w"] / 2) * W, (b["y"] + b["h"] / 2) * H
        cands.append((fi, i, cx, cy, b["w"] * W, b["h"] * H, d["confidence"]))
    g = gt.get(c)
    for fi, i, cx, cy, bw, bh, conf in cands:
        v = emb.get((fi, i))
        if v is None: continue
        lab_gt = np.nan
        if g and (fi in g["balls"] or fi in g["nob"]):
            ds = [np.hypot(cx - x, cy - y) for x, y in g["balls"].get(fi, [])]
            if not ds: lab_gt = 0
            elif min(ds) <= POS_PX: lab_gt = 1
            elif min(ds) >= NEG_PX: lab_gt = 0
        lab_ck = np.nan
        for d in checks.get(c, []):
            if d["frame_index"] != fi: continue
            b = d["bbox"]
            if abs((b["x"] + b["w"] / 2) * W - cx) < 3 and abs((b["y"] + b["h"] / 2) * H - cy) < 3:
                lab_ck = 1 if d["verdict"] == "ball" else 0
        if np.isnan(lab_gt) and np.isnan(lab_ck): continue
        rows.append(dict(clip=c, seg=int(c[-3:]), frame=fi, cand=i, conf=conf, cx=cx, cy=cy, bw=bw, bh=bh,
                         lab_gt=lab_gt, lab_ck=lab_ck, rf=rf.get((c, fi), ""),
                         ncand=perf[fi], **{f"v{k}": float(x) for k, x in enumerate(v)}))
df = pd.DataFrame(rows)
df.to_parquet(f"{OUT}/ball_cands_labelled.parquet") if False else df.to_pickle(f"{OUT}/ball_cands_labelled.pkl")
print(len(df), "rows;", df["clip"].nunique(), "clips")
print("gt labelled:", df["lab_gt"].notna().sum(), "pos", int((df.lab_gt == 1).sum()), "neg", int((df.lab_gt == 0).sum()))
print("check labelled:", df["lab_ck"].notna().sum(), "pos", int((df.lab_ck == 1).sum()), "neg", int((df.lab_ck == 0).sum()))
print("rf split of rows:", df["rf"].replace("", "none").value_counts().to_dict())
print(df.groupby("clip").agg(n=("conf", "size"), pos=("lab_gt", lambda s: (s == 1).sum()), neg=("lab_gt", lambda s: (s == 0).sum()), ck=("lab_ck", lambda s: s.notna().sum())).to_string())
