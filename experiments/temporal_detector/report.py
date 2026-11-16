"""Aggregate per-image detections of all runs into the ablation report (CPU, numpy + cv2 only).

    uv run --no-project --python /usr/bin/python3 --with numpy --with opencv-python-headless \
        python report.py --runs /mnt/storage/exp/temporal/runs --dataset /mnt/storage/footy_data/multiframe_v1/k2 \
        --out docs/experiments/temporal_detector_ablation

Expects  <runs>/preds_ref.json  (production RT-DETR-L v11, frame t, 1280)  and  preds_A_s<seed>.json / preds_B_s<seed>.json,
optionally <runs>/curve_<A|B>_s<seed>.jsonl and <runs>/time_<A|B>_s<seed>.json.
Writes <out>.md, <out>.json and <out>/example_*.png.

Definitions (all in original 1920x1080 pixels)
  hit@10   the highest-confidence ball detection of the image is within 10 px of a GT ball centre
           (denominator: images that have a GT ball; images with several balls use the nearest)
  any@10   some detection (conf >= 0.001) lies within 10 px of a GT ball (candidate recall, ignores ranking)
  AP50     COCO/Ultralytics-style 101-point AP at IoU 0.5 over the subset's images, ball class only
  top-1 precision at conf t: among ALL images of the subset (also ball-free ones) whose top-1 conf >= t, the fraction that is a hit
  FP rate  fraction of ball-free images whose top-1 conf >= t
Paired CIs: percentile bootstrap (10,000 resamples) of mean(hit_B - hit_A) over the same images; "clip" = resampling
whole clips (images from one clip are correlated), "img" = resampling images. With several seeds the per-image hit is
averaged over the seeds of each arm first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HIT_PX = 10.0
THRS = (0.1, 0.25, 0.5)
N_BOOT = 10_000


# ------------------------------------------------------------------------------------------------ metrics
def iou_cxcywh(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def ap50(images: list[dict]) -> float:
    n_gt = sum(len(im["gt"]) for im in images)
    if n_gt == 0:
        return float("nan")
    rec = []  # (conf, tp)
    for im in images:
        used = set()
        for d in sorted(im["dets"], key=lambda d: -d[4]):
            best, bj = 0.0, -1
            for j, g in enumerate(im["gt"]):
                if j in used:
                    continue
                v = iou_cxcywh(d, g)
                if v > best:
                    best, bj = v, j
            if best >= 0.5:
                used.add(bj)
            rec.append((d[4], 1.0 if best >= 0.5 else 0.0))
    if not rec:
        return 0.0
    rec.sort(key=lambda r: -r[0])
    tp = np.array([r[1] for r in rec])
    tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = tpc / n_gt, tpc / (tpc + fpc)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], precision, [0.0])))))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def per_image(im: dict) -> dict:
    """hit/any/top1 info for one image of one model."""
    dets = im["dets"]
    top = max(dets, key=lambda d: d[4]) if dets else None
    out = {"top_conf": top[4] if top else 0.0, "top_xy": (top[0], top[1]) if top else None}
    if im["gt"]:
        gts = np.array([[g[0], g[1]] for g in im["gt"]])
        if top:
            out["dist"] = float(np.min(np.hypot(gts[:, 0] - top[0], gts[:, 1] - top[1])))
        else:
            out["dist"] = float("inf")
        out["hit"] = float(out["dist"] <= HIT_PX)
        out["any"] = float(any(np.min(np.hypot(gts[:, 0] - d[0], gts[:, 1] - d[1])) <= HIT_PX for d in dets))
    return out


# ------------------------------------------------------------------------------------------------ data
def load_runs(runs: Path, dataset: Path) -> dict[str, dict]:
    """Per-model {id|split: image record}. Early runs were evaluated before evaluate.py mapped split 'val' to meta's
    'valid', so neighbour status / clip id / strict flag are (re)filled from meta.jsonl here for every run."""
    meta = {}
    for line in open(dataset / "meta.jsonl"):
        r = json.loads(line)
        meta[("val" if r["split"] == "valid" else r["split"], r["id"])] = r
    out = {}
    for p in sorted(runs.glob("preds_*.json")):
        d = json.loads(p.read_text())
        for im in d["images"]:
            m = meta[(im["split"], im["id"])]
            im["prev_status"], im["next_status"] = m["prev"]["status"], m["next"]["status"]
            im["clip_id"], im["strict_heldout"] = m["clip_id"], bool(m.get("strict_heldout", False))
        out[p.stem.removeprefix("preds_")] = {im["id"] + "|" + im["split"]: im for im in d["images"]} | {"__meta__": d}
    return out


def subsets(base: list[dict], motion_edges: tuple[float, float]) -> dict[str, list[str]]:
    """name -> list of image keys (id|split). `base` = the images of one model (all models share ids)."""
    def key(im):
        return im["id"] + "|" + im["split"]

    ok = lambda im: im["prev_status"] == "ok" and im["next_status"] == "ok"  # noqa: E731
    S: dict[str, list[str]] = {}
    for sp in ("val", "test", "heldout"):
        S[sp] = [key(i) for i in base if i["split"] == sp]
    S["val / clip arsenal_example (74% of val balls)"] = [key(i) for i in base if i["split"] == "val" and i["clip_id"] == "arsenal_example"]
    S["val / all other clips"] = [key(i) for i in base if i["split"] == "val" and i["clip_id"] != "arsenal_example"]
    S["heldout_strict"] = [key(i) for i in base if i["split"] == "heldout" and i["strict_heldout"]]
    S["test+heldout"] = S["test"] + S["heldout"]
    S["val+test+heldout"] = S["val"] + S["test"] + S["heldout"]
    for name, pred in (("both_ok", ok), ("fallback", lambda i: not ok(i))):
        S[f"test+heldout / neighbours {name}"] = [key(i) for i in base if i["split"] in ("test", "heldout") and pred(i)]
    lo, hi = motion_edges
    for name, pred in (("low", lambda m: m < lo), ("mid", lambda m: lo <= m < hi), ("high", lambda m: m >= hi)):
        S[f"test+heldout / ball motion {name}"] = [
            key(i) for i in base if i["split"] in ("test", "heldout") and i["gt"] and ok(i)
            and pred(max(i["motion_prev"], i["motion_next"]))]
    return S


def bootstrap_diff(hit_a: np.ndarray, hit_b: np.ndarray, clips: np.ndarray | None, rng) -> tuple[float, float, float]:
    d = hit_b - hit_a
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if clips is None:
        idx = rng.integers(0, n, size=(N_BOOT, n))
        m = d[idx].mean(1)
    else:
        uniq, inv = np.unique(clips, return_inverse=True)
        sums = np.bincount(inv, weights=d, minlength=len(uniq))
        cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
        pick = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
        m = sums[pick].sum(1) / cnt[pick].sum(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------------------------------------------ examples
def crop_example(root: Path, im: dict, preds: dict[str, tuple], title: str, dst: Path, r: int = 40, z: int = 5):
    sys_path = str(root / "images" / im["split"] / (im["id"] + ".tiff"))
    ok, pages = cv2.imdecodemulti(np.fromfile(sys_path, np.uint8), cv2.IMREAD_UNCHANGED)
    img5 = np.stack(pages, axis=2)
    ih, iw = img5.shape[:2]  # a few clips are 1280x720: boxes in the json are in the 1920x1080 frame, convert back
    sx, sy = 1920 / iw, 1080 / ih
    gx, gy = im["gt"][0][0] / sx, im["gt"][0][1] / sy
    preds = {k: ((v[0] / sx, v[1] / sy, v[2]) if v[0] is not None else v) for k, v in preds.items()}
    x0, y0 = int(np.clip(gx - r, 0, iw - 2 * r)), int(np.clip(gy - r, 0, ih - 2 * r))
    sl = (slice(y0, y0 + 2 * r), slice(x0, x0 + 2 * r))
    bgr = np.ascontiguousarray(img5[sl][..., 1:4])
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    big = cv2.resize(bgr, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    cv2.circle(big, (int((gx - x0) * z), int((gy - y0) * z)), 11 * z // 2 + 6, (0, 255, 255), 1, cv2.LINE_AA)
    colours = {"A": (255, 128, 0), "B": (0, 200, 0), "ref": (200, 0, 200)}
    for k, (px, py, conf) in preds.items():
        if px is None:
            continue
        cv2.drawMarker(big, (int((px - x0) * z), int((py - y0) * z)), colours[k], cv2.MARKER_CROSS, 18, 2)
    panels = [big]
    for nb in (0, 4):
        d = np.clip(np.abs(g - img5[sl][..., nb].astype(np.int16)) * 2, 0, 255).astype(np.uint8)
        panels.append(cv2.cvtColor(cv2.resize(d, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR))
    row = np.hstack(panels)
    bar = np.full((44, row.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    leg = "  ".join(f"{k}: " + ("none" if v[0] is None else f"conf {v[2]:.2f} d={np.hypot((v[0] - gx) * sx, (v[1] - gy) * sy):.0f}px") for k, v in preds.items())
    cv2.putText(bar, leg + "   yellow=GT  left=RGB(t)  mid=|t-prev|x2  right=|next-t|x2", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(dst), np.vstack([bar, row]))


# ------------------------------------------------------------------------------------------------ main
def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="prefix, writes <out>.md/.json and <out>/*.png")
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--meta", default="", help="free text put at the top of the md (training config etc.)")
    a = ap.parse_args()

    runs = load_runs(a.runs, a.dataset)
    tags = [t for t in runs]
    seeds = sorted({t.split("_s")[1] for t in tags if t.startswith(("A_s", "B_s"))})
    arms = {arm: [f"{arm}_s{s}" for s in seeds if f"{arm}_s{s}" in runs] for arm in "AB"}
    assert arms["A"] and arms["B"], f"need A and B runs, found {tags}"
    first = arms["A"][0]
    base = [v for k, v in runs[first].items() if k != "__meta__"]
    # motion terciles over images with a GT ball and both neighbours ok
    ms = np.array([max(i["motion_prev"], i["motion_next"]) for i in base if i["split"] in ("test", "heldout")
                   and i["gt"] and i["prev_status"] == "ok" and i["next_status"] == "ok"])
    edges = (float(np.percentile(ms, 100 / 3)), float(np.percentile(ms, 200 / 3)))
    S = subsets(base, edges)
    rng = np.random.default_rng(0)

    models = {"RT-DETR-L v11 (production, frame t)": ["ref"] if "ref" in runs else []}
    for arm, nm in (("A", "A  YOLO 3ch (frame t)"), ("B", "B  YOLO 5ch (t, t-2, t+2)")):
        for t in arms[arm]:
            models[f"{nm} seed {t.split('_s')[1]}"] = [t]
        if len(arms[arm]) > 1:
            models[f"{nm} mean of seeds"] = arms[arm]
    models = {k: v for k, v in models.items() if v}

    pim = {t: {k: per_image(v) for k, v in runs[t].items() if k != "__meta__"} for t in runs}
    by_key = {k: v for k, v in runs[first].items() if k != "__meta__"}
    res: dict = {"model_tags": models, "seeds": seeds, "motion_edges_p90_absdiff": edges, "subsets": {}, "models": list(models)}

    def metric_block(tag_list: list[str], keys: list[str]) -> dict:
        gt_keys = [k for k in keys if by_key[k]["gt"]]
        free = [k for k in keys if not by_key[k]["gt"]]
        out: dict = {"n_img": len(keys), "n_gt_img": len(gt_keys), "n_balls": sum(len(by_key[k]["gt"]) for k in keys),
                     "n_ball_free": len(free)}
        if not gt_keys:
            return out
        out["hit10"] = float(np.mean([np.mean([pim[t][k]["hit"] for t in tag_list]) for k in gt_keys]))
        out["any10"] = float(np.mean([np.mean([pim[t][k]["any"] for t in tag_list]) for k in gt_keys]))
        out["ap50"] = float(np.mean([ap50([runs[t][k] for k in keys]) for t in tag_list]))
        for thr in THRS:
            prec, cov, fp = [], [], []
            for t in tag_list:
                pred = [k for k in keys if pim[t][k]["top_conf"] >= thr]
                ok_ = [k for k in pred if by_key[k]["gt"] and pim[t][k]["hit"] == 1.0]
                prec.append(len(ok_) / len(pred) if pred else float("nan"))
                cov.append(len(ok_) / len(gt_keys))
                fp.append(float(np.mean([pim[t][k]["top_conf"] >= thr for k in free])) if free else float("nan"))
            out[f"prec@{thr}"], out[f"hit@conf{thr}"], out[f"fp_free@{thr}"] = (float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan") for v in (prec, cov, fp))
        return out

    res["subsets"] = {}
    for sname, keys in S.items():
        res["subsets"][sname] = {mname: metric_block(tl, keys) for mname, tl in models.items()}

    # paired differences
    def hits(tag_list, keys):
        return np.array([np.mean([pim[t][k]["hit"] for t in tag_list]) for k in keys])

    paired = {}
    for sname, keys in S.items():
        gk = [k for k in keys if by_key[k]["gt"]]
        if not gk:
            continue
        ha, hb = hits(arms["A"], gk), hits(arms["B"], gk)
        clips = np.array([by_key[k]["clip_id"] or "none" for k in gk])
        row = {"n": len(gk), "n_clips": int(len(set(clips))), "A": float(ha.mean()), "B": float(hb.mean())}
        row["diff_img"] = bootstrap_diff(ha, hb, None, rng)
        row["diff_clip"] = bootstrap_diff(ha, hb, clips, rng)
        if len(arms["A"]) > 1:
            row["per_seed"] = {s: float(np.mean([pim[f"B_s{s}"][k]["hit"] - pim[f"A_s{s}"][k]["hit"] for k in gk])) for s in seeds
                               if f"A_s{s}" in runs and f"B_s{s}" in runs}
        # discordant pairs (seed-averaged hits rounded to majority)
        row["B_only"] = int(np.sum((hb > 0.5) & (ha <= 0.5)))
        row["A_only"] = int(np.sum((ha > 0.5) & (hb <= 0.5)))
        any_a = np.array([np.mean([pim[t][k]["any"] for t in arms["A"]]) for k in gk])
        any_b = np.array([np.mean([pim[t][k]["any"] for t in arms["B"]]) for k in gk])
        row["any10"] = {"A": float(any_a.mean()), "B": float(any_b.mean()), "diff_img": bootstrap_diff(any_a, any_b, None, rng),
                        "diff_clip": bootstrap_diff(any_a, any_b, clips, rng)}
        paired[sname] = row
    res["paired_B_minus_A_hit10"] = paired

    # curves
    curves = {}
    for p in sorted(a.runs.glob("curve_*.jsonl")):
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if "ball_ap50" in r]
        if not rows:
            continue
        # training wall time summed over sessions (elapsed_s restarts at 0 when a run is resumed after a crash)
        wall, seg_max, prev = 0.0, 0.0, 0.0
        for r in rows:
            if r["elapsed_s"] < prev:
                wall, seg_max = wall + seg_max, 0.0
            seg_max, prev = r["elapsed_s"], r["elapsed_s"]
        wall += seg_max
        # one row per epoch (the callback fires twice at the end of training; resumed runs repeat an epoch)
        first = {}
        for r in rows:
            first.setdefault(r["epoch"], r)  # the epoch's own validation (a second row at the end is best.pt's re-validation)
        rows = list(first.values())
        apv = np.array([r["ball_ap50"] for r in rows])
        tp = p.stem.removeprefix("curve_")
        t_file = a.runs / f"time_{tp}.json"
        curves[tp] = {"epochs": len(rows), "best_ball_ap50": float(apv.max()), "best_epoch": rows[int(apv.argmax())]["epoch"],
                      "last": float(apv[-1]), "mean_last10": float(apv[-10:].mean()),
                      "ap50_at": {str(r["epoch"]): r["ball_ap50"] for r in rows if r["epoch"] % 10 == 0 or r["epoch"] == len(rows)},
                      "val_loss_note": "train loss at last epoch " + str(rows[-1]["loss"]),
                      "train_seconds": wall, "wall": json.loads(t_file.read_text()) if t_file.exists() else None}
    res["curves"] = curves

    # examples: discordant images on test/heldout, seed-averaged
    ex_dir = a.out
    ex_dir.mkdir(parents=True, exist_ok=True)
    cand = [k for k in S["test+heldout"] if by_key[k]["gt"]]
    ha, hb = hits(arms["A"], cand), hits(arms["B"], cand)
    # seed-averaged top-1 hit differs by >= 0.5 (both seeds agree -> 1.0); strongest agreement first
    b_wins = [k for k, x, y in sorted(zip(cand, ha, hb, strict=True), key=lambda t: -(t[2] - t[1])) if y - x >= 0.5]
    a_wins = [k for k, x, y in sorted(zip(cand, ha, hb, strict=True), key=lambda t: -(t[1] - t[2])) if x - y >= 0.5]
    both_miss = [k for k, x, y in zip(cand, ha, hb, strict=True) if x == 0 and y == 0]
    rng2 = np.random.default_rng(1)
    pick = []
    for name, lst, n in (("B_hit_A_miss", b_wins, round(a.n_examples * 0.6)), ("A_hit_B_miss", a_wins, round(a.n_examples * 0.4))):
        sel = lst[:n]
        pick += [(name, k) for k in sel]
    ex_meta = []
    for j, (name, k) in enumerate(pick):
        im = by_key[k]
        preds = {}
        for lab, tl in (("A", arms["A"][0]), ("B", arms["B"][0]), ("ref", "ref")):
            if tl in runs:
                p_ = pim[tl][k]
                preds[lab] = (p_["top_xy"][0] if p_["top_xy"] else None, p_["top_xy"][1] if p_["top_xy"] else None, p_["top_conf"])
        dst = ex_dir / f"example_{j:02d}_{name}.png"
        crop_example(a.dataset, im, preds, f"{name}  {k.split('|')[0][:60]} ({im['split']}) motion={max(im['motion_prev'], im['motion_next']):.0f}", dst)
        ex_meta.append({"file": dst.name, "kind": name, "id": im["id"], "split": im["split"], "motion": max(im["motion_prev"], im["motion_next"])})
    res["examples"] = ex_meta
    res["discordant"] = {"B_hit_A_miss": len(b_wins), "A_hit_B_miss": len(a_wins), "both_miss": len(both_miss), "n": len(cand)}

    json.dump(res, open(str(a.out) + ".json", "w"), indent=1, default=lambda o: None if o != o else o)
    write_md(res, a, S, arms)
    print("wrote", str(a.out) + ".md")


def write_md(res, a, S, arms):
    L = []
    L.append("# Temporal detector ablation: single frame vs camera-aligned neighbours\n")
    if a.meta:
        L.append(a.meta + "\n")
    L.append("Counts: " + ", ".join(f"{k}: {len(v)} images" for k, v in S.items() if " / " not in k) + ".\n")
    cols = ["val", "test", "heldout", "heldout_strict"]
    for metric, title in (("hit10", "Top-1 ball recall @10 px (hit@10)"), ("any10", "Candidate recall @10 px (any detection within 10 px)"),
                          ("ap50", "Ball AP50")):
        L.append(f"## {title}\n")
        n_row = {c: res["subsets"][c][next(iter(res["models"]))] for c in cols}
        L.append("| model | " + " | ".join(f"{c} (n={n_row[c]['n_gt_img']})" for c in cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        for m in res["models"]:
            L.append(f"| {m} | " + " | ".join(fmt(res["subsets"][c][m].get(metric)) for c in cols) + " |")
        L.append("")
    L.append("## Paired difference B - A in hit@10 (bootstrap 95% CI)\n")
    L.append("| subset | n imgs | n clips | A | B | B-A | 95% CI (images) | 95% CI (clips) | B-only / A-only hits | per seed B-A |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s, r in res["paired_B_minus_A_hit10"].items():
        d, di, dc = r["diff_img"], r["diff_img"], r["diff_clip"]
        ps = ", ".join(f"s{k}: {v:+.3f}" for k, v in r.get("per_seed", {}).items())
        L.append(f"| {s} | {r['n']} | {r['n_clips']} | {fmt(r['A'])} | {fmt(r['B'])} | {d[0]:+.3f} | [{di[1]:+.3f}, {di[2]:+.3f}] | "
                 f"[{dc[1]:+.3f}, {dc[2]:+.3f}] | {r['B_only']} / {r['A_only']} | {ps} |")
    L.append("")
    L.append("## Paired difference B - A in candidate recall any@10 (does the detector at least propose the ball?)\n")
    L.append("| subset | n imgs | A | B | B-A | 95% CI (images) | 95% CI (clips) |")
    L.append("|---|---|---|---|---|---|---|")
    for s_, r in res["paired_B_minus_A_hit10"].items():
        q = r["any10"]
        L.append(f"| {s_} | {r['n']} | {fmt(q['A'])} | {fmt(q['B'])} | {q['diff_img'][0]:+.3f} | [{q['diff_img'][1]:+.3f}, {q['diff_img'][2]:+.3f}] | "
                 f"[{q['diff_clip'][1]:+.3f}, {q['diff_clip'][2]:+.3f}] |")
    L.append("")
    L.append("## Top-1 precision and false positives by confidence threshold (val+test+heldout)\n")
    sub = res["subsets"]["val+test+heldout"]
    L.append("| model | " + " | ".join(f"prec@{t} / recall@{t}" for t in THRS) + " | " + " | ".join(f"FP(ball-free)@{t}" for t in THRS) + " |")
    L.append("|---|" + "---|" * (2 * len(THRS)))
    for m in res["models"]:
        L.append(f"| {m} | " + " | ".join(f"{fmt(sub[m].get(f'prec@{t}'))} / {fmt(sub[m].get(f'hit@conf{t}'))}" for t in THRS) + " | "
                 + " | ".join(fmt(sub[m].get(f"fp_free@{t}")) for t in THRS) + " |")
    L.append(f"\nBall-free images in val+test+heldout: {sub[res['models'][0]]['n_ball_free']} (FP rate is noisy).\n")
    L.append("## By neighbour status and ball motion (hit@10, test+heldout)\n")
    subs = [s for s in res["subsets"] if "/" in s and not s.startswith("val /")]
    L.append("| model | " + " | ".join(f"{s} (n={res['subsets'][s][res['models'][0]]['n_gt_img']})" for s in subs) + " |")
    L.append("|---|" + "---|" * len(subs))
    for m in res["models"]:
        L.append(f"| {m} | " + " | ".join(fmt(res["subsets"][s][m].get("hit10")) for s in subs) + " |")
    e = res["motion_edges_p90_absdiff"]
    L.append(f"\nMotion score = max over the two neighbours of the p90 of |gray(t) - aligned neighbour| in a 15x15 window at the GT ball; "
             f"terciles over test+heldout balls with both neighbours ok: low < {e[0]:.1f} <= mid < {e[1]:.1f} <= high.\n")
    L.append("Paired B-A for these subsets:\n")
    L.append("| subset | n | B-A | 95% CI (images) | 95% CI (clips) |")
    L.append("|---|---|---|---|---|")
    for s in subs:
        r = res["paired_B_minus_A_hit10"].get(s)
        if r:
            L.append(f"| {s} | {r['n']} | {r['diff_img'][0]:+.3f} | [{r['diff_img'][1]:+.3f}, {r['diff_img'][2]:+.3f}] | "
                     f"[{r['diff_clip'][1]:+.3f}, {r['diff_clip'][2]:+.3f}] |")
    L.append("\n## Training curves (ball AP50 on val, each epoch)\n")
    L.append("Ultralytics' own validator at the last epoch ('last') vs this report's explicit-inference AP50 on the same val split "
             "(cross-check of the evaluation code path; small differences come from letterbox padding and the AP matching).\n")
    L.append("| run | epochs | best ball AP50 (epoch) | last (Ultralytics val) | last (report.py val AP50) | mean last 10 | AP50 at epoch ... | training time |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t, c in res["curves"].items():
        pts = ", ".join(f"{k}: {v:.3f}" for k, v in c["ap50_at"].items())
        mine = next((res["subsets"]["val"][m].get("ap50") for m, tl in res["model_tags"].items() if tl == [t]), None)
        L.append(f"| {t} | {c['epochs']} | {c['best_ball_ap50']:.3f} ({c['best_epoch']}) | {c['last']:.3f} | {fmt(mine)} | {c['mean_last10']:.3f} | {pts} | {c['train_seconds'] / 3600:.2f} h |")
    L.append("\n## Examples where A and B disagree (test + heldout)\n")
    d = res["discordant"]
    L.append(f"Images where the seed-averaged top-1 hit differs by >= 0.5: B better {d['B_hit_A_miss']}, A better {d['A_hit_B_miss']}; both miss {d['both_miss']} of {d['n']}. Shown: the clearest ones (all seeds agree first); the A / B markers are the seed-0 top-1 detections.\n")
    for ex in res["examples"]:
        L.append(f"![{ex['kind']}]({a.out.name}/{ex['file']})\n")
    Path(str(a.out) + ".md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
