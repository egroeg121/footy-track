"""Score WASB candidates (and RT-DETR machine labels) against human GT (CPU, stdlib only).

Per eval frame (ball frames and explicit no_ball frames), with candidates (x, y, score) in 1080p px:
  * hit@d       top-1 candidate (highest score, no threshold) within d px of a GT ball centre
  * oracle@d    ANY candidate within d px (upper bound of what a tracker could pick)
  * sweep       over score thresholds s: emitted = candidates with score >= s
                  recall   = ball frames whose top-1 emitted candidate is within d / ball frames
                  prec_f   = those hits / frames (ball or no_ball) where anything was emitted
                  prec_det = emitted candidates within d of GT / all emitted candidates (all frames)
                  nb_fp    = no_ball frames with any emission / no_ball frames
                  f1       = harmonic mean of recall and prec_f
  * best threshold = the swept s with max f1 at d=10.

RT-DETR candidates: machine_labels/<20250925 alias>.jsonl rows tagged in_play_ball/ball.
Usage: score.py --pred-dir runs/wasb-v1/pred --evalset evalset.json [--v11 v11_match.json] --out results.json
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

DS = (5, 10, 20)
THR = [round(0.1 + 0.05 * i, 2) for i in range(0, 18)]   # 0.10 .. 0.95
ML = "/mnt/storage/footy_data/machine_labels"
W, H = 1920, 1080


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def frame_hits(cands, gts, d):
    top = max(cands, key=lambda c: c[2]) if cands else None
    top_hit = bool(top and gts and min(dist(top, g) for g in gts) <= d)
    any_hit = bool(gts and any(min(dist(c, g) for g in gts) <= d for c in cands))
    return top, top_hit, any_hit


def evaluate(frames, d=10, thr=None):
    """frames: list of dict(kind, gt, cands[(x,y,s)]). Returns metric dict at distance d, plus sweep."""
    balls = [f for f in frames if f["kind"] == "ball"]
    nbs = [f for f in frames if f["kind"] == "no_ball"]
    out = {"n_ball": len(balls), "n_no_ball": len(nbs)}
    if balls:
        top1 = sum(frame_hits(f["cands"], f["gt"], d)[1] for f in balls)
        anyh = sum(frame_hits(f["cands"], f["gt"], d)[2] for f in balls)
        out["top1_hit"], out["oracle_hit"] = top1, anyh
        out["top1_recall"], out["oracle_recall"] = top1 / len(balls), anyh / len(balls)
    sweep = []
    for s in (THR if thr is None else thr):
        tp = emit_f = det_tp = det_all = 0
        for f in frames:
            em = [c for c in f["cands"] if c[2] >= s]
            det_all += len(em)
            if em:
                emit_f += 1
            if f["kind"] == "ball":
                det_tp += sum(1 for c in em if f["gt"] and min(dist(c, g) for g in f["gt"]) <= d)
                top = max(em, key=lambda c: c[2]) if em else None
                if top and f["gt"] and min(dist(top, g) for g in f["gt"]) <= d:
                    tp += 1
        rec = tp / len(balls) if balls else None
        prec = tp / emit_f if emit_f else None
        nbfp = sum(1 for f in nbs if any(c[2] >= s for c in f["cands"])) / len(nbs) if nbs else None
        f1 = 2 * rec * prec / (rec + prec) if rec and prec else 0.0
        sweep.append({"s": s, "recall": rec, "prec_f": prec, "prec_det": det_tp / det_all if det_all else None,
                      "emitting_frames": emit_f, "tp_frames": tp, "nb_fp_rate": nbfp, "f1": f1,
                      "dets_per_frame": det_all / len(frames) if frames else 0})
    out["sweep"] = sweep
    if sweep:
        best = max(sweep, key=lambda r: r["f1"])
        out["best"] = best
    return out


def load_wasb(pred_dir):
    """{mode: {clip: [frame dict]}}"""
    modes = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            for m, c in r["pred"].items():
                modes[m][r["clip"]].append({"frame": r["frame"], "kind": r["kind"], "gt": r["gt"],
                                            "cands": [(x, y, s) for x, y, s, _ in c]})
    return modes


def load_rtdetr(cid, frames_needed):
    """in_play_ball/ball candidates per needed frame from the machine label file (None if absent)."""
    p = f"{ML}/{re.sub(r'^arsenal_mancity_', 'arsenal_mancity_20250925_', cid)}.jsonl"
    if not os.path.exists(p):
        return None
    need = set(frames_needed)
    by = defaultdict(list)
    for line in open(p):
        d = json.loads(line)
        if d["frame_index"] in need and d["tags"][0] in ("in_play_ball", "ball"):
            by[d["frame_index"]].append((d["center"]["x"] * W, d["center"]["y"] * H, d.get("confidence", 0.0)))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--v11", default="")
    ap.add_argument("--v11-thr", type=float, default=3.0, help="thumbnail mean-abs-diff below which a frame is the same image")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev = json.load(open(a.evalset))["clips"]
    modes = load_wasb(a.pred_dir)
    v11 = json.load(open(a.v11)) if a.v11 else {}
    res = {"distances": DS, "modes": {}, "per_clip": {}, "rtdetr": {}, "v11_thr": a.v11_thr}

    # --- WASB: pooled + per clip, every mode
    for m, by_clip in modes.items():
        allf = [dict(f, clip=c) for c, fl in by_clip.items() for f in fl]
        res["modes"][m] = {str(d): evaluate(allf, d) for d in DS}
        res["per_clip"][m] = {c: {"n_ball": sum(f["kind"] == "ball" for f in fl),
                                  "n_no_ball": sum(f["kind"] == "no_ball" for f in fl),
                                  "d10": {k: v for k, v in evaluate(fl, 10, thr=[0.3]).items() if k != "sweep"},
                                  "at_0.3": evaluate(fl, 10, thr=[0.3])["sweep"][0]} for c, fl in by_clip.items()}
        # dense-clip pooled groups
        groups = {"dense3": ["arsenal_example", "arsenal_astonvilla_seg080", "arsenal_bournemouth_1st_seg070"],
                  "mancity": [c for c in by_clip if "mancity" in c]}
        res.setdefault("groups", {}).setdefault(m, {})
        for g, cl in groups.items():
            gf = [dict(f, clip=c) for c in cl if c in by_clip for f in by_clip[c]]
            res["groups"][m][g] = {str(d): evaluate(gf, d) for d in DS}

    # --- RT-DETR on the same frames (Man City clips that have machine labels)
    clean_rows, all_rows = [], []
    ref_mode = "full_bilinear"
    for cid, c in ev.items():
        if "mancity" not in cid:
            continue
        need = [int(k) for k in c["ball"]] + list(c["no_ball"])
        rt = load_rtdetr(cid, need)
        if rt is None:
            continue
        vm = v11.get(cid, {})
        for fr in need:
            kind = "ball" if str(fr) in c["ball"] else "no_ball"
            row = {"clip": cid, "frame": fr, "kind": kind, "gt": c["ball"].get(str(fr), []),
                   "cands": rt.get(fr, [])}
            m = vm.get(str(fr))
            row["split"] = m["split"] if m and m["d"] <= a.v11_thr else ("unchecked" if not m else "none")
            all_rows.append(row)
    res["rtdetr"]["n_frames_total"] = len(all_rows)
    splits = defaultdict(int)
    for r in all_rows:
        splits[(r["kind"], r["split"])] += 1
    res["rtdetr"]["split_counts"] = {f"{k[0]}:{k[1]}": v for k, v in sorted(splits.items())}
    clean = [r for r in all_rows if r["split"] in ("none", "valid", "test")]   # not in v11 train
    strict = [r for r in all_rows if r["split"] == "none"]                      # in no v11 split
    res["rtdetr"]["all"] = {str(d): evaluate(all_rows, d) for d in DS}
    res["rtdetr"]["clean_not_in_train"] = {str(d): evaluate(clean, d) for d in DS}
    res["rtdetr"]["clean_no_split"] = {str(d): evaluate(strict, d) for d in DS}
    keys = {(r["clip"], r["frame"]) for r in clean}
    keys_s = {(r["clip"], r["frame"]) for r in strict}
    # WASB modes on exactly the same frame sets
    res["same_frames"] = {}
    for m, by_clip in modes.items():
        for name, ks in (("all_mancity_rt_clips", {(r["clip"], r["frame"]) for r in all_rows}),
                         ("clean_not_in_train", keys), ("clean_no_split", keys_s)):
            fl = [dict(f, clip=c) for c, l in by_clip.items() for f in l if (c, f["frame"]) in ks]
            res["same_frames"].setdefault(name, {})[m] = {str(d): evaluate(fl, d) for d in DS}
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
