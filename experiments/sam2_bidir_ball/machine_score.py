"""Score S1 (gap fill) and S2 (re-anchoring) against B0/B1 with the offline-ball harness metrics.

usage: machine_score.py --prep prep.json --tracks <dir with mtracks__*.json> --out <dir> [--model small --variant crop512]

S1: B1 frames unchanged; frames with no B1 prediction receive a SAM2 position when the gate passes.
S2: B1 anchors unchanged; on every other frame a gate-passing SAM2 position REPLACES the B1 point (or
    adds one where B1 has none).
Gate: forward pass (previous anchor) and backward pass (next anchor) both present and within AGREE_PX
of each other -> their mean; or exactly one pass exists (the other anchor is out of reach), it reports
the ball present, and the frame is within SINGLE_PASS_MAX frames of its anchor.
Scoring uses footy_track.ball_offline.metrics.score_clip on the harness's scored frames.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from footy_track.ball_offline.metrics import Counts, score_clip, wilson  # noqa: E402
from machine_common import AGREE_PX, CONFIGS, MAIN_CLIPS, SINGLE_PASS_MAX, thin  # noqa: E402


def load_tracks(tdir: Path, clip: str, model: str, variant: str):
    p = tdir / f"mtracks__{clip}__{model}__{variant}.json"
    if not p.exists():
        return {}, None
    d = json.loads(p.read_text())
    return {(t["anchor"], t["dir"]): t for t in d["tracks"]}, d


def sam2_decisions(clip: dict, tracks: dict, anchors: list[int], frames) -> dict:
    """frame -> dict(pos, rule, anchor, dist) for every frame in ``frames`` where the gate passes."""
    A = sorted(anchors)
    aset = set(A)
    cuts = clip["cuts"]
    out = {}
    import bisect
    for f in frames:
        if f in aset:
            continue
        i = bisect.bisect_left(A, f)
        ap = A[i - 1] if i > 0 else None
        an = A[i] if i < len(A) else None
        tf = tracks.get((ap, "fwd")) if ap is not None else None
        tb = tracks.get((an, "bwd")) if an is not None else None
        pf = tf["frames"].get(str(f)) if tf else None
        pb = tb["frames"].get(str(f)) if tb else None
        rec = None
        if pf is not None and pb is not None:
            if pf[0] and pb[0] and math.hypot(pf[1] - pb[1], pf[2] - pb[2]) <= AGREE_PX:
                rec = {"pos": ((pf[1] + pb[1]) / 2, (pf[2] + pb[2]) / 2), "rule": "agree",
                       "anchor": ap if f - ap <= an - f else an, "dist": min(f - ap, an - f),
                       "agree_px": math.hypot(pf[1] - pb[1], pf[2] - pb[2]), "ap": ap, "an": an}
        elif pf is not None and pb is None:
            if pf[0] and f - ap <= SINGLE_PASS_MAX:
                rec = {"pos": (pf[1], pf[2]), "rule": "single", "anchor": ap, "dist": f - ap, "agree_px": None, "ap": ap, "an": None}
        elif pb is not None and pf is None:
            if pb[0] and an - f <= SINGLE_PASS_MAX:
                rec = {"pos": (pb[1], pb[2]), "rule": "single", "anchor": an, "dist": an - f, "agree_px": None, "ap": None, "an": an}
        if rec is not None:
            lo, hi = sorted((rec["anchor"], f))
            rec["crosses_cut"] = any(lo < c <= hi for c in cuts)
            out[f] = rec
    return out


def raw_passes(tracks: dict, anchors: list[int], f: int) -> dict:
    """What the forward / backward passes say at frame f (before the gate)."""
    import bisect
    A = sorted(anchors)
    i = bisect.bisect_left(A, f)
    ap = A[i - 1] if i > 0 else None
    an = A[i] if i < len(A) else None
    out = {"prev_anchor_dist": (f - ap) if ap is not None else None, "next_anchor_dist": (an - f) if an is not None else None,
           "fwd": None, "bwd": None}
    tf = tracks.get((ap, "fwd")) if ap is not None else None
    tb = tracks.get((an, "bwd")) if an is not None else None
    pf = tf["frames"].get(str(f)) if tf else None
    pb = tb["frames"].get(str(f)) if tb else None
    if pf is not None:
        out["fwd"] = {"present": bool(pf[0]), "score": pf[4]}
    if pb is not None:
        out["bwd"] = {"present": bool(pb[0]), "score": pb[4]}
    return out


def fuse(clip: dict, dec: dict, mode: str) -> dict[int, tuple]:
    b1 = {int(t): tuple(v) for t, v in clip["b1"].items()}
    pred = dict(b1)
    for f, r in dec.items():
        if mode == "S1" and f in b1:
            continue
        pred[f] = r["pos"]
    return pred


def per_frame_stats(pred, clip, scored):
    """frame -> (kind, tp, fp) for the paired bootstrap."""
    gt = {int(t): v for t, v in clip["gt"].items()}
    nb = set(clip["no_ball"])
    out = {}
    for t in scored:
        if t in gt:
            p = pred.get(t)
            if p is None:
                out[t] = ("gt", 0, 0)
            else:
                d = math.hypot(p[0] - gt[t][0], p[1] - gt[t][1])
                out[t] = ("gt", int(d <= 10), int(d > 10))
        elif t in nb:
            out[t] = ("nb", 0, int(t in pred))
    return out


def blocks(frames_by_clip: dict[str, list[int]], gap: int = 25) -> list[tuple[str, int]]:
    """cluster id per (clip, frame): consecutive scored frames closer than ``gap`` share a block."""
    res = {}
    for c, fs in frames_by_clip.items():
        fs = sorted(fs)
        b = 0
        for i, f in enumerate(fs):
            if i and f - fs[i - 1] > gap:
                b += 1
            res[(c, f)] = (c, b)
    return res


def boot(samples: dict[str, dict], names: list[str], base: str, clusters: dict, n_boot: int = 5000, seed: int = 0):
    """Paired block bootstrap: difference to ``base`` in recall@10 / precision@10 for each name.

    samples[method][(clip, frame)] = (kind, tp, fp)."""
    keys = list(samples[base].keys())
    by_cluster = defaultdict(list)
    for k in keys:
        by_cluster[clusters[k]].append(k)
    cl = list(by_cluster)
    rng = random.Random(seed)
    n_gt_total = sum(1 for k in keys if samples[base][k][0] == "gt")

    def agg(m, ks):
        tp = sum(samples[m][k][1] for k in ks)
        fp = sum(samples[m][k][2] for k in ks)
        n = sum(1 for k in ks if samples[m][k][0] == "gt")
        return tp, fp, n

    res = {}
    for m in names:
        d_rec, d_prec = [], []
        for _ in range(n_boot):
            ks = [k for c in (rng.choice(cl) for _ in cl) for k in by_cluster[c]]
            tpm, fpm, n = agg(m, ks)
            tpb, fpb, _ = agg(base, ks)
            if n == 0:
                continue
            d_rec.append(tpm / n - tpb / n)
            pm = tpm / (tpm + fpm) if tpm + fpm else float("nan")
            pb = tpb / (tpb + fpb) if tpb + fpb else float("nan")
            if pm == pm and pb == pb:
                d_prec.append(pm - pb)
        tpm, fpm, n = agg(m, keys)
        tpb, fpb, _ = agg(base, keys)
        res[m] = {
            "d_recall": (tpm - tpb) / n_gt_total,
            "d_recall_ci95": [float(np.percentile(d_rec, 2.5)), float(np.percentile(d_rec, 97.5))],
            "d_precision": (tpm / (tpm + fpm) - tpb / (tpb + fpb)) if (tpm + fpm) and (tpb + fpb) else None,
            "d_precision_ci95": [float(np.percentile(d_prec, 2.5)), float(np.percentile(d_prec, 97.5))] if d_prec else None,
            "n_blocks": len(cl),
        }
    return res


def mcnemar_exact(fixed: int, broke: int) -> float:
    """two-sided exact binomial p for discordant pairs (fixed vs broke)."""
    n = fixed + broke
    if n == 0:
        return 1.0
    k = min(fixed, broke)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--variant", default="crop512")
    ap.add_argument("--roles", default="test", help="test | dev | test,dev | indep")
    ap.add_argument("--ceil", default=None, help="ceil.json from machine_ceiling.py")
    args = ap.parse_args()
    prep = json.loads(Path(args.prep).read_text())
    ceil = json.loads(Path(args.ceil).read_text()) if args.ceil else {}
    tdir = Path(args.tracks)
    roles = set(args.roles.split(","))
    out = {"model": args.model, "variant": args.variant, "configs": {}, "baselines": {}}

    clips = {c: v for c, v in prep["clips"].items() if v["role"] in roles}
    tracks_all = {c: load_tracks(tdir, c, args.model, args.variant) for c in clips}
    perf = {c: (d or {}).get("perf") for c, (_, d) in tracks_all.items()}
    out["perf_by_clip"] = perf
    tot_sec = sum(sum(t["sec"] for t in tr.values()) for tr, _ in tracks_all.values())
    tot_frames = sum(sum(t["n_frames"] for t in tr.values()) for tr, _ in tracks_all.values())
    out["runtime"] = {"gpu_seconds": round(tot_sec, 1), "frames_propagated": tot_frames,
                      "sec_per_frame": round(tot_sec / max(tot_frames, 1), 4)}

    # ---- baselines ------------------------------------------------------------
    def pooled(pred_of):
        tot = Counts()
        for c, cl in clips.items():
            gt = {int(t): tuple(v) for t, v in cl["gt"].items()}
            tot += score_clip(pred_of(c, cl), gt, set(cl["no_ball"]), cl["n_frames"], scored=set(cl["scored"]))
        return tot

    def summarize(tot: Counts) -> dict:
        s = tot.summary()
        lo, hi = wilson(s["tp@10"], s["n_gt_ball"])
        s["recall@10_wilson"] = [lo, hi]
        return s

    def main_cov(pred_of):
        n = p = 0
        for c, cl in clips.items():
            if c in MAIN_CLIPS:
                n += cl["n_frames"]
                p += len(pred_of(c, cl))
        return p / n if n else float("nan")

    methods = {"B0": lambda c, cl: {int(t): tuple(v) for t, v in cl["b0"].items()},
               "B1": lambda c, cl: {int(t): tuple(v) for t, v in cl["b1"].items()}}
    for name, fn in methods.items():
        s = summarize(pooled(fn))
        s["coverage_main_clips"] = main_cov(fn)
        out["baselines"][name] = s

    # ---- configs --------------------------------------------------------------
    samples: dict[str, dict] = {"B0": {}, "B1": {}}
    clusters: dict = {}
    frames_by_clip = {c: sorted(cl["scored"]) for c, cl in clips.items()}
    cl_map = blocks(frames_by_clip)
    for c, cl in clips.items():
        for name in ("B0", "B1"):
            for t, v in per_frame_stats(methods[name](c, cl), cl, cl["scored"]).items():
                samples[name][(c, t)] = v
        for t in cl["scored"]:
            clusters[(c, t)] = cl_map[(c, t)]
    detail = {}
    for thr, sp in CONFIGS:
        key = f"thr{thr}_sp{sp}"
        res_cfg = {"thr": thr, "spacing": sp}
        preds = {"S1": {}, "S2": {}}
        decs = {}
        anchors_of = {}
        stats = {"n_anchors": 0, "fills": {"S1": defaultdict(int), "S2": defaultdict(int)}}
        for c, cl in clips.items():
            pf = {int(t) for t in cl["b1"]}
            anchors = [n["t"] for n in thin(cl["nodes"], pf, thr, sp)]
            stats["n_anchors"] += len(anchors)
            anchors_of[c] = anchors
            tracks, _ = tracks_all[c]
            if c in MAIN_CLIPS:
                frames = range(cl["n_frames"])
            else:
                rel = set(cl["scored"]) | set(cl["not_broadcast"]) | set(cl["out_of_play"])
                frames = sorted(rel)
            dec = sam2_decisions(cl, tracks, anchors, frames)
            decs[c] = dec
            for mode in ("S1", "S2"):
                preds[mode][c] = fuse(cl, dec, mode)
                b1 = cl["b1"]
                for f, r in dec.items():
                    new = (f not in {int(t) for t in b1}) if mode == "S1" else True
                    if mode == "S1" and str(f) in b1:
                        continue
                    stats["fills"][mode][r["rule"]] += 1
                    if r["crosses_cut"]:
                        stats["fills"][mode]["crosses_cut"] += 1
                    if str(f) in b1:
                        stats["fills"][mode]["replaced_b1"] += 1
        for mode in ("S1", "S2"):
            tot = pooled(lambda c, cl, mode=mode: preds[mode][c])
            s = summarize(tot)
            cov = [len(preds[mode][c]) / cl["n_frames"] for c, cl in clips.items() if c in MAIN_CLIPS]
            n_f = sum(cl["n_frames"] for c, cl in clips.items() if c in MAIN_CLIPS)
            n_p = sum(len(preds[mode][c]) for c, cl in clips.items() if c in MAIN_CLIPS)
            s["coverage_main_clips"] = n_p / n_f if n_f else float("nan")
            res_cfg[mode] = s
            for c, cl in clips.items():
                for t, v in per_frame_stats(preds[mode][c], cl, cl["scored"]).items():
                    samples.setdefault(f"{key}:{mode}", {})[(c, t)] = v
        for mode in ("S1", "S2"):
            n_fill_gt = n_fill_ok = 0
            for c, cl in clips.items():
                gtd = {int(t): v for t, v in cl["gt"].items()}
                b1k = {int(t) for t in cl["b1"]}
                for f, r in decs[c].items():
                    if f in gtd and f in set(cl["scored"]) and (mode == "S2" or f not in b1k):
                        n_fill_gt += 1
                        n_fill_ok += int(math.hypot(r["pos"][0] - gtd[f][0], r["pos"][1] - gtd[f][1]) <= 10)
            res_cfg[f"{mode}_gate_passing_points_on_GT_frames"] = {"n": n_fill_gt, "within10px": n_fill_ok}
        res_cfg["n_anchors"] = stats["n_anchors"]
        res_cfg["fills"] = {m: dict(v) for m, v in stats["fills"].items()}
        # ---- paired bootstrap vs B1 and B0 ---------------------------------------
        names = [f"{key}:S1", f"{key}:S2"]
        res_cfg["bootstrap_vs_B1"] = boot(samples, names, "B1", clusters)
        res_cfg["bootstrap_vs_B0"] = boot(samples, names, "B0", clusters)
        iid = {k: (k[0], k[1]) for k in clusters}
        res_cfg["bootstrap_iid_vs_B1"] = boot(samples, names, "B1", iid)
        # ---- transitions S2 vs B1 on GT frames -----------------------------------
        for mode in ("S1", "S2"):
            fixed = broke = same_hit = same_miss = 0
            fixed_in = fixed_out = 0
            for (c, t), (kind, tp, fp) in samples[f"{key}:{mode}"].items():
                if kind != "gt":
                    continue
                b = samples["B1"][(c, t)]
                if b[1] and not tp:
                    broke += 1
                elif tp and not b[1]:
                    fixed += 1
                    if ceil:
                        if t in set(ceil.get(c, {}).get("reachable", [])):
                            fixed_in += 1
                        else:
                            fixed_out += 1
                elif tp:
                    same_hit += 1
                else:
                    same_miss += 1
            res_cfg[f"{mode}_vs_B1_gt_transitions"] = {"fixed": fixed, "broke": broke, "both_hit": same_hit, "both_miss": same_miss,
                                                       "fixed_inside_detector_ceiling": fixed_in, "fixed_beyond_detector_ceiling": fixed_out,
                                                       "mcnemar_exact_p": mcnemar_exact(fixed, broke)}
        # ---- no-ball / out-of-shot ----------------------------------------------
        rel_of = {c: set(cl["scored"]) | set(cl["not_broadcast"]) | set(cl["out_of_play"]) for c, cl in clips.items()}
        nb = {"no_ball_frames": 0, "no_ball_b1_emits": 0, "no_ball_sam2_gate_pass": 0,
              "S1_emits": 0, "S2_emits": 0, "per_frame": []}
        ob = {"not_broadcast": {"n": 0, "S1": 0, "S2": 0, "b1": 0}, "out_of_play": {"n": 0, "S1": 0, "S2": 0, "b1": 0}}
        for c, cl in clips.items():
            for f in cl["no_ball"]:
                if f not in set(cl["scored"]):
                    continue
                nb["no_ball_frames"] += 1
                e1 = str(f) in cl["b1"]
                nb["no_ball_b1_emits"] += int(e1)
                nb["no_ball_sam2_gate_pass"] += int(f in decs[c])
                es1 = f in preds["S1"][c]
                es2 = f in preds["S2"][c]
                nb["S1_emits"] += int(es1)
                nb["S2_emits"] += int(es2)
                raw = raw_passes(tracks_all[c][0], anchors_of[c], f)
                nb["per_frame"].append({"clip": c, "frame": f, "role": cl["role"], "b1": e1, "gate": f in decs[c], "S1": es1, "S2": es2, **raw})
            for kind in ("not_broadcast", "out_of_play"):
                for f in cl[kind]:
                    if c not in MAIN_CLIPS and f not in rel_of[c]:
                        continue
                    ob[kind]["n"] += 1
                    ob[kind]["b1"] += int(str(f) in cl["b1"])
                    ob[kind]["S1"] += int(f in preds["S1"][c])
                    ob[kind]["S2"] += int(f in preds["S2"][c])
        pf_ = nb["per_frame"]
        nb["n_covered_by_a_pass"] = sum(1 for x in pf_ if x["fwd"] or x["bwd"])
        nb["n_covered_by_both_passes"] = sum(1 for x in pf_ if x["fwd"] and x["bwd"])
        nb["n_a_pass_says_ball"] = sum(1 for x in pf_ if (x["fwd"] and x["fwd"]["present"]) or (x["bwd"] and x["bwd"]["present"]))
        nb["n_both_passes_say_ball"] = sum(1 for x in pf_ if x["fwd"] and x["bwd"] and x["fwd"]["present"] and x["bwd"]["present"])
        res_cfg["no_ball"] = nb
        res_cfg["out_of_shot"] = ob
        out["configs"][key] = res_cfg
        detail[key] = {c: {str(f): r for f, r in d.items()} for c, d in decs.items()}
    out["clusters"] = {"n_blocks_total": len(set(clusters.values())), "block_gap_frames": 25}
    outd = Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.variant}_{'-'.join(sorted(roles))}"
    (outd / f"machine_summary_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    (outd / f"machine_decisions_{tag}.json").write_text(json.dumps(detail))
    print(json.dumps({k: out[k] for k in ("baselines", "runtime")}, indent=1, default=float)[:1500])
    for key, r in out["configs"].items():
        for mode in ("S1", "S2"):
            s = r[mode]
            b = r["bootstrap_vs_B1"][f"{key}:{mode}"]
            print(f"{key} {mode}: R@10 {s['recall@10']:.3f} P@10 {s['precision@10']:.3f} tp={s['tp@10']} fp={s['fp@10']} "
                  f"cov={s['coverage_main_clips']:.3f} | dR {b['d_recall']:+.3f} {b['d_recall_ci95']} | fills {r['fills'][mode]} "
                  f"| trans {r[mode+'_vs_B1_gt_transitions']}")


if __name__ == "__main__":
    main()
