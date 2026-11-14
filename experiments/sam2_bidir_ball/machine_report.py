"""Markdown tables for docs/experiments/sam2_machine_anchors.md from the scoring JSONs (stdlib).

usage: machine_report.py --dir <dir with machine_summary_*.json, ablate_*.json, machine_verdicts.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from math import isnan


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and isnan(x)) else f"{x:.3f}"


def wil(s):
    lo, hi = s["recall@10_wilson"]
    return f"[{lo:.2f}-{hi:.2f}]"


def row(name, s, extra=""):
    return (f"| {name} | {s['n_gt_ball']} | {f3(s['recall@10'])} {wil(s)} | {f3(s['precision@10'])} ({s['tp@10']}/{s['tp@10'] + s['fp@10']}) "
            f"| {f3(s['f1@10'])} | {f3(s['recall@5'])} / {f3(s['recall@20'])} | {f3(s['coverage_main_clips'])} | {f3(s['median_err_px'])} |{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    L = []
    for role, title in (("test", "Test set (9 clean Man City clips, harness `test`)"), ("indep", "Independent check: arsenal_example (not in detector training; detections regenerated)")):
        p = d / f"machine_summary_small_crop512_{role}.json"
        if not p.exists():
            continue
        S = json.loads(p.read_text())
        L.append(f"### {title}\n")
        b = S["baselines"]
        L.append("| method | GT ball frames | recall@10 [95% Wilson] | precision@10 (TP/emitted) | F1@10 | recall@5 / @20 | coverage (full-propagation clips) | median err px | fixed / broke vs B1 (inside / beyond detector ceiling) | McNemar p | dR vs B1, iid 95% CI | gate fills (agree / single) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        L.append(row("B0 per-frame (tau 0.4)", b["B0"], " - | - | - | - |"))
        L.append(row("B1 Viterbi (camera on)", b["B1"], " - | - | - | - |"))
        for key, c in S["configs"].items():
            for mode in ("S1", "S2"):
                tr = c[f"{mode}_vs_B1_gt_transitions"]
                bt = c["bootstrap_iid_vs_B1"][f"{key}:{mode}"]
                fl = c["fills"][mode]
                ex = (f" {tr['fixed']} / {tr['broke']} ({tr['fixed_inside_detector_ceiling']} / {tr['fixed_beyond_detector_ceiling']}) | {tr['mcnemar_exact_p']:.4f} | "
                      f"{bt['d_recall']:+.3f} [{bt['d_recall_ci95'][0]:+.3f}, {bt['d_recall_ci95'][1]:+.3f}] | {fl.get('agree', 0)} / {fl.get('single', 0)} |")
                L.append(row(f"{mode} anchors conf>={c['thr']}, spacing>={c['spacing']}", c[mode], ex))
        n_nb = b["B1"]["n_no_ball"]
        L.append(f"\nGT ball frames scored: {b['B1']['n_gt_ball']}; explicit no_ball frames scored: {n_nb}. Anchors in each config: "
                 + ", ".join(f"{k}: {v['n_anchors']}" for k, v in S["configs"].items()) + f". SAM2 runtime for these tracks: {S['runtime']}.\n")
    for role in ("test", "dev", "indep"):
        p = d / f"machine_summary_small_crop512_{role}.json"
        if not p.exists():
            continue
        S = json.loads(p.read_text())
        c = S["configs"]["thr0.7_sp10"]
        nb = c["no_ball"]
        L.append(f"### No-ball and out-of-shot frames, role={role} (primary config conf>=0.7, spacing>=10)\n")
        L.append(f"explicit no_ball frames scored: {nb['no_ball_frames']}; B1 emits a point on {nb['no_ball_b1_emits']}; frames inside the reach of at least one SAM2 pass: "
                 f"{nb['n_covered_by_a_pass']}, of both passes: {nb['n_covered_by_both_passes']}; at least one raw pass says 'ball present': {nb['n_a_pass_says_ball']}; "
                 f"both passes say present: {nb['n_both_passes_say_ball']}; gate passes (a SAM2 fill would be emitted): {nb['no_ball_sam2_gate_pass']}; "
                 f"final emissions S1 / S2: {nb['S1_emits']} / {nb['S2_emits']} (all of them B1's own points).\n")
        ob = c["out_of_shot"]
        L.append(f"not_broadcast frames: n={ob['not_broadcast']['n']}, B1 emits {ob['not_broadcast']['b1']}, S1 {ob['not_broadcast']['S1']}, S2 {ob['not_broadcast']['S2']}; "
                 f"out_of_play frames: n={ob['out_of_play']['n']}.\n")
    for role in ("test", "indep"):
        for thr in ("0.7", "0.8"):
            p = d / f"ablate_{role}_thr{thr}.json"
            if not p.exists():
                continue
            A = json.loads(p.read_text())
            L.append(f"### Gate ablation ({role}, anchors conf>={thr}, spacing>=10; S1 gap fill)\n")
            L.append(f"B1: recall@10 {A['B1']['recall@10']:.3f}, precision@10 {A['B1']['precision@10']:.3f} (TP {A['B1']['tp@10']}, FP {A['B1']['fp@10']})\n")
            L.append("| agree px | single-pass max (frames) | R@10 | P@10 | TP | FP | fills in gaps | fills wrong on GT frames |")
            L.append("|---|---|---|---|---|---|---|---|")
            for r in A["rows"]:
                if r["agree_px"] != 5.0:
                    continue
                L.append(f"| {r['agree_px']:g} | {r['single_max']} | {r['recall@10']:.3f} | {r['precision@10']:.3f} | {r['tp']} | {r['fp']} | {r['fills']} | {r['wrong_fills_on_gt']} |")
            L.append("")
    p = d / "machine_verdicts.json"
    if p.exists():
        V = json.loads(p.read_text())["summary"]
        L.append("### Ball Check verdict probe (candidate-level, primary config)\n")
        L.append("| method | ball verdicts hit (n) | recall | not_ball verdicts hit by an emitted point (n) | emission rate on not_ball |")
        L.append("|---|---|---|---|---|")
        for m, a in V.items():
            L.append(f"| {m} | {a['pos_hit']}/{a['pos']} | {f3(a['recall_on_ball'])} | {a['neg_hit']}/{a['neg']} | {f3(a['emission_on_not_ball'])} |")
        L.append("")
    print("\n".join(L))


if __name__ == "__main__":
    main()
