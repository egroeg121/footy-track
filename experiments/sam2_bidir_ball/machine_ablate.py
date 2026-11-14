"""Gate ablation for S1 (gap fill): single-pass distance limit and agreement tolerance.

usage: machine_ablate.py --prep prep.json --tracks <dir> --roles test|indep|dev [--thr 0.7 --spacing 10]
Prints one row per (agree_px, single_max): recall@10, precision@10, TP/FP, fills, wrong fills on GT.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import machine_score as MS  # noqa: E402
from footy_track.ball_offline.metrics import Counts, score_clip  # noqa: E402
from machine_common import MAIN_CLIPS, thin  # noqa: E402


def run(clips, tracks_all, thr, sp, agree_px, single_max, mode="S1"):
    MS.AGREE_PX = agree_px
    MS.SINGLE_PASS_MAX = single_max
    tot = Counts()
    fills = wrong = 0
    for c, cl in clips.items():
        pf = {int(t) for t in cl["b1"]}
        anchors = [n["t"] for n in thin(cl["nodes"], pf, thr, sp)]
        frames = range(cl["n_frames"]) if c in MAIN_CLIPS else sorted(set(cl["scored"]) | set(cl["not_broadcast"]))
        dec = MS.sam2_decisions(cl, tracks_all[c], anchors, frames)
        pred = MS.fuse(cl, dec, mode)
        gt = {int(t): tuple(v) for t, v in cl["gt"].items()}
        tot += score_clip(pred, gt, set(cl["no_ball"]), cl["n_frames"], scored=set(cl["scored"]))
        for f, r in dec.items():
            if mode == "S1" and str(f) in cl["b1"]:
                continue
            fills += 1
            if f in gt and math.hypot(r["pos"][0] - gt[f][0], r["pos"][1] - gt[f][1]) > 10:
                wrong += 1
    return tot.summary(), fills, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--roles", default="test")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--spacing", type=int, default=10)
    ap.add_argument("--model", default="small")
    ap.add_argument("--variant", default="crop512")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prep = json.loads(Path(a.prep).read_text())
    roles = set(a.roles.split(","))
    clips = {c: v for c, v in prep["clips"].items() if v["role"] in roles}
    tracks_all = {c: MS.load_tracks(Path(a.tracks), c, a.model, a.variant)[0] for c in clips}
    rows = []
    b1 = Counts()
    for c, cl in clips.items():
        gt = {int(t): tuple(v) for t, v in cl["gt"].items()}
        b1 += score_clip({int(t): tuple(v) for t, v in cl["b1"].items()}, gt, set(cl["no_ball"]), cl["n_frames"], scored=set(cl["scored"]))
    s = b1.summary()
    print(f"B1: R@10 {s['recall@10']:.3f} P@10 {s['precision@10']:.3f} tp={s['tp@10']} fp={s['fp@10']}")
    print("| agree px | single-pass max (frames) | R@10 | P@10 | TP | FP | fills in gaps | fills wrong on GT frames |")
    print("|---|---|---|---|---|---|---|---|")
    for agree in (3.0, 5.0, 8.0):
        for single in (0, 4, 8, 12, 25):
            s, fills, wrong = run(clips, tracks_all, a.thr, a.spacing, agree, single)
            rows.append({"agree_px": agree, "single_max": single, "recall@10": s["recall@10"], "precision@10": s["precision@10"],
                         "tp": s["tp@10"], "fp": s["fp@10"], "fills": fills, "wrong_fills_on_gt": wrong})
            print(f"| {agree:g} | {single} | {s['recall@10']:.3f} | {s['precision@10']:.3f} | {s['tp@10']} | {s['fp@10']} | {fills} | {wrong} |")
    if a.out:
        Path(a.out).write_text(json.dumps({"B1": {k: s[k] for k in ("recall@10", "precision@10", "tp@10", "fp@10")}, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
