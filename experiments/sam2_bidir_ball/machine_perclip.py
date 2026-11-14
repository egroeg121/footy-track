"""Per-clip B0 / B1 / S1 / S2 recall@10 and precision@10 for the primary config (CPU).

usage: machine_perclip.py --prep prep.json --tracks <dir> --roles test --thr 0.7 --spacing 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import machine_score as MS  # noqa: E402
from footy_track.ball_offline.metrics import score_clip  # noqa: E402
from machine_common import MAIN_CLIPS, thin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--roles", default="test")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--spacing", type=int, default=10)
    a = ap.parse_args()
    prep = json.loads(Path(a.prep).read_text())
    print("| clip | GT ball frames | B0 R / P | B1 R / P | S1 R / P | S2 R / P |")
    print("|---|---|---|---|---|---|")
    for c, cl in prep["clips"].items():
        if cl["role"] not in a.roles.split(","):
            continue
        gt = {int(t): tuple(v) for t, v in cl["gt"].items()}
        scored = set(cl["scored"])
        n_gt = sum(1 for t in gt if t in scored)
        if n_gt < 10:
            continue
        tracks, _ = MS.load_tracks(Path(a.tracks), c, "small", "crop512")
        pf = {int(t) for t in cl["b1"]}
        anchors = [n["t"] for n in thin(cl["nodes"], pf, a.thr, a.spacing)]
        frames = range(cl["n_frames"]) if c in MAIN_CLIPS else sorted(scored)
        dec = MS.sam2_decisions(cl, tracks, anchors, frames)
        res = {}
        for name, pred in (("B0", {int(t): tuple(v) for t, v in cl["b0"].items()}), ("B1", {int(t): tuple(v) for t, v in cl["b1"].items()}),
                           ("S1", MS.fuse(cl, dec, "S1")), ("S2", MS.fuse(cl, dec, "S2"))):
            s = score_clip(pred, gt, set(cl["no_ball"]), cl["n_frames"], scored=scored).summary()
            res[name] = f"{s['recall@10']:.3f} / {s['precision@10']:.3f} ({s['tp@10']}/{s['tp@10'] + s['fp@10']})"
        print(f"| {c} | {n_gt} | {res['B0']} | {res['B1']} | {res['S1']} | {res['S2']} |")


if __name__ == "__main__":
    main()
