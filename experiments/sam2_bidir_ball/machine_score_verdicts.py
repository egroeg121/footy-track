"""Ball Check verdict probe: do SAM2 fills land on confirmed non-balls?

Same candidate-level metric as ``scripts/eval_offline_ball_verdicts.py`` (recall on ``ball`` verdicts,
emission rate on ``not_ball`` verdicts within 10 px), for B1, S1 and S2 (primary anchor config).
``not_ball`` emission of a method = it emitted a point within 10 px of a candidate a human said is not
the ball. SAM2 fills at other places on the frame are not counted here (their truth is unknown).

usage: machine_score_verdicts.py --prep prep_verdicts.json --tracks <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from machine_score import load_tracks, sam2_decisions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--variant", default="crop512")
    args = ap.parse_args()
    prep = json.loads(Path(args.prep).read_text())
    agg = {m: defaultdict(int) for m in ("B1", "S1", "S2")}
    per_item = []
    for cid, c in prep["clips"].items():
        tracks, _ = load_tracks(Path(args.tracks), cid, args.model, args.variant)
        frames = sorted({it[0] for it in c["items"]})
        dec = sam2_decisions({"cuts": c["cuts"]}, tracks, c["anchors"], frames)
        b1 = {int(t): tuple(v) for t, v in c["b1"].items()}
        for t, xy, v in c["items"]:
            preds = {"B1": b1.get(t)}
            d = dec.get(t)
            preds["S1"] = preds["B1"] if (t in b1 or d is None) else d["pos"]
            preds["S2"] = d["pos"] if d is not None and t not in set(c["anchors"]) else preds["B1"]
            for m, p in preds.items():
                a = agg[m]
                hit = p is not None and math.hypot(p[0] - xy[0], p[1] - xy[1]) <= 10
                if v == "ball":
                    a["pos"] += 1
                    a["pos_hit"] += int(hit)
                else:
                    a["neg"] += 1
                    a["neg_hit"] += int(hit)
            fill_only = d is not None and t not in b1
            per_item.append({"clip": cid, "t": t, "verdict": v, "b1_emits": t in b1, "gate_pass": d is not None,
                             "fill_in_gap": fill_only, "rule": d["rule"] if d else None,
                             "S1_hit": preds["S1"] is not None and math.hypot(preds["S1"][0] - xy[0], preds["S1"][1] - xy[1]) <= 10,
                             "S2_hit": preds["S2"] is not None and math.hypot(preds["S2"][0] - xy[0], preds["S2"][1] - xy[1]) <= 10})
    out = {m: dict(a) for m, a in agg.items()}
    for m, a in out.items():
        a["recall_on_ball"] = a["pos_hit"] / a["pos"] if a["pos"] else None
        a["emission_on_not_ball"] = a["neg_hit"] / a["neg"] if a["neg"] else None
    outd = Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "machine_verdicts.json").write_text(json.dumps({"summary": out, "items": per_item}, indent=1))
    for m, a in out.items():
        print(m, a)
    n_fill = sum(1 for i in per_item if i["fill_in_gap"])
    print("verdict frames: total", len(per_item), "B1-gap frames with a gate-passing SAM2 fill", n_fill,
          "| of which on a not_ball location:", sum(1 for i in per_item if i["fill_in_gap"] and i["verdict"] != "ball" and i["S1_hit"]))


if __name__ == "__main__":
    main()
