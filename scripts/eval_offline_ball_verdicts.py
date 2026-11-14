#!/usr/bin/env python
# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408
"""Independent cross-check of B0/B1 against Ball Check verdicts.

Ball Check (``ball_gt_marks/ball_checks/``) holds human yes/no verdicts on
individual detector candidates.  Unlike the GT marks these include confirmed
*non*-balls, so they bound how often a method emits a point that a human said
is not the ball, without relying on the sparse GT frame selection.  Verdicts
are sampled from the detector's own candidates (borderline-heavy), so recall on
``ball`` verdicts is bounded by 1 by construction and the numbers are only
comparable between methods, not absolute.

    uv run python scripts/eval_offline_ball_verdicts.py \
        --results docs/experiments/offline_ball_baseline.json

Frames within 5 frames of a detector-training image are skipped.  Results are
merged into the results JSON under ``ball_check`` and appended to the markdown.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from footy_track.ball_offline import baselines as bl
from footy_track.ball_offline.data import (
    H_PX,
    W_PX,
    contaminated_frames,
    data_root,
    load_clip,
    machine_stem_map,
    roboflow_frames,
)
from footy_track.ball_offline.solver import SolverParams
from footy_track.clip_id import canonical_clip_id


def load_verdicts(root: Path):
    latest = {}
    for p in sorted((root / "ball_gt_marks" / "ball_checks").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            latest[(canonical_clip_id(r["clip"]), r["frame_index"], r["box_index"])] = r
    out = defaultdict(list)
    for (cid, t, _), r in latest.items():
        v = r["verdict"]
        if v == "ball" and r.get("bbox"):
            b = r["bbox"]
        elif v == "corrected" and r.get("corrected_bbox"):
            b, v = r["corrected_bbox"], "ball"
        elif v == "not_ball" and r.get("bbox"):
            b = r["bbox"]
        else:
            continue  # unsure / tombstone
        out[cid].append(
            (t, ((b["x"] + b["w"] / 2) * W_PX, (b["y"] + b["h"] / 2) * H_PX), v)
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="docs/experiments/offline_ball_baseline.json")
    ap.add_argument("--out-md", default="docs/experiments/offline_ball_baseline.md")
    args = ap.parse_args()
    res = json.loads(Path(args.results).read_text())
    cam = res["B1"]["camera_selected_on_dev"]
    p1 = SolverParams(**res["B1"]["tuned_params"]["camera_on" if cam else "camera_off"])
    tau = res["B0"]["tau_selected_on_dev"]
    gate = res["B2"]["gate_px_selected_on_dev"]
    root = data_root()
    ml = machine_stem_map(root)
    rf = roboflow_frames(root)
    verdicts = load_verdicts(root)
    methods = {
        f"B0 tau={tau}": lambda c: bl.b0_per_frame(c, tau),
        "B1": lambda c: bl.b1_solver(c, p1)[0],
        "B2": lambda c: bl.b2_rescue(c, p1, gate_px=gate)[0],
    }
    agg = {
        m: dict(pos=0, pos_hit=0, neg=0, neg_hit=0, on_pos=0, on_any=0) for m in methods
    }
    n_skipped_contam = 0
    clips_used = 0
    for cid, items in sorted(verdicts.items()):
        if cid not in ml:
            continue
        bad = contaminated_frames(cid, rf)
        items = [it for it in items if it[0] not in bad]
        n_skipped_contam += len(verdicts[cid]) - len(items)
        if not items:
            continue
        clip = load_clip(cid, root, with_camera=p1.camera)
        clips_used += 1
        for m, fn in methods.items():
            pred = fn(clip)
            a = agg[m]
            for t, (x, y), v in items:
                pr = pred.get(t)
                if v == "ball":
                    a["pos"] += 1
                    if pr is not None and np.hypot(pr[0] - x, pr[1] - y) <= 10:
                        a["pos_hit"] += 1
                else:
                    a["neg"] += 1
                    if pr is not None and np.hypot(pr[0] - x, pr[1] - y) <= 10:
                        a["neg_hit"] += 1
            # emitted points that coincide with a verdict point: share that are positive
            by_t = defaultdict(list)
            for t, xy, v in items:
                by_t[t].append((xy, v))
            for t, lst in by_t.items():
                pr = pred.get(t)
                if pr is None:
                    continue
                near = [
                    v for xy, v in lst if np.hypot(pr[0] - xy[0], pr[1] - xy[1]) <= 10
                ]
                if near:
                    a["on_any"] += 1
                    a["on_pos"] += int("ball" in near)
    out = {
        "clips_used": clips_used,
        "verdicts_skipped_contaminated": n_skipped_contam,
        "methods": {},
    }
    for m, a in agg.items():
        out["methods"][m] = {
            **a,
            "recall_on_ball_verdicts": a["pos_hit"] / a["pos"] if a["pos"] else None,
            "emission_rate_on_not_ball_verdicts": a["neg_hit"] / a["neg"]
            if a["neg"]
            else None,
            "precision_of_emissions_landing_on_a_verdict": a["on_pos"] / a["on_any"]
            if a["on_any"]
            else None,
        }
    res["ball_check"] = out
    Path(args.results).write_text(json.dumps(res, indent=1, default=float))
    L = [
        "\n## 8. Cross-check against Ball Check verdicts\n",
        f"{clips_used} clips with machine labels, {out['methods'][next(iter(agg))]['pos']} confirmed-ball and "
        f"{out['methods'][next(iter(agg))]['neg']} confirmed-not-ball candidates "
        f"({n_skipped_contaminated(n_skipped_contam)} verdicts dropped for detector-training overlap). "
        "Candidate-level, independent of the GT frame selection; verdicts are sampled from detector candidates so "
        "only compare methods with each other.\n",
        "| method | ball verdicts hit (n) | recall | not_ball verdicts emitted (n) | emission rate on not_ball | emitted points on a verdict that are positive |",
        "|---|---|---|---|---|---|",
    ]
    for m, v in out["methods"].items():

        def f(x):
            return "n/a" if x is None else f"{x:.3f}"

        L.append(
            f"| {m} | {v['pos_hit']}/{v['pos']} | {f(v['recall_on_ball_verdicts'])} | {v['neg_hit']}/{v['neg']} | "
            f"{f(v['emission_rate_on_not_ball_verdicts'])} | {v['on_pos']}/{v['on_any']} = {f(v['precision_of_emissions_landing_on_a_verdict'])} |"
        )
    with open(args.out_md, "a") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def n_skipped_contaminated(n: int) -> int:
    return n


if __name__ == "__main__":
    main()
