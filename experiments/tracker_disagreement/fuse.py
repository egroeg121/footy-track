"""Per-frame positions of the three trackers on the chosen Norwich clips (mini, one thread-capped process; numpy only).

  T1  A++ (round-3 YOLO11s 3ch, seed 0) every-frame candidates -> harness B1 trajectory solver, camera on, dev-tuned parameters, raw (RT-DETR) calibration knots
      (the harness is the offline-ball-tracking worktree, imported read-only, as in experiments/temporal_detector/pipeline_eval.py; the empty ground truth means no human
      label is read)
  T2  SAM2.1-small crop512   (track_seeds.py --engine sam2)
  T3  SAM 3 box crop512      (track_seeds.py --engine sam3)
Both SAM trackers: per processed frame the nearest-seed (or S1-averaged) position; a tracker "has a ball" when the SAM object-score logit is > 0 (the model's own
object-present decision) or the frame is a seed frame.  Seed frames are flagged (they are detections, not tracking).

    OMP_NUM_THREADS=1 uv run --no-project --python /usr/bin/python3 --with numpy python fuse.py --inputs IN --npz CLIPS_OUT --sam2 DIR --sam3 DIR --out fused.json.gz
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HARNESS = Path("/home/george/footy-track/.claude/worktrees/agent-a8507f51f3d63f415/src")
CAMERA_ON = dict(min_conf=0.3, bias=1.0, emit_scale=1.0, kappa=1.5, lam_pos=0.5, lam_vel=0.0, vmax=40.0, r0=12.0, dmax=2.0,
                 mu=0.02, static_px=3.0, max_gap=25, camera=True, fill_gap=10, fill_min_p=0.8, fill_mode="kalman")
SCORE_MIN = 0.0


def csr(z, key):
    ptr, det = z[f"{key}_ptr"], z[f"{key}_det"].astype(float)
    n = int(z["n_frames"])
    rows = [det[ptr[t]:ptr[t + 1]] for t in range(n)]
    rows = [r[np.argsort(-r[:, 4])] if len(r) else r for r in rows]
    cnt = np.array([len(r) for r in rows])
    new_ptr = np.zeros(n + 1, np.int64)
    new_ptr[1:] = np.cumsum(cnt)
    arr = np.concatenate(rows, axis=0) if cnt.sum() else np.zeros((0, 5))
    return new_ptr, arr


def t1_clip(clip_id: str, z, bl, hd, hs):
    n = int(z["n_frames"])
    pp, parr = z["players_ptr"], z["players_xywh"].astype(float)
    players = [parr[pp[t]:pp[t + 1]].reshape(-1, 4) for t in range(n)]
    shift, cut = hd.estimate_camera_shift(players)
    ptr, arr = csr(z, "App_s0")
    clip = hd.ClipData(clip_id=clip_id, n_frames=n, ptr=ptr, xy=arr[:, 0:2].copy(), conf=arr[:, 4].copy(), wh=arr[:, 2:4].copy(), gt={}, no_ball=set(),
                       cam_shift=shift, cam_cut=cut, players=players, gt_aliases=[], gt_conflicts=0)
    pred, nodes = bl.b1_solver(clip, hs.SolverParams(**CAMERA_ON))
    node_t = {nd.t: nd for nd in nodes}
    rptr, rarr = csr(z, "ref")
    out = {"x": [None] * n, "y": [None] * n, "node": [0] * n, "conf": [None] * n, "top1": [None] * n, "top1_conf": [None] * n, "rt_top1": [None] * n, "rt_conf": [None] * n}
    for t in range(n):
        if t in pred:
            out["x"][t], out["y"][t] = round(float(pred[t][0]), 2), round(float(pred[t][1]), 2)
        if t in node_t:
            out["node"][t], out["conf"][t] = 1, round(float(node_t[t].conf), 4)
        s, e = ptr[t], ptr[t + 1]
        if e > s:
            out["top1"][t], out["top1_conf"][t] = [round(float(arr[s, 0]), 1), round(float(arr[s, 1]), 1)], round(float(arr[s, 4]), 4)
        s, e = rptr[t], rptr[t + 1]
        if e > s:
            out["rt_top1"][t], out["rt_conf"][t] = [round(float(rarr[s, 0]), 1), round(float(rarr[s, 1]), 1)], round(float(rarr[s, 4]), 4)
    # player context per frame: number of player boxes within 60 px of the emitted position, and whether it sits inside a (slightly grown) player box
    near = [0] * n
    inside = [0] * n
    for t in range(n):
        if out["x"][t] is None or not len(players[t]):
            continue
        d = np.hypot(players[t][:, 0] - out["x"][t], players[t][:, 1] - out["y"][t])
        near[t] = int((d < 60).sum())
        inside[t] = int(np.any((abs(players[t][:, 0] - out["x"][t]) < players[t][:, 2] / 2 + 3) & (abs(players[t][:, 1] - out["y"][t]) < players[t][:, 3] / 2 + 3)))
    out["players_near"], out["in_player_box"] = near, inside
    out["cut"] = [int(c) for c in cut]
    return out


def sam_clip(path: Path, n: int, seeds: dict):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else None
    out = {"x": [None] * n, "y": [None] * n, "score": [None] * n, "gate": [None] * n, "dist": [None] * n, "seed_src": [None] * n, "is_seed": [0] * n, "any_row": [0] * n}
    if rows is None:
        return None
    for r in rows:
        t = r["frame"]
        if t >= n:
            continue
        out["any_row"][t] = 1
        out["score"][t], out["gate"][t], out["dist"][t] = r["score"], r["gate"], r["dist"]
        out["seed_src"][t] = seeds.get((r["clip"], r["seed"]), "?")
        out["is_seed"][t] = int(r["gate"] == "seed")
        if r["gate"] == "seed" or r["score"] > SCORE_MIN:
            out["x"][t], out["y"][t] = round(r["x"], 2), round(r["y"], 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--sam2", type=Path, required=True)
    ap.add_argument("--sam3", type=Path, required=True)
    ap.add_argument("--seeds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(HARNESS))
    from footy_track.ball_offline import baselines as bl, data as hd, solver as hs
    meta = json.loads((a.inputs / "clips.json").read_text())
    seeds = {}
    for l in a.seeds.read_text().splitlines():
        if l.strip():
            s = json.loads(l)
            seeds[(s["clip"], s["id"])] = s["source"]
    out = {"clips": {}, "exclude": meta["exclude"], "judged": meta["judged"], "params_camera_on": CAMERA_ON, "score_min": SCORE_MIN}
    for clip, cm in meta["clips"].items():
        z = np.load(a.npz / f"{clip}.npz")
        n = int(z["n_frames"])
        assert n == cm["n_frames"], (clip, n, cm["n_frames"])
        t1 = t1_clip(clip, z, bl, hd, hs)
        t2 = sam_clip(a.sam2 / f"{clip}.labels.jsonl", n, seeds)
        t3 = sam_clip(a.sam3 / f"{clip}.labels.jsonl", n, seeds)
        out["clips"][clip] = {"n": n, "t1": t1, "t2": t2, "t3": t3}
        print(clip, n, "t1", sum(x is not None for x in t1["x"]), "t2", sum(x is not None for x in (t2 or {"x": []})["x"]), "t3", sum(x is not None for x in (t3 or {"x": []})["x"]), flush=True)
    with gzip.open(a.out, "wt") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
