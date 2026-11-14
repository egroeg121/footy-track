"""Pool selection + B1 + S1 track spec for the pseudo-label set (CPU, mini; one process).

1. Clip pool = Man City broadcast segments with machine_labels, minus
   * the 9 trajectory test clips (docs/experiments/offline_ball_baseline.json split.test),
   * every clip containing a v11 val / test / heldout image (multiframe_v1/k2/meta.jsonl),
   * arsenal_example and anything arsenal_norwich (none exist in machine_labels; listed for the record).
2. B1 (harness, camera on, dev-tuned params) over each pool clip; nodes + detector boxes.
3. S1 anchors (thr 0.8, spacing 10, border nodes kept) and the SAM2 tracks needed to cover every
   B1 gap frame (forward from the previous anchor, backward from the next, <= 50 frames each).

Writes: exclusions.json, pool_b1.json (per clip B1/nodes/cuts), spec_pool.json (tracks).

usage: OMP_NUM_THREADS=1 python pseudo_prep.py --out <dir>
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "sam2_bidir_ball"))

from footy_track.ball_offline import baselines as bl  # noqa: E402
from footy_track.ball_offline.data import data_root, load_clip, machine_stem_map  # noqa: E402
from footy_track.ball_offline.solver import SolverParams, calibrated_p  # noqa: E402
from machine_common import REACH, thin  # noqa: E402

THR, SPACING = 0.8, 10
META = Path("/mnt/storage/footy_data/multiframe_v1/k2/meta.jsonl")


def exclusions(all_clips: list[str], results: dict) -> dict:
    rows = [json.loads(line) for line in META.read_text().splitlines() if line.strip()]
    by_split: dict[str, set[str]] = {}
    for r in rows:
        by_split.setdefault(r["split"], set()).add(r["clip_id"])
    v11_eval = by_split["valid"] | by_split["test"] | by_split["heldout"]
    traj_test = set(results["split"]["test"])
    named = {c for c in all_clips if c == "arsenal_example" or "norwich" in c}
    pool = sorted(c for c in all_clips if c not in v11_eval and c not in traj_test and c not in named)
    return {
        "machine_label_clips": len(all_clips),
        "v11_eval_clips_total": len(v11_eval),
        "v11_eval_clip_counts_by_split": {s: len(by_split[s]) for s in ("valid", "test", "heldout")},
        "v11_train_clip_count": len(by_split["train"]),
        "trajectory_test_clips": sorted(traj_test),
        "excluded_v11_val_test_heldout": sorted(c for c in all_clips if c in v11_eval),
        "excluded_trajectory_test": sorted(c for c in all_clips if c in traj_test),
        "excluded_named": sorted(named),
        "n_excluded_v11": sum(1 for c in all_clips if c in v11_eval),
        "n_excluded_traj_only": sum(1 for c in all_clips if c in traj_test and c not in v11_eval),
        "n_excluded_total": len(all_clips) - len(pool),
        "pool": pool,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=str(ROOT / "docs/experiments/offline_ball_baseline.json"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads(Path(a.results).read_text())
    p1 = SolverParams(**res["B1"]["tuned_params"]["camera_on"])
    root = data_root()
    stems = machine_stem_map(root)
    ex = exclusions(sorted(stems), res)
    manifest = json.loads((root / "machine_labels" / "_manifest.json").read_text())["clips"]
    ex["pool_frames"] = sum(manifest[stems[c].stem]["frames"] for c in ex["pool"])
    (out / "exclusions.json").write_text(json.dumps(ex, indent=1))
    print(f"machine_labels clips {ex['machine_label_clips']}; excluded: v11 val/test/heldout {ex['n_excluded_v11']}, "
          f"trajectory test only {ex['n_excluded_traj_only']}, total {ex['n_excluded_total']}; pool {len(ex['pool'])} clips, {ex['pool_frames']} frames", flush=True)

    pool_b1, spec = {}, {"reach": REACH, "clips": {}}
    t0 = time.time()
    for i, cid in enumerate(ex["pool"], 1):
        clip = load_clip(cid, root, with_camera=True)
        pred, nodes = bl.b1_solver(clip, p1)
        nj = []
        for n in nodes:
            xy, _ = clip.cands(n.t)
            k = int(np.argmin(np.hypot(xy[:, 0] - n.xy[0], xy[:, 1] - n.xy[1]))) if len(xy) else None
            s, e = clip.ptr[n.t], clip.ptr[n.t + 1]
            wh = clip.wh[s:e][k] if k is not None else np.array([16.0, 16.0])
            nj.append({"t": n.t, "xy": [round(n.xy[0], 2), round(n.xy[1], 2)], "conf": round(n.conf, 4), "p": round(float(calibrated_p(np.array([n.conf]))[0]), 4),
                       "linked": n.linked, "wh": [round(float(wh[0]), 1), round(float(wh[1]), 1)],
                       "box": [n.xy[0] - wh[0] / 2, n.xy[1] - wh[1] / 2, n.xy[0] + wh[0] / 2, n.xy[1] + wh[1] / 2]})
        pf = {int(t) for t in pred}
        anchors = thin(nj, pf, THR, SPACING)
        A = [x["t"] for x in anchors]
        box = {x["t"]: x["box"] for x in anchors}
        need: dict[tuple[int, str], dict] = {}
        n_gap = 0
        for f in range(clip.n_frames):
            if f in pf:
                continue
            n_gap += 1
            j = bisect.bisect_left(A, f)
            if j > 0 and f - A[j - 1] <= REACH:
                k = (A[j - 1], "fwd")
                need[k] = {"t": A[j - 1], "dir": "fwd", "reach": max(f - A[j - 1], need.get(k, {}).get("reach", 0)), "box": box[A[j - 1]]}
            if j < len(A) and A[j] - f <= REACH:
                k = (A[j], "bwd")
                need[k] = {"t": A[j], "dir": "bwd", "reach": max(A[j] - f, need.get(k, {}).get("reach", 0)), "box": box[A[j]]}
        pool_b1[cid] = {"n_frames": clip.n_frames, "stem": stems[cid].stem, "nodes": nj,
                        "b1": {str(t): [round(v[0], 2), round(v[1], 2)] for t, v in pred.items()},
                        "cuts": [int(t) for t in np.where(clip.cam_cut)[0]], "anchors": A,
                        "gt": {str(t): list(g.xy) for t, g in clip.gt.items()}, "no_ball": sorted(clip.no_ball)}
        spec["clips"][cid] = {"video": str(root / "clips" / f"{stems[cid].stem}.mp4"), "n_frames": clip.n_frames,
                              "tracks": list(need.values()), "full": False}
        print(f"[{i}/{len(ex['pool'])}] {cid:26s} frames={clip.n_frames:5d} nodes={len(nj):5d} b1={len(pred):5d} gap={n_gap:4d} "
              f"anchors={len(A):4d} tracks={len(need):4d} steps={sum(v['reach'] for v in need.values()):5d} ({time.time()-t0:.0f}s)", flush=True)
    (out / "pool_b1.json").write_text(json.dumps(pool_b1))
    (out / "spec_pool.json").write_text(json.dumps(spec))
    tot = sum(v["reach"] for c in spec["clips"].values() for v in c["tracks"])
    print(f"total SAM2 steps {tot} (~{tot * 0.15 / 60:.0f} min at 0.15 s/frame uncontended)")


if __name__ == "__main__":
    main()
