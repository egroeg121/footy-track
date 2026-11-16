"""Round 3 step 2 (mini, one thread-capped process): harness B1 over RT-DETR machine labels of one target match, S1 anchors and the
SAM2 track spec. Generalises experiments/pseudo_labels_v1/pseudo_prep.py (Man City only) to a labels directory.

    OMP_NUM_THREADS=1 python r3_prep.py --match bournemouth --labels-dir <dir with <clip>.jsonl [+ <clip>.meta.json]> --out <dir>

A fake harness root `<out>/root` holds machine_labels (symlinks + manifest) and an EMPTY ball_gt_marks, so no human label of any match
(incl. Nottingham Forest) can be read. Frame indices are those of the labels (for 50 fps clips: original // stride).
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HARNESS = Path("/home/george/footy-track/.claude/worktrees/agent-a8507f51f3d63f415")
SAM2 = Path("/home/george/footy-track/.claude/worktrees/agent-a61a56024c56ea1ff/experiments/sam2_bidir_ball")
sys.path.insert(0, str(HARNESS / "src"))
sys.path.insert(0, str(SAM2))

from footy_track.ball_offline import baselines as bl  # noqa: E402
from footy_track.ball_offline.data import load_clip, machine_stem_map  # noqa: E402
from footy_track.ball_offline.solver import SolverParams, calibrated_p  # noqa: E402
from machine_common import REACH, thin  # noqa: E402
from pipeline_eval import CAMERA_ON  # noqa: E402

THR, SPACING = 0.8, 10


def make_root(labels_dir: Path, root: Path) -> list[str]:
    ml = root / "machine_labels"
    ml.mkdir(parents=True, exist_ok=True)
    (root / "ball_gt_marks").mkdir(exist_ok=True)  # deliberately empty
    clips = {}
    man_src = labels_dir / "_manifest.json"
    src_manifest = json.loads(man_src.read_text())["clips"] if man_src.exists() else {}
    for p in sorted(labels_dir.glob("*.jsonl")):
        link = ml / p.name
        if not link.exists():
            link.symlink_to(p.resolve())
        meta = labels_dir / f"{p.stem}.meta.json"
        n = json.loads(meta.read_text())["frames"] if meta.exists() else src_manifest[p.stem]["frames"]
        clips[p.stem] = {"frames": n}
    (ml / "_manifest.json").write_text(json.dumps({"clips": clips}))
    return sorted(clips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", default="", help="comma list (default all in the labels dir)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = a.out / "root"
    clips = make_root(a.labels_dir, root)
    if a.clips:
        clips = [c for c in clips if c in set(a.clips.split(","))]
    stems = machine_stem_map(root)
    p1 = SolverParams(**CAMERA_ON)
    pool_b1, spec = {}, {"reach": REACH, "clips": {}}
    t0 = time.time()
    for i, cid in enumerate(clips, 1):
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
                        "cuts": [int(t) for t in np.where(clip.cam_cut)[0]], "anchors": A, "gt": {}, "no_ball": []}
        spec["clips"][cid] = {"video": f"{cid}.mp4", "n_frames": clip.n_frames, "tracks": list(need.values()), "full": False}
        print(f"[{i}/{len(clips)}] {cid:34s} frames={clip.n_frames:5d} nodes={len(nj):5d} b1={len(pred):5d} gap={n_gap:4d} anchors={len(A):4d} "
              f"tracks={len(need):4d} steps={sum(v['reach'] for v in need.values()):5d} ({time.time() - t0:.0f}s)", flush=True)
    (a.out / "pool_b1.json").write_text(json.dumps(pool_b1))
    (a.out / "spec.json").write_text(json.dumps(spec))
    tot = sum(v["reach"] for c in spec["clips"].values() for v in c["tracks"])
    print(f"{a.match}: {len(clips)} clips, total SAM2 steps {tot} (~{tot * 0.18 / 60:.0f} min at 0.18 s/frame)")


if __name__ == "__main__":
    main()
