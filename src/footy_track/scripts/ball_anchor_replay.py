"""Replay evaluation of the anchor-iteration loop on labelled TRAIN clips (design ``ball_anchor_iteration.md`` task 13).

    python -m footy_track.scripts.ball_anchor_replay [--marks-dir D ...] [--cand-dir D ...] [--min-gt 10]
        [--clips a,b] [--policy design|requeue|both] [--order design|margin] [--window-frames 250] [--tol-px 20] [--error-rate 0]
        [--synthetic N] [--json out.json]

Oracle = human ball GT sidecars (default ``ball_gt_marks`` + ``ball_gt_marks_ball_train_v1``), candidates = RT-DETR
jsonl (``ball_candidates`` then ``machine_labels``). Clips without candidates are listed as missing, never guessed.
``--synthetic N`` replays N synthetic noisy clips instead (harness proof, NOT a measurement). Read-only; CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from footy_track import splits
from footy_track.ball_mining import anchor_replay as R  # noqa: N812

ROOT = Path(os.environ.get("FOOTY_DATA") or "/mnt/storage/footy_data")
MARKS = ["ball_gt_marks", "ball_gt_marks_ball_train_v1"]
CANDS = ["ball_candidates", "machine_labels"]


def run(a: argparse.Namespace) -> dict:
    policies = R.POLICIES if a.policy == "both" else (a.policy,)
    out: dict = {"policies": {}, "missing": [], "skipped_non_train": [], "synthetic": bool(a.synthetic)}
    jobs: list[tuple[str, dict, object, bool]] = []
    if a.synthetic:
        for i in range(a.synthetic):
            gt, cs = R.synthetic_clip(f"synthetic_{i:02d}", seed=i)
            jobs.append((f"synthetic_{i:02d}", gt, cs, True))
    else:
        gt_all, info = R.load_gt([ROOT / m for m in a.marks_dir or MARKS], tol_px=a.tol_px)
        out["gt_info"] = info
        want = set(a.clips.split(",")) if a.clips else None
        for clip, gt in sorted(gt_all.items()):
            if want and clip not in want or len(gt) < a.min_gt:
                continue
            if splits.split_of_name(clip) != "train":
                out["skipped_non_train"].append(clip)
                continue
            cs = R.load_machine_candidates(clip, [ROOT / d for d in a.cand_dir or CANDS])
            if cs is None:
                out["missing"].append({"clip": clip, "gt_frames": len(gt), "missing": "detector candidates (no jsonl in cand dirs)"})
                continue
            jobs.append((clip, gt, cs, False))
    for pol in policies:
        cfg = R.ReplayConfig(a.tol_px, a.target, pol, a.order, a.error_rate, a.hop_cap, a.window_frames, bridge_stride=a.bridge_stride)
        rows = [R.simulate_clip(c, gt, cs, cfg, synthetic=syn).row() for c, gt, cs, syn in jobs]
        out["policies"][pol] = {"rows": rows, "summary": R.summarize(rows)}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--marks-dir", action="append")
    p.add_argument("--cand-dir", action="append")
    p.add_argument("--min-gt", type=int, default=10)
    p.add_argument("--clips")
    p.add_argument("--policy", choices=[*R.POLICIES, "both"], default="both")
    p.add_argument("--order", choices=R.ORDERS, default="design")
    p.add_argument("--window-frames", type=int, default=R.A.WINDOW_FRAMES)
    p.add_argument("--bridge-stride", type=int, default=5, help="extra tracker frames every N frames (0 = GT frames only)")
    p.add_argument("--hop-cap", type=int, default=R.A.HOP_CAP)
    p.add_argument("--tol-px", type=float, default=R.TOL_PX)
    p.add_argument("--target", type=float, default=R.TARGET)
    p.add_argument("--error-rate", type=float, default=0.0)
    p.add_argument("--synthetic", type=int, default=0)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    out = run(a)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    label = "SYNTHETIC (harness proof, not a measurement)" if a.synthetic else "REAL train clips"
    for pol, d in out["policies"].items():
        print(f"\n## {label}: policy={pol} order={a.order} window={a.window_frames} err={a.error_rate}")
        print(R.format_table(d["rows"]))
        print(d["summary"])
    if out["missing"]:
        print("\nmissing candidates:", out["missing"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
