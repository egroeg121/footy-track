"""``python -m footy_track.ball_offline`` : freeze GT, run baselines, score any tracker.

    snapshot   copy the current GT of an eval set into a dated, hashed directory
    predict    run a reference tracker (b0, viterbi) on a snapshot, write tracker jsonl
    score      score a tracker's per-frame ball jsonl against a snapshot
    verdicts   candidate-level cross-check against Ball Check verdicts
    ceiling    detector upper bound on the scored frames

Every JSON/markdown report records the snapshot hash it was scored against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from footy_track.ball_offline import baselines as bl
from footy_track.ball_offline import evaluate as E
from footy_track.ball_offline import score as S
from footy_track.ball_offline.snapshot import (
    Snapshot,
    create_snapshot,
    resolve_snapshot,
    snapshots_root,
)
from footy_track.ball_offline.sources import EvalSetSpec, eval_set

TRACKER_FORMAT = (
    'tracker jsonl: one row per frame with a ball: {"clip": ..., "frame_index": n, "x": px, "y": px} '
    "(pixels at 1920x1080; see footy_track.ball_offline.score)"
)


def spec_for(snap: Snapshot, args) -> EvalSetSpec:
    """The snapshot's eval set, with candidate path overridable from the CLI."""
    m = snap.manifest
    return eval_set(
        m["eval_set"],
        gt_dir=getattr(args, "gt_dir", None) or m["gt_dir"],
        candidates_dir=getattr(args, "candidates_dir", None) or m["candidates_dir"],
        manifest_dir=getattr(args, "manifest_dir", None),
    )


def _add_set_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--candidates-dir", help="override the candidate directory")
    p.add_argument(
        "--manifest-dir",
        help="detector run manifest dir (frame counts), if not the candidate dir",
    )


def cmd_snapshot(args) -> int:
    spec = eval_set(
        args.eval_set,
        gt_dir=args.gt_dir,
        candidates_dir=args.candidates_dir,
        manifest_dir=args.manifest_dir,
    )
    snap = create_snapshot(spec, Path(args.out_root) if args.out_root else None)
    m = snap.manifest
    print(f"snapshot  {snap.path}")
    print(f"hash      {snap.hash}")
    print(
        f"eval set  {m['eval_set']}: {m['n_clips']} clips, {m['n_test_clips']} test, "
        f"{m['n_test_clean_ball']} clean test ball + {m['n_test_no_ball']} no_ball, "
        f"{m['n_dev_ball']} dev ball (detector-contaminated)"
    )
    if m["skipped_no_candidates"]:
        print(f"skipped (GT but no candidates): {m['skipped_no_candidates']}")
    return 0


def run_baseline(
    evs: list[E.ClipEval], method: str
) -> dict[str, dict[int, tuple[float, float]]]:
    """``method``: ``b0`` (top-1 per frame, tau 0.4) | ``viterbi`` (main's ViterbiBallTracker)."""
    fn = bl.METHODS[method]
    return {ev.clip.clip_id: fn(ev.clip) for ev in evs}


def cmd_predict(args) -> int:
    snap = resolve_snapshot(args.snapshot)
    spec = spec_for(snap, args)
    roles = set(args.roles.split(","))
    clips = [c for c, v in snap.clips.items() if v.role in roles]
    evs = E.build_sets(snap, spec, clips=clips)
    if args.include_verdict_clips:
        from footy_track.ball_offline.data import load_clip, machine_stem_map  # noqa: PLC0415
        from footy_track.ball_offline.verdicts import load_verdicts  # noqa: PLC0415

        have = machine_stem_map(spec.candidates_dir)
        seen = {ev.clip.clip_id for ev in evs}
        for cid in sorted(load_verdicts(spec.verdicts_dir)[0]):
            if cid in have and cid not in seen:
                clip = load_clip(cid, spec, gt=({}, set(), [], 0))
                evs.append(E.ClipEval(clip, set(), set(), set(), "verdict"))
    preds = run_baseline(evs, args.method)
    S.write_predictions(args.out, preds)
    n = sum(len(p) for p in preds.values())
    print(
        f"wrote {args.out}: {n} points over {len(preds)} clips ({args.method}, snapshot {snap.label()})"
    )
    return 0


def cmd_ceiling(args) -> int:
    """Detector upper bound: scored GT frames with any candidate >= conf within R px."""
    snap = resolve_snapshot(args.snapshot)
    spec = spec_for(snap, args)
    print(
        f"ceiling on snapshot {snap.label()}: any candidate >= {args.min_conf} within {args.radius:g} px"
    )
    for role in ("test", "dev"):
        evs = [e for e in E.build_sets(snap, spec) if e.role == role]
        hit, n, _ = E.theoretical_ceiling(evs, args.radius, args.min_conf)
        print(f"  {role}: {hit} / {n} = {hit / n if n else float('nan'):.3f}")
    return 0


def _report_md(
    snap: Snapshot,
    lp: S.LoadedPreds,
    args,
    boot: dict | None,
    base: S.LoadedPreds | None,
) -> tuple[str, dict]:
    res: dict = {
        "snapshot": {
            "hash": snap.hash,
            "short": snap.short,
            "eval_set": snap.eval_set,
            "path": str(snap.path),
            "created_utc": snap.manifest["created_utc"],
        },
        "tracker": {
            "path": lp.path,
            "sha256": lp.sha256,
            "rows": lp.n_rows,
            "rows_without_position": lp.n_dropped_nan,
            "duplicate_frame_rows": lp.n_duplicate_frames,
        },
        "roles": {},
    }
    L = [
        f"Measured on the offline-ball harness (snapshot {snap.label()}, "
        f"{snap.manifest['created_utc'][:10]}); tracker `{Path(lp.path).name}` sha256 {lp.sha256[:12]}.\n"
    ]
    for role in args.roles.split(","):
        per, pooled, warnings = S.score_snapshot(snap, lp, role)
        res["roles"][role] = {"pooled": pooled, "per_clip": per, "warnings": warnings}
        title = (
            "test (clean, detector-uncontaminated)"
            if role == "test"
            else "dev (detector-contaminated; optimistic)"
        )
        L += [f"### {title}\n", S.TABLE_HEAD, S.summary_row(Path(lp.path).stem, pooled)]
        if base is not None:
            _, bp, _ = S.score_snapshot(snap, base, role)
            L.append(S.summary_row(Path(base.path).stem + " (baseline)", bp))
        L.append("")
        for w in warnings:
            L.append(f"- WARNING: {w}")
        if args.per_clip:
            L += ["", "| clip | GT | recall@10 | precision@10 |", "|---|---|---|---|"]
            for cid, s in per.items():
                L.append(
                    f"| {cid} | {s['n_gt_ball']} | {S.fmt(s['recall@10'], 2)} | {S.fmt(s['precision@10'], 2)} |"
                )
            L.append("")
    if boot is not None:
        res["paired_bootstrap_vs_baseline"] = boot
        rd, pdf = boot["recall_diff"], boot["precision_diff"]
        L += [
            f"Paired bootstrap, tracker minus baseline (test, {boot['n_boot']} resamples of "
            f"{boot['n_blocks']} {'frames' if boot['block'] == 1 else f'blocks of {boot["block"]} frames'}):",
            f"- recall@10 diff {rd['point']:+.3f} [95% CI {rd['lo']:+.3f}, {rd['hi']:+.3f}], P(>0)={rd['p_gt_0']:.2f}",
            f"- precision@10 diff {pdf['point']:+.3f} [95% CI {pdf['lo']:+.3f}, {pdf['hi']:+.3f}], P(>0)={pdf['p_gt_0']:.2f}",
            f"- frames hit only by tracker: {boot['a_hit_b_miss']}, only by baseline: {boot['b_hit_a_miss']}",
        ]
    return "\n".join(L) + "\n", res


def cmd_score(args) -> int:
    snap = resolve_snapshot(args.snapshot)
    lp = S.load_predictions(args.pred, args.units)
    base = S.load_predictions(args.baseline, args.units) if args.baseline else None
    boot = None
    if base is not None and "test" in args.roles.split(","):
        boot = S.paired_bootstrap(
            S.frame_units(snap, lp, "test"),
            S.frame_units(snap, base, "test"),
            n_boot=args.n_boot,
            block=args.block,
            seed=args.seed,
        )
    md, res = _report_md(snap, lp, args, boot, base)
    print(md)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=1, default=float))
    return 0


def cmd_verdicts(args) -> int:
    from footy_track.ball_offline.verdicts import run_verdicts  # noqa: PLC0415

    return run_verdicts(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m footy_track.ball_offline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("snapshot", help="freeze the current GT of an eval set")
    p.add_argument(
        "--eval-set",
        default="mancity",
        help="mancity | norwich | <name> (then --gt-dir/--candidates-dir)",
    )
    p.add_argument(
        "--gt-dir", help="labeller marks dir (the labeller's FOOTY_GT_MARKS_DIR)"
    )
    p.add_argument(
        "--out-root", help=f"default {snapshots_root()} (env FOOTY_BALL_SNAPSHOTS)"
    )
    _add_set_args(p)
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser(
        "predict", help="run a baseline on a snapshot, write tracker jsonl"
    )
    p.add_argument(
        "--snapshot", required=True, help="snapshot dir, dir name or hash prefix"
    )
    p.add_argument("--method", choices=sorted(bl.METHODS), default="viterbi")
    p.add_argument("--roles", default="test,dev")
    p.add_argument(
        "--include-verdict-clips",
        action="store_true",
        help="also run on clips that only have Ball Check verdicts (for `verdicts`)",
    )
    p.add_argument("--out", required=True)
    _add_set_args(p)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser(
        "score", help="score a tracker jsonl against a snapshot", epilog=TRACKER_FORMAT
    )
    p.add_argument("--snapshot", required=True)
    p.add_argument("--pred", required=True, help="tracker jsonl")
    p.add_argument("--units", choices=["px", "norm"], default="px")
    p.add_argument(
        "--baseline",
        help="second tracker jsonl: adds a paired bootstrap (pred - baseline)",
    )
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument(
        "--block", type=int, default=1, help="bootstrap block length in frames"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--roles",
        default="test",
        help="test (clean) and/or dev (contaminated), comma separated",
    )
    p.add_argument("--per-clip", action="store_true")
    p.add_argument("--out-json")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser(
        "verdicts", help="cross-check trackers against Ball Check verdicts"
    )
    p.add_argument("--snapshot", required=True)
    p.add_argument(
        "--pred", action="append", required=True, help="tracker jsonl (repeatable)"
    )
    p.add_argument("--out-json")
    p.set_defaults(fn=cmd_verdicts)

    p = sub.add_parser("ceiling", help="detector upper bound on a snapshot")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--min-conf", type=float, default=0.1)
    _add_set_args(p)
    p.set_defaults(fn=cmd_ceiling)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
