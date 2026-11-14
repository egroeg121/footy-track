"""Pre-label pitch calibration frames (classical candidates + an identifier).

    # Claude via the Claude Code CLI (subscription login, no API key)
    python -m footy_track.scripts.prelabel_calibration run --identifier claude-cli \
        --clips arsenal_mancity_seg010 --every 25 --limit 40

    # live Claude pilot (needs ANTHROPIC_API_KEY and `uv sync --extra llm`)
    python -m footy_track.scripts.prelabel_calibration run --identifier claude \
        --clips arsenal_mancity_seg010 --every 25 --limit 40

    # no-cost debugging from existing human GT (frames with GT are normally skipped: --no-skip-gt)
    python -m footy_track.scripts.prelabel_calibration run --identifier mock --frames 0,349,417 \
        --clips arsenal_mancity_seg010 --no-skip-gt --mock-sigma 20

    # how precise must Claude be? noise sweep over the human-GT frames (on a COPY of the GT dir)
    python -m footy_track.scripts.prelabel_calibration sweep --clip arsenal_mancity_seg010 \
        --frames 0,349,417 --sigmas 0,10,20,40 --seeds 8

    # compare a run with human GT
    python -m footy_track.scripts.prelabel_calibration eval --run <run_id> --out eval.json

Proposals land in ``<gt dir>/calibration_proposals/`` (review them at
``/calibration``, "Review proposals"). The GT dir is ``--gt-dir`` or
``FOOTY_GT_MARKS_DIR``; clips come from ``--clips-dir`` or ``FOOTY_CLIPS_DIR``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from footy_track.calibration.prelabel import evaluate as ev
from footy_track.calibration.prelabel.identifier import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    ClaudeCLIIdentifier,
    ClaudeIdentifier,
    MockIdentifier,
)
from footy_track.calibration.prelabel.pipeline import RunConfig, run_prelabel

EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _dirs(args) -> tuple[Path, Path]:
    gt = args.gt_dir or os.environ.get("FOOTY_GT_MARKS_DIR")
    if not gt:
        sys.exit("set --gt-dir or FOOTY_GT_MARKS_DIR")
    clips = args.clips_dir or os.environ.get("FOOTY_CLIPS_DIR") or Path(__file__).parents[3] / "eval_data" / "clips"
    return Path(gt), Path(clips)


def _cmd_run(args) -> int:
    gt, clips = _dirs(args)
    if args.identifier == "claude-cli":
        model = "opus" if args.model is None else args.model
        ident = ClaudeCLIIdentifier(model=model, effort=args.effort, timeout=args.timeout)
    elif args.identifier == "claude":
        if args.model is None:
            args.model = DEFAULT_MODEL
        ident = ClaudeIdentifier(model=args.model, effort=args.effort, fallbacks=not args.no_fallbacks)
    else:
        ident = MockIdentifier(
            gt, corrupt=args.mock_corrupt, sigma_px=args.mock_sigma, drop_prob=args.mock_drop,
            mislabel_prob=args.mock_mislabel, lines=args.mock_lines,
        )
    cfg = RunConfig(
        gt_dir=gt, clips_dir=clips, clips=args.clips,
        frames=[int(x) for x in args.frames.split(",")] if args.frames else None,
        every=args.every, skip_gt=args.skip_gt, limit=args.limit,
        render_dir=Path(args.render_dir) if args.render_dir else None,
        verify=args.verify, run_id=args.run_id, refine=not args.no_refine,
        k1=args.k1, fit_k1=args.fit_k1,
    )
    s = run_prelabel(cfg, ident)
    print(f"run_id={s.run_id} frames={s.n_frames} proposed={s.n_proposed} failed={s.n_failed}")
    if s.usage_total:
        print(f"usage: {json.dumps(s.usage_total)}")
    return 0


def _cmd_rerefine(args) -> int:
    from footy_track.calibration.prelabel.rerefine import rerefine  # noqa: PLC0415

    gt, clips = _dirs(args)
    dirs = [clips] + [Path(d) for d in (args.extra_clips_dir or [])]
    s = rerefine(gt, dirs, clips=args.clips, runs=args.runs, fit_k1=args.fit_k1, dry_run=args.dry_run,
                 workers=args.workers)
    print(f"rerefined={s.rerefined} now_failed={s.now_failed} kept_previous={s.kept_previous} unchanged={s.unchanged} errors={len(s.errors)}"
          + (" (dry run: nothing written)" if args.dry_run else ""))
    for e in s.errors:
        print("error:", e)
    return 0


def _cmd_sweep(args) -> int:
    from footy_track.calibration.prelabel import sweep as sw  # noqa: PLC0415

    gt, clips = _dirs(args)
    res = sw.noise_sweep(
        gt, clips, args.clip, [int(x) for x in args.frames.split(",")],
        sigmas=[float(x) for x in args.sigmas.split(",")], n_seeds=args.seeds, drop_prob=args.drop,
        mislabel_prob=args.mislabel, lines=args.lines, workers=args.workers,
    )
    print(sw.format_sweep(res))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")
    return 0


def _cmd_eval(args) -> int:
    gt, _ = _dirs(args)
    res = ev.evaluate(gt, args.run, args.success_px)
    print(ev.format_table(res))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir")
    ap.add_argument("--clips-dir")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate proposals")
    r.add_argument("--identifier", choices=["claude", "claude-cli", "mock"], default="claude")
    r.add_argument("--clips", nargs="+", required=True, help="clip names or aliases")
    r.add_argument("--frames", help="comma-separated frame indices (default: keyframes)")
    r.add_argument("--every", type=int, default=25, help="keyframe stride (default 25)")
    r.add_argument("--skip-gt", action=argparse.BooleanOptionalAction, default=True,
                   help="skip frames that already have human GT (default on; --no-skip-gt to include them)")
    r.add_argument("--no-refine", action="store_true", help="coarse fit only (no line refinement)")
    r.add_argument("--k1", type=float, default=0.0,
                   help="lens distortion (division model) held fixed, e.g. a per-clip value; 0 = off (default)")
    r.add_argument("--fit-k1", action="store_true", help="estimate k1 per frame jointly with H (off by default)")
    r.add_argument("--limit", type=int, help="max frames processed in total")
    r.add_argument("--model", help="default: claude-opus-5-5 (claude) / opus alias (claude-cli)")
    r.add_argument("--timeout", type=float, default=300.0, help="claude-cli: seconds per call")
    r.add_argument("--effort", choices=EFFORTS, default=DEFAULT_EFFORT)
    r.add_argument("--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    r.add_argument("--verify", action="store_true", help="second Claude call judging the overlay (extra cost)")
    r.add_argument("--mock-corrupt", type=int, default=0, help="mock only: swap k label pairs")
    r.add_argument("--mock-sigma", type=float, default=0.0, help="mock only: label noise std in px @1920 wide")
    r.add_argument("--mock-drop", type=float, default=0.0, help="mock only: probability of dropping a point")
    r.add_argument("--mock-mislabel", type=float, default=0.0, help="mock only: probability of a wrong id")
    r.add_argument("--mock-lines", action="store_true", help="mock only: also emit noisy line labels")
    r.add_argument("--render-dir", help="also save candidate/overlay/diagram images here")
    r.add_argument("--run-id")
    r.set_defaults(fn=_cmd_run)
    rr = sub.add_parser("rerefine", help="re-run detection + refinement on 'proposed' items (no Claude calls)")
    rr.add_argument("--clips", nargs="+")
    rr.add_argument("--runs", nargs="+")
    rr.add_argument("--fit-k1", action=argparse.BooleanOptionalAction, default=True)
    rr.add_argument("--extra-clips-dir", nargs="+", help="more directories to search for videos")
    rr.add_argument("--dry-run", action="store_true")
    rr.add_argument("--workers", type=int, default=2)
    rr.set_defaults(fn=_cmd_rerefine)
    sw = sub.add_parser("sweep", help="noise sweep: success vs label noise, before/after line refinement")
    sw.add_argument("--clip", required=True)
    sw.add_argument("--frames", required=True, help="comma-separated GT frame indices")
    sw.add_argument("--sigmas", default="0,10,20,40", help="comma-separated noise std in px @1920 wide")
    sw.add_argument("--seeds", type=int, default=8)
    sw.add_argument("--drop", type=float, default=0.0)
    sw.add_argument("--mislabel", type=float, default=0.0)
    sw.add_argument("--lines", action="store_true", help="also give noisy line labels")
    sw.add_argument("--workers", type=int, default=1)
    sw.add_argument("--out", help="write JSON here")
    sw.set_defaults(fn=_cmd_sweep)
    e = sub.add_parser("eval", help="proposals vs human GT")
    e.add_argument("--run", help="run id (default: all runs)")
    e.add_argument("--success-px", type=float, default=ev.SUCCESS_PX)
    e.add_argument("--out", help="write JSON here")
    e.set_defaults(fn=_cmd_eval)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
