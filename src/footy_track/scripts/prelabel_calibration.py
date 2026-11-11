"""Pre-label pitch calibration frames (classical candidates + an identifier).

    # live Claude pilot (needs ANTHROPIC_API_KEY and `uv sync --extra llm`)
    python -m footy_track.scripts.prelabel_calibration run --identifier claude \
        --clips arsenal_mancity_seg010 --every 25 --limit 40

    # no-cost debugging from existing human GT
    python -m footy_track.scripts.prelabel_calibration run --identifier mock --frames 0,349,417 \
        --clips arsenal_mancity_seg010

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
    if args.identifier == "claude":
        ident = ClaudeIdentifier(model=args.model, effort=args.effort, fallbacks=not args.no_fallbacks)
    else:
        ident = MockIdentifier(gt, corrupt=args.mock_corrupt)
    cfg = RunConfig(
        gt_dir=gt, clips_dir=clips, clips=args.clips,
        frames=[int(x) for x in args.frames.split(",")] if args.frames else None,
        every=args.every, skip_gt=args.skip_gt, limit=args.limit,
        render_dir=Path(args.render_dir) if args.render_dir else None,
        verify=args.verify, run_id=args.run_id,
    )
    s = run_prelabel(cfg, ident)
    print(f"run_id={s.run_id} frames={s.n_frames} proposed={s.n_proposed} failed={s.n_failed}")
    if s.usage_total:
        print(f"usage: {json.dumps(s.usage_total)}")
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
    r.add_argument("--identifier", choices=["claude", "mock"], default="claude")
    r.add_argument("--clips", nargs="+", required=True, help="clip names or aliases")
    r.add_argument("--frames", help="comma-separated frame indices (default: keyframes)")
    r.add_argument("--every", type=int, default=25, help="keyframe stride (default 25)")
    r.add_argument("--skip-gt", action="store_true", help="skip frames that already have human GT")
    r.add_argument("--limit", type=int, help="max frames processed in total")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--effort", choices=EFFORTS, default=DEFAULT_EFFORT)
    r.add_argument("--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    r.add_argument("--verify", action="store_true", help="second Claude call judging the overlay (extra cost)")
    r.add_argument("--mock-corrupt", type=int, default=0, help="mock only: swap k label pairs")
    r.add_argument("--render-dir", help="also save candidate/overlay/diagram images here")
    r.add_argument("--run-id")
    r.set_defaults(fn=_cmd_run)
    e = sub.add_parser("eval", help="proposals vs human GT")
    e.add_argument("--run", help="run id (default: all runs)")
    e.add_argument("--success-px", type=float, default=ev.SUCCESS_PX)
    e.add_argument("--out", help="write JSON here")
    e.set_defaults(fn=_cmd_eval)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
