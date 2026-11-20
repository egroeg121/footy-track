"""Detector suggestions for a ``ball_place`` labelling job (LAB-1479), no blind control.

    python -m footy_track.scripts.job_suggest suggest --job j-xxxx [--device cpu] [--weights W] [--root R]
    python -m footy_track.scripts.job_suggest report  --job j-xxxx [--radius 10] [--root R]

``suggest`` runs the production RT-DETR (threads capped by $BALL_POOL_THREADS / $OMP_NUM_THREADS, default 4; run it under
``nice``) over every frame of the job's items and writes ``<FD>/job_suggestions/<job_id>.suggest.<sha8>.json`` only.
Heldout clips only (eval-only inference). ``report`` prints, per stratum, coverage, share of accepted suggestions and
agreement with RT-DETR on the job's labels. See ``footy_track/ball_eval/job_suggest.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from footy_track.ball_eval import job_suggest as J  # noqa: N812
from footy_track.ball_eval import panel as P  # noqa: N812


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None, help="data root (default $FOOTY_DATA)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("suggest")
    s.add_argument("--job", required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--weights", default=None)
    s.add_argument("--batch", type=int, default=4)
    r = sub.add_parser("report")
    r.add_argument("--job", required=True)
    r.add_argument("--radius", type=float, default=10.0)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "suggest":
            out = J.suggest_job(a.job, root=a.root, weights=Path(a.weights) if a.weights else None, device=a.device,
                                batch=a.batch, log=lambda m: print(m, file=sys.stderr, flush=True))
            sf = J.find_suggest_file(a.job, a.root)
            if sf is not None:
                from footy_track.labeller import jobs  # noqa: PLC0415

                out["coverage"] = J.coverage(jobs.read_job(a.job), json.loads(sf.read_text()))
        else:
            out = J.report(a.job, root=a.root, r_px=a.radius)
    except (P.PanelError, OSError, ValueError) as e:
        print(json.dumps({"job_id": a.job, "status": "error", "error": str(e)}))
        return 1
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
