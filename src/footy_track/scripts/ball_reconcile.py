"""Ball reconciler CLI: ``python -m footy_track.scripts.ball_reconcile --db store.duckdb [--json out.json]``.

Read-only: diffs selected_frame vs labelling-job cards/verdicts vs frame_label_state per job family and exits nonzero on any
violation (see ``footy_track.ball_mining.reconcile``). ``--apply`` is reserved (decision: report-only) and refuses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from footy_track.ball_mining import reconcile
from footy_track.feature_store.store import FeatureStore
from footy_track.labeller import jobs_client

APPLY_REFUSAL = "--apply is not implemented: the reconciler is report-only for now (re-queueing is a decision still open)."


def main(argv: list[str] | None = None, *, client=None, store: FeatureStore | None = None, split_fn=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, help="feature store DuckDB (required unless a store is injected)")
    ap.add_argument("--family", choices=[*reconcile.FAMILIES, "all"], default="all")
    ap.add_argument("--dataset", default=None, help="restrict to jobs of this labelling dataset")
    ap.add_argument("--marks-dir", type=Path, default=None, help="marks dir holding suspects/<job>.jsonl (default: the --dataset's)")
    ap.add_argument("--stuck-days", type=float, default=reconcile.DEFAULT_STUCK_DAYS)
    ap.add_argument("--json", type=Path, help="write the JSON report here (default: print it to stdout)")
    ap.add_argument("--apply", action="store_true", help="reserved; refuses")
    args = ap.parse_args(argv)
    if args.apply:
        print(APPLY_REFUSAL, file=sys.stderr)
        return 2
    if store is None:
        if args.db is None:
            ap.error("--db is required")
        store = FeatureStore.open(args.db, create=False)
    fams = tuple(reconcile.FAMILIES) if args.family == "all" else (args.family,)
    marks = args.marks_dir
    if marks is None and args.dataset:
        from footy_track.labeller import datasets  # noqa: PLC0415

        try:
            marks = datasets.get(args.dataset).marks_dir
        except datasets.UnknownDatasetError:
            marks = None
    rep = reconcile.reconcile(store, client or jobs_client, families=fams, stuck_days=args.stuck_days, dataset=args.dataset, split_fn=split_fn, marks_dir=marks)
    text = json.dumps(rep, indent=2, default=str)
    if args.json:
        args.json.write_text(text)
    else:
        print(text)
    print(reconcile.summary(rep), file=sys.stderr)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
