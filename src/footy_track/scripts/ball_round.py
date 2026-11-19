"""Plan / create / status of one training-miner round (docs/design/ball_sampling.md §3, §6).

    python -m footy_track.scripts.ball_round plan   --machine-labels /mnt/storage/footy_data/ball_candidates --out plan.json
    python -m footy_track.scripts.ball_round plan   --pool /mnt/storage/footy_data/ball_pool
    python -m footy_track.scripts.ball_round create --pool ... [--force] [--dry-run] [--prepare]
    python -m footy_track.scripts.ball_round status
    python -m footy_track.scripts.ball_round refit --job <ref>      # Prefect flow ball-refit (hook every_n 50)
    python -m footy_track.scripts.ball_round close --job <ref>      # Prefect flow ball-round-close (hook on_done)

``refit`` and ``close`` run OUTSIDE the labeller server (the job's hooks only queue the Prefect deployments): they read the
verdicts through the labeller API (``$FOOTY_LABELLER_URL`` or ``--labeller-url``), write ``<marks>/ball_mining/state/vNNN.json``
and POST the new order to ``/api/jobs/<ref>/reorder`` (``refit``), or write the tier-3 auto-accept files
``<ball_auto>/<scorer_version>/<match>.jsonl`` and record the round as closed (``close``). Both print one JSON line; exit 2 on
an API or input error (nothing is retried here).

``plan`` never touches the labeller. ``create`` posts one ``ball_check`` job (key ``ball-round/<date>/<n>``) unless the
tick rule says no (open training cards < 100 and today's cards < 600; ``--force`` overrides). The pool is either the
Parquet written by ``ball-pool-score`` (``--pool``, a file or a directory of them) or RT-DETR jsonl (``--machine-labels``,
single source, ``p_real`` = raw confidence).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
from pathlib import Path

from footy_track.ball_mining import close as close_mod
from footy_track.ball_mining import refit as refit_mod
from footy_track.ball_mining import round as rnd
from footy_track.labeller import jobs_client


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default=rnd.DEFAULT_DATASET)
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-match-cap", type=float, default=0.15)
    p.add_argument(
        "--data-root",
        default=None,
        help="footy_data root (segments.json for the +-10 s eval exclusion)",
    )
    p.add_argument(
        "--marks-dir",
        default=None,
        help="override the dataset's marks dir (verdicts, gold, state)",
    )
    p.add_argument("--no-gold", action="store_true")
    p.add_argument(
        "--gold-datasets",
        default=",".join(rnd.RoundConfig().gold_datasets),
        help="registry datasets the gold GT and the gold cards' videos come from (comma separated; default %(default)s)",
    )
    p.add_argument(
        "--no-link",
        action="store_true",
        help="do not register the dataset / link the cards' clips into its clips dir (tests; cards then will not render)",
    )
    p.add_argument(
        "--quotas",
        default=None,
        help='JSON, e.g. \'{"U":.35,"M":.2,"N":.15,"R":.15,"A":.1,"G":.05}\'',
    )


def _pool_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pool", help="pool Parquet file, or a directory of them")
    p.add_argument(
        "--machine-labels",
        nargs="*",
        default=[],
        help="RT-DETR jsonl files / directories (ball rows)",
    )
    p.add_argument(
        "--match",
        default=None,
        help="match key for --machine-labels (default: derived from the clip name)",
    )
    p.add_argument("--fps", type=float, default=25.0)


def _cfg(a: argparse.Namespace) -> rnd.RoundConfig:
    cfg = rnd.RoundConfig(
        dataset=a.dataset,
        budget=a.budget,
        seed=a.seed,
        per_match_cap=a.per_match_cap,
        use_gold=not a.no_gold,
        gold_datasets=tuple(x for x in a.gold_datasets.split(",") if x),
        link_clips=not a.no_link,
    )
    if a.data_root:
        cfg.data_root = a.data_root
    if a.quotas:
        cfg.quotas = json.loads(a.quotas)
    return cfg


def _print_report(rep: dict) -> None:
    print(
        f"cards {rep['n_cards']} / budget {rep['budget']}  (pool rows {rep['pool_rows_in']}, minable {rep['pool_rows_minable']})"
    )
    print(
        f"targets {rep['targets']}\ngot     {rep['got']}\nspilled to R: {rep['spilled_to_R']}"
    )
    cap = rep["per_match_cap"]
    print(
        f"per-match cap {cap['cards']} cards{' (relaxed to equal share)' if cap['relaxed_to_equal_share'] else ''}"
    )
    print("per match x bucket:")
    for m, row in rep["by_match_bucket"].items():
        print(f"  {m}: {dict(sorted(row.items()))}  total {sum(row.values())}")
    if rep.get("gold_unresolved_clips"):
        print("gold clips with no video (left out):", rep["gold_unresolved_clips"])
    if rep["refused"]:
        print("REFUSED (not train):", json.dumps(rep["refused"]))
    if rep["excluded"]:
        print("excluded:", rep["excluded"])
    print("inclusion probabilities (R, A):", json.dumps(rep["inclusion_probabilities"]))


def _job_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--job", required=True, help="the round job: id or exact title")
    p.add_argument("--labeller-url", default=None, help="default $FOOTY_LABELLER_URL")
    p.add_argument(
        "--marks-dir",
        default=None,
        help="override the dataset's marks dir (state, closed marker)",
    )
    p.add_argument(
        "--params", default=None, help='JSON: {"min_per_class", "target", "delta"}'
    )


def _run_job_cmd(a: argparse.Namespace) -> int:
    """``refit`` / ``close``: one JSON line on stdout; rc 0 ok, 2 on an API / input error."""
    if a.labeller_url:
        os.environ["FOOTY_LABELLER_URL"] = a.labeller_url
    params = json.loads(a.params) if a.params else None
    try:
        if a.cmd == "refit":
            out = refit_mod.refit_round(
                a.job, marks_dir=a.marks_dir, params=params, reorder=not a.no_reorder
            )
        else:
            out = close_mod.close_round(a.job, pool=a.pool, root=a.out_root, marks_dir=a.marks_dir, data_root=a.data_root,
                                        params=params, force=a.force, allow_flagged=a.allow_flagged)  # fmt: skip
    except (
        jobs_client.JobsError,
        urllib.error.URLError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as e:
        print(
            json.dumps(
                {
                    "job": a.job,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}"[:500],
                }
            )
        )
        return 2
    print(
        json.dumps({"status": "ok", **out} if "status" not in out else out, default=str)
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="print the would-be cards and bucket counts")
    _common(p)
    _pool_args(p)
    p.add_argument("--out", help="write {cards, report} JSON here")
    p = sub.add_parser("create", help="submit one ball_check job (tick rule applies)")
    _common(p)
    _pool_args(p)
    p.add_argument("--force", action="store_true", help="ignore the tick rule")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="mine and prepare but submit nothing (the dataset is still registered and the clips linked: both idempotent)",
    )
    p.add_argument("--out", help="write the cards of a --dry-run here as JSON")
    p.add_argument(
        "--prepare",
        action="store_true",
        help="also queue the ballcheck-prepare render run",
    )
    p = sub.add_parser(
        "status", help="open / today counts, tick decision, latest state"
    )
    _common(p)
    p = sub.add_parser(
        "refit",
        help="refit from the round's verdicts, write state, POST the new order (outside the server)",
    )
    _job_args(p)
    p.add_argument(
        "--no-reorder", action="store_true", help="only write the state file"
    )
    p = sub.add_parser(
        "close",
        help="final refit, tier-3 auto-accept files, record the round as closed",
    )
    _job_args(p)
    p.add_argument(
        "--pool",
        default=close_mod.DEFAULT_POOL,
        help="pool Parquet file or directory (default %(default)s)",
    )
    p.add_argument(
        "--out-root",
        default=None,
        help=f"auto-accept root (default $FOOTY_BALL_AUTO_ROOT or {close_mod.DEFAULT_OUT_ROOT})",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="footy_data root (segments.json for the +-10 s eval exclusion)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="redo a closed round / close a job that is not done",
    )
    p.add_argument(
        "--allow-flagged",
        action="store_true",
        help="write auto-accepts even if the gold check flagged the round",
    )
    a = ap.parse_args(argv)
    if a.cmd in ("refit", "close"):
        return _run_job_cmd(a)
    cfg = _cfg(a)
    if a.cmd == "status":
        print(json.dumps(rnd.status(cfg), indent=1, default=str))
        return 0
    pool = rnd.load_pool(a.pool, a.machine_labels, match=a.match, fps=a.fps)
    marks = Path(a.marks_dir) if a.marks_dir else None
    if a.cmd == "plan":
        res = rnd.plan_round(pool, cfg, marks=marks)
        _print_report(res.report)
        if a.out:
            Path(a.out).write_text(
                json.dumps(
                    {"report": res.report, "cards": res.cards}, indent=1, default=str
                )
            )
            print(f"wrote {a.out}")
        return 0
    out = rnd.create_round(
        pool, cfg, force=a.force, dry_run=a.dry_run, prepare=a.prepare, marks=marks
    )
    if a.out and out.get("cards"):
        Path(a.out).write_text(json.dumps(out["cards"], indent=1, default=str))
    print(
        json.dumps(
            {k: v for k, v in out.items() if k not in ("report", "cards")}, indent=1, default=str
        )
    )
    if out.get("report"):
        _print_report(out["report"])
    return (
        0
        if out.get("created")
        or out.get("dry_run")
        or not out.get("tick", {}).get("allowed", True)
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
