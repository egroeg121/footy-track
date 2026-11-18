"""Plan / create / status of one training-miner round (docs/design/ball_sampling.md §3, §6).

    python -m footy_track.scripts.ball_round plan   --machine-labels /mnt/storage/footy_data/ball_candidates --out plan.json
    python -m footy_track.scripts.ball_round plan   --pool /mnt/storage/footy_data/ball_pool
    python -m footy_track.scripts.ball_round create --pool ... [--force] [--dry-run] [--prepare]
    python -m footy_track.scripts.ball_round status

``plan`` never touches the labeller. ``create`` posts one ``ball_check`` job (key ``ball-round/<date>/<n>``) unless the
tick rule says no (open training cards < 100 and today's cards < 600; ``--force`` overrides). The pool is either the
Parquet written by ``ball-pool-score`` (``--pool``, a file or a directory of them) or RT-DETR jsonl (``--machine-labels``,
single source, ``p_real`` = raw confidence).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from footy_track.ball_mining import round as rnd


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
    if rep["refused"]:
        print("REFUSED (not train):", json.dumps(rep["refused"]))
    if rep["excluded"]:
        print("excluded:", rep["excluded"])
    print("inclusion probabilities (R, A):", json.dumps(rep["inclusion_probabilities"]))


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
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--prepare",
        action="store_true",
        help="also queue the ballcheck-prepare render run",
    )
    p = sub.add_parser(
        "status", help="open / today counts, tick decision, latest state"
    )
    _common(p)
    a = ap.parse_args(argv)
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
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != "report"}, indent=1, default=str
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
