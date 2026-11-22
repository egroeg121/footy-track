"""Anchor-search CLI (docs/design/ball_anchor_iteration.md task 5, sections 2.3, 7.2, 7.4, 7.8).

    python -m footy_track.scripts.ball_anchor_search run   --job <ref> [--dry-run] [--window-frames 250] [--store feature.duckdb]
    python -m footy_track.scripts.ball_anchor_search sweep [--dataset train] [--dry-run]

``run``    load the round job through the labeller API (``$FOOTY_LABELLER_URL`` or ``--labeller-url``), search every window
           that has an anchor (``anchor_search.run_search``, writes ``<marks>/search/<window>/vNNN.json``, a window whose input
           digest is unchanged writes nothing), append the new cards (bucket ``S``) within the open / daily caps and reorder
           (``anchor_search.push_search``). Prints one JSON line. Idempotent: a second run appends 0.
``sweep``  the nightly backstop (flow input ``{job_id: null, mode: "sweep"}``): ``run`` over every live ``ball-round/*`` job.
           A job whose windows are all reused (same digests) and that gets no cards is reported ``noop``; the whole sweep is a
           no-op when every job is.

Both take the per-job mutex shared with ``ball_round refit`` / ``close`` (``refit.job_lock``, the local twin of the Prefect
concurrency tag ``ball-job:<job_id>``): when another run holds it the job is reported ``skipped_busy`` and the run exits 0,
so two overlapping triggers (hook + sweep) converge on one result. Exit 2 on an API / input error.

What the Prefect flow ``ball-anchor-search`` (JOBS-REPO, ``~/code/jobs``, NOT edited from this repo) must do
-----------------------------------------------------------------------------------------------------------
* deployment ``ball-anchor-search/ball-anchor-search``, cpu pool (mini); parameters
  ``{job_id: str | None, hook: str | None, trigger: dict | None, dataset: str | None, labeller_url: str | None, mode: str | None}``
  (the labeller's hook sends ``job_id, hook, trigger, dataset, labeller_url?`` exactly like ``ball-refit``);
* ``mode == "sweep"`` or ``job_id is None``  ->  ``python -m footy_track.scripts.ball_anchor_search sweep [--labeller-url U]``;
  otherwise  ->  ``... run --job <job_id> [--labeller-url U]``; fail the flow run on a non-zero exit, log the JSON line;
* concurrency tags ``ball-job:<job_id>`` (limit 1, shared with ``ball-refit`` and ``ball-round-close``) and
  ``anchor-search:<window>`` where a window is known; CPU only, never waits for the GPU;
* nightly ``ball-anchor-sweep`` after ``footy-nightly-datasets`` ingest (03:30 Europe/London) and from ``autolabel-queue-tick``;
* allow the deployment in ``FOOTY_HOOKS_PREFECT_ALLOW``; redeploy only when no parent flow is running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

from footy_track.ball_mining import anchor_controls as ctl
from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.ball_mining import refit
from footy_track.labeller import datasets, jobs_client

ERRORS = (
    jobs_client.JobsError,
    urllib.error.URLError,
    OSError,
    RuntimeError,
    ValueError,
    KeyError,
    A.SplitError,
)


def full_job(doc: dict) -> dict:
    """The job dict the generator reads (``results`` returns the job without its items)."""
    return {**doc["job"], "items": [p["item"] for p in doc.get("items") or []]}


def load_controls_gold(dataset: str, marks: Path):
    """``(gold rows, playable)`` for the S-gold controls of a job of ``dataset``: the gold GT of the dataset's own marks and of the
    round's gold datasets (as ``round.plan_round`` reads them), restricted at draw time to clips the dataset's clips dir serves
    (``playable`` returns the served stem). ``(None, None)`` when nothing can be read (controls are then simply not added)."""
    from footy_track.ball_mining import round as rnd  # noqa: PLC0415
    from footy_track.ball_mining import traindata  # noqa: PLC0415
    from footy_track.ball_mining.gold import load_gold_multi  # noqa: PLC0415
    from footy_track.ball_mining.miner import default_match_of  # noqa: PLC0415

    dirs = [Path(marks)]
    for n in rnd.RoundConfig.gold_datasets:
        try:
            dirs.append(datasets.get(n).marks_dir)
        except datasets.UnknownDatasetError:
            continue
    try:
        gold = load_gold_multi(dirs, match_of=lambda c: default_match_of(c) or c)
        resolver = traindata.ClipResolver([datasets.get(dataset).clips_dir])
    except (datasets.UnknownDatasetError, OSError, ValueError):
        return None, None

    def playable(clip: str) -> str | None:
        p = resolver.find(clip)
        return p.stem if p else None

    return gold, playable


def run_job(
    ref: str,
    *,
    client=None,
    marks_dir: str | Path | None = None,
    store=None,
    window_frames: int = A.WINDOW_FRAMES,
    dry_run: bool = False,
    today: str | None = None,
    pool_path: Path | None = None,
    root: Path | None = None,
    teacher_version: str | None = None,
    run_id: str | None = None,
    open_max: int | None = None,
    daily_max: int | None = None,
    lock_wait_s: float = 0.0,
    gold=None,
    playable=None,
    controls: bool = True,
    control_rate: float = ctl.CONTROL_RATE,
) -> dict:
    """Search + push for job ``ref`` under the job mutex. Returns a JSON-able summary with ``status`` ``ok`` / ``skipped_busy`` /
    ``skipped`` and ``noop`` (nothing new was written or appended)."""
    client = client or jobs_client
    doc = client.results(ref)
    job = doc["job"]
    out = {"job": job["id"], "key": job.get("key")}
    if job.get("status") not in refit.LIVE:
        return {
            **out,
            "status": "skipped",
            "reason": f"job is {job.get('status')}",
            "noop": True,
        }
    marks = refit.resolve_marks(job["dataset"], marks_dir)
    try:
        with refit.job_lock(marks, job["id"], wait_s=lock_wait_s):
            doc = client.results(
                ref
            )  # re-read under the lock: rev and cards as they are now
            fj = full_job(doc)
            pending = [
                (it["clip"], int(it["frame_index"]))
                for it in fj["items"]
                if it.get("frame_index") is not None
            ]
            # wrong-anchor controls: verdicts -> suspect anchors (recorded once), released by the regenerated search
            rows = refit.rows_from_results(doc)
            found = ctl.detect_suspects(rows)
            new_sus = found if dry_run else ctl.record_suspects(marks, job["id"], found, ts=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
            entries = ctl.read_suspects(marks, job["id"]) if not dry_run else [*ctl.read_suspects(marks, job["id"]), *found]
            sus = ctl.suspect_keys(entries, job["id"])
            prev = [p for k in sorted(sus) if (p := A.latest_search(marks, A.window_id_of(k[0], k[1] // window_frames * window_frames, window_frames)))]
            docs = A.run_search(fj, marks_dir=marks, root=root, pool_path=pool_path, store=store, pending=pending, suspects=sus,
                                teacher_version=teacher_version, run_id=run_id, window_frames=window_frames, write=not dry_run)  # fmt: skip
            if controls and gold is None:
                gold, playable = load_controls_gold(job["dataset"], marks)
            push = A.push_search(
                ref,
                docs,
                client=client,
                today=today,
                open_max=open_max,
                daily_max=daily_max,
                dry_run=dry_run,
                gold=gold if controls else None,
                playable=playable,
                selected=A.resolve_selected(fj, store),
                control_rate=control_rate,
            )
    except refit.JobBusy as e:
        return {**out, "status": "skipped_busy", "detail": str(e), "noop": True}
    new_files = sum(1 for d in docs if not d.get("reused"))
    impact = ctl.suspect_impact(doc, entries, prev_docs=prev)
    controls_report = {"agreement": ctl.agreement(rows), "suspect_anchors": len(sus), "newly_suspect": [e["anchor_id"] for e in new_sus],
                       "impact": impact, "controls_added": push.get("controls_added", 0)}  # fmt: skip
    changed = bool(push.get("added")) or (
        new_files > 0
        and not dry_run
        and (any(d.get("cards") for d in docs if not d.get("reused")) or bool(sus))  # a suspect released = a new, maybe card-less, file
    )
    return {**out, "status": "ok", "dry_run": dry_run, "windows": len(docs), "new_files": new_files, "reused": len(docs) - new_files,
            "cards_offered": push.get("offered", 0), "push": push, "controls": controls_report, "noop": not changed}  # fmt: skip


def sweep(*, client=None, dataset: str | None = None, **kw) -> dict:
    """:func:`run_job` over every live ``ball-round/*`` job (optionally of ``dataset``); a failing job is reported and the
    others still run. ``noop`` when every job was."""
    client = client or jobs_client
    f = {"kind": "ball_check", "progress": "true"}
    if dataset:
        f["dataset"] = dataset
    results, errors = [], 0
    for j in client.list_jobs(**f):
        if not refit.is_round_job(j) or j.get("status") not in refit.LIVE:
            continue
        try:
            results.append(run_job(j["id"], client=client, **kw))
        except ERRORS as e:
            errors += 1
            results.append(
                {
                    "job": j["id"],
                    "key": j.get("key"),
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}"[:500],
                }
            )
    return {"status": "error" if errors else "ok", "mode": "sweep", "jobs": results, "n_jobs": len(results),
            "n_changed": sum(1 for r in results if not r.get("noop")), "noop": all(r.get("noop") for r in results)}  # fmt: skip


def _open_store(path: str | None):
    if not path:
        return None
    from footy_track.scripts.crowded_frames import open_readonly  # noqa: PLC0415

    return open_readonly(path)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--labeller-url", default=None, help="default $FOOTY_LABELLER_URL")
    p.add_argument(
        "--marks-dir",
        default=None,
        help="override the dataset's marks dir (search files, lock)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="search without writing files and report what would be appended",
    )
    p.add_argument("--window-frames", type=int, default=A.WINDOW_FRAMES)
    p.add_argument(
        "--store",
        default=None,
        help="feature store DuckDB (selected_frame set F, terminal frames); default: the job's cards",
    )
    p.add_argument(
        "--pool",
        default=None,
        help="pool Parquet as the candidate source (default: the job's job_suggest file)",
    )
    p.add_argument("--teacher-version", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--open-max", type=int, default=None)
    p.add_argument("--daily-max", type=int, default=None)
    p.add_argument(
        "--lock-wait",
        type=float,
        default=0.0,
        help="seconds to wait for the job mutex before skipping (default 0)",
    )


def _kw(a: argparse.Namespace) -> dict:
    return {"marks_dir": a.marks_dir, "store": _open_store(a.store), "window_frames": a.window_frames, "dry_run": a.dry_run,
            "pool_path": Path(a.pool) if a.pool else None, "teacher_version": a.teacher_version, "run_id": a.run_id,
            "open_max": a.open_max, "daily_max": a.daily_max, "lock_wait_s": a.lock_wait}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="search + append + reorder for one round job")
    p.add_argument("--job", required=True, help="the round job: id or exact title")
    _common(p)
    p = sub.add_parser(
        "sweep", help="run over every live ball-round job (nightly backstop)"
    )
    p.add_argument("--dataset", default=None)
    _common(p)
    a = ap.parse_args(argv)
    if a.labeller_url:
        os.environ["FOOTY_LABELLER_URL"] = a.labeller_url
    try:
        out = (
            run_job(a.job, **_kw(a))
            if a.cmd == "run"
            else sweep(dataset=a.dataset, **_kw(a))
        )
    except ERRORS as e:
        print(
            json.dumps(
                {
                    "job": getattr(a, "job", None),
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}"[:500],
                }
            )
        )
        return 2
    print(json.dumps(out, default=str))
    return 2 if out.get("status") == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
