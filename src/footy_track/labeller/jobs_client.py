#!/usr/bin/env python3
"""Submit labelling jobs to the main labeller and fetch the results (stdlib only).

Single file, no footy_track imports: copy it anywhere, or fetch it from the server,
for a branch / worktree / Prefect flow that has not merged main::

    curl -O http://merlin:8011/api/jobs_client.py

    from jobs_client import submit, wait, results
    job = submit("Nottingham Forest windows r1", "ball_place", [
        {"clip": "arsenal_nottinghamforest_seg003", "frame_start": 10, "frame_end": 259, "step": 5},
    ], dataset="nottinghamforest", key="my-branch/nottinghamforest-windows-r1")
    wait(job["id"], poll=60)                 # blocks until done / cancelled
    labels = results(job["id"])["labels"]    # canonical rows, see README section 19

Command line (``python jobs_client.py`` or ``python -m footy_track.scripts.labelling_jobs``)::

    submit --title T --kind ball_place --dataset nottinghamforest --items windows.csv [--step 5] [--key K]
    submit --title T --kind ball_check --dataset D --items cards.json --prepare [--where mini|desktop] [--allow-large]
    prepare REF [--hold] [--where mini|desktop]     # (re)render an existing ball_check job's cards ahead of time
    status REF | results REF [-o out.json] | wait REF [--poll 60] [--timeout S] | list | done REF | cancel REF
    hooks REF [--set hooks.json] [--run NAME]       # labelling hooks: show / replace / run one now (README 19c)
    submit ... --hooks hooks.json                   # hooks at submit time
    gc [--dry-run]                                  # stale prepare runs, idle/finished renders, cache size cap

Ball Check items are pointers ``{clip, frame_index, box_index, source?}`` into a registered review list,
or inline cards ``{clip, frame_index, bbox{x,y,w,h normalised}, other_bbox?, label?, ...hidden keys}`` that
need no list file or registry edit. A job holds at most 500 cards (``--allow-large`` raises it to the hard
ceiling of 2000). ``--prepare`` also queues a Prefect run that renders every card's still and clip to disk,
so the phone only downloads finished files; the job shows as ``preparing`` until it is done.

REF is a job id or an exact (unique) job title. The server is ``$FOOTY_LABELLER_URL``
(default ``http://merlin:8011``). ``requester.branch`` and ``commit`` are filled in
from git, ``host`` from the hostname, unless you pass ``requester`` yourself.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = "http://merlin:8011"
TERMINAL = ("done", "cancelled")
#: Mirrors ``jobs.MAX_CARDS_DEFAULT`` / ``MAX_CARDS_HARD`` so the CLI can refuse before posting.
DEFAULT_MAX_CARDS = 500
HARD_MAX_CARDS = 2000
_INT_COLUMNS = {
    "frame", "frame_start", "frame_end", "start_frame", "end_frame", "step", "frame_index", "box_index",
}  # fmt: skip


class JobsError(RuntimeError):
    """The server refused a request (``status`` is the HTTP code, ``body`` the JSON reply)."""

    def __init__(self, status: int, body: object) -> None:
        detail = body.get("detail") if isinstance(body, dict) else body
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.body = body


def base_url(url: str | None = None) -> str:
    return (url or os.environ.get("FOOTY_LABELLER_URL") or DEFAULT_URL).rstrip("/")


def _request(method: str, path: str, *, url: str | None = None, body: object = None, timeout: float = 60) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(  # noqa: S310 (operator-supplied http(s) URL)
        base_url(url) + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = raw
        raise JobsError(e.code, payload) from None


def _ref(ref: str) -> str:
    return urllib.parse.quote(ref, safe="")


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=False)  # noqa: S603, S607
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def detect_requester() -> dict:
    """``{branch, commit, host}`` of the calling checkout (best effort; empty values dropped)."""
    r = {"branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "commit": _git("rev-parse", "--short=12", "HEAD"),
         "host": socket.gethostname()}  # fmt: skip
    if os.environ.get("FOOTY_JOB_SESSION"):
        r["session"] = os.environ["FOOTY_JOB_SESSION"]
    return {k: v for k, v in r.items() if v}


def submit(
    title: str,
    kind: str = "other",
    items: list[dict] | None = None,
    *,
    dataset: str | None = None,
    why: str = "",
    device: str = "either",
    priority: int = 2,
    requester: dict | None = None,
    key: str | None = None,
    url: str | None = None,
    timeout: float = 60,
    prepare: bool | str = False,
    max_cards: int | None = None,
    hooks: list[dict] | None = None,
) -> dict:
    """Create a job; returns ``{id, url, status, created, ...}``. Re-submitting the same ``key``
    returns the existing job (``created`` False). ``requester=None`` auto-detects from git.

    ``prepare=True`` (or ``"mini"`` / ``"desktop"``, where to render) also queues the render run and
    returns the job as ``preparing``. ``max_cards`` raises the per-job card cap (default 500) up to
    the hard ceiling of 2000. ``hooks`` is the labelling-hooks config (see ``hooks.py``): things the
    server runs *while the job is being labelled* (refit, reorder the remaining items, queue a Prefect run)."""
    items = items or []
    if kind == "ball_check":
        cap = max_cards or DEFAULT_MAX_CARDS
        if len(items) > HARD_MAX_CARDS:
            raise JobsError(
                422,
                f"{len(items)} cards is above the hard ceiling of {HARD_MAX_CARDS}; split the batch",
            )
        if len(items) > cap:
            raise JobsError(
                422,
                f"{len(items)} cards is above the cap of {cap}; pass --allow-large "
                f"(max_cards) to go up to {HARD_MAX_CARDS}, or split the batch",
            )
    body: dict = {
        "title": title, "kind": kind, "items": items or [], "why": why, "device": device,
        "priority": priority, "requester": detect_requester() if requester is None else requester,
    }  # fmt: skip
    if dataset:
        body["dataset"] = dataset
    if key:
        body["key"] = key
    if max_cards:
        body["max_cards"] = max_cards
    if hooks:
        body["hooks"] = hooks
    if prepare:
        body["prepare"] = {"where": prepare} if isinstance(prepare, str) else {}
        body["prepare"] = body["prepare"] or True
    return _request("POST", "/api/jobs", url=url, body=body, timeout=timeout)


def prepare(
    ref: str, *, hold: bool = False, where: str | None = None, url: str | None = None
) -> dict:
    """Queue the render run for an existing ball_check job (``hold``: an open job shows as
    ``preparing`` until it is done). Refused when a run is already queued/running, over the queued
    cards cap, or when the disk is below its floor."""
    body: dict = {"hold": hold}
    if where:
        body["where"] = where
    return _request("POST", f"/api/jobs/{_ref(ref)}/prepare", url=url, body=body)


def get_hooks(ref: str, *, url: str | None = None) -> dict:
    """The job's hook config, runtime state (counter, last run, backoff, in-flight run) and history."""
    return _request("GET", f"/api/jobs/{_ref(ref)}/hooks", url=url)


def set_hooks(ref: str, hooks: list[dict], *, url: str | None = None) -> dict:
    """Replace the job's hooks (``[]`` removes them). Raises ``JobsError`` 422 on a bad config."""
    return _request("PUT", f"/api/jobs/{_ref(ref)}/hooks", url=url, body={"hooks": hooks})


def run_hook(ref: str, name: str, *, url: str | None = None) -> dict:
    """Run one hook now (skips the debounce; refused while it is in flight or backing off)."""
    return _request("POST", f"/api/jobs/{_ref(ref)}/hooks/{_ref(name)}/run", url=url, timeout=120)


def reorder(ref: str, order: list[int], *, by: str = "api", note: str | None = None, if_rev: int | None = None,
            url: str | None = None) -> dict:  # fmt: skip
    """Put these *unanswered* item positions first, in this order. Answered items never move; positions
    that are answered / unknown are skipped (counted in the reply). ``if_rev``: 409 if the job changed."""
    body = {"order": order, "by": by, "note": note, "if_rev": if_rev}
    return _request("POST", f"/api/jobs/{_ref(ref)}/reorder", url=url, body={k: v for k, v in body.items() if v is not None})


def append(ref: str, items: list[dict], *, by: str = "api", note: str | None = None, if_rev: int | None = None,
           url: str | None = None) -> dict:  # fmt: skip
    """Append items to a live job (same shapes as submit; caps / queued-card cap / disk floor apply)."""
    body = {"items": items, "by": by, "note": note, "if_rev": if_rev}
    return _request("POST", f"/api/jobs/{_ref(ref)}/append", url=url, body={k: v for k, v in body.items() if v is not None})


def gc(*, dry_run: bool = False, url: str | None = None) -> dict:
    """Housekeeping on the server: stale prepare runs, idle/finished renders, cache size cap."""
    return _request(
        "POST",
        "/api/jobs/gc" + ("?dry_run=true" if dry_run else ""),
        url=url,
        timeout=300,
    )


def get(ref: str, *, url: str | None = None) -> dict:
    """Status + progress (computed from the label store) of a job, by id or exact title."""
    return _request("GET", f"/api/jobs/{_ref(ref)}", url=url)


def list_jobs(*, url: str | None = None, **filters: str) -> list[dict]:
    """Jobs filtered by ``status``, ``title`` (substring), ``kind``, ``dataset``, ``requester.branch``."""
    q = urllib.parse.urlencode({k: v for k, v in filters.items() if v is not None})
    return _request("GET", "/api/jobs" + (f"?{q}" if q else ""), url=url)["jobs"]


def results(ref: str, *, url: str | None = None) -> dict:
    """The labels for exactly the job's items (schema ``labelling-job-results/v1``)."""
    return _request("GET", f"/api/jobs/{_ref(ref)}/results", url=url, timeout=300)


def done(ref: str, *, url: str | None = None) -> dict:
    return _request("POST", f"/api/jobs/{_ref(ref)}/done", url=url)


def cancel(ref: str, *, url: str | None = None) -> dict:
    return _request("POST", f"/api/jobs/{_ref(ref)}/cancel", url=url)


def wait(ref: str, *, poll: float = 30, timeout: float | None = None, url: str | None = None, on_poll=None) -> dict:
    """Block until the job is done or cancelled; returns its final status document.

    Polls every ``poll`` seconds. Raises ``TimeoutError`` after ``timeout`` seconds (None = forever).
    ``on_poll(job)`` is called after each poll (progress logging).
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        job = get(ref, url=url)
        if on_poll:
            on_poll(job)
        if job["status"] in TERMINAL:
            return job
        if deadline is not None and time.monotonic() + poll > deadline:
            raise TimeoutError(f"job {job['id']} still {job['status']} after {timeout}s")
        time.sleep(poll)


# ---------------------------------------------------------------------------
# Items from a file (CLI)
# ---------------------------------------------------------------------------


def read_items(path: str, *, step: int | None = None) -> list[dict]:
    """Items from a ``.csv`` (header row; numeric columns become ints; extra columns such as
    ``stratum``/``reason`` are kept) or ``.json`` (a list, or ``{"items": [...]}``)."""
    if path.lower().endswith(".json"):
        with open(path) as f:
            doc = json.load(f)
        rows = doc["items"] if isinstance(doc, dict) else doc
    else:
        with open(path, newline="") as f:
            rows = [
                {k: (int(v) if k in _INT_COLUMNS and v.strip() != "" else v) for k, v in row.items() if v is not None}
                for row in csv.DictReader(f)
            ]
    if step is not None:
        rows = [{**r, "step": r.get("step", step)} for r in rows]
    return rows


def _load_json(path: str) -> object:
    with open(path) as f:
        return json.load(f)


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2))


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="labeller base URL (default $FOOTY_LABELLER_URL or http://merlin:8011)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="create a job (prints id, url, status)")
    s.add_argument("--title", required=True)
    s.add_argument("--kind", default="ball_place", choices=["ball_place", "ball_check", "calibration", "frame_type", "identity", "broadcast_review", "other"])
    s.add_argument("--dataset")
    s.add_argument("--items", help="CSV or JSON file of items")
    s.add_argument("--step", type=int, help="default step for window items without one")
    s.add_argument("--why", default="")
    s.add_argument("--device", default="either", choices=["phone", "laptop", "either"])
    s.add_argument("--priority", type=int, default=2)
    s.add_argument(
        "--key", help="idempotency key: re-submitting it returns the same job"
    )
    s.add_argument(
        "--prepare",
        action="store_true",
        help="ball_check: also render every card ahead of time (Prefect run)",
    )
    s.add_argument(
        "--where",
        choices=["mini", "desktop"],
        help="where --prepare renders (default: the server's choice)",
    )
    s.add_argument(
        "--allow-large",
        action="store_true",
        help=f"allow more than {DEFAULT_MAX_CARDS} cards, up to {HARD_MAX_CARDS}",
    )
    s.add_argument(
        "--max-cards", type=int, help=f"explicit card cap, at most {HARD_MAX_CARDS}"
    )
    s.add_argument("--hooks", help="JSON file: list of labelling hooks (README 19c)")
    hp = sub.add_parser("hooks", help="show / replace / run a job's labelling hooks")
    hp.add_argument("ref", help="job id or exact title")
    hp.add_argument("--set", dest="set_file", help="JSON file with the new hook list ([] removes them)")
    hp.add_argument("--run", metavar="NAME", help="run this hook now")
    for name, helptext in (("status", "status + progress"), ("results", "canonical label rows"), ("wait", "block until done"),
                           ("done", "close the job"), ("cancel", "cancel the job")):  # fmt: skip
        p = sub.add_parser(name, help=helptext)
        p.add_argument("ref", help="job id or exact title")
        if name == "results":
            p.add_argument("-o", "--out", help="write JSON here instead of stdout")
        if name == "wait":
            p.add_argument("--poll", type=float, default=30)
            p.add_argument("--timeout", type=float)
    pp = sub.add_parser(
        "prepare", help="render an existing ball_check job's cards ahead of time"
    )
    pp.add_argument("ref", help="job id or exact title")
    pp.add_argument(
        "--hold",
        action="store_true",
        help="show the job as 'preparing' (not offered) until done",
    )
    pp.add_argument("--where", choices=["mini", "desktop"])
    gp = sub.add_parser(
        "gc",
        help="expire idle/finished renders, enforce the cache cap, fail stale prepare runs",
    )
    gp.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be deleted, delete nothing",
    )
    ls = sub.add_parser("list", help="list jobs")
    for f in ("status", "title", "kind", "dataset", "branch"):
        ls.add_argument(f"--{f}")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "submit":
            items = read_items(a.items, step=a.step) if a.items else []
            res = submit(a.title, a.kind, items, dataset=a.dataset, why=a.why, device=a.device,
                         priority=a.priority, key=a.key, url=a.url, prepare=(a.where or True) if a.prepare else False,
                         max_cards=HARD_MAX_CARDS if a.allow_large and not a.max_cards else a.max_cards,
                         hooks=_load_json(a.hooks) if a.hooks else None)  # fmt: skip
            _print(res)
            if (res.get("prepare") or {}).get("state") == "failed":
                print(
                    f"warning: job created but preparation could not be queued: {res['prepare'].get('error')}",
                    file=sys.stderr,
                )
                return 2
        elif a.cmd == "prepare":
            _print(prepare(a.ref, hold=a.hold, where=a.where, url=a.url).get("prepare"))
        elif a.cmd == "hooks":
            if a.set_file:
                _print(set_hooks(a.ref, _load_json(a.set_file), url=a.url))
            elif a.run:
                _print(run_hook(a.ref, a.run, url=a.url))
            else:
                _print(get_hooks(a.ref, url=a.url))
        elif a.cmd == "gc":
            _print(gc(dry_run=a.dry_run, url=a.url))
        elif a.cmd == "status":
            j = get(a.ref, url=a.url)
            p = j.get("progress") or {}
            print(f"{j['id']}  {j['status']}  {p.get('done')}/{p.get('total')}  {j['title']}\n{j['url']}")
            if j.get("prepare"):
                pr = j["prepare"]
                print(
                    f"prepare: {pr.get('state')} {pr.get('prepared')}/{pr.get('total')} ({pr.get('where')})"
                    + (f" error: {pr['error']}" if pr.get("error") else "")
                )
        elif a.cmd == "results":
            text = json.dumps(results(a.ref, url=a.url), indent=2)
            if a.out:
                with open(a.out, "w") as f:
                    f.write(text + "\n")
            else:
                print(text)
        elif a.cmd == "wait":
            j = wait(a.ref, poll=a.poll, timeout=a.timeout, url=a.url,
                     on_poll=lambda j: print(f"{j['status']} {(j.get('progress') or {}).get('done')}/{(j.get('progress') or {}).get('total')}",
                                             file=sys.stderr, flush=True))  # fmt: skip
            print(f"{j['id']} {j['status']}")
            return 0 if j["status"] == "done" else 1
        elif a.cmd == "done":
            _print(done(a.ref, url=a.url))
        elif a.cmd == "cancel":
            _print(cancel(a.ref, url=a.url))
        else:
            for j in list_jobs(url=a.url, status=a.status, title=a.title, kind=a.kind, dataset=a.dataset,
                               **{"requester.branch": a.branch}):  # fmt: skip
                p = j.get("progress") or {}
                print(f"{j['id']}  {j['status']:<11} {p.get('done')}/{p.get('total')}  {j['kind']:<11} {j['dataset']:<8} {j['title']}")
    except (JobsError, TimeoutError, urllib.error.URLError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
