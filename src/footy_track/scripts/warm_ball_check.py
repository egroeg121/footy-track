"""Warm a labeller server's Ball Check caches for the review-list cards still to do.

A restart empties the in-memory crop / clip caches, so the first phone to open
each card waits for a video decode. This walks the not-yet-verdicted cards of
``source=list`` (the server's own ``/ball_check/queue``, so judged cards and
file order come from the same code the page uses) and GETs each card's crop and
clip once, sequentially, with a pause between requests so it stays out of the way.

    nice -n 19 python -m footy_track.scripts.warm_ball_check --base http://127.0.0.1:8011 --dataset nottinghamforest

``--dataset`` selects the labeller dataset (default: the server's default). Started
in the background by ``scripts/labeller_deploy.sh`` after restarting the service,
for each dataset it lists in WARM_DATASETS (skip with ``--no-warm``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import quote

MAX_QUEUE = 500  # server-side cap on /ball_check/queue?limit=


def _get(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (local http)
        return r.read()


def _ds(dataset: str | None) -> str:
    return f"&dataset={quote(dataset, safe='')}" if dataset else ""


def card_urls(item: dict, dataset: str | None = None) -> list[str]:
    clip, f, box = quote(str(item["clip"]), safe=""), int(item["frame_index"]), int(item["box_index"])
    return [
        f"/ball_check/crop/{clip}/{f}/{box}.jpg?src=list&pad=6&reticle=false{_ds(dataset)}",
        f"/ball_check/clip/{clip}/{f}/{box}.mp4?pad=6&src=list{_ds(dataset)}",
    ]


def fetch_todo(base: str, wait_s: float, dataset: str | None = None) -> list[dict]:
    """The undone list cards in file order; waits for the server to come up."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            q = json.loads(_get(f"{base}/ball_check/queue?source=list&limit={MAX_QUEUE}{_ds(dataset)}", 30))
            return q.get("items", [])
        except (urllib.error.URLError, OSError, ValueError) as e:
            if time.monotonic() > deadline:
                raise SystemExit(f"warm: server at {base} not ready: {e}") from e
            time.sleep(2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="e.g. http://127.0.0.1:8011")
    ap.add_argument("--dataset", default=None, help="labeller dataset name (default: server default)")
    ap.add_argument("--sleep", type=float, default=0.5, help="pause between requests (s)")
    ap.add_argument("--wait", type=float, default=120, help="max seconds to wait for the server")
    ap.add_argument("--limit", type=int, default=MAX_QUEUE)
    args = ap.parse_args(argv)
    base = args.base.rstrip("/")

    todo = fetch_todo(base, args.wait, args.dataset)[: args.limit]
    print(f"warm: {len(todo)} undone cards from {base}", flush=True)
    ok = failed = 0
    for n, item in enumerate(todo, 1):
        for path in card_urls(item, args.dataset):
            try:
                _get(base + path, 120)
                ok += 1
            except (urllib.error.URLError, OSError) as e:
                failed += 1
                print(f"warm: {path}: {e}", file=sys.stderr, flush=True)
            time.sleep(args.sleep)
        if n % 25 == 0:
            print(f"warm: {n}/{len(todo)} cards", flush=True)
    print(f"warm: done, {ok} ok, {failed} failed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
