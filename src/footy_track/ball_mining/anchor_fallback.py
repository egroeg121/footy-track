"""Fallback full-frame check job (design ``ball_anchor_iteration.md`` task 7; sections 3.1, 3.4, 4 rule 3, 7.0).

A frame the anchor search cannot give a confirmed candidate for is **never** defaulted to ``no_ball``. It goes to an explicit
human full-frame check: one ``ball_place`` job (labeller job mode ``/labeller?job=<id>``, laptop). Only that check may write
``no_ball`` (``FeatureStore.write_ball_terminal_state(..., full_frame_check=True)``).

Producing the job
-----------------
:func:`fallback_entries` collects, from the search documents of :func:`anchor_search.run_search`, the frames that need the
check: ``unreachable`` (no eligible candidate), ``beyond_cap`` (no confirmed seed within the hop cap) and frames whose card was
judged ``unsure`` twice (:func:`unsure_twice`). :func:`push_fallback` submits them through ``jobs_client`` (injectable):

* train family (source job ``ball-round/...``): one job per window, key ``ball-place/<window_id>``;
* eval family (source job ``ball-eval/<panel>/<match>`` or ``ball-eval-panel/<ver>/<match>``): ONE job per panel and match,
  key ``ball-eval-place/<panel>/<match>``. The family follows the split of the source job (train vs eval) and must agree
  with its key; the two namespaces never mix (design 7.0).

Idempotent: ``submit`` with an existing key returns that job; frames already in it are skipped and only the missing ones are
appended (``if_rev`` + bounded 409 retry). A rerun adds 0. A finished fallback job cannot take frames; they are reported in
``closed``. Items are plain ``{clip, frame}`` units carrying ``split``, ``window_id``, ``source: "fallback"``, ``reason``,
``generator_version``, ``source_job``: no suggestion is attached (a ``job_suggest`` file may be built for the job separately;
anchoring bias is accepted by the design). Frames outside the source job's selected set F or of another split are refused
(task 1 guard, :func:`anchor_search.check_emittable`).

Consuming the result
--------------------
:func:`ingest_results` reads the fallback job's results document (``labels``) and writes the terminal ball state of each of
the job's own frames: a human ball box -> ``ball``; the job-mode "No ball" action (sidecar ``no_ball`` marker) -> ``no_ball``
with ``full_frame_check=True``; ``not_broadcast`` -> ``unusable``. A frame without a label is not written at all.
Precedence when one frame has several marks: ``not_broadcast`` > ``ball`` > ``no_ball``.

A placed box is also a new anchor. Anchors are not stored anywhere else: :func:`anchor_search.load_anchors` reads the sidecar
human labels (``<marks>/<clip>.jsonl``) of the **source** job's frames, and the fallback frame is one of them, so the next
``anchor_search.run_search(source_job)`` sees the box as an anchor, its input digest changes, and a new ``search/<window>/vNNN.json``
is written. :func:`reseed` is that call.

CLI (``python -m footy_track.ball_mining.anchor_fallback``)::

    push   --job REF [--marks DIR] [--dry-run]   # run_search (no write) + push_fallback
    ingest --job REF [--db PATH] [--dry-run]     # fallback job results -> frame_label_state
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.clip_id import canonical_clip_id
from footy_track.feature_store.schema import LabelMethod
from footy_track.labeller import jobs_client
from footy_track.labeller.job_edit import MAX_APPEND_PER_CALL

SOURCE = "fallback"
TRAIN_KEY_PREFIX = "ball-place"
EVAL_KEY_PREFIX = "ball-eval-place"
EVAL_SOURCE_PREFIXES = ("ball-eval", "ball-eval-panel")
REASON_BEYOND_CAP = "beyond_cap"
REASON_UNSURE_X2 = "unsure_x2"
APPEND_TRIES = 3
TRAIN_SPLIT = "train"


# ---------------------------------------------------------------------------
# namespaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    name: str  # train | eval
    eval_tail: str | None  # "<panel>/<match>" for the eval family

    def key(self, window_id: str) -> str:
        if self.name == "eval":
            return f"{EVAL_KEY_PREFIX}/{self.eval_tail}"
        return f"{TRAIN_KEY_PREFIX}/{window_id}"


def family_of(job: dict) -> Family:
    """Job family of the SOURCE job, from its split (train vs eval) checked against its key namespace."""
    split = A.job_split(job)
    key = str(job.get("key") or "")
    is_eval_key = any(key.startswith(p + "/") for p in EVAL_SOURCE_PREFIXES)
    if split is None:
        raise A.SplitError(f"job {job.get('id')!r} has no clips; no family")
    if split == TRAIN_SPLIT:
        if is_eval_key:
            raise A.SplitError(f"job {job.get('id')!r} has an eval key {key!r} but a train split")
        return Family("train", None)
    if key.startswith(f"{A.rnd.KEY_PREFIX}/"):
        raise A.SplitError(f"job {job.get('id')!r} is a train round key {key!r} but its split is {split!r}")
    tail = key.split("/", 1)[1] if is_eval_key else str(job.get("id"))
    return Family("eval", tail)


# ---------------------------------------------------------------------------
# which frames need the check
# ---------------------------------------------------------------------------


def unsure_twice(marks_dir: Path, selected: Iterable[A.Key]) -> set[A.Key]:
    """Frames of ``selected`` whose Ball Check card was answered ``unsure`` at least twice and is still ``unsure`` (design 3.1:
    the second ``unsure`` is ``deferred(unsure_x2)``; here it is routed to the full-frame check instead of being left)."""
    want = set(selected)
    counts: dict[A.Key, int] = {}
    last: dict[tuple, str | None] = {}
    vdir = Path(marks_dir) / "ball_checks"
    if not vdir.is_dir():
        return set()
    for path in sorted(vdir.glob("*.jsonl")):
        for raw in path.read_text().splitlines():
            try:
                rec = json.loads(raw)
                k = (canonical_clip_id(rec.get("clip") or path.stem), int(rec["frame_index"]))
                card = (k, rec.get("source") or "sidecar", int(rec["box_index"]))
            except (ValueError, KeyError, TypeError):
                continue
            if k not in want:
                continue
            last[card] = rec.get("verdict")
            if rec.get("verdict") == "unsure":
                counts[k] = counts.get(k, 0) + 1
    live_other = {c[0] for c, v in last.items() if v not in (None, "unsure")}
    return {k for k, n in counts.items() if n >= 2 and k not in live_other}


def fallback_entries(
    docs: Iterable[dict],
    *,
    selected: frozenset[A.Key],
    split: str | None,
    unsure: Iterable[A.Key] = (),
    terminal: Iterable[A.Key] = (),
    anchors: Iterable[A.Key] = (),
    window_frames: int = A.WINDOW_FRAMES,
) -> list[dict]:
    """``[{clip, frame, window_id, reason}]``, sorted, one per frame: unreachable + beyond-cap + unsure-twice frames, minus frames
    that are already terminal or already hold a human anchor. Every frame must be in ``selected`` and in ``split``
    (``SplitError`` otherwise, nothing is returned)."""
    found: dict[A.Key, dict] = {}
    for d in docs:
        for r in d.get("unreachable") or []:
            found.setdefault((r["clip"], int(r["frame"])), {"window_id": r.get("window_id") or d.get("window_id"), "reason": r.get("reason")})
        for r in d.get("beyond_cap") or []:
            found.setdefault((r["clip"], int(r["frame"])), {"window_id": d.get("window_id"), "reason": REASON_BEYOND_CAP})
    wins = A.split_windows(unsure, window_frames)
    for wid, (clip, _s, fs) in wins.items():
        for f in fs:
            found.setdefault((clip, f), {"window_id": wid, "reason": REASON_UNSURE_X2})
    skip = {(canonical_clip_id(c), int(f)) for c, f in (*terminal, *anchors)}
    out = [
        {"clip": canonical_clip_id(c), "frame": f, "window_id": v["window_id"], "reason": v["reason"]}
        for (c, f), v in sorted(found.items())
        if (canonical_clip_id(c), f) not in skip
    ]
    A.check_emittable([{"clip": e["clip"], "frame_index": e["frame"]} for e in out], selected, split)
    return out


def _item(e: dict, *, split: str | None, generator_version: str | None, source_job: str) -> dict:
    return {
        "clip": e["clip"], "frame": int(e["frame"]), "split": split, "window_id": e["window_id"], "source": SOURCE,
        "reason": e["reason"], "generator_version": generator_version, "source_job": source_job,
    }  # fmt: skip


def _live(job: dict) -> bool:
    return job.get("status") in A.LIVE


def _rev_conflict(e: Exception) -> bool:
    return A._is_rev_conflict(e)  # noqa: SLF001


def _push_group(client, key: str, title: str, items: list[dict], dataset: str | None, why: str) -> dict:
    out = {"key": key, "offered": len(items), "created": False, "added": 0, "skipped_existing": 0, "closed": 0}
    first = items[:MAX_APPEND_PER_CALL]
    sub = client.submit(
        title, "ball_place", first, dataset=dataset, why=why, device="laptop", key=key, requester={"session": "anchor-fallback"}
    )
    out["job"] = sub["id"]
    out["created"] = bool(sub.get("created"))
    if out["created"]:
        out["added"] = len(first)
        rest = items[MAX_APPEND_PER_CALL:]
    else:
        rest = items
    conflicts = 0
    while rest:
        res = client.results(sub["id"])
        job = res["job"]
        have = {(p["item"]["clip"], int(p["item"]["frame"])) for p in res.get("items") or [] if "frame" in p["item"]}
        fresh = [it for it in rest if (it["clip"], it["frame"]) not in have]
        out["skipped_existing"] += len(rest) - len(fresh)
        if fresh and not _live(job):
            out["closed"] = len(fresh)
            out["status"] = job.get("status")
            break
        rest = []
        batch, rest = fresh[:MAX_APPEND_PER_CALL], fresh[MAX_APPEND_PER_CALL:]
        if not batch:
            break
        try:
            r = client.append(sub["id"], batch, by=A.APPEND_BY, note=f"fallback {len(batch)} frames", if_rev=int(job.get("rev") or 0))
        except jobs_client.JobsError as e:
            if not _rev_conflict(e):
                raise
            conflicts += 1
            if conflicts >= APPEND_TRIES:
                out["append"] = "rev_conflict"
                break
            rest = batch + rest
            continue
        out["added"] += int(r["added"])
        out["skipped_existing"] += int(r.get("skipped_duplicates") or 0)
    return out


def push_fallback(
    job: dict,
    docs: Iterable[dict],
    *,
    client=None,
    store=None,
    marks_dir: Path | None = None,
    clip_of_game=lambda g: g,
    game_of_clip=lambda c: c,
    include_beyond_cap: bool = True,
    dry_run: bool = False,
    window_frames: int = A.WINDOW_FRAMES,
) -> dict:
    """Create / top up the fallback ``ball_place`` job(s) of the search documents ``docs`` of source job ``job``.

    ``client`` is anything with ``submit`` / ``results`` / ``append`` (default ``jobs_client``). Returns
    ``{"family", "frames", "groups": [...]}``; per group ``created`` / ``added`` / ``skipped_existing`` / ``closed``. Nothing
    is ever written to the feature store here."""
    client = client or jobs_client
    fam = family_of(job)
    scope = A.job_scope(job)
    docs = list(docs)
    if not include_beyond_cap:
        docs = [{**d, "beyond_cap": []} for d in docs]
    marks = Path(marks_dir) if marks_dir else A.datasets.get(job.get("dataset")).marks_dir
    selected = A.resolve_selected(job, store, clip_of_game=clip_of_game)
    term = A.terminal_frames(store, selected, game_of_clip=game_of_clip) if store is not None else ()
    anchorset = A.load_anchors(job, marks_dir=marks)
    entries = fallback_entries(
        docs, selected=selected, split=scope.split, unsure=unsure_twice(marks, selected), terminal=term,
        anchors=anchorset.anchors, window_frames=window_frames,
    )  # fmt: skip
    gen = next((d.get("generator_version") for d in docs if d.get("generator_version")), None)
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(fam.key(e["window_id"]), []).append(_item(e, split=scope.split, generator_version=gen, source_job=scope.job_id))
    out: dict = {"family": fam.name, "frames": len(entries), "groups": []}
    for key, items in sorted(groups.items()):
        title = f"Full-frame check {key.split('/', 1)[1]}"
        why = "no confirmed candidate: place the ball or mark No ball (the only way a frame ends as no_ball)"
        if dry_run:
            out["groups"].append({"key": key, "offered": len(items)})
        else:
            out["groups"].append(_push_group(client, key, title, items, job.get("dataset"), why))
    return out


# ---------------------------------------------------------------------------
# results -> frame_label_state
# ---------------------------------------------------------------------------

_PRECEDENCE = ("not_broadcast", "ball", "no_ball")  # first match wins when a frame has several marks


def decide(rows: list[dict]) -> str | None:
    """Terminal state of one frame from its label rows: ``unusable`` / ``ball`` / ``no_ball`` or None (nothing decided)."""
    st = {r.get("status") for r in rows}
    if "not_broadcast" in st:
        return "unusable"
    if "ball" in st and any(r.get("bbox") or r.get("center") or r.get("label") for r in rows if r.get("status") == "ball"):
        return "ball"
    if "no_ball" in st:
        return "no_ball"
    return None


def ingest_results(
    res: dict,
    store,
    *,
    game_of_clip=lambda c: c,
    selected: frozenset[A.Key] | None = None,
    dry_run: bool = False,
) -> dict:
    """Write the terminal ball state of every decided frame of a fallback job's results document.

    Only frames that are items of the job (``source == "fallback"``) and, when given, in ``selected`` (the source job's F) are
    considered; the job's items must all be of one split (``SplitError``). ``ball`` -> ``(ball, complete)``, the job-mode
    "No ball" action -> ``no_ball`` through ``full_frame_check=True`` (the only path that may write it), ``not_broadcast`` ->
    ``unusable``. Re-ingest is idempotent (same rows upserted). Returns counts and the frames per state."""
    job = res["job"]
    key = str(job.get("key") or "")
    if job.get("kind") != "ball_place" or not (key.startswith(TRAIN_KEY_PREFIX + "/") or key.startswith(EVAL_KEY_PREFIX + "/")):
        raise A.SplitError(f"job {job.get('id')!r} (key {key!r}) is not a fallback full-frame check job")
    items = [p["item"] for p in res.get("items") or []]
    splits_ = {A.clip_split(canonical_clip_id(it["clip"])) for it in items if "clip" in it}
    if len(splits_) > 1:
        raise A.SplitError(f"fallback job {job.get('id')!r} spans more than one split ({sorted(splits_)})")
    meta: dict[A.Key, dict] = {}
    for it in items:
        if it.get("source") == SOURCE and "frame" in it:
            meta[(canonical_clip_id(it["clip"]), int(it["frame"]))] = it
    by_frame: dict[A.Key, list[dict]] = {}
    for r in res.get("labels") or []:
        k = (canonical_clip_id(r["clip"]), int(r["frame_index"]))
        if k in meta and (selected is None or k in selected):
            by_frame.setdefault(k, []).append(r)
    out: dict = {"ball": [], "no_ball": [], "unusable": [], "undecided": 0, "written": 0}
    for k in sorted(meta):
        if selected is not None and k not in selected:
            continue
        state = decide(by_frame.get(k, []))
        if state is None:
            out["undecided"] += 1
            continue
        out[state].append(k)
        if dry_run:
            continue
        out["written"] += store.write_ball_terminal_state(
            game_of_clip(k[0]), k[1], state, method=LabelMethod.HAND_FRAME, full_frame_check=state == "no_ball",
            generator_version=meta[k].get("generator_version"),
        )  # fmt: skip
    return out


def reseed(job: dict, **kw) -> list[dict]:
    """Re-run the search of the source job after fallback placements: the placed boxes are sidecar human labels of frames in the
    job's F, so they are read as anchors by ``load_anchors`` and change the window digest (a new search file)."""
    return A.run_search(job, **kw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="anchor_fallback", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="search the source job (no file written) and create / top up its fallback job(s)")
    p.add_argument("--job", required=True, help="source job id or title (ball_check round / eval job)")
    p.add_argument("--marks", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    i = sub.add_parser("ingest", help="write the terminal ball states of a fallback job's results to the feature store")
    i.add_argument("--job", required=True, help="fallback job id or title")
    i.add_argument("--db", default=None, help="feature store path (default: the shared store)")
    i.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "push":
        src = jobs_client.get(a.job)
        from footy_track.labeller import jobs as jobs_mod  # noqa: PLC0415

        job = jobs_mod.read_job(src["id"]) or src
        docs = A.run_search(job, marks_dir=a.marks, write=False)
        print(json.dumps(push_fallback(job, docs, marks_dir=a.marks, dry_run=a.dry_run), indent=2, default=str))
        return 0
    from footy_track.feature_store import FeatureStore  # noqa: PLC0415

    res = jobs_client.results(a.job)
    from footy_track.feature_store.lock import default_db_path  # noqa: PLC0415

    store = FeatureStore.open(a.db or default_db_path())
    try:
        print(json.dumps(ingest_results(res, store, dry_run=a.dry_run), indent=2, default=str))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
