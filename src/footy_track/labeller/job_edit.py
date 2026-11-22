"""Job mutations for labelling hooks: reorder the unanswered items, append items, keep history.

Hooks (``hooks.py``) and the HTTP API use these to steer a job *while it is being labelled*.
The rules that make that safe:

* **Answered items never move and are never removed.** An item counts as answered when every unit it
  holds has a label in the real store (the same rule as the job's progress). :func:`reorder_unanswered`
  leaves answered items at their positions and only permutes the unanswered ones among the
  positions they already occupy, so a page that is walking the job in order keeps its place and
  ``items[i]`` of an answered item is stable.
* **Appends respect the job bounds** (``jobs.py``): the per-job card/frame cap, for prepared
  ball_check jobs the queued-card cap and the free-disk floor, and a per-call cap. Items already in
  the job (same unit) are skipped, so a hook can re-append the same candidates harmlessly.
* **Provenance and history.** Appended items carry ``added_by = {by, ts, note?}`` (an item key, so it
  is echoed in results); the job keeps a ``rev`` counter bumped by every reorder/append (pages poll it
  to pick up a new order) and a ``hook_history`` list of the last ``MAX_HISTORY`` reorders, appends,
  hook runs and refits.
"""

from __future__ import annotations

from footy_track.labeller import jobs
from footy_track.labeller.jobs import JobError

MAX_HISTORY = 50
#: Items one append call may add (a hook topping a job up should do so in small steps).
MAX_APPEND_PER_CALL = 500


def _editable(job_id: str) -> dict:
    job = jobs.read_job(job_id)
    if job is None:
        raise JobError(404, f"no job {job_id!r}")
    if job["status"] not in jobs.LIVE:
        raise JobError(409, f"job is {job['status']}; only open / in_progress jobs can be reordered or appended to")
    return job


def _push_history(job: dict, entry: dict) -> None:
    hist = list(job.get("hook_history") or [])
    hist.append({"ts": jobs._now(), **entry})
    job["hook_history"] = hist[-MAX_HISTORY:]


def _save(job: dict, *, bump_rev: bool) -> None:
    if bump_rev:
        job["rev"] = int(job.get("rev") or 0) + 1
    job["updated"] = jobs._now()
    jobs._write_atomic(jobs._job_path(job["id"]), job)


def answered_positions(job: dict) -> set[int]:
    """Positions of the items whose every unit is labelled (same rule as the job's progress)."""
    c = jobs.compute(job)
    if not c["supported"]:
        raise JobError(422, c.get("error") or f"job kind {job['kind']!r} has no computed progress, so no way to tell answered items")
    return {p["index"] for p in c["items"] if p["total"] and p["done"] >= p["total"]}


def unanswered_items(job: dict) -> list[tuple[int, dict]]:
    """``(position, item)`` of the items not fully labelled yet, in job order."""
    done = answered_positions(job)
    return [(i, it) for i, it in enumerate(job["items"]) if i not in done]


# ---------------------------------------------------------------------------
# Hook config / state / history
# ---------------------------------------------------------------------------


def set_hooks(job_id: str, raw: object) -> dict:
    """Replace the job's hook config (validated; at most ``hooks.MAX_HOOKS_PER_JOB``)."""
    from footy_track.labeller import hooks  # noqa: PLC0415

    with jobs._LOCK:
        job = jobs.read_job(job_id)
        if job is None:
            raise JobError(404, f"no job {job_id!r}")
        cleaned = hooks.validate_hooks(job["kind"], raw)
        if cleaned:
            job["hooks"] = cleaned
        else:
            job.pop("hooks", None)
        keep = {h["name"] for h in cleaned}
        state = {k: v for k, v in (job.get("hook_state") or {}).items() if k in keep}
        if state:
            job["hook_state"] = state
        else:
            job.pop("hook_state", None)
        _save(job, bump_rev=False)
    hooks.forget(job_id, keep)
    return job


def record(job_id: str, entry: dict, *, state: tuple[str, dict] | None = None) -> None:
    """Append a history entry (and optionally merge ``state`` into ``hook_state[name]``)."""
    with jobs._LOCK:
        job = jobs.read_job(job_id)
        if job is None:
            return
        _push_history(job, entry)
        if state is not None:
            name, st = state
            job["hook_state"] = {**(job.get("hook_state") or {}), name: {**(job.get("hook_state") or {}).get(name, {}), **st}}
        _save(job, bump_rev=False)


# ---------------------------------------------------------------------------
# Reorder
# ---------------------------------------------------------------------------


def reorder_unanswered(
    job_id: str, order: list, *, by: str = "api", note: str | None = None, if_rev: int | None = None, meta: dict | None = None
) -> dict:
    """Put the unanswered items named in ``order`` (current item positions) first, in that order,
    ahead of the other unanswered items (which keep their relative order).

    Answered items stay exactly where they are; unanswered items are re-dealt into the positions
    the unanswered items already held. Positions that are answered, unknown or repeated are skipped
    and counted in the reply, never an error (the user may have answered between the hook reading
    the job and writing the order). ``if_rev`` makes the call fail with 409 when the job was
    reordered/appended in between.
    """
    if not isinstance(order, list):
        raise JobError(422, "order must be a list of item positions")
    with jobs._LOCK:
        job = _editable(job_id)
        if if_rev is not None and int(job.get("rev") or 0) != if_rev:
            raise JobError(409, f"job is at rev {job.get('rev') or 0}, not {if_rev}; re-read it", rev=int(job.get("rev") or 0))
        items = job["items"]
        done = answered_positions(job)
        slots = [i for i in range(len(items)) if i not in done]
        skipped = {"answered": 0, "unknown": 0, "duplicate": 0}
        want: list[int] = []
        seen: set[int] = set()
        for p in order:
            if isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < len(items):
                skipped["unknown"] += 1
            elif p in done:
                skipped["answered"] += 1
            elif p in seen:
                skipped["duplicate"] += 1
            else:
                want.append(p)
                seen.add(p)
        new_seq = want + [i for i in slots if i not in seen]
        if new_seq == slots:
            return {"changed": False, "rev": int(job.get("rev") or 0), "unanswered": len(slots), "skipped": skipped}
        out = list(items)
        for slot, src in zip(slots, new_seq, strict=True):
            out[slot] = items[src]
        job["items"] = out
        _push_history(job, {"event": "reorder", "by": by, "n_unanswered": len(slots), "n_pinned_first": len(want),
                            **({"note": note} if note else {}), **(meta or {})})  # fmt: skip
        _save(job, bump_rev=True)
        return {"changed": True, "rev": job["rev"], "unanswered": len(slots), "skipped": skipped}


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


def _dedupe_key(kind: str, item: dict):
    if kind == "ball_check":
        if jobs.is_inline(item) or "bbox" in item:
            b = item.get("bbox") or {}
            return ("card", item["clip"], item["frame_index"], tuple(round(float(b.get(k, 0)), 5) for k in "xywh"))
        return ("ptr", item.get("source", "sidecar"), item["clip"], item["frame_index"], item.get("box_index"))
    return tuple((item["clip"], u) for u in jobs._item_units(kind, item)) or ("item", item.get("clip"))


def _frame_cap(job: dict) -> tuple[int | None, int | None]:
    cap, hard = jobs._unit_cap(job["kind"], job.get("max_cards"))
    if cap is not None and job.get("max_cards") is None:
        n = sum(len(jobs._item_units(job["kind"], it)) for it in job["items"])
        if n > cap:
            cap = hard  # the job was submitted with a raised cap that was not stored
    return cap, hard


def _check_split(job: dict, new: list[dict]) -> None:
    """One job = one split: refuse (422, nothing written) items of another split than the job's own."""
    from footy_track.ball_mining import anchor_search  # noqa: PLC0415 (it imports the labeller)

    try:
        anchor_search.check_append_split(job, new)
    except anchor_search.SplitError as e:
        raise JobError(422, f"split guard: {e}") from e


def append_items(
    job_id: str, raw_items: object, *, by: str = "api", note: str | None = None, if_rev: int | None = None
) -> dict:
    """Append items (same shapes as at submit) to a live job, with provenance, inside its bounds.

    Refused (nothing written) with 422 over the job's card/frame cap or ``MAX_APPEND_PER_CALL``, 429 when
    a prepared ball_check job would pass the queued-card cap, 507 below the disk floor, and 422 when an item is of
    another split than the job's own (``splits.split_of``; a job is eval-heldout, eval-in-match or train, never mixed). Items whose
    unit is already in the job are skipped and counted.
    """
    if not isinstance(raw_items, list) or not all(isinstance(x, dict) for x in raw_items):
        raise JobError(422, "items must be a list of objects")
    if len(raw_items) > MAX_APPEND_PER_CALL:
        raise JobError(422, f"{len(raw_items)} items in one append is above the per-call cap of {MAX_APPEND_PER_CALL}")
    with jobs._LOCK:
        job = _editable(job_id)
        kind = job["kind"]
        if kind == "other":
            raise JobError(422, "'other' jobs have no item shape; append is not supported")
        if if_rev is not None and int(job.get("rev") or 0) != if_rev:
            raise JobError(409, f"job is at rev {job.get('rev') or 0}, not {if_rev}; re-read it", rev=int(job.get("rev") or 0))
        new = jobs._normalise_items(kind, raw_items) if raw_items else []
        _check_split(job, new)
        have = {_dedupe_key(kind, it) for it in job["items"]}
        fresh: list[dict] = []
        for it in new:
            k = _dedupe_key(kind, it)
            if k not in have:
                have.add(k)
                fresh.append(it)
        skipped = len(new) - len(fresh)
        if not fresh:
            return {"added": 0, "skipped_duplicates": skipped, "total": len(job["items"]), "rev": int(job.get("rev") or 0)}
        n_new = sum(len(jobs._item_units(kind, it)) for it in fresh)
        n_old = sum(len(jobs._item_units(kind, it)) for it in job["items"])
        cap, hard = _frame_cap(job)
        if cap is not None and n_old + n_new > cap:
            raise JobError(
                422,
                f"appending {n_new} would make {n_old + n_new} {'cards' if kind == 'ball_check' else 'frames'}, above the job's cap of {cap}"
                f" (hard ceiling {hard})",
                cap=cap, hard_cap=hard, have=n_old,
            )  # fmt: skip
        if kind == "ball_check" and (job.get("prepare") or {}).get("state") in ("queued", "running", "ready"):
            jobs._check_prepare_allowed(len(jobs.todo_items(job)) + len(fresh), exclude=job_id)
        stamp = {"by": by, "ts": jobs._now(), **({"note": note} if note else {})}
        for it in fresh:
            it["added_by"] = dict(stamp)
        jobs._assign_inline_boxes(job_id, job["dataset"], fresh, start=len(job["items"]))
        job["items"] = [*job["items"], *fresh]
        _push_history(job, {"event": "append", "by": by, "added": len(fresh), "skipped_duplicates": skipped,
                            **({"note": note} if note else {})})  # fmt: skip
        _save(job, bump_rev=True)
        return {"added": len(fresh), "skipped_duplicates": skipped, "total": len(job["items"]), "rev": job["rev"]}

