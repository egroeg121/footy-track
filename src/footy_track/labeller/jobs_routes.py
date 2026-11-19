"""HTTP API for labelling jobs (``/api/jobs``). See ``jobs.py`` and README section 19."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from footy_track.labeller import ball_check, datasets, hooks, job_edit, jobs, suggestions

router = APIRouter()


def _err(e: jobs.JobError) -> JSONResponse:
    return JSONResponse({"detail": e.detail, **e.extra}, status_code=e.status)


def _abs(request: Request, path: str) -> str:
    return str(request.base_url).rstrip("/") + path


def _view(request: Request, job: dict, c: dict | None, *, items: bool = True) -> dict:
    out = jobs.public(job, c, items=items)
    out["url"] = _abs(request, out["url"])
    return out


@router.post("/api/jobs")
async def submit_job(request: Request, body: dict):
    """Create a labelling job. Re-posting the same ``key`` returns the existing job (200)
    instead of a duplicate; a new job is 201. Body: see README section 19."""
    try:
        job, created = await asyncio.to_thread(jobs.submit, body, datasets.current_name())
        if created and (job.get("prepare") or {}).get("state") == "queued":
            job = await asyncio.to_thread(
                jobs.dispatch_prepare, job
            )  # queue the Prefect run
    except jobs.JobError as e:
        return _err(e)
    out = {
        "id": job["id"],
        "url": _abs(request, jobs.deep_link(job)),
        "status": job["status"],
        "created": created,
        "title": job["title"],
        "dataset": job["dataset"],
    }
    if job.get("prepare"):
        out["prepare"] = job["prepare"]
    return JSONResponse(out, status_code=201 if created else 200)


@router.get("/api/jobs")
async def list_jobs(
    request: Request,
    status: str | None = None,
    title: str | None = None,
    kind: str | None = None,
    dataset: str | None = None,
    branch: str | None = Query(None, alias="requester.branch"),
    progress: bool = False,
    limit: int = 200,
) -> dict:
    """Jobs, newest last. ``title`` matches exactly or as a case-insensitive substring;
    ``requester.branch`` is exact. Progress is computed for open/in_progress jobs (all with
    ``progress=true``)."""

    def work() -> list[dict]:
        out = []
        found = jobs.filter_jobs(status=status, title=title, kind=kind, dataset=dataset, branch=branch)
        for stored in found[-limit:]:
            j, c = stored, None
            if progress or stored["status"] in jobs.ACTIVE:
                j, c = jobs.job_with_progress(stored)
            out.append(_view(request, j, c, items=False))
        if status:  # auto-transitions may have moved a job out of the requested status
            out = [j for j in out if j["status"] == status]
        return out

    return {"jobs": await asyncio.to_thread(work)}


@router.post("/api/jobs/gc")
async def jobs_gc(dry_run: bool = False) -> dict:
    """Housekeeping: stale prepare runs, idle/finished renders, the render-cache size cap. Jobs and
    verdicts are never deleted. Also runs hourly inside the server."""
    return await asyncio.to_thread(lambda: jobs.gc(dry_run=dry_run))


@router.get("/api/jobs_client.py")
async def jobs_client_source() -> PlainTextResponse:
    """The dependency-free client, for a branch that has not merged main:
    ``curl -O http://mac-mini-i7:8011/api/jobs_client.py``."""
    return PlainTextResponse(Path(__file__).with_name("jobs_client.py").read_text())


@router.get("/api/jobs/{ref:path}/results")
async def job_results(request: Request, ref: str):
    """The labels for exactly this job's items, in the canonical documented JSON."""

    def work():
        job, c = jobs.job_with_progress(jobs.resolve(ref))
        doc = jobs.results_doc(job, c)
        doc["job"]["url"] = _abs(request, doc["job"]["url"])
        return doc

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


@router.get("/api/jobs/{ref:path}/prepare_plan")
async def job_prepare_plan(ref: str, pads: str = "6", skip_judged: bool = True):
    """What the ``ballcheck-prepare`` flow renders: every card still to judge with its geometry,
    clip file and exact disk-cache keys (``cached`` says which are already on disk)."""

    def work():
        job = jobs.resolve(ref)
        if job["kind"] != "ball_check":
            raise jobs.JobError(422, "only ball_check jobs have a prepare plan")
        try:
            pad_list = tuple(float(p) for p in pads.split(",") if p.strip()) or (
                ball_check.PHONE_PAD,
            )
        except ValueError:
            raise jobs.JobError(422, "pads must be comma-separated numbers") from None
        return ball_check.prepare_plan(job, pad_list, skip_judged=skip_judged)

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


@router.post("/api/jobs/{ref:path}/prepare_event")
async def job_prepare_event(ref: str, body: dict):
    """Progress reports from the prepare flow: ``{action: run|progress|finish|abort, run_id?,
    prepared?, total?, failed?, ok?, error?}``."""

    def work():
        job = jobs.resolve(ref)
        kw = {k: v for k, v in body.items() if k != "action"}
        return jobs.prepare_event(job["id"], str(body.get("action")), **kw)["prepare"]

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


@router.post("/api/jobs/{ref:path}/prepare")
async def job_prepare(request: Request, ref: str, body: dict | None = None):
    """Queue a ``ballcheck-prepare`` run for an existing ball_check job (caps, duplicates and free
    disk are checked). Body (all optional): ``hold`` (an open job shows as ``preparing`` until the
    run finishes; default false), ``where`` (``mini`` or ``desktop``)."""
    body = body or {}

    def work():
        job = jobs.resolve(ref)
        job = jobs.start_prepare(
            job["id"], hold=bool(body.get("hold")), where=body.get("where")
        )
        return _view(request, job, None, items=False)

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


@router.get("/api/jobs/{ref:path}/rev")
async def job_rev(ref: str):
    """Cheap change marker for pages that walk a job in order: ``rev`` goes up whenever a hook (or
    the API) reorders or appends to the job."""

    def work():
        job = jobs.resolve(ref)
        return {"id": job["id"], "rev": int(job.get("rev") or 0), "status": job["status"]}

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


def _hooks_view(job: dict) -> dict:
    return {"id": job["id"], "rev": int(job.get("rev") or 0), "hooks": job.get("hooks") or [],
            "state": hooks.runtime_view(job), "history": job.get("hook_history") or [],
            "actions": sorted(hooks.ACTIONS)}  # fmt: skip


@router.get("/api/jobs/{ref:path}/hooks")
async def get_job_hooks(ref: str):
    """The job's hook config, per-hook runtime (counter, last run, backoff, in-flight run) and the
    history of reorders / appends / hook runs; ``actions`` lists the registered in-process actions."""
    try:
        return await asyncio.to_thread(lambda: _hooks_view(jobs.resolve(ref)))
    except jobs.JobError as e:
        return _err(e)


@router.put("/api/jobs/{ref:path}/hooks")
async def put_job_hooks(ref: str, body: dict):
    """Replace the job's hooks. Body ``{"hooks": [{name, trigger{every_n?, min_interval_s?, on_done?},
    action{type: builtin|prefect, ...}}]}``; ``{"hooks": []}`` removes them. See ``hooks.py``."""

    def work():
        job = jobs.resolve(ref)
        return _hooks_view(job_edit.set_hooks(job["id"], body.get("hooks")))

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


@router.post("/api/jobs/{ref:path}/hooks/{name}/run")
async def run_job_hook(ref: str, name: str):
    """Run one hook now (skips the debounce, not the one-in-flight rule or the failure backoff)."""

    def work():
        return hooks.run_now(jobs.resolve(ref)["id"], name)

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


@router.post("/api/jobs/{ref:path}/reorder")
async def job_reorder(ref: str, body: dict):
    """Reorder the *unanswered* items: ``{order: [item positions], by?, note?, if_rev?}``. The listed
    unanswered items go first in that order; answered items never move. See ``job_edit.py``."""

    def work():
        job = jobs.resolve(ref)
        return job_edit.reorder_unanswered(job["id"], body.get("order"), by=str(body.get("by") or "api")[:80],
                                           note=body.get("note"), if_rev=body.get("if_rev"))  # fmt: skip

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


@router.post("/api/jobs/{ref:path}/append")
async def job_append(ref: str, body: dict):
    """Append items to a live job: ``{items: [...], by?, note?, if_rev?}`` (same shapes as at submit;
    caps, queued-card cap and disk floor apply; appended items carry ``added_by``)."""

    def work():
        job = jobs.resolve(ref)
        return job_edit.append_items(job["id"], body.get("items"), by=str(body.get("by") or "api")[:80],
                                     note=body.get("note"), if_rev=body.get("if_rev"))  # fmt: skip

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


@router.get("/api/jobs/{ref:path}/frames")
async def job_frames(ref: str):
    """The job's frames in order with ``done`` flags (same rule as progress). Drives the
    labeller's job mode (``/labeller?job=<id>``)."""

    def work():
        job = jobs.resolve(ref)
        if job["kind"] == "ball_place":
            return jobs.frame_list(job)
        raise jobs.JobError(422, f"job kind {job['kind']!r} has no labeller frame list")

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


@router.get("/api/jobs/{ref:path}/suggestion")
async def job_suggestion(ref: str, clip: str, frame: int):
    """The detector suggestion (``{x,y,w,h,conf,detector}``, normalised) for one frame of a ``ball_place`` job, or
    ``suggestion: null`` (blind card, no candidate, nothing precomputed). Never a label: see ``suggestions.py``."""

    def work():
        return suggestions.suggestion_for(jobs.resolve(ref), clip, frame)

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


async def _close(request: Request, ref: str, new: str):
    def work():
        job = jobs.close(jobs.resolve(ref)["id"], new)
        return _view(request, job, None)

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)


@router.post("/api/jobs/{ref:path}/done")
async def job_done(request: Request, ref: str):
    return await _close(request, ref, "done")


@router.post("/api/jobs/{ref:path}/cancel")
async def job_cancel(request: Request, ref: str):
    return await _close(request, ref, "cancelled")


@router.post("/api/jobs/{ref:path}/reopen")
async def job_reopen(request: Request, ref: str):
    return await _close(request, ref, "open")


@router.get("/api/jobs/{ref:path}")
async def get_job(request: Request, ref: str):
    """A job by id or exact (unique) title, with progress computed from the label store."""

    def work():
        job, c = jobs.job_with_progress(jobs.resolve(ref))
        return _view(request, job, c)

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return _err(e)
