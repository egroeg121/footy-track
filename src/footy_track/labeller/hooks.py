"""Labelling hooks: run something *while* a job is being labelled.

Every label write on the site (Ball Check verdicts, ball_place marks, calibration GT, frame-type
flags, identity pair/review records, broadcast-review verdicts) calls :func:`emit`. The call only
puts a small event on a bounded in-memory queue; a single daemon thread matches it to the live jobs
that contain that unit, counts it against each job's hooks and, when a trigger fires, runs the
hook's action on a small thread pool. Nothing here can slow or break the write path: ``emit`` never
blocks, never raises, drops the event when the queue is full, and every error in the worker is
logged and swallowed.

Per-job config (``job["hooks"]``, set at submit or with ``PUT /api/jobs/<id>/hooks``)::

    {"name": "calibrate",                      # unique in the job, [a-z0-9_.-]
     "trigger": {"every_n": 20,                # run after every 20 label events on this job's units
                 "min_interval_s": 60,         # debounce: never more often than this (default 60)
                 "on_done": false},            # also run once when the job completes
     "action": {"type": "builtin", "name": "broadcast_calibrate", "params": {}}}
    {"name": "rescore", "trigger": {"every_n": 50}, "action":
        {"type": "prefect", "deployment": "flow-name/deployment-name", "params": {...}}}

Bounds (all per ``(job, hook)`` unless noted): at most one run in flight; ``min_interval_s``
debounce (prefect actions: at least ``MIN_PREFECT_INTERVAL_S``); after a failed run no new run for
``BACKOFF_S`` (10 min, as ``jobs.maybe_reprepare``); a run that never reports back is treated as
failed after ``INFLIGHT_STALE_S``; at most ``MAX_HOOKS_PER_JOB`` hooks per job and
``MAX_PREFECT_INFLIGHT`` prefect runs in flight across all jobs; ``EVENT_QUEUE_MAX`` queued events.

Actions
-------
``register_action(name, fn, kinds=...)`` adds an in-process action; ``fn(ctx: HookContext) -> dict``
runs on a worker thread with the job's dataset current, and returns a small JSON-able summary that is
stored in the job's ``hook_history``. A ``{"noop": "reason"}`` result is a success that did nothing.
It steers the job through ``job_edit.reorder_unanswered`` / ``append_items`` (answered items are never
touched). A ``prefect`` action queues ``deployment`` with ``{job_id, hook, trigger, dataset, labeller_url?,
**params}`` through ``prefect_client`` and stays "in flight" until Prefect reports a terminal state (the
flow talks back through the same reorder/append API).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import random
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import datasets, job_edit, jobs

LOGGER = logging.getLogger(__name__)

MAX_HOOKS_PER_JOB = 8
MIN_INTERVAL_DEFAULT_S = 60.0
MIN_PREFECT_INTERVAL_S = 30.0
BACKOFF_S = 600.0
INFLIGHT_STALE_S = 2 * 3600.0
EVENT_QUEUE_MAX = 2000
ACTION_WORKERS = 3
MAX_PREFECT_INFLIGHT = 3
PREFECT_POLL_S = 15.0
_PARAMS_MAX_BYTES = 4000
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,40}$")
_DEPLOYMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_PREFECT_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "CRASHED"}
_PREFECT_BAD = {"FAILED", "CANCELLED", "CRASHED"}

#: Which job kinds a label write of ``event.kind`` can answer a unit of.
_JOB_KINDS = {
    "calibration_gt": ("calibration",),
    "frame_type": ("calibration", "frame_type"),
}
#: Kinds whose clip names may differ by alias; matched through ``canonical_clip_id``.
_CANON_KINDS = ("ball_place", "calibration", "frame_type", "identity")


def enabled() -> bool:
    return os.environ.get("FOOTY_LABELLING_HOOKS", "1") not in ("0", "false", "off", "")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelEvent:
    """One label write. ``unit`` is a frame index (frame kinds, broadcast_review), ``(source,
    frame_index, box_index)`` (ball_check) or None (identity: the clip is the unit)."""

    kind: str
    dataset: str
    clip: str
    unit: Any = None
    verdict: str | None = None
    ts: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "dataset": self.dataset, "clip": self.clip,
                "unit": list(self.unit) if isinstance(self.unit, tuple) else self.unit,
                "verdict": self.verdict, "ts": self.ts}  # fmt: skip


_Q: queue.Queue = queue.Queue(maxsize=EVENT_QUEUE_MAX)
_STATS = {"emitted": 0, "dropped": 0, "errors": 0}
_THREAD_LOCK = threading.Lock()
_worker: threading.Thread | None = None
_EXEC: ThreadPoolExecutor | None = None
_PENDING = 0
_PENDING_LOCK = threading.Lock()


def emit(kind: str, *, clip: str, unit: Any = None, verdict: str | None = None, dataset: str | None = None, **meta) -> None:
    """Report one label write. Cheap, non-blocking, never raises (see the module docstring).

    ``kind`` is the store that was written: ``ball_check``, ``ball_place``, ``calibration_gt``,
    ``frame_type``, ``identity``, ``broadcast_review``. ``dataset`` defaults to the request's.
    """
    if not enabled():
        return
    try:
        ds = dataset or datasets.current_name()
        _ensure_started()
        _Q.put_nowait(("label", LabelEvent(kind, ds, str(clip), unit, verdict, meta=meta)))
        _STATS["emitted"] += 1
    except queue.Full:
        _STATS["dropped"] += 1
        LOGGER.warning("hooks: event queue full, dropping a %s event", kind)
    except Exception:  # noqa: BLE001 - the write path must never see a hook problem
        _STATS["errors"] += 1
        LOGGER.exception("hooks: could not emit %s event", kind)


def job_done(job_id: str) -> None:
    """A job just completed: run its ``on_done`` hooks (queued; called from ``jobs._set_status``)."""
    if not enabled():
        return
    try:
        _ensure_started()
        _Q.put_nowait(("done", job_id))
    except Exception:  # noqa: BLE001
        LOGGER.exception("hooks: could not queue on_done for %s", job_id)


def _ensure_started() -> None:
    global _worker, _EXEC  # noqa: PLW0603
    if _worker is not None and _worker.is_alive():
        return
    with _THREAD_LOCK:
        if _worker is not None and _worker.is_alive():
            return
        if _EXEC is None:
            _EXEC = ThreadPoolExecutor(max_workers=ACTION_WORKERS, thread_name_prefix="label-hook")
        _worker = threading.Thread(target=_loop, name="label-hooks", daemon=True)
        _worker.start()


def _loop() -> None:
    while True:
        item = _Q.get()
        try:
            what, payload = item
            if what == "label":
                _handle_event(payload)
            elif what == "done":
                _handle_done(payload)
        except Exception:  # noqa: BLE001
            _STATS["errors"] += 1
            LOGGER.exception("hooks: error handling %s", item[0] if item else item)
        finally:
            _Q.task_done()


def drain(timeout: float = 5.0) -> bool:
    """Wait until the event queue is empty and no action is running (tests, scripts)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if _Q.unfinished_tasks == 0 and _PENDING == 0:
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# Matching an event to jobs
# ---------------------------------------------------------------------------

_SCAN: dict[str, tuple[tuple, dict | None]] = {}


def _hooked_jobs() -> list[dict]:
    """The live jobs that have hooks, read through a per-file (mtime, size) cache so a label write
    costs a directory listing, not a parse of every job."""
    d = jobs.jobs_dir()
    if not d.is_dir():
        return []
    out = []
    live = set()
    for p in d.glob("*.json"):
        live.add(str(p))
        try:
            st = p.stat()
        except OSError:
            continue
        sig = (st.st_mtime_ns, st.st_size)
        hit = _SCAN.get(str(p))
        if hit is None or hit[0] != sig:
            try:
                doc = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            keep = doc if isinstance(doc, dict) and doc.get("hooks") and doc.get("id") else None
            _SCAN[str(p)] = (sig, keep)
            hit = _SCAN[str(p)]
        if hit[1] is not None and hit[1].get("status") in jobs.LIVE:
            out.append(hit[1])
    for k in [k for k in _SCAN if k not in live and k.startswith(str(d))]:
        _SCAN.pop(k, None)
    return out


def _same_clip(kind: str, a: str, b: str) -> bool:
    if kind in _CANON_KINDS:
        return canonical_clip_id(a) == canonical_clip_id(b)
    return a == b


def event_matches(job: dict, ev: LabelEvent) -> bool:  # noqa: PLR0911, PLR0912
    """Does the written unit belong to this job (same dataset, a job kind the write can answer)?"""
    kind = job["kind"]
    if job.get("dataset") != ev.dataset or kind not in _JOB_KINDS.get(ev.kind, (ev.kind,)):
        return False
    for it in job.get("items") or []:
        if not _same_clip(kind, it.get("clip", ""), ev.clip):
            continue
        if kind in ("ball_place", "calibration", "frame_type"):
            u = ev.unit
            if not isinstance(u, int):
                continue
            if "frame" in it:
                if it["frame"] == u:
                    return True
            else:
                lo, hi, step = it.get("frame_start"), it.get("frame_end"), it.get("step") or 1
                if lo is not None and hi is not None and lo <= u <= hi and (u - lo) % step == 0:
                    return True
        elif kind == "broadcast_review":
            if it.get("frame") == ev.unit:
                return True
        elif kind == "ball_check":
            if isinstance(ev.unit, tuple) and tuple(jobs._item_units("ball_check", it)[0]) == tuple(ev.unit):
                return True
        elif kind == "identity":
            return True
    return False


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _num(v: object, what: str, *, minimum: float = 0) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < minimum:
        raise jobs.JobError(422, f"{what} must be a number >= {minimum:g}")
    return float(v)


def validate_hooks(kind: str, raw: object) -> list[dict]:  # noqa: PLR0912, PLR0915
    """Normalise a ``hooks`` list (raises ``JobError`` 422). ``None`` / ``[]`` -> ``[]``."""
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(h, dict) for h in raw):
        raise jobs.JobError(422, "hooks must be a list of objects")
    if len(raw) > MAX_HOOKS_PER_JOB:
        raise jobs.JobError(422, f"a job can have at most {MAX_HOOKS_PER_JOB} hooks")
    _load_builtins()
    out, names = [], set()
    for i, h in enumerate(raw):
        where = f"hooks[{i}]"
        name = h.get("name")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise jobs.JobError(422, f"{where}.name must match {_NAME_RE.pattern}")
        if name in names:
            raise jobs.JobError(422, f"{where}.name {name!r} is used twice")
        names.add(name)
        trig = h.get("trigger")
        if not isinstance(trig, dict):
            raise jobs.JobError(422, f"{where}.trigger must be an object {{every_n?, min_interval_s?, on_done?}}")
        every_n = trig.get("every_n")
        if every_n is not None and (isinstance(every_n, bool) or not isinstance(every_n, int) or every_n < 1):
            raise jobs.JobError(422, f"{where}.trigger.every_n must be a positive integer")
        on_done = bool(trig.get("on_done", False))
        if every_n is None and not on_done:
            raise jobs.JobError(422, f"{where}.trigger needs every_n and/or on_done")
        interval = _num(trig.get("min_interval_s", MIN_INTERVAL_DEFAULT_S), f"{where}.trigger.min_interval_s")
        act = h.get("action")
        if not isinstance(act, dict):
            raise jobs.JobError(422, f"{where}.action must be an object")
        params = act.get("params") or {}
        if not isinstance(params, dict) or len(json.dumps(params, default=str)) > _PARAMS_MAX_BYTES:
            raise jobs.JobError(422, f"{where}.action.params must be an object under {_PARAMS_MAX_BYTES} bytes of JSON")
        typ = act.get("type")
        if typ == "builtin":
            a = ACTIONS.get(act.get("name"))
            if a is None:
                raise jobs.JobError(422, f"{where}.action.name {act.get('name')!r} is not a registered action: {sorted(ACTIONS)}")
            if a.kinds is not None and kind not in a.kinds:
                raise jobs.JobError(422, f"action {a.name!r} is for {list(a.kinds)} jobs, not {kind!r}")
            clean_act = {"type": "builtin", "name": a.name, "params": params}
        elif typ == "prefect":
            dep = act.get("deployment")
            if not isinstance(dep, str) or not _DEPLOYMENT_RE.fullmatch(dep):
                raise jobs.JobError(422, f"{where}.action.deployment must look like 'flow-name/deployment-name'")
            allow = [d.strip() for d in os.environ.get("FOOTY_HOOKS_PREFECT_ALLOW", "").split(",") if d.strip()]
            if allow and dep not in allow:
                raise jobs.JobError(422, f"deployment {dep!r} is not in $FOOTY_HOOKS_PREFECT_ALLOW {allow}")
            if interval < MIN_PREFECT_INTERVAL_S:
                raise jobs.JobError(422, f"{where}.trigger.min_interval_s must be >= {MIN_PREFECT_INTERVAL_S:g} for a prefect action")
            clean_act = {"type": "prefect", "deployment": dep, "params": params}
        else:
            raise jobs.JobError(422, f"{where}.action.type must be 'builtin' or 'prefect'")
        trigger = {"min_interval_s": interval, "on_done": on_done}
        if every_n is not None:
            trigger["every_n"] = every_n
        out.append({"name": name, "trigger": trigger, "action": clean_act})
    return out


# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------


@dataclass
class HookContext:
    """What an action gets. ``job`` is a fresh read taken when the run started."""

    job: dict
    hook: dict
    trigger: str  # every_n | on_done | manual
    event: LabelEvent | None
    params: dict

    @property
    def by(self) -> str:
        return f"hook:{self.hook['name']}"

    def reorder(self, order: list[int], **kw) -> dict:
        return job_edit.reorder_unanswered(self.job["id"], order, by=self.by, **kw)

    def append(self, items: list[dict], **kw) -> dict:
        return job_edit.append_items(self.job["id"], items, by=self.by, **kw)


@dataclass
class Action:
    name: str
    fn: Callable[[HookContext], dict | None]
    kinds: tuple[str, ...] | None = None
    doc: str = ""


ACTIONS: dict[str, Action] = {}


def register_action(name: str, fn: Callable[[HookContext], dict | None], *, kinds: tuple[str, ...] | None = None) -> None:
    """Register an in-process action under ``name`` (``kinds``: job kinds it may be attached to)."""
    ACTIONS[name] = Action(name, fn, kinds, (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "")


def _load_builtins() -> None:
    """Import the modules that register actions (kind-specific ones live next to their kind)."""
    with contextlib.suppress(ImportError):
        from footy_track.labeller import broadcast_review  # noqa: F401, PLC0415


def _shuffle_unanswered(ctx: HookContext) -> dict:
    """Reshuffle the job's unanswered items (any kind). params: ``seed``."""
    pos = [i for i, _ in job_edit.unanswered_items(ctx.job)]
    rng = random.Random(ctx.params.get("seed", f"{ctx.job['id']}:{ctx.job.get('rev', 0)}"))
    rng.shuffle(pos)
    return {"reordered": len(pos), **ctx.reorder(pos, note="shuffle")}


def _sort_by_field(ctx: HookContext) -> dict:
    """Order the unanswered items by an item key (any kind). params: ``key``, ``descending``."""
    key = ctx.params.get("key")
    if not isinstance(key, str):
        raise ValueError("sort_by_field needs params.key")
    rows = [(i, it.get(key)) for i, it in job_edit.unanswered_items(ctx.job)]
    have = sorted((r for r in rows if isinstance(r[1], (int, float))), key=lambda r: r[1], reverse=bool(ctx.params.get("descending")))
    order = [i for i, _ in have] + [i for i, v in rows if not isinstance(v, (int, float))]
    return {"sorted": len(have), **ctx.reorder(order, note=f"sort by {key}")}


register_action("shuffle_unanswered", _shuffle_unanswered)
register_action("sort_by_field", _sort_by_field)


# ---------------------------------------------------------------------------
# Runtime state (in memory; last_run / backoff are also persisted in the job)
# ---------------------------------------------------------------------------


@dataclass
class _Runtime:
    n_since: int = 0
    last_run: float = 0.0
    fail_until: float = 0.0
    last_error: str | None = None
    in_flight: dict | None = None
    polling: bool = False


_RT: dict[tuple[str, str], _Runtime] = {}
_RT_LOCK = threading.RLock()


def _runtime(job: dict, name: str) -> _Runtime:
    key = (job["id"], name)
    with _RT_LOCK:
        rt = _RT.get(key)
        if rt is None:
            saved = (job.get("hook_state") or {}).get(name) or {}
            rt = _Runtime(last_run=float(saved.get("last_run") or 0), fail_until=float(saved.get("fail_until") or 0),
                          last_error=saved.get("last_error"))  # fmt: skip
            _RT[key] = rt
        return rt


def forget(job_id: str, keep: set[str]) -> None:
    """Drop runtime state of hooks that were removed from a job."""
    with _RT_LOCK:
        for k in [k for k in _RT if k[0] == job_id and k[1] not in keep]:
            _RT.pop(k, None)


def runtime_view(job: dict) -> dict:
    """Per-hook runtime for the API: counter, last run, backoff, in-flight run."""
    out = {}
    for h in job.get("hooks") or []:
        rt = _runtime(job, h["name"])
        out[h["name"]] = {"n_since_last_run": rt.n_since, "last_run": rt.last_run or None,
                          "backoff_until": rt.fail_until or None, "last_error": rt.last_error,
                          "in_flight": rt.in_flight}  # fmt: skip
    return out


def _prefect_in_flight() -> int:
    with _RT_LOCK:
        return sum(1 for rt in _RT.values() if rt.in_flight and rt.in_flight.get("type") == "prefect")


# ---------------------------------------------------------------------------
# Triggering
# ---------------------------------------------------------------------------


def _handle_event(ev: LabelEvent) -> None:
    wants_done_check = []
    for job in _hooked_jobs():
        if not event_matches(job, ev):
            continue
        for hook in job["hooks"]:
            trig = hook["trigger"]
            rt = _runtime(job, hook["name"])
            if trig.get("every_n"):
                with _RT_LOCK:
                    rt.n_since += 1
                    ready = rt.n_since >= trig["every_n"]
                if ready:
                    _try_start(job, hook, rt, ev, "every_n")
            if trig.get("on_done"):
                wants_done_check.append(job["id"])
    for job_id in dict.fromkeys(wants_done_check):  # completion is derived: look after the write
        job = jobs.read_job(job_id)
        if job is not None:
            with datasets.using(job["dataset"]):
                jobs.job_with_progress(job)  # flips to done -> jobs._set_status -> job_done()


def _handle_done(job_id: str) -> None:
    job = jobs.read_job(job_id)
    if job is None:
        return
    for hook in job.get("hooks") or []:
        if hook["trigger"].get("on_done"):
            _try_start(job, hook, _runtime(job, hook["name"]), None, "on_done", force=True)


def _try_start(job: dict, hook: dict, rt: _Runtime, ev: LabelEvent | None, trigger: str, *, force: bool = False) -> bool:
    """Start the hook's action if the bounds allow. ``force`` (manual run / on_done) skips the debounce."""
    now = time.time()
    typ = hook["action"]["type"]
    stale = None
    poll = start = False
    with _RT_LOCK:
        fl = rt.in_flight
        if fl is not None:
            if now - fl["started"] > INFLIGHT_STALE_S:
                rt.in_flight = None
                rt.fail_until = now + BACKOFF_S
                rt.last_error = "run went stale (no completion report)"
                stale = {"event": "error", "hook": hook["name"], "error": rt.last_error}
            elif fl.get("type") == "prefect" and not rt.polling and now - fl.get("checked", fl["started"]) >= PREFECT_POLL_S:
                rt.polling = poll = True
        elif now < rt.fail_until:
            pass  # backing off after a failure
        elif not force and now - rt.last_run < hook["trigger"].get("min_interval_s", MIN_INTERVAL_DEFAULT_S):
            pass  # debounce
        elif typ == "prefect" and _prefect_in_flight() >= MAX_PREFECT_INFLIGHT:
            pass
        else:
            rt.in_flight = {"type": typ, "started": now, "trigger": trigger, "run_id": None}
            rt.last_run = now
            rt.n_since = 0
            start = True
    if stale:
        _persist(job["id"], hook["name"], rt, stale)
    if poll:
        _submit(_poll_prefect, job["id"], hook, rt)
    if start:
        _submit(_run, job["id"], hook, rt, ev, trigger)
    return start


def _submit(fn, *args) -> None:
    global _PENDING  # noqa: PLW0603
    with _PENDING_LOCK:
        _PENDING += 1

    def wrapped():
        global _PENDING  # noqa: PLW0603
        try:
            fn(*args)
        except Exception:  # noqa: BLE001
            _STATS["errors"] += 1
            LOGGER.exception("hooks: action crashed")
        finally:
            with _PENDING_LOCK:
                _PENDING -= 1

    assert _EXEC is not None
    _EXEC.submit(wrapped)


def _persist(job_id: str, name: str, rt: _Runtime, entry: dict) -> None:
    with contextlib.suppress(Exception):
        job_edit.record(job_id, entry, state=(name, {"last_run": rt.last_run, "fail_until": rt.fail_until, "last_error": rt.last_error}))


def _jsonable(d: object) -> object:
    try:
        return json.loads(json.dumps(d, default=str))
    except (TypeError, ValueError):
        return str(d)


def _run(job_id: str, hook: dict, rt: _Runtime, ev: LabelEvent | None, trigger: str) -> None:
    name = hook["name"]
    act = hook["action"]
    entry: dict = {"event": "run", "hook": name, "trigger": trigger, "action": act.get("name") or act.get("deployment")}
    try:
        job = jobs.read_job(job_id)
        if job is None:
            raise RuntimeError("job vanished")
        with datasets.using(job["dataset"]):
            if act["type"] == "builtin":
                a = ACTIONS.get(act["name"])
                if a is None:
                    raise RuntimeError(f"action {act['name']!r} is not registered")
                res = a.fn(HookContext(job, hook, trigger, ev, dict(act.get("params") or {}))) or {}
                entry.update(ok=True, result=_jsonable(res))
                with _RT_LOCK:
                    rt.in_flight = None
                    rt.last_error = None
            else:
                run_id = _queue_prefect(job, hook, trigger)
                entry.update(ok=True, run_id=run_id)
                with _RT_LOCK:
                    if rt.in_flight is not None:
                        rt.in_flight.update(run_id=run_id, checked=time.time())
    except Exception as e:  # noqa: BLE001 - recorded, backed off, never propagated
        LOGGER.warning("hooks: %s on %s failed: %s", name, job_id, e)
        with _RT_LOCK:
            rt.in_flight = None
            rt.fail_until = time.time() + BACKOFF_S
            rt.last_error = f"{type(e).__name__}: {e}"[:300]
        entry.update(ok=False, error=rt.last_error)
    _persist(job_id, name, rt, entry)


def _queue_prefect(job: dict, hook: dict, trigger: str) -> str:
    from footy_track.labeller import prefect_client  # noqa: PLC0415

    act = hook["action"]
    params = {**(act.get("params") or {}), "job_id": job["id"], "hook": hook["name"], "trigger": trigger, "dataset": job["dataset"]}
    if os.environ.get("FOOTY_LABELLER_URL"):
        params["labeller_url"] = os.environ["FOOTY_LABELLER_URL"]
    return prefect_client.create_flow_run(params, deployment=act["deployment"], name=f"hook-{hook['name']}-{job['id']}")


def _poll_prefect(job_id: str, hook: dict, rt: _Runtime) -> None:
    from footy_track.labeller import prefect_client  # noqa: PLC0415

    entry = None
    try:
        with _RT_LOCK:
            fl = dict(rt.in_flight or {})
        if not fl.get("run_id"):
            return
        try:
            state = prefect_client.flow_run_state(fl["run_id"])
        except Exception as e:  # noqa: BLE001 - Prefect unreachable: keep waiting until the stale timeout
            LOGGER.info("hooks: could not poll run %s: %s", fl["run_id"], e)
            return
        with _RT_LOCK:
            if rt.in_flight is None:
                return
            rt.in_flight["checked"] = time.time()
            if state in _PREFECT_TERMINAL:
                rt.in_flight = None
                if state in _PREFECT_BAD:
                    rt.fail_until = time.time() + BACKOFF_S
                    rt.last_error = f"prefect run {fl['run_id']} ended {state}"
                    entry = {"event": "error", "hook": hook["name"], "error": rt.last_error}
                else:
                    rt.last_error = None
                    entry = {"event": "run_finished", "hook": hook["name"], "run_id": fl["run_id"], "state": state}
    finally:
        with _RT_LOCK:
            rt.polling = False
    if entry:
        _persist(job_id, hook["name"], rt, entry)


def run_now(job_id: str, name: str) -> dict:
    """Run one hook now (``POST /api/jobs/<id>/hooks/<name>/run``): skips the debounce, not the
    one-in-flight rule or the backoff. Returns once an in-process action has finished."""
    job = jobs.read_job(job_id)
    if job is None:
        raise jobs.JobError(404, f"no job {job_id!r}")
    hook = next((h for h in job.get("hooks") or [] if h["name"] == name), None)
    if hook is None:
        raise jobs.JobError(404, f"job has no hook {name!r}")
    if job["status"] not in jobs.LIVE and job["status"] != "done":
        raise jobs.JobError(409, f"job is {job['status']}")
    _ensure_started()
    rt = _runtime(job, name)
    before = len((jobs.read_job(job_id) or {}).get("hook_history", []))
    if not _try_start(job, hook, rt, None, "manual", force=True):
        raise jobs.JobError(409, "hook is in flight, backing off or at the prefect in-flight cap", **runtime_view(job).get(name, {}))
    if hook["action"]["type"] == "builtin":
        drain(30)
    fresh = jobs.read_job(job_id) or job
    return {"hook": name, "history": (fresh.get("hook_history") or [])[before:], "state": runtime_view(fresh).get(name)}


def _reset_for_tests() -> None:
    """Forget all runtime state (tests only)."""
    with _RT_LOCK:
        _RT.clear()
    _SCAN.clear()
    _STATS.update(emitted=0, dropped=0, errors=0)


def stats() -> dict:
    return {**_STATS, "queued": _Q.qsize(), "prefect_in_flight": _prefect_in_flight()}


__all__ = [
    "ACTIONS", "HookContext", "LabelEvent", "drain", "emit", "event_matches", "job_done", "register_action",
    "run_now", "runtime_view", "validate_hooks",
]  # fmt: skip
