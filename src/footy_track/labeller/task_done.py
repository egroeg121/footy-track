"""Self-completing tasks: evaluate a task's ``done_when`` for ``GET /api/tasks``.

A task may carry ``done_when`` (one of the shapes below). The server evaluates it per
request (cached ``CACHE_TTL`` seconds) and reports ``done`` / ``progress``; a task
without one (and with nothing inferable, see :func:`effective_done_when`) never
completes by itself and is removed with the CLI ``done`` command as before.

Shapes (all optional keys in brackets)::

    {"gt": {"clip": C, "frame": N [, "since": "added" | "YYYY-MM-DD"]}}
        calibration GT exists for that frame (clip aliases resolve through
        ``canonical_clip_id``, exactly like the calibration routes). ``since`` ignores
        GT saved before that date, for "recheck" tasks whose frame already had GT.
    {"proposal": {"clip": C, "run": R, "frame": N}, "status_in": ["accepted", "edited"]}
        that proposal's status is in ``status_in`` (default accepted + edited).
    {"frames": [[C, N], ...] [, "since": ...]}
        every frame has GT, or a ``not_broadcast`` / ``too_few_points`` frame-type mark;
        progress ``{"done": k, "total": n}``.
    {"count_zero": true}
        the task's own ``count_url`` returns 0 (same parsing as the page's chip).

Any shape may add ``"dataset": NAME`` (default: the default dataset, which is what a
link without ``?dataset=`` opens). Failures to evaluate mean "not done", never an error.

Done-ness is derived, not stored, except that the first time a task is seen done its
timestamp goes to a sidecar ``<tasks file>.done.json`` so done tasks can age off the
board after ``DONE_TTL_DAYS``. If the condition turns false again (GT deleted) the task
is active again and its timestamp is forgotten.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from footy_track.calibration.frame_type import read_frame_types
from footy_track.calibration.prelabel.proposals import read_proposals
from footy_track.labeller import calibration as cal
from footy_track.labeller import datasets
from footy_track.labeller.review import _gt_marks_dir

CACHE_TTL = 5.0
CALIB = "http://mac-mini-i7:8011"  # the one site; tasks.CALIB is this
DONE_TTL_DAYS = 2
COUNT_TIMEOUT_S = 2.0
#: Frame-type marks that settle a frame without GT. ``too_few_points`` is not written
#: on main yet; reading it here costs nothing and makes it count the day it lands.
SETTLED_FRAME_TYPES = ("not_broadcast", "too_few_points")
DEFAULT_PROPOSAL_STATUSES = ("accepted", "edited")

_cache: dict[tuple, tuple[float, tuple[bool, dict | None]]] = {}


# ----------------------------------------------------------------- inference


def effective_done_when(task: dict) -> tuple[dict | None, bool]:
    """``(done_when, inferred)``. Explicit ``done_when`` wins; job tasks complete themselves
    already (they leave the board when the job closes). Otherwise infer from the task:
    a ``count_url`` -> ``count_zero``; else a ``/calibration?clip=X&frame=N`` url -> a GT
    check on that frame, only counting GT saved on/after the task's ``added`` date."""
    if isinstance(task.get("done_when"), dict) and task["done_when"]:
        return task["done_when"], False
    if task.get("job"):
        return None, False
    if task.get("count_url"):
        return {"count_zero": True}, True
    u = urlparse(task.get("url") or "")
    q = parse_qs(u.query)
    clip, frame = (q.get("clip") or [None])[0], (q.get("frame") or [None])[0]
    if u.path.rstrip("/") == "/calibration" and clip and frame and frame.lstrip("-").isdigit():
        dw: dict = {"gt": {"clip": clip, "frame": int(frame), "since": "added"}}
        if (q.get("dataset") or [None])[0]:
            dw["dataset"] = q["dataset"][0]
        return dw, True
    return None, False


# ---------------------------------------------------------------- evaluation


def _since(spec: dict, task: dict) -> str | None:
    s = spec.get("since")
    return (str(task.get("added") or "") or None) if s == "added" else (str(s) if s else None)


def _gt_ok(clip: str, frame: int, since: str | None) -> bool:
    try:
        rec = cal._read_gt(clip)["frames"].get(str(int(frame)))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if rec is None:
        return False
    ts = str(rec.get("ts") or "")[:10]
    return not (since and ts and ts < since)  # no timestamp: accept


def _settled(clip: str, frame: int, since: str | None) -> bool:
    if _gt_ok(clip, frame, since):
        return True
    try:
        types = read_frame_types(_gt_marks_dir(), clip).get("frames", {})
    except (OSError, ValueError):
        return False
    return ((types.get(str(int(frame))) or {}).get("type")) in SETTLED_FRAME_TYPES


def pick_count(j, path: str | None = None) -> int | None:
    """Same rule as ``pickCount`` in ``web/main.html``: ``count_path`` else total/count/remaining."""
    if isinstance(j, (int, float)) and not isinstance(j, bool):
        return int(j)
    if path:
        for k in path.split("."):
            j = j.get(k) if isinstance(j, dict) else None
        return int(j) if isinstance(j, (int, float)) and not isinstance(j, bool) else None
    if isinstance(j, dict):
        for k in ("total", "count", "remaining"):
            if isinstance(j.get(k), (int, float)) and not isinstance(j.get(k), bool):
                return int(j[k])
    return None


def fetch_json(url: str):  # patched in tests
    with urllib.request.urlopen(url, timeout=COUNT_TIMEOUT_S) as r:  # noqa: S310 - task urls are operator-written
        return json.loads(r.read())


def resolve_url(url: str, base: str | None) -> str:
    """Count URLs are written for browsers (``http://mac-mini-i7:8011/...`` or relative). Anything on
    this server's own path space is fetched through the request's own base url instead."""
    u = urlparse(url)
    if base and (not u.netloc or u.netloc == urlparse(CALIB).netloc):
        return base.rstrip("/") + u.path + (f"?{u.query}" if u.query else "")
    return url


def _evaluate(task: dict, dw: dict, base: str | None) -> tuple[bool, dict | None]:
    since = _since(dw, task)
    if "gt" in dw:
        g = dw["gt"]
        return _gt_ok(g["clip"], g["frame"], _since(g, task) or since), None
    if "proposal" in dw:
        p = dw["proposal"]
        want = set(dw.get("status_in") or DEFAULT_PROPOSAL_STATUSES)
        rec = read_proposals(_gt_marks_dir(), p["clip"])["runs"].get(p["run"], {}).get("frames", {}).get(str(int(p["frame"])))
        return bool(rec) and rec.get("status") in want, None
    if "frames" in dw:
        frames = [(c, int(n)) for c, n in dw["frames"]]
        k = sum(_settled(c, n, since) for c, n in frames)
        return bool(frames) and k == len(frames), {"done": k, "total": len(frames)}
    if dw.get("count_zero"):
        if not task.get("count_url"):
            return False, None
        n = pick_count(fetch_json(resolve_url(task["count_url"], base)), task.get("count_path"))
        return n == 0, (None if n is None else {"remaining": n})
    return False, None


def evaluate(task: dict, dw: dict, base: str | None = None) -> tuple[bool, dict | None]:
    """``(done, progress)``, cached ``CACHE_TTL`` s. Never raises."""
    key = (dw.get("dataset") or "", json.dumps(dw, sort_keys=True, default=str), task.get("count_url") or "",
           task.get("count_path") or "", str(task.get("added") or ""))
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    try:
        with datasets.using(dw.get("dataset")):
            res = _evaluate(task, dw, base)
    except Exception:  # noqa: BLE001 - a bad spec / unreachable count url means "not done", not a 500
        res = (False, None)
    if len(_cache) > 512:
        _cache.clear()
    _cache[key] = (now, res)
    return res


# -------------------------------------------------------------- done sidecar


def sidecar_path(tasks_file: Path) -> Path:
    return tasks_file.with_name(tasks_file.name + ".done.json")


def _read_sidecar(path: Path) -> dict[str, str]:
    try:
        d = json.loads(path.read_text())
        return {str(k): str(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def now_utc() -> datetime:  # patched in tests
    return datetime.now(UTC)


async def annotate(tasks: list[dict], tasks_file: Path, base: str | None = None) -> list[dict]:
    """Return ``tasks`` with ``done`` (+ ``done_at``, ``progress``, effective ``done_when``)
    set; done tasks older than ``DONE_TTL_DAYS`` are dropped; active first, done last."""
    plans = [effective_done_when(t) for t in tasks]
    results = await asyncio.gather(
        *(asyncio.to_thread(evaluate, t, dw, base) if dw else _none() for t, (dw, _) in zip(tasks, plans, strict=True))
    )
    sidecar_file = sidecar_path(tasks_file)
    seen = _read_sidecar(sidecar_file)
    stamp = now_utc()
    dirty, active, done = False, [], []
    for t, (dw, inferred), (is_done, progress) in zip(tasks, plans, results, strict=True):
        out = dict(t)
        if dw:
            out["done_when"], out["done_when_inferred"] = dw, inferred
        out["done"] = bool(is_done)
        if progress:
            out["progress"] = progress
        tid = str(t.get("id") or "")
        if not is_done:
            if tid in seen:
                del seen[tid]
                dirty = True
            active.append(out)
            continue
        if tid not in seen:
            seen[tid] = stamp.isoformat(timespec="seconds")
            dirty = True
        out["done_at"] = seen[tid]
        try:
            age = stamp - datetime.fromisoformat(seen[tid])
        except ValueError:
            age = timedelta(0)
        if age <= timedelta(days=DONE_TTL_DAYS):
            done.append(out)
    if dirty:
        keep = {k: v for k, v in seen.items() if k in {str(t.get("id")) for t in tasks}}
        with contextlib.suppress(OSError):  # read-only mount: done-ness still reported, it just never ages off
            _write_json_atomic(sidecar_file, keep)
    return active + done


async def _none() -> tuple[bool, None]:
    return False, None
