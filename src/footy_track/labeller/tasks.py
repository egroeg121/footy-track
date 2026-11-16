"""Shared labelling task board + server info (``/api/tasks``, ``/api/server_info``).

The task list is one JSON file shared by every labeller server (calib :8011,
norwich :8012, ...). Sessions edit it with ``python -m
footy_track.scripts.labelling_tasks`` (atomic write, stable ids); the servers
re-read it on every request, so no redeploy is needed to add or drop a task.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Request

from footy_track.labeller import jobs, task_done

DEFAULT_TASKS_FILE = "/mnt/storage/footy_data/labelling_tasks.json"
CALIB = task_done.CALIB

#: Written only when the tasks file does not exist yet.
SEED_TASKS: list[dict] = [
    {
        "id": "t-seed-bmth-seg030",
        "title": "Fix rejected Bournemouth penalty box (seg030 f205)",
        "why": "Machine fit was rejected; drag the box corners onto the real lines and save.",
        "url": f"{CALIB}/calibration?clip=arsenal_bournemouth_1st_seg030&frame=205&proposal=bmth_pre1",
        "device": "laptop",
        "priority": 1,
        "added": "2026-11-15",
        "owner": "george",
    },
    {
        "id": "t-seed-bmth-seg050",
        "title": "Fix rejected Bournemouth penalty box (seg050 f57)",
        "why": "Machine fit was rejected; drag the box corners onto the real lines and save.",
        "url": f"{CALIB}/calibration?clip=arsenal_bournemouth_1st_seg050&frame=57&proposal=bmth_pre1",
        "device": "laptop",
        "priority": 1,
        "added": "2026-11-15",
        "owner": "george",
    },
    {
        "id": "t-seed-mancity-recheck",
        "title": "Recheck Man City labels the PTZ rig disagrees with",
        "why": "5 frames, all arsenal_mancity_*: seg080 f87 (opens here), seg080 f52, seg031 f328, "
        "seg010 f375, seg010 f575.",
        "url": f"{CALIB}/calibration?clip=arsenal_mancity_seg080&frame=87",
        "device": "phone",
        "priority": 2,
        "added": "2026-11-15",
        "owner": "george",
    },
    {
        "id": "t-seed-bmth-nonbroadcast",
        "title": "Mark non-broadcast frames among failed Bournemouth",
        "why": "Swipe through failed fits; press Not broadcast on replays, close-ups, crowd and graphics.",
        "url": f"{CALIB}/calibration/review?run=bmth_pre1&status=failed",
        "device": "phone",
        "priority": 2,
        "count_url": f"{CALIB}/calibration/api/proposals?status=failed&run=bmth_pre1&limit=1",
        "count_label": "failed left",
        "added": "2026-11-15",
        "owner": "george",
    },
]

DEVICES = ("phone", "laptop", "either")


def tasks_path() -> Path:
    return Path(os.environ.get("FOOTY_TASKS_FILE") or DEFAULT_TASKS_FILE).expanduser()


def write_tasks_atomic(path: Path, tasks: list[dict]) -> None:
    """Write ``{"tasks": [...]}`` via temp file + rename in the same dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"tasks": tasks}, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _task_sort_key(t: dict) -> tuple:
    prio = t.get("priority")
    return (prio if isinstance(prio, (int, float)) else 2, str(t.get("added") or ""))


def load_tasks(path: Path | None = None, *, seed: bool = True) -> tuple[list[dict], str | None]:
    """Return ``(tasks, error)``. Never raises.

    Missing file: seed it (best effort) and return the seed. Malformed file:
    return ``([], message)`` and leave the file untouched.
    """
    path = path or tasks_path()
    if not path.exists():
        if not seed:
            return [], None
        tasks = [dict(t) for t in SEED_TASKS]
        with contextlib.suppress(OSError):  # read-only mount etc.: still show the seed
            write_tasks_atomic(path, tasks)
        return tasks, None
    try:
        data = json.loads(path.read_text())
        raw = data["tasks"] if isinstance(data, dict) else data
        if not isinstance(raw, list):
            raise ValueError("'tasks' must be a list")
        tasks = [t for t in raw if isinstance(t, dict) and t.get("title")]
    except (OSError, ValueError, KeyError) as e:
        return [], f"{path}: {type(e).__name__}: {e}"
    tasks.sort(key=_task_sort_key)
    return tasks, None


def new_task_id(existing: list[dict]) -> str:
    """Short stable id (``t-<base36 time>``), de-duplicated against ``existing``."""
    used = {t.get("id") for t in existing}
    n = int(time.time())
    while True:
        digits, x = "", n
        while x:
            x, r = divmod(x, 36)
            digits = "0123456789abcdefghijklmnopqrstuvwxyz"[r] + digits
        tid = f"t-{digits}"
        if tid not in used:
            return tid
        n += 1


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- server info


def _git(repo: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(  # noqa: S603
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def compute_git_info(repo: Path) -> dict:
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return {
        "sha": _git(repo, "rev-parse", "HEAD"),
        "short_sha": _git(repo, "rev-parse", "--short", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git(repo, "describe", "--always", "--dirty", "--tags"),
        "dirty": None if status is None else bool(status),
    }


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GIT_INFO = compute_git_info(_REPO_ROOT)  # once at startup: the checkout can't move under a running process
_STARTED = time.time()
_FOOTY_ENV_KEYS = (
    "FOOTY_CLIPS_DIR",
    "FOOTY_GT_MARKS_DIR",
    "FOOTY_CALIB_EXTRA_CLIPS_DIRS",
    "FOOTY_BALL_CHECK_LIST",
    "FOOTY_BALL_CANDIDATES_DIR",
    "FOOTY_TASKS_FILE",
    "FOOTY_LABELLER_DATASETS",
    "FOOTY_LABELLING_JOBS_DIR",
)

router = APIRouter()


@router.get("/api/tasks")
async def api_tasks(request: Request):
    tasks, error = load_tasks()
    # Open labelling jobs (POST /api/jobs) appear automatically, beside the hand-added tasks.
    job_tasks = await asyncio.to_thread(jobs.board_tasks)
    if job_tasks:
        tasks = sorted(tasks + job_tasks, key=_task_sort_key)
    # done_when (explicit or inferred, see task_done.py): done tasks sort last, old ones drop off.
    tasks = await task_done.annotate(tasks, tasks_path(), str(request.base_url))
    out = {"tasks": tasks, "file": str(tasks_path()), "n_jobs": len(job_tasks)}
    if error:
        out["error"] = error
    return out


@router.get("/api/server_info")
async def api_server_info():
    port = os.environ.get("FOOTY_SERVICE_PORT")
    return {
        "name": os.environ.get("FOOTY_SERVICE_NAME") or None,
        "port": int(port) if port and port.isdigit() else None,
        **_GIT_INFO,
        "repo": str(_REPO_ROOT),
        "started": datetime.fromtimestamp(_STARTED, UTC).isoformat(timespec="seconds"),
        "uptime_s": int(time.time() - _STARTED),
        "env": {k: os.environ.get(k) for k in _FOOTY_ENV_KEYS if os.environ.get(k)},
    }
