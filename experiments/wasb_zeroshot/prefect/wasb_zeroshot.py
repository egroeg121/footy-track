"""WASB-SBDT zero-shot ball-detection experiment as a GPU Prefect job.

Experiment code: footy-track branch ``exp/wasb-zeroshot``, ``experiments/wasb_zeroshot/``,
snapshotted to the mini at ``/mnt/storage/exp/wasb/code`` (``CODE_SRC``). Inputs (evalset.json,
33 clips, the WASB hrnet source + soccer checkpoint) live on the mini at
``/mnt/storage/exp/wasb/inputs`` and are rsynced to the desktop ``~/exp/wasb/`` on every run
(size/mtime skip, so only the first run moves bytes). Python env: the desktop's shared footy venv
(``~/code/jobs-cache/footy-venv``: torch 2.8 cu128 + opencv + numpy), no install step.
Results: desktop ``~/exp/wasb/runs/<run_name>/`` then rsynced to the mini
``/mnt/storage/exp/wasb/runs/<run_name>/``.

Idempotent: run_wasb.py skips clips whose pred file exists, so a crash-rerun resumes.
Stdlib + prefect only (+ footy_common).
"""
from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from prefect import flow, get_run_logger
from prefect.runtime import flow_run

from footy_common import CODE_CACHE, MINI, SSH_OPTS, rsync, run_logged, sh

HOME = Path(os.environ.get("HOME", "/home/george"))
EXP = HOME / "exp" / "wasb"
MINI_STORE = "/mnt/storage/exp/wasb"
PY = CODE_CACHE / "footy-venv" / "bin" / "python"


@flow(name="wasb-zeroshot", persist_result=False)
def wasb_zeroshot(run_name: str, modes: str = "", clips: str = "", max_frames: int = 0) -> dict:
    """Run experiments/wasb_zeroshot/run_wasb.py on the desktop GPU."""
    log = get_run_logger()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}", run_name or ""):
        raise ValueError(f"bad run_name {run_name!r}")
    if not PY.exists():
        raise RuntimeError(f"shared venv missing: {PY}")
    for d in ("code", "inputs", f"runs/{run_name}"):
        (EXP / d).mkdir(parents=True, exist_ok=True)
    log.info("code: %s", rsync(f"{MINI}:{MINI_STORE}/code/", f"{EXP}/code/", ["--delete", "--exclude", "__pycache__"]))
    log.info("inputs: %s", rsync(f"{MINI}:{MINI_STORE}/inputs/", f"{EXP}/inputs/"))
    out = EXP / "runs" / run_name
    cmd = [str(PY), "-u", str(EXP / "code" / "run_wasb.py"), "--inputs", str(EXP / "inputs"), "--out", str(out)]
    if modes:
        cmd += ["--modes", modes]
    if clips:
        cmd += ["--clips", clips]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    rc, tail = run_logged(cmd, log, cwd=str(EXP / "code"), env=env, log_file=out / "run.log")
    if rc != 0:
        if rc < 0 or rc in (137, 143):
            raise SystemExit(rc)  # killed by signal -> Crashed -> gpu retry automation semantics
        raise RuntimeError(f"run_wasb.py exited {rc}:\n" + "\n".join(tail[-15:]))
    sh(["ssh", *SSH_OPTS, MINI, f"mkdir -p {shlex.quote(MINI_STORE + '/runs/' + run_name)}"])
    back = rsync(f"{out}/", f"{MINI}:{MINI_STORE}/runs/{run_name}/", ["--exclude", "*.tmp"])
    log.info("results pushed: %s", back)
    summary = {"run_name": run_name, "flow_run_id": str(flow_run.id), "result_dir": f"{MINI_STORE}/runs/{run_name}"}
    from prefect.variables import Variable
    Variable.set(f"wasb-zeroshot-{flow_run.id}", summary, overwrite=True)
    return summary
