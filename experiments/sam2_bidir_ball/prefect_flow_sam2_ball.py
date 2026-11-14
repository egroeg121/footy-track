"""SAM2.1 bidirectional ball propagation experiment as a GPU Prefect job.

Experiment code: footy-track branch ``exp/sam2-bidir-ball``, ``experiments/sam2_bidir_ball/``,
snapshotted to the mini at ``/mnt/storage/footy/sam2_exp/code`` (``CODE_SRC``), which this flow
rsyncs to the desktop ``~/sam2-exp/code`` on every run (so the run always uses the staged
snapshot, never a stale desktop copy). Clips and GT marks are pulled from the mini
(``/mnt/storage/footy_data``) with rsync (size/mtime skip, so only the first run moves bytes).
Results: desktop ``~/sam2-exp/runs/<run_name>/`` then rsynced to the mini
``/mnt/storage/footy/sam2_exp/runs/<run_name>/``.

Idempotent: ``sam2_tracks.py`` skips finished (run, anchor, direction) tracks in an existing
output file, so a crash-rerun resumes. Stdlib + prefect only (+ footy_common).
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from prefect import flow, get_run_logger
from prefect.runtime import flow_run

from footy_common import MINI, SSH_OPTS, UV, rsync, run_logged, sh

HOME = Path(os.environ.get("HOME", "/home/george"))
EXP = HOME / "sam2-exp"
MINI_STORE = "/mnt/storage/footy/sam2_exp"
CODE_SRC = f"{MINI}:{MINI_STORE}/code/"
FD = "/mnt/storage/footy_data"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_{m}.pt"
CKPTS = ["tiny", "small", "base_plus"]
VIDEO_SRC = {"arsenal_example": f"{FD}/arsenal_mancity_example_video.mp4"}  # verified identical (md5) alias


def valid(name: str) -> str:
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.,=-]{0,80}", name or ""):
        raise ValueError(f"bad name {name!r}")
    return name


def ensure_env(log) -> None:
    """One-time env: venv with torch 2.8 + sam2 (PyPI) + checkpoints. No-op once present."""
    venv_py = EXP / "venv" / "bin" / "python"
    if not venv_py.exists():
        log.info("creating venv")
        sh([UV, "venv", "--python", "3.12", str(EXP / "venv")])
        sh([UV, "pip", "install", "--python", str(venv_py), "torch==2.8.0", "torchvision==0.23.0", "numpy",
            "pillow", "opencv-python-headless", "hydra-core", "iopath", "tqdm", "setuptools", "wheel"])
    if subprocess.run([str(venv_py), "-c", "import sam2"], capture_output=True).returncode:
        env = {**os.environ, "SAM2_BUILD_CUDA": "0"}
        subprocess.run([UV, "pip", "install", "--python", str(venv_py), "--no-build-isolation", "sam2"],
                       check=True, env=env)
    (EXP / "ckpt").mkdir(parents=True, exist_ok=True)
    for m in CKPTS:
        dst = EXP / "ckpt" / f"sam2.1_hiera_{m}.pt"
        if not dst.exists():
            log.info("downloading %s", dst.name)
            subprocess.run(["curl", "-sSfL", "-o", str(dst) + ".tmp", CKPT_URL.format(m=m)], check=True)
            os.replace(str(dst) + ".tmp", dst)


@flow(name="sam2-ball", persist_result=False)
def sam2_ball(run_name: str, clips: str = "arsenal_example", models: str = "small",
              variants: str = "full,crop512", ns: str = "10,25,50", dtype: str = "fp16",
              prompt: str = "box", hole_area: int = -1, max_anchors: int = 0, tag: str = "",
              score: bool = True) -> dict:
    """Run sam2_tracks.py on the desktop GPU for the given clips/models/variants."""
    log = get_run_logger()
    run_name = valid(run_name)
    clip_list = [valid(c) for c in clips.split(",") if c]
    ensure_env(log)
    for d in ("code", "data/videos", "data/gt", f"runs/{run_name}"):
        (EXP / d).mkdir(parents=True, exist_ok=True)
    log.info("code sync: %s", rsync(CODE_SRC, f"{EXP}/code/", ["--delete", "--exclude", "__pycache__"]))
    for c in clip_list:
        src = VIDEO_SRC.get(c, f"{FD}/ballcheck_clips/{c}.mp4")
        log.info("video %s: %s", c, rsync(f"{MINI}:{src}", f"{EXP}/data/videos/{c}.mp4", ["-L"]))
        log.info("gt %s: %s", c, rsync(f"{MINI}:{FD}/ball_gt_marks/{c}.jsonl", f"{EXP}/data/gt/{c}.jsonl"))
    out = EXP / "runs" / run_name
    cmd = [str(EXP / "venv" / "bin" / "python"), "-u", str(EXP / "code" / "sam2_tracks.py"),
           "--clips", ",".join(clip_list), "--data", str(EXP / "data"), "--out", str(out),
           "--ckpt", str(EXP / "ckpt"), "--models", models, "--variants", variants, "--ns", ns,
           "--dtype", dtype, "--prompt", prompt, "--max-anchors", str(max_anchors)]
    if hole_area >= 0:
        cmd += ["--hole-area", str(hole_area)]
    if tag:
        cmd += ["--tag", tag]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1", "HF_HUB_OFFLINE": "1"}
    rc, tail = run_logged(cmd, log, cwd=str(EXP / "code"), env=env, log_file=out / "run.log")
    if rc != 0:
        if rc < 0 or rc in (137, 143):
            raise SystemExit(rc)  # killed by signal -> Crashed -> gpu retry automation semantics
        raise RuntimeError(f"sam2_tracks.py exited {rc}:\n" + "\n".join(tail[-15:]))
    # results back to the mini (small JSON)
    sh(["ssh", *SSH_OPTS, MINI, f"mkdir -p {shlex.quote(MINI_STORE + '/runs/' + run_name)}"])
    back = rsync(f"{out}/", f"{MINI}:{MINI_STORE}/runs/{run_name}/", ["--exclude", "*.tmp"])
    log.info("results pushed: %s", back)
    summary = {"run_name": run_name, "flow_run_id": str(flow_run.id), "clips": clip_list,
               "result_dir": f"{MINI_STORE}/runs/{run_name}", "tail": tail[-5:]}
    from prefect.variables import Variable
    Variable.set(f"sam2-ball-{flow_run.id}", {k: v for k, v in summary.items() if k != "tail"}, overwrite=True)
    return summary
