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


def _machine_video_src(clip: str) -> str:
    if clip == "arsenal_example":
        return f"{FD}/arsenal_mancity_example_video.mp4"
    kind = clip.replace("arsenal_mancity_", "")
    return f"{FD}/clips/arsenal_mancity_20250925_{kind}.mp4"


@flow(name="sam2-machine", persist_result=False)
def sam2_machine(run_name: str, spec: str, model: str = "small", variant: str = "crop512",
                 dtype: str = "fp16", clips: str = "", max_tracks: int = 0) -> dict:
    """SAM2 tracks from machine anchors (spec.json staged at <mini>/sam2_exp/machine/<spec>/spec.json)."""
    import json as _json
    log = get_run_logger()
    run_name, spec = valid(run_name), valid(spec)
    ensure_env(log)
    for d in ("code", "data/videos", f"machine/{spec}", f"runs/{run_name}"):
        (EXP / d).mkdir(parents=True, exist_ok=True)
    log.info("code sync: %s", rsync(CODE_SRC, f"{EXP}/code/", ["--delete", "--exclude", "__pycache__"]))
    log.info("spec sync: %s", rsync(f"{MINI}:{MINI_STORE}/machine/{spec}/spec.json", f"{EXP}/machine/{spec}/spec.json"))
    spec_doc = _json.loads((EXP / "machine" / spec / "spec.json").read_text())
    want = [c for c in spec_doc["clips"] if spec_doc["clips"][c]["tracks"] and (not clips or c in clips.split(","))]
    for c in want:
        log.info("video %s: %s", c, rsync(f"{MINI}:{_machine_video_src(c)}", f"{EXP}/data/videos/{c}.mp4", ["-L"]))
    out = EXP / "runs" / run_name
    cmd = [str(EXP / "venv" / "bin" / "python"), "-u", str(EXP / "code" / "machine_tracks.py"),
           "--spec", str(EXP / "machine" / spec / "spec.json"), "--videos", str(EXP / "data" / "videos"),
           "--out", str(out), "--ckpt", str(EXP / "ckpt"), "--model", model, "--variant", variant, "--dtype", dtype]
    if clips:
        cmd += ["--clips", clips]
    if max_tracks:
        cmd += ["--max-tracks", str(max_tracks)]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1", "HF_HUB_OFFLINE": "1"}
    rc, tail = run_logged(cmd, log, cwd=str(EXP / "code"), env=env, log_file=out / "run.log")
    if rc != 0:
        if rc < 0 or rc in (137, 143):
            raise SystemExit(rc)
        raise RuntimeError(f"machine_tracks.py exited {rc}:\n" + "\n".join(tail[-15:]))
    sh(["ssh", *SSH_OPTS, MINI, f"mkdir -p {shlex.quote(MINI_STORE + '/runs/' + run_name)}"])
    log.info("results pushed: %s", rsync(f"{out}/", f"{MINI}:{MINI_STORE}/runs/{run_name}/", ["--exclude", "*.tmp"]))
    return {"run_name": run_name, "flow_run_id": str(flow_run.id), "result_dir": f"{MINI_STORE}/runs/{run_name}"}


@flow(name="sam2-detect", persist_result=False)
def sam2_detect(clip: str = "arsenal_example") -> dict:
    """Run the project RT-DETR detector on one clip (machine_labels format) -> <mini>/sam2_exp/detect/<clip>/."""
    log = get_run_logger()
    clip = valid(clip)
    for d in ("code", "data/videos", f"detect/{clip}"):
        (EXP / d).mkdir(parents=True, exist_ok=True)
    log.info("code sync: %s", rsync(CODE_SRC, f"{EXP}/code/", ["--delete", "--exclude", "__pycache__"]))
    vid = EXP / "data" / "videos" / f"{clip}.mp4"
    log.info("video: %s", rsync(f"{MINI}:{_machine_video_src(clip)}", str(vid), ["-L"]))
    py = str(HOME / "footy-track-web" / "repo" / ".venv" / "bin" / "python")
    weights = str(HOME / "footy-track-web/repo/model_saves/detector/rtdetr_1920_v11/best.pt")
    out = EXP / "detect" / clip
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    rc, tail = run_logged([py, "-u", str(EXP / "code" / "detect_clip.py"), str(vid), "--weights", weights,
                           "--out", str(out), "--name", clip], log, cwd=str(EXP / "code"), env=env,
                          log_file=out / "run.log")
    if rc != 0:
        if rc < 0 or rc in (137, 143):
            raise SystemExit(rc)
        raise RuntimeError("detect_clip.py failed:\n" + "\n".join(tail[-15:]))
    sh(["ssh", *SSH_OPTS, MINI, f"mkdir -p {shlex.quote(MINI_STORE + '/detect/' + clip)}"])
    log.info("pushed: %s", rsync(f"{out}/", f"{MINI}:{MINI_STORE}/detect/{clip}/", ["--exclude", "*.tmp"]))
    return {"clip": clip, "flow_run_id": str(flow_run.id)}
