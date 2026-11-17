"""PTZ V3 temporal calibration of ONE broadcast segment mp4 (production run, no ground truth, no eval).

    python -m footy_track.scripts.ptz_temporal_segment \
        --video /.../arsenal_mancity_20250925_seg010.mp4 --clip arsenal_mancity_seg010 \
        --rig-json src/footy_track/calibration/camera_rigs/emirates.json --out-root /scratch/temporal

Thin wrapper over :mod:`footy_track.scripts.ptz_temporal` (the research driver): same ``prep`` -> ``refit_anchors`` ->
``motion`` -> ``segments`` -> ``solve_variant("full")`` chain as ``ptz_temporal solve``, but

* the video is given explicitly (the driver only looks in ``eval_clips_kf10``);
* the rig is read from ``--rig-json`` (the driver fits a leave-clip-out rig from GT, which needs the GT dir);
* only the ``full`` variant is solved (no hold / sm3 / sm10 / sl1 / est / nomotion ablations, no GT);
* the anchor step shrinks for short segments so every segment gets a few anchor attempts;
* the result files are written atomically, ``states_full.npy`` LAST: its presence means the segment is done.

Everything is cached (and therefore resumable) under ``<out-root>/<clip>/``: grey frames, masks, lines-only anchors
(written after every anchor), motion pairs. Re-running after a crash continues from those caches.

Output (small; the caches are large and stay where they were made):
``states_full.npy`` (N, 3) pan / tilt / log-f per frame (NaN where the segment has no main-camera shot or no solve),
``solve_full.json`` per motion-connected run, ``anchors_lines.json``, ``anchors_refit.json``, ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from footy_track.calibration import ptz
from footy_track.scripts import ptz_temporal as pte

#: the driver evaluates every EV_STEP-th frame; anchors are taken at multiples of ANCHOR_STEP (itself a multiple of it)
MIN_ANCHOR_STEP = 25


def anchor_step_for(n_frames: int, wanted: int) -> int:
    """``wanted`` (100 in the report), but small enough for >= ~3 anchor attempts in a short segment."""
    third = int(round(n_frames / 3 / pte.EV_STEP)) * pte.EV_STEP
    return int(max(pte.EV_STEP, min(wanted, max(MIN_ANCHOR_STEP, third))))


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 (a git archive export has no .git)
        return os.environ.get("PTZ_CODE_SHA")


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def run(video: Path, clip: str, rig_json: Path, out_root: Path, anchor_step: int = 100) -> dict:
    t0 = time.time()
    cpu0 = time.process_time()
    pte.ROOT = out_root
    pte.clip_video = lambda _clip: video  # noqa: E731
    pte.clip_gt = lambda _clip: []  # noqa: E731 (production: no labels)
    n = pte.n_frames(video)
    pte.ANCHOR_STEP = anchor_step_for(n, anchor_step)
    out = out_root / clip
    out.mkdir(parents=True, exist_ok=True)
    rig = ptz.Rig.from_json(rig_json)
    pte.log(f"{clip}: {n} frames, anchor step {pte.ANCHOR_STEP}, rig {rig_json}")

    cache, shots, shot_info, anchors = pte.prep(clip, rig)
    anchors = pte.refit_anchors(cache, anchors, rig)
    pairs = pte.motion(cache, shots, shot_info, anchors, rig)
    segs = pte.segments(shot_info, pairs)
    pte.log("motion-connected main-camera runs:", segs)
    states, infos = pte.solve_variant(cache, segs, pairs, anchors, rig, [], None, "full")

    valid = np.isfinite(states).all(axis=1)
    meta = {"clip": clip, "video": str(video), "frames": int(cache.n), "anchor_step": pte.ANCHOR_STEP,
            "rig_json": str(rig_json), "rig_C": [float(x) for x in rig.C], "code_sha": _git_sha(),
            "shots": shot_info, "runs": [list(map(int, s)) for s in segs],
            "calibrated_frames": int(valid.sum()), "calibrated_frac": float(valid.mean()) if cache.n else 0.0,
            "anchors_ok": int(sum(bool(a.get("ok")) for a in anchors.values())), "anchors_tried": len(anchors),
            "wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - cpu0, 1)}
    _atomic_write(out / "solve_full.json", lambda p: p.write_text(json.dumps(infos, default=float)))
    _atomic_write(out / "meta.json", lambda p: p.write_text(json.dumps(meta, indent=1, default=float)))

    def _save(p: Path) -> None:
        with open(p, "wb") as f:  # np.save would append .npy to a name that does not end in it
            np.save(f, states)

    _atomic_write(out / "states_full.npy", _save)  # last: the completion marker
    pte.log(f"done {clip}: {meta['calibrated_frames']}/{meta['frames']} frames calibrated, "
            f"{meta['wall_s']:.0f}s wall, {meta['cpu_s']:.0f}s cpu")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--clip", required=True, help="output dir name under --out-root, e.g. arsenal_mancity_seg010")
    ap.add_argument("--rig-json", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=pte.ROOT)
    ap.add_argument("--anchor-step", type=int, default=100)
    a = ap.parse_args()
    run(a.video, a.clip, a.rig_json, a.out_root, a.anchor_step)


if __name__ == "__main__":
    main()
