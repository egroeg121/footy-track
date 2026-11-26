"""Label-free rig search of one match (design ``calibration_match_flow.md`` 2.2 (a)+(b), experiment D arm E2).

    PYTHONPATH=src nice -n 19 python -m footy_track.scripts.ptz_rig_search \\
        --match arsenal_mancity --video-dir /desk/video/arsenal_mancity --segments-json .../segments.json \\
        --work-dir /desk/rigsearch/arsenal_mancity --out-dir /desk/rigsearch/arsenal_mancity/out [--grid emirates]

Steps: one frame per segment (the middle one), searched lines-only at the GENERIC prior
(``camera_rigs/generic_gantry_prior_v2.json``; no GT-derived rig anywhere); the accepted frames become single-frame units;
``fit_shared_multistart`` from the generic prior, ``rel_tol=1e-8``, ``n_iter=100`` on the finalists, k1 free; then, per unit,
the data cost of the pose re-optimised at the fitted rig (the per-segment fixed-rig cost of design 2.2 (d)).

Writes to ``--out-dir``: ``<stem>.json`` (the fitted rig, a rig JSON; ``<stem>`` defaults to ``<match>_ms``) and
``rig_search.json`` (inputs, per-start screening and final costs, chosen rig, per-unit costs, CPU seconds, ``contaminated``).
``rig_search.json`` is written LAST: it is the done marker. An aborted search (see
:func:`footy_track.calibration.rig_search.assess_search`) writes ``rig_search.json`` with ``status: "aborted"`` and exits 3.
The anchor frames are cached in ``--work-dir`` (a killed run resumes).
"""

# ruff: noqa: PLR0915
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from footy_track.calibration import ptz
from footy_track.calibration import ptz_temporal as pt
from footy_track.calibration import rig_search as rs
from footy_track.scripts import ptz_temporal as T


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _code_sha() -> str | None:
    if os.environ.get("PTZ_CODE_SHA"):
        return os.environ["PTZ_CODE_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 (a git archive export has no .git)
        return None


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--match", required=True, help="footy_data dir name, e.g. arsenal_mancity")
    ap.add_argument("--video-dir", required=True, type=Path, help="dir with the segment mp4s named as in segments.json")
    ap.add_argument("--segments-json", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path, help="resumable cache (masks, anchors)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prior", type=Path, default=rs.GENERIC_PRIOR, help="generic rig prior (never a GT-fitted rig)")
    ap.add_argument("--stem", default=None, help="rig file stem; default <match>_ms")
    ap.add_argument("--grid", choices=["emirates", "vitality"], default="emirates",
                    help="lines-only anchor grid of the ground's camera (vitality = the lower Bournemouth camera)")
    ap.add_argument("--pose-prior-from", type=Path, default=None,
                    help="rig JSON whose info.pose_prior (recovery band + side_cx_m variants) is copied into the fitted rig; "
                         "marked in rig_search.json (indirect input)")
    ap.add_argument("--segments", default="", help="comma-separated segment indices (default all); smoke tests")
    ap.add_argument("--rel-tol", type=float, default=rs.REL_TOL)
    ap.add_argument("--n-iter", type=int, default=rs.N_ITER)
    ap.add_argument("--no-abort", action="store_true", help="write the rig even when assess_search would abort")
    a = ap.parse_args(argv)

    t_wall, cpu0 = time.time(), cpu_seconds()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.work_dir.mkdir(parents=True, exist_ok=True)
    stem = a.stem or f"{a.match}_ms"
    prior_doc = rs.load_json(a.prior)
    prior = ptz.Rig.from_json(a.prior)
    seg_doc = json.loads(a.segments_json.read_text())
    want = {int(x) for x in a.segments.split(",") if x}
    segs = []
    for raw in seg_doc["segments"]:
        sg = {**raw, "path": a.video_dir / raw["file"]}
        if sg["path"].exists() and (not want or int(sg["index"]) in want):
            segs.append(sg)
    key = "bmth" if a.grid == "vitality" else a.match  # shared_anchor picks the Vitality grid by this key
    T.log(f"{len(segs)} segments of {a.match}; prior {a.prior.name} C {prior.C.tolist()} k1 {prior.k1}")

    def log(*x):
        T.log(f"[cpu {cpu_seconds() - cpu0:.0f}s]", *x)

    # --- 1. label-free anchors: the middle frame of every segment, lines-only search at the generic prior
    sp = a.work_dir / "anchors_generic.json"
    store = json.loads(sp.read_text()) if sp.exists() else {}
    n = 0
    for sg in segs:
        for t in T.sample_frames(sg, 1):
            if f"{sg['file']}:{t}" in store:
                continue
            T.shared_anchor(key, sg, t, prior, store, a.work_dir)
            n += 1
            if n % 5 == 0:
                _atomic_json(sp, store)
                log(f"  {len(store)} frames, {sum(v.get('ok', False) for v in store.values())} anchored")
    _atomic_json(sp, store)
    units, unit_seg = [], []  # one single-frame unit per anchored segment (uid = position; seg index kept for the report)
    for sg in segs:
        for u in T.shared_units(key, [sg], store, a.work_dir, 1):
            units.append(replace(u, uid=len(units)))
            unit_seg.append(int(sg["index"]))
    log(f"{len(units)} anchored units from {len(segs)} segments")

    contaminated = False  # nothing in this run comes from hand labels; see ``indirect_inputs``
    res: dict = {
        "schema": 1, "match": a.match, "status": "ok", "contaminated": contaminated, "stem": stem,
        "inputs": {
            "prior": str(a.prior), "prior_sha256": _sha256(a.prior), "prior_C": prior.C.tolist(), "prior_k1": prior.k1,
            "segments_json_sha256": _sha256(a.segments_json), "n_segments": len(segs), "grid": a.grid,
            "rel_tol": a.rel_tol, "n_iter": a.n_iter, "top_k": rs.TOP_K, "screen_stride": rs.SCREEN_STRIDE,
            "free": ["Cx", "Cy", "Cz", "roll", "k1"], "code_sha": _code_sha(), "segments_filter": sorted(want),
        },
        "indirect_inputs": [],
        "n_units": len(units), "n_anchored_ok": int(sum(v.get("ok", False) for v in store.values())),
    }
    pose_prior = None
    if a.pose_prior_from:
        pose_prior = rs.load_json(a.pose_prior_from).get("info", {}).get("pose_prior")
        res["inputs"]["pose_prior_from"] = str(a.pose_prior_from)
        res["indirect_inputs"].append({
            "name": "pose_prior", "file": a.pose_prior_from.name,
            "note": "tilt-vs-zoom band + side_cx_m, derived from solver outputs over the same match's segments (not hand labels), "
                    "but those solves used the GT-fitted rig: a second-order dependence. Used by the recovery pass of weak segments only."})

    if len(units) < 3:
        res.update(status="aborted", reasons=[f"only {len(units)} anchored units"])
        _atomic_json(a.out_dir / "rig_search.json", res)
        return 3

    # --- 2. E2: multistart shared fit from the generic prior, tight rule on the finalists, k1 free
    states_at = pt.Shared.from_rig(prior)
    free = ("Cx", "Cy", "Cz", "roll", "k1")
    t0 = time.time()
    shn, _, info = pt.fit_shared_multistart(
        units, prior, T.PITCH, states_at=states_at, offsets=pt.seed_offsets(), top_k=rs.TOP_K,
        screen_stride=rs.SCREEN_STRIDE, free=free, log=log, rel_tol=a.rel_tol, n_iter=a.n_iter)
    fit_s = time.time() - t0
    starts = info["starts"]
    finalists = [s for s in starts if "final_cost" in s]
    why = rs.assess_search(len(units), len(segs), [s["final_C"] for s in finalists])
    res.update(
        C=shn.C.tolist(), roll_deg=float(np.degrees(shn.roll)), roll=float(shn.roll), k1=float(shn.k1),
        cost=info["stage2"]["cost"], iters=info["stage2"].get("iters"), n_line_frames=info.get("n_line_frames"),
        finalist_spread_m=rs.finalist_spread([s["final_C"] for s in finalists]), best_start=info["best_start"],
        starts=starts, fit_wall_s=round(fit_s, 1))
    if why and not a.no_abort:
        res.update(status="aborted", reasons=why, cpu_s=round(cpu_seconds() - cpu0, 1))
        _atomic_json(a.out_dir / "rig_search.json", res)
        log("ABORTED:", why)
        return 3

    # --- 3. per-unit fixed-rig cost (design 2.2 (d)): poses re-optimised at the fitted rig, rig held
    fitted = ptz.Rig(shn.C, shn.roll, float(shn.k1), prior.sigma_C, prior.pitch, prior.info)
    unit_costs = []
    for u, si in zip(units, unit_seg, strict=True):
        us = replace(u, states=pt.reseed_states(u.states, states_at, pt.Shared.from_rig(fitted)))
        _, _, inf = pt.fit_shared([us], fitted, T.PITCH, free=(), n_iter=15)
        unit_costs.append({"segment": si, "frame": int(u.frames[0]), "cost": float(inf["stage2"]["cost"]),
                           "n_line_frames": inf.get("n_line_frames")})
    res["unit_costs"] = unit_costs
    c = np.array([u["cost"] for u in unit_costs])
    res["unit_cost_summary"] = {"median": float(np.median(c)), "p90": float(np.percentile(c, 90)), "max": float(c.max())}

    doc = rs.rig_document(shn.C, shn.roll, shn.k1, prior_doc, pose_prior=pose_prior,
                          source=f"{a.match} label-free multistart shared fit (E2) from {a.prior.name}; no GT-derived input")
    rig_path = a.out_dir / f"{stem}.json"
    _atomic_json(rig_path, doc)
    res.update(rig_json=rig_path.name, rig_hash=rs.rig_hash(doc), rig_tag=rs.rig_tag(stem, doc), rig_doc=doc,
               cpu_s=round(cpu_seconds() - cpu0, 1), wall_s=round(time.time() - t_wall, 1))
    _atomic_json(a.out_dir / "rig_search.json", res)  # last: the done marker
    log(f"rig {res['rig_tag']}: C {np.round(shn.C, 2).tolist()} roll {res['roll_deg']:.3f} deg k1 {shn.k1:.4f}, "
        f"{res['cpu_s']:.0f} CPU-s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
