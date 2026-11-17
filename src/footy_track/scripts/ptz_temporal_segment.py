"""PTZ V3 temporal calibration of ONE broadcast segment mp4 (production run: no ground truth, no eval).

    python -m footy_track.scripts.ptz_temporal_segment \
        --video /.../arsenal_mancity_20250925_seg010.mp4 --clip arsenal_mancity_seg010 \
        --rig-json src/footy_track/calibration/camera_rigs/emirates.json --out-root /scratch/temporal

Thin wrapper over the research driver :mod:`footy_track.scripts.ptz_temporal` (``prep`` + ``solve --variants full`` with
``--video`` / ``--rig``): the same ``load_clip`` -> ``solve_variant("full")`` -> ``export_calib`` chain, in-process, plus

* no GT lookups (``clip_gt`` returns nothing: the GT directory does not exist on the desktop);
* the anchor step shrinks for short segments so every segment gets a few anchor attempts;
* ``line_f1_full.json``: line-evidence F1 of the SOLVED camera on every evidence frame, and the lines-only search F1 /
  acceptance of every anchor, so a consumer can tell confident frames from uncertain ones without GT;
* ``meta.json`` and ``solve_full.json``; every file is written atomically.

Work directory: ``<out-root>/<clip>@<rig stem>/`` (the driver's convention: anchors and motion depend on the rig).
Output (small): ``calib_full.npz`` (frame, pan, tilt, logf, H (N,3,3) normalised undistorted image -> pitch m, rig),
``states_full.npy``, ``line_f1_full.json``, ``solve_full.json``, ``meta.json``, ``anchors_lines.json``,
``anchors_refit.json``, ``pairs_g1_5.pkl``. **``calib_full.npz`` is written LAST: its presence means the segment is done.**
Caches (grey frames, masks, anchors, pairs) make a re-run resume. k1 stays fixed at the rig value (it is poorly
constrained by the lines; see reports/ptz_v3.md).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from footy_track.calibration import ptz, ptz_recover as pr
from footy_track.calibration import ptz_temporal as pt
from footy_track.scripts import ptz_temporal as pte

MIN_ANCHOR_STEP = 25
RECOVER_BELOW = 0.5  # run the recovery pass when the standard pass accepts less than this fraction of the frames
REC_FRAMES = 12  # lines-only candidate frames per segment in recovery stage A (stage B adds finders and doubles them)
F1_ACCEPT = 0.5  # the driver's acceptance threshold for an evidence frame (cmd_eval coverage)


def anchor_step_for(n_frames: int, wanted: int) -> int:
    """``wanted`` (100 in the report), but small enough for >= ~3 anchor attempts in a short segment (a multiple of EV_STEP)."""
    third = int(round(n_frames / 3 / pte.EV_STEP)) * pte.EV_STEP
    return int(max(pte.EV_STEP, min(wanted, max(MIN_ANCHOR_STEP, third))))


def _git_sha() -> str | None:
    if os.environ.get("PTZ_CODE_SHA"):
        return os.environ["PTZ_CODE_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 (a git archive export has no .git)
        return None


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _ranges(mask: np.ndarray) -> list[list[int]]:
    """Runs of True as [start, end] inclusive."""
    out, a = [], None
    for t, m in enumerate(mask):
        if m and a is None:
            a = t
        if not m and a is not None:
            out.append([a, t - 1])
            a = None
    if a is not None:
        out.append([a, len(mask) - 1])
    return out


def line_f1_report(cache, states, shared, anchors, segs, prior: dict | None = None) -> dict:
    """F1 of the solved camera against the line evidence of every evidence frame (every 5th frame), the anchors' search F1,
    and ``accepted_ranges``: frames whose two neighbouring evidence frames both reach ``F1_ACCEPT`` (the driver's coverage rule).
    With a ``prior`` (recovery pass) an evidence frame also has to sit on the rig's pose band (:func:`pr.in_prior`)."""
    ev = list(cache.ev_frames)
    f1 = pte.line_f1(cache, states, shared, ev)
    good = {t: f1.get(t, -1.0) >= F1_ACCEPT and (prior is None or pr.in_prior(prior, *states[t])) for t in ev}
    acc = np.zeros(cache.n, bool)
    for t in range(cache.n):
        prev = max((e for e in ev if e <= t), default=None)
        nxt = min((e for e in ev if e >= t), default=None)
        acc[t] = all(good[e] for e in (prev, nxt) if e is not None) and bool(np.all(np.isfinite(states[t])))
    anc = {t: {"ok": bool(a.get("ok")), "search_f1": a.get("f1"), "solved_f1": f1.get(int(t))} for t, a in anchors.items()}
    vals = list(f1.values())
    return {"threshold": F1_ACCEPT, "evidence_f1": {str(t): round(v, 4) for t, v in f1.items()}, "anchors": anc,
            "accepted_ranges": _ranges(acc), "accepted_frac_of_frames": float(acc.mean()) if cache.n else 0.0,
            "median_evidence_f1": float(np.median(vals)) if vals else None,
            "main_camera_runs": [list(map(int, s)) for s in segs]}


def candidate_frames(cache, runs: list[tuple[int, int]], n: int) -> list[int]:
    """``n`` evidence frames spread evenly over the runs (all of them when there are fewer)."""
    ev = [t for t in cache.ev_frames if any(a <= t <= b for a, b in runs)]
    if len(ev) <= n:
        return ev
    return [ev[i] for i in np.unique(np.linspace(0, len(ev) - 1, n).round().astype(int))]


def usable_candidates(cands: dict, frames: list[int]) -> dict[int, dict]:
    """Best usable record per frame: on the pose band and a relaxed F1 with C at the rig (what the joint solve uses)."""
    out = {}
    for t in frames:
        recs = [r for r in (cands.get(str(t)) or {}).values() if r and r["prior_ok"] and r["f1_fixed"] >= pr.F1_CAND]
        if recs:
            out[t] = max(recs, key=lambda r: r["f1_fixed"])
    return out


def recover_pass(cache, rig, shot_info, out: Path) -> dict:
    """Recovery of a segment the standard pass left uncalibrated (see :mod:`ptz_recover`). Stage A: trim non-pitch tails,
    multi-start prior-band lines-only candidates on ``REC_FRAMES`` frames, motion confirmation, joint solve. Stage B
    (only when stage A still accepts < ``RECOVER_BELOW``): finder candidates and twice the frames. Returns the better of
    the stages: ``{"states", "infos", "segs", "anchors", "f1rep", "stages", "runs"}``."""
    prior = pr.pose_prior(rig)
    pitch = pte.PITCH
    sh = pt.Shared.from_rig(rig)
    pm = {t: pt.pmain_of(cache, t) for t in cache.ev_frames}
    base_runs = [(x["start"], x["end"]) for x in shot_info if x["p_main"] >= 0.5]
    runs = pr.trim_runs(base_runs, cache, pm)
    pte.log(f"recovery: main shots {base_runs} -> trimmed runs {runs}")
    cp = out / "anchors_rec.json"
    cands = json.loads(cp.read_text()) if cp.exists() else {}
    best, stages = None, []
    for stage, (n_fr, use_finders) in enumerate(((REC_FRAMES, False), (2 * REC_FRAMES, True))):
        frames = candidate_frames(cache, runs, n_fr)
        for t in frames:
            rec = cands.setdefault(str(t), {})
            ev = cache.evidence(t)
            if ev is None:
                continue
            if "lines" not in rec:
                rec["lines"] = pr.lines_candidate(ev, rig, pitch, prior)
                cp.write_text(json.dumps(cands, default=float))
            if use_finders and "finder" not in rec:
                rec["finder"] = pr.finder_anchor(cache, t, rig, pitch, prior)
                cp.write_text(json.dumps(cands, default=float))
        use = usable_candidates(cands, frames)
        pte.log(f"recovery stage {'AB'[stage]}: {len(use)}/{len(frames)} usable candidates "
                f"{ {t: (round(r['f1'], 2), round(r['f1_fixed'], 2), r['src']) for t, r in use.items()} }")
        info = {"stage": "AB"[stage], "frames": len(frames), "usable": len(use)}
        if not use:
            stages.append(info | {"accepted_frac": 0.0})
            continue
        # motion on the trimmed runs with the candidates' zoom as the nominal focal length
        fnom = pte.f_nominal(cache, {str(t): {"ok": True, "logf": r["logf"]} for t, r in use.items()})
        pairs = []
        for a, b in runs:
            ps, st = pt.compute_motion(cache, a, b, fnom[a:b + 1], rig.k1, gaps=(1, 5), log=pte.log)
            pairs += ps
        pairs, weak = pt.reject_weak_pairs(pairs)
        segs = pte.segments([{"start": a, "end": b, "p_main": 1.0} for a, b in runs], pairs)
        pairs_by = {(p.i, p.j): p for p in pairs}
        states = {t: np.array(r["refit"]["fixed"]) for t, r in use.items()}
        strong = {t: r["f1"] >= pr.F1_STRONG and r["f1_fixed"] >= 0.6 for t, r in use.items()}
        kept, partners = pr.confirm_anchors(states, pairs_by, sh, strong=strong)
        anchors = {str(t): {**r, "ok": t in kept} for t, r in use.items()}
        pte.log(f"recovery stage {'AB'[stage]}: confirmed {sorted(kept)} of {sorted(use)}; partners {partners}")
        st_arr, infos = pte.solve_variant(cache, segs, pairs, anchors, rig, [], None, "recover")
        f1rep = line_f1_report(cache, st_arr, sh, anchors, segs, prior)
        info |= {"confirmed": sorted(kept), "weak_pairs": weak, "accepted_frac": f1rep["accepted_frac_of_frames"]}
        stages.append(info)
        pte.log(f"recovery stage {'AB'[stage]}: accepted {100 * info['accepted_frac']:.0f}%")
        if best is None or info["accepted_frac"] > best["f1rep"]["accepted_frac_of_frames"]:
            best = {"states": st_arr, "infos": infos, "segs": segs, "anchors": anchors, "f1rep": f1rep, "runs": runs}
        if info["accepted_frac"] >= RECOVER_BELOW:
            break
    if best is not None:
        best["stages"] = stages
    return best or {"stages": stages, "runs": runs}


def run(video: Path, clip: str, rig_json: Path, out_root: Path, anchor_step: int = 100, recover: bool = True) -> dict:
    t0, cpu0 = time.time(), time.process_time()
    pte.ROOT = out_root
    pte.VIDEO_PATH = video
    pte.RIG_PATH = rig_json
    pte.WORK_SUFFIX = f"@{rig_json.stem}"
    pte.clip_gt = lambda _clip: []  # noqa: E731 (production: no labels, and no GT dir on the desktop)
    n = pte.n_frames(video)
    pte.ANCHOR_STEP = anchor_step_for(n, anchor_step)
    out = pte.wd(clip)
    out.mkdir(parents=True, exist_ok=True)
    pte.log(f"{clip}: {n} frames (container), anchor step {pte.ANCHOR_STEP}, rig {rig_json}, work dir {out}")

    rig, cache, shot_info, anchors, pairs, segs = pte.load_clip(clip)  # prep, refit anchors, motion, weak-pair guard
    pte.log("motion-connected main-camera runs:", segs)
    states, infos = pte.solve_variant(cache, segs, pairs, anchors, rig, [], None, "full")
    shared = pt.Shared.from_rig(rig)

    f1rep = line_f1_report(cache, states, shared, anchors, segs)
    recovery = None
    if recover and f1rep["accepted_frac_of_frames"] < RECOVER_BELOW:
        pte.log(f"standard pass accepts {100 * f1rep['accepted_frac_of_frames']:.0f}% of the frames: recovery pass")
        rec = recover_pass(cache, rig, shot_info, out)
        recovery = {"standard_accepted_frac": f1rep["accepted_frac_of_frames"], "stages": rec["stages"],
                    "trimmed_runs": [list(map(int, r)) for r in rec["runs"]], "used": False}
        if rec.get("f1rep") and rec["f1rep"]["accepted_frac_of_frames"] > f1rep["accepted_frac_of_frames"]:
            recovery["used"] = True
            recovery["standard_anchors"] = f1rep["anchors"]
            states, infos, segs, anchors = rec["states"], rec["infos"], rec["segs"], rec["anchors"]
            f1rep = rec["f1rep"]
    valid = np.isfinite(states).all(axis=1)
    meta = {"clip": clip, "video": str(video), "frames_decoded": int(cache.n), "frames_container": n,
            "anchor_step": pte.ANCHOR_STEP, "rig_json": str(rig_json), "rig_C": [float(x) for x in rig.C],
            "rig_k1": float(rig.k1), "code_sha": _git_sha(), "shots": shot_info, "runs": f1rep["main_camera_runs"],
            "calibrated_frames": int(valid.sum()), "calibrated_frac": float(valid.mean()) if cache.n else 0.0,
            "accepted_frac": f1rep["accepted_frac_of_frames"], "median_evidence_f1": f1rep["median_evidence_f1"],
            "anchors_ok": int(sum(bool(a.get("ok")) for a in anchors.values())), "anchors_tried": len(anchors),
            "recovery": recovery, "wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - cpu0, 1)}
    _atomic(out / "solve_full.json", lambda p: p.write_text(json.dumps(infos, default=float)))
    _atomic(out / "line_f1_full.json", lambda p: p.write_text(json.dumps(f1rep)))
    _atomic(out / "meta.json", lambda p: p.write_text(json.dumps(meta, indent=1, default=float)))

    def _states(p: Path) -> None:
        with open(p, "wb") as f:
            np.save(f, states)

    _atomic(out / "states_full.npy", _states)
    tmp = out / ".export"
    tmp.mkdir(exist_ok=True)
    pte.export_calib(tmp, "full", states, shared)
    os.replace(tmp / "calib_full.npz", out / "calib_full.npz")  # last: the completion marker
    tmp.rmdir()
    pte.log(f"done {clip}: {meta['calibrated_frames']}/{meta['frames_decoded']} frames calibrated "
            f"({100 * meta['accepted_frac']:.0f}% with F1>={F1_ACCEPT} evidence on both sides), "
            f"{meta['wall_s']:.0f}s wall, {meta['cpu_s']:.0f}s cpu")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--clip", required=True, help="e.g. arsenal_mancity_seg010; work dir is <out-root>/<clip>@<rig stem>")
    ap.add_argument("--rig-json", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=pte.ROOT)
    ap.add_argument("--anchor-step", type=int, default=100)
    ap.add_argument("--no-recover", action="store_true", help="skip the recovery pass for segments the standard pass leaves weak")
    a = ap.parse_args()
    run(a.video, a.clip, a.rig_json, a.out_root, a.anchor_step, recover=not a.no_recover)


if __name__ == "__main__":
    main()
