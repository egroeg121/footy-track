"""Label-free rig (C, roll, k1) of a ground from the broadcast video itself (no GT labels).

    python -m footy_track.scripts.ptz_new_ground anchors --ground spurs --video-root ~/footy/grounds/video \
        --matches tottenham-hotspur-vs-coventry-city-2026-27 --per 4 --jobs 8
    python -m footy_track.scripts.ptz_new_ground fit --ground spurs --prior 0,-58,12 --k1 0.02
    python -m footy_track.scripts.ptz_new_ground report --ground spurs --rig camera_rigs/spurs.json

Generalises the ``shared`` / ``sharedms`` sub-commands of :mod:`footy_track.scripts.ptz_temporal` (``reports/ptz_v3.md``
addendum A/B: C within 0.14-0.3 m of the GT-fitted rig on Man City / Bournemouth) to a ground that has no rig:

``anchors``  from ``--per`` frames of every broadcast segment (sequential decode, no seeks): white-line / grass masks,
             main-camera probability, and a lines-only search (wide pan x tilt x zoom grid, C at ``--prior`` with a
             loose sigma) -> ``<out>/<ground>/anchors.json``.  Frames are the wide, line-rich ones by construction: only
             main-camera frames (p >= 0.5) whose best hypothesis explains the lines (F1 >= ``--min-f1``) are used.
``fit``      multi-start shared fit of (C, roll [, k1]) over those frames (``pt.fit_shared_multistart``), then the
             per-frame agreement of every unit at the fitted rig, a unit-bootstrap of the result, and an optional
             k1 profile; writes the rig json (with an ``info`` block).
``reanchor`` repeat ``anchors`` at the fitted rig (tight C sigma): more accepted frames, poses consistent with the fit
             (then ``fit`` again with ``--prior`` = the fitted C).

The pitch is symmetric under x -> -x and y -> -y, so the camera is placed on the -y side with pan ~ 90 deg whatever
side of the real stadium it is on.  The 105 x 68 m pitch model is assumed (real pitches differ by a metre or two).
"""

# ruff: noqa: PLR0915, PLR0912, PLR0913
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import ptz
from footy_track.calibration import ptz_temporal as pt
from footy_track.calibration.pitch_model import PitchModel

PITCH = PitchModel(105.0, 68.0)
#: wide lines-only grid (any camera height): pan 40-145 deg, tilt 2-50 deg, f 900-9500 px
GRID = ptz._grid(n_pan=36, n_tilt=14, n_f=14, pan_range=(np.deg2rad(40.0), np.deg2rad(145.0)),
                 tilt_range=(np.deg2rad(2.0), np.deg2rad(50.0)), f_range=(900.0, 9500.0))
SEARCH = dict(n_cells=20, n_refine=4, grid=GRID)
OUT = Path(os.environ.get("PTZ_GROUNDS_DIR", Path.home() / "footy/grounds/work"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _vec(txt: str) -> np.ndarray:
    return np.array([float(x) for x in txt.split(",")])


def segments(video_root: Path, match: str) -> list[dict]:
    d = video_root / match
    sj = d / "segments.json"
    segs = json.loads(sj.read_text())["segments"] if sj.exists() else []
    out = [{**s, "path": str(d / s["file"]), "match": match} for s in segs if (d / s["file"]).exists()]
    if not out:  # no segments.json shipped: every mp4
        out = [{"file": p.name, "path": str(p), "match": match, "frame_count": 0} for p in sorted(d.glob("*.mp4"))]
    return out


def _work(args):
    """One segment: decode sequentially to the sampled frames, masks + pmain + lines-only search for each."""
    seg, per, rig_d, out_root, min_pm = args
    rig = ptz.Rig(np.array(rig_d["C"]), rig_d["roll"], rig_d["k1"], tuple(rig_d["sigma_C"]), (105.0, 68.0))
    stem = Path(seg["file"]).stem
    root = Path(out_root) / seg["match"] / stem
    cap = cv2.VideoCapture(seg["path"])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ts = sorted({int(n * (k + 1) / (per + 1)) for k in range(per)}) if n >= 60 else [n // 2]
    cache = pt.ClipCache(Path(seg["path"]), root)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    recs = {}
    t = -1
    for target in ts:
        key = f"{seg['file']}:{target}"
        while t < target:
            if not cap.grab():
                break
            t += 1
        if t < target:
            break
        if not cache.mask_path(target).exists():
            ok, f = cap.retrieve()
            if not ok:
                continue
            cache._save_masks(target, f)
        pm = pt.pmain_of(cache, target)
        ev = cache.evidence(target)
        if ev is None or not pm >= min_pm:
            recs[key] = {"ok": False, "reason": "not main", "pmain": pm, "t": target, "seg": seg["file"],
                         "match": seg["match"]}
            continue
        t0 = time.time()
        cam, hyps, info = ptz.search_lines(ev, rig, PITCH, margin=pt.MARGIN, min_f1=pt.MIN_F1, **SEARCH)
        rec = {"ok": cam is not None, "reason": info.get("reason"), "f1": info.get("f1"), "pmain": pm, "t": target,
               "seg": seg["file"], "match": seg["match"], "secs": time.time() - t0, "margin": info.get("margin")}
        best = cam or (hyps[0].cam if hyps else None)
        if best is not None:
            rec.update(pan=best.pan, tilt=best.tilt, logf=best.logf, C=[float(c) for c in best.C])
        recs[key] = rec
    cap.release()
    return recs


def cmd_anchors(a):
    out = OUT / a.ground
    out.mkdir(parents=True, exist_ok=True)
    store_p = out / a.store
    store = json.loads(store_p.read_text()) if store_p.exists() else {}
    if a.rig:
        r = ptz.Rig.from_json(a.rig)
    else:
        r = ptz.Rig(_vec(a.prior), 0.0, a.k1, tuple(_vec(a.sigma)), (105.0, 68.0))
    rig_d = {"C": [float(c) for c in r.C], "roll": float(r.roll), "k1": float(r.k1), "sigma_C": list(r.sigma_C)}
    log("prior", rig_d)
    jobs = []
    for m in a.matches.split(","):
        segs = segments(Path(a.video_root), m)
        if a.max_segs:
            segs = segs[:: max(1, len(segs) // a.max_segs)]
        log(m, len(segs), "segments")
        jobs += [(s, a.per, rig_d, str(out / "cache"), a.min_pm) for s in segs]
    done = {f"{j[0]['file']}:" for j in jobs}  # noqa: F841 (resume is per frame key below)
    todo = [j for j in jobs if not any(k.startswith(j[0]["file"] + ":") for k in store)]
    log(len(todo), "of", len(jobs), "segments to do")
    t0 = time.time()
    with mp.get_context("fork").Pool(a.jobs) as pool:
        for i, recs in enumerate(pool.imap_unordered(_work, todo), 1):
            store.update(recs)
            if i % 10 == 0 or i == len(todo):
                store_p.write_text(json.dumps(store))
                ok = sum(v.get("ok", False) for v in store.values())
                log(f"  {i}/{len(todo)} segments, {len(store)} frames, {ok} anchored, {time.time() - t0:.0f}s")
    store_p.write_text(json.dumps(store))


def units_of(out: Path, store: dict, min_f1: float, max_per_seg: int = 99) -> tuple[list, list[dict]]:
    units, recs = [], []
    per_seg: dict[str, int] = {}
    for key, r in sorted(store.items(), key=lambda kv: -(kv[1].get("f1") or 0)):
        if "pan" not in r or (r.get("pmain") or 0) < 0.5:
            continue
        if not (r.get("ok") or (r.get("f1") or 0) >= min_f1):
            continue
        sk = f"{r['match']}/{r['seg']}"
        if per_seg.get(sk, 0) >= max_per_seg:
            continue
        cache = pt.ClipCache(Path("x"), out / "cache" / r["match"] / Path(r["seg"]).stem)
        ev = cache.evidence(r["t"])
        if ev is None:
            continue
        per_seg[sk] = per_seg.get(sk, 0) + 1
        units.append(pt.Unit(len(units), [r["t"]], np.array([[r["pan"], r["tilt"], r["logf"]]]), {r["t"]: ev}))
        recs.append({"key": key, **{k: r[k] for k in ("match", "seg", "t", "f1", "ok")}})
    return units, recs


def unit_f1(units, poses, shn: pt.Shared) -> np.ndarray:
    out = []
    for u, p in zip(units, poses, strict=True):
        cam = pt.cam_from_state(p[0], shn)
        out.append(float(ptz.line_support([cam], u.evidence[u.frames[0]], PITCH)[0]))
    return np.array(out)


def cmd_fit(a):
    out = OUT / a.ground
    store = json.loads((out / a.store).read_text())
    units, recs = units_of(out, store, a.min_f1, a.max_per_seg)
    prior = ptz.Rig(_vec(a.prior), 0.0, a.k1, (5.0, 12.0, 6.0), (105.0, 68.0))
    if a.prior_sigma:
        prior = prior.with_sigma(tuple(_vec(a.prior_sigma)))
    free = tuple(a.free.split(","))
    log(f"{len(units)} units from {len({(r['match'], r['seg']) for r in recs})} segments; prior C {prior.C.tolist()} "
        f"k1 {prior.k1} free {free}")
    states_at = pt.Shared.from_rig(ptz.Rig(_vec(a.anchor_C or a.prior), 0.0, a.k1, (1, 1, 1), (105.0, 68.0)))
    offs = pt.seed_offsets(step=tuple(_vec(a.step)), n=tuple(int(k) for k in _vec(a.nseed)))
    t0 = time.time()
    shn, poses, info = pt.fit_shared_multistart(units, prior, PITCH, states_at=states_at, offsets=offs, top_k=a.top_k,
                                                screen_stride=a.stride, free=free, n_iter=a.iters, log=log)
    f1 = unit_f1(units, [np.asarray(p) for p in poses], shn)
    res = {"C": shn.C.tolist(), "roll_deg": float(np.degrees(shn.roll)), "k1": shn.k1, "cost": info["stage2"]["cost"],
           "n_units": len(units), "n_segments": len({(r["match"], r["seg"]) for r in recs}),
           "f1_median": float(np.median(f1)), "f1_ge_0.6": float((f1 >= 0.6).mean()), "f1_ge_0.5": float((f1 >= 0.5).mean()),
           "prior_C": prior.C.tolist(), "free": list(free), "secs": time.time() - t0,
           "finalists": [s for s in info["starts"] if "final_C" in s]}
    for tag, key in (("input_f1_median", [r.get("f1") or 0 for r in recs]),):
        res[tag] = float(np.median(key))
    # unit bootstrap from the winning start (plain fit: cheaper than a multistart per resample)
    if a.boot:
        rng = np.random.default_rng(0)
        bs = []
        start = pt.replace(pt.Shared.from_rig(prior), C=shn.C, roll=shn.roll, k1=shn.k1)
        for b in range(a.boot):
            idx = rng.choice(len(units), size=len(units), replace=True)
            us = [pt.replace(units[i], uid=k, states=np.asarray(poses[i])) for k, i in enumerate(idx)]
            sb, _, _ = pt.fit_shared(us, prior, PITCH, free=free, n_iter=a.iters, start=start)
            bs.append([*sb.C, np.degrees(sb.roll), sb.k1])
            log(f"  boot {b}: C {np.round(sb.C, 2).tolist()}")
        bs = np.array(bs)
        res["bootstrap"] = {"n": a.boot, "mean": bs.mean(0).tolist(), "std": bs.std(0).tolist(),
                            "names": ["Cx", "Cy", "Cz", "roll_deg", "k1"]}
    (out / f"fit_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez(out / f"fit_{a.tag}_units.npz", f1=f1, keys=np.array([r["key"] for r in recs]),
             poses=np.array([np.asarray(p)[0] for p in poses]))
    log("RESULT", {k: v for k, v in res.items() if k not in ("finalists",)})


def cmd_rig(a):
    """Write the rig json from a fit result."""
    out = OUT / a.ground
    res = json.loads((out / f"fit_{a.tag}.json").read_text())
    info = {"ground": a.ground, "fit_date": time.strftime("%Y-%m-%d"), "method": "label-free shared multi-start fit of "
            "(C, roll) over main-camera frames with lines-only anchors; k1 fixed (see footy_track.scripts.ptz_new_ground)",
            "n_frames": res["n_units"], "n_segments": res["n_segments"], "cost": res["cost"],
            "line_f1_median_at_fit": res["f1_median"], "frac_f1_ge_0.6": res["f1_ge_0.6"], "prior_C": res["prior_C"],
            "finalists": res["finalists"], "bootstrap": res.get("bootstrap"), "matches": a.matches.split(",")}
    if a.extra:
        info.update(json.loads(a.extra))
    rig = {"C": res["C"], "roll": float(np.radians(res["roll_deg"])), "k1": res["k1"], "sigma_C": [1.0, 3.0, 1.0],
           "pitch": [105.0, 68.0], "info": info}
    Path(a.rig_out).write_text(json.dumps(rig, indent=1, default=float))
    log("wrote", a.rig_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["anchors", "fit", "rig"])
    ap.add_argument("--ground", required=True)
    ap.add_argument("--matches", default="")
    ap.add_argument("--video-root", default=str(Path.home() / "footy/grounds/video"))
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--max-segs", type=int, default=0)
    ap.add_argument("--min-pm", type=float, default=0.5)
    ap.add_argument("--prior", default="0,-58,12")
    ap.add_argument("--sigma", default="3,10,5", help="anchors: per-frame C sigma (m)")
    ap.add_argument("--prior-sigma", default=None, help="fit: shared prior sigma, default 5,12,6")
    ap.add_argument("--anchor-C", default=None, help="fit: the C the anchors were solved at (default --prior)")
    ap.add_argument("--rig", default=None, help="anchors: rig json instead of --prior (reanchor)")
    ap.add_argument("--store", default="anchors.json")
    ap.add_argument("--k1", type=float, default=0.02)
    ap.add_argument("--free", default="Cx,Cy,Cz,roll")
    ap.add_argument("--min-f1", type=float, default=0.5)
    ap.add_argument("--max-per-seg", type=int, default=99)
    ap.add_argument("--step", default="3,6,3")
    ap.add_argument("--nseed", default="1,2,2")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--rig-out", default=None)
    ap.add_argument("--extra", default=None)
    a = ap.parse_args()
    {"anchors": cmd_anchors, "fit": cmd_fit, "rig": cmd_rig}[a.cmd](a)


if __name__ == "__main__":
    main()
