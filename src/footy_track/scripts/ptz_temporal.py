"""PTZ V3 prototype driver: temporal calibration of one Emirates clip, scored against human GT.

    python -m footy_track.scripts.ptz_temporal prep  --clip arsenal_mancity_seg010   # cache, anchors, motion
    python -m footy_track.scripts.ptz_temporal solve --clip arsenal_mancity_seg010   # per-segment joint solves
    python -m footy_track.scripts.ptz_temporal eval  --clip arsenal_mancity_seg010   # (a) (b) (c) vs GT + coverage/jitter

Everything is cached under ``/mnt/storage/footy_data/scratch/temporal/<clip>/``. One process, no workers.
"""

# ruff: noqa: PLR0915, PLR0912, PLC0415, E702, C408, C416, F841, PLC3002, PLW0603
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration import ptz
from footy_track.calibration import ptz_temporal as pt
from footy_track.calibration.pitch_model import PitchModel
from footy_track.scripts import ptz_validate as pv

ROOT = Path(os.environ.get("PTZ_TEMPORAL_DIR", "/mnt/storage/footy_data/scratch/temporal"))
CLIPS = Path("/mnt/storage/footy_data/eval_clips_kf10")
FOOTY_DATA_ROOT = Path("/mnt/storage/footy_data")
#: rig JSON given on the command line (``--rig``): used as the clip's rig instead of the leave-clip-out Emirates fit, and
#: keeps its own work directory (``<clip>@<rig stem>``) because anchors and motion depend on the rig
RIG_PATH: Path | None = None
WORK_SUFFIX = ""
#: ground -> (GT match key of ``ptz_validate``, default rig JSON, broadcast segment dir, segment file pattern)
RIGS_DIR = Path(__file__).resolve().parents[1] / "calibration" / "camera_rigs"
BMTH_DIR = FOOTY_DATA_ROOT / "arsenal_bournmouth_1st_half" / "split_video_broadcast_frames"


def wd(clip: str) -> Path:
    """Work directory of a clip (caches, anchors, pairs, states, eval)."""
    return ROOT / f"{clip}{WORK_SUFFIX}"
PITCH = PitchModel()
EV_STEP = 5
STATIC_FLOW = 1.0  # work px / frame: a 'near-static' camera pair
SEARCH_LIGHT = dict(n_cells=20, n_refine=4)
ANCHOR_STEP = 50


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[cpu {time.process_time():.0f}s]", *a, flush=True)


def clip_gt(clip: str) -> list[pv.GTFrame]:
    m = pv.match_of(clip)
    return [f for f in pv.load_gt((m,) if m else ()) if f.clip == clip]


def clip_video(clip: str) -> Path:
    """``arsenal_mancity_seg010`` -> the eval clip; ``arsenal_bournemouth_1st_seg060`` -> the broadcast segment file."""
    for name in (clip, clip.removeprefix("arsenal_")):
        if (CLIPS / f"{name}.mp4").exists():
            return CLIPS / f"{name}.mp4"
    if pv.match_of(clip) == "bmth":
        idx = int(clip.rsplit("_seg", 1)[1])
        p = BMTH_DIR / f"Bournemouth vs Arsenal 1_seg{idx:03d}.mp4"
        if p.exists():
            return p
    raise FileNotFoundError(clip)


def n_frames(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def leave_clip_out_rig(clip: str) -> ptz.Rig:
    """Emirates rig refitted on all GT frames *except* this clip's (cached); ``--rig`` overrides it."""
    if RIG_PATH is not None:
        return ptz.Rig.from_json(RIG_PATH)
    if pv.match_of(clip) == "bmth":
        return ptz.Rig.from_json(RIGS_DIR / "vitality.json")
    p = ROOT / "rigs" / f"emirates_without_{clip}.json"
    if p.exists():
        return ptz.Rig.from_json(p)
    gt = [f for f in pv.load_gt(("mancity", "villa")) if f.clip != clip]
    log(f"fitting leave-clip-out rig on {len(gt)} GT frames")
    rig, _ = pv.fit(gt)
    p.parent.mkdir(parents=True, exist_ok=True)
    rig.to_json(p)
    return rig


def prep(clip: str, rig: ptz.Rig, gaps=(1, 5)):
    video = clip_video(clip)
    gt = clip_gt(clip)
    ev = set(range(0, n_frames(video), EV_STEP)) | {f.frame for f in gt}
    cache = pt.ClipCache(video, wd(clip)).build(ev, log=log)
    log(f"cache: {cache.n} frames, {len(cache.ev_frames)} evidence frames")
    # shots: hard cuts, main-camera probability per shot
    shots = pt.segment_shots(cache.hd)
    pm = {t: pt.pmain_of(cache, t) for t in cache.ev_frames}
    shot_info = []
    for a, b in shots:
        v = [pm[t] for t in cache.ev_frames if a <= t <= b and np.isfinite(pm[t])]
        shot_info.append({"start": a, "end": b, "p_main": float(np.median(v)) if v else float("nan"), "n_ev": len(v)})
    log("shots:", shot_info)
    # lines-only anchors on every evidence frame (cached)
    ap = wd(clip) / "anchors_lines.json"
    anchors = json.loads(ap.read_text()) if ap.exists() else {}
    gt_t = {f.frame for f in gt}
    for t in cache.ev_frames:
        # lines-only anchors cost ~15 CPU-s a frame: every ANCHOR_STEP-th frame and the scored GT frames only
        if str(t) in anchors or not (t % ANCHOR_STEP == 0 or t in gt_t):
            continue
        evd = cache.evidence(t)
        if evd is None:
            continue
        # scored GT frames: the full V2 search (baseline (a)); the others: a lighter one (3x cheaper) as anchors only
        anchors[str(t)] = pt.lines_anchor(evd, rig, PITCH, **({} if t in gt_t else SEARCH_LIGHT))
        ap.write_text(json.dumps(anchors))
        log(f"  anchor {t}: {anchors[str(t)]['ok']} f1 {anchors[str(t)]['f1']} {anchors[str(t)]['secs']:.0f}s "
            f"({len(anchors)} done)")
    ap.write_text(json.dumps(anchors))
    return cache, shots, shot_info, anchors


def refit_anchors(cache, anchors, rig) -> dict:
    """``search_lines`` optimises C inside its prior, but the anchor record keeps (pan, tilt, log f) only: re-run the line
    refinement from it twice -- C free within the rig prior (reproduces the lines-only camera, baseline (a)) and C
    hard-fixed to the rig (consistent anchor for the joint solve, which keeps C fixed)."""
    rp = cache.root / "anchors_refit.json"
    done = json.loads(rp.read_text()) if rp.exists() else {}
    for t, a in anchors.items():
        if t in done or "pan" not in a:
            continue
        ev = cache.evidence(int(t))
        cam0 = ptz.Camera(a["pan"], a["tilt"], a["logf"], rig.roll, np.array(rig.C, float), rig.k1)
        cf = ptz._refine_lines(cam0, ev, rig, PITCH)
        cx = ptz._refine_lines(cam0, ev, rig.with_sigma((0.0, 0.0, 0.0)), PITCH)
        done[t] = {"free": [cf.pan, cf.tilt, cf.logf, *cf.C], "fixed": [cx.pan, cx.tilt, cx.logf]}
        rp.write_text(json.dumps(done))
    for t, d in done.items():
        if t in anchors:
            anchors[t]["refit"] = d
    return anchors


def f_nominal(cache, anchors) -> np.ndarray:
    ts = sorted(int(t) for t, a in anchors.items() if a["ok"])
    if not ts:
        return np.full(cache.n, pt.F_NOM)
    f = np.exp([anchors[str(t)]["logf"] for t in ts])
    return np.exp(np.interp(np.arange(cache.n), ts, np.log(f)))


def motion(cache, shots, shot_info, anchors, rig, gaps=(1, 5)):
    mp = cache.root / f"pairs_g{'_'.join(map(str, gaps))}.pkl"
    if mp.exists():
        return pickle.loads(mp.read_bytes())
    fnom = f_nominal(cache, anchors)
    allp = []
    for sh in shot_info:
        if not sh["p_main"] >= 0.5:
            continue
        a, b = sh["start"], sh["end"]
        log(f"motion shot {a}-{b}")
        ps, st = pt.compute_motion(cache, a, b, fnom[a:b + 1], rig.k1, gaps=gaps, log=log)
        log(f"  {len(ps)} pairs, {st}")
        allp += ps
    mp.write_bytes(pickle.dumps(allp))
    return allp


def segments(shot_info, pairs) -> list[tuple[int, int]]:
    """Motion-connected runs of frames inside main-camera shots (a failed gap-1 pair ends a segment)."""
    ok1 = {p.i for p in pairs if p.j - p.i == 1 and p.ok}
    out = []
    for sh in shot_info:
        if not sh["p_main"] >= 0.5:
            continue
        a = sh["start"]
        for t in range(sh["start"], sh["end"]):
            if t not in ok1:
                if t - a + 1 >= 8:
                    out.append((a, t))
                a = t + 1
        if sh["end"] - a + 1 >= 8:
            out.append((a, sh["end"]))
    return out


def anchors_state(anchors, ts) -> dict[int, np.ndarray]:
    return {t: np.array(anchors[str(t)]["refit"]["fixed"]) for t in ts
            if str(t) in anchors and anchors[str(t)]["ok"] and "refit" in anchors[str(t)]}


FINDER_SIGMA = 2.5  # px (canonical): the finders' points are single detections, not fitted lines


def solve_variant(cache, segs, pairs, anchors, rig, gt_frames, hold: int | None, tag: str, free_shared=(),
                  finder_pts: dict | None = None, **kw):
    """All segments of the clip. ``hold``: withhold line evidence and anchors within +-hold frames of every GT frame."""
    N = cache.n
    states = np.full((N, 3), np.nan)
    infos = []
    held = set()
    if hold is not None:
        for g in gt_frames:
            held |= set(range(g - hold, g + hold + 1))
    for a, b in segs:
        ev_frames = [t for t in cache.ev_frames if a <= t <= b and t not in held]
        anc = anchors_state(anchors, [t for t in ev_frames])
        t = time.time()
        pts = [(t, xy, obs, FINDER_SIGMA) for t, (xy, obs) in (finder_pts or {}).items() if a <= t <= b and t not in held]
        res = pt.solve_segment(cache, a, b, [p for p in pairs if a <= p.i and p.j <= b], anc, rig, PITCH,
                               evidence_frames=ev_frames, free_shared=free_shared, log=log, points=pts or None, **kw)
        states[a - 0:b + 1] = res.states
        infos.append({"seg": [a, b], "n_anchors": res.n_anchors, "dropped": res.dropped_anchors, "info": res.info,
                      "secs": time.time() - t, "shared": res.shared.vec().tolist()})
        log(f"[{tag}] segment {a}-{b}: anchors {res.n_anchors} dropped {res.dropped_anchors} {time.time() - t:.0f}s "
            f"{res.info}")
    return states, infos


EXTRA_VARIANTS = {"sm3": {"sigma_motion": 3.0}, "sm10": {"sigma_motion": 10.0}, "sl1": {"sigma_line": 1.5}}


def err5(cam, truth) -> dict:
    """Pitch error (m) of ``cam`` over the 5 m grid of the GT-visible pitch."""
    P, uv, near = pv.gt_grid(truth, step=5.0)
    if len(P) < 6:
        return {"failed": True}
    xy, ok = cam.image_to_pitch(uv)
    e = np.linalg.norm(xy - P, axis=1)
    e = np.where(ok & np.isfinite(e), np.minimum(e, 100.0), 100.0)
    return {"failed": False, "med": float(np.median(e)), "p90": float(np.percentile(e, 90)),
            "max": float(e.max())}


def finder_cam(cache, t, rig):
    """Per-frame finders (centre + penalty-area template) -> points -> PTZ point solve. -> (cam or None, info)."""
    from footy_track.calibration import pitch_finders as pf

    bgr = cache.read_full(t)
    dist = geo.make_distortion(rig.k1)
    uv, xy, kinds = [], [], []
    t0 = time.time()
    for kind in ("centre", "d"):
        try:
            det = pf.find_points(kind, bgr, dist=dist, pitch=PITCH)
        except Exception as e:  # noqa: BLE001
            log("finder error", kind, t, repr(e)[:80])
            continue
        if det.result is not None and det.result.points:
            kinds.append(kind)
            for p in det.result.points:
                uv.append(p.uv)
                xy.append(p.pitch_xy)
    info = {"kinds": kinds, "n": len(uv), "secs": time.time() - t0}
    if len(uv) < 2:
        return None, info
    cam = ptz.solve_points(np.array(uv), np.array(xy), rig)
    return cam, info


def finder_points(cache, t, rig) -> dict:
    """Centre-circle and penalty-area finder points of frame ``t`` (cached by the caller): pitch metres (n,2) and observed
    canonical-px image points (n,2) as the joint solver's ``points`` residuals want them."""
    from footy_track.calibration import pitch_finders as pf

    bgr = cache.read_full(t)
    dist = geo.make_distortion(rig.k1)
    uv, xy, kinds = [], [], []
    for kind in ("centre", "d"):
        try:
            det = pf.find_points(kind, bgr, dist=dist, pitch=PITCH)
        except Exception as e:  # noqa: BLE001
            log("finder error", kind, t, repr(e)[:80])
            continue
        if det.result is not None and det.result.points:
            kinds.append(kind)
            uv += [p.uv for p in det.result.points]
            xy += [p.pitch_xy for p in det.result.points]
    return {"kinds": kinds, "xy": np.asarray(xy, float).reshape(-1, 2).tolist(),
            "obs": (np.asarray(uv, float).reshape(-1, 2) * ptz.CANON).tolist()}


def finder_point_set(cache, frames, rig) -> dict[int, tuple]:
    """``frames`` -> {t: (xy, obs)} for the frames where the finders found >= 2 points (cached in ``finder_points.json``)."""
    fp = cache.root / "finder_points.json"
    store = json.loads(fp.read_text()) if fp.exists() else {}
    for t in frames:
        if str(t) not in store:
            store[str(t)] = finder_points(cache, t, rig)
            fp.write_text(json.dumps(store))
            log(f"  finders {t}: {store[str(t)]['kinds']} {len(store[str(t)]['xy'])} pts")
    return {int(t): (np.array(v["xy"]), np.array(v["obs"])) for t, v in store.items()
            if int(t) in set(frames) and len(v["xy"]) >= 2}


def stats(recs, key="med"):
    v = np.array([r[key] for r in recs if r and not r.get("failed")])
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95)), "mean": float(v.mean()), "gt5m": int((v > 5).sum())}


def draw_lines(img, project, color, thick=2):
    """Pitch markings through ``project(xy) -> (px (N,2) in img coords, ok)``."""
    for poly in PITCH.polylines().values():
        d = geo._densify(np.asarray(poly, np.float64), 0.5)
        uv, ok = project(d)
        for k in range(len(d) - 1):
            if ok[k] and ok[k + 1] and np.abs(uv[k]).max() < 1e5 and np.abs(uv[k + 1]).max() < 1e5:
                cv2.line(img, tuple(np.rint(uv[k]).astype(int)), tuple(np.rint(uv[k + 1]).astype(int)), color, thick,
                         cv2.LINE_AA)


def cam_proj(cam, w, h):
    def f(xy):
        uv, ok = cam.pitch_to_image(xy)
        return uv * [w, h], ok
    return f


def render(cache, t, cams: dict, truth=None, width=960):
    """Frame t with the pitch markings of each camera in ``cams`` (name -> (cam, BGR colour)); truth in red."""
    f = cache.read_full(t)
    h = int(round(f.shape[0] * width / f.shape[1]))
    img = cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA)
    if truth is not None:
        draw_lines(img, lambda xy: (lambda r: (r[0] * [width, h], r[1]))(truth.to_image(xy)), (0, 0, 255), 1)
    y = 18
    for name, (cam, col) in cams.items():
        if cam is not None:
            draw_lines(img, cam_proj(cam, width, h), col, 2)
        cv2.putText(img, f"{name}{'' if cam is not None else ' -'}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        y += 20
    cv2.putText(img, f"frame {t}", (width - 120, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


def load_clip(clip):
    rig = leave_clip_out_rig(clip)
    cache, shots, shot_info, anchors = prep(clip, rig)
    anchors = refit_anchors(cache, anchors, rig)
    pairs = motion(cache, shots, shot_info, anchors, rig)
    segs = segments(shot_info, pairs)
    return rig, cache, shot_info, anchors, pairs, segs


def export_calib(out: Path, tag: str, states: np.ndarray, shared: pt.Shared) -> None:
    """``calib_<tag>.npz``: the per-frame output of a solve. ``frame`` (N,) clip frame index, ``pan`` / ``tilt`` / ``logf``
    (N,) radians / ln(px at 1920) with NaN where the frame has no calibration, ``H`` (N,3,3) the undistorted-normalised-image
    -> pitch-metres homography of each calibrated frame (NaN elsewhere), ``rig`` = (Cx, Cy, Cz, roll rad, k1)."""
    n = len(states)
    H = np.full((n, 3, 3), np.nan)
    for t in range(n):
        if np.all(np.isfinite(states[t])):
            H[t] = pt.cam_from_state(states[t], shared).homography()
    np.savez(out / f"calib_{tag}.npz", frame=np.arange(n), pan=states[:, 0], tilt=states[:, 1], logf=states[:, 2], H=H,
             rig=shared.vec())


def cmd_solve(args):
    rig, cache, shot_info, anchors, pairs, segs = load_clip(args.clip)
    log("segments", segs)
    gt = clip_gt(args.clip)
    out = wd(args.clip)
    want = set(args.variants.split(","))
    gtf = [f.frame for f in gt]
    shr = pt.Shared.from_rig(rig)
    for tag, hold in (("full", None), ("hold", 2)):
        if tag not in want or ((out / f"states_{tag}.npy").exists() and not args.force):
            continue
        st, infos = solve_variant(cache, segs, pairs, anchors, rig, gtf, hold, tag)
        np.save(out / f"states_{tag}.npy", st)
        (out / f"solve_{tag}.json").write_text(json.dumps(infos, default=float))
        export_calib(out, tag, st, shr)
    if "sweeps" in want:
        for tag, kw in EXTRA_VARIANTS.items():
            if (out / f"states_{tag}.npy").exists() and not args.force:
                continue
            st, infos = solve_variant(cache, segs, pairs, anchors, rig, gtf, None, tag, **kw)
            np.save(out / f"states_{tag}.npy", st)
    if "fin" in want and not ((out / "states_fin.npy").exists() and not args.force):
        # finder points (centre circle / penalty area) as anchors next to the lines-only ones: at the anchor cadence + GT frames
        fr = [t for t in cache.ev_frames if t % ANCHOR_STEP == 0 or t in set(gtf)]
        fpts = finder_point_set(cache, fr, rig)
        log(f"finder points on {len(fpts)} of {len(fr)} frames")
        st, infos = solve_variant(cache, segs, pairs, anchors, rig, gtf, None, "fin", finder_pts=fpts)
        np.save(out / "states_fin.npy", st)
        (out / "solve_fin.json").write_text(json.dumps(infos, default=float))
        export_calib(out, "fin", st, shr)
    # rig quantities estimated from the line evidence of every segment of the match (no GT): solve the clip with them
    est = None
    if args.est:  # JSON with C / roll_deg / k1 (optionally nested under --est-key)
        est = json.loads(Path(args.est).read_text())
        est = est[args.est_key] if args.est_key else est
    elif (ROOT / "shared" / "shared_mancity.json").exists() and args.clip.startswith(("arsenal_mancity", "arsenal_astonvilla")):
        est = json.loads((ROOT / "shared" / "shared_mancity.json").read_text())["lines_plus_motion"]
    if est is not None and "est" in want:
        sh = pt.Shared(np.array(est["C"]), np.radians(est["roll_deg"]), est["k1"])
        for tag, hold in ((f"est{args.est_tag}", None), (f"esthold{args.est_tag}", 2)):
            if (out / f"states_{tag}.npy").exists() and not args.force:
                continue
            st, infos = solve_variant(cache, segs, pairs, anchors, rig, gtf, hold, tag, shared=sh)
            np.save(out / f"states_{tag}.npy", st)
            (out / f"shared_{tag}.json").write_text(json.dumps(sh.vec().tolist()))
            (out / f"solve_{tag}.json").write_text(json.dumps(infos, default=float))
            export_calib(out, tag, st, sh)
    # ablation: no motion constraints (per-frame line evidence + anchors only, same solver)
    if "nomotion" not in want or ((out / "states_nomotion.npy").exists() and not args.force):
        return
    st, infos = solve_variant(cache, segs, pairs, anchors, rig, gtf, None, "nomotion", use_motion=False)
    np.save(out / "states_nomotion.npy", st)


def line_f1(cache, st, shared, ts):
    out = {}
    for t in ts:
        ev = cache.evidence(t)
        if ev is None or not np.all(np.isfinite(st[t])):
            continue
        out[t] = float(ptz.line_support([pt.cam_from_state(st[t], shared)], ev, PITCH)[0])
    return out


def cmd_eval(args):
    rig, cache, shot_info, anchors, pairs, segs = load_clip(args.clip)
    gt = clip_gt(args.clip)
    out = wd(args.clip)
    sh = pt.Shared.from_rig(rig)
    res = {"clip": args.clip, "n_gt": len(gt), "rig": {"C": rig.C.tolist(), "k1": rig.k1, "roll": rig.roll}}
    S_ = {f.stem.removeprefix("states_"): np.load(f) for f in sorted(out.glob("states_*.npy"))}
    recs = {"a_lines": [], "b_finders": [], **{f"c_{v}": [] for v in S_}}
    per = []
    fp = out / "finders.json"
    fcache = json.loads(fp.read_text()) if fp.exists() else {}
    SH = {v: (pt.Shared.from_vec(np.array(json.loads((out / f"shared_{v}.json").read_text())))
              if (out / f"shared_{v}.json").exists() else sh) for v in S_}
    for g in gt:
        truth = g.truth()
        row = {"key": g.key, "t": g.frame}
        a = anchors.get(str(g.frame))
        ca = None
        if a and "refit" in a:
            q = a["refit"]["free"]
            ca = ptz.Camera(q[0], q[1], q[2], rig.roll, np.array(q[3:6]), rig.k1)
        row["a_ok"] = bool(a and a["ok"])
        row["a_f1"] = a.get("f1") if a else None
        ra = err5(ca, truth) if ca is not None else None
        row["a_best"] = ra
        recs["a_lines"].append(ra if row["a_ok"] else None)
        # (b) finders
        if str(g.frame) not in fcache:
            cam, info = finder_cam(cache, g.frame, rig)
            fcache[str(g.frame)] = {"info": info, "params": None if cam is None else
                                    [cam.pan, cam.tilt, cam.logf, cam.roll, *cam.C, cam.k1]}
            fp.write_text(json.dumps(fcache))
        fc = fcache[str(g.frame)]
        row["b_info"] = fc["info"]
        if fc["params"]:
            q = fc["params"]
            cb = ptz.Camera(q[0], q[1], q[2], q[3], np.array(q[4:7]), q[7])
            rb = err5(cb, truth)
            row["b"] = rb
            recs["b_finders"].append(rb)
        else:
            recs["b_finders"].append(None)
        for v, st in S_.items():
            if np.all(np.isfinite(st[g.frame])):
                rc = err5(pt.cam_from_state(st[g.frame], SH[v]), truth)
                row[f"c_{v}"] = rc
                recs[f"c_{v}"].append(rc)
            else:
                recs[f"c_{v}"].append(None)
        per.append(row)
    res["per_frame"] = per
    res["summary"] = {k: stats([r for r in v if r]) | {"answered": int(sum(r is not None for r in v)), "of": len(v)}
                      for k, v in recs.items()}
    # (a) on abstained frames: error of the best (rejected) hypothesis, to show what abstention costs
    # coverage: frames of main segments with a calibration the evidence agrees with
    ev_ts = [t for t in cache.ev_frames]
    main_frames = sum(b - a + 1 for a, b in segs)
    cov = {}
    for v, st in S_.items():
        f1 = line_f1(cache, st, SH[v], ev_ts)
        good = {t: f1.get(t, -1.0) >= 0.5 for t in ev_ts}
        acc = np.zeros(cache.n, bool)
        for t in range(cache.n):  # accepted when the evidence frames on both sides of t agree with the solved camera
            prev = max((e for e in ev_ts if e <= t), default=None)
            nxt = min((e for e in ev_ts if e >= t), default=None)
            acc[t] = all(good[e] for e in (prev, nxt) if e is not None)
        valid = np.array([np.all(np.isfinite(st[t])) for t in range(cache.n)])
        cov[v] = {"f1_median_ev": float(np.median(list(f1.values()))) if f1 else None,
                  "frac_ev_f1_ge_0.5": float(np.mean(list(good.values()))), "accepted_frac_of_main": float((acc & valid).sum() / max(main_frames, 1)),
                  "calibrated_frac_of_main": float(valid.sum() / max(main_frames, 1))}
    ak = [t for t in ev_ts if t % ANCHOR_STEP == 0 and str(t) in anchors]
    cov["a_lines"] = {"accepted_frac_of_anchor_frames": float(np.mean([anchors[str(t)]["ok"] for t in ak])), "n": len(ak)}
    cov["b_finders"] = {"answered_of_gt": float(np.mean([r is not None for r in recs["b_finders"]]))}
    res["coverage"] = cov
    # jitter: frame-to-frame jump of the projected pitch centre where the camera is near static (LK inlier flow)
    flow = {}
    for p in pairs:
        if p.j - p.i == 1 and p.ok:
            flow[p.i] = float(np.median(np.linalg.norm(p.b - p.a, axis=1)))
    jit = {}
    for v, st in S_.items():
        cs = [pt.cam_from_state(st[t], SH[v]) if np.all(np.isfinite(st[t])) else None for t in range(cache.n)]
        u = np.array([(c.pitch_to_image(np.zeros((1, 2)))[0][0] * ptz.CANON) if c else [np.nan, np.nan] for c in cs])
        jump = np.linalg.norm(np.diff(u, axis=0), axis=1)  # (n-1,) px at 1920 between t and t+1
        acc2 = np.linalg.norm(u[2:] - 2 * u[1:-1] + u[:-2], axis=1)  # second difference: ~0 for a smooth pan
        slow = np.array([flow.get(t, 99.0) < STATIC_FLOW for t in range(cache.n - 1)])
        jit[v] = {"n_slow_pairs": int(slow.sum()),
                  "jump_px_median_slow": float(np.nanmedian(jump[slow])) if slow.any() else None,
                  "jump_px_p95_slow": float(np.nanpercentile(jump[slow], 95)) if slow.any() else None,
                  "flow_px_median_slow": float(np.median([flow[t] * 2 for t in range(cache.n - 1) if slow[t]])) if slow.any() else None,
                  "accel_px_median_all": float(np.nanmedian(acc2)), "accel_px_p95_all": float(np.nanpercentile(acc2, 95)),
                  "accel_px_max_all": float(np.nanmax(acc2))}
    res["jitter_near_static"] = jit
    res["n_static_pairs"] = int(sum(f < STATIC_FLOW for f in flow.values()))
    res["motion"] = {"pairs": len(pairs), "ok": int(sum(p.ok for p in pairs)),
                     "median_inliers": float(np.median([p.n_in for p in pairs if p.ok])),
                     "median_rms_canon_px": float(np.nanmedian([p.rms for p in pairs if p.ok]))}
    (out / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("summary", "coverage", "jitter_near_static", "motion", "n_static_pairs")},
                     indent=1, default=float))


def cmd_sheet(args):
    """Contact sheet (3 x N tiles) + overlay video. GT frames when the clip has any (truth in red, solved cameras on top),
    else evenly spaced frames; ``--est-tag`` adds the est-rig variant, anchors (lines-only) are drawn where they exist."""
    rig, cache, shot_info, anchors, pairs, segs = load_clip(args.clip)
    gt = clip_gt(args.clip)
    out = wd(args.clip)
    sh = pt.Shared.from_rig(rig)
    st = np.load(out / "states_full.npy")
    sth = np.load(out / "states_hold.npy") if (out / "states_hold.npy").exists() else None
    ste = out / f"states_est{args.est_tag}.npy"
    ste, she = (np.load(ste), pt.Shared.from_vec(np.array(json.loads((out / f"shared_est{args.est_tag}.json").read_text())))) \
        if ste.exists() else (None, None)
    if gt:
        gts = sorted(gt, key=lambda f: f.frame)
        pick = [gts[i] for i in sorted(set(np.linspace(0, len(gts) - 1, min(12, len(gts))).astype(int)))]
        todo = [(g.frame, g.truth()) for g in pick]
    else:
        ts = sorted(t for t in cache.ev_frames if np.all(np.isfinite(st[t])))
        todo = [(ts[i], None) for i in sorted(set(np.linspace(0, len(ts) - 1, 12).astype(int)))] if ts else []
    tiles = []
    for t, truth in todo:
        a = anchors.get(str(t))
        ca = None
        if a and a.get("ok") and "refit" in a:
            q = a["refit"]["free"]
            ca = ptz.Camera(q[0], q[1], q[2], rig.roll, np.array(q[3:6]), rig.k1)
        fin = lambda s_, h_: pt.cam_from_state(s_[t], h_) if s_ is not None and np.all(np.isfinite(s_[t])) else None
        cams = {"temporal(full)": (fin(st, sh), (0, 255, 0)), "temporal(hold)": (fin(sth, sh), (0, 255, 255)),
                "temporal(est rig)": (fin(ste, she), (255, 0, 255)), "lines-only": (ca, (255, 128, 0))}
        tiles.append(cv2.resize(render(cache, t, cams, truth), (640, 360)))
    while tiles and len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    dest = Path(args.out or ROOT)
    dest.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest / f"{args.clip}{WORK_SUFFIX}_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # overlay video: every 5th frame, temporal (full) in green, 960 wide
    vw = cv2.VideoWriter(str(dest / f"{args.clip}{WORK_SUFFIX}_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (960, 540))
    for t in range(0, cache.n, 5):
        c = pt.cam_from_state(st[t], sh) if np.all(np.isfinite(st[t])) else None
        vw.write(render(cache, t, {"temporal": (c, (0, 255, 0))}))
    vw.release()
    log("wrote", dest)


MATCH_DIRS = {"mancity": "arsenal_mancity", "villa": "arsenal_astonvilla", "bmth": "arsenal_bournmouth_1st_half"}
FOOTY_DATA = Path("/mnt/storage/footy_data")


def match_segments(match: str) -> list[dict]:
    """The broadcast-only segments of a match (``split_video_broadcast_frames/segments.json``), with the video path."""
    d = FOOTY_DATA / MATCH_DIRS[match] / "split_video_broadcast_frames"
    segs = json.loads((d / "segments.json").read_text())["segments"]
    for sg in segs:
        sg["path"] = d / sg["file"]
    return [sg for sg in segs if sg["path"].exists()]


#: 3.6x fewer grid cells than the production lines-only grid, restricted to the pan / tilt range of the Emirates main
#: camera: the shared fit only needs *some* confident frames per segment, not an answer for every one
SHARED_GRID = ptz._grid(n_pan=36, n_tilt=11, n_f=12, pan_range=(np.deg2rad(55.0), np.deg2rad(125.0)),
                        tilt_range=(np.deg2rad(5.0), np.deg2rad(32.0)), f_range=(1400.0, 8000.0))
#: Vitality (Bournemouth): a lower camera (7.8 m), so the pan / tilt / zoom ranges are the full lines-only ones
SHARED_GRID_BMTH = ptz._grid(n_pan=36, n_tilt=14, n_f=14, pan_range=(np.deg2rad(40.0), np.deg2rad(145.0)),
                             tilt_range=(np.deg2rad(2.0), np.deg2rad(50.0)), f_range=(900.0, 9500.0))
SHARED_SEARCH = dict(n_cells=20, n_refine=4, grid=SHARED_GRID)


def sample_frames(seg: dict, per: int) -> list[int]:
    n = seg["frame_count"]
    return [int(n * (k + 1) / (per + 1)) for k in range(per)] if n >= 100 else [n // 2]


def shared_anchor(match: str, seg: dict, t: int, rig: ptz.Rig, store: dict, root: Path):
    """Evidence + main-camera p + (reduced) lines-only anchor of frame ``t`` of a segment (cached)."""
    key = f"{seg['file']}:{t}"
    cache = pt.ClipCache(seg["path"], root / match / Path(seg["file"]).stem)
    if key in store:
        return store[key]
    (cache.root / "masks").mkdir(parents=True, exist_ok=True)
    if not cache.mask_path(t).exists():
        f = cache.read_full(t)
        if f is None:
            store[key] = {"ok": False, "reason": "no frame"}
            return store[key]
        cache._save_masks(t, f)
    pm = pt.pmain_of(cache, t)
    ev = cache.evidence(t)
    if ev is None or not pm >= 0.5:
        store[key] = {"ok": False, "reason": "not main" if ev is not None else "no mask", "pmain": pm}
        return store[key]
    t0 = time.time()
    kw = {**SHARED_SEARCH, "grid": SHARED_GRID_BMTH} if match == "bmth" else SHARED_SEARCH
    cam, hyps, info = ptz.search_lines(ev, rig, PITCH, margin=pt.MARGIN, min_f1=pt.MIN_F1, **kw)
    rec = {"ok": cam is not None, "reason": info.get("reason"), "f1": info.get("f1"), "pmain": pm,
           "secs": time.time() - t0}
    best = cam or (hyps[0].cam if hyps else None)
    if best is not None:
        rec.update(pan=best.pan, tilt=best.tilt, logf=best.logf)
    store[key] = rec
    return rec


def shared_units(match: str, segs: list[dict], store: dict, root: Path, per: int, min_f1: float | None = None) -> list:
    """One single-frame :class:`pt.Unit` per anchored frame in ``store`` (the anchor only initialises the pose). With
    ``min_f1`` also the main-camera frames whose best hypothesis explains at least that F1 although the search abstained
    (low margin / support): the robust shared fit tolerates a few wrong ones, and Bournemouth's lines are weak
    (most frames sit at F1 0.4-0.55 against the 0.55 acceptance threshold)."""
    units = []
    for sg in segs:
        cache = pt.ClipCache(sg["path"], root / match / Path(sg["file"]).stem)
        for t in sample_frames(sg, per):
            r = store.get(f"{sg['file']}:{t}")
            if not r or "pan" not in r or not (r.get("ok") or (min_f1 is not None and (r.get("f1") or 0) >= min_f1
                                                               and (r.get("pmain") or 0) >= 0.5)):
                continue
            ev = cache.evidence(t)
            if ev is None:
                continue
            st = np.array([[r["pan"], r["tilt"], r["logf"]]])
            units.append(pt.Unit(len(units), [t], st, {t: ev}))
    return units


def cmd_shared(args):
    """Rig quantities (C, roll, k1) from the line evidence of frames sampled from EVERY broadcast segment of the match
    (+ the motion pairs of the evaluation clip), no human labels; compared with the GT-fitted rig."""
    root = ROOT / "shared"
    root.mkdir(parents=True, exist_ok=True)
    rig = leave_clip_out_rig(args.clip)
    sp = root / f"anchors_{args.match}.json"
    store = json.loads(sp.read_text()) if sp.exists() else {}
    segs = match_segments(args.match)
    log(f"{len(segs)} segments of {args.match}")
    n = 0
    for sg in segs:
        for t in sample_frames(sg, args.per):
            if f"{sg['file']}:{t}" in store:
                continue
            shared_anchor(args.match, sg, t, rig, store, root)
            n += 1
            if n % 5 == 0:
                sp.write_text(json.dumps(store))
                log(f"  {len(store)} frames, {sum(v.get('ok', False) for v in store.values())} anchored")
    sp.write_text(json.dumps(store))
    units = shared_units(args.match, segs, store, root, args.per)
    log(f"{len(units)} anchored frames from {len({u.uid for u in units})} samples")
    res = {"n_units": len(units), "rig_prior": {"C": rig.C.tolist(), "roll_deg": float(np.degrees(rig.roll)), "k1": rig.k1}}
    full_rig = ptz.Rig.from_json(RIGS_DIR / ("vitality.json" if args.match == "bmth" else "emirates.json"))
    res["rig_gt_all_frames"] = {"C": full_rig.C.tolist(), "roll_deg": float(np.degrees(full_rig.roll)), "k1": full_rig.k1}
    for tag, with_clip in (("lines_only", False), ("lines_plus_motion", True)):
        if with_clip and not (args.motion or args.match in ("mancity", "villa")):
            continue
        us = list(units)
        if with_clip:
            rig2, cache, shot_info, anchors, pairs, segs_ = load_clip(args.clip)
            stf = np.load(wd(args.clip) / "states_full.npy")
            fr = [t for t in range(0, cache.n, 5) if np.all(np.isfinite(stf[t]))]
            ev = {t: cache.evidence(t) for t in fr if cache.evidence(t) is not None}
            us.append(pt.Unit(uid, fr, stf[fr], ev, [p for p in pairs if p.i % 5 == 0 and p.j - p.i == 5]))
        t0 = time.time()
        shn, _, info = pt.fit_shared(us, rig, PITCH, log=log)
        res[tag] = {"C": shn.C.tolist(), "roll_deg": float(np.degrees(shn.roll)), "k1": shn.k1, "info": info,
                    "secs": time.time() - t0}
        log(tag, res[tag])
        (root / f"shared_{args.match}.json").write_text(json.dumps(res, indent=1, default=float))


def _vec3(txt: str | None) -> np.ndarray:
    return np.array([float(x) for x in txt.split(",")]) if txt else np.zeros(3)


def cmd_sharedms(args):
    """Shared (C, roll, k1) fit from several starts of C (``pt.fit_shared_multistart``) over the cached anchored frames of
    ``--match`` (run ``shared`` first). ``--rig`` JSON (default: the leave-``--clip``-out Emirates rig) is the prior;
    ``--perturb dx,dy,dz`` moves its C (a deliberately bad start); ``--single`` runs the plain one-start fit instead;
    ``--free`` / ``--k1`` fix quantities (k1 study); ``--motion`` adds the motion pairs of ``--clip`` (Emirates only)."""
    root = ROOT / "shared"
    rig = ptz.Rig.from_json(args.rig) if args.rig else leave_clip_out_rig(args.clip)
    pert = _vec3(args.perturb)
    rig = ptz.Rig(rig.C + pert, rig.roll, rig.k1 if args.k1 is None else args.k1, (5.0, 10.0, 3.0), rig.pitch, rig.info)
    store = json.loads((root / f"anchors_{args.match}.json").read_text())
    segs = match_segments(args.match)
    units = shared_units(args.match, segs, store, root, args.per, args.min_f1 if args.min_f1 is not None else
                         (0.45 if args.match == "bmth" else None))
    free = tuple(args.free.split(",")) if args.free else ("Cx", "Cy", "Cz", "roll", "k1")
    states_at = pt.Shared.from_rig(ptz.Rig.from_json(args.rig) if args.rig else leave_clip_out_rig(args.clip))
    if args.motion:
        rig2, cache, shot_info, anchors, pairs, segs_ = load_clip(args.clip)
        stf = np.load(wd(args.clip) / "states_full.npy")
        fr = [t for t in range(0, cache.n, 5) if np.all(np.isfinite(stf[t]))]
        ev = {t: cache.evidence(t) for t in fr if cache.evidence(t) is not None}
        units.append(pt.Unit(len(units), fr, stf[fr], ev, [p for p in pairs if p.i % 5 == 0 and p.j - p.i == 5]))
    log(f"{len(units)} units, prior C {np.round(rig.C, 2).tolist()} k1 {rig.k1:.4f} free {free}")
    t0 = time.time()
    if args.single:
        us = [pt.replace(u, states=pt.reseed_states(u.states, states_at, pt.Shared.from_rig(rig))) for u in units]
        shn, _, info = pt.fit_shared(us, rig, PITCH, free=free, n_iter=args.iters, log=log)
    else:
        offs = pt.seed_offsets(step=tuple(_vec3(args.step)), n=tuple(int(k) for k in _vec3(args.nseed)))
        shn, _, info = pt.fit_shared_multistart(units, rig, PITCH, states_at=states_at, offsets=offs, top_k=args.top_k, screen_stride=args.stride,
                                                free=free, n_iter=args.iters, log=log)
    res = {"prior_C": rig.C.tolist(), "perturb": pert.tolist(), "free": list(free), "C": shn.C.tolist(),
           "roll_deg": float(np.degrees(shn.roll)), "k1": shn.k1, "cost": info["stage2"]["cost"], "n_units": len(units),
           "secs": time.time() - t0, "starts": info.get("starts")}
    tag = args.tag or f"{args.match}_{'single' if args.single else 'ms'}"
    (root / f"ms_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    log("result", {k: v for k, v in res.items() if k != "starts"})


def cmd_jitter(args):
    """Per-frame lines-only (a) on consecutive frames of the most static span vs the temporal solve there."""
    rig, cache, shot_info, anchors, pairs, segs = load_clip(args.clip)
    sh = pt.Shared.from_rig(rig)
    flow = {p.i: float(np.median(np.linalg.norm(p.b - p.a, axis=1))) for p in pairs if p.j - p.i == 1 and p.ok}
    n = args.n
    best = None
    for t in sorted(flow):
        w = [flow.get(t + k, 9.0) for k in range(n)]
        if max(w) < 2.0 and (best is None or sum(w) < best[0]):
            best = (sum(w), t)
    if best is None:
        log("no static span found")
        return
    t0 = best[1]
    log(f"static span {t0}..{t0 + n} (summed flow {best[0]:.2f} px)")
    jp = wd(args.clip) / "jitter_lines.json"
    rec = json.loads(jp.read_text()) if jp.exists() else {}
    for t in range(t0, t0 + n + 1):
        if str(t) in rec:
            continue
        if not cache.mask_path(t).exists():
            cache._save_masks(t, cache.read_full(t))
        rec[str(t)] = pt.lines_anchor(cache.evidence(t), rig, PITCH)
        jp.write_text(json.dumps(rec))
        log("lines-only", t, rec[str(t)]["ok"])
    stf = np.load(wd(args.clip) / "states_full.npy")

    def jumps(get):
        u = []
        for t in range(t0, t0 + n + 1):
            c = get(t)
            u.append(c.pitch_to_image(np.zeros((1, 2)))[0][0] * ptz.CANON if c is not None else [np.nan, np.nan])
        u = np.array(u)
        return (np.linalg.norm(np.diff(u, axis=0), axis=1).tolist(),
                np.linalg.norm(u[2:] - 2 * u[1:-1] + u[:-2], axis=1).tolist())

    def lines_cam(t):
        r = rec.get(str(t))
        if not r or "pan" not in r:
            return None
        cam0 = ptz.Camera(r["pan"], r["tilt"], r["logf"], rig.roll, np.array(rig.C, float), rig.k1)
        return ptz._refine_lines(cam0, cache.evidence(t), rig, PITCH)  # C free inside its prior, as the search returns it

    def temporal_cam(t):
        return pt.cam_from_state(stf[t], sh) if np.all(np.isfinite(stf[t])) else None

    jl, al = jumps(lines_cam)
    jt, at = jumps(temporal_cam)
    out = {"span": [t0, t0 + n], "flow_px_canon_median": float(2 * np.median([flow[t] for t in range(t0, t0 + n)])),
           "lines_only": {"jump_px": jl, "accel_px": al, "accel_median": float(np.nanmedian(al)),
                          "accel_max": float(np.nanmax(al))},
           "temporal": {"jump_px": jt, "accel_px": at, "accel_median": float(np.nanmedian(at)),
                        "accel_max": float(np.nanmax(at))},
           "lines_only_accepted": [bool(rec[str(t)]["ok"]) for t in range(t0, t0 + n + 1)]}
    (wd(args.clip) / "jitter.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "solve", "eval", "sheet", "shared", "sharedms", "jitter"])
    ap.add_argument("--clip", default="arsenal_mancity_seg010")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--match", default="mancity")
    ap.add_argument("--per", type=int, default=1)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--anchor-step", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rig", default=None, help="rig JSON (prior / start); default leave-clip-out Emirates (Vitality for bmth)")
    ap.add_argument("--variants", default="full,hold,sweeps,est,nomotion", help="solve: variants to run")
    ap.add_argument("--est", default=None, help="solve: JSON with the shared-fit rig (C, roll_deg, k1) for the 'est' variant")
    ap.add_argument("--est-key", default=None)
    ap.add_argument("--est-tag", default="", help="solve: suffix of the est variant names (est<tag>, esthold<tag>)")
    ap.add_argument("--perturb", default=None, help="dx,dy,dz added to the rig C (sharedms)")
    ap.add_argument("--k1", type=float, default=None, help="override the rig k1 (fixed unless k1 is in --free)")
    ap.add_argument("--free", default=None, help="comma list of Cx,Cy,Cz,roll,k1 (sharedms)")
    ap.add_argument("--single", action="store_true", help="sharedms: plain single-start fit")
    ap.add_argument("--motion", action="store_true", help="sharedms: add the motion pairs of --clip")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--step", default="3,4,3", help="sharedms seed grid step (m)")
    ap.add_argument("--nseed", default="1,2,1", help="sharedms seed grid half-widths (cells)")
    ap.add_argument("--min-f1", type=float, default=None, help="sharedms: also use abstained frames with F1 >= this (default 0.45 for bmth)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--stride", type=int, default=2, help="sharedms: screen seeds on every n-th unit")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    global RIG_PATH, WORK_SUFFIX
    if args.rig and args.cmd != "sharedms":
        RIG_PATH = Path(args.rig)
        WORK_SUFFIX = f"@{RIG_PATH.stem}"
    if args.anchor_step:
        global ANCHOR_STEP
        ANCHOR_STEP = args.anchor_step
    if args.cmd == "prep":
        load_clip(args.clip)
    elif args.cmd == "solve":
        cmd_solve(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "sheet":
        cmd_sheet(args)
    elif args.cmd == "shared":
        cmd_shared(args)
    elif args.cmd == "sharedms":
        cmd_sharedms(args)
    elif args.cmd == "jitter":
        cmd_jitter(args)


if __name__ == "__main__":
    main()
