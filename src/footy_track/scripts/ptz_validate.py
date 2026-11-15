"""V1 offline validation of the PTZ rig hypothesis (docs/design/calibration_ptz.md section 4).

Emirates GT only (Man City + Aston Villa matches), no LLM calls, one process, no multiprocessing::

    python -m footy_track.scripts.ptz_validate rigs      --out DIR   # rig fits: all frames, leave-match-out, leave-shot-out, bootstrap
    python -m footy_track.scripts.ptz_validate points    --out DIR   # (a) all GT points, (b) random 2/3-pt, (c) near-collinear + baselines
    python -m footy_track.scripts.ptz_validate lines     --out DIR   # (d) lines-only semantic multi-hypothesis search on the video frame
    python -m footy_track.scripts.ptz_validate zoom      --out DIR   # C vs focal length (axis slope), per-frame k1 from the line mask
    python -m footy_track.scripts.ptz_validate synthetic --out DIR   # the same machinery on known synthetic cameras
    python -m footy_track.scripts.ptz_validate report    --out DIR   # markdown tables from the JSON results

Metric: pitch-metre error against the stored GT homography (+ its own k1) over a 3 m grid of visible pitch points
(GT image point -> pitch under the estimate vs the grid point), split into the upper half (far) and lower half
(near) of the *visible* grid by image row. Folds: ``match`` (rig from one match, tested on the other) and ``shot``
(rig from the other clips of the same match; clip == shot group, the conservative grouping since the clips are
pre-cut to 1-3 shots).
"""

# ruff: noqa: PLR0915, PLR0912, PLC0415, E702
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration import ptz
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.refine import decompose_G

GT_ROOT = Path(os.environ.get("FOOTY_GT_MARKS_DIR", "/mnt/storage/footy_data/ball_gt_marks"))
CLIP_DIRS = [Path("/mnt/storage/footy_data/ballcheck_clips"), Path("/home/george/footy-track-calib/eval_data/clips")]
PITCH = PitchModel()
SIGMAS = {"fixed": (0.0, 0.0, 0.0), "tight": (0.3, 1.0, 0.3), "mid": (1.0, 3.0, 1.0), "loose": (5.0, 10.0, 3.0),
          "free": (100.0, 100.0, 100.0)}
FAIL_M = 5.0  # a frame whose median grid error exceeds this is a catastrophic failure
CAP_M = 100.0


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@dataclass
class GTFrame:
    clip: str
    match: str
    frame: int
    ids: list[str]
    img: np.ndarray  # (n,2) observed normalised
    pitch: np.ndarray  # (n,2) metres
    H: np.ndarray
    dist: dict | None
    size: tuple[int, int]
    ratio: float | None  # |r1|/|r2| of the GT H decomposed as a square-pixel pinhole (1 = physical)
    source: str

    @property
    def key(self) -> str:
        return f"{self.clip}:{self.frame}"

    @property
    def plausible(self) -> bool:
        return self.ratio is not None and 0.8 <= self.ratio <= 1.25

    def truth(self) -> HTruth:
        return HTruth(self.H, self.dist)


class HTruth:
    """Ground truth from a stored GT homography (+ distortion)."""

    def __init__(self, H, dist):
        self.H, self.dist = np.asarray(H, float), dist

    def to_image(self, xy):
        uv, w = geo.pitch_to_image(self.H, xy, self.dist)
        return uv, (w > 1e-6) & np.all(np.isfinite(uv), axis=1)

    def to_pitch(self, uv):
        xy, w = geo.image_to_pitch(self.H, uv, self.dist)
        return xy, np.isfinite(xy).all(axis=1)


class CamTruth:
    """Ground truth from a known synthetic camera."""

    def __init__(self, cam: ptz.Camera):
        self.cam = cam

    def to_image(self, xy):
        return self.cam.pitch_to_image(xy)

    def to_pitch(self, uv):
        return self.cam.image_to_pitch(uv)


def match_of(clip: str) -> str | None:
    if clip.startswith("arsenal_mancity"):
        return "mancity"
    if clip.startswith("arsenal_astonvilla"):
        return "villa"
    if clip.startswith(("arsenal_bournemouth", "bournemouth")):
        return "bmth"
    return None


def load_gt(matches: tuple[str, ...] = ("mancity", "villa")) -> list[GTFrame]:
    """GT frames with >= 4 correspondences of the given matches (V1: the Emirates ones; V2 passes ``("bmth",)``)."""
    out = []
    for f in sorted((GT_ROOT / "calibration_gt").glob("*.json")):
        m = match_of(f.stem)
        if m is None or m not in matches:
            continue
        for k, r in sorted(json.loads(f.read_text())["frames"].items(), key=lambda kv: int(kv[0])):
            cs = r.get("correspondences") or []
            if len(cs) < 4 or not r.get("H"):
                continue
            H = np.array(r["H"], float)
            size = tuple(r.get("image_size") or (1920, 1080))
            G = np.linalg.inv(H @ np.diag([1 / size[0], 1 / size[1], 1]))
            dec = decompose_G(G / np.linalg.norm(G), size)
            src = (r.get("origin") or {}).get("source") or "hand"
            out.append(GTFrame(
                f.stem, m, int(k), [c["pitch_id"] for c in cs], np.array([c["image"] for c in cs], float),
                np.array([c["pitch_xy"] for c in cs], float), H, r.get("distortion"), size,
                None if dec is None else float(dec[3]), src))
    return out


def gt_grid(truth, step: float = 3.0, margin: float = 0.02):
    """Visible pitch grid under the truth: -> pitch pts (N,2), observed image pts (N,2), near mask (N,) (lower half)."""
    gx, gy = np.meshgrid(np.arange(-52.5, 52.51, step), np.arange(-34.0, 34.01, step))
    P = np.column_stack([gx.ravel(), gy.ravel()])
    uv, ok = truth.to_image(P)
    ok &= np.all((uv > margin) & (uv < 1 - margin), axis=1)
    P, uv = P[ok], uv[ok]
    if len(P) < 6:
        return P, uv, np.zeros(len(P), bool)
    return P, uv, uv[:, 1] >= np.median(uv[:, 1])


def error_record(map_fn, truth, n_sub: int = 40) -> dict:
    """Pitch error (m) of ``map_fn(uv_norm) -> (xy, ok)`` over the truth's visible grid."""
    P, uv, near = gt_grid(truth)
    if len(P) < 6:
        return {"failed": True, "reason": "no visible grid"}
    xy, ok = map_fn(uv)
    e = np.linalg.norm(xy - P, axis=1)
    e = np.where(ok & np.isfinite(e), np.minimum(e, CAP_M), CAP_M)
    idx = np.unique(np.linspace(0, len(P) - 1, min(n_sub, len(P))).astype(int))
    return {"failed": False, "med": float(np.median(e)), "p90": float(np.percentile(e, 90)),
            "near_med": float(np.median(e[near])) if near.any() else None,
            "far_med": float(np.median(e[~near])) if (~near).any() else None,
            "err": [round(float(v), 2) for v in e[idx]], "near": [bool(v) for v in near[idx]]}


def cam_map(cam: ptz.Camera):
    return lambda uv: cam.image_to_pitch(uv)


def h_map(H, k1):
    def f(uv):
        u = ptz.undistort_px(uv * ptz.CANON, k1) / ptz.CANON
        xy, w = geo.apply_homography(H, u)
        return xy, np.isfinite(xy).all(axis=1)

    return f


# --------------------------------------------------------------------------
# rigs
# --------------------------------------------------------------------------


def to_rigframes(frames: list[GTFrame]) -> list[ptz.RigFrame]:
    return [ptz.RigFrame(f.img, f.pitch, f.key, f.clip) for f in frames]


_INIT: dict[str, ptz.Camera] = {}  # per-frame (pan, tilt, f) of earlier fits: warm start only (C is re-estimated)


def fit(frames: list[GTFrame], **kw):
    kw.setdefault("sigma_C", SIGMAS["mid"])
    init = [_INIT.get(f.key) for f in frames]
    rig, cams = ptz.fit_rig(to_rigframes(frames), init=init if all(c is not None for c in init) else None, **kw)
    for f, c in zip(frames, cams, strict=True):
        _INIT.setdefault(f.key, c)
    return rig, cams


def rig_dict(rig: ptz.Rig) -> dict:
    return {"C": [round(float(c), 3) for c in rig.C], "roll_deg": float(np.degrees(rig.roll)), "k1": float(rig.k1),
            "info": rig.info}


def bootstrap_rig(frames: list[GTFrame], n_boot: int, seed: int = 0) -> dict:
    """Bootstrap over shots (clips): spread of C, roll, k1."""
    rng = np.random.default_rng(seed)
    clips = sorted({f.clip for f in frames})
    Cs, rolls, k1s = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(clips, len(clips))
        fr = [f for c in pick for f in frames if f.clip == c]
        try:
            rig, _ = fit(fr)
        except Exception:  # noqa: BLE001
            continue
        Cs.append(rig.C)
        rolls.append(np.degrees(rig.roll))
        k1s.append(rig.k1)
    Cs = np.array(Cs)
    return {"n": len(Cs), "C_mean": Cs.mean(0).tolist(), "C_std": Cs.std(0).tolist(),
            "C_p5": np.percentile(Cs, 5, axis=0).tolist(), "C_p95": np.percentile(Cs, 95, axis=0).tolist(),
            "roll_std_deg": float(np.std(rolls)), "k1_std": float(np.std(k1s)),
            "corr_Cy_Cz": float(np.corrcoef(Cs[:, 1], Cs[:, 2])[0, 1])}


def cmd_rigs(args):
    gt = load_gt()
    out = {"frames": [{"key": f.key, "match": f.match, "n": len(f.ids), "ratio": f.ratio, "src": f.source}
                      for f in gt]}
    by = {m: [f for f in gt if f.match == m] for m in ("mancity", "villa")}
    for m, fr in by.items():
        t = time.time()
        rig, cams = fit(fr)
        out[f"rig_{m}"] = rig_dict(rig)
        out[f"rig_{m}"]["per_frame"] = [
            {"key": f.key, "rms_px": c.info["rms_px"], "f": c.f, "pan": float(np.degrees(c.pan)),
             "tilt": float(np.degrees(c.tilt))} for f, c in zip(fr, cams, strict=True)]
        plaus = [f for f in fr if f.plausible]
        rig_p, _ = fit(plaus)
        out[f"rig_{m}_plausible"] = rig_dict(rig_p)
        print(m, "all", rig_dict(rig)["C"], "plausible", rig_dict(rig_p)["C"], f"{time.time() - t:.0f}s", flush=True)
        rig.to_json(Path(args.out) / f"rig_{m}.json")
    out["boot_mancity"] = bootstrap_rig(by["mancity"], args.boot)
    print("bootstrap mancity", out["boot_mancity"], flush=True)
    out["boot_villa"] = bootstrap_rig(by["villa"], args.boot)
    print("bootstrap villa", out["boot_villa"], flush=True)
    (Path(args.out) / "rigs.json").write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------


def make_folds(gt: list[GTFrame], train_filter: str = "all") -> list[tuple[str, str, ptz.Rig, list[GTFrame]]]:
    """-> [(fold kind, fold id, rig, test frames)]: leave-match-out both ways, leave-shot-out per clip."""
    sel = (lambda fr: [f for f in fr if f.plausible]) if train_filter == "plausible" else (lambda fr: fr)
    folds = []
    for tr, te in (("mancity", "villa"), ("villa", "mancity")):
        rig, _ = fit(sel([f for f in gt if f.match == tr]))
        folds.append(("match", f"{tr}->{te}", rig, [f for f in gt if f.match == te]))
    for m in ("mancity", "villa"):
        fm = [f for f in gt if f.match == m]
        for clip in sorted({f.clip for f in fm}):
            train = sel([f for f in fm if f.clip != clip])
            if len(train) < 6:
                continue
            rig, _ = fit(train)
            folds.append(("shot", clip, rig, [f for f in fm if f.clip == clip]))
    return folds


def get_folds(args, gt):
    cache = Path(args.out) / f"folds_{args.train_filter}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        byk = {f.key: f for f in gt}
        folds = []
        for k, fid, rd, keys in d:
            rig = ptz.Rig(np.array(rd["C"]), np.radians(rd["roll_deg"]), rd["k1"], SIGMAS["mid"])
            folds.append((k, fid, rig, [byk[x] for x in keys]))
        return folds
    folds = make_folds(gt, args.train_filter)
    cache.write_text(json.dumps([(k, fid, rig_dict(r), [f.key for f in te]) for k, fid, r, te in folds]))
    for k, fid, r, te in folds:
        print("fold", k, fid, np.round(r.C, 1), f"k1={r.k1:.3f}", f"n_test={len(te)}", flush=True)
    return folds


# --------------------------------------------------------------------------
# points experiments (a) (b) (c) + baselines
# --------------------------------------------------------------------------

CENTRE_IDS = {"CENTRE_SPOT", "HALFWAY_TOP", "HALFWAY_BOTTOM", "CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM",
              "CENTRE_CIRCLE_LEFT", "CENTRE_CIRCLE_RIGHT"}


def collinear_groups(f: GTFrame) -> list[list[int]]:
    """Index groups of >= 2 labelled points lying on one pitch line (shared x or y, within 0.05 m)."""
    groups = []
    for ax in (0, 1):
        for v in sorted({round(float(p[ax]), 1) for p in f.pitch}):
            g = [i for i, p in enumerate(f.pitch) if abs(p[ax] - v) < 0.05]
            if len(g) >= 2 and g not in groups:
                groups.append(g)
    return groups


def free_h(img, pitch, k1):
    """Free 8-DOF homography baseline (DLT, same lens k1). None when under-determined or degenerate."""
    if len(img) < 4:
        return None
    try:
        u = ptz.undistort_px(img * ptz.CANON, k1) / ptz.CANON
        H, _ = geo.fit_homography(u, pitch, "dlt")
    except ValueError:
        return None
    return H


def run_methods(f: GTFrame, idx, rig: ptz.Rig, truth, rec_base: dict, sink: list, methods: set[str]):
    """One trial: solve from correspondences ``idx`` of frame ``f`` with the PTZ rig (3 tolerance settings), the
    free-C pinhole and the free-H baselines; append one record per method."""
    img, pit = f.img[idx], f.pitch[idx]
    n = len(idx)
    for name in ("ptz_fixed", "ptz_tight", "ptz_mid", "ptz_loose"):
        if name not in methods:
            continue
        cam = ptz.solve_points(img, pit, rig.with_sigma(SIGMAS[name[4:]]))
        rec = error_record(cam_map(cam), truth) if cam is not None else {"failed": True, "reason": "no solution"}
        sink.append({**rec_base, "method": name, "n_pts": n, **rec})
    if "pinhole_freeC" in methods:
        cam = ptz.solve_points(img, pit, rig.with_sigma(SIGMAS["free"])) if n >= 4 else None
        rec = error_record(cam_map(cam), truth) if cam is not None else {"failed": True, "reason": "n<4"}
        sink.append({**rec_base, "method": "pinhole_freeC", "n_pts": n, **rec})
    if "free_H" in methods:
        H = free_h(img, pit, rig.k1)
        rec = error_record(h_map(H, rig.k1), truth) if H is not None else {"failed": True, "reason": "n<4/degenerate"}
        sink.append({**rec_base, "method": "free_H", "n_pts": n, **rec})


def cmd_points(args):
    gt = load_gt()
    folds = get_folds(args, gt)
    rng = np.random.default_rng(args.seed)
    sink: list[dict] = []
    allm = {"ptz_fixed", "ptz_tight", "ptz_mid", "ptz_loose", "pinhole_freeC", "free_H"}
    t0 = time.time()
    for kind, fid, rig, test in folds:
        for f in test:
            truth = f.truth()
            base = {"fold": kind, "fold_id": fid, "key": f.key, "match": f.match, "plausible": f.plausible,
                    "n_gt": len(f.ids)}
            n = len(f.ids)
            # (a) all GT points; no-lens-prior ablation for the mid tolerance
            run_methods(f, np.arange(n), rig, truth, {**base, "exp": "a"}, sink, allm)
            cam = ptz.solve_points(f.img, f.pitch, rig.with_sigma(SIGMAS["mid"]).with_k1(0.0))
            if cam is not None:
                sink.append({**base, "exp": "a", "method": "ptz_mid_k1zero", "n_pts": n, **error_record(cam_map(cam), truth)})
            # (b) random subsets
            for k in (2, 3, 4, 5):
                if k >= n:
                    continue
                for _ in range(args.draws):
                    idx = np.sort(rng.choice(n, k, replace=False))
                    run_methods(f, idx, rig, truth, {**base, "exp": f"b{k}"}, sink, allm)
            # (c) near-collinear subsets
            cidx = [i for i, p in enumerate(f.ids) if p in CENTRE_IDS]
            subsets = []
            if len(cidx) >= 2:
                subsets.append(("c_centre", cidx))
            for g in collinear_groups(f):
                others = [i for i in range(n) if i not in g]
                for o in others[:6]:
                    subsets.append(("c_line1", sorted([*g, o])))
            rng.shuffle(subsets)
            for tag, idx in subsets[: args.max_c]:
                run_methods(f, np.array(idx), rig, truth, {**base, "exp": tag}, sink, allm)
        print(f"fold {kind} {fid}: {len(sink)} records {time.time() - t0:.0f}s", flush=True)
    (Path(args.out) / f"points_{args.train_filter}.json").write_text(json.dumps(sink))


# --------------------------------------------------------------------------
# lines experiment (d)
# --------------------------------------------------------------------------


def find_clip_video(clip: str):
    from footy_track.calibration.prelabel.pipeline import find_video

    for d in CLIP_DIRS:
        if d.is_dir() and (v := find_video(d, clip)) is not None:
            return v
    return None


def frame_evidence(f: GTFrame, cache_dir: Path) -> ptz.LineEvidence | None:
    import cv2

    from footy_track.calibration.prelabel.candidates import (
        grass_mask,
        line_mask,
        resize_to_work,
    )
    from footy_track.calibration.video import read_frame

    cp = cache_dir / f"{f.clip}_{f.frame}.npz"
    if cp.exists():
        d = np.load(cp)
        lines, grass = d["lines"], d["grass"]
    else:
        v = find_clip_video(f.clip)
        if v is None:
            return None
        bgr = read_frame(v, f.frame)
        if bgr is None:
            return None
        work, _ = resize_to_work(bgr)
        grass = grass_mask(work)
        lines = line_mask(work, grass)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, lines=lines, grass=grass)
        cv2.imwrite(str(cache_dir / f"{f.clip}_{f.frame}.jpg"), work)
    return ptz.LineEvidence.from_masks(lines, grass)


def cmd_lines(args):
    gt = load_gt()
    folds = get_folds(args, gt)
    cache = Path(args.out) / "masks"
    sink: list[dict] = []
    variants = args.variants.split(",")
    t0 = time.time()
    for kind, fid, rig, test in folds:
        for f in test:
            if kind == "match" and fid != "mancity->villa":
                continue  # reverse direction has no extra information for (d); shot folds cover every frame
            ev = frame_evidence(f, cache)
            if ev is None:
                print("no video for", f.key, flush=True)
                continue
            truth = f.truth()
            for var in variants:
                if kind == "match" and var != "mid":
                    continue
                r = rig.with_sigma(SIGMAS[var])
                t = time.time()
                cam, hyps, info = ptz.search_lines(ev, r)
                base = {"exp": "d", "fold": kind, "fold_id": fid, "key": f.key, "match": f.match,
                        "plausible": f.plausible, "variant": var, "method": f"lines_{var}", "n_gt": len(f.ids),
                        "secs": time.time() - t, "margin": info.get("margin"), "support": info.get("support"),
                        "reason": info.get("reason"), "n_distinct": info.get("n_distinct")}
                if hyps:  # score and error of the best hypothesis even when it abstained (to study the abstain rule)
                    best = hyps[0].cam
                    base["best_err"] = error_record(cam_map(best), truth)
                    base["best_params"] = best.params()
                if cam is None:
                    sink.append({**base, "failed": True, "abstain": True})
                else:
                    sink.append({**base, "abstain": False, **error_record(cam_map(cam), truth)})
        print(f"fold {kind} {fid}: {len(sink)} records {time.time() - t0:.0f}s", flush=True)
        (Path(args.out) / "lines.json").write_text(json.dumps(sink))
    (Path(args.out) / "lines.json").write_text(json.dumps(sink))


# --------------------------------------------------------------------------
# zoom: C vs f, k1
# --------------------------------------------------------------------------


def cmd_zoom(args):
    gt = load_gt()
    out = {}
    for m in ("mancity", "villa", "both"):
        fr = [f for f in gt if (m in ("both", f.match))]
        rig0, c0 = fit(fr)
        rig1, c1 = fit(fr, fit_axis_slope=True)
        fs = np.array([c.f for c in c0])
        rms0 = np.sqrt(np.mean([c.info["rms_px"] ** 2 for c in c0]))
        rms1 = np.sqrt(np.mean([c.info["rms_px"] ** 2 for c in c1]))
        # per-frame C from the plain GT decomposition vs f (the table of the proposal), plausible frames only
        dec = []
        for f in fr:
            if not f.plausible:
                continue
            G = np.linalg.inv(f.H @ np.diag([1 / f.size[0], 1 / f.size[1], 1]))
            d = decompose_G(G / np.linalg.norm(G), f.size)
            dec.append((d[0] * ptz.CANON_W / f.size[0], *d[2]))
        dec = np.array(dec)
        from scipy.stats import spearmanr

        out[m] = {"n": len(fr), "rig_C": rig0.C.tolist(), "rig_C_slope": rig1.C.tolist(),
                  "axis_slope_m_per_logf": rig1.info.get("axis_slope"), "rms_px_no_slope": float(rms0),
                  "rms_px_slope": float(rms1), "cost_no_slope": rig0.info["cost"], "cost_slope": rig1.info["cost"],
                  "f_range": [float(fs.min()), float(fs.max())],
                  "decomp_spearman_f_vs_Cxyz": [float(spearmanr(dec[:, 0], dec[:, j]).statistic) for j in (1, 2, 3)],
                  "decomp_n": len(dec)}
        print(m, json.dumps(out[m]), flush=True)
    # bootstrap the slope (shots) on the full set
    rng = np.random.default_rng(0)
    clips = sorted({f.clip for f in gt})
    sl = []
    for _ in range(args.boot):
        pick = rng.choice(clips, len(clips))
        fr = [f for c in pick for f in gt if f.clip == c]
        try:
            r, _ = fit(fr, fit_axis_slope=True)
            sl.append(r.info["axis_slope"])
        except Exception:  # noqa: BLE001
            pass
    out["axis_slope_boot"] = {"n": len(sl), "mean": float(np.mean(sl)), "std": float(np.std(sl)),
                              "p5": float(np.percentile(sl, 5)), "p95": float(np.percentile(sl, 95))}
    print("axis slope bootstrap", out["axis_slope_boot"], flush=True)
    # per-frame k1 from the white-line mask: refine (pan, tilt, log f, k1) at the full-data rig, C fixed
    rig, cams = fit(gt)
    rig = rig.with_sigma(SIGMAS["fixed"])
    cache = Path(args.out) / "masks"
    k1s = []
    for f, c in zip(gt, cams, strict=True):
        ev = frame_evidence(f, cache)
        if ev is None:
            continue
        cam = ptz.Camera(c.pan, c.tilt, c.logf, rig.roll, rig.C.copy(), rig.k1)
        try:
            cam0 = ptz._refine_lines(cam, ev, rig, PITCH, fit_k1=False)
            cam1 = ptz._refine_lines(cam0, ev, rig, PITCH, fit_k1=True)
        except Exception:  # noqa: BLE001
            continue
        s0, s1 = ptz.score_cameras([cam0, cam1], ev, PITCH)
        k1s.append({"key": f.key, "match": f.match, "f": cam1.f, "k1": cam1.k1, "score_k1fixed": float(s0),
                    "score_k1free": float(s1), "n": len(f.ids), "gt_k1": (f.dist or {}).get("k1")})
        print(f.key, f"f={cam1.f:.0f} k1={cam1.k1:+.3f} score {s0:.3f}->{s1:.3f}", flush=True)
    out["k1_lines"] = k1s
    if len(k1s) > 4:
        from scipy.stats import spearmanr

        a = np.array([(k["f"], k["k1"]) for k in k1s])
        out["k1_lines_summary"] = {"n": len(k1s), "median": float(np.median(a[:, 1])), "iqr": [float(np.percentile(a[:, 1], 25)), float(np.percentile(a[:, 1], 75))],
                                   "spearman_f_k1": float(spearmanr(a[:, 0], a[:, 1]).statistic)}
        print(out["k1_lines_summary"], flush=True)
    (Path(args.out) / "zoom.json").write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------
# synthetic
# --------------------------------------------------------------------------


def synth_frames(rng, n, rig_true, noise_px, pitch=PITCH):
    """Random known cameras at the rig with keypoint correspondences (noisy)."""
    frames, cams = [], []
    ids, xy = pitch.keypoint_array()
    while len(frames) < n:
        pan = np.radians(rng.uniform(55, 125))
        tilt = np.radians(rng.uniform(8, 24))
        f = float(np.exp(rng.uniform(np.log(2000), np.log(6000))))
        cam = ptz.Camera(pan, tilt, np.log(f), rig_true.roll, rig_true.C.copy(), rig_true.k1)
        uv, ok = cam.pitch_to_image(xy)
        ok &= np.all((uv > 0.03) & (uv < 0.97), axis=1)
        if ok.sum() < 4:
            continue
        sel = np.flatnonzero(ok)
        if len(sel) > 8:
            sel = rng.choice(sel, int(rng.integers(4, 9)), replace=False)
        img = uv[sel] + rng.normal(0, noise_px, (len(sel), 2)) / ptz.CANON
        frames.append(ptz.RigFrame(img, xy[sel], f"s{len(frames)}", f"s{len(frames)}"))
        cams.append(cam)
    return frames, cams


def cmd_synthetic(args):
    rng = np.random.default_rng(args.seed)
    true = ptz.Rig(np.array([0.5, -64.0, 15.0]), np.radians(0.4), 0.02)
    res = {"settings": {"C": true.C.tolist(), "k1": true.k1, "roll_deg": 0.4, "noise_px": args.noise}}
    fr, cams = synth_frames(rng, 30, true, args.noise)
    rig, _ = ptz.fit_rig(fr[:22], C0=(0.0, -65.0, 15.0), sigma_C=SIGMAS["mid"])
    res["rig_fit"] = {"C": rig.C.tolist(), "C_err_m": float(np.linalg.norm(rig.C - true.C)),
                      "k1": rig.k1, "roll_deg": float(np.degrees(rig.roll))}
    print("synthetic rig", res["rig_fit"], flush=True)
    # a wrong prior mean (2 m off) for the per-frame tests: the tolerance must absorb it
    prior = ptz.Rig(rig.C, rig.roll, rig.k1, SIGMAS["mid"])
    sink = []
    for fcam, frame in zip(cams[22:], fr[22:], strict=True):
        truth = CamTruth(fcam)
        gtf = GTFrame("syn", "syn", 0, [], frame.img, frame.pitch, np.eye(3), None, (1920, 1080), 1.0, "synthetic")
        base = {"key": frame.key, "fold": "synthetic", "plausible": True, "n_gt": len(frame.img)}
        n = len(frame.img)
        run_methods(gtf, np.arange(n), prior, truth, {**base, "exp": "a"}, sink,
                    {"ptz_fixed", "ptz_mid", "ptz_loose", "pinhole_freeC", "free_H"})
        for k in (2, 3):
            for _ in range(args.draws):
                idx = np.sort(rng.choice(n, k, replace=False))
                run_methods(gtf, idx, prior, truth, {**base, "exp": f"b{k}"}, sink,
                            {"ptz_fixed", "ptz_mid", "free_H"})
        # (d) lines with clutter and dropout
        lines, grass = ptz.render_synthetic(fcam, width=1568)
        lines = lines & (rng.random(lines.shape) > 0.15)
        import cv2

        for _ in range(8):  # clutter: bright edges that are not pitch lines (inside the grass region)
            ys, xs = np.nonzero(grass)
            i, j = rng.integers(0, len(ys), 2)
            cv2.line(lines.view(np.uint8), (int(xs[i]), int(ys[i])), (int(xs[j]), int(ys[j])), 1, 3)
        ev = ptz.LineEvidence.from_masks(lines, grass)
        for var in ("mid", "fixed"):
            cam, hyps, info = ptz.search_lines(ev, prior.with_sigma(SIGMAS[var]))
            rec = {**base, "exp": "d", "method": f"lines_{var}", "margin": info.get("margin"),
                   "reason": info.get("reason")}
            if hyps:
                rec["best_err"] = error_record(cam_map(hyps[0].cam), truth)
            sink.append({**rec, "abstain": True, "failed": True} if cam is None
                        else {**rec, "abstain": False, **error_record(cam_map(cam), truth)})
        print("frame", frame.key, "done", flush=True)
    res["records"] = sink
    (Path(args.out) / "synthetic.json").write_text(json.dumps(res))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def wquant(vals, w, q):
    vals, w = np.asarray(vals), np.asarray(w)
    if not len(vals):
        return float("nan")
    o = np.argsort(vals)
    cw = np.cumsum(w[o]) / w.sum()
    return float(vals[o][min(np.searchsorted(cw, q / 100.0), len(vals) - 1)])


def summarise(recs: list[dict]) -> dict:
    """Trial-level (median grid error per trial) and point-level (frame-weighted near/far) statistics."""
    n = len(recs)
    ok = [r for r in recs if not r.get("failed")]
    abst = sum(1 for r in recs if r.get("abstain"))
    fail = n - len(ok) - abst
    med = np.array([r["med"] for r in ok])
    cat = int((med > FAIL_M).sum())
    out = {"n": n, "fail": fail / n if n else float("nan"), "abstain": abst / n if n else float("nan"),
           "cat": cat / n if n else float("nan")}
    if ok:
        out.update(median=float(np.median(med)), p90=float(np.percentile(med, 90)), p95=float(np.percentile(med, 95)))
        ne, fe, nw, fw = [], [], [], []
        for r in ok:
            e, nr = np.array(r["err"]), np.array(r["near"])
            if nr.any():
                ne += list(e[nr]); nw += [1.0 / nr.sum()] * int(nr.sum())
            if (~nr).any():
                fe += list(e[~nr]); fw += [1.0 / (~nr).sum()] * int((~nr).sum())
        out.update(near_med=wquant(ne, nw, 50), far_med=wquant(fe, fw, 50), far_p90=wquant(fe, fw, 90),
                   near_p90=wquant(ne, nw, 90))
    return out


def fmt(s: dict) -> str:
    def g(k, p=2):
        v = s.get(k)
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"

    return (f"| {s['n']} | {100 * s['fail']:.0f}% | {100 * s['abstain']:.0f}% | {100 * s['cat']:.0f}% | {g('median')} | "
            f"{g('p90')} | {g('p95')} | {g('near_med')} | {g('far_med')} | {g('far_p90')} |")


HDR = ("| method | n | fail | abstain | >5 m | median | p90 | p95 | near med | far med | far p90 |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|")


def cmd_report(args):
    out = Path(args.out)
    lines = []
    for fname in ("points_all.json", "points_plausible.json"):
        p = out / fname
        if not p.exists():
            continue
        recs = json.loads(p.read_text())
        for fold in ("shot", "match"):
            for subset in ("all frames", "plausible GT only"):
                rs = [r for r in recs if r["fold"] == fold and (subset == "all frames" or r["plausible"])]
                if not rs:
                    continue
                lines.append(f"\n#### {fname} / leave-{fold}-out / {subset}\n")
                for exp in ("a", "b2", "b3", "b4", "b5", "c_centre", "c_line1"):
                    sub = [r for r in rs if r["exp"] == exp]
                    if not sub:
                        continue
                    lines.append(f"\n{exp}\n\n{HDR}")
                    for m in sorted({r["method"] for r in sub}):
                        lines.append(f"| {m} " + fmt(summarise([r for r in sub if r["method"] == m])))
    p = out / "lines.json"
    if p.exists():
        recs = json.loads(p.read_text())
        for fold in ("shot", "match"):
            for subset in ("all frames", "plausible GT only"):
                rs = [r for r in recs if r["fold"] == fold and (subset == "all frames" or r["plausible"])]
                if not rs:
                    continue
                lines.append(f"\n#### lines (d) / leave-{fold}-out / {subset}\n\n{HDR}")
                for m in sorted({r["method"] for r in rs}):
                    lines.append(f"| {m} " + fmt(summarise([r for r in rs if r["method"] == m])))
    lines += extra_tables(out)
    print("\n".join(lines))


def extra_tables(out: Path) -> list[str]:
    """Per-direction match folds, error by zoom tercile, and the frames that disagree most with the rig."""
    lines: list[str] = []
    p = out / "points_all.json"
    if not p.exists():
        return lines
    recs = json.loads(p.read_text())
    rigs = json.loads((out / "rigs.json").read_text()) if (out / "rigs.json").exists() else {}
    for fid in ("mancity->villa", "villa->mancity"):
        lines.append(f"\n#### leave-match-out, rig fitted on {fid.split('->')[0]}, tested on {fid.split('->')[1]}\n")
        for exp in ("a", "b2", "b3", "c_line1"):
            rs = [r for r in recs if r["fold"] == "match" and r["fold_id"] == fid and r["exp"] == exp]
            if rs:
                lines.append(f"\n{exp}\n\n{HDR}")
                for m in ("free_H", "pinhole_freeC", "ptz_mid"):
                    lines.append(f"| {m} " + fmt(summarise([r for r in rs if r["method"] == m])))
    # error by zoom tercile (leave-shot-out, ptz_mid)
    f_of = {}
    for m in ("mancity", "villa"):
        for pf in rigs.get(f"rig_{m}", {}).get("per_frame", []):
            f_of[pf["key"]] = pf["f"]
    if f_of:
        fs = np.array(sorted(f_of.values()))
        t1, t2 = np.percentile(fs, [33.3, 66.7])
        lines.append(f"\n#### leave-shot-out, ptz_mid by focal length (tercile edges {t1:.0f} / {t2:.0f} px)\n")
        for exp in ("a", "b2", "b3"):
            lines.append(f"\n{exp}\n\n{HDR}")
            for name, lo, hi in (("f low", 0, t1), ("f mid", t1, t2), ("f high", t2, 1e9)):
                rs = [r for r in recs if r["fold"] == "shot" and r["method"] == "ptz_mid" and r["exp"] == exp
                      and lo <= f_of.get(r["key"], -1) < hi]
                if rs:
                    lines.append(f"| {name} " + fmt(summarise(rs)))
    # frames that disagree most with the rig: (a) ptz_mid p90 / median over the leave-shot-out fold
    rs = [r for r in recs if r["fold"] == "shot" and r["exp"] == "a" and r["method"] == "ptz_mid" and not r.get("failed")]
    rs.sort(key=lambda r: -r["p90"])
    lines.append("\n#### frames the rig explains worst (leave-shot-out, all GT points, ptz_mid)\n")
    lines.append("| frame | n pts | GT source | pinhole ratio | median m | p90 m |\n|---|---|---|---|---|---|")
    gt = {f.key: f for f in load_gt()}
    for r in rs[:12]:
        g = gt[r["key"]]
        lines.append(f"| {r['key']} | {r['n_gt']} | {g.source} | {'-' if g.ratio is None else f'{g.ratio:.2f}'} | "
                     f"{r['med']:.2f} | {r['p90']:.2f} |")
    return lines


class Svg:
    """Tiny dependency-free SVG plot (the shared venv has no usable matplotlib)."""

    def __init__(self, title, xlabel, ylabel, xlim, ylim, logx=False, w=520, h=340):
        self.w, self.h, self.xlim, self.ylim, self.logx = w, h, xlim, ylim, logx
        self.m = (58, 14, 34, 40)  # left, right, top, bottom
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" '
                      f'font-size="11"><rect width="{w}" height="{h}" fill="white"/>',
                      f'<text x="{w / 2}" y="16" text-anchor="middle" font-size="13">{title}</text>',
                      f'<text x="{w / 2}" y="{h - 6}" text-anchor="middle">{xlabel}</text>',
                      f'<text transform="translate(12,{h / 2}) rotate(-90)" text-anchor="middle">{ylabel}</text>']
        lm, r, t, b = self.m
        self.parts.append(f'<rect x="{lm}" y="{t}" width="{w - lm - r}" height="{h - t - b}" fill="none" stroke="#999"/>')
        for v in np.linspace(*ylim, 5):
            y = self.py(v)
            self.parts.append(f'<line x1="{lm}" x2="{w - r}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eee"/>'
                              f'<text x="{lm - 4}" y="{y + 4:.1f}" text-anchor="end">{v:.3g}</text>')
        ticks = [0.1, 0.3, 1, 3, 10, 30, 100] if logx else np.linspace(*xlim, 6)
        for v in ticks:
            if xlim[0] <= v <= xlim[1]:
                x = self.px(v)
                self.parts.append(f'<line y1="{t}" y2="{h - b}" x1="{x:.1f}" x2="{x:.1f}" stroke="#eee"/>'
                                  f'<text x="{x:.1f}" y="{h - b + 14}" text-anchor="middle">{v:.3g}</text>')
        self.ly = t + 6

    def px(self, v):
        lm, r, _, _ = self.m
        a, b = self.xlim
        f = (np.log(v) - np.log(a)) / (np.log(b) - np.log(a)) if self.logx else (v - a) / (b - a)
        return lm + f * (self.w - lm - r)

    def py(self, v):
        _, _, t, b = self.m
        a, c = self.ylim
        return self.h - b - (v - a) / (c - a) * (self.h - t - b)

    def points(self, xs, ys, color, label):
        for x, y in zip(xs, ys, strict=True):
            self.parts.append(f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="2.6" fill="{color}" opacity="0.8"/>')
        self.legend(color, label)

    def line(self, xs, ys, color, label, dash=""):
        pts = " ".join(f"{self.px(max(x, self.xlim[0])):.1f},{self.py(y):.1f}" for x, y in zip(xs, ys, strict=True))
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" '
                          f'stroke-dasharray="{dash}"/>')
        self.legend(color, label)

    def hline(self, y, color):
        self.parts.append(f'<line x1="{self.m[0]}" x2="{self.w - self.m[1]}" y1="{self.py(y):.1f}" y2="{self.py(y):.1f}" '
                          f'stroke="{color}" stroke-dasharray="4 3"/>')

    def legend(self, color, label):
        x = self.m[0] + 8
        self.parts.append(f'<rect x="{x}" y="{self.ly - 8}" width="10" height="3" fill="{color}"/>'
                          f'<text x="{x + 14}" y="{self.ly - 4}">{label}</text>')
        self.ly += 13

    def save(self, path):
        Path(path).write_text("\n".join([*self.parts, "</svg>"]))


def cmd_plots(args):
    out = Path(args.out)
    gt = load_gt()
    rigs = json.loads((out / "rigs.json").read_text())
    cols = {"mancity": "#1b6ca8", "villa": "#c2571a"}
    dec_rows = {}
    for m in cols:
        rows = []
        for f in gt:
            if f.match != m or not f.plausible:
                continue
            G = np.linalg.inv(f.H @ np.diag([1 / f.size[0], 1 / f.size[1], 1]))
            d = decompose_G(G / np.linalg.norm(G), f.size)
            rows.append((d[0] * ptz.CANON_W / f.size[0], *d[2]))
        dec_rows[m] = np.array(rows)
    for j, (name, lim) in enumerate((("x", (-8, 8)), ("y", (-90, -50)), ("z", (10, 22)))):
        sv = Svg(f"Camera centre {name} (m) per GT frame vs zoom", "focal length (px at 1920 wide)", f"C {name} (m)",
                 (1500, 6000), lim)
        for m, col in cols.items():
            a = dec_rows[m]
            sv.points(a[:, 0], a[:, j + 1], col, f"{m}: per-frame decomposition")
            sv.hline(rigs[f"rig_{m}"]["C"][j], col)
        sv.save(out / f"C{name}_vs_f.svg")
    recs = json.loads((out / "points_all.json").read_text())
    sv = Svg("Leave-shot-out: error CDF", "median pitch error over visible grid (m, log)",
             "fraction of trials solved below x", (0.05, 100), (0, 1), logx=True)
    spec = (("b2", "ptz_mid", "PTZ 2 random pts", "#1b6ca8", ""), ("b3", "ptz_mid", "PTZ 3 random pts", "#2a9d8f", ""),
            ("c_line1", "ptz_mid", "PTZ line + 1 pt", "#8a5cc2", ""), ("b4", "ptz_mid", "PTZ 4 random pts", "#1b6ca8", "4 3"),
            ("b4", "free_H", "free H 4 random pts", "#c2571a", "4 3"), ("b5", "free_H", "free H 5 random pts", "#c2571a", "1 3"))
    for exp, meth, lab, col, dash in spec:
        rs = [r for r in recs if r["fold"] == "shot" and r["exp"] == exp and r["method"] == meth]
        v = np.sort([r["med"] for r in rs if not r.get("failed")])
        sv.line(v, np.arange(1, len(v) + 1) / len(rs), col, lab, dash)
    lp = out / "lines.json"
    if lp.exists():
        lr = [r for r in json.loads(lp.read_text()) if r["fold"] == "shot" and r["method"] == "lines_mid"]
        v = np.sort([r["med"] for r in lr if not r.get("failed")])
        sv.line(v, np.arange(1, len(v) + 1) / len(lr), "#111", "PTZ lines only (abstain: curve stops <1)")
    sv.save(out / "error_cdf.svg")
    zp = out / "zoom.json"
    if zp.exists() and (k := json.loads(zp.read_text()).get("k1_lines")):
        sv = Svg("k1 fitted on the line mask (C fixed) vs zoom", "focal length (px at 1920 wide)", "k1", (1500, 6000),
                 (-0.02, 0.06))
        for m, col in cols.items():
            kk = [x for x in k if x["match"] == m]
            sv.points([x["f"] for x in kk], [x["k1"] for x in kk], col, m)
        sv.save(out / "k1_vs_f.svg")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rigs", "points", "lines", "zoom", "synthetic", "report", "plots"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--max-c", type=int, default=8)
    ap.add_argument("--boot", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--train-filter", choices=["all", "plausible"], default="all")
    ap.add_argument("--variants", default="mid,fixed,loose")
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"rigs": cmd_rigs, "points": cmd_points, "lines": cmd_lines, "zoom": cmd_zoom, "synthetic": cmd_synthetic,
     "report": cmd_report, "plots": cmd_plots}[args.cmd](args)


if __name__ == "__main__":
    main()
