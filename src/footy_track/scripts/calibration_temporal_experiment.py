"""Experiment: carry a keyframe calibration across a clip with frame-to-frame camera motion.

Question: can a calibration (homography, lens k1) be carried from a labelled keyframe to the other
frames of a clip by tracking camera motion, and how accurate is that vs per-frame fitting?

Stages (each writes under ``--out``; the human GT is only read)::

    motion    --clip C --config lk_grass      estimate + cache pairwise motion (ClipMotion .npz)
    gt        --clip C                        cache GT frames, line masks, evaluation grids
    prop      --clip C --configs ...          pure propagation, all ordered GT pairs  -> prop_<C>.csv
    refine    --clip C --config lk_grass      + per-frame refine / static baselines   -> refine_<C>.csv
    keyframes --clip C --config lk_grass      1 / 2 / 4 keyframes, held-out GT frames -> keyframes_<C>.csv
    track     --clip C --config lk_grass      re-anchor policies + smoothing          -> track_<C>.json
    report                                    tables (results.md) + figures

Configs (``--config``): ``lk_grass`` (sparse LK + RANSAC, grass/line mask), ``lk_boxes`` (LK, whole frame minus
machine-label person boxes), ``lk_all`` (LK, whole frame), ``ecc_grass``, ``lkecc_grass``.

Run with ``OMP_NUM_THREADS=1`` (at most 2 processes on the Mac mini).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration import temporal as tp
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.pipeline import find_video, overlay_render
from footy_track.calibration.refine import dt_residual, model_samples
from footy_track.calibration.video import read_frame
from footy_track.clip_id import canonical_clip_id

PITCH = PitchModel()
GT_DIR = Path(os.environ.get("FOOTY_GT_MARKS_DIR", "/mnt/storage/footy_data/ball_gt_marks"))
CLIPS_DIR = Path(os.environ.get("FOOTY_CLIPS_DIR", "/mnt/storage/footy_data/ballcheck_clips"))
BOXES_DIR = Path("/mnt/storage/footy_data/machine_labels")
CONFIGS = {
    "lk_grass": {"method": "lk", "mask_mode": "grass"},
    "lk_boxes": {"method": "lk", "mask_mode": "all", "boxes": True},
    "lk_all": {"method": "lk", "mask_mode": "all"},
    "ecc_grass": {"method": "ecc", "mask_mode": "grass"},
    "lkecc_grass": {"method": "lk_ecc", "mask_mode": "grass"},
}
WORK_WIDTH = 1280  # refinement working width
SAMPLES = model_samples(PITCH)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_gt(clip: str) -> dict[int, dict]:
    doc = json.loads((GT_DIR / "calibration_gt" / f"{canonical_clip_id(clip)}.json").read_text())
    out = {}
    for k, r in doc["frames"].items():
        out[int(k)] = {
            "H": np.array(r["H"], float),
            "img": np.array([c["image"] for c in r["correspondences"]], float),
            "pitch": np.array([c["pitch_xy"] for c in r["correspondences"]], float),
            "size": tuple(r["image_size"]),
            "origin": (r.get("origin") or {}).get("source", "human"),
        }
    return out


def boxes_for(clip: str):
    cand = [BOXES_DIR / f"{clip}.jsonl"]
    cand += [p for p in BOXES_DIR.glob("*.jsonl") if canonical_clip_id(p.stem) == canonical_clip_id(clip)]
    for p in cand:
        if p.exists():
            return tp.load_person_boxes(p)
    return None


def motion_path(out: Path, clip: str, cfg: str) -> Path:
    return out / f"motion_{canonical_clip_id(clip)}__{cfg}.npz"


def load_motion(out: Path, clip: str, cfg: str) -> tp.ClipMotion:
    return tp.ClipMotion.load(motion_path(out, clip, cfg))


def _err_stats(e: np.ndarray) -> dict:
    return {"mean": float(np.mean(e)), "med": float(np.median(e)), "p95": float(np.percentile(e, 95)), "max": float(np.max(e))}


def px_err(H: np.ndarray, g: dict, dist: dict | None = None) -> tuple[float, float]:
    W, Hh = g["size"]
    e = geo.reprojection_errors(H, g["img"], g["pitch"], (W, Hh), dist)["px"]
    e = e[np.isfinite(e)]
    return (float(np.mean(e)), float(np.max(e))) if len(e) else (float("nan"), float("nan"))


def grid_err(H: np.ndarray, g: dict, pts: np.ndarray, dist: dict | None = None) -> dict:
    a, _ = geo.image_to_pitch(H, pts, dist)
    ref, _ = geo.apply_homography(g["H"], pts)
    return _err_stats(np.linalg.norm(a - ref, axis=1))


# --------------------------------------------------------------------------
# stage: motion
# --------------------------------------------------------------------------


def cmd_motion(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    video = find_video(CLIPS_DIR, a.clip)
    cfg = dict(CONFIGS[a.config])
    boxes = boxes_for(a.clip) if cfg.pop("boxes", False) else None
    if a.config == "lk_boxes" and boxes is None:
        print(f"no machine labels for {a.clip}; skipped")
        return
    t0 = time.perf_counter()
    cm = tp.clip_motion(video, frames=(a.start, a.stop) if a.stop else None, max_gap=a.max_gap, scale=a.scale, boxes=boxes,
                        progress=lambda i: print(f"  frame {i}", flush=True), **cfg)
    cm.save(motion_path(out, a.clip, a.config))
    ms = cm.ms.mean(0)
    print(f"{a.clip} {a.config}: {cm.n} frames, cuts {cm.cuts}, ok(d=1) {cm.ok[:, 0].mean():.3f}, "
          f"ms/frame decode {ms[0]:.1f} mask {ms[1]:.1f} motion(4 pairs) {ms[2]:.1f}; wall {time.perf_counter() - t0:.0f}s")


# --------------------------------------------------------------------------
# stage: gt cache
# --------------------------------------------------------------------------


def gt_cache_path(out: Path, clip: str) -> Path:
    return out / f"gt_{canonical_clip_id(clip)}.npz"


def cmd_gt(a):
    out = Path(a.out)
    video = find_video(CLIPS_DIR, a.clip)
    gt = load_gt(a.clip)
    store: dict[str, np.ndarray] = {}
    for f in sorted(gt):
        frame = read_frame(video, f)
        m = tp.line_masks(frame, WORK_WIDTH)
        h, w = m.grass.shape
        # evaluation lattice: grass pixels that GT maps inside (a margin around) the pitch
        gx, gy = np.meshgrid(np.linspace(0.04, 0.96, 24), np.linspace(0.04, 0.96, 14))
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        er = cv2.erode(m.grass.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        on = er[np.clip((pts[:, 1] * h).astype(int), 0, h - 1), np.clip((pts[:, 0] * w).astype(int), 0, w - 1)]
        xy, wv = geo.apply_homography(gt[f]["H"], pts)
        ins = (wv > 1e-9) & (np.abs(xy[:, 0]) <= PITCH.hl + 3) & (np.abs(xy[:, 1]) <= PITCH.hw + 3)
        store[f"grid_{f}"] = pts[on & ins]
        store[f"grass_{f}"] = np.packbits(m.grass)
        store[f"lines_{f}"] = np.packbits(m.lines)
        store[f"far_{f}"] = np.packbits(m.far)
        store[f"shape_{f}"] = np.array([h, w])
        cv2.imwrite(str(out / f"frame_{canonical_clip_id(a.clip)}_{f}.jpg"), cv2.resize(frame, (1280, 720)))
        print(f"  gt frame {f}: {len(store[f'grid_{f}'])} grid points", flush=True)
    np.savez_compressed(gt_cache_path(out, a.clip), **store)


class GT:
    """GT frames + cached masks of a clip."""

    def __init__(self, out: Path, clip: str):
        self.clip, self.out = clip, out
        self.gt = load_gt(clip)
        self.z = np.load(gt_cache_path(out, clip))

    def masks(self, f: int) -> tp.LineMasks:
        h, w = self.z[f"shape_{f}"]
        un = lambda k: np.unpackbits(self.z[f"{k}_{f}"])[: h * w].reshape(h, w).astype(bool)  # noqa: E731
        return tp.LineMasks(un("grass"), un("lines"), un("far"))

    def grid(self, f: int) -> np.ndarray:
        return self.z[f"grid_{f}"]

    def frame(self, f: int) -> np.ndarray:
        return cv2.imread(str(self.out / f"frame_{canonical_clip_id(self.clip)}_{f}.jpg"))


def line_cost(H: np.ndarray, m: tp.LineMasks, trunc: float = 20.0) -> float:
    """Mean truncated distance (px at the 1280 working width) of the model lines to the line mask."""
    h, w = m.lines.shape
    G = np.linalg.inv(np.asarray(H) @ np.diag([1.0 / w, 1.0 / h, 1.0]))
    G = G / np.linalg.norm(G)
    c, n = dt_residual(G, SAMPLES, m.lines, m.grass, trunc)
    return c if n >= 30 else float("nan")


# --------------------------------------------------------------------------
# stage: pure propagation
# --------------------------------------------------------------------------


def rz_trajectory(cm: tp.ClipMotion, H_key: np.ndarray, key: int, aspect: float) -> dict[int, np.ndarray]:
    """Calibration at every reachable frame from ``key`` using rotation + zoom projected d=1 motions only."""
    f0 = tp.focal_from_homography(H_key, aspect)
    out: dict[int, np.ndarray] = {}
    if f0 is None:
        return out
    for direction in (1, -1):
        H, f, t = H_key.copy(), f0, key
        while True:
            u = t + direction
            if not (cm.start <= u < cm.start + cm.n) or cm.shot[u - cm.start] != cm.shot[t - cm.start]:
                break
            pair = cm.pair(t, u)  # t -> u
            if pair is None or not pair.ok:
                break
            Mc, f, _ = tp.project_rot_zoom(pair.H_norm, f, aspect)
            H = geo.orient_homography(H @ np.linalg.inv(Mc), np.array([[0.5, 0.5]]))
            out[u] = H
            t = u
    return out


def cmd_prop(a):
    out = Path(a.out)
    g = GT(out, a.clip)
    frames = sorted(g.gt)
    rows = []
    for cfg in a.configs:
        cm = load_motion(out, a.clip, cfg)
        variants = {"chain": cm}
        if not a.no_d1:
            d1 = tp.ClipMotion(**{**cm.__dict__, "ok": cm.ok.copy()})
            d1.ok[:, 1:] = False
            variants["chain_d1"] = d1
        for vname, cmv in variants.items():
            for k in [f for f in frames if cm.start <= f < cm.start + cm.n]:
                rz = rz_trajectory(cm, g.gt[k]["H"], k, g.gt[k]["size"][0] / g.gt[k]["size"][1]) if (vname == "chain_d1") else {}
                for t in frames:
                    if t == k or not (cm.start <= t < cm.start + cm.n):
                        continue
                    shot_ok = cmv.shot[t - cmv.start] == cmv.shot[k - cmv.start]
                    for mname in (["lk"] if vname == "chain" else ["lk_d1", "rotzoom_d1"]):
                        H = None
                        if mname == "rotzoom_d1":
                            H = rz.get(t)
                        elif shot_ok:
                            H = tp.propagate(g.gt[k]["H"], cmv, k, t)
                        row = {"clip": a.clip, "cfg": cfg, "model": f"{cfg}/{'chain' if mname == 'lk' else mname}", "key": k, "t": t,
                               "gap": abs(t - k), "same_shot": bool(shot_ok), "ok": H is not None}
                        if H is not None:
                            row.update({f"m_{s}": v for s, v in grid_err(H, g.gt[t], g.grid(t)).items()})
                            row["px_mean"], row["px_max"] = px_err(H, g.gt[t])
                            row["dt"] = line_cost(H, g.masks(t))
                        rows.append(row)
        print(f"{a.clip} {cfg}: done", flush=True)
    # GT floor: refit cost of GT itself
    for t in frames if not a.no_d1 else []:
        rows.append({"clip": a.clip, "cfg": "gt", "model": "gt", "key": t, "t": t, "gap": 0, "same_shot": True, "ok": True,
                     "m_mean": 0.0, "m_med": 0.0, "m_p95": 0.0, "m_max": 0.0, "px_mean": px_err(g.gt[t]["H"], g.gt[t])[0],
                     "px_max": px_err(g.gt[t]["H"], g.gt[t])[1], "dt": line_cost(g.gt[t]["H"], g.masks(t))})
    write_csv(out / f"prop_{canonical_clip_id(a.clip)}{'__' + a.tag if a.tag else ''}.csv", rows)


def write_csv(path: Path, rows: list[dict]):
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if v in ("True", "False"):
                    r[k] = v == "True"
                else:
                    with contextlib.suppress(ValueError, TypeError):
                        r[k] = float(v) if ("." in v or "e" in v or v == "nan") else int(v)
            rows.append(r)
    return rows


# --------------------------------------------------------------------------
# stage: refine
# --------------------------------------------------------------------------


def pick_keys(frames: list[int], n: int) -> list[int]:
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, n)).astype(int))
    return [frames[i] for i in idx]


def cmd_refine(a):
    out = Path(a.out)
    g = GT(out, a.clip)
    cm = load_motion(out, a.clip, a.config)
    frames = sorted(g.gt)
    keys = pick_keys(frames, a.n_keys)
    rows = []
    for ki, k in enumerate(keys):
        Hk = g.gt[k]["H"]
        for t in frames[:: a.target_stride]:
            if t == k or cm.shot[t - cm.start] != cm.shot[k - cm.start]:
                continue
            m = g.masks(t)
            variants = {
                "prop": tp.propagate(Hk, cm, k, t),
                "static": Hk,
            }
            for vname, H0 in list(variants.items()):
                if H0 is None:
                    continue
                for tag, kw in (("", {}), ("+k1", {"fit_k1": True})):
                    if tag and (vname == "static" or ki > 0):  # k1 variant only from the first keyframe (cost)
                        continue
                    t0 = time.perf_counter()
                    r = tp.refine_frame(H0, m, PITCH, **kw)
                    dt = time.perf_counter() - t0
                    dist = geo.make_distortion(r.k1, m.lines.shape[1] / m.lines.shape[0]) if r.k1 else None
                    row = {"clip": a.clip, "cfg": a.config, "variant": f"{vname}+refine{tag}", "key": k, "t": t,
                           "gap": abs(t - k), "applied": r.applied, "reason": r.reason, "secs": dt, "k1": r.k1,
                           "dt_before": r.cost_before, "dt_after": r.cost_after}
                    row.update({f"m_{s}": v for s, v in grid_err(r.H, g.gt[t], g.grid(t), dist).items()})
                    row["px_mean"], row["px_max"] = px_err(r.H, g.gt[t], dist)
                    rows.append(row)
                    print(f"  {k}->{t} {row['variant']}: applied={r.applied} mean={row['m_mean']:.2f}m {dt:.1f}s", flush=True)
        write_csv(out / f"refine_{canonical_clip_id(a.clip)}.csv", rows)
    # refinement started from the GT itself: how much does it move a "correct" H, and its line cost
    for t in frames[:: a.target_stride]:
        m = g.masks(t)
        r = tp.refine_frame(g.gt[t]["H"], m, PITCH)
        row = {"clip": a.clip, "cfg": a.config, "variant": "gt+refine", "key": t, "t": t, "gap": 0, "applied": r.applied,
               "reason": r.reason, "secs": 0.0, "k1": r.k1, "dt_before": r.cost_before, "dt_after": r.cost_after}
        row.update({f"m_{s}": v for s, v in grid_err(r.H, g.gt[t], g.grid(t)).items()})
        row["px_mean"], row["px_max"] = px_err(r.H, g.gt[t])
        rows.append(row)
    write_csv(out / f"refine_{canonical_clip_id(a.clip)}.csv", rows)


# --------------------------------------------------------------------------
# stage: keyframe sparsity
# --------------------------------------------------------------------------


def blend(H0: np.ndarray, H1: np.ndarray, w1: float) -> np.ndarray:
    A, B = H0 / np.linalg.norm(H0), H1 / np.linalg.norm(H1)
    if np.sum(A * B) < 0:
        B = -B
    return (1 - w1) * A + w1 * B


def cmd_keyframes(a):
    out = Path(a.out)
    g = GT(out, a.clip)
    cm = load_motion(out, a.clip, a.config)
    frames = sorted(g.gt)
    # the longest shot's GT frames
    shot_of = {f: int(cm.shot[f - cm.start]) for f in frames if cm.start <= f < cm.start + cm.n}
    best = max(set(shot_of.values()), key=lambda s: sum(1 for v in shot_of.values() if v == s))
    pool = [f for f in frames if shot_of.get(f) == best]
    rows = []
    for n_keys in (1, 2, 4):
        if len(pool) < n_keys + 2:
            continue
        # every subset of n_keys evenly spread choices (shifted) to average over luck
        combos = []
        for shift in range(0, max(1, len(pool) // max(n_keys, 1))):
            idx = sorted({min(len(pool) - 1, shift + round(i * (len(pool) - 1 - shift) / max(n_keys - 1, 1))) if n_keys > 1 else min(len(pool) - 1, shift * 3)
                          for i in range(n_keys)})
            if len(idx) == n_keys and idx not in combos:
                combos.append(idx)
        for idx in combos:
            keys = [pool[i] for i in idx]
            for t in pool:
                if t in keys:
                    continue
                gaps = sorted(keys, key=lambda k: abs(k - t))
                H_near = tp.propagate(g.gt[gaps[0]]["H"], cm, gaps[0], t)
                if H_near is None:
                    continue
                e_near = grid_err(H_near, g.gt[t], g.grid(t))
                row = {"clip": a.clip, "n_keys": n_keys, "keys": " ".join(map(str, keys)), "t": t,
                       "gap": abs(gaps[0] - t), "mode": "nearest", **{f"m_{s}": v for s, v in e_near.items()}}
                rows.append(row)
                lo = [k for k in keys if k < t]
                hi = [k for k in keys if k > t]
                if lo and hi:
                    k0, k1 = max(lo), min(hi)
                    H0, H1 = tp.propagate(g.gt[k0]["H"], cm, k0, t), tp.propagate(g.gt[k1]["H"], cm, k1, t)
                    if H0 is not None and H1 is not None:
                        Hb = blend(H0, H1, (t - k0) / (k1 - k0))
                        rows.append({**row, "mode": "blend", **{f"m_{s}": v for s, v in grid_err(Hb, g.gt[t], g.grid(t)).items()}})
    write_csv(out / f"keyframes_{canonical_clip_id(a.clip)}.csv", rows)
    print(f"{a.clip}: {len(rows)} rows (shot {best}, pool {pool})")


# --------------------------------------------------------------------------
# stage: tracking with re-anchoring + smoothing
# --------------------------------------------------------------------------


def sg_smooth(y: np.ndarray, t: np.ndarray, half: int, deg: int = 2) -> np.ndarray:
    """Savitzky-Golay-style local polynomial smoothing of (n, d) samples at times ``t`` (window shrinks at the ends)."""
    n = len(t)
    out = np.empty_like(y, dtype=float)
    for i in range(n):
        h = min(half, i, n - 1 - i)
        lo, hi = i - h, i + h + 1
        d = min(deg, hi - lo - 1)
        x = t[lo:hi] - t[i]
        for j in range(y.shape[1]):
            out[i, j] = np.polyval(np.polyfit(x, y[lo:hi, j], d), 0.0) if d >= 0 and hi - lo > d else y[i, j]
    return out


PROBE = np.array([[0.1, 0.3], [0.9, 0.3], [0.9, 0.95], [0.1, 0.95], [0.5, 0.5]])


def h_to_params(H):
    xy, _ = geo.apply_homography(H, PROBE)
    return xy.ravel()


def params_to_h(p):
    H = cv2.getPerspectiveTransform(PROBE[:4].astype(np.float32), p.reshape(-1, 2)[:4].astype(np.float32)).astype(float)
    return geo.orient_homography(H, np.array([[0.5, 0.5]]))


def cmd_track(a):  # noqa: PLR0912, PLR0915
    out = Path(a.out)
    g = GT(out, a.clip)
    cm = load_motion(out, a.clip, a.config)
    video = find_video(CLIPS_DIR, a.clip)
    frames = sorted(g.gt)
    key0 = a.key if a.key is not None else frames[0]
    shot = cm.shot[key0 - cm.start]
    last = max(f for f in frames if cm.shot[f - cm.start] == shot)
    step = a.step
    sample = sorted(set(range(key0, last + 1, step)) | {f for f in frames if key0 <= f <= last})
    policies = {"chain_only": None, "refine_every_5": 5, "refine_every_25": 25, "refine_every_50": 50, "trigger": "trigger"}
    state = {p: {"H": g.gt[key0]["H"].copy(), "frame": key0, "dt0": None, "n_ref": 0, "lost": False} for p in policies}
    hist = {p: {} for p in policies}  # t -> H (after the policy's action at t)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, key0)
    t_cur = key0
    t_ref = 0.0
    n_ref_calls = 0
    ss = set(sample)
    while t_cur <= last:
        ok = cap.grab()
        if not ok:
            break
        if t_cur in ss:
            _, fr = cap.retrieve()
            m = tp.line_masks(fr, WORK_WIDTH) if t_cur != key0 else g.masks(t_cur)
            for p, pol in policies.items():
                st = state[p]
                if st["lost"]:
                    continue
                if t_cur == key0:
                    hist[p][t_cur] = st["H"]
                    st["dt0"] = line_cost(st["H"], m)
                    continue
                Hp = tp.propagate(st["H"], cm, st["frame"], t_cur)
                if Hp is None:
                    st["lost"] = True
                    continue
                do = False
                if isinstance(pol, int):
                    do = (t_cur - st["frame"]) >= pol
                elif pol == "trigger":
                    c = line_cost(Hp, m)
                    do = (not np.isfinite(c)) or c > 1.5 * st["dt0"] + 1.0 or (t_cur - st["frame"]) >= 100
                if do:
                    t0 = time.perf_counter()
                    r = tp.refine_frame(Hp, m, PITCH)
                    t_ref += time.perf_counter() - t0
                    n_ref_calls += 1
                    st["n_ref"] += 1
                    if r.applied:
                        Hp = r.H
                    st["H"], st["frame"] = Hp, t_cur
                    st["dt0"] = line_cost(Hp, m)
                hist[p][t_cur] = Hp
        t_cur += 1
    cap.release()
    res = {"clip": a.clip, "key": key0, "last": last, "step": step, "refine_secs_mean": t_ref / max(n_ref_calls, 1), "policies": {}}
    for p in policies:
        errs = {}
        for f in frames:
            if f in hist[p] and f != key0:
                errs[f] = grid_err(hist[p][f], g.gt[f], g.grid(f))
        res["policies"][p] = {"n_refine": state[p]["n_ref"], "lost": state[p]["lost"],
                              "errors": {str(f): e for f, e in errs.items()}}
    # smoothing of the refine_every_5 sequence (params of 5 image probe points, local quadratic over +-half samples)
    seq = hist["refine_every_5"]
    ts = np.array(sorted(seq))
    if len(ts) > 8:
        P = np.stack([h_to_params(seq[t]) for t in ts])
        for half in (1, 2, 3):
            Ps = sg_smooth(P, ts.astype(float), half)
            Hs = {int(t): params_to_h(Ps[i]) for i, t in enumerate(ts)}
            errs = {str(f): grid_err(Hs[f], g.gt[f], g.grid(f)) for f in frames if f in Hs and f != key0}
            res["policies"][f"refine_every_5+sg{2 * half + 1}"] = {"n_refine": state["refine_every_5"]["n_ref"], "lost": False,
                                                                 "errors": errs}
        # jitter: mean frame-to-frame (5 frames) pitch-position change of the probe points
        # jitter: mean change of the probe points' pitch position between consecutive samples, raw vs smoothed
        def _jit(P_):
            return float(np.mean(np.linalg.norm(np.diff(P_, axis=0).reshape(len(P_) - 1, -1, 2), axis=2)))

        res["jitter_m_per_sample"] = {"raw": _jit(P), **{f"sg{2 * h + 1}": _jit(sg_smooth(P, ts.astype(float), h)) for h in (1, 2, 3)}}
    (out / f"track_{canonical_clip_id(a.clip)}.json").write_text(json.dumps(res, indent=1))
    for p, v in res["policies"].items():
        e = [x["mean"] for x in v["errors"].values()]
        print(f"{p:28s} refines={v['n_refine']:3d} lost={v['lost']} mean-of-mean err {np.mean(e) if e else float('nan'):.2f} m over {len(e)} GT frames")


# --------------------------------------------------------------------------
# stage: report
# --------------------------------------------------------------------------


GAP_BINS = [(1, 25), (26, 50), (51, 100), (101, 200), (201, 400), (401, 700)]


def _bin(rows, lo, hi):
    return [r for r in rows if lo <= r["gap"] <= hi]


def _fmt(x):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"


def _agg(rows, key="m_mean", stat=np.mean):
    v = [r[key] for r in rows if r.get("ok", True) and isinstance(r.get(key), float) and np.isfinite(r[key])]
    return stat(v) if v else None


def cmd_report(a):  # noqa: PLR0912, PLR0915
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out = Path(a.out)
    md = ["# Temporal calibration propagation - results\n",
          "Setup: GT = accepted calibration GT (plain homography, no k1 recorded; 4-point GT fit their clicks exactly). Error = mean over "
          "a lattice of grass pixels of the pitch-plane distance (m) between the estimated and the GT homography at the target frame. "
          "Motion: sparse LK + RANSAC at 480 px, chained over pairs of up to 4 frames (fewest hops). Cuts detected: see section 6; "
          "no GT pair straddles a cut. Timings in the section-6 table are wall-clock on a Mac mini at load 20-55 and are inflated; "
          "the CPU-time table at the end is the reference. Run with OpenCV 4.11 (the main venv was being re-synced). Sections are "
          "numbered after the experiment brief (3 = floors, inside 2).\n"]
    prop = [r for p in sorted(out.glob("prop_*.csv")) for r in read_csv(p)]
    models = sorted({r["model"] for r in prop if r["model"] != "gt"})
    clips = sorted({r["clip"] for r in prop})

    md.append("## 1. Pure propagation (no refinement): error vs frame gap\n")
    md.append("Mean over grass grid points of |H_prop(p) - H_GT(p)| in metres, averaged over all ordered GT (key, target) "
              "pairs in the same shot; cell = mean / p95-of-pair-means (n pairs). Pairs whose chain is broken (cut, failed pairs) "
              "are counted separately in section 6.\n")
    for clip in clips:
        md.append(f"### {clip}\n")
        md.append("| model | " + " | ".join(f"gap {lo}-{hi}" for lo, hi in GAP_BINS) + " |")
        md.append("|---|" + "---|" * len(GAP_BINS))
        for mo in models:
            rr = [r for r in prop if r["model"] == mo and r["clip"] == clip and r["ok"]]
            if not rr:
                continue
            cells = []
            for lo, hi in GAP_BINS:
                b = _bin(rr, lo, hi)
                cells.append("-" if not b else f"{_agg(b):.2f} / {_agg(b, stat=lambda v: np.percentile(v, 95)):.2f} (n={len(b)})")
            md.append(f"| {mo} | " + " | ".join(cells) + " |")
        md.append("")

    # figure: error vs gap (propagation models, other clips pooled, refinement variants)
    ref = [r for p in sorted(out.glob("refine_*.csv")) for r in read_csv(p)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    def curve(ax, rr, label, **kw):
        xs, ys, lo, hi = [], [], [], []
        for glo, ghi in GAP_BINS:
            bb = [r for r in rr if glo <= r["gap"] <= ghi and isinstance(r.get("m_mean"), float)]
            if len(bb) >= 3:
                v = np.array([r["m_mean"] for r in bb])
                xs.append(np.mean([r["gap"] for r in bb]))
                ys.append(np.median(v))
                lo.append(np.percentile(v, 10))
                hi.append(np.percentile(v, 90))
        if xs:
            ax.plot(xs, ys, "o-", label=label, **kw)
            if kw.get("band"):
                pass
        return xs, lo, hi

    s10 = [r for r in prop if "seg010" in r["clip"] and r["ok"]]
    for mo in sorted({r["model"] for r in s10}):
        curve(axes[0], [r for r in s10 if r["model"] == mo], mo)
    others = [r for r in prop if "seg010" not in r["clip"] and r["ok"]]
    for mo in sorted({r["model"] for r in others}):
        curve(axes[1], [r for r in others if r["model"] == mo], mo)
    s10r = [r for r in ref if "seg010" in r["clip"]]
    base = [r for r in s10 if r["model"] == "lk_grass/chain"]
    curve(axes[2], base, "propagation only")
    for v in ("prop+refine", "static+refine"):
        curve(axes[2], [r for r in s10r if r["variant"] == v], v)
    for ax, title in zip(axes, ("arsenal_mancity_seg010: propagation (median, all GT pairs)",
                                "other clips pooled (4 clips, few pairs)", "seg010: with per-frame refinement"), strict=True):
        ax.axhline(1.0, color="grey", ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("frame gap keyframe -> target")
        ax.set_ylabel("mean grid error (m), median over pairs")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "error_vs_gap.png", dpi=130)
    plt.close(fig)

    # refine table
    if ref:
        md.append("## 2. Propagation + per-frame refinement\n")
        md.append("Same metric (mean grid error, metres; cell = mean / p95 of pair means). 'prop' = pure propagation (from section 1), "
                  "'static' = nearest keyframe H unchanged, '+refine' = `refine_homography` (k1 fixed 0) from that start, "
                  "'+refine+k1' fits k1 jointly (scored against the k1=0 GT, so this penalises a k1 the GT lacks). "
                  "Keyframes: 3 evenly spread GT frames of each clip, every 2nd GT frame as target (compute budget).\n")
        for clip in sorted({r["clip"] for r in ref}):
            md.append(f"### {clip}\n")
            md.append("| variant | " + " | ".join(f"gap {lo}-{hi}" for lo, hi in GAP_BINS) + " | applied | mean s/refine |")
            md.append("|---|" + "---|" * (len(GAP_BINS) + 2))
            keys = {r["key"] for r in ref if r["clip"] == clip and r["variant"] != "gt+refine"}
            cfg = next(r["cfg"] for r in ref if r["clip"] == clip)
            base = [r for r in prop if r["clip"] == clip and r["model"] == f"{cfg}/chain" and r["ok"] and r["key"] in keys]
            rows_v = {"prop (no refine)": base}
            for v in ("static+refine", "prop+refine", "prop+refine+k1"):
                rows_v[v] = [r for r in ref if r["clip"] == clip and r["variant"] == v]
            rows_v["static (no refine)"] = [{**r, "gap": r["gap"]} for r in _static_rows(out, clip, keys)]
            for v, rr in rows_v.items():
                if not rr:
                    continue
                cells = []
                for lo, hi in GAP_BINS:
                    b = _bin(rr, lo, hi)
                    cells.append("-" if not b else f"{_agg(b):.2f} / {_agg(b, stat=lambda x: np.percentile(x, 95)):.2f} (n={len(b)})")
                app = np.mean([r["applied"] for r in rr]) if "applied" in rr[0] else float("nan")
                secs = np.mean([r["secs"] for r in rr]) if "secs" in rr[0] else float("nan")
                md.append(f"| {v} | " + " | ".join(cells) + f" | {_fmt(app)} | {_fmt(secs)} |")
            md.append("")
        md.append("### Overall, all gaps (seg010 unless stated)\n")
        md.append("| variant | n | median (m) | mean (m) | frac < 1 m | frac < 2 m | frac > 5 m | refine applied |")
        md.append("|---|---|---|---|---|---|---|---|")
        for clip in sorted({r["clip"] for r in ref}):
            keys = {r["key"] for r in ref if r["clip"] == clip and r["variant"] != "gt+refine"}
            cfg = next(r["cfg"] for r in ref if r["clip"] == clip)
            tgts = {r["t"] for r in ref if r["clip"] == clip and r["variant"] == "prop+refine"}
            sets = {
                "propagation only": [r for r in prop if r["clip"] == clip and r["model"] == f"{cfg}/chain" and r["ok"]
                                     and r["key"] in keys and r["t"] in tgts],
                "static H, no refine": [r for r in _static_rows(out, clip, keys) if r["t"] in tgts],
            }
            for v in ("static+refine", "prop+refine", "prop+refine+k1"):
                sets[v] = [r for r in ref if r["clip"] == clip and r["variant"] == v]
            for v, rr in sets.items():
                e = np.array([r["m_mean"] for r in rr if isinstance(r.get("m_mean"), float)])
                if len(e):
                    app = np.mean([r["applied"] for r in rr]) if "applied" in rr[0] else float("nan")
                    md.append(f"| {clip}: {v} | {len(e)} | {np.median(e):.2f} | {e.mean():.2f} | {np.mean(e < 1):.2f} | "
                              f"{np.mean(e < 2):.2f} | {np.mean(e > 5):.2f} | {_fmt(app)} |")
        md.append("")
        md.append("### Floors\n")
        md.append("The human/Claude GT is itself approximate: a 4-point GT fits its own clicks exactly (corr. px error ~ 0) but its model lines sit "
                  "off the painted lines by ~3-5 px; refinement started from GT moves the grid by the 'GT -> refined' amount.\n")
        md.append("| clip | GT corr. px err | GT line-DT cost (px @1280) | refine-from-GT moves grid by (m, mean) |")
        md.append("|---|---|---|---|")
        for clip in sorted({r["clip"] for r in ref}):
            gtr = [r for r in ref if r["clip"] == clip and r["variant"] == "gt+refine"]
            pg = [r for r in prop if r["clip"] == clip and r["model"] == "gt"]
            md.append(f"| {clip} | {_fmt(_agg(pg, 'px_mean'))} | {_fmt(_agg(pg, 'dt'))} | {_fmt(_agg(gtr))} |")
        pair = [r for r in prop if "seg010" in r["clip"] and r["key"] == 349 and r["t"] == 350 and r["model"].endswith("/chain")]
        pair += [r for r in prop if "seg010" in r["clip"] and r["key"] == 350 and r["t"] == 349 and r["model"].endswith("/chain")]
        if pair:
            md.append(f"\nGT-to-GT agreement: GT frames 349 and 350 are consecutive frames labelled separately; propagating one onto the "
                      f"other (1 frame of motion, ~0) differs from the other's GT by {np.mean([r['m_mean'] for r in pair]):.2f} m mean "
                      f"(max {np.max([r['m_max'] for r in pair]):.1f} m): the labelling noise floor of this metric.\n")
        md.append("")

    # keyframe sparsity
    kf = [r for p in sorted(out.glob("keyframes_*.csv")) for r in read_csv(p)]
    if kf:
        md.append("## 4. Keyframe sparsity (held-out GT frames, pure propagation)\n")
        md.append("`nearest`: propagate from the temporally nearest keyframe; `blend`: matrix blend of forward- and backward-propagated "
                  "H from the two bracketing keyframes (only for frames between two keyframes). Fraction of held-out frames with "
                  "mean grid error < 1 m / < 2 m, and p95 of the per-frame mean error.\n")
        md.append("| clip | keyframes | mode | held-out frames | frac < 1 m | frac < 2 m | p95 mean err (m) | median (m) | max gap |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for clip in sorted({r["clip"] for r in kf}):
            for nk in (1, 2, 4):
                for mode in ("nearest", "blend"):
                    rr = [r for r in kf if r["clip"] == clip and r["n_keys"] == nk and r["mode"] == mode]
                    if not rr:
                        continue
                    v = np.array([r["m_mean"] for r in rr])
                    md.append(f"| {clip} | {nk} | {mode} | {len(rr)} | {np.mean(v < 1):.2f} | {np.mean(v < 2):.2f} | "
                              f"{np.percentile(v, 95):.2f} | {np.median(v):.2f} | {max(r['gap'] for r in rr)} |")
        md.append("")

    # tracking
    tr = [json.loads(p.read_text()) for p in sorted(out.glob("track_*.json"))]
    if tr:
        md.append("## 5. Re-anchoring policies on a whole shot (chain + refine on the way)\n")
        md.append("Start from the GT of the first frame; every 5th frame the policy may refine the propagated H on the image "
                  "(and the result becomes the new anchor). Error against GT at all other GT frames of the shot (grid mean, m). "
                  "`trigger` refines when the model-line distance cost exceeds 1.5x the last anchor's + 1 px, or after 100 frames.\n")
        md.append("| clip | policy | refine calls | GT frames scored | mean err (m) | median | max | frac < 1 m |")
        md.append("|---|---|---|---|---|---|---|---|")
        for t in tr:
            for p, v in t["policies"].items():
                e = np.array([x["mean"] for x in v["errors"].values()])
                if len(e):
                    md.append(f"| {t['clip']} | {p} | {v['n_refine']} | {len(e)} | {e.mean():.2f} | {np.median(e):.2f} | {e.max():.2f} | "
                              f"{np.mean(e < 1):.2f} |")
        md.append("")
        md.append(f"Mean refine time in that run: {', '.join(f'{t['clip']} {t['refine_secs_mean']:.2f}s' for t in tr)}.\n")

    # cuts and speed
    md.append("## 6. Shot cuts, motion success and speed\n")
    md.append("| clip | config | frames | cuts (frame index) | pairs ok d=1 | ok d=4 | decode ms | mask ms | motion ms (4 pairs) | ms/pair |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(out.glob("motion_*.npz")):
        cm = tp.ClipMotion.load(p)
        clip, cfg = p.stem[len("motion_"):].split("__")
        ms = cm.ms.mean(0)
        md.append(f"| {clip} | {cfg} | {cm.n} | {cm.cuts} | {cm.ok[:, 0].mean():.3f} | {cm.ok[:, -1].mean():.3f} | {ms[0]:.1f} | {ms[1]:.1f} | "
                  f"{ms[2]:.1f} | {ms[2] / max(cm.max_gap, 1):.1f} |")
    md.append("")
    for bp in sorted(out.glob("bench_*.json")):
        md.append(f"### CPU time per operation ({bp.stem[len('bench_'):]}, single thread, `time.process_time`, 480 px working scale)\n")
        md.append("| operation | median ms | mean ms | n |")
        md.append("|---|---|---|---|")
        for k, v in json.loads(bp.read_text()).items():
            md.append(f"| {k} | {v['median']:.1f} | {v['mean']:.1f} | {v['n']} |")
        md.append("")
    (out / "results.md").write_text("\n".join(md))
    print("wrote", out / "results.md")


def _static_rows(out: Path, clip: str, keys) -> list[dict]:
    g = GT(out, clip)
    rows = []
    for k in keys:
        for t in g.gt:
            if t != k:
                rows.append({"key": k, "t": t, "gap": abs(t - k), "ok": True, **{f"m_{s}": v for s, v in grid_err(g.gt[k]["H"], g.gt[t], g.grid(t)).items()}})
    return rows


# --------------------------------------------------------------------------
# stage: speed (CPU time, so it is meaningful on a loaded box)
# --------------------------------------------------------------------------


def cmd_bench(a):  # noqa: PLR0915
    out = Path(a.out)
    video = find_video(CLIPS_DIR, a.clip)
    ct = time.process_time
    frames = list(tp.iter_video_small(video, a.scale, a.start, a.start + a.n))
    sizes = frames[0][2]
    res: dict[str, list[float]] = {}

    def add(k, v):
        res.setdefault(k, []).append(v * 1000)

    preps = []
    for _, small, size in frames:
        for mode in ("grass", "all"):
            t0 = ct()
            p = tp.prep_frame(small, None, None, a.scale, mode, size_full=size, already_small=True)
            add(f"prep/mask ({mode})", ct() - t0)
        preps.append(p)
    preps_g = [tp.prep_frame(small, None, None, a.scale, "grass", size_full=size, already_small=True) for _, small, size in frames]
    for method in ("lk", "lk_ecc", "ecc"):
        for d in (1, 4):
            for pa, pb in zip(preps_g[:-d], preps_g[d:], strict=True):
                pa.pts = None
                t0 = ct()
                tp._pair(pa, pb, method)
                add(f"pair {method} d={d} (features+track+RANSAC)", ct() - t0)
    for _, small, _ in frames[:20]:
        t0 = ct()
        tp._hist_thumb(small)
        add("cut histogram", ct() - t0)
    # full-resolution refinement
    g = GT(out, a.clip)
    f0 = sorted(g.gt)[len(g.gt) // 2]
    full = cv2.imread(str(out / f"frame_{canonical_clip_id(a.clip)}_{f0}.jpg"))
    for ww in (960, 1280):
        t0 = ct()
        m = tp.line_masks(full, ww)
        add(f"line masks @{ww}", ct() - t0)
        Hk = g.gt[f0]["H"] @ np.array([[1, 0, 0.01], [0, 1, -0.008], [0, 0, 1.0]])
        for _ in range(3):
            t0 = ct()
            tp.refine_frame(Hk, m, PITCH)
            add(f"refine_homography @{ww} (k1 fixed)", ct() - t0)
        t0 = ct()
        tp.refine_frame(Hk, m, PITCH, fit_k1=True)
        add(f"refine_homography @{ww} (fit k1)", ct() - t0)
    print(f"{a.clip}: decoded frames {sizes}, scale {a.scale}, CPU ms (single thread)")
    for k, v in res.items():
        print(f"  {k:52s} median {np.median(v):7.1f}  mean {np.mean(v):7.1f}  n={len(v)}")
    (out / f"bench_{canonical_clip_id(a.clip)}.json").write_text(json.dumps({k: {"median": float(np.median(v)), "mean": float(np.mean(v)), "n": len(v)} for k, v in res.items()}, indent=1))


# --------------------------------------------------------------------------
# figures: before/after overlay of a far-from-keyframe frame
# --------------------------------------------------------------------------


def cmd_overlay(a):
    out = Path(a.out)
    g = GT(out, a.clip)
    cm = load_motion(out, a.clip, a.config)
    k, t = a.key, a.target
    Hk = g.gt[k]["H"]
    Hp = tp.propagate(Hk, cm, k, t)
    m = g.masks(t)
    r = tp.refine_frame(Hp, m, PITCH)
    fr = g.frame(t)
    panels = []
    for name, H in (("GT", g.gt[t]["H"]), (f"propagated from {k} (gap {abs(t - k)})", Hp), ("propagated + refined", r.H),
                    (f"static H of {k} + refined", tp.refine_frame(Hk, m, PITCH).H)):
        im = overlay_render(fr, H, PITCH, None)
        e = grid_err(H, g.gt[t], g.grid(t))
        cv2.putText(im, f"{name}: err {e['mean']:.2f} m (max {e['max']:.1f})", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(im, f"{name}: err {e['mean']:.2f} m (max {e['max']:.1f})", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        panels.append(cv2.resize(im, (960, 540)))
    grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    cv2.imwrite(str(out / f"overlay_{canonical_clip_id(a.clip)}_{k}_to_{t}.jpg"), grid)
    print("wrote overlay", k, t)


# --------------------------------------------------------------------------


def main(argv=None):  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="scratch dir for caches and results")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("motion")
    s.add_argument("--clip", required=True)
    s.add_argument("--config", default="lk_grass", choices=sorted(CONFIGS))
    s.add_argument("--max-gap", type=int, default=4)
    s.add_argument("--scale", type=int, default=480)
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--stop", type=int, default=0)
    s.set_defaults(fn=cmd_motion)
    s = sub.add_parser("gt")
    s.add_argument("--clip", required=True)
    s.set_defaults(fn=cmd_gt)
    s = sub.add_parser("prop")
    s.add_argument("--clip", required=True)
    s.add_argument("--configs", nargs="+", default=["lk_grass"])
    s.add_argument("--no-d1", action="store_true", help="skip the d=1-only and rotation+zoom variants and the GT floor rows")
    s.add_argument("--tag", default="", help="suffix of the output csv (keep earlier runs)")
    s.set_defaults(fn=cmd_prop)
    s = sub.add_parser("refine")
    s.add_argument("--clip", required=True)
    s.add_argument("--config", default="lk_grass")
    s.add_argument("--n-keys", type=int, default=3)
    s.add_argument("--target-stride", type=int, default=2)
    s.set_defaults(fn=cmd_refine)
    s = sub.add_parser("keyframes")
    s.add_argument("--clip", required=True)
    s.add_argument("--config", default="lk_grass")
    s.set_defaults(fn=cmd_keyframes)
    s = sub.add_parser("track")
    s.add_argument("--clip", required=True)
    s.add_argument("--config", default="lk_grass")
    s.add_argument("--key", type=int, default=None)
    s.add_argument("--step", type=int, default=5)
    s.set_defaults(fn=cmd_track)
    s = sub.add_parser("bench")
    s.add_argument("--clip", required=True)
    s.add_argument("--scale", type=int, default=480)
    s.add_argument("--start", type=int, default=100)
    s.add_argument("--n", type=int, default=25)
    s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("overlay")
    s.add_argument("--clip", required=True)
    s.add_argument("--config", default="lk_grass")
    s.add_argument("--key", type=int, required=True)
    s.add_argument("--target", type=int, required=True)
    s.set_defaults(fn=cmd_overlay)
    s = sub.add_parser("report")
    s.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
