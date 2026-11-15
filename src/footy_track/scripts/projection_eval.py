"""Evaluate keyframe -> per-frame calibration -> pitch positions on clips with calibration GT.

    PYTHONPATH=src python -m footy_track.scripts.projection_eval --out OUT [--clips arsenal_mancity_seg010,arsenal_mancity_seg020]
        [--motion-dir DIR]   # reuse ClipMotion npz (``motion_<clip>__lk_boxes.npz``) instead of recomputing

For every scenario (a set of GT frames used as keyframes) it calibrates the clip with the keyframes only, then scores:

(a) calibration error in metres on the held-out GT frames (all GT frames not used as keyframes): distance between the
    estimated and the GT homography over a lattice of grass pixels, pooled over frames; split near half (bottom) /
    far half (top) of the image; separately for the frames the calibration calls ``ok``.
(c) the same boxes (machine-label person boxes at the held-out GT frames, foot point) projected through the GT H and
    through the estimated H (and through the nearest keyframe's H unchanged as a no-tracking baseline): player
    position error in metres.
(b) on the whole clip: fraction of projected players inside the pitch (+5 m), frame-to-frame jitter of player
    positions (Hungarian match between consecutive frames, gate 1.5 m; the per-frame-pair median displacement of the
    matched players is a calibration-induced common shift) and the coverage of ``ok`` frames.

Writes ``eval.json`` and ``results_tables.md`` under ``--out`` (the calibrations as parquet next to them).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from footy_track.calibration import geometry as geo
from footy_track.calibration import temporal as tp
from footy_track.calibration.clip_calibration import (
    ClipCalibration,
    MaskStore,
    Policy,
    calibrate_clip,
)
from footy_track.calibration.prelabel.pipeline import find_video
from footy_track.clip_id import canonical_clip_id
from footy_track.projection.players import (
    load_detections,
    project_boxes,
    project_players,
)
from footy_track.scripts.calibrate_clip import (
    CLIPS_DIR,
    find_boxes_path,
    load_gt_keyframes,
    pick_nearest,
)

PERSON = ("player", "referee", "coach", "person", "player_sub")
PITCH = (105.0, 68.0)


def scenarios(clip: str, gt_frames: list[int]) -> dict[str, tuple[list[int], bool]]:
    """name -> (keyframes, refine)."""
    if clip.endswith("seg010"):
        return {
            "k6_every100": (pick_nearest(gt_frames, [0, 100, 200, 300, 400, 500]), True),
            "k6_every100_norefine": (pick_nearest(gt_frames, [0, 100, 200, 300, 400, 500]), False),
            "k2": (pick_nearest(gt_frames, [150, 450]), True),
            "k1_mid": (pick_nearest(gt_frames, [300]), True),
            "k1_mid_norefine": (pick_nearest(gt_frames, [300]), False),
            "k1_first": (pick_nearest(gt_frames, [0]), True),
        }
    n = len(gt_frames)
    if "seg020" in clip:
        return {"k1": ([gt_frames[n // 2]], True), "k2": ([gt_frames[0], gt_frames[-1]], True)}
    return {"k1_mid": ([gt_frames[n // 2]], True), "k1_first": ([gt_frames[0]], True)}  # generalisation clips: 3-4 GT frames each


def err_stats(e: np.ndarray) -> dict:
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "mean": float(e.mean()), "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
            "max": float(e.max()), "lt1": float((e < 1).mean()), "lt2": float((e < 2).mean())}


def halves(e: np.ndarray, y: np.ndarray) -> dict:
    return {"all": err_stats(e), "near": err_stats(e[y >= 0.5]), "far": err_stats(e[y < 0.5])}


def grass_points(masks: MaskStore, f: int, step: int = 24) -> np.ndarray:
    m = masks.get(f)
    h, w = m.grass.shape
    ys, xs = np.nonzero(m.grass[step // 2::step, step // 2::step])
    return np.stack([(xs * step + step // 2) / w, (ys * step + step // 2) / h], 1)


def pitch_pts(H, pts, dist=None):
    return geo.image_to_pitch(H, pts, dist)[0]


def held_out_errors(cal: ClipCalibration, gt: dict, keys: list[int], masks: MaskStore, dets: pd.DataFrame | None) -> dict:
    e_all, y_all, e_ok, y_ok, e_static, y_static = [], [], [], [], [], []
    p_all, py_all, p_ok, py_ok, p_static, py_static = [], [], [], [], [], []
    held = [f for f in sorted(gt) if f not in keys]
    per_frame = []
    n_ok = 0
    for f in held:
        pts = grass_points(masks, f)
        Hg = gt[f][0]
        Hk = gt[min(keys, key=lambda k: abs(k - f))][0]
        ref = pitch_pts(Hg, pts)
        He = cal.H_any(f)
        row = cal.row(f)
        d = {"frame": f, "ok": bool(row["ok"]), "source": row["source"], "gap": int(row["gap"]), "key": int(row["key_frame"]),
             "line_cost": float(row["line_cost"]), "confidence": float(row["confidence"])}
        e_s = np.linalg.norm(pitch_pts(Hk, pts) - ref, axis=1)
        e_static += list(e_s)
        y_static += list(pts[:, 1])
        if He is not None:
            e = np.linalg.norm(pitch_pts(He, pts, cal.distortion(f)) - ref, axis=1)
            e_all += list(e)
            y_all += list(pts[:, 1])
            d["mean_err_m"] = float(np.mean(e))
            if row["ok"]:
                n_ok += 1
                e_ok += list(e)
                y_ok += list(pts[:, 1])
        # (c) players
        if dets is not None:
            b = dets[(dets["frame"] == f) & dets["label"].isin(PERSON)]
            if len(b):
                bx = b[["x", "y", "w", "h"]].to_numpy()
                g = project_boxes(bx, Hg, None, pitch_dims=PITCH)
                inb = (np.abs(g[:, 0]) <= PITCH[0] / 2 + 5) & (np.abs(g[:, 1]) <= PITCH[1] / 2 + 5)
                foot_y = (bx[:, 1] + bx[:, 3])[inb]
                gg = g[inb]
                s = np.linalg.norm(project_boxes(bx[inb], Hk, None) - gg, axis=1)
                p_static += list(s)
                py_static += list(foot_y)
                if He is not None:
                    pe = np.linalg.norm(project_boxes(bx[inb], He, cal.distortion(f)) - gg, axis=1)
                    p_all += list(pe)
                    py_all += list(foot_y)
                    d["player_mean_err_m"] = float(pe.mean()) if len(pe) else None
                    if row["ok"]:
                        p_ok += list(pe)
                        py_ok += list(foot_y)
        per_frame.append(d)
    A = lambda x: np.asarray(x, float)  # noqa: E731
    return {"n_held_out": len(held), "n_held_out_ok": n_ok,
            "calib_all": halves(A(e_all), A(y_all)), "calib_ok": halves(A(e_ok), A(y_ok)),
            "calib_static_keyframe": halves(A(e_static), A(y_static)),
            "players_all": halves(A(p_all), A(py_all)), "players_ok": halves(A(p_ok), A(py_ok)),
            "players_static_keyframe": halves(A(p_static), A(py_static)), "per_frame": per_frame}


def jitter(pos: pd.DataFrame, cal: ClipCalibration, gate: float = 1.5) -> dict:
    """Frame-to-frame displacement (m) of matched players and the common shift of each frame pair."""
    p = pos[(pos["label"] == "player") & pos["calib_ok"] & pos["in_bounds"]]
    by = {f: g[["x_m", "y_m"]].to_numpy() for f, g in p.groupby("frame")}
    shot = cal.table.set_index("frame")["shot_id"]
    refined = set(cal.table.loc[cal.table["source"] == "refined", "frame"])
    disp, common, common_ref = [], [], []
    for f in sorted(by):
        if f + 1 not in by or shot[f] != shot[f + 1]:
            continue
        a, b = by[f], by[f + 1]
        D = np.linalg.norm(a[:, None] - b[None], axis=2)
        r, c = linear_sum_assignment(D)
        ok = D[r, c] < gate
        if ok.sum() < 4:
            continue
        v = b[c[ok]] - a[r[ok]]
        disp += list(np.linalg.norm(v, axis=1))
        cm = float(np.linalg.norm(np.median(v, axis=0)))
        common.append(cm)
        if f + 1 in refined or f in refined:
            common_ref.append(cm)
    disp, common = np.array(disp), np.array(common)
    q = lambda x, k: float(np.percentile(x, k)) if len(x) else None  # noqa: E731
    return {"pairs_matched": int(len(disp)), "frame_pairs": int(len(common)),
            "disp_median_m": q(disp, 50), "disp_p95_m": q(disp, 95), "disp_frac_gt_0p4m": float((disp > 0.4).mean()) if len(disp) else None,
            "common_shift_median_m": q(common, 50), "common_shift_p95_m": q(common, 95),
            "common_shift_frac_gt_0p3m": float((common > 0.3).mean()) if len(common) else None,
            "common_shift_at_refinement_median_m": q(np.array(common_ref), 50), "n_refinement_pairs": len(common_ref)}


def whole_clip(cal: ClipCalibration, dets: pd.DataFrame | None) -> dict:
    t = cal.table
    shots_with_key = set(t.loc[t["source"] == "keyframe", "shot_id"])
    in_key_shot = t["shot_id"].isin(shots_with_key)
    out = {"frames": len(t), "ok_frac_all_frames": float(t["ok"].mean()),
           "ok_frac_keyframed_shots": float(t.loc[in_key_shot, "ok"].mean()), "frames_in_keyframed_shots": int(in_key_shot.sum()),
           "sources": t["source"].value_counts().to_dict(), "stats": {k: v for k, v in cal.stats.items() if k in ("n_refine", "n_accepted", "n_rejected", "refine_s", "sweep_s")}}
    if dets is not None:
        pos = project_players(dets, cal, PITCH)
        pl = pos[(pos["label"] == "player") & pos["calib_ok"]]
        out["players"] = {"n_boxes": int((pos["label"] == "player").sum()), "n_projected": int(len(pl)),
                          "in_pitch_frac": float(pl["in_pitch"].mean()), "in_bounds_5m_frac": float(pl["in_bounds"].mean()),
                          "median_players_per_frame": float(pl.groupby("frame").size().median())}
        out["jitter"] = jitter(pos, cal)
    return out


def _row(label: str, st: dict) -> str:
    if not st.get("n"):
        return f"| {label} | 0 | - | - | - | - | - |"
    return f"| {label} | {st['n']} | {st['mean']:.2f} | {st['median']:.2f} | {st['p95']:.2f} | {st['lt1']:.2f} | {st['lt2']:.2f} |"


def write_tables(res: dict, path: Path) -> None:
    """Markdown tables (a), (c), (b) from ``eval.json``-style results."""
    hdr = "| scenario | n points | mean (m) | median | p95 | < 1 m | < 2 m |\n|---|---|---|---|---|---|---|"
    out = []
    for title, key, base in (("(a) Calibration error vs held-out GT, grass lattice, `ok` frames", "calib_ok", "calib_static_keyframe"),
                             ("(c) Player foot-point position error, GT H vs estimated H, `ok` frames", "players_ok", "players_static_keyframe")):
        out += [f"### {title}", "", hdr]
        for k, v in res.items():
            ho = v["held_out"]
            for half in ("all", "near", "far"):
                out.append(_row(f"{k} (keys {v['keyframes']}) {half}", ho[key][half]))
            out.append(_row(f"{k} - static nearest-keyframe H, all", ho[base]["all"]))
        out.append("")
    out += ["### (b) Whole clip: coverage, players in the pitch, jitter", "",
            "| scenario | ok frames (all / keyframed shot) | players in pitch / +5 m | disp median / p95 (m) | frac > 0.4 m | common shift median / p95 (m) | refined / rejected |",
            "|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        c, j, p = v["clip_level"], v["clip_level"].get("jitter") or {}, v["clip_level"].get("players") or {}
        if not p:
            continue
        out.append(f"| {k} | {c['ok_frac_all_frames']:.3f} / {c['ok_frac_keyframed_shots']:.3f} | {p['in_pitch_frac']:.3f} / {p['in_bounds_5m_frac']:.3f} | "
                   f"{j['disp_median_m']:.2f} / {j['disp_p95_m']:.2f} | {j['disp_frac_gt_0p4m']:.3f} | "
                   f"{j['common_shift_median_m']:.2f} / {j['common_shift_p95_m']:.2f} | {c['stats']['n_accepted']} / {c['stats']['n_rejected']} |")
    path.write_text("\n".join(out) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", default="arsenal_mancity_seg010,arsenal_mancity_seg020")
    ap.add_argument("--motion-dir", type=Path, help="reuse motion_<clip>__lk_boxes.npz from the temporal experiment")
    ap.add_argument("--only", help="comma list of scenario names")
    ap.add_argument("--tables-only", action="store_true", help="just rewrite results_tables.md from eval.json")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    res_path = a.out / "eval.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    if a.tables_only:
        write_tables(res, a.out / "results_tables.md")
        return 0
    for clip in a.clips.split(","):
        cid = canonical_clip_id(clip)
        video = find_video(CLIPS_DIR, clip)
        gt_full = load_gt_keyframes(clip)
        gt = {f: (H, d) for f, (H, d) in gt_full.items()}
        bp = find_boxes_path(clip)
        dets = load_detections(bp) if bp else None
        boxes = tp.load_person_boxes(bp) if bp else None
        mp = a.out / f"{cid}.motion.npz"
        if mp.exists():
            motion = tp.ClipMotion.load(mp)
        elif a.motion_dir and (a.motion_dir / f"motion_{cid}__lk_boxes.npz").exists():
            motion = tp.ClipMotion.load(a.motion_dir / f"motion_{cid}__lk_boxes.npz")
        else:
            motion = tp.clip_motion(video, boxes=boxes, mask_mode="all")
            motion.save(mp)
        print(f"{cid}: {motion.n} frames, cuts {motion.cuts}", flush=True)
        pol0 = Policy()
        kp = a.out / f"{cid}.masks.npz"
        masks = MaskStore.load(kp) if kp.exists() else MaskStore(pol0.work_width)
        t0 = time.perf_counter()
        n0 = len(masks)
        masks.ensure(video, set(range(motion.start, motion.start + motion.n, pol0.check_every)) | set(gt), lambda i: print(f"  masks {i}", flush=True) if i % 100 == 0 else None)
        if len(masks) != n0:
            masks.save(kp)
            print(f"  masks {len(masks)} frames in {time.perf_counter() - t0:.0f}s", flush=True)
        gt_h = {f: gt[f] for f in gt}
        for name, (keys, refine) in scenarios(cid, sorted(gt)).items():
            if a.only and name not in a.only.split(","):
                continue
            key = f"{cid}/{name}"
            t0 = time.perf_counter()
            cal = calibrate_clip(video, {k: gt_h[k] for k in keys}, boxes, Policy(refine=refine), motion=motion, masks=masks)
            cal.save(a.out / f"{cid}.{name}.calibration.parquet")
            r = {"clip": cid, "scenario": name, "keyframes": keys, "refine": refine, "calibrate_s": time.perf_counter() - t0,
                 "held_out": held_out_errors(cal, {f: (H, d) for f, (H, d) in gt_h.items()}, keys, masks, dets),
                 "clip_level": whole_clip(cal, dets)}
            res[key] = r
            res_path.write_text(json.dumps(res, indent=1, default=float))
            ho = r["held_out"]["calib_all"]["all"]
            print(f"  {name:24s} keys={keys} n_ho={r['held_out']['n_held_out']} mean={ho.get('mean', float('nan')):.2f} "
                  f"med={ho.get('median', float('nan')):.2f} ok={r['clip_level']['ok_frac_all_frames']:.2f} "
                  f"({r['calibrate_s']:.0f}s, refines {cal.stats['n_refine']}/{cal.stats['n_accepted']} accepted)", flush=True)
    write_tables(res, a.out / "results_tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
