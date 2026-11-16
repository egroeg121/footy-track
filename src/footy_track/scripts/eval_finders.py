# ruff: noqa: PLR0911, PLR0912, PLR0915, PLR2004
"""Offline evaluation of the two editor finders (``calibration.pitch_finders``) against the calibration GT.

Same frames, cached masks and GT conventions as ``eval_centre_circle``. For each frame both finders run;
for every returned point the pixel error against the GT homography's projection of the *same pitch id*
is measured (at 1920 x 1080), split by observed / derived. When a finder returns >= 4 non-collinear points
the editor's fit is reproduced from them alone (``geo.fit_homography`` on the points undistorted with the
GT lens, as ``/calibration/api/fit`` does) and its pitch-metre error against the GT homography is
measured on a 5 m grid of the pitch that the GT puts inside the frame.

Single process, CPU-light, reads ``eval_centre_circle``'s mask cache (adds ``*.far.npz`` far-touchline
strips next to it)::

    OMP_NUM_THREADS=1 nice -n 19 PYTHONPATH=src python -m footy_track.scripts.eval_finders
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import centre_circle as cc
from footy_track.calibration import geometry as geo
from footy_track.calibration import pitch_finders as pf
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel import candidates as cand
from footy_track.calibration.prelabel.pipeline import find_video
from footy_track.calibration.video import read_frame
from footy_track.scripts import eval_centre_circle as ev0

GT_ROOT, CLIP_DIRS, OUT = ev0.GT_ROOT, ev0.CLIP_DIRS, ev0.OUT
PX = np.array([ev0.PX_W, ev0.PX_H])
GROSS_PX = 15.0
GRID_M = 5.0
KINDS = ("centre", "d")


def load_frame(clip: str, idx: int, video, cache: Path):
    """(mask, far mask, work bgr) from the cache, decoding the frame when the cache lacks it."""
    key = f"{clip}_{idx}"
    mp, fp, farp = cache / f"{key}.mask.npz", cache / f"{key}.jpg", cache / f"{key}.far.npz"
    if mp.exists() and fp.exists():
        z = np.load(mp)
        mask = np.unpackbits(z["m"], axis=None)[: int(z["n"])].reshape(z["shape"]).astype(bool)
        work = cv2.imread(str(fp))
    else:
        if video is None:
            return None
        frame = read_frame(video, idx)
        if frame is None:
            return None
        work = cc.resize_work(frame)
        mask = cc.line_mask_for(frame)
        np.savez_compressed(mp, m=np.packbits(mask), n=mask.size, shape=np.array(mask.shape))
        cv2.imwrite(str(fp), work, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if farp.exists():
        z = np.load(farp)
        far = np.unpackbits(z["m"], axis=None)[: int(z["n"])].reshape(z["shape"]).astype(bool)
    else:
        far = cand.far_edge_line(work, cand.grass_mask(work))
        np.savez_compressed(farp, m=np.packbits(far), n=far.size, shape=np.array(far.shape))
    return mask, far, work


def gt_present(rec: dict) -> dict[str, bool]:
    s = ev0.gt_structures(rec)
    return {"centre": "centre" in s, "d": any(k.startswith("pen_arc") for k in s)}


def gt_project(rec: dict, pm: PitchModel, pts_m) -> tuple[np.ndarray, np.ndarray]:
    """Observed normalised GT image points of pitch metres, and whether each is inside the frame."""
    pts = np.array(pts_m, float).reshape(-1, 2)
    xy, w = geo.pitch_to_image(np.array(rec["H"]), pts, rec.get("distortion"))
    inb = (w > 0) & ((xy > 0) & (xy < 1)).all(1)
    return xy, inb


def run_finder(kind: str, mask, far, dist, pitch: PitchModel):
    if kind == "centre":
        return pf.find_centre(mask, dist, pitch, far_mask=far)
    return pf.find_d(mask, dist, pitch)


def editor_fit_error(points, pm_used: PitchModel, rec: dict, pm_true: PitchModel) -> dict:
    """Reproduce the editor fit from ``points`` and measure metre error on a 5 m grid (visible GT pitch)."""
    dist = rec.get("distortion")
    img = np.array([p.uv for p in points])
    ptc = np.array([p.pitch_xy for p in points])
    try:
        H, _ = geo.fit_homography(geo.undistort_points(img, dist), ptc)
    except ValueError as exc:
        return {"ok": False, "why": str(exc)[:80]}
    Hgt = np.array(rec["H"])
    gx = np.arange(-pm_true.hl, pm_true.hl + 1e-6, GRID_M)
    gy = np.arange(-pm_true.hw, pm_true.hw + 1e-6, GRID_M)
    grid = np.array([(x, y) for x in gx for y in gy])
    xy_u, w = geo.apply_homography(np.linalg.inv(Hgt), grid)  # GT undistorted normalised image
    xy_obs = geo.distort_points(xy_u, dist)
    vis = (w > 0) & ((xy_obs > 0) & (xy_obs < 1)).all(1) & np.isfinite(xy_obs).all(1)
    if vis.sum() < 4:
        return {"ok": False, "why": "no visible grid"}
    est, _ = geo.apply_homography(H, xy_u[vis])
    err = np.linalg.norm(est - grid[vis], axis=1)
    return {
        "ok": True,
        "n_grid": int(vis.sum()),
        "med_m": float(np.median(err)),
        "p95_m": float(np.percentile(err, 95)),
        "max_m": float(err.max()),
    }


def run(args) -> dict:
    cache = OUT / "cache"
    rows = []
    t0 = time.time()
    for f in sorted(GT_ROOT.glob("*.json")):
        doc = json.loads(f.read_text())
        clip = f.stem
        video = None
        for d in CLIP_DIRS:
            if d.is_dir():
                video = video or find_video(d, clip)
        for k, rec in sorted(doc["frames"].items(), key=lambda kv: int(kv[0])):
            idx = int(k)
            lf = load_frame(clip, idx, video, cache)
            if lf is None:
                continue
            mask, far, _ = lf
            dist = rec.get("distortion") if args.lens == "gt" else None
            pm_true = PitchModel(*rec["pitch_dims"])
            present = gt_present(rec)
            row = {"clip": clip, "idx": idx, "gt": present, "runs": {}}
            for kind in KINDS:
                if args.only and kind != args.only:
                    continue
                t1, c1 = time.perf_counter(), time.process_time()
                det = run_finder(kind, mask, far, dist, pm_true)
                dt = time.perf_counter() - t1
                cpu = time.process_time() - c1
                r = det.result
                e = {"answered": r is not None, "reason": det.reason, "sec": dt, "cpu": cpu, "gt_present": present[kind]}
                if kind == "centre":
                    _, fin = gt_project(rec, pm_true, [pm_true.keypoints()["HALFWAY_TOP"]])
                    e["gt_far_in"] = bool(fin[0])
                if r is not None:
                    g_uv, g_in = gt_project(rec, pm_true, [p.pitch_xy for p in r.points])
                    err = np.linalg.norm((np.array([p.uv for p in r.points]) - g_uv) * PX, axis=1)
                    e.update(
                        conf=r.confidence,
                        info={kk: vv for kk, vv in r.info.items() if kk in ("source", "end", "f1", "margin", "touchline_ends", "width_est_m", "template_precision", "template", "ambiguous_dropped", "n_elements")},
                        points=[
                            {"id": p.pitch_id or "FREE", "derived": p.derived, "conf": p.confidence, "err": float(er), "gt_in": bool(gi)}
                            for p, er, gi in zip(r.points, err, g_in, strict=True)
                        ],
                    )
                    pts = r.points
                    if len(pts) >= 4:
                        e["fit"] = editor_fit_error(pts, pm_true, rec, pm_true)
                        if not e["fit"]["ok"]:
                            e["fit"]["n_pts"] = len(pts)
                    if args.width_sens and kind == "centre" and "touchline_ends" in r.info:
                        e["width_sens"] = {}
                        for dw in (-2.0, 2.0):
                            pm_w = PitchModel(pm_true.length, pm_true.width + dw)
                            det2 = pf.find_centre(mask, dist, pm_w, far_mask=far)
                            if det2.result is not None and len(det2.result.points) >= 4:
                                e["width_sens"][str(dw)] = editor_fit_error(det2.result.points, pm_w, rec, pm_true)
                row["runs"][kind] = e
            rows.append(row)
            print(
                f"{clip}_{idx}: "
                + " | ".join(
                    f"{kd}: "
                    + (
                        f"n={len(e['points'])} med={np.median([p['err'] for p in e['points']]):.1f}px"
                        + (f" fit={e['fit']['med_m']:.2f}m" if e.get("fit", {}).get("ok") else "")
                        + f" {e['sec']:.1f}s"
                        if e["answered"]
                        else f"abstain {e['sec']:.1f}s"
                    )
                    + ("" if e["gt_present"] else " [no GT]")
                    for kd, e in row["runs"].items()
                ),
                flush=True,
            )
    print(f"{len(rows)} frames in {time.time() - t0:.0f}s")
    return {"rows": rows}


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarise(res: dict) -> str:
    out = []
    for kind in KINDS:
        rs = [r for r in res["rows"] if kind in r["runs"]]
        if not rs:
            continue
        pres = [r for r in rs if r["runs"][kind]["gt_present"]]
        absent = [r for r in rs if not r["runs"][kind]["gt_present"]]
        ans = [r for r in pres if r["runs"][kind]["answered"]]
        out.append(f"== {kind} ==")
        out.append(
            f"GT frames with structure in shot: {len(pres)}; answered {len(ans)} ({100 * len(ans) / max(1, len(pres)):.0f}%), "
            f"abstained {len(pres) - len(ans)}"
        )
        fa = [r for r in absent if r["runs"][kind]["answered"]]
        # an answer on a frame without the full structure is fine when its points really are in the GT frame and right
        bad_fa = [
            (r["clip"], r["idx"])
            for r in fa
            if any(p["err"] > GROSS_PX for p in r["runs"][kind]["points"] if p["gt_in"])
            or not any(p["gt_in"] for p in r["runs"][kind]["points"])
        ]
        out.append(
            f"answers on {len(absent)} frames without the structure in shot: {len(fa)}; of those with a wrong / "
            f"not-in-frame point: {len(bad_fa)} {bad_fa[:8]}"
        )
        for name, sel in (
            ("all points", lambda p: True),
            ("observed (derived=false)", lambda p: not p["derived"]),
            ("derived", lambda p: p["derived"]),
        ):
            errs = np.array([p["err"] for r in ans for p in r["runs"][kind]["points"] if sel(p)])
            if len(errs):
                out.append(
                    f"  px error @1920 [{name}]: n={len(errs)} median {np.median(errs):.1f}, p95 {_pct(errs, 95):.1f}, "
                    f"gross(>{GROSS_PX:.0f}) {(errs > GROSS_PX).sum()} ({100 * (errs > GROSS_PX).mean():.0f}%)"
                )
        by_id: dict[str, list[float]] = {}
        for r in ans:
            for p in r["runs"][kind]["points"]:
                by_id.setdefault(p["id"].split("_", 1)[1] if kind == "d" else p["id"], []).append(p["err"])
        out.append("  per id (n, median px, gross): " + "; ".join(f"{k} {len(v)}/{np.median(v):.1f}/{sum(e > GROSS_PX for e in v)}" for k, v in sorted(by_id.items())))
        npts = [len(r["runs"][kind]["points"]) for r in ans]
        out.append(f"  points per answered frame: {np.bincount(npts).tolist() if npts else []} (index = count)")
        wrong = [(r["clip"], r["idx"]) for r in ans if any(p["err"] > 3 * GROSS_PX and p["gt_in"] for p in r["runs"][kind]["points"])]
        out.append(f"  frames with a point > {3 * GROSS_PX:.0f}px off: {len(wrong)} {wrong[:10]}")
        fits = [r["runs"][kind]["fit"] for r in ans if "fit" in r["runs"][kind]]
        ok = [f for f in fits if f["ok"]]
        out.append(f"  frames with >=4 points: {len(fits)} (answered frames {len(ans)}); editor fit ok {len(ok)}")
        if ok:
            med = np.array([f["med_m"] for f in ok])
            p95 = np.array([f["p95_m"] for f in ok])
            out.append(
                f"  editor-fit pitch error on 5 m grid, per-frame median: median {np.median(med):.2f} m, p90 {_pct(med, 90):.2f} m, "
                f"max {med.max():.2f} m; per-frame p95: median {np.median(p95):.2f} m, worst {p95.max():.2f} m; "
                f"fits with median > 1 m: {(med > 1).sum()}"
            )
        sec = np.array([r["runs"][kind].get("cpu", r["runs"][kind]["sec"]) for r in rs])
        out.append(f"  CPU time per frame (finder only, 1280 px mask cached): median {np.median(sec):.2f}s, p95 {_pct(sec, 95):.2f}s, max {sec.max():.2f}s")
        if kind == "centre":
            ws = [r["runs"][kind]["width_sens"] for r in ans if r["runs"][kind].get("width_sens")]
            for dw in ("-2.0", "2.0"):
                v = [w[dw]["med_m"] for w in ws if dw in w and w[dw]["ok"]]
                base = [
                    r["runs"][kind]["fit"]["med_m"]
                    for r in ans
                    if r["runs"][kind].get("width_sens", {}).get(dw, {}).get("ok") and r["runs"][kind].get("fit", {}).get("ok")
                ]
                if v:
                    out.append(
                        f"  width {'+' if dw[0] != '-' else ''}{dw} m: editor-fit median error {np.median(v):.2f} m (n={len(v)}) vs {np.median(base):.2f} m with the true width"
                    )
            vis_far = [r for r in pres if r["runs"][kind].get("gt_far_in")]
            found = [r for r in vis_far if r["runs"][kind]["answered"] and "HALFWAY_TOP" in {p["id"] for p in r["runs"][kind]["points"]}]
            out.append(f"  far touchline end: GT puts HALFWAY_TOP in frame on {len(vis_far)} of the centre frames, found on {len(found)}")
    return "\n".join(out)


def contact_sheet(res: dict, path: Path, lens: str = "gt", per_kind: int = 6) -> None:
    cache = OUT / "cache"
    tiles = []
    for kind in KINDS:
        cand_rows = []
        for r in res["rows"]:
            e = r["runs"].get(kind)
            if e is None:
                continue
            if e["answered"]:
                worst = max(p["err"] for p in e["points"])
                n = len(e["points"])
                cand_rows.append((0 if e["gt_present"] else 1, -n, worst, r, e))
            elif e["gt_present"]:
                cand_rows.append((2, 0, 1e9, r, e))
        good = sorted([c for c in cand_rows if c[0] == 0 and c[1] <= -4], key=lambda c: c[2])
        picks = good[:: max(1, len(good) // 3)][:3] + sorted([c for c in cand_rows if c[0] == 0 and c[2] > GROSS_PX], key=lambda c: -c[2])[:1]
        picks += [c for c in cand_rows if c[0] == 1][:1] + [c for c in cand_rows if c[0] == 2][:1]
        picks = picks[:per_kind]
        for _, _, _, r, e in picks:
            key = f"{r['clip']}_{r['idx']}"
            img = cv2.imread(str(cache / f"{key}.jpg"))
            if img is None:
                continue
            doc = json.loads((GT_ROOT / f"{r['clip']}.json").read_text())["frames"][str(r["idx"])]
            dist = doc.get("distortion") if lens == "gt" else None
            pm = PitchModel(*doc["pitch_dims"])
            loaded = load_frame(r["clip"], r["idx"], None, cache)
            mask, far, _ = loaded
            det = run_finder(kind, mask, far, dist, pm)
            vis = pf.draw_overlay(img, det, dist, label=kind)
            h, w = vis.shape[:2]
            if det.result is not None:
                g_uv, _ = gt_project(doc, pm, [p.pitch_xy for p in det.result.points])
                for u, v in g_uv:
                    cv2.drawMarker(vis, (int(u * w), int(v * h)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            tag = "ok" if e["gt_present"] else "NO-GT"
            fit = f" fit={e['fit']['med_m']:.2f}m" if e.get("fit", {}).get("ok") else ""
            err = max((p["err"] for p in e.get("points", [])), default=float("nan"))
            cv2.putText(vis, f"{key} [{kind}] {tag} maxerr={err:.0f}px{fit}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            tiles.append(cv2.resize(vis, (640, int(640 * h / w))))
    if not tiles:
        return
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", choices=["gt", "none"], default="gt")
    ap.add_argument("--only", choices=list(KINDS))
    ap.add_argument("--width-sens", action="store_true", help="re-run the centre finder with width +-2 m")
    ap.add_argument("--out", default=str(OUT / "eval_v2.json"))
    ap.add_argument("--sheet", default=str(OUT / "contact_sheet_v2.jpg"))
    ap.add_argument("--from-json", help="skip the run, summarise / draw a saved result")
    args = ap.parse_args()
    if args.from_json:
        res = json.loads(Path(args.from_json).read_text())
    else:
        res = run(args)
        Path(args.out).write_text(json.dumps(res))
    print(summarise(res))
    contact_sheet(res, Path(args.sheet), args.lens)


if __name__ == "__main__":
    main()
