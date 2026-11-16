# ruff: noqa: PLR0911, PLR0912, PLR0915
"""Offline evaluation of ``calibration.centre_circle`` against the calibration GT.

For every GT frame whose homography puts the two ``CENTRE_CIRCLE_*`` points (or both ends of a penalty-D,
``{L,R}_PEN_ARC_*``) inside the frame, run the detector and compare with the GT-projected points. Frames
where the structure is *not* in shot count as false answers when the detector answers.

Single process, decodes only the GT frames, caches masks + work frames under ``--cache``::

    OMP_NUM_THREADS=1 nice -n 19 PYTHONPATH=src python -m footy_track.scripts.eval_centre_circle
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
from footy_track.calibration.pitch_model import (
    CIRCLE_RADIUS,
    PEN_BOX_DEPTH,
    PEN_SPOT_DIST,
    PitchModel,
)
from footy_track.calibration.prelabel.pipeline import find_video
from footy_track.calibration.video import read_frame

GT_ROOT = Path("/mnt/storage/footy_data/ball_gt_marks/calibration_gt")
CLIP_DIRS = [
    Path("/mnt/storage/footy_data/eval_clips_kf10"),
    Path("/mnt/storage/footy_data/ballcheck_clips_kf10"),
]
OUT = Path("/mnt/storage/footy_data/scratch/centre_circle")
PX_W, PX_H = 1920.0, 1080.0
GROSS_PX = 15.0
MARGIN = 0.01


def _in_frame(xy: np.ndarray, w: np.ndarray) -> bool:
    return bool((w > 0).all() and ((xy > MARGIN) & (xy < 1 - MARGIN)).all())


def gt_structures(rec: dict, strict: bool = True) -> dict[str, dict]:
    """``kind[:end]`` -> {ids, uv (2,2) observed, pole_h, kind} for structures with both points in frame."""
    pm = PitchModel(*rec["pitch_dims"])
    H = np.array(rec["H"])
    dist = rec.get("distortion")
    Hinv = np.linalg.inv(H)
    kp = pm.keypoints()
    out = {}
    specs = {
        "centre": ("CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM"),
        "pen_arc:L": ("L_PEN_ARC_TOP", "L_PEN_ARC_BOTTOM"),
        "pen_arc:R": ("R_PEN_ARC_TOP", "R_PEN_ARC_BOTTOM"),
    }
    for key, ids in specs.items():
        pts = np.array([kp[i] for i in ids])
        xy, w = geo.pitch_to_image(H, pts, dist)
        if strict and not _in_frame(xy, w):
            continue
        if not strict and not ((w > 0).all() and (np.abs(xy - 0.5) < 1.5).all()):
            continue  # behind the camera / absurdly far off
        # pole of the line wrt the circle: centre circle -> point at infinity along x; D -> finite point on the axis
        if key == "centre":
            pole_m = np.array([1.0, 0.0, 0.0])
        else:
            s = -1.0 if key.endswith("L") else 1.0
            d = PEN_BOX_DEPTH - PEN_SPOT_DIST
            spot_x = s * (pm.hl - PEN_SPOT_DIST)
            # pole of the box front (distance d from the spot, away from the goal) wrt the circle
            pole_m = np.array([spot_x - s * (CIRCLE_RADIUS**2 / d), 0.0, 1.0])
        Hi = Hinv  # pitch -> undistorted normalised image
        ph = Hi @ pole_m
        out[key] = {"ids": ids, "uv": xy, "pole_h": ph}
    return out


def _axis_angle_err(det: cc.CirclePoints, gt_pole_h: np.ndarray) -> float:
    """Degrees between the detected axis direction at the chord midpoint and the GT one at the same point."""
    d_gt = (gt_pole_h[:2] - det.midpoint_u * gt_pole_h[2]) * np.array([16.0, 9.0])
    d_dt = det.axis_direction()
    n = np.linalg.norm(d_gt)
    if n == 0:
        return float("nan")
    c = abs(float(d_dt @ (d_gt / n)))  # a direction, sign-free
    return float(np.degrees(np.arccos(min(1.0, c))))


def run(args) -> dict:
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
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
            gts = gt_structures(rec)
            key = f"{clip}_{idx}"
            mp, fp = cache / f"{key}.mask.npz", cache / f"{key}.jpg"
            if mp.exists() and fp.exists():
                z = np.load(mp)
                mask = (
                    np.unpackbits(z["m"], axis=None)[: int(z["n"])]
                    .reshape(z["shape"])
                    .astype(bool)
                )
                work = cv2.imread(str(fp))
            else:
                if video is None:
                    continue
                frame = read_frame(video, idx)
                if frame is None:
                    continue
                work = cc.resize_work(frame)
                mask = cc.line_mask_for(frame)
                np.savez_compressed(
                    mp, m=np.packbits(mask), n=mask.size, shape=np.array(mask.shape)
                )
                cv2.imwrite(str(fp), work, [cv2.IMWRITE_JPEG_QUALITY, 90])
            dist = rec.get("distortion") if args.lens == "gt" else None
            row = {
                "clip": clip,
                "idx": idx,
                "gt": sorted(gts),
                "lens": bool(dist),
                "runs": {},
            }
            for kind in ("centre", "pen_arc"):
                det = cc.detect_from_mask(
                    mask, dist, (kind,), min_confidence=args.min_conf
                )
                r = det.result
                entry = {
                    "answered": r is not None,
                    "reason": det.reason,
                    "notes": det.notes[:2],
                }
                if r is not None:
                    entry.update(
                        conf=r.confidence,
                        ids=list(r.pitch_ids),
                        uv=r.points_uv.tolist(),
                        info=r.info,
                    )
                    keys = [kk for kk in gts if kk.split(":")[0] == kind]
                    if keys:
                        # the end (L/R) is a detector output: score against the GT structure it names (or best)
                        best = None
                        for kk in keys:
                            g = gts[kk]
                            e = np.linalg.norm(
                                (r.points_uv - g["uv"]) * np.array([PX_W, PX_H]), axis=1
                            )
                            tot = float(e.max())
                            if best is None or tot < best[0]:
                                best = (tot, kk, e)
                        _, kk, e = best
                        g = gts[kk]
                        entry["gt_key"] = kk
                        entry["err_px"] = e.tolist()
                        entry["ids_ok"] = list(r.pitch_ids) == list(g["ids"])
                        entry["axis_err_deg"] = _axis_angle_err(r, g["pole_h"])
                rows_has = kind in {kk.split(":")[0] for kk in gts}
                if r is not None and not rows_has:
                    # not fully in shot: a detection is still right if it lands on a GT structure that is partly off-frame
                    loose = [
                        g
                        for kk, g in gt_structures(rec, strict=False).items()
                        if kk.split(":")[0] == kind
                    ]
                    if loose:
                        entry["partial_err_px"] = min(
                            float(
                                np.linalg.norm(
                                    (r.points_uv - g["uv"]) * np.array([PX_W, PX_H]),
                                    axis=1,
                                ).max()
                            )
                            for g in loose
                        )
                entry["gt_present"] = rows_has
                row["runs"][kind] = entry
            rows.append(row)
            print(
                f"{key}: gt={sorted(gts)} "
                + " | ".join(
                    f"{kd}: "
                    + (
                        f"ans conf={e['conf']:.2f} err={max(e['err_px']):.1f}"
                        if e["answered"] and "err_px" in e
                        else ("ans(FALSE)" if e["answered"] else "abstain")
                    )
                    for kd, e in row["runs"].items()
                ),
                flush=True,
            )
    print(f"{len(rows)} frames in {time.time() - t0:.0f}s")
    return {"rows": rows}


def summarise(res: dict) -> str:
    out = []
    for kind in ("centre", "pen_arc"):
        pres = [r for r in res["rows"] if r["runs"][kind]["gt_present"]]
        absent = [r for r in res["rows"] if not r["runs"][kind]["gt_present"]]
        ans = [r for r in pres if r["runs"][kind]["answered"]]
        errs = np.array([max(r["runs"][kind]["err_px"]) for r in ans])
        errs_all = np.array([e for r in ans for e in r["runs"][kind]["err_px"]])
        gross = [
            (r["clip"], r["idx"], round(max(r["runs"][kind]["err_px"]), 1))
            for r in ans
            if max(r["runs"][kind]["err_px"]) > GROSS_PX
        ]
        fa_all = [r for r in absent if r["runs"][kind]["answered"]]
        part = [
            (r["clip"], r["idx"], round(r["runs"][kind]["partial_err_px"], 1))
            for r in fa_all
            if r["runs"][kind].get("partial_err_px", 1e9) <= GROSS_PX
        ]
        fa = [
            (r["clip"], r["idx"])
            for r in fa_all
            if r["runs"][kind].get("partial_err_px", 1e9) > GROSS_PX
        ]
        axis = np.array(
            [
                r["runs"][kind]["axis_err_deg"]
                for r in ans
                if np.isfinite(r["runs"][kind].get("axis_err_deg", np.nan))
            ]
        )
        idbad = [(r["clip"], r["idx"]) for r in ans if not r["runs"][kind]["ids_ok"]]
        out.append(f"== {kind} ==")
        out.append(
            f"GT frames with structure in shot: {len(pres)}; answered {len(ans)} ({100 * len(ans) / max(1, len(pres)):.0f}%), "
            f"abstained {len(pres) - len(ans)} ({100 * (len(pres) - len(ans)) / max(1, len(pres)):.0f}%)"
        )
        if ans:
            out.append(
                f"per-point px error @1920: median {np.median(errs_all):.1f}, p95 {np.percentile(errs_all, 95):.1f}; "
                f"per-frame worst-point median {np.median(errs):.1f}, p95 {np.percentile(errs, 95):.1f}"
            )
            out.append(f"gross (>{GROSS_PX:.0f} px): {len(gross)}  {gross}")
            out.append(f"wrong pitch ids: {len(idbad)} {idbad}")
            if len(axis):
                out.append(
                    f"axis-direction error deg: median {np.median(axis):.1f}, p95 {np.percentile(axis, 95):.1f} (n={len(axis)})"
                )
        out.append(
            f"answers on {len(absent)} frames without both points in shot: {len(fa_all)}; of those matching a partly-visible GT structure (<={GROSS_PX:.0f}px): {len(part)} {part}; true false answers: {len(fa)} {fa}"
        )
    return "\n".join(out)


def contact_sheet(
    res: dict,
    path: Path,
    lens: str = "gt",
    n_good: int = 6,
    n_bad: int = 6,
    min_conf: float | None = None,
) -> None:
    """Tile of overlays: detector (green far / orange near, ellipse, line) vs GT-projected points (red crosses)."""
    cache = OUT / "cache"
    cand = []  # (sort key, row, kind)
    for r in res["rows"]:
        for kind, e in r["runs"].items():
            if e["gt_present"] and e["answered"] and "err_px" in e:
                cand.append((max(e["err_px"]), r, kind, "answered"))
            elif e["gt_present"]:
                cand.append((1e9, r, kind, "missed"))
            elif e["answered"]:
                cand.append((1e8, r, kind, "partial-view ok" if e.get("partial_err_px", 1e9) <= GROSS_PX else "FALSE"))
    answered = sorted([c for c in cand if c[3] == "answered"], key=lambda c: c[0])
    good = answered[:: max(1, len(answered) // n_good)][:n_good]
    bad = (
        [c for c in answered if c[0] > GROSS_PX][-3:]
        + [c for c in cand if c[3] in ("FALSE", "partial-view ok")][:2]
        + [c for c in cand if c[3] == "missed"][:3]
    )
    picks = (good + bad)[: n_good + n_bad]
    tiles = []
    for _, r, kind, tag in picks:
        key = f"{r['clip']}_{r['idx']}"
        img = cv2.imread(str(cache / f"{key}.jpg"))
        if img is None:
            continue
        doc = json.loads((GT_ROOT / f"{r['clip']}.json").read_text())["frames"][
            str(r["idx"])
        ]
        dist = doc.get("distortion") if lens == "gt" else None
        mask = np.load(cache / f"{key}.mask.npz")
        m = (
            np.unpackbits(mask["m"], axis=None)[: int(mask["n"])]
            .reshape(mask["shape"])
            .astype(bool)
        )
        det = cc.detect_from_mask(m, dist, (kind,), min_confidence=min_conf)
        vis = cc.draw_overlay(img, det, dist)
        h, w = vis.shape[:2]
        for g in gt_structures(doc).values():
            for u, v in g["uv"]:
                cv2.drawMarker(
                    vis,
                    (int(u * w), int(v * h)),
                    (0, 0, 255),
                    cv2.MARKER_TILTED_CROSS,
                    18,
                    2,
                )
        err = max(r["runs"][kind].get("err_px", [float("nan")]))
        err = err if np.isfinite(err) else r["runs"][kind].get("partial_err_px", float("nan"))
        label = f"{key} [{kind}] {tag} err={err:.1f}px"
        cv2.putText(
            vis, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        tiles.append(cv2.resize(vis, (640, int(640 * h / w))))
    if not tiles:
        return
    th = max(t.shape[0] for t in tiles)
    tiles = [
        cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for t in tiles
    ]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lens",
        choices=["gt", "none"],
        default="gt",
        help="distortion handed to the detector",
    )
    ap.add_argument(
        "--min-conf",
        type=float,
        default=None,
        help="override the per-kind confidence floor",
    )
    ap.add_argument("--out", default=str(OUT / "eval.json"))
    ap.add_argument("--sheet", default=str(OUT / "contact_sheet.jpg"))
    args = ap.parse_args()
    res = run(args)
    Path(args.out).write_text(json.dumps(res))
    print(summarise(res))
    contact_sheet(res, Path(args.sheet), args.lens, min_conf=args.min_conf)


if __name__ == "__main__":
    main()
