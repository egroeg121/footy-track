"""Evaluate the calibration editor's lens support on labelled GT frames (docs: PR "calibration editor lens").

For each GT frame, start from the GT points fitted *without* distortion (a plain 8-DOF homography, what the
editor produced before) and report the long-boundary-line residual near the frame edges and the line F1 for

* ``none``  - the plain homography,
* ``rig``   - the same points refit with the ground's suggested ``k1`` (``lens.default_distortion``; NOT applied by default in the editor),
* ``snap``  - ``lens.snap_to_lines`` as the editor runs it with the lens off (from ``none`` and from the suggested k1, best accepted answer),

and render a before/after sheet (frame edge crop + top-down warp near the touchline). Read-only on the GT
and the clips; run one process at a time (``nice -n 19``, ``OMP_NUM_THREADS=1``).

    python -m footy_track.scripts.calib_lens_eval --out /mnt/storage/footy_data/scratch/calib_bend
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration import lens
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.pipeline import find_video
from footy_track.calibration.video import read_frame

GT_DIR = Path("/mnt/storage/footy_data/ball_gt_marks/calibration_gt")
CLIP_DIRS = [Path("/mnt/storage/footy_data/eval_clips_kf10"), Path("/mnt/storage/footy_data/ballcheck_clips_kf10")]

#: (clip, frame) - Man City + Villa (Emirates) and Bournemouth (Vitality); >= 5 hand/machine points each
FRAMES = [
    ("arsenal_mancity_seg000", 1901), ("arsenal_mancity_seg020", 56), ("arsenal_mancity_seg020", 131),
    ("arsenal_mancity_seg104", 184), ("arsenal_mancity_seg104", 110), ("arsenal_mancity_seg104", 36),
    ("arsenal_mancity_seg140", 44), ("arsenal_mancity_seg140", 133), ("arsenal_mancity_seg100", 1242),
    ("arsenal_astonvilla_seg020", 36), ("arsenal_astonvilla_seg020", 5), ("arsenal_astonvilla_seg050", 302),
    ("arsenal_astonvilla_seg080", 403), ("arsenal_astonvilla_seg110", 405),
    ("arsenal_bournemouth_1st_seg050", 87), ("arsenal_bournemouth_1st_seg050", 49),
    ("arsenal_bournemouth_1st_seg050", 57), ("arsenal_bournemouth_1st_seg050", 27),
    ("arsenal_bournemouth_1st_seg009", 29), ("arsenal_bournemouth_1st_seg015", 0),
]
#: frames drawn on the sheet
SHEET = [
    ("arsenal_mancity_seg020", 56), ("arsenal_mancity_seg000", 1901), ("arsenal_mancity_seg104", 36),
    ("arsenal_astonvilla_seg050", 302), ("arsenal_astonvilla_seg110", 405),
]


def _fit(img: np.ndarray, xy: np.ndarray, dist: dict | None) -> np.ndarray:
    H, _ = geo.fit_homography(geo.undistort_points(img, dist), xy, method="dlt")
    return H


def _frame(clip: str, idx: int):
    for d in CLIP_DIRS:
        v = find_video(d, clip)
        if v is not None:
            return read_frame(v, idx)
    return None


def evaluate_frame(clip: str, idx: int) -> dict | None:
    return _evaluate(clip, idx)


def _evaluate(clip: str, idx: int) -> dict | None:
    rec = json.loads((GT_DIR / f"{clip}.json").read_text())["frames"][str(idx)]
    fr = _frame(clip, idx)
    if fr is None:
        return None
    pitch = PitchModel(*rec["pitch_dims"])
    corr = rec["correspondences"]
    img = np.array([c["image"] for c in corr], float)
    xy = np.array([c["pitch_xy"] for c in corr], float)
    ev = lens.Evidence.from_frame(fr)
    rig = lens.default_distortion(clip, fr.shape[1] / fr.shape[0])
    out = {"clip": clip, "idx": idx, "n": len(corr), "frame": fr, "pitch": pitch, "img": img, "xy": xy, "ev": ev}

    def metrics(tag: str, H: np.ndarray, d: dict | None) -> None:
        e = lens.edge_residual(ev, H, d, pitch)
        out[tag] = {"H": H, "dist": d, "f1": lens.line_f1(ev, H, d, pitch)["f1"], **e}

    H0 = _fit(img, xy, None)
    metrics("none", H0, None)
    if rig:
        metrics("rig", _fit(img, xy, rig), rig)
    # the editor with the lens off: from the plain fit and from the ground's suggested k1, better accepted answer wins
    r = lens.snap_best(ev, H0, None, pitch, img, xy, suggested=(_fit(img, xy, rig), rig) if rig else None)
    metrics("snap", r.H, r.distortion)
    out["snap"].update(accepted=r.accepted, reason=r.reason, k1=r.k1_after, shift=r.max_click_shift_px,
                       points=r.points)
    return out


# --------------------------------------------------------------------------
# sheet
# --------------------------------------------------------------------------


def _lines_px(H, dist, pitch, names, size):
    w, h = size
    runs = []
    for n in names:
        poly = pitch.polylines()[n]
        for run in geo.project_polyline_to_image(H, poly, 0.5, dist):
            runs.append(np.round(run * np.array([w, h])).astype(np.int32))
    return runs


def _edge_crop(res: dict, tag: str, box, colour) -> np.ndarray:
    fr = res["frame"].copy()
    h, w = fr.shape[:2]
    v = res[tag]
    names = ["TOUCHLINE_TOP", "TOUCHLINE_BOTTOM", "GOAL_LINE_L", "GOAL_LINE_R"]
    x0, y0, x1, y1 = box
    crop = fr[y0:y1, x0:x1]
    z = 4.0
    crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    for run in geo.project_polyline_to_image(v["H"], res["pitch"].polylines()["TOUCHLINE_TOP"], 0.1, v["dist"]) + \
            geo.project_polyline_to_image(v["H"], res["pitch"].polylines()["TOUCHLINE_BOTTOM"], 0.1, v["dist"]) + \
            [r for n in names[2:] for r in geo.project_polyline_to_image(v["H"], res["pitch"].polylines()[n], 0.1, v["dist"])]:
        px = (run * np.array([w, h]) - np.array([x0, y0])) * z
        cv2.polylines(crop, [np.round(px).astype(np.int32)], False, colour, 1, cv2.LINE_AA)
    return crop


def _edge_box(res: dict) -> tuple[int, int, int, int]:
    """Crop box (px) on the frame edge where a touchline is furthest out: where the bend shows."""
    h, w = res["frame"].shape[:2]
    v, pitch = res["snap"], res["pitch"]
    best = None
    for side in (0, 1):
        pts = []
        for n in ("TOUCHLINE_TOP", "TOUCHLINE_BOTTOM"):
            for run in geo.project_polyline_to_image(v["H"], pitch.polylines()[n], 0.5, v["dist"]):
                m = (run[:, 0] > 0) & (run[:, 0] < 1) & (run[:, 1] > 0) & (run[:, 1] < 1)
                m &= (run[:, 0] < 0.2) if side == 0 else (run[:, 0] > 0.8)
                pts.extend(run[m])
        if pts and (best is None or len(pts) > len(best[1])):
            best = (side, np.array(pts))
    bw, bh = int(w * 0.22), int(h * 0.12)
    if best is None:
        return 0, 0, bw, bh
    side, pts = best
    cy = float(np.median(pts[:, 1]) * h)
    x0 = 0 if side == 0 else w - bw
    y0 = int(np.clip(cy - bh / 2, 0, h - bh))
    return x0, y0, x0 + bw, y0 + bh


def _topdown_crop(res: dict, tag: str, box) -> np.ndarray:
    """Top-down warp (20 px/m) near the touchline at the chosen frame edge, with the pitch lines drawn on it."""
    pitch, v, fr = res["pitch"], res[tag], res["frame"]
    h, w = fr.shape[:2]
    scale, margin = 20.0, 3.0
    ras = geo.warp_to_pitch(fr, v["H"], pitch, scale, margin, v["dist"])
    x0, y0, x1, y1 = box
    centre = np.array([[(x0 + x1) / 2 / w, (y0 + y1) / 2 / h]])
    m, _ = geo.image_to_pitch(res["snap"]["H"], centre, res["snap"]["dist"])
    cx, cy = m[0]
    T = geo.pitch_raster_transform(pitch, scale, margin)
    px = T @ np.array([cx, cy, 1.0])
    half_w, half_h = 160, 100
    x_a, y_a = int(px[0] - half_w), int(px[1] - half_h)
    canvas = np.zeros((2 * half_h, 2 * half_w, 3), np.uint8)
    xs, ys = max(x_a, 0), max(y_a, 0)
    xe, ye = min(x_a + 2 * half_w, ras.shape[1]), min(y_a + 2 * half_h, ras.shape[0])
    if xe > xs and ye > ys:
        canvas[ys - y_a:ye - y_a, xs - x_a:xe - x_a] = ras[ys:ye, xs:xe]
    canvas = cv2.resize(canvas, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    for n in ("TOUCHLINE_TOP", "TOUCHLINE_BOTTOM", "GOAL_LINE_L", "GOAL_LINE_R"):
        p = pitch.polylines()[n]
        q = ((T @ np.concatenate([p, np.ones((len(p), 1))], 1).T).T[:, :2] - np.array([x_a, y_a])) * 2.0
        cv2.polylines(canvas, [np.round(q).astype(np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
    return canvas


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_sheet(results: list[dict]) -> np.ndarray:
    rows = []
    for res in results:
        box = _edge_box(res)
        s, r0 = res["snap"], res["none"]
        cells = [
            _label(_edge_crop(res, "none", box, (0, 0, 255)),
                   f"{res['clip'][8:]}:{res['idx']} plain H (red) edge {r0['median_px']:.1f}px F1 {r0['f1']:.2f}"),
            _label(_edge_crop(res, "snap", box, (0, 255, 0)),
                   f"snapped k1={s['k1']:.3f} (green) edge {s['median_px']:.1f}px F1 {s['f1']:.2f}"),
            _label(_topdown_crop(res, "none", box), "top-down warp, plain H (magenta = pitch lines)"),
            _label(_topdown_crop(res, "snap", box), "top-down warp, snapped (lens removed)"),
        ]
        hh = max(c.shape[0] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, hh - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in cells]
        rows.append(np.concatenate(cells, axis=1))
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    return np.concatenate(rows, axis=0)


# --------------------------------------------------------------------------


def _fmt(x, nd=1):
    return "  -  " if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    frames = FRAMES[: a.limit] if a.limit else FRAMES
    results = []
    lines = ["| frame | n | edge px med/p90 none | rig | snap | F1 none | rig | snap | k1 snap | click shift px | snap |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for clip, idx in frames:
        t0 = time.time()
        try:
            r = evaluate_frame(clip, idx)
        except (OSError, KeyError, ValueError) as exc:   # bad sectors / missing frame: skip
            print(f"skip {clip}:{idx}: {exc}", flush=True)
            continue
        if r is None:
            print(f"skip {clip}:{idx}: no frame", flush=True)
            continue
        results.append(r)
        n, g, s = r["none"], r.get("rig"), r["snap"]
        lines.append(
            f"| {clip[8:]}:{idx} | {r['n']} | {_fmt(n['median_px'])}/{_fmt(n['p90_px'])} | "
            f"{_fmt(g and g['median_px'])}/{_fmt(g and g['p90_px'])} | {_fmt(s['median_px'])}/{_fmt(s['p90_px'])} | "
            f"{n['f1']:.3f} | {_fmt(g and g['f1'], 3)} | {s['f1']:.3f} | {s['k1']:.4f} | "
            f"{_fmt(s['shift'])} | {'ok' if s['accepted'] else 'refused'} |"
        )
        print(lines[-1], f"({time.time() - t0:.0f}s)", flush=True)
    for ground, pre in (("emirates", ("arsenal_mancity", "arsenal_astonvilla")), ("vitality", ("arsenal_bournemouth",))):
        rs = [r for r in results if r["clip"].startswith(pre)]
        if not rs:
            continue

        def med(tag, key, rs=rs):
            v = [r[tag][key] for r in rs if tag in r and np.isfinite(r[tag][key])]
            return statistics.median(v) if v else float("nan")

        acc = [r for r in rs if r["snap"]["accepted"]]
        lines.append(
            f"\n{ground} ({len(rs)} frames, snap accepted {len(acc)}): median edge residual px none {med('none', 'median_px'):.2f}"
            f" / rig {med('rig', 'median_px'):.2f} / snap {med('snap', 'median_px'):.2f}; median F1 none "
            f"{med('none', 'f1'):.3f} / rig {med('rig', 'f1'):.3f} / snap {med('snap', 'f1'):.3f}; "
            f"median snapped k1 {statistics.median([r['snap']['k1'] for r in acc]) if acc else float('nan'):.4f}"
        )
    (a.out / "table.md").write_text("\n".join(lines) + "\n")
    sheet = [r for r in results if (r["clip"], r["idx"]) in SHEET and r["snap"]["accepted"]]
    if sheet:
        cv2.imwrite(str(a.out / "sheet.jpg"), render_sheet(sheet), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
