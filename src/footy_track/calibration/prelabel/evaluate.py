"""Compare a proposals run against human calibration GT on the same frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration.prelabel.proposals import list_clips_with_proposals, read_proposals
from footy_track.clip_id import canonical_clip_id

SUCCESS_PX_STRICT = 5.0
LOC_TOL_PX = 8.0  # two labels "at the same image location"
SUCCESS_PX = 10.0  # frame success: mean reprojection error of the GT points


def read_gt(gt_dir: Path, clip: str) -> dict:
    p = Path(gt_dir) / "calibration_gt" / f"{canonical_clip_id(clip)}.json"
    return json.loads(p.read_text()).get("frames", {}) if p.exists() else {}


def gt_px_errors(H: np.ndarray, gt: dict) -> np.ndarray:
    """Per-GT-point reprojection error in pixels of the GT image (NaN where a point does not project)."""
    iw, ih = gt.get("image_size") or (1920, 1080)
    uv = np.array([c["image"] for c in gt["correspondences"]], float)
    xy = np.array([c["pitch_xy"] for c in gt["correspondences"]], float)
    back, w = geo.apply_homography(np.linalg.inv(np.asarray(H, float)), xy)
    px = np.linalg.norm((back - uv) * np.array([iw, ih], float), axis=1)
    px[(w <= geo.W_EPS) | ~np.isfinite(px)] = np.nan
    return px


def eval_frame(gt: dict, prop: dict, loc_tol_px: float = LOC_TOL_PX) -> dict:
    iw, ih = gt.get("image_size") or prop.get("image_size") or (1920, 1080)
    size = np.array([iw, ih], float)
    out = {"status": prop["status"], "score": prop.get("score"), "failure": prop.get("failure"),
           "n_gt_points": len(gt["correspondences"]), "n_prop_points": len(prop.get("correspondences", []))}
    pc = prop.get("correspondences", [])
    inl = prop.get("inliers") or [True] * len(pc)
    # keypoint agreement where both labelled the same image location
    n_loc = n_agree = 0
    for g in gt["correspondences"]:
        for p, ok in zip(pc, inl, strict=False):
            if not ok:
                continue
            d = np.linalg.norm((np.array(g["image"]) - np.array(p.get("image_label") or p["image"])) * size)
            if d <= loc_tol_px:
                n_loc += 1
                n_agree += int(g.get("pitch_id") == p.get("pitch_id"))
                break
    out.update(n_same_location=n_loc, n_agree=n_agree,
               agreement=(n_agree / n_loc) if n_loc else None)
    out.update(mean_px=None, median_px=None, mean_m=None, pre_mean_px=None, refine_applied=None,
               dt_residual=prop.get("dt_residual"))
    out["refine_applied"] = (prop.get("refine") or {}).get("applied")
    if prop.get("H") is not None and prop["status"] != "failed":
        H = np.array(prop["H"], float)
        px = gt_px_errors(H, gt)
        xy = np.array([c["pitch_xy"] for c in gt["correspondences"]], float)
        uv = np.array([c["image"] for c in gt["correspondences"]], float)
        fwd, _ = geo.apply_homography(H, uv)
        m = np.linalg.norm(fwd - xy, axis=1)
        if np.isfinite(px).any():
            out.update(mean_px=float(np.nanmean(px)), median_px=float(np.nanmedian(px)))
        if np.isfinite(m).any():
            out["mean_m"] = float(np.nanmean(m))
        pre = prop.get("pre_refine")
        if pre and pre.get("H") is not None:
            ppx = gt_px_errors(np.array(pre["H"], float), gt)
            if np.isfinite(ppx).any():
                out["pre_mean_px"] = float(np.nanmean(ppx))
    return out


def evaluate(gt_dir: Path, run_id: str | None = None, success_px: float = SUCCESS_PX) -> dict:
    """One row per (clip, frame) that has both GT and a proposal in the run."""
    rows = []
    runs_seen = set()
    for clip in list_clips_with_proposals(gt_dir):
        gts = read_gt(gt_dir, clip)
        runs = read_proposals(gt_dir, clip)["runs"]
        for rid, run in runs.items():
            if run_id and rid != run_id:
                continue
            for k, prop in run["frames"].items():
                if k in gts:
                    runs_seen.add(rid)
                    rows.append({"clip": clip, "frame": int(k), "run_id": rid,
                                 **eval_frame(gts[k], prop)})
    rows.sort(key=lambda r: (r["run_id"], r["clip"], r["frame"]))
    ok = [r for r in rows if r["mean_px"] is not None and r["mean_px"] <= success_px]
    mp = [r["mean_px"] for r in rows if r["mean_px"] is not None]
    agree_n = sum(r["n_agree"] for r in rows)
    agree_d = sum(r["n_same_location"] for r in rows)
    summary = {
        "runs": sorted(runs_seen), "n_frames": len(rows), "success_px": success_px,
        "n_success": len(ok), "success_rate": len(ok) / len(rows) if rows else None,
        "n_failed": sum(r["status"] == "failed" for r in rows),
        "mean_px_over_frames": float(np.mean(mp)) if mp else None,
        "keypoint_agreement": agree_n / agree_d if agree_d else None,
        "n_same_location": agree_d,
    }
    return {"summary": summary, "frames": rows}


def format_table(res: dict) -> str:
    def f(v, spec):
        return "-" if v is None else format(v, spec)

    head = (f"{'clip':26s} {'frame':>5s} {'status':9s} {'score':>5s} {'gt':>3s} {'pr':>3s} {'loc':>3s} {'agr':>5s} "
            f"{'pre px':>7s} {'px':>7s} {'m':>6s}")
    lines = [head, "-" * len(head)]
    for r in res["frames"]:
        lines.append(
            f"{r['clip'][:26]:26s} {r['frame']:5d} {r['status']:9s} {f(r['score'], '5.2f'):>5s} "
            f"{r['n_gt_points']:3d} {r['n_prop_points']:3d} {r['n_same_location']:3d} "
            f"{f(r['agreement'], '5.2f'):>5s} {f(r.get('pre_mean_px'), '7.1f'):>7s} {f(r['mean_px'], '7.1f'):>7s} "
            f"{f(r['mean_m'], '6.2f'):>6s}"
        )
    s = res["summary"]
    lines.append("-" * len(head))
    lines.append(
        f"frames={s['n_frames']}  success(<= {s['success_px']:.0f}px)={s['n_success']}/{s['n_frames']}"
        f" ({f(s['success_rate'], '.0%')})  failed={s['n_failed']}  "
        f"keypoint agreement={f(s['keypoint_agreement'], '.0%')} over {s['n_same_location']} shared locations  "
        f"mean px={f(s['mean_px_over_frames'], '.1f')}"
    )
    return "\n".join(lines)
