"""P3 scoring: per-frame H (calib_full.npz states + rig) of Man City segments against calibration GT, split by GT pitch depth.

    python scripts/ptz_p3/score_gt.py --label NAME DIR_TEMPLATE [DIR_TEMPLATE...]
DIR_TEMPLATE contains ``{i:03d}`` for the segment index, e.g. /mnt/.../arsenal_mancity_seg{i:03d}@emirates.
Error = |image_to_pitch(GT-visible 5 m grid point observed uv) - GT pitch point|, as ``ptz_temporal.err5``; frames with no
calibration (NaN state) count as unanswered. Depth thirds by GT pitch y (camera on the y<0 side): near < -11.33 <= mid <= 11.33 < far.
Light (48 GT frames): run with an ephemeral `uv run --no-project --with numpy --with scipy --with opencv-python-headless`, PYTHONPATH=src."""
from __future__ import annotations

import argparse
import json

import numpy as np

from footy_track.calibration import geometry as geo  # noqa: F401
from footy_track.calibration import ptz_temporal as pt
from footy_track.scripts import ptz_validate as pv

GT_SEGS = (0, 10, 20, 31, 50, 60, 80, 90, 100, 104, 110, 120, 140, 150)
THIRDS = ((-99, -34 / 3), (-34 / 3, 34 / 3), (34 / 3, 99))


def score_dir(d: str, idx: int, gt_frames, rig_override=None):
    from pathlib import Path
    z = Path(d) / "calib_full.npz"
    if not z.exists():
        return None
    npz = np.load(z)
    rig = npz["rig"]
    sh = pt.Shared(C=np.asarray(rig[:3], float), roll=float(rig[3]), k1=float(rig[4])) if hasattr(pt, "Shared") else None
    states = np.stack([npz["pan"], npz["tilt"], npz["logf"]], 1)
    out = []
    for g in gt_frames:
        P, uv, near = pv.gt_grid(g.truth(), step=5.0)
        rec = {"seg": idx, "frame": g.frame, "n_grid": int(len(P)), "answered": False}
        if len(P) < 6:
            rec["skip"] = "no grid"
            out.append(rec)
            continue
        if g.frame >= len(states) or not np.all(np.isfinite(states[g.frame])):
            out.append(rec)
            continue
        cam = pt.cam_from_state(states[g.frame], sh)
        xy, ok = cam.image_to_pitch(uv)
        e = np.linalg.norm(xy - P, axis=1)
        e = np.where(ok & np.isfinite(e), np.minimum(e, 100.0), 100.0)
        rec.update(answered=True, e=e.tolist(), y=P[:, 1].tolist(), med=float(np.median(e)))
        out.append(rec)
    return out


def summarise(recs):
    fr = [r for r in recs if "skip" not in r]
    ans = [r for r in fr if r["answered"]]
    res = {"frames": len(fr), "answered": len(ans)}
    if not ans:
        return res
    meds = np.array([r["med"] for r in ans])
    res["frame_median_m"] = {"median": float(np.median(meds)), "p90": float(np.percentile(meds, 90)), "p95": float(np.percentile(meds, 95))}
    E = np.concatenate([r["e"] for r in ans]); Y = np.concatenate([r["y"] for r in ans])
    res["points"] = int(len(E))
    res["within_pct"] = {f"{t}m": float(100 * np.mean(E <= t)) for t in (1, 2, 5)}
    res["pooled_m"] = {"median": float(np.median(E)), "p90": float(np.percentile(E, 90)), "p95": float(np.percentile(E, 95))}
    for name, (lo, hi) in zip(("near", "mid", "far"), THIRDS):
        m = (Y >= lo) & (Y < hi)
        if m.any():
            e = E[m]
            res[name] = {"n": int(m.sum()), "median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "p95": float(np.percentile(e, 95)),
                         "within_pct": {f"{t}m": float(100 * np.mean(e <= t)) for t in (1, 2, 5)}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--segs", default=",".join(map(str, GT_SEGS)))
    ap.add_argument("templates", nargs="+", help="dir template with {i:03d}; first existing wins per segment")
    a = ap.parse_args()
    gt = [f for f in pv.load_gt(("mancity",)) if f.clip.startswith("arsenal_mancity_seg")]
    recs = []
    for i in map(int, a.segs.split(",")):
        gf = [f for f in gt if f.clip == f"arsenal_mancity_seg{i:03d}"]
        for tpl in a.templates:
            r = score_dir(tpl.format(i=i), i, gf)
            if r is not None:
                recs += r
                break
    s = summarise(recs)
    print(json.dumps({"label": a.label, **s}, indent=1))
    json.dump({"label": a.label, "summary": s, "frames": recs}, open(a.out, "w"))


if __name__ == "__main__":
    main()
