"""Refit rejected/failed calibration proposals as pinhole cameras into a NEW run (no Claude calls).

Reuses the labels stored in each record. Originals are never modified; results go to run ``--run-id``
(default ``pinhole1``) with status ``proposed`` and ``source_run``/``source_status`` links. A non-pinhole
control refit is computed too and only written to ``--report`` (JSON), never to the sidecars.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration.frame_type import not_broadcast_frames
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.pipeline import find_video, gt_frames
from footy_track.calibration.prelabel.proposals import list_clips_with_proposals, read_proposals, update_proposals
from footy_track.calibration.prelabel.rerefine import labels_from_record, new_record
from footy_track.calibration.refine import decompose_G, model_samples
from footy_track.calibration.video import read_frame


def wait_for_load(limit: float = 10.0) -> None:
    while float(open("/proc/loadavg").read().split()[0]) > limit:
        time.sleep(30)


def candidates(gt: Path, pattern: str) -> list[tuple]:
    """(clip, idx, run_id, rec) for rejected/failed frames, newest run per frame, minus GT / not_broadcast."""
    out = []
    for clip in list_clips_with_proposals(gt):
        if not re.match(pattern, clip):
            continue
        doc = read_proposals(gt, clip)
        skip = gt_frames(gt, clip) | not_broadcast_frames(gt, clip)
        best: dict[int, tuple] = {}
        for rid, run in sorted(doc["runs"].items(), key=lambda kv: kv[1].get("created") or ""):
            for k, rec in run["frames"].items():
                if rec.get("status") in ("rejected", "failed") and int(k) not in skip:
                    best[int(k)] = (rid, rec)  # later runs overwrite earlier
        for idx, (rid, rec) in sorted(best.items()):
            out.append((clip, idx, rid, rec))
    return out


def metrics(rec: dict, size, pitch: PitchModel) -> dict:
    H = np.asarray(rec["H"], float)
    G = np.linalg.inv(H @ np.diag([1.0 / size[0], 1.0 / size[1], 1.0]))
    dec = decompose_G(G / np.linalg.norm(G), size)
    d = {"ratio": None, "cam": None, "f": None}
    if dec:
        d.update(ratio=dec[3], cam=[float(v) for v in dec[2]], f=dec[0])
    d["line_cost"] = (rec.get("refine") or {}).get("cost_after")
    d["k1"] = (rec.get("distortion") or {}).get("k1", 0.0)
    d["score"] = rec.get("score")
    d["refine_applied"] = (rec.get("refine") or {}).get("applied")
    return d


def mean_shift(a: dict, b: dict, size, pitch: PitchModel):
    pts = model_samples(pitch)

    def proj(r):
        xy, w = geo.pitch_to_image(np.asarray(r["H"], float), pts, r.get("distortion"))
        return xy * np.array(size), w

    xa, wa = proj(a)
    xb, wb = proj(b)
    ok = np.isfinite(xa).all(1) & np.isfinite(xb).all(1) & (wa > 1e-9) & (wb > 1e-9)
    for x in (xa, xb):
        ok &= (x[:, 0] >= 0) & (x[:, 0] < size[0]) & (x[:, 1] >= 0) & (x[:, 1] < size[1])
    return float(np.linalg.norm(xa[ok] - xb[ok], axis=1).mean()) if ok.any() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--clips-dir", required=True, type=Path, action="append")
    ap.add_argument("--pattern", default=r"arsenal_(bournemouth|astonvilla)_")
    ap.add_argument("--run-id", default="pinhole1")
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--write", action="store_true", help="write the new run into the sidecars")
    a = ap.parse_args()

    cands = candidates(a.gt_dir, a.pattern)
    print(f"{len(cands)} candidates", flush=True)
    report, new_recs = [], {}
    for n, (clip, idx, rid, old) in enumerate(cands):
        wait_for_load()
        row = {"clip": clip, "idx": idx, "source_run": rid, "source_status": old["status"]}
        labels, _ = labels_from_record(old)
        if not labels or (old.get("view") or {}).get("unusable"):
            row["skip"] = "no labels / unusable"
            report.append(row)
            continue
        video = next((v for d in a.clips_dir if (v := find_video(d, clip))), None)
        frame = read_frame(video, idx) if video else None
        if frame is None:
            row["skip"] = "video/frame unavailable"
            report.append(row)
            continue
        ih, iw = frame.shape[:2]
        pitch = PitchModel(*(old.get("pitch_dims") or (105.0, 68.0)))
        try:
            ctrl = new_record(old, frame, pitch, True, camera=None)
            pin = new_record(old, frame, pitch, True, camera=True)
        except Exception as e:  # noqa: BLE001
            row["skip"] = f"{type(e).__name__}: {e}"
            report.append(row)
            continue
        row["old"] = metrics(old, (iw, ih), pitch) if old.get("H") else None
        row["control"] = {"ok": ctrl["status"] == "proposed", "failure": ctrl.get("failure")}
        row["pinhole"] = {"ok": pin["status"] == "proposed", "failure": pin.get("failure")}
        if ctrl["status"] == "proposed":
            row["control"].update(metrics(ctrl, (iw, ih), pitch))
        if pin["status"] == "proposed":
            row["pinhole"].update(metrics(pin, (iw, ih), pitch))
            row["pinhole"]["refine_reason"] = (pin.get("refine") or {}).get("reason")
            if old.get("H"):
                row["shift_vs_old"] = mean_shift(old, pin, (iw, ih), pitch)
            if ctrl["status"] == "proposed":
                row["shift_vs_control"] = mean_shift(ctrl, pin, (iw, ih), pitch)
            pin.update(source_run=rid, source_status=old["status"], pinhole=True)
            pin["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            new_recs.setdefault(clip, {})[str(idx)] = pin
            row["new_H"], row["new_dist"] = pin["H"], pin.get("distortion")
        row["old_H"] = old.get("H")
        row["old_dist"] = old.get("distortion")
        row["ctrl_H"] = ctrl["H"] if ctrl["status"] == "proposed" else None
        row["ctrl_dist"] = ctrl.get("distortion")
        report.append(row)
        print(f"[{n + 1}/{len(cands)}] {clip}:{idx} pin={row['pinhole']['ok']} ctrl={row['control']['ok']}", flush=True)
        a.report.write_text(json.dumps(report, indent=1))
    a.report.write_text(json.dumps(report, indent=1))

    if a.write:
        created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        params = {"pinhole": True, "fit_k1": True, "source": "refit of rejected/failed labels, no Claude call"}
        for clip, frames in new_recs.items():
            def _store(doc, frames=frames):
                run = doc["runs"].setdefault(
                    a.run_id, {"source": "pinhole-refit", "created": created, "params": params, "frames": {}})
                for k, rec in frames.items():
                    run["frames"][k] = rec

            update_proposals(a.gt_dir, clip, _store)
        print(f"wrote {sum(len(f) for f in new_recs.values())} frames in {len(new_recs)} sidecars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
