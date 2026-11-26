"""Per-segment diagnostic and run report of the ``calibrate-match`` flow (design ``calibration_match_flow.md`` 2.2 (d), 3).

No per-segment rig fit. For every segment of a match, after the fixed-rig solves:

* the **fixed-rig cost**: the data cost of the segment's one rig-search frame with its pose re-optimised at the shared rig
  (``rig_search.json["unit_costs"]``; only segments that gave a confident anchor frame have one);
* the **calibrated / accepted fraction** and the anchors from the run's ``meta.json``, the camera variant a recovery pass
  chose (``rig_variant``) and whether the recovery result replaced the standard one.

Flags (label-free, thresholds provisional until the eval of design section 3 exists; they are recorded in the output):

* ``missing``: no finished result dir;
* ``uncalibrated``: < :data:`UNCALIBRATED_BELOW` of the frames calibrated (nothing usable: H = NaN, queue for the editor);
* ``other_camera``: a recovery pass moved the camera to a side position (``rig_variant`` set and used): the segment is a
  different camera position, flagged and kept apart from the shared rig, never forced onto it;
* ``recovered``: the recovery pass ran and its result replaced the standard one (same-camera or variant);
* ``low_accept``: calibrated but < :data:`LOW_ACCEPT_BELOW` of the frames accepted (F1 >= 0.5 on both sides);
* ``high_cost``: the fixed-rig cost is a robust outlier among the segments' costs (modified z > :data:`COST_Z`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

UNCALIBRATED_BELOW = 0.05
LOW_ACCEPT_BELOW = 0.5
COST_Z = 3.5


def modified_z(x: np.ndarray) -> np.ndarray:
    """Iglewicz-Hoaglin modified z-score (0.6745 (x - median) / MAD); zeros when the MAD is 0."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return np.zeros_like(x) if not mad else 0.6745 * (x - med) / mad


def flag_row(row: dict, cost_z: float | None) -> list[str]:
    flags = []
    if row.get("status") == "missing":
        return ["missing"]
    if (row.get("calibrated_frac") or 0.0) < UNCALIBRATED_BELOW:
        flags.append("uncalibrated")
    elif (row.get("accepted_frac") or 0.0) < LOW_ACCEPT_BELOW:
        flags.append("low_accept")
    if row.get("recovery_used"):
        flags.append("recovered")
        if row.get("rig_variant") and not str(row["rig_variant"]).startswith("centre"):
            flags.append("other_camera")
    if cost_z is not None and cost_z > COST_Z:
        flags.append("high_cost")
    return flags


def segment_diagnostics(segments: list[dict], temporal: Path, match: str, tag: str,
                        unit_costs: list[dict] | None = None) -> list[dict]:
    """One row per segment of ``segments.json`` (list of its segment dicts), read from ``<temporal>/<match>_segNNN@<tag>/``."""
    cost = {int(u["segment"]): float(u["cost"]) for u in (unit_costs or [])}
    z_of: dict[int, float] = {}
    if len(cost) >= 5:
        ks = sorted(cost)
        z_of = dict(zip(ks, modified_z(np.array([cost[k] for k in ks])).tolist(), strict=True))
    rows = []
    for seg in sorted(segments, key=lambda s: int(s["index"])):
        i = int(seg["index"])
        d = temporal / f"{match}_seg{i:03d}@{tag}"
        row: dict = {"segment": i, "frame_count": int(seg["frame_count"]), "fixed_rig_cost": cost.get(i),
                     "cost_z": z_of.get(i)}
        mp = d / "meta.json"
        if not (d / "calib_full.npz").is_file() or not mp.is_file():
            row.update(status="missing")
        else:
            m = json.loads(mp.read_text())
            rec = m.get("recovery") or {}
            row.update(status="done", frames=m.get("frames_decoded"), calibrated_frac=round(float(m["calibrated_frac"]), 4),
                       accepted_frac=round(float(m["accepted_frac"]), 4), median_evidence_f1=m.get("median_evidence_f1"),
                       anchors=f"{m.get('anchors_ok')}/{m.get('anchors_tried')}", rig_variant=m.get("rig_variant"),
                       recovery_ran=bool(rec), recovery_used=bool(rec.get("used")), cpu_s=m.get("cpu_s"),
                       code_sha=(m.get("code_sha") or "")[:12])
        row["flags"] = flag_row(row, row["cost_z"])
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    flags: dict[str, list[int]] = {}
    for r in rows:
        for f in r["flags"]:
            flags.setdefault(f, []).append(r["segment"])
    done = [r for r in rows if r["status"] == "done"]
    frames = sum(r["frames"] or 0 for r in done)
    return {
        "segments": len(rows), "done": len(done), "missing": len(rows) - len(done), "frames": frames,
        "calibrated_frac": round(sum((r["frames"] or 0) * r["calibrated_frac"] for r in done) / frames, 4) if frames else 0.0,
        "accepted_frac": round(sum((r["frames"] or 0) * r["accepted_frac"] for r in done) / frames, 4) if frames else 0.0,
        "cpu_h": round(sum(r.get("cpu_s") or 0.0 for r in done) / 3600, 3),
        "flags": {k: {"n": len(v), "segments": v} for k, v in sorted(flags.items())},
        "thresholds": {"uncalibrated_below": UNCALIBRATED_BELOW, "low_accept_below": LOW_ACCEPT_BELOW, "cost_modified_z": COST_Z},
    }


def render_markdown(report: dict) -> str:
    rs = report.get("rig_search") or {}
    rig = rs.get("rig") or {}
    L = [f"# calibrate-match: {report['match']} ({report['run_id']})", ""]
    L.append(f"- code: `{report.get('code_sha')}`; rig tag `{report.get('rig_tag')}`; contaminated: **{report.get('contaminated')}**")
    if rig:
        L.append(f"- rig: C = {rig['C']}, roll = {rig['roll_deg']:.3f} deg, k1 = {rig['k1']:.4f}; cost {rs.get('cost'):.0f} "
                 f"over {rs.get('n_units')} units; finalist spread {rs.get('finalist_spread_m'):.2f} m")
    for ind in rs.get("indirect_inputs") or []:
        L.append(f"- indirect input `{ind['name']}` ({ind['file']}): {ind['note']}")
    cpu = report.get("cpu_h") or {}
    L.append(f"- CPU-h: {cpu}")
    s = report["segments_summary"]
    L += ["", "## Segments", "",
          f"{s['done']}/{s['segments']} done, {s['frames']} frames; calibrated {100 * s['calibrated_frac']:.1f}%, "
          f"accepted {100 * s['accepted_frac']:.1f}%.", ""]
    for k, v in s["flags"].items():
        L.append(f"- `{k}` ({v['n']}): {v['segments']}")
    cb = report.get("combine")
    if cb:
        t = cb["totals"]
        L += ["", "## Combine", "", f"`{cb['version_id']}` ({cb['status']}): {t['frames']} frames, "
              f"{t['frames_accepted']} accepted ({100 * t['accepted_frac']:.1f}%), {t['frames_calibrated']} calibrated "
              f"({100 * t['calibrated_frac']:.1f}%), {t['segments_recovered']} segments recovered, "
              f"{t['segments_missing']} missing."]
    pr = report.get("proposals")
    if pr:
        L += ["", "## Proposals", "", f"{pr['run_id']}: picked {pr['n_picked']}, wrote {pr['n_written']} in {pr['n_clips']} clips."]
        for name, d in pr["design"].items():
            if not name.startswith("_"):
                L.append(f"- {name}: pool {d['pool_frames']}, expected {d['expected']}, picked {d['picked']}")
    L += ["", "## Segment table", "", "| seg | frames | calibrated | accepted | cost | variant | flags |", "|---|---|---|---|---|---|---|"]
    for r in report["segments"]:
        L.append(f"| {r['segment']:03d} | {r['frame_count']} | {r.get('calibrated_frac', '')} | {r.get('accepted_frac', '')} | "
                 f"{'' if r['fixed_rig_cost'] is None else round(r['fixed_rig_cost'], 1)} | {r.get('rig_variant') or ''} | "
                 f"{', '.join(r['flags'])} |")
    return "\n".join(L) + "\n"
