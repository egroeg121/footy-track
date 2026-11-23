"""Per-segment diagnostic and run report of one ``calibrate-match`` run (see ``calibration.match_diagnostic``).

    PYTHONPATH=src python -m footy_track.scripts.calibration_match_report diagnose --match M --temporal DIR --tag STEM-HASH8 \\
        [--rig-search rig_search.json] --out diagnostic.json
    PYTHONPATH=src python -m footy_track.scripts.calibration_match_report report --match M --run-id ID --diagnostic diagnostic.json \\
        [--rig-search ...] [--combine-manifest ...] [--proposals-json ...] [--cpu-json ...] --out-dir DIR

``diagnose`` reads the per-segment result dirs and writes the rows + summary; ``report`` assembles ``<run-id>.json`` and
``<run-id>.md`` from it and the other stage outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from footy_track.calibration import match_calibration as mc
from footy_track.calibration import match_diagnostic as md


def _load(p: Path | None):
    return json.loads(p.read_text()) if p else None


def _rig_summary(rs: dict | None) -> dict | None:
    if not rs:
        return None
    keep = ("status", "contaminated", "n_units", "n_anchored_ok", "cost", "finalist_spread_m", "rig_tag", "rig_hash", "cpu_s",
            "indirect_inputs", "reasons", "unit_cost_summary", "inputs")
    out = {k: rs[k] for k in keep if k in rs}
    if "C" in rs:
        out["rig"] = {"C": rs["C"], "roll_deg": rs["roll_deg"], "k1": rs["k1"]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--match", required=True)
    d.add_argument("--temporal", type=Path, required=True)
    d.add_argument("--tag", required=True, help="the '@' suffix of the segment dirs, <stem>-<hash8>")
    d.add_argument("--rig-search", type=Path, default=None)
    d.add_argument("--data-dir", type=Path, default=None)
    d.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--match", required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument("--diagnostic", type=Path, required=True)
    r.add_argument("--rig-search", type=Path, default=None)
    r.add_argument("--combine-manifest", type=Path, default=None)
    r.add_argument("--proposals-json", type=Path, default=None)
    r.add_argument("--cpu-json", type=Path, default=None, help="{'rig_search': h, 'segments': h, ...} CPU-hours per phase")
    r.add_argument("--code-sha", default=None)
    r.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.cmd == "diagnose":
        rs = _load(a.rig_search)
        segs = mc.read_segments_json(a.match, a.data_dir)["segments"]
        rows = md.segment_diagnostics(segs, a.temporal, a.match, a.tag, (rs or {}).get("unit_costs"))
        a.out.write_text(json.dumps({"match": a.match, "tag": a.tag, "segments": rows, "summary": md.summarise(rows)}, indent=1))
        s = md.summarise(rows)
        print(f"{a.match} @{a.tag}: {s['done']}/{s['segments']} done, calibrated {100 * s['calibrated_frac']:.1f}% "
              f"accepted {100 * s['accepted_frac']:.1f}%, cpu {s['cpu_h']} h; flags " + ", ".join(f"{k}={v['n']}" for k, v in s["flags"].items()))
        return 0

    diag = _load(a.diagnostic)
    rs = _load(a.rig_search)
    cb = _load(a.combine_manifest)
    cpu = _load(a.cpu_json) or {}
    cpu.setdefault("segments", diag["summary"]["cpu_h"])
    cpu["total"] = round(sum(v for k, v in cpu.items() if k != "total"), 3)
    report = {
        "match": a.match, "run_id": a.run_id, "rig_tag": diag["tag"], "code_sha": a.code_sha,
        "contaminated": bool((rs or {}).get("contaminated", False)), "cpu_h": cpu, "rig_search": _rig_summary(rs),
        "segments_summary": diag["summary"], "segments": diag["segments"],
        "combine": None if cb is None else {k: cb[k] for k in ("version_id", "status", "totals", "files", "warnings")},
        "proposals": _load(a.proposals_json),
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{a.run_id}.json").write_text(json.dumps(report, indent=1, default=float))
    (a.out_dir / f"{a.run_id}.md").write_text(md.render_markdown(report))
    print(f"wrote {a.out_dir / (a.run_id + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
