# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408, B007, SIM105
"""Candidate-level cross-check of trackers against Ball Check verdicts.

Ball Check (``<gt_dir>/ball_checks/*.jsonl``) holds human yes/no verdicts on single
detector candidates.  Unlike the GT marks these include confirmed *non*-balls,
so they bound how often a tracker emits a point a human said is not the ball,
independently of which frames happen to carry GT.  Verdicts are sampled from the
detector's own candidates (borderline-heavy), so numbers are only comparable
between trackers, not absolute.  Frames within 5 of a detector-training image are
skipped.  The verdict files are not part of the GT snapshot hash; the report
records their fingerprint instead.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from footy_track.ball_offline import score as S
from footy_track.ball_offline.data import contaminated_frames, roboflow_frames
from footy_track.ball_offline.snapshot import resolve_snapshot
from footy_track.clip_id import canonical_clip_id

W_PX, H_PX = 1920.0, 1080.0


def load_verdicts(directory: Path) -> tuple[dict, str]:
    """``{clip: [(frame, (x, y) px, "ball" | "not_ball")]}`` (latest verdict per candidate
    wins; ``corrected`` counts as ball at the corrected box; unsure / tombstones dropped)
    and a fingerprint of the files."""
    latest = {}
    h = hashlib.sha256()
    for p in sorted(directory.glob("*.jsonl")):
        h.update(f"{p.name}:{p.stat().st_size};".encode())
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            latest[(canonical_clip_id(r["clip"]), r["frame_index"], r["box_index"])] = r
    out = defaultdict(list)
    for (cid, t, _), r in latest.items():
        v = r["verdict"]
        if v == "ball" and r.get("bbox"):
            b = r["bbox"]
        elif v == "corrected" and r.get("corrected_bbox"):
            b, v = r["corrected_bbox"], "ball"
        elif v == "not_ball" and r.get("bbox"):
            b = r["bbox"]
        else:
            continue
        out[cid].append(
            (t, ((b["x"] + b["w"] / 2) * W_PX, (b["y"] + b["h"] / 2) * H_PX), v)
        )
    return out, h.hexdigest()[:16]


def score_verdicts(
    verdicts: dict, preds: dict[str, dict], skip: dict[str, set[int]], clips=None
) -> dict:
    a = dict(pos=0, pos_hit=0, neg=0, neg_hit=0, on_pos=0, on_any=0)
    used = 0
    for cid, items in sorted(verdicts.items()):
        if clips is not None and cid not in clips:
            continue
        items = [it for it in items if it[0] not in skip.get(cid, ())]
        if not items:
            continue
        used += 1
        pred = preds.get(cid, {})
        for t, (x, y), v in items:
            pr = pred.get(t)
            hit = pr is not None and np.hypot(pr[0] - x, pr[1] - y) <= 10
            if v == "ball":
                a["pos"] += 1
                a["pos_hit"] += int(hit)
            else:
                a["neg"] += 1
                a["neg_hit"] += int(hit)
        by_t = defaultdict(list)
        for t, xy, v in items:
            by_t[t].append((xy, v))
        for t, lst in by_t.items():
            pr = pred.get(t)
            if pr is None:
                continue
            near = [v for xy, v in lst if np.hypot(pr[0] - xy[0], pr[1] - xy[1]) <= 10]
            if near:
                a["on_any"] += 1
                a["on_pos"] += int("ball" in near)
    a["clips_used"] = used
    a["recall_on_ball_verdicts"] = a["pos_hit"] / a["pos"] if a["pos"] else None
    a["emission_rate_on_not_ball_verdicts"] = (
        a["neg_hit"] / a["neg"] if a["neg"] else None
    )
    a["precision_of_emissions_landing_on_a_verdict"] = (
        a["on_pos"] / a["on_any"] if a["on_any"] else None
    )
    return a


def run_verdicts(args) -> int:
    from footy_track.ball_offline.cli import spec_for  # noqa: PLC0415

    snap = resolve_snapshot(args.snapshot)
    spec = spec_for(snap, args)
    verdicts, fp = load_verdicts(spec.verdicts_dir)
    rf = roboflow_frames()
    skip = {cid: contaminated_frames(cid, rf) for cid in verdicts}
    out = {
        "snapshot": snap.hash,
        "eval_set": snap.eval_set,
        "verdicts_fingerprint": fp,
        "trackers": {},
    }
    print(
        f"Ball Check cross-check on {spec.verdicts_dir} (verdict files fp {fp}), snapshot {snap.label()}"
    )
    print(
        "| tracker | ball verdicts hit | recall | not_ball verdicts emitted | emission rate | emitted points on a verdict that are positive |"
    )
    print("|---|---|---|---|---|---|")
    for path in args.pred:
        lp = S.load_predictions(path)
        r = score_verdicts(verdicts, lp.preds, skip, clips=set(lp.preds))
        out["trackers"][path] = r

        def f(x):
            return "n/a" if x is None else f"{x:.3f}"

        print(
            f"| {Path(path).stem} | {r['pos_hit']}/{r['pos']} | {f(r['recall_on_ball_verdicts'])} | "
            f"{r['neg_hit']}/{r['neg']} | {f(r['emission_rate_on_not_ball_verdicts'])} | "
            f"{r['on_pos']}/{r['on_any']} = {f(r['precision_of_emissions_landing_on_a_verdict'])} |"
        )
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=1, default=float))
    return 0
