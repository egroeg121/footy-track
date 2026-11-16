"""Scoring of seed-propagated ball tracks against human GT (mini CPU side, numpy only).

A *sample* = (clip, seed, GT frame g != seed): the prediction of the seed's forward track (g > seed) or backward track (g < seed) at g.
Odd original frames of 50 fps clips (spec ``gt_odd``) are scored against the mean of the two neighbouring processed frames.  Errors are in
1080p-equivalent px (the clip's native px * 1080/h), as in ~/sam2-exp score.py.  Distances are in processed frames (25 fps equivalent).

    hit      = present (and score > gate) and error <= 10 px          (recall@10 numerator; absent counts as a miss)
    coverage = fraction of GT-ball samples where the tracker emits a position
    precision-on-no-ball = fraction of GT ``no_ball`` frames (in the seed window) where the tracker emits nothing
    first loss = distance to the first GT frame of a (seed, direction) track that is a miss; censored at the last GT frame when no miss.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

BINS = [(0, 25), (25, 100), (100, 250), (250, 10_000)]
BIN_NAMES = ["0-25", "25-100", "100-250", "250+"]
HIT_PX = 10.0


def load_tracks(track_dir: str | Path, tag: str, protocol: str = "oracle") -> dict[str, dict]:
    out = {}
    for p in sorted(Path(track_dir).glob(f"tracks__*__{tag}__{protocol}.json")):
        d = json.loads(p.read_text())
        out[d["meta"]["clip"]] = d
    return out


def _pred(track: dict | None, p: float):
    """Prediction at processed frame p (half-integer p = mean of the two neighbours). -> (present, x, y, score) or None if outside the track."""
    if track is None:
        return None
    fr = track["frames"]
    if abs(p - round(p)) < 1e-6:
        r = fr.get(str(int(round(p))))
        return None if r is None else (bool(r[0]), r[1], r[2], r[4])
    a, b = fr.get(str(int(math.floor(p)))), fr.get(str(int(math.ceil(p))))
    if a is None or b is None:
        return None
    if not (a[0] and b[0]):
        return (False, None, None, min(a[4], b[4]))
    return (True, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2, min(a[4], b[4]))


def samples(spec: dict, docs: dict[str, dict], clips: list[str] | None = None, max_reach: int | None = None) -> list[dict]:
    """All (seed, GT frame) samples, ball and no_ball.  Rows: clip, seed, dir, t, g, dist, odd, kind, present, score, err, gx, gy, x, y."""
    rows = []
    mr = max_reach or spec.get("max_reach", 400)
    for name, doc in docs.items():
        if clips is not None and name not in clips:
            continue
        clip = spec["clips"][name]
        sc = 1080.0 / clip["h"]
        tr = {(t["seed"], t["dir"]): t for t in doc["tracks"]}
        seeds = sorted({t["seed"] for t in doc["tracks"]})
        gts = [(float(k), v, False, "ball") for k, v in clip["gt"].items()] + [(int(k) + 0.5, v, True, "ball") for k, v in clip.get("gt_odd", {}).items()]
        gts += [(float(f), None, False, "noball") for f in clip["no_ball"]]
        for sid in seeds:
            t = next(x["t"] for x in doc["tracks"] if x["seed"] == sid)
            for g, box, odd, kind in gts:
                if g == t or abs(g - t) > mr:
                    continue
                d = "fwd" if g > t else "bwd"
                pr = _pred(tr.get((sid, d)), g)
                if pr is None:
                    continue
                present, x, y, score = pr
                row = {"clip": name, "seed": sid, "dir": d, "t": t, "g": g, "dist": abs(g - t), "odd": odd, "kind": kind, "present": present, "score": score,
                       "x": x, "y": y, "sc": sc}
                if kind == "ball":
                    gx, gy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    row.update(gx=gx, gy=gy, err=(math.hypot(x - gx, y - gy) * sc) if present else None)
                rows.append(row)
    return rows


def bin_of(d: float) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= d < hi:
            return i
    return len(BINS) - 1


def hits(rows: list[dict], gate: float | None = None) -> np.ndarray:
    """hit indicator per ball row (gate: additionally requires score > gate)."""
    return np.array([bool(r["present"] and r["err"] is not None and r["err"] <= HIT_PX and (gate is None or r["score"] > gate)) for r in rows], dtype=float)


def boot_ci(vals: np.ndarray, clusters: np.ndarray | None = None, n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """mean and 95% percentile CI; frame-level bootstrap (clusters None) or clip-level cluster bootstrap."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (float("nan"),) * 3
    if clusters is None:
        idx = rng.integers(0, len(vals), size=(n, len(vals)))
        m = vals[idx].mean(1)
    else:
        u = np.unique(clusters)
        sums = np.array([vals[clusters == c].sum() for c in u])
        cnt = np.array([(clusters == c).sum() for c in u], dtype=float)
        idx = rng.integers(0, len(u), size=(n, len(u)))
        m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_diff(a: np.ndarray, b: np.ndarray, clusters: np.ndarray | None = None, n: int = 4000) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    f = boot_ci(d, None, n)
    out = {"diff": f[0], "ci_frame": [f[1], f[2]]}
    if clusters is not None:
        c = boot_ci(d, clusters, n)
        out["ci_clip"] = [c[1], c[2]]
    return out


def summarize(rows: list[dict], gate: float | None = None, n_boot: int = 2000) -> dict:
    ball = [r for r in rows if r["kind"] == "ball"]
    nob = [r for r in rows if r["kind"] == "noball"]
    h = hits(ball, gate)
    pres = np.array([bool(r["present"] and (gate is None or r["score"] > gate)) for r in ball], dtype=float)
    cl = np.array([r["clip"] for r in ball])
    out = {"n_samples": len(ball), "n_clips": int(len(set(cl))), "n_seeds": len({(r["clip"], r["seed"]) for r in ball})}
    m, lo, hi = boot_ci(h, None, n_boot)
    out["recall10"] = {"mean": m, "ci_frame": [lo, hi]}
    if len(set(cl)) > 1:
        out["recall10"]["ci_clip"] = list(boot_ci(h, cl, n_boot)[1:])
    out["coverage"] = float(pres.mean()) if len(pres) else None
    # precision of what is emitted (frame-level)
    out["precision_emitted"] = float(h.sum() / pres.sum()) if pres.sum() else None
    out["by_dist"] = {}
    for i, name in enumerate(BIN_NAMES):
        idx = np.array([bin_of(r["dist"]) == i for r in ball])
        if idx.sum():
            mm, l, hh = boot_ci(h[idx], None, n_boot)
            out["by_dist"][name] = {"n": int(idx.sum()), "recall10": mm, "ci": [l, hh], "coverage": float(pres[idx].mean()), "clips": int(len(set(cl[idx])))}
        else:
            out["by_dist"][name] = {"n": 0}
    # miss anatomy
    miss = [r for r in ball if not (r["present"] and r["err"] is not None and r["err"] <= HIT_PX and (gate is None or r["score"] > gate))]
    if miss:
        absent = sum(1 for r in miss if not (r["present"] and (gate is None or r["score"] > gate)))
        near = sum(1 for r in miss if r["present"] and (gate is None or r["score"] > gate) and r["err"] is not None and r["err"] <= 50)
        out["miss_anatomy"] = {"n": len(miss), "absent": absent / len(miss), "drift_10_50": near / len(miss), "wrong_object_gt50": 1 - (absent + near) / len(miss)}
    # no-ball frames
    if nob:
        emit = [bool(r["present"] and (gate is None or r["score"] > gate)) for r in nob]
        out["no_ball"] = {"n": len(nob), "silent_rate": 1 - float(np.mean(emit)), "false_emit_rate": float(np.mean(emit)),
                          "by_dist_n": {BIN_NAMES[bin_of(r["dist"])]: 0 for r in nob}}
        out["no_ball"].pop("by_dist_n")
    out["first_loss"] = first_loss(ball, gate)
    return out


def first_loss(ball: list[dict], gate: float | None = None) -> dict:
    """Per (clip, seed, dir): distance of the first miss, censored at the last GT frame. Mean / median over tracks that have >= 1 GT frame."""
    by: dict[tuple, list] = {}
    for r in ball:
        by.setdefault((r["clip"], r["seed"], r["dir"]), []).append(r)
    vals, cens = [], []
    for k, rs in by.items():
        rs = sorted(rs, key=lambda r: r["dist"])
        loss = next((r["dist"] for r in rs if not (r["present"] and r["err"] is not None and r["err"] <= HIT_PX and (gate is None or r["score"] > gate))), None)
        vals.append(loss if loss is not None else rs[-1]["dist"])
        cens.append(loss is None)
    if not vals:
        return {"n_tracks": 0}
    return {"n_tracks": len(vals), "mean": float(np.mean(vals)), "median": float(np.median(vals)), "censored_frac": float(np.mean(cens))}


def by_clip(rows: list[dict], gate: float | None = None) -> dict:
    out = {}
    for c in sorted({r["clip"] for r in rows}):
        rs = [r for r in rows if r["clip"] == c and r["kind"] == "ball"]
        if rs:
            h = hits(rs, gate)
            out[c] = {"n": len(rs), "recall10": float(h.mean())}
    return out


# ---------------------------------------------------------------------------------------------- gates
def key_of(r: dict) -> tuple:
    return (r["clip"], r["seed"], r["g"])


def gate_curve(rows: list[dict], thresholds=(-9, -3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14)) -> list[dict]:
    """Coverage / precision-of-emitted / recall as the SAM score-logit gate is raised (own presence head, no other information)."""
    ball = [r for r in rows if r["kind"] == "ball"]
    out = []
    for th in thresholds:
        h = hits(ball, th)
        emit = np.array([bool(r["present"] and r["score"] > th) for r in ball], dtype=float)
        out.append({"gate": th, "coverage": float(emit.mean()) if len(emit) else None, "precision": float(h.sum() / emit.sum()) if emit.sum() else None,
                    "recall10": float(h.mean()) if len(h) else None})
    return out


def agree_rows(rows_a: list[dict], rows_b: list[dict], px: float = 5.0) -> list[dict]:
    """Two configs (same seeds / frames) must agree within ``px`` (1080p-equivalent) for the frame to be emitted; emitted position = config A's.
    Returned rows are A's rows with ``present`` cleared where the gate fails (so summarize() reports coverage / precision of the gated output)."""
    b = {key_of(r): r for r in rows_b}
    out = []
    for r in rows_a:
        r2 = dict(r)
        o = b.get(key_of(r))
        if r["kind"] == "ball" and r["present"]:
            ok = o is not None and o["present"] and math.hypot(o["x"] - r["x"], o["y"] - r["y"]) * r["sc"] <= px
            r2["present"] = bool(ok)
            if not ok:
                r2["err"] = None
        out.append(r2)
    return out


def s1_rows(spec: dict, docs: dict[str, dict], px: float = 5.0, max_reach: int | None = None) -> list[dict]:
    """Round-3 S1 gate from seed pairs: for a GT frame g between two seeds a < g < b the forward track of a and the backward track of b must agree within
    ``px``.  Returns one row per (clip, a, b, g) with present = gate passed (position = mean of both); also ``nearest`` = the ungated prediction of the
    nearer seed for the same frames (the single-seed baseline on identical frames)."""
    mr = max_reach or spec.get("max_reach", 400)
    rows = []
    for name, doc in docs.items():
        clip = spec["clips"][name]
        sc = 1080.0 / clip["h"]
        tr = {(t["seed"], t["dir"]): t for t in doc["tracks"]}
        seeds = sorted({(t["t"], t["seed"]) for t in doc["tracks"]})
        gts = [(float(k), v) for k, v in clip["gt"].items()] + [(int(k) + 0.5, v) for k, v in clip.get("gt_odd", {}).items()]
        for (ta, sa), (tb, sb) in zip(seeds, seeds[1:]):
            for g, box in gts:
                if not (ta < g < tb) or g - ta > mr or tb - g > mr:
                    continue
                pa, pb = _pred(tr.get((sa, "fwd")), g), _pred(tr.get((sb, "bwd")), g)
                if pa is None or pb is None:
                    continue
                gx, gy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                both = pa[0] and pb[0]
                d = math.hypot(pa[1] - pb[1], pa[2] - pb[2]) * sc if both else None
                ok = both and d <= px
                near = pa if (g - ta) <= (tb - g) else pb
                x, y = ((pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2) if ok else (None, None)
                rows.append({"clip": name, "a": sa, "b": sb, "g": g, "gate": bool(ok), "err": (math.hypot(x - gx, y - gy) * sc) if ok else None, "d_ab": d,
                             "near_err": (math.hypot(near[1] - gx, near[2] - gy) * sc) if near[0] else None, "min_dist": min(g - ta, tb - g)})
    return rows


def s1_summary(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    em = [r for r in rows if r["gate"]]
    hit_em = [r for r in em if r["err"] <= HIT_PX]
    near_hit = [r for r in rows if r["near_err"] is not None and r["near_err"] <= HIT_PX]
    return {"n_gt_frames_between_seeds": n, "gate_pass_rate": len(em) / n, "precision_of_emitted": (len(hit_em) / len(em)) if em else None,
            "recall10_gated": len(hit_em) / n, "recall10_nearest_seed_ungated": len(near_hit) / n}
