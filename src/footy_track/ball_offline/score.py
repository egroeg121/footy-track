# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408, B007, SIM105
"""Score any tracker's per-frame ball positions against a frozen snapshot.

Tracker output format (jsonl, one row per frame with a ball)::

    {"clip": "arsenal_mancity_seg010", "frame_index": 412, "x": 1031.5, "y": 402.0}

* ``clip`` is any alias of the clip (``clip_id.canonical_clip_id`` is applied);
* ``x``/``y`` are pixels at 1920x1080 (``--units px``, the default).  Normalised
  coordinates are accepted either as ``x``/``y`` with ``--units norm`` or as the
  labeller's ``"center": {"x", "y"}`` (always normalised); ``bbox`` (normalised
  top-left xywh) is used if there is no centre;
* ``x: null`` or an absent frame means "no prediction"; optional ``conf`` breaks ties
  when a frame has several rows (highest wins; otherwise the first).

Frames the snapshot does not score are ignored.  A snapshot clip for which the
tracker emitted no rows at all is counted (every GT ball there is a miss) and
listed in ``warnings`` so a partial run cannot masquerade as a full one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from footy_track.ball_offline.metrics import Counts, Pred, score_clip, wilson
from footy_track.ball_offline.snapshot import Snapshot
from footy_track.clip_id import canonical_clip_id

W_PX = 1920.0
H_PX = 1080.0


@dataclasses.dataclass
class LoadedPreds:
    preds: dict[str, Pred]
    n_rows: int
    n_dropped_nan: int
    n_duplicate_frames: int
    sha256: str
    path: str


def load_predictions(path: Path | str, units: str = "px") -> LoadedPreds:
    path = Path(path)
    h = hashlib.sha256()
    best: dict[tuple[str, int], tuple[float, tuple[float, float]]] = {}
    n_rows = n_nan = n_dup = 0
    with path.open("rb") as f:
        for raw in f:
            h.update(raw)
            line = raw.decode().strip()
            if not line:
                continue
            r = json.loads(line)
            n_rows += 1
            xy = _xy(r, units)
            if xy is None or not (math.isfinite(xy[0]) and math.isfinite(xy[1])):
                n_nan += 1
                continue
            key = (canonical_clip_id(str(r["clip"])), int(r["frame_index"]))
            conf = float(r.get("conf", 0.0) or 0.0)
            if key in best:
                n_dup += 1
                if conf <= best[key][0]:
                    continue
            best[key] = (conf, xy)
    preds: dict[str, Pred] = defaultdict(dict)
    for (cid, t), (_, xy) in best.items():
        preds[cid][t] = xy
    return LoadedPreds(dict(preds), n_rows, n_nan, n_dup, h.hexdigest(), str(path))


def _xy(r: dict, units: str) -> tuple[float, float] | None:
    if isinstance(r.get("center"), dict):
        c = r["center"]
        return float(c["x"]) * W_PX, float(c["y"]) * H_PX
    if r.get("x") is not None and r.get("y") is not None:
        x, y = float(r["x"]), float(r["y"])
        return (x * W_PX, y * H_PX) if units == "norm" else (x, y)
    b = r.get("bbox")
    if isinstance(b, dict):
        return (b["x"] + b["w"] / 2) * W_PX, (b["y"] + b["h"] / 2) * H_PX
    return None


def write_predictions(path: Path | str, preds: dict[str, Pred]) -> None:
    with Path(path).open("w") as f:
        for cid in sorted(preds):
            for t in sorted(preds[cid]):
                x, y = preds[cid][t]
                f.write(
                    json.dumps(
                        {
                            "clip": cid,
                            "frame_index": t,
                            "x": round(x, 2),
                            "y": round(y, 2),
                        }
                    )
                    + "\n"
                )


def scored_frames(snap: Snapshot, role: str) -> dict[str, set[int]]:
    """Frames scored per clip: clean frames for test clips; every labelled frame for dev clips."""
    out = {}
    for cid, c in snap.clips.items():
        if c.role != role:
            continue
        out[cid] = c.clean_scored if role == "test" else (set(c.gt) | c.no_ball)
    return out


def score_snapshot(
    snap: Snapshot, lp: LoadedPreds, role: str = "test"
) -> tuple[dict[str, dict], dict, list[str]]:
    """``(per_clip summaries, pooled summary, warnings)`` for the clips of ``role``."""
    per: dict[str, Counts] = {}
    warnings: list[str] = []
    for cid, frames in scored_frames(snap, role).items():
        c = snap.clips[cid]
        pred = lp.preds.get(cid, {})
        if cid not in lp.preds:
            warnings.append(
                f"no predictions for clip {cid} ({role}): all its GT counts as missed"
            )
        gt = {t: g.xy for t, g in c.gt.items()}
        per[cid] = score_clip(pred, gt, c.no_ball, c.n_frames, scored=frames)
    pooled = Counts()
    for c in per.values():
        pooled += c
    unknown = sorted(set(lp.preds) - set(snap.clips))
    if unknown:
        warnings.append(
            f"{len(unknown)} predicted clip(s) not in the snapshot were ignored: {unknown[:5]}"
        )
    return {k: v.summary() for k, v in per.items()}, pooled.summary(), warnings


def fmt(x, nd=3) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def summary_row(name: str, s: dict) -> str:
    lo, hi = wilson(s["tp@10"], s["n_gt_ball"])
    return (
        f"| {name} | {s['n_gt_ball']} | {fmt(s['recall@10'])} [{fmt(lo, 2)}-{fmt(hi, 2)}] "
        f"| {fmt(s['precision@10'])} ({s['tp@10']}/{s['tp@10'] + s['fp@10']}) | {fmt(s['f1@10'])} "
        f"| {fmt(s['recall@5'])} / {fmt(s['recall@20'])} | {fmt(s['coverage'])} "
        f"| {fmt(s['median_err_px'], 1)} |"
    )


TABLE_HEAD = (
    "| method | GT ball frames | recall@10px [95% Wilson] | precision@10px (TP/emitted on scored) "
    "| F1@10 | recall@5 / @20 | coverage | median err px |\n"
    "|---|---|---|---|---|---|---|---|"
)


# --------------------------------------------------------------------------
# paired bootstrap


@dataclasses.dataclass
class Units:
    """Per scored frame outcome at radius ``r``: ``gt`` (1 if a GT ball frame), ``tp``, ``fp``."""

    keys: list[tuple[str, int]]
    gt: np.ndarray
    tp: np.ndarray
    fp: np.ndarray


def frame_units(
    snap: Snapshot, lp: LoadedPreds, role: str = "test", r: float = 10.0
) -> Units:
    keys, gts, tps, fps = [], [], [], []
    for cid, frames in sorted(scored_frames(snap, role).items()):
        c = snap.clips[cid]
        pred = lp.preds.get(cid, {})
        for t in sorted(frames):
            keys.append((cid, t))
            p = pred.get(t)
            if t in c.gt:
                g = c.gt[t].xy
                hit = p is not None and math.hypot(p[0] - g[0], p[1] - g[1]) <= r
                gts.append(1)
                tps.append(int(hit))
                fps.append(int(p is not None and not hit))
            else:  # no_ball frame
                gts.append(0)
                tps.append(0)
                fps.append(int(p is not None))
    return Units(keys, np.array(gts), np.array(tps), np.array(fps))


def paired_bootstrap(
    a: Units,
    b: Units,
    n_boot: int = 10000,
    block: int = 1,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap of ``a - b`` for recall and precision over the shared scored frames.

    ``block`` > 1 resamples runs of ``block`` consecutive frames of one clip together
    (labelled frames are autocorrelated), ``block == 1`` resamples single frames."""
    if a.keys != b.keys:
        raise ValueError("units must cover the same scored frames")
    rng = np.random.default_rng(seed)
    bid: dict[tuple[str, int], int] = {}
    ids = np.empty(len(a.keys), dtype=int)
    for i, (cid, t) in enumerate(a.keys):
        ids[i] = bid.setdefault((cid, t // block), len(bid))
    nb = len(bid)

    def agg(x):
        return np.bincount(ids, weights=x, minlength=nb)

    G, TA, FA, TB, FB = (agg(v) for v in (a.gt, a.tp, a.fp, b.tp, b.fp))
    idx = rng.integers(0, nb, size=(n_boot, nb))
    g = G[idx].sum(1)
    ta, fa, tb, fb = (v[idx].sum(1) for v in (TA, FA, TB, FB))
    with np.errstate(invalid="ignore", divide="ignore"):
        rec_a, rec_b = ta / g, tb / g
        pre_a, pre_b = ta / (ta + fa), tb / (tb + fb)

    def stat(pt, boot):
        ok = np.isfinite(boot)
        lo, hi = np.percentile(boot[ok], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {
            "point": float(pt),
            "lo": float(lo),
            "hi": float(hi),
            "p_gt_0": float((boot[ok] > 0).mean()),
        }

    def ratio(t, f):
        return t / (t + f) if (t + f) else float("nan")

    n_gt = a.gt.sum()
    return {
        "n_units": len(a.keys),
        "n_blocks": nb,
        "block": block,
        "n_boot": n_boot,
        "recall_diff": stat(a.tp.sum() / n_gt - b.tp.sum() / n_gt, rec_a - rec_b),
        "precision_diff": stat(
            ratio(a.tp.sum(), a.fp.sum()) - ratio(b.tp.sum(), b.fp.sum()),
            pre_a - pre_b,
        ),
        "a_hit_b_miss": int(((a.tp == 1) & (b.tp == 0) & (a.gt == 1)).sum()),
        "b_hit_a_miss": int(((b.tp == 1) & (a.tp == 0) & (a.gt == 1)).sum()),
    }
