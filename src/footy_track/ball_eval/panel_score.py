"""Score a ball model on the evaluation panel (``panel.py``): Horvitz-Thompson ratios + cluster bootstrap CIs.

Inputs
    panel      ``<FD>/ball_eval_panel/<version>/<match>.json`` (windows, frames, exact inclusion probabilities).
    human      blind placements: the panel dataset's sidecars (``ball_gt_marks_<version>/<clip>.jsonl``) and/or a
               labelling-job results document. Only tier-1 (``labeller``) ball centres, ``no_ball`` and
               ``not_broadcast`` markers count.
    model      predictions, one top-1 ball per frame or none: jsonl ``{clip, frame, x, y, score}`` (``x`` null or a
               missing row = no prediction). Adapters for the ``machine_labels`` and detection-cache formats.

Coordinates. Human marks are normalised (0..1); the estimator works in **pixels of the clip frame** (``width`` x
``height`` recorded in the panel, i.e. the frame the labeller showed). Predictions are pixels of that same frame
(``units="norm"`` converts normalised ones). ``r`` (default 10 and 25 px) is the Euclidean distance between the
predicted and the human ball centre.

Per-frame outcome (frames with a human label in the panel's windows only):

    ball present  + prediction within r of a human centre   ``hit``
    ball present  + prediction farther than r               ``off``
    ball present  + no prediction                           ``miss``
    no ball       + prediction                              ``false_ball``
    no ball       + no prediction                           ``true_empty``

``not_broadcast`` frames are out of the population (the metric is a domain estimate) and are dropped; frames with
no label yet drop their whole window unless ``include_partial``.

Metrics (per radius), each a ratio of Horvitz-Thompson totals, weight ``1 / pi_frame``:

    recall@r      = hit / (hit + off + miss)
    precision@r   = hit / (hit + off + false_ball)
    false_ball    = false_ball / (false_ball + true_empty)

Reported per match and pooled two ways: **match-averaged** (unweighted mean of the per-match ratios: "a typical
new match", the headline) and **frame-weighted** (ratio of the pooled HT totals). CIs: stratified cluster
bootstrap, i.e. windows are resampled with replacement within each match (the stratum), 2000 draws by default,
seedable; the percentile 95 % interval is reported. QA cards (repeats, gold) are never part of any estimate;
:func:`qa_report` reports repeat consistency and gold agreement separately.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from footy_track.ball_eval import panel as P  # noqa: N812
from footy_track.ball_eval.detection_replay import BALL_TAGS
from footy_track.clip_id import canonical_clip_id

SCHEMA = "ball-eval-panel-score/v1"
OUTCOMES = ("hit", "off", "miss", "false_ball", "true_empty")
METRICS = ("recall", "precision", "false_ball")
DEFAULT_RADII = (10.0, 25.0)
DEFAULT_DRAWS = 2000
BALL, NO_BALL, NOT_BROADCAST = "ball", "no_ball", "not_broadcast"

Key = tuple[str, int]  # (clip, frame)


# --------------------------------------------------------------------------------------------
# Human labels
# --------------------------------------------------------------------------------------------
@dataclass
class Human:
    state: str  # ball | no_ball | not_broadcast
    centres: list[tuple[float, float]]  # normalised
    #: the box was a detector suggestion accepted as-is (marker ``accepted_suggestion``). Audit only: it is an
    #: ordinary tier-1 label everywhere, the flag feeds the "share accepted" stat and the blind-vs-shown check.
    accepted: bool = False


def _centre(row: dict) -> tuple[float, float] | None:
    c = row.get("center")
    if isinstance(c, dict) and "x" in c and "y" in c:
        return float(c["x"]), float(c["y"])
    b = row.get("bbox")
    if isinstance(b, dict) and all(k in b for k in "xywh"):
        return b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    return None


def _add_row(out: dict[Key, Human], clip: str, row: dict) -> None:
    try:
        frame = int(row["frame_index"])
    except (KeyError, TypeError, ValueError):
        return
    tags = [t for t in (row.get("tags") or []) if isinstance(t, str)]
    key = (clip, frame)
    if "not_broadcast" in tags:
        out.setdefault(key, Human(NOT_BROADCAST, []))
    elif "no_ball" in tags:
        out.setdefault(key, Human(NO_BALL, []))
    elif "labeller" in tags and any(t in BALL_TAGS or t == "ball" for t in tags):
        c = _centre(row)
        if c is not None:
            h = out.get(key)
            acc = P.SUGGEST_MARKER_TAG in tags
            if h is None or h.state != BALL:
                out[key] = Human(
                    BALL, [c], acc
                )  # a ball mark beats a marker on the same frame
            else:
                h.centres.append(c)
                h.accepted = h.accepted or acc


def load_human_sidecars(marks_dir: Path, clips: Iterable[str]) -> dict[Key, Human]:
    """Tier-1 marks of the given clips from ``<marks_dir>/<clip>.jsonl`` (exact stems; nothing is alias-merged:
    the QA alias clips must stay separate from their originals)."""
    out: dict[Key, Human] = {}
    for clip in sorted(set(clips)):
        p = Path(marks_dir) / f"{clip}.jsonl"
        if not p.is_file():
            continue
        for raw in p.read_text().splitlines():
            if raw.strip():
                try:
                    _add_row(out, clip, json.loads(raw))
                except ValueError:
                    continue
    return out


def load_human_results(path: Path) -> dict[Key, Human]:
    """Marks from a labelling-job results document (``GET /api/jobs/<ref>/results``)."""
    doc = json.loads(Path(path).read_text())
    out: dict[Key, Human] = {}
    for row in doc.get("labels", []):
        status = row.get("status")
        tags = list(row.get("tags") or [])
        if status == "ball" and row.get("tier", 1) == 1 and "labeller" not in tags:
            tags.append("labeller")
        if status == "ball" and not any(t in BALL_TAGS or t == "ball" for t in tags):
            tags.append("in_play_ball")
        if status in (NO_BALL, NOT_BROADCAST) and status not in tags:
            tags.append(status)
        _add_row(out, row["clip"], {**row, "tags": tags})
    return out


# --------------------------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------------------------
Pred = tuple[float, float, float]  # x_px, y_px, score


def load_predictions(
    path: Path,
    *,
    units: str = "px",
    size: tuple[int, int] | None = None,
    min_score: float = 0.0,
) -> dict[Key, Pred]:
    """Predictions jsonl ``{clip, frame, x, y, score}`` -> ``{(clip, frame): (x_px, y_px, score)}`` (top-1 by score).

    ``x`` / ``y`` null or missing = no prediction for that frame. ``units="norm"`` needs ``size=(width, height)``.
    """
    out: dict[Key, Pred] = {}
    for raw in Path(path).read_text().splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        if d.get("x") is None or d.get("y") is None:
            continue
        score = float(d.get("score") if d.get("score") is not None else 1.0)
        if score < min_score:
            continue
        x, y = float(d["x"]), float(d["y"])
        if units == "norm":
            if size is None:
                raise ValueError("norm predictions need the clip frame size")
            x, y = x * size[0], y * size[1]
        key = (d["clip"], int(d["frame"]))
        if key not in out or score > out[key][2]:
            out[key] = (x, y, score)
    return out


def predictions_from_machine_labels(
    path: Path, clip: str, size: tuple[int, int], *, min_score: float = 0.0
) -> dict[Key, Pred]:
    """Top-1 ball per frame from a ``machine_labels`` / pipeline detections jsonl
    (``{frame_index, bbox{x,y,w,h normalised top-left}, confidence, tags[0]}``)."""
    out: dict[Key, Pred] = {}
    for raw in Path(path).read_text().splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        tags = d.get("tags") or []
        if not tags or not (tags[0] in BALL_TAGS or tags[0] == "ball"):
            continue
        b, conf = d.get("bbox"), float(d.get("confidence", 1.0))
        if not isinstance(b, dict) or conf < min_score:
            continue
        key = (clip, int(d["frame_index"]))
        if key not in out or conf > out[key][2]:
            out[key] = (
                (b["x"] + b["w"] / 2) * size[0],
                (b["y"] + b["h"] / 2) * size[1],
                conf,
            )
    return out


def predictions_from_cache(
    path: Path, clip: str, size: tuple[int, int], *, min_score: float = 0.0
) -> dict[Key, Pred]:
    """Top-1 candidate per frame from a ``ball_tracking_eval`` candidate cache (``detection_replay.read_cache``)."""
    from footy_track.ball_eval.detection_replay import read_cache  # noqa: PLC0415

    out: dict[Key, Pred] = {}
    for f, cands in read_cache(Path(path)).frames:
        best = max(
            (c for c in cands if c.confidence >= min_score),
            key=lambda c: c.confidence,
            default=None,
        )
        if best is not None:
            out[(clip, int(f))] = (
                (best.x + best.w / 2) * size[0],
                (best.y + best.h / 2) * size[1],
                float(best.confidence),
            )
    return out


def find_clip_file(directory: Path, clip: str) -> Path | None:
    """``<directory>/<stem>.jsonl`` for ``clip`` or any alias with the same canonical clip id."""
    p = Path(directory) / f"{clip}.jsonl"
    if p.is_file():
        return p
    cid = canonical_clip_id(clip)
    for cand in sorted(Path(directory).glob("*.jsonl")):
        if canonical_clip_id(cand.stem) == cid:
            return cand
    return None


# --------------------------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------------------------
def classify(
    human: Human, pred: Pred | None, r_px: float, size: tuple[int, int]
) -> str | None:
    """Outcome of one frame at radius ``r_px`` (``None`` = not scored: not a main-camera frame)."""
    if human.state == NOT_BROADCAST:
        return None
    if human.state == NO_BALL:
        return "true_empty" if pred is None else "false_ball"
    if pred is None:
        return "miss"
    d = min(
        math.hypot((pred[0] - cx * size[0]), (pred[1] - cy * size[1]))
        for cx, cy in human.centres
    )
    return "hit" if d <= r_px else "off"


def ratios(t: np.ndarray) -> dict[str, np.ndarray]:
    """Metric ratios from HT totals ``t[..., 5]`` (order of ``OUTCOMES``); NaN where the denominator is 0."""
    hit, off, miss, fb, te = (t[..., i] for i in range(5))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "recall": np.where(hit + off + miss > 0, hit / (hit + off + miss), np.nan),
            "precision": np.where(hit + off + fb > 0, hit / (hit + off + fb), np.nan),
            "false_ball": np.where(fb + te > 0, fb / (fb + te), np.nan),
        }


def ht_total(values: np.ndarray, pi: np.ndarray) -> float:
    """Horvitz-Thompson estimate of a population total from sampled ``values`` with inclusion probabilities ``pi``."""
    return float(np.sum(np.asarray(values, float) / np.asarray(pi, float)))


# --------------------------------------------------------------------------------------------
# Per-window tables
# --------------------------------------------------------------------------------------------
@dataclass
class MatchTable:
    match: str
    size: tuple[int, int]
    counts: np.ndarray  # (n_scored_windows, n_radii, 5) integer frame counts
    weights: np.ndarray  # (n_scored_windows,) 1 / pi_frame
    n_windows: int
    n_unlabelled_frames: int
    n_dropped_windows: int
    n_not_broadcast: int


def match_table(
    panel: dict,
    human: dict[Key, Human],
    preds: dict[Key, Pred],
    radii: tuple[float, ...],
    *,
    include_partial: bool = False,
) -> MatchTable:
    w, h = panel.get("width"), panel.get("height")
    if not w or not h:
        raise ValueError(
            f"{panel['match']}: panel has no clip frame size (built without a probe); cannot score in pixels"
        )
    size = (int(w), int(h))
    rows, wts = [], []
    unlabelled = dropped = not_bc = 0
    for win in panel["windows"]:
        frames = [(win["clip"], f) for f in win["frames"]]
        missing = sum(1 for k in frames if k not in human)
        unlabelled += missing
        if missing and not include_partial:
            dropped += 1
            continue
        c = np.zeros((len(radii), 5), dtype=np.int64)
        for k in frames:
            hm = human.get(k)
            if hm is None:
                continue
            for ri, r in enumerate(radii):
                o = classify(hm, preds.get(k), r, size)
                if o is not None:
                    c[ri, OUTCOMES.index(o)] += 1
            not_bc += hm.state == NOT_BROADCAST
        rows.append(c)
        wts.append(1.0 / win["pi_frame"])
    counts = np.stack(rows) if rows else np.zeros((0, len(radii), 5), dtype=np.int64)
    return MatchTable(
        panel["match"],
        size,
        counts,
        np.asarray(wts, float),
        len(panel["windows"]),
        unlabelled,
        dropped,
        not_bc,
    )


def _totals(t: MatchTable, idx: np.ndarray | None = None) -> np.ndarray:
    """HT totals ``(n_radii, 5)`` of the windows ``idx`` (all when None; repeats allowed: bootstrap)."""
    c = t.counts if idx is None else t.counts[idx]
    wt = t.weights if idx is None else t.weights[idx]
    return np.tensordot(wt, c, axes=(0, 0))


def _ci(draws: np.ndarray) -> tuple[float, float]:
    d = draws[~np.isnan(draws)]
    if d.size < 2:
        return (float("nan"), float("nan"))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _round(v: float | None) -> float | None:
    return None if v is None or math.isnan(v) else round(float(v), 4)


def _cell(est: float, draws: np.ndarray) -> dict:
    lo, hi = _ci(draws)
    return {"est": _round(est), "lo": _round(lo), "hi": _round(hi)}


def score(
    panels: list[dict],
    human: dict[Key, Human],
    preds: dict[Key, Pred],
    *,
    radii: tuple[float, ...] = DEFAULT_RADII,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    include_partial: bool = False,
    model: str | None = None,
    blind_only: bool = False,
    root: Path | None = None,
) -> dict:
    """The full score report (see the module docstring). Deterministic for a given ``seed``.

    ``blind_only`` scores the model on blind windows only (the random blind control and labels placed before
    suggestions existed): the suggesting detector is the model most favoured by its own suggestions."""
    groups = {p["match"]: window_groups(p, human, root) for p in panels}
    scored = panels
    if blind_only:
        scored = [
            {
                **p,
                "windows": [
                    w
                    for w in p["windows"]
                    if groups[p["match"]][w["window_id"]] != "shown"
                ],
            }
            for p in panels
        ]
    tables = [
        match_table(p, human, preds, radii, include_partial=include_partial)
        for p in scored
    ]
    rng = np.random.default_rng(seed)

    # bootstrap: per match, resample its windows with replacement; keep per-draw HT totals (draws, n_radii, 5)
    boot: list[np.ndarray | None] = []
    for t in tables:
        n = len(t.weights)
        if n == 0:
            boot.append(None)
            continue
        idx = rng.integers(0, n, size=(draws, n))
        boot.append(np.einsum("dn,dnrk->drk", t.weights[idx], t.counts[idx]))

    point = [_totals(t) if len(t.weights) else None for t in tables]

    def metric_block(est_t: np.ndarray, draw_t: np.ndarray | None) -> dict:
        out = {}
        e = ratios(est_t)
        d = ratios(draw_t) if draw_t is not None else None
        for ri, r in enumerate(radii):
            out[f"r{r:g}"] = {
                m: _cell(e[m][ri], d[m][:, ri] if d is not None else np.array([]))
                for m in METRICS
            }
        return out

    report_matches: dict[str, dict] = {}
    for t, pt, bt in zip(tables, point, boot, strict=True):
        report_matches[t.match] = {
            "n_windows": t.n_windows,
            "n_windows_scored": int(len(t.weights)),
            "n_windows_dropped_incomplete": t.n_dropped_windows,
            "n_frames_unlabelled": t.n_unlabelled_frames,
            "n_frames_not_broadcast": int(t.n_not_broadcast),
            "frame_size": list(t.size),
            "counts": {
                f"r{r:g}": {
                    o: int(t.counts[:, ri, oi].sum()) for oi, o in enumerate(OUTCOMES)
                }
                for ri, r in enumerate(radii)
            }
            if len(t.weights)
            else {},
            "metrics": metric_block(pt, bt) if pt is not None else {},
        }

    live = [i for i, p in enumerate(point) if p is not None]
    pooled: dict = {}
    if live:
        # frame-weighted: ratio of the pooled HT totals
        fw_est = sum(point[i] for i in live)
        fw_draws = sum(boot[i] for i in live)
        pooled["frame_weighted"] = metric_block(fw_est, fw_draws)
        # match-averaged: unweighted mean of the per-match ratios (a match with an undefined ratio is skipped)
        ma = {}
        per_est = [ratios(point[i]) for i in live]
        per_draw = [ratios(boot[i]) for i in live]
        for ri, r in enumerate(radii):
            cell = {}
            for m in METRICS:
                e = np.array([pe[m][ri] for pe in per_est])
                dd = np.stack([pd[m][:, ri] for pd in per_draw], axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", RuntimeWarning
                    )  # all-NaN rows: a draw with no defined ratio
                    est = float(np.nanmean(e))
                    dr = np.nanmean(dd, axis=1)
                cell[m] = _cell(est, dr)
            ma[f"r{r:g}"] = cell
        pooled["match_averaged"] = ma
        pooled["n_matches"] = len(live)
    return {
        "schema": SCHEMA,
        "panel_version": panels[0]["panel_version"] if panels else None,
        "model": model,
        "radii_px": list(radii),
        "units": "pixels of the clip frame (frame_size per match); human marks are normalised and scaled by it",
        "estimator": "Horvitz-Thompson ratio, weight 1/pi_frame; stratified (by match) cluster (window) bootstrap",
        "draws": draws,
        "seed": seed,
        "ci": "percentile 95%",
        "include_partial": include_partial,
        "blind_only": blind_only,
        "panel_code_sha": sorted({p.get("code_sha") for p in panels}),
        "matches": report_matches,
        "pooled": pooled,
        "qa": qa_report(panels, human),
        "suggestions": suggestion_report(
            panels,
            human,
            preds,
            groups,
            radii=radii,
            draws=draws,
            seed=seed,
            include_partial=include_partial,
        ),
    }


# --------------------------------------------------------------------------------------------
# Suggestions: share accepted, and blind-vs-shown bias check
# --------------------------------------------------------------------------------------------
def suggest_pre_labelled(panel: dict, root: Path | None = None) -> set[str] | None:
    """``clip:frame`` keys that were already labelled when the match's suggest file appeared (None = no suggest
    file yet, so no label of this panel was ever made with a suggestion on offer)."""
    sf = P.find_suggest_file(panel["match"], panel["panel_version"], root)
    if sf is None:
        return None
    mp = P.suggest_meta_path(sf)
    try:
        return set(json.loads(mp.read_text()).get("pre_labelled") or [])
    except (OSError, ValueError):
        return set()  # unknown history: treat every label as made with the suggestion on offer


def window_groups(
    panel: dict, human: dict[Key, Human], root: Path | None = None
) -> dict[str, str]:
    """``{window_id: "blind" | "blind_equiv" | "shown"}``.

    ``blind``: the random blind control (``panel.blind_windows``): never had a suggestion. ``blind_equiv``: a
    ``shown`` window whose labels were all placed before suggestions existed (George's first Forest labels),
    so also unsuggested. ``shown``: the rest. The comparison treats blind and blind_equiv together."""
    blind = P.blind_windows(panel)
    pre = suggest_pre_labelled(panel, root)
    out: dict[str, str] = {}
    for w in panel["windows"]:
        wid = w["window_id"]
        if wid in blind:
            out[wid] = "blind"
            continue
        labelled = [f"{w['clip']}:{f}" for f in w["frames"] if (w["clip"], f) in human]
        if pre is None or (labelled and all(k in pre for k in labelled)):
            out[wid] = "blind_equiv"
        else:
            out[wid] = "shown"
    return out


def _group_cells(
    tables: list[MatchTable], radii: tuple[float, ...], draws: int, rng
) -> tuple[dict, dict]:
    """Frame-weighted pooled metric cells of ``tables`` and the raw ``(est, draws)`` ratio arrays (for a difference)."""
    ests, boots = [], []
    for t in tables:
        n = len(t.weights)
        if n == 0:
            continue
        idx = rng.integers(0, n, size=(draws, n))
        boots.append(np.einsum("dn,dnrk->drk", t.weights[idx], t.counts[idx]))
        ests.append(_totals(t))
    if not ests:
        return {}, {}
    e, d = ratios(sum(ests)), ratios(sum(boots))
    cells = {
        f"r{r:g}": {m: _cell(e[m][ri], d[m][:, ri]) for m in METRICS}
        for ri, r in enumerate(radii)
    }
    return cells, {"est": e, "draws": d}


def _compare(
    blind: list[MatchTable],
    shown: list[MatchTable],
    radii: tuple[float, ...],
    draws: int,
    rng,
) -> dict:
    bc, braw = _group_cells(blind, radii, draws, rng)
    sc, sraw = _group_cells(shown, radii, draws, rng)
    out: dict = {
        "n_windows_blind": int(sum(len(t.weights) for t in blind)),
        "n_windows_shown": int(sum(len(t.weights) for t in shown)),
        "blind": bc,
        "shown": sc,
        "diff_blind_minus_shown": {},
    }
    if braw and sraw:
        for ri, r in enumerate(radii):
            out["diff_blind_minus_shown"][f"r{r:g}"] = {
                m: _cell(
                    braw["est"][m][ri] - sraw["est"][m][ri],
                    braw["draws"][m][:, ri] - sraw["draws"][m][:, ri],
                )
                for m in METRICS
            }
    return out


def suggestion_report(
    panels: list[dict],
    human: dict[Key, Human],
    preds: dict[Key, Pred],
    groups: dict[str, dict[str, str]],
    *,
    radii: tuple[float, ...],
    draws: int,
    seed: int,
    include_partial: bool,
) -> dict:
    """Per match and pooled: the share of labels that were accepted suggestions, and the model's metrics on the
    blind vs the shown windows with the difference (blind - shown) and its 95 % CI. A large gap means the
    suggestions shift the labels. Accepted boxes are ordinary tier-1 labels and stay in every estimate."""
    rng = np.random.default_rng(seed + 7)
    blind_t: list[MatchTable] = []
    shown_t: list[MatchTable] = []
    per_match: dict[str, dict] = {}
    tot = {"labelled_balls": 0, "accepted": 0, "shown_balls": 0, "shown_accepted": 0}
    for p in panels:
        g = groups[p["match"]]

        def sub(kinds: tuple[str, ...], p=p, g=g) -> dict:
            return {**p, "windows": [w for w in p["windows"] if g[w["window_id"]] in kinds]}

        bt = match_table(sub(("blind", "blind_equiv")), human, preds, radii, include_partial=include_partial)
        st = match_table(sub(("shown",)), human, preds, radii, include_partial=include_partial)
        blind_t.append(bt)
        shown_t.append(st)
        c = dict.fromkeys(tot, 0)
        for w in p["windows"]:
            for f in w["frames"]:
                h = human.get((w["clip"], f))
                if h is None or h.state != BALL:
                    continue
                c["labelled_balls"] += 1
                c["accepted"] += h.accepted
                if g[w["window_id"]] == "shown":
                    c["shown_balls"] += 1
                    c["shown_accepted"] += h.accepted
        for k, v in c.items():
            tot[k] += v
        per_match[p["match"]] = {
            "windows": {
                k: sum(1 for v in g.values() if v == k)
                for k in ("blind", "blind_equiv", "shown")
            },
            **c,
            "share_accepted_of_ball_labels": _share(c["accepted"], c["labelled_balls"]),
            "share_accepted_of_shown_ball_labels": _share(c["shown_accepted"], c["shown_balls"]),
            "comparison": _compare([bt], [st], radii, draws, rng),
        }
    return {
        "note": "blind = random 25% control + labels placed before suggestions existed (blind_equiv); shown = the rest. "
        "Accepted suggestions are ordinary tier-1 labels; the marker is audit only.",
        "pooled": {
            **tot,
            "share_accepted_of_ball_labels": _share(tot["accepted"], tot["labelled_balls"]),
            "share_accepted_of_shown_ball_labels": _share(tot["shown_accepted"], tot["shown_balls"]),
            "comparison": _compare(blind_t, shown_t, radii, draws, rng),
        },
        "matches": per_match,
    }


def _share(a: int, n: int) -> float | None:
    return round(a / n, 4) if n else None


# --------------------------------------------------------------------------------------------
# QA: repeat consistency and gold agreement (never part of the estimates)
# --------------------------------------------------------------------------------------------
def _spread_stats(d: list[float], r_px: float) -> dict:
    if not d:
        return {"n": 0}
    a = np.asarray(d, float)
    return {
        "n": len(d),
        "median_px": round(float(np.median(a)), 2),
        "p90_px": round(float(np.percentile(a, 90)), 2),
        "max_px": round(float(a.max()), 2),
        "frac_le_5px": round(float(np.mean(a <= 5)), 3),
        f"frac_le_{r_px:g}px": round(float(np.mean(a <= r_px)), 3),
    }


def qa_report(panels: list[dict], human: dict[Key, Human], r_px: float = 10.0) -> dict:
    """Repeat consistency (pixel spread between two blind placements of the same frame) and gold agreement
    (blind placement vs existing tier-1 GT), pooled over the given panels. Pending cards are counted, not scored."""
    rep_d: list[float] = []
    gold_d: list[float] = []
    out = {
        "repeat": {
            "cards": 0,
            "pending": 0,
            "both_ball": 0,
            "presence_disagree": 0,
            "both_no_ball": 0,
        },
        "gold": {"cards": 0, "pending": 0, "placed": 0, "said_no_ball": 0},
    }
    for p in panels:
        size = (int(p.get("width") or 0), int(p.get("height") or 0))
        for q in p["qa"]:
            a = human.get((q["clip"], q["frame"]))
            if q["qa"] == "repeat":
                out["repeat"]["cards"] += 1
                b = human.get((q["of"]["clip"], q["of"]["frame"]))
                if a is None or b is None or NOT_BROADCAST in (a.state, b.state):
                    out["repeat"]["pending"] += a is None or b is None
                elif a.state == NO_BALL and b.state == NO_BALL:
                    out["repeat"]["both_no_ball"] += 1
                elif a.state != b.state:
                    out["repeat"]["presence_disagree"] += 1
                else:
                    out["repeat"]["both_ball"] += 1
                    rep_d.append(
                        min(
                            math.hypot((ax - bx) * size[0], (ay - by) * size[1])
                            for ax, ay in a.centres
                            for bx, by in b.centres
                        )
                    )
            elif q["qa"] == "gold":
                out["gold"]["cards"] += 1
                if a is None:
                    out["gold"]["pending"] += 1
                elif a.state == BALL:
                    out["gold"]["placed"] += 1
                    gx, gy = q["gold"]["x"], q["gold"]["y"]
                    gold_d.append(
                        min(
                            math.hypot((ax - gx) * size[0], (ay - gy) * size[1])
                            for ax, ay in a.centres
                        )
                    )
                elif a.state == NO_BALL:
                    out["gold"]["said_no_ball"] += 1
    out["repeat"]["spread"] = _spread_stats(rep_d, r_px)
    out["gold"]["agreement"] = _spread_stats(gold_d, r_px)
    return out
