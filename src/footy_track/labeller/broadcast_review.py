"""Broadcast-frame review: yes/no on frames the broadcast classifier was unsure about.

The per-match Prefect flow ``broadcast-review`` (jobs repo) runs the frame classifier over a
full match, picks the frames nearest the decision boundary plus a small random sample, writes
them as JPEGs to ``<dataset clips_dir>/<match>/f<frame>.jpg`` and submits a ``broadcast_review``
labelling job (item = ``{clip: <match key>, frame, ts_s, p_yes, pred, reason}``).

Verdicts live in ``<dataset marks_dir>/broadcast_review/<match>.json``::

    {"frames": {"12345": {"verdict": "yes"|"no"|"unsure", "ts": "...", "p_yes": 0.43, "pred": "No",
                          "reason": "edge"}}}

``yes`` = a main-camera broadcast view (what the classifier calls ``Yes``), ``no`` = anything
else (replay, close-up, crowd, graphic ...), ``unsure`` = skip (counts as answered, excluded
from training export). ``scripts/export_broadcast_review.py`` turns these into the
``train/Yes`` ``train/No`` folder layout ``scripts/train_classifier.py --local-dataset`` reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import random
import re
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from footy_track.labeller import datasets, hooks, job_edit, jobs

VERDICTS = ("yes", "no", "unsure")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
_LOCK = threading.RLock()
_WEB = Path(__file__).parent / "web"

router = APIRouter()


def check_match(match: str) -> str:
    if not _NAME.fullmatch(match or "") or ".." in match:
        raise ValueError(f"bad match key {match!r}")
    return match


def frame_path(ds: datasets.Dataset, match: str, frame: int) -> Path:
    return ds.clips_dir / check_match(match) / f"f{int(frame)}.jpg"


def store_path(ds: datasets.Dataset, match: str) -> Path:
    return ds.marks_dir / "broadcast_review" / f"{check_match(match)}.json"


def read_store(ds: datasets.Dataset, match: str) -> dict[int, dict]:
    """``{frame: {verdict, ts, ...}}`` ({} when nothing is labelled yet)."""
    try:
        doc = json.loads(store_path(ds, match).read_text())
    except (OSError, ValueError):
        return {}
    out: dict[int, dict] = {}
    for k, v in ((doc or {}).get("frames") or {}).items():
        if isinstance(v, dict) and str(k).isdigit() and v.get("verdict") in VERDICTS:
            out[int(k)] = v
    return out


def _write_atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def set_verdict(ds: datasets.Dataset, match: str, frame: int, verdict: str | None, extra: dict | None = None) -> dict:
    """Record (or, with ``verdict=None``, clear) the verdict for one frame."""
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {list(VERDICTS)}")
    path = store_path(ds, match)
    with _LOCK:
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            doc = {}
        frames = doc.setdefault("frames", {})
        if verdict is None:
            frames.pop(str(int(frame)), None)
        else:
            frames[str(int(frame))] = {
                **{k: v for k, v in (extra or {}).items() if k in ("p_yes", "pred", "reason", "ts_s")},
                "verdict": verdict,
                "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        _write_atomic(path, doc)
    hooks.emit("broadcast_review", clip=match, unit=int(frame), verdict=verdict, dataset=ds.name)
    return frames.get(str(int(frame))) or {}


# ---------------------------------------------------------------------------
# Calibration of the classifier's p_yes (labelling hook ``broadcast_calibrate``)
# ---------------------------------------------------------------------------
#
# The classifier's p_yes is badly miscalibrated (measured on 200 verdicts of one match: the empirical
# 50/50 point is p_yes ~ 0.005, not 0.5). While George labels, the hook refits a Platt scaling on
# logit(p_yes) from every yes/no verdict of the dataset and re-sorts the frames still to label by how
# close they are to that fitted boundary (uncertainty sampling on the logit scale, where the scores
# cluster at 0, ~0.5 and 1). Each fit is written as a new immutable ``calibration/vNNN.json``.

P_EPS = 1e-6
CAL_MIN_PER_CLASS = 10
CAL_JITTER = 0.5  # logit units of noise on the uncertainty score
CAL_BUCKET_S = 60.0  # frames in one time bucket compete: the k-th most uncertain pays k * penalty
CAL_BUCKET_PENALTY = 0.15
_CAL_NAME = re.compile(r"^v(\d{3,})\.json$")


def logit(p: float) -> float:
    p = min(max(float(p), P_EPS), 1.0 - P_EPS)
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def fit_platt(x, y, iters: int = 100) -> tuple[float, float]:
    """Platt scaling: ``P(yes) = sigmoid(a * x + b)`` with ``x = logit(p_yes)``.

    Newton's method on the cross-entropy with Platt's smoothed targets (``(N+ + 1)/(N+ + 2)`` and
    ``1/(N- + 2)``), which keeps ``a`` finite when the verdicts are (nearly) separable, as they are here."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
    t = np.where(y > 0.5, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def nll(a: float, b: float) -> float:
        z = a * x + b
        return float(np.sum(np.logaddexp(0.0, z) - t * z))

    a, b = 1.0, 0.0
    cur = nll(a, b)
    for _ in range(iters):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = np.array([np.sum((p - t) * x), np.sum(p - t)])
        w = p * (1 - p)
        h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        lam = 1.0
        while lam > 1e-6:
            na, nb = a - lam * step[0], b - lam * step[1]
            new = nll(na, nb)
            if new <= cur:
                break
            lam /= 2
        else:
            break
        done = abs(cur - new) < 1e-12
        a, b, cur = na, nb, new
        if done:
            break
    return float(a), float(b)


def calibration_dir(ds: datasets.Dataset) -> Path:
    return ds.marks_dir / "broadcast_review" / "calibration"


def _versions(ds: datasets.Dataset) -> list[tuple[int, Path]]:
    d = calibration_dir(ds)
    if not d.is_dir():
        return []
    return sorted((int(m.group(1)), p) for p in d.iterdir() if (m := _CAL_NAME.match(p.name)))


def latest_calibration(ds: datasets.Dataset) -> dict | None:
    """The newest calibration (``{version, a, b, n, boundary_p_yes, ...}``) or None."""
    for _, p in reversed(_versions(ds)):
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and "a" in doc and "b" in doc:
            return doc
    return None


def calibrated_p(p_yes: float, cal: dict) -> float:
    return sigmoid(cal["a"] * logit(p_yes) + cal["b"])


def collect_training(ds: datasets.Dataset) -> dict:
    """``{x, y, n_yes, n_no, n_unsure, n_no_p, matches}`` from every yes/no verdict (with a stored
    ``p_yes``) of every match in the dataset; ``unsure`` is excluded."""
    root = ds.marks_dir / "broadcast_review"
    xs: list[float] = []
    ys: list[int] = []
    n_unsure = n_no_p = 0
    matches = []
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            frames = (json.loads(path.read_text()) or {}).get("frames") or {}
        except (OSError, ValueError):
            continue
        used = 0
        for rec in frames.values():
            if not isinstance(rec, dict):
                continue
            v = rec.get("verdict")
            if v == "unsure":
                n_unsure += 1
            elif v in ("yes", "no"):
                if isinstance(rec.get("p_yes"), (int, float)) and not isinstance(rec["p_yes"], bool):
                    xs.append(logit(rec["p_yes"]))
                    ys.append(1 if v == "yes" else 0)
                    used += 1
                else:
                    n_no_p += 1
        if used:
            matches.append(path.stem)
    n_yes = sum(ys)
    return {"x": xs, "y": ys, "n_yes": n_yes, "n_no": len(ys) - n_yes, "n_unsure": n_unsure, "n_no_p": n_no_p, "matches": matches}


def fit_dataset(ds: datasets.Dataset, *, min_per_class: int = CAL_MIN_PER_CLASS, job_id: str | None = None, write: bool = True) -> dict:
    """Fit (and, with ``write``, store as the next version) the calibration for a dataset.

    Returns ``{"fitted": True, "calibration": {...}}`` or ``{"fitted": False, "noop": reason, ...counts}``.
    A fit on the same verdict count as the latest version is not stored again (that version is returned)."""
    tr = collect_training(ds)
    counts = {k: tr[k] for k in ("n_yes", "n_no", "n_unsure", "n_no_p")}
    if tr["n_yes"] < min_per_class or tr["n_no"] < min_per_class:
        return {"fitted": False, "noop": f"need >= {min_per_class} yes and >= {min_per_class} no verdicts with a p_yes, have "
                                         f"{tr['n_yes']} yes / {tr['n_no']} no", **counts}  # fmt: skip
    prev = latest_calibration(ds)
    if prev is not None and prev.get("n") == len(tr["y"]) and prev.get("n_yes") == tr["n_yes"]:
        return {"fitted": True, "reused": True, "calibration": prev, **counts}
    a, b = fit_platt(tr["x"], tr["y"])
    if not a > 0:
        return {"fitted": False, "noop": f"degenerate fit (slope a={a:.3g} <= 0): the verdicts do not rise with p_yes", **counts}
    cal = {
        "schema": "broadcast-calibration/v1", "method": "platt on logit(p_yes), smoothed targets", "p_clip": P_EPS,
        "a": a, "b": b, "n": len(tr["y"]), "n_yes": tr["n_yes"], "n_no": tr["n_no"], "n_unsure_excluded": tr["n_unsure"],
        "boundary_logit": -b / a, "boundary_p_yes": sigmoid(-b / a), "matches": tr["matches"],
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **({"job_id": job_id} if job_id else {}),
    }  # fmt: skip
    if write:
        with _LOCK:
            d = calibration_dir(ds)
            v = (_versions(ds)[-1][0] if _versions(ds) else 0) + 1
            while True:
                cal["version"] = v
                if jobs._write_atomic(d / f"v{v:03d}.json", cal, exclusive=True):
                    break
                v += 1
    return {"fitted": True, "reused": False, "calibration": cal, **counts}


def uncertainty_order(job_id: str, version: int, items: list[tuple[int, dict]], cal: dict, *, jitter: float = CAL_JITTER,
                      bucket_s: float = CAL_BUCKET_S, penalty: float = CAL_BUCKET_PENALTY) -> list[int]:  # fmt: skip
    """Item positions, most uncertain first. Score = ``|a * logit(p_yes) + b|`` (the distance to the
    fitted 50/50 boundary on the logit scale) plus Gaussian jitter, plus ``penalty`` per more-uncertain
    frame already ranked in the same time bucket so one cluster of near-identical frames cannot hog the
    top. Deterministic for (job, version). Items without a ``p_yes`` go last, in their old order."""
    rng = random.Random(f"{job_id}:{version}")
    scored, unscored = [], []
    for pos, it in items:
        p = it.get("p_yes")
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            scored.append((abs(cal["a"] * logit(p) + cal["b"]) + rng.gauss(0.0, jitter), pos, it))
        else:
            unscored.append(pos)
    buckets: dict[tuple, list] = {}
    for s, pos, it in scored:
        ts = it.get("ts_s")
        key = (it.get("clip"), int(ts // bucket_s) if isinstance(ts, (int, float)) else pos)
        buckets.setdefault(key, []).append((s, pos))
    final = []
    for rows in buckets.values():
        rows.sort()
        final.extend((s + penalty * k, pos) for k, (s, pos) in enumerate(rows))
    final.sort()
    return [pos for _, pos in final] + unscored


def _broadcast_calibrate(ctx: hooks.HookContext) -> dict:
    """Refit Platt scaling on the dataset's yes/no verdicts, then reorder the job's remaining frames
    by uncertainty. params: ``min_per_class`` (10), ``reorder`` (true), ``jitter``, ``bucket_s``, ``penalty``."""
    ds = datasets.get(ctx.job["dataset"])
    p = ctx.params
    fit = fit_dataset(ds, min_per_class=int(p.get("min_per_class", CAL_MIN_PER_CLASS)), job_id=ctx.job["id"])
    if not fit["fitted"]:
        return {"noop": fit["noop"], **{k: fit[k] for k in ("n_yes", "n_no")}}
    cal = fit["calibration"]
    out = {"version": cal["version"], "a": round(cal["a"], 4), "b": round(cal["b"], 4), "n": cal["n"],
           "boundary_p_yes": round(cal["boundary_p_yes"], 6), "reused": fit["reused"]}  # fmt: skip
    if p.get("reorder", True):
        todo = job_edit.unanswered_items(ctx.job)
        order = uncertainty_order(ctx.job["id"], cal["version"], todo, cal, jitter=float(p.get("jitter", CAL_JITTER)),
                                  bucket_s=float(p.get("bucket_s", CAL_BUCKET_S)), penalty=float(p.get("penalty", CAL_BUCKET_PENALTY)))  # fmt: skip
        res = ctx.reorder(order, note=f"uncertainty around p_yes={cal['boundary_p_yes']:.4g} (calibration v{cal['version']})",
                          meta={"calibration_version": cal["version"], "n": cal["n"]})  # fmt: skip
        out.update(reordered=res["unanswered"], changed=res["changed"])
    return out


hooks.register_action("broadcast_calibrate", _broadcast_calibrate, kinds=("broadcast_review",))


# ---------------------------------------------------------------------------
# Per-match evaluation of the classifier against the verdicts (worst match first)
# ---------------------------------------------------------------------------
#
# ``pred`` / ``p_yes`` are the scores.npz values the selection was made from (video-decoded frames, stored on
# the job item and echoed into each verdict): they are NOT re-scored from the JPEGs, whose JPEG round trip
# flips calls near the boundary. Only the ``random`` stratum is an unbiased sample of the match, so it alone
# gives the headline accuracy (with a Wilson interval); the edge strata (``edge_mid`` / ``edge_boundary`` /
# ``edge_cal``) were picked for being uncertain, so their error count is reported separately, never pooled.
# ``unsure`` verdicts are excluded. The random stratum is stratified by predicted class (>= 25 % each), so it is
# class-balanced rather than strictly uniform: read the figure as "accuracy on a class-balanced random sample".

RANDOM_BUCKET = "random"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for ``k`` successes in ``n`` trials (None when ``n == 0``)."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def _pred_yes(rec: dict) -> bool | None:
    pred = rec.get("pred")
    if pred in ("Yes", "No"):
        return pred == "Yes"
    p = rec.get("p_yes")
    return p > 0.5 if isinstance(p, (int, float)) and not isinstance(p, bool) else None


def eval_match(rows: list[dict]) -> dict:
    """``rows`` = ``[{verdict, pred?, p_yes?, reason?}]`` (verdict None = not labelled yet) -> the match's numbers."""
    rnd_n = rnd_ok = edge_n = edge_wrong = unsure = no_pred = 0
    fp = fn = 0  # edge errors split: model said Yes but human No (fp) / the reverse (fn)
    for r in rows:
        v = r.get("verdict")
        if v == "unsure":
            unsure += 1
        if v not in ("yes", "no"):
            continue
        pred = _pred_yes(r)
        if pred is None:
            no_pred += 1
            continue
        ok = pred == (v == "yes")
        if r.get("reason") == RANDOM_BUCKET:
            rnd_n += 1
            rnd_ok += ok
        else:
            edge_n += 1
            if not ok:
                edge_wrong += 1
                fp += pred
                fn += not pred
    ci = wilson(rnd_ok, rnd_n)
    return {"random": {"n": rnd_n, "correct": rnd_ok, "acc": rnd_ok / rnd_n if rnd_n else None,
                       "ci": list(ci) if ci else None},
            "edge": {"n": edge_n, "wrong": edge_wrong, "false_yes": fp, "false_no": fn},
            "unsure": unsure, "no_pred": no_pred}  # fmt: skip


def match_split(match: str) -> str:
    """``train`` / ``heldout`` (or the raw label, ``unknown`` when the register cannot be read) from the one
    split authority (``footy_track.splits``; read-only here)."""
    try:
        from footy_track import splits  # noqa: PLC0415

        s = splits.split_label(match)
    except Exception:  # noqa: BLE001 - a metrics table must not crash on a missing register
        return "unknown"
    return "heldout" if s == "eval_heldout_match" else s


def pool_by_split(matches: list[dict]) -> list[dict]:
    """Pooled numbers per split: random-stratum accuracy over all the split's matches (Wilson 95 %), edge errors, counts."""
    out = []
    for split in sorted({m["split"] for m in matches}):
        ms_ = [m for m in matches if m["split"] == split]
        n = sum(m["random"]["n"] for m in ms_)
        k = sum(m["random"]["correct"] for m in ms_)
        ci = wilson(k, n)
        out.append({"split": split, "matches": len(ms_), "total": sum(m["total"] for m in ms_),
                    "labelled": sum(m["labelled"] for m in ms_),
                    "random": {"n": n, "correct": k, "acc": k / n if n else None, "ci": list(ci) if ci else None},
                    "edge": {"n": sum(m["edge"]["n"] for m in ms_), "wrong": sum(m["edge"]["wrong"] for m in ms_)}})
    return out


def match_summary(ds: datasets.Dataset) -> list[dict]:
    """One row per match of the dataset (every broadcast_review job's clips plus any verdict file), ranked
    worst-first by random-stratum accuracy; matches with no random verdict yet come last."""
    by_match: dict[str, dict] = {}
    for job in jobs.all_jobs():
        if job.get("kind") != "broadcast_review" or job.get("dataset") != ds.name or job.get("status") == "cancelled":
            continue
        for it in job["items"]:
            m = by_match.setdefault(it["clip"], {"items": {}, "jobs": []})
            m["items"].setdefault(it["frame"], it)
            if job["id"] not in m["jobs"]:
                m["jobs"].append(job["id"])
                m.setdefault("title", job["title"].removeprefix("Broadcast review: "))
    root = ds.marks_dir / "broadcast_review"
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        by_match.setdefault(path.stem, {"items": {}, "jobs": []})
    out = []
    for match, m in by_match.items():
        try:
            store = read_store(ds, match)
        except ValueError:
            continue
        rows, total = [], len(m["items"])
        for frame, it in m["items"].items():
            v = store.get(frame)
            rows.append({**it, "verdict": v["verdict"] if v else None})
        for frame, v in store.items():  # verdicts of frames no longer in any job still count
            if frame not in m["items"]:
                rows.append({**v, "verdict": v["verdict"]})
        e = eval_match(rows)
        out.append({"match": match, "split": match_split(match), "title": m.get("title") or match, "job_id": (m["jobs"] or [None])[0],
                    "total": total, "labelled": sum(1 for r in rows if r["verdict"]), **e})
    out.sort(key=lambda r: (r["random"]["acc"] is None,
                            r["random"]["acc"] if r["random"]["acc"] is not None else 0.0,
                            r["random"]["ci"][0] if r["random"]["ci"] else 0.0, r["match"]))
    return out


# ---------------------------------------------------------------------------
# Routes (the dataset comes from ?dataset= via the middleware)
# ---------------------------------------------------------------------------


@router.get("/broadcast_review", response_class=HTMLResponse)
async def page() -> HTMLResponse:
    return HTMLResponse((_WEB / "broadcast_review.html").read_text(), headers={"Cache-Control": "no-store, max-age=0"})


@router.get("/broadcast_review/api/summary")
async def summary_view():
    """Per-match accuracy of the classifier against the verdicts, worst first (see ``match_summary``)."""
    ds = datasets.current()
    matches = await asyncio.to_thread(match_summary, ds)
    return {"dataset": ds.name, "matches": matches, "pooled": pool_by_split(matches)}


@router.get("/broadcast_review/frame/{match}/{frame}.jpg")
async def frame_jpg(match: str, frame: int):
    try:
        p = frame_path(datasets.current(), match, frame)
    except ValueError:
        raise HTTPException(404) from None
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@router.get("/broadcast_review/api/job/{ref:path}")
async def job_view(ref: str):
    """The job's frames in order with ``done``/``verdict`` (+ the model's call, which the page shows alongside the frame)."""
    def work():
        job = jobs.resolve(ref)
        if job["kind"] != "broadcast_review":
            raise jobs.JobError(422, f"job kind {job['kind']!r} is not broadcast_review")
        ds = datasets.get(job["dataset"])
        stores: dict[str, dict] = {}
        cal = latest_calibration(ds)
        rows = []
        for i, it in enumerate(job["items"]):
            st = stores.get(it["clip"])
            if st is None:  # setdefault(k, read_store()) would read the file for every item
                st = stores[it["clip"]] = read_store(ds, it["clip"])
            v = st.get(it["frame"])
            row = {**it, "item": i, "verdict": v["verdict"] if v else None, "done": bool(v)}
            if cal is not None and isinstance(it.get("p_yes"), (int, float)):
                row["p_cal"] = calibrated_p(it["p_yes"], cal)
            rows.append(row)
        # ``rev`` changes when a hook reorders/appends: the page re-reads and keeps the user's place.
        return {"id": job["id"], "title": job["title"], "dataset": job["dataset"], "status": job["status"],
                "rev": int(job.get("rev") or 0), "calibration": cal,
                "total": len(rows), "done": sum(r["done"] for r in rows), "frames": rows}  # fmt: skip

    try:
        return await asyncio.to_thread(work)
    except jobs.JobError as e:
        return JSONResponse({"error": e.detail}, status_code=e.status)


@router.put("/broadcast_review/api/verdict/{match}/{frame}")
async def put_verdict(match: str, frame: int, body: dict):
    try:
        row = await asyncio.to_thread(set_verdict, datasets.current(), match, frame, body.get("verdict"), body)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"ok": True, "row": row}


@router.delete("/broadcast_review/api/verdict/{match}/{frame}")
async def delete_verdict(match: str, frame: int):
    try:
        await asyncio.to_thread(set_verdict, datasets.current(), match, frame, None)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"ok": True}
