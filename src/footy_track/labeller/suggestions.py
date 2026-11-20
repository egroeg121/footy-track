"""Detector suggestions for ``ball_place`` job cards (eval panel): ``GET /api/jobs/<id>/suggestion?clip=&frame=``.

A suggestion is **not a label**: the page draws it as a dashed overlay and only an explicit accept (key ``Y``)
saves it, as an ordinary human box tagged ``[in_play_ball, labeller, accepted_suggestion, suggest_det:<sha8>]``.

Where it comes from. The job item carries ``sampling = {panel, match, window_id, suggest, qa, ...}``. The
suggestion is read from ``<FD>/ball_eval_panel/<panel>/<match>.suggest.<sha8>.json`` (``panel_suggest.py``). Whether
the card may show one is ``sampling.suggest`` (``blind`` | ``shown``); items of jobs created before that flag
existed carry none, so it is derived at request time from the panel file by the same seeded rule that
``panel.job_items`` uses (``panel.blind_windows``): the live jobs need no mutation. QA cards are always blind.

No suggestion (blind card, no candidate, no suggest file, any other kind of job) answers ``{"suggestion": null}``;
the page then shows nothing. "No ball" is never suggested.
"""

from __future__ import annotations

import json
from pathlib import Path

from footy_track import splits
from footy_track.ball_eval import panel as P  # noqa: N812
from footy_track.labeller import jobs

_CACHE: dict[str, tuple[tuple, object]] = {}


def _cached(path: Path, parse):
    try:
        st = path.stat()
    except OSError:
        return None
    sig = (st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(str(path))
    if hit and hit[0] == sig:
        return hit[1]
    try:
        val = parse(path)
    except (OSError, ValueError):
        return None
    _CACHE[str(path)] = (sig, val)
    return val


def _item_for(job: dict, clip: str, frame: int) -> dict | None:
    for it in job.get("items") or []:
        if it.get("clip") == clip and frame in jobs._item_units(job["kind"], it):  # noqa: SLF001
            return it
    return None


def suggestion_for(job: dict, clip: str, frame: int, root: Path | None = None) -> dict:
    out: dict = {"clip": clip, "frame": frame, "suggestion": None}
    if job.get("kind") != "ball_place":
        return out
    item = _item_for(job, clip, frame)
    sampling = (item or {}).get("sampling")
    root = root or splits.data_root()
    if not isinstance(sampling, dict) or not sampling.get("panel") or not sampling.get("match"):
        return _job_suggestion(job, out, root) if item is not None else out
    version, match = sampling["panel"], sampling["match"]
    blind: set[str] | None = None
    if sampling.get("suggest") not in (P.SUGGEST_BLIND, P.SUGGEST_SHOWN) and not sampling.get("qa"):
        doc = _cached(P.panel_path(match, version, root), P.read_panel)
        blind = P.blind_windows(doc) if isinstance(doc, dict) else None
    flag = P.suggest_flag(sampling, blind)
    out["suggest"] = flag
    if flag != P.SUGGEST_SHOWN:
        return out
    sf = P.find_suggest_file(match, version, root)
    if sf is None:
        return out
    table = _cached(sf, lambda p: json.loads(p.read_text()))
    box = (table or {}).get(f"{clip}:{frame}")
    if isinstance(box, dict) and all(k in box for k in ("x", "y", "w", "h")):
        sha8 = sf.name.split(".suggest.")[1].removesuffix(".json")
        out["suggestion"] = {**{k: float(box[k]) for k in ("x", "y", "w", "h")}, "conf": float(box.get("conf", 0.0)), "detector": sha8}
    return out


def _job_suggestion(job: dict, out: dict, root: Path) -> dict:
    """Non-panel ``ball_place`` job (LAB-1479): the job-keyed file ``<FD>/job_suggestions/<job_id>.suggest.<sha8>.json``
    (``ball_eval/job_suggest.py``). No blind control: every frame with a candidate is offered, except frames that were
    already labelled when the file was written."""
    from footy_track.ball_eval import job_suggest as J  # noqa: N812, PLC0415

    sf = J.find_suggest_file(str(job.get("id")), root)
    if sf is None:
        return out
    doc = _cached(sf, lambda p: json.loads(p.read_text()))
    if not isinstance(doc, dict):
        return out
    key = f"{out['clip']}:{out['frame']}"
    box = (doc.get("boxes") or {}).get(key)
    if key in set(doc.get("pre_labelled") or []) or not (isinstance(box, dict) and all(k in box for k in ("x", "y", "w", "h"))):
        return out
    sha8 = sf.name.split(".suggest.")[1].removesuffix(".json")
    out["suggest"] = P.SUGGEST_SHOWN
    out["suggestion"] = {**{k: float(box[k]) for k in ("x", "y", "w", "h")}, "conf": float(box.get("conf", 0.0)), "detector": sha8}
    return out


def _entry_cands(entry: dict | None) -> list[list[float]]:
    """Compact ``[x, y, w, h, conf, tid]`` rows of a file entry: its ``cands`` (top-k), else the top-1 alone (old files)."""
    if not isinstance(entry, dict):
        return []
    rows = entry.get("cands") or [entry]
    return [[round(float(c[k]), 4) for k in ("x", "y", "w", "h", "conf")] + [int(c.get("tid", 0))]
            for c in rows if all(k in c for k in ("x", "y", "w", "h"))]


def candidates_for_job(job: dict, root: Path | None = None) -> dict:
    """Everything the page needs to roll suggestions forward in the browser with no further request (LAB-1479):
    ``{"frames": {"clip:frame": [[x,y,w,h,conf,tid], ...]}, "size": {clip: [w,h]}, "detector": {clip: sha8}}``.
    Same rules as :func:`suggestion_for`: panel items honour their blind flag (blind items are absent), job-keyed files
    carry no blind set; frames already labelled when a job file was written are absent. One request per job."""
    out: dict = {"job_id": job.get("id"), "frames": {}, "size": {}, "detector": {}}
    if job.get("kind") != "ball_place":
        return out
    root = root or splits.data_root()
    from footy_track.ball_eval import job_suggest as J  # noqa: N812, PLC0415

    jf = J.find_suggest_file(str(job.get("id")), root)
    jdoc = _cached(jf, lambda p: json.loads(p.read_text())) if jf else None
    for it in job.get("items") or []:
        sampling = it.get("sampling")
        pre: set = set()
        if isinstance(sampling, dict) and sampling.get("panel") and sampling.get("match"):
            version, match = sampling["panel"], sampling["match"]
            blind = None
            if sampling.get("suggest") not in (P.SUGGEST_BLIND, P.SUGGEST_SHOWN) and not sampling.get("qa"):
                doc = _cached(P.panel_path(match, version, root), P.read_panel)
                blind = P.blind_windows(doc) if isinstance(doc, dict) else None
            if P.suggest_flag(sampling, blind) != P.SUGGEST_SHOWN:
                continue
            sf = P.find_suggest_file(match, version, root)
            table = _cached(sf, lambda p: json.loads(p.read_text())) if sf else None
            meta = _cached(P.suggest_meta_path(sf), lambda p: json.loads(p.read_text())) if sf else None
            size = (meta or {}).get("size")
            sha8 = sf.name.split(".suggest.")[1].removesuffix(".json") if sf else ""
        elif isinstance(jdoc, dict):
            table, pre = jdoc.get("boxes"), set(jdoc.get("pre_labelled") or [])
            size, sha8 = jdoc.get("size"), jf.name.split(".suggest.")[1].removesuffix(".json")
        else:
            continue
        if not table:
            continue
        for f in jobs._item_units(job["kind"], it):  # noqa: SLF001
            key = f"{it['clip']}:{f}"
            if key in pre:
                continue
            if rows := _entry_cands(table.get(key)):
                out["frames"][key] = rows
        if size:
            out["size"][it["clip"]] = list(size)
        out["detector"][it["clip"]] = sha8
    return out
