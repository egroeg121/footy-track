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
    if not isinstance(sampling, dict) or not sampling.get("panel") or not sampling.get("match"):
        return out
    root = root or splits.data_root()
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
