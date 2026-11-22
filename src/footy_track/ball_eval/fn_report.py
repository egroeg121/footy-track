"""False-negative report of an eval job (design ``ball_anchor_iteration.md`` task 10, section 4.1; George 2026-11-22).

Read from the FINAL terminal states of the eval selected set F (``selected_frame`` of the job key joined with the ball
``frame_label_state``): every frame of F is ``ball``, ``no_ball``, ``unusable``, ``deferred`` or still ``outstanding``. The report
is only ``complete`` when nothing is outstanding; an incomplete one says so and lists the frames, it never counts them as misses.

For every scored model VERSION (one ``--model label=path`` per version) a ball frame is a **miss** when the human placed a ball
and the model has no detection within ``r_px`` (default 20) of the human centre. It is split into ``no_detection`` (the model had
nothing on the frame) and ``off`` (it had detections, all farther than ``r_px``). Models are scored on the same frames, so the
versions are directly comparable; nothing is merged across versions.

Human ball centres are the job's own anchors (``anchor_search.load_anchors``: sidecar tier-1 boxes and ``ball`` / ``corrected``
verdicts). A terminal ``ball`` frame with no recoverable centre is counted separately (``ball_without_centre``), never as a hit
or miss. The counts are raw frames; weight them by the panel's ``pi_frame`` before quoting a population number
(``panel_score.score`` does that for the headline). Output is ``ball-eval-fn-report/v1`` JSON; the candidate source recorded on
the job's cards is echoed so the report reads together with the version that pre-filled the cards.

CLI::

    python -m footy_track.ball_eval.fn_report --job REF --db feature.duckdb --model v3=preds_v3.jsonl [--model ...] [--out report.json]

Prediction rows: ``{clip, frame, x, y}`` (+ optional ``w`` / ``h`` / ``score``), pixels by default (``--norm`` for normalised);
several rows per frame are all detections.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.clip_id import canonical_clip_id

SCHEMA = "ball-eval-fn-report/v1"
R_PX = 20.0
TERMINAL = ("ball", "no_ball", "unusable", "deferred")
LIST_CAP = 200

Key = tuple[str, int]
Dets = Mapping[Key, list[tuple[float, float]]]  # pixel centres per frame


def load_detections(path: Path, *, norm: bool = False, size: tuple[int, int] = A.FRAME_SIZE) -> dict[Key, list[tuple[float, float]]]:
    """All detections per frame from a jsonl (``x`` / ``y`` = centre; null x = no detection)."""
    out: dict[Key, list[tuple[float, float]]] = {}
    for raw in Path(path).read_text().splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        if d.get("x") is None or d.get("y") is None:
            continue
        x, y = float(d["x"]), float(d["y"])
        if norm:
            x, y = x * size[0], y * size[1]
        out.setdefault((canonical_clip_id(d["clip"]), int(d["frame"])), []).append((x, y))
    return out


def terminal_states(store, frames: Iterable[Key], *, game_of_clip=lambda c: c) -> dict[Key, str]:
    by_clip: dict[str, list[int]] = {}
    for c, f in frames:
        by_clip.setdefault(c, []).append(f)
    out: dict[Key, str] = {}
    for c, fs in by_clip.items():
        out.update({(c, f): s for f, s in store.read_ball_terminal_states(game_of_clip(c), fs).items()})
    return out


def methods(store, frames: Iterable[Key], *, game_of_clip=lambda c: c) -> dict[Key, str]:
    """``frame_label_state.method`` of the ball row per frame (``verdict`` = found by a card, ``hand_frame`` = by the full-frame check)."""
    out: dict[Key, str] = {}
    want = set(frames)
    for c in {c for c, _ in want}:
        df = store.read_frame_label_states(game_of_clip(c), "ball")
        for r in df.itertuples(index=False):
            if (c, int(r.frame_index)) in want:
                out[(c, int(r.frame_index))] = str(r.method)
    return out


def candidate_sources(job: Mapping) -> list[dict]:
    """Distinct ``candidate_source`` blocks recorded on the job's cards (sampling or search block)."""
    seen: dict[str, dict] = {}
    for it in job.get("items") or []:
        for blk in ((it.get("search") or {}).get("candidate_source"), (it.get("sampling") or {}).get("candidate_source")):
            if isinstance(blk, dict):
                seen.setdefault(json.dumps(blk, sort_keys=True), blk)
    return list(seen.values())


def _nearest(dets: list[tuple[float, float]], cx: float, cy: float) -> float | None:
    return min((math.hypot(x - cx, y - cy) for x, y in dets), default=None)


def build_report(
    job: Mapping,
    *,
    selected: Iterable[Key],
    states: Mapping[Key, str],
    centres: Mapping[Key, tuple[float, float]],
    models: Mapping[str, Dets],
    window_of: Mapping[Key, str] | None = None,
    by_method: Mapping[Key, str] | None = None,
    r_px: float = R_PX,
    size: tuple[int, int] = A.FRAME_SIZE,
) -> dict:
    """The report document. ``centres`` are normalised human ball centres; ``models`` maps a version label to its detections."""
    F = sorted(set(selected))
    counts: Counter[str] = Counter(states.get(k, "outstanding") for k in F)
    outstanding = [k for k in F if k not in states]
    ball = [k for k in F if states.get(k) == "ball"]
    located = [k for k in ball if k in centres]
    rep: dict = {
        "schema": SCHEMA, "job": job.get("id"), "job_key": job.get("key"), "r_px": r_px, "n_selected": len(F),
        "complete": not outstanding, "by_state": {s: counts.get(s, 0) for s in (*TERMINAL, "outstanding")},
        "outstanding": [{"clip": c, "frame": f} for c, f in outstanding[:LIST_CAP]],
        "ball_without_centre": len(ball) - len(located), "n_ball_frames": len(located),
        "found_by": dict(Counter("check" if (by_method or {}).get(k) == "hand_frame" else "card" for k in ball)) if by_method else None,
        "candidate_sources": candidate_sources(job), "versions": {},
    }  # fmt: skip
    for label in sorted(models):
        dets = models[label]
        missed, no_det = [], 0
        for k in located:
            cx, cy = centres[k][0] * size[0], centres[k][1] * size[1]
            d = _nearest(dets.get(k, []), cx, cy)
            if d is None or d > r_px:
                no_det += d is None
                missed.append({"clip": k[0], "frame": k[1], "window_id": (window_of or {}).get(k), "nearest_px": None if d is None else round(d, 2)})
        n = len(located)
        rep["versions"][label] = {
            "n_ball_frames": n, "n_missed": len(missed), "missed_no_detection": no_det, "missed_off": len(missed) - no_det,
            "recall": round(1 - len(missed) / n, 6) if n else None, "miss_rate": round(len(missed) / n, 6) if n else None,
            "missed_frames": missed[:LIST_CAP], "missed_frames_truncated": len(missed) > LIST_CAP,
        }  # fmt: skip
    return rep


def report_for_job(job: dict, store, models: Mapping[str, Dets], *, marks_dir: Path | None = None, r_px: float = R_PX, size: tuple[int, int] = A.FRAME_SIZE) -> dict:
    """Assemble the inputs from the store, the job and the marks, then :func:`build_report`."""
    selected = A.resolve_selected(job, store)
    anchors = A.load_anchors(A.job_with_selected(job, selected), marks_dir=marks_dir).anchors
    window_of: dict[Key, str] = {}
    df = store.read_selected_frames(job_key=job.get("key"))
    for r in df.itertuples(index=False):
        if r.window_id:
            window_of[(canonical_clip_id(r.game_id), int(r.frame_index))] = r.window_id
    states = terminal_states(store, selected)
    return build_report(
        job, selected=selected, states=states, centres={k: (a.cx, a.cy) for k, a in anchors.items()}, models=models,
        window_of=window_of, by_method=methods(store, [k for k, s in states.items() if s == "ball"]), r_px=r_px, size=size,
    )  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fn_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--job", required=True, help="eval job id or title")
    ap.add_argument("--db", default=None, help="feature store path (default: the shared store)")
    ap.add_argument("--model", action="append", required=True, help="label=path.jsonl, one per scored model version")
    ap.add_argument("--norm", action="store_true", help="predictions are normalised")
    ap.add_argument("--radius", type=float, default=R_PX)
    ap.add_argument("--marks", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    from footy_track.feature_store import FeatureStore  # noqa: PLC0415
    from footy_track.feature_store.lock import default_db_path  # noqa: PLC0415
    from footy_track.labeller import jobs as jobs_mod  # noqa: PLC0415
    from footy_track.labeller import jobs_client  # noqa: PLC0415

    src = jobs_client.get(a.job)
    job = jobs_mod.read_job(src["id"]) or src
    models = {}
    for spec in a.model:
        label, _, path = spec.partition("=")
        models[label] = load_detections(Path(path), norm=a.norm)
    store = FeatureStore.open(a.db or default_db_path())
    try:
        rep = report_for_job(job, store, models, marks_dir=a.marks, r_px=a.radius)
    finally:
        store.close()
    text = json.dumps(rep, indent=2)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0 if rep["complete"] else 1  # non-zero = the eval set is not finished: the numbers are provisional


if __name__ == "__main__":
    sys.exit(main())
