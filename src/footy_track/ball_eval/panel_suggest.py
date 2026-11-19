"""Precompute detector suggestions for the evaluation panel's frames (``panel.py suggest``).

George's ask: even for eval labelling the page should show a detector box he can accept or replace. The panel
stays valid as an evaluation set because (a) a random ``BLIND_FRAC`` (25 %) of its windows never get a
suggestion (``panel.blind_windows``) and every QA card is blind, so blind-vs-shown labels can be compared for
systematic bias (``panel_score``), and (b) an accepted box carries the audit marker ``accepted_suggestion`` plus
the detector sha8. An accepted box is still ordinary tier-1 ``labeller`` GT; the marker is for audit only.

HELDOUT-DATA RULE. Heldout matches never enter training. Running the production detector on a heldout panel
frame is eval-only inference (nothing is trained, mined or cached into a training pool), so it is allowed;
:func:`suggest_match` still refuses a match that is not registered heldout.

Output ``<panel_dir>/<match>.suggest.<detector_sha8>.json``::

    {"<clip>:<frame>": {"x": .., "y": .., "w": .., "h": .., "conf": ..} | null, ...}

normalised top-left xywh of the TOP-1 ball candidate (highest confidence, floor ``candidates.CONF``), ``null``
when the detector found no ball candidate. Every non-QA frame of every panel window has a key. A
``<match>.suggest-meta.<sha8>.json`` records the detector, timings and the panel frames that were already
labelled when the file appeared (labels placed before suggestions existed are blind-equivalent).

Speed: 1280 px RT-DETR on CPU is ~2 s / frame, ~20 min / match. Resumable per clip (``.partial``).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from footy_track.ball_eval import panel as P  # noqa: N812

IMGSZ = 1280
BATCH = 4


def frame_key(clip: str, frame: int) -> str:
    return f"{clip}:{int(frame)}"


def top1(balls) -> dict | None:
    """Highest-confidence row of an ``(n, 5)`` x, y, w, h, conf array as a dict (None when empty)."""
    if balls is None or len(balls) == 0:
        return None
    r = max(balls, key=lambda row: float(row[4]))
    return {k: round(float(v), 6) for k, v in zip(("x", "y", "w", "h", "conf"), r, strict=True)}


def panel_frames(doc: dict) -> dict[str, list[int]]:
    """``{clip: sorted local frames}`` of the panel windows (QA cards are blind: no suggestion needed)."""
    out: dict[str, set[int]] = {}
    for w in doc["windows"]:
        out.setdefault(w["clip"], set()).update(int(f) for f in w["frames"])
    return {c: sorted(f) for c, f in sorted(out.items())}


def _read_frames(video: Path, wanted: list[int]):
    """Yield ``(frame, bgr)`` for the wanted frames by sequential decode (frame-exact; seeking is not)."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise P.PanelError(f"cannot open {video}")
    want = set(wanted)
    last = max(wanted)
    try:
        i = 0
        while i <= last and cap.grab():
            if i in want:
                ok, bgr = cap.retrieve()
                if ok:
                    yield i, bgr
            i += 1
    finally:
        cap.release()


def suggest_match(
    match: str,
    *,
    panel_version: str = P.PANEL_VERSION,
    root: Path | None = None,
    weights: Path | None = None,
    device: str = "cpu",
    imgsz: int = IMGSZ,
    batch: int = BATCH,
    detector=None,
    labelled_keys: Callable[[], set[str]] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Detect the top-1 ball on every panel frame of ``match`` and write the suggest file. Resumable; an
    existing finished file for the same detector is left alone (status ``exists``)."""
    from footy_track.ball_mining import candidates as C  # noqa: N812, PLC0415

    P.require_heldout(match)
    doc = P.read_panel(P.panel_path(match, panel_version, root))
    if weights is None:
        from footy_track.detectors.ultralytics import current_best_detector_path  # noqa: PLC0415

        weights = current_best_detector_path()
    sha8 = C.sha8(C.sha256_file(weights))
    out = P.suggest_path(match, sha8, panel_version, root)
    if out.is_file():
        return {"match": match, "status": "exists", "path": str(out), "detector_sha8": sha8}
    partial = out.with_name(out.name + ".partial")
    sugg: dict[str, dict | None] = {}
    if partial.is_file():
        sugg = json.loads(partial.read_text())
    seg_dir = Path(doc["segments_json"]).parent
    files = {Path(s["file"]).stem: seg_dir / s["file"] for s in P.load_segments(Path(doc["segments_json"]))["segments"]}
    frames = panel_frames(doc)
    total = sum(len(v) for v in frames.values())
    if detector is None:
        detector = C.make_detector(weights, imgsz=imgsz, conf=C.CONF, device=device)
    t0 = time.monotonic()
    done_now = 0
    for clip, fl in frames.items():
        todo = [f for f in fl if frame_key(clip, f) not in sugg]
        if not todo:
            continue
        pending: list[tuple[int, object]] = []

        def flush(pending=pending, clip=clip) -> None:
            nonlocal done_now
            if not pending:
                return
            results = detector.predict_frames([b for _, b in pending])
            C.limit_threads()  # ultralytics' first cpu predict() resets torch threads
            for (f, _), res in zip(pending, results, strict=True):
                balls, _ = C.rows_of(res.detections)
                sugg[frame_key(clip, f)] = top1(balls)
            done_now += len(pending)
            pending.clear()

        for f, bgr in _read_frames(files[clip], todo):
            pending.append((f, bgr))
            if len(pending) >= batch:
                flush()
        flush()
        for f in todo:  # frames the decoder never reached (clip shorter than the panel says)
            sugg.setdefault(frame_key(clip, f), None)
        P._atomic_write(partial, sugg, exclusive=False)  # noqa: SLF001
        el = time.monotonic() - t0
        log(f"[suggest] {match} {len(sugg)}/{total} frames, {el:.0f}s ({el / max(done_now, 1):.2f} s/frame)")
    elapsed = time.monotonic() - t0
    pre = sorted(labelled_keys()) if labelled_keys else []
    meta = {
        "schema": "ball-eval-panel-suggest/v1",
        "match": match,
        "panel_version": panel_version,
        "created_at": P.now_iso(),
        "detector": {"sha8": sha8, "sha256": C.sha256_file(weights), "weights": str(weights), "imgsz": imgsz, "conf_floor": C.CONF},
        "device": device,
        "threads": C.thread_cap(),
        "n_frames": total,
        "n_with_suggestion": sum(1 for v in sugg.values() if v),
        "seconds_this_run": round(elapsed, 1),
        "seconds_per_frame": round(elapsed / done_now, 3) if done_now else None,
        "blind_windows": sorted(P.blind_windows(doc)),
        "pre_labelled": pre,
        "note": "pre_labelled = panel frames already labelled when this file was written: blind-equivalent labels",
    }
    P._atomic_write(P.suggest_meta_path(out), meta, exclusive=False)  # noqa: SLF001
    P._atomic_write(out, sugg, exclusive=True, sort_keys=False)  # noqa: SLF001
    partial.unlink(missing_ok=True)
    meta["status"] = "created"
    meta["detector_sha8"] = sha8
    meta["path"] = str(out)
    return meta


def labelled_panel_keys(doc: dict, root: Path | None = None) -> set[str]:
    """Panel frames (``clip:frame``) that already have a human label in the panel's own marks dir (read only)."""
    from footy_track.ball_eval.panel_score import load_human_sidecars  # noqa: PLC0415

    human = load_human_sidecars(
        P.marks_dir(doc["panel_version"], root), {w["clip"] for w in doc["windows"]}
    )
    want = {frame_key(w["clip"], f) for w in doc["windows"] for f in w["frames"]}
    return {frame_key(c, f) for (c, f) in human if frame_key(c, f) in want}


def load_suggestions(path: Path) -> dict[str, dict | None]:
    return json.loads(Path(path).read_text())


def thread_env() -> dict:
    return {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "BALL_POOL_THREADS")}
