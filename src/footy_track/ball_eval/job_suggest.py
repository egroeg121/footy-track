"""Detector suggestions for ANY ``ball_place`` job (LAB-1479), keyed by job id, not by eval panel.

A dashed suggested box on every frame with a detector candidate; ``Y`` accepts it as George's own human box tagged
``accepted_suggestion`` (audit only, still an ordinary tier-1 label), drawing replaces it. Unlike the eval panel there
is NO blind control here (George's call): every frame with a candidate is offered. Frames already labelled when the
file was written (``pre_labelled``) are left as they are and never get a suggestion served.

HELDOUT-DATA RULE. Eval-only inference: :func:`suggest_job` refuses a job with any clip outside a registered heldout
match; nothing is trained, mined or cached into a training pool.

File ``<FD>/job_suggestions/<job_id>.suggest.<detector_sha8>.json``::

    {"schema": "job-suggest/v1", "job_id", "detector": {...}, "size": [w, h], "pre_labelled": ["<clip>:<frame>", ...],
     "boxes": {"<clip>:<frame>": {x,y,w,h,conf,tid, "cands": [{x,y,w,h,conf,tid} x<=5]} | null}}

``boxes`` has a key for every frame of every item (null = no candidate); the entry is the top-1 plus up to 5
candidates (conf >= 0.1) with offline ``tid`` tracklet ids per window (``tracklets.py``). Never writes the job, its sidecars or the marks.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from footy_track import match_splits, splits
from footy_track.ball_eval import panel as P  # noqa: N812
from footy_track.ball_eval import panel_suggest as PS  # noqa: N812
from footy_track.ball_eval.panel_score import BALL, Human, load_human_sidecars
from footy_track.labeller import datasets, jobs

SCHEMA = "job-suggest/v1"


def suggest_dir(root: Path | None = None) -> Path:
    return (root or splits.data_root()) / "job_suggestions"


def suggest_path(job_id: str, sha8: str, root: Path | None = None) -> Path:
    return suggest_dir(root) / f"{job_id}.suggest.{sha8}.json"


def find_suggest_file(job_id: str, root: Path | None = None) -> Path | None:
    found = sorted(suggest_dir(root).glob(f"{job_id}.suggest.*.json"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def window_id(item: dict) -> str:
    return f"{item['clip']}:{item.get('frame_start', item.get('frame', 0))}"


def job_frames(job: dict) -> dict[str, list[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for it in job.get("items") or []:
        out[it["clip"]].update(int(f) for f in jobs._item_units(job["kind"], it))  # noqa: SLF001
    return {c: sorted(f) for c, f in sorted(out.items())}


def require_heldout_job(job: dict) -> None:
    bad = sorted(c for c in job_frames(job) if not match_splits.is_heldout_clip(c))
    if bad:
        raise P.PanelError(f"job {job['id']} has clips outside a registered heldout match ({bad[:3]}); eval-only inference refused")


def job_human(job: dict) -> dict[tuple[str, int], Human]:
    """Tier-1 human labels of the job's frames (read only), via the same sidecar parser as the panel scorer."""
    ds = datasets.get(job.get("dataset"))
    human: dict[tuple[str, int], Human] = {}
    for clip, frames in job_frames(job).items():
        stem = jobs._gt_stem(ds.marks_dir, clip)  # noqa: SLF001
        h = load_human_sidecars(ds.marks_dir, [stem])
        human.update({(clip, f): v for (c, f), v in h.items() if c == stem and f in set(frames)})
    return human


def _video(job: dict, clip: str) -> Path:
    ds = datasets.get(job.get("dataset"))
    for d in (ds.clips_dir, *ds.extra_clips_dirs):
        p = Path(d) / f"{clip}.mp4"
        if p.exists():
            return p
    raise P.PanelError(f"no video for clip {clip!r} in dataset {ds.name!r}")


def suggest_job(
    job_id: str,
    *,
    root: Path | None = None,
    suggester: PS.Suggester | None = None,
    weights: Path | None = None,
    device: str = "cpu",
    batch: int = PS.BATCH,
    detector=None,
    log: Callable[[str], None] = print,
) -> dict:
    """Top-1 RT-DETR box of every frame of the job's items. Resumable per clip (``.partial``); an existing finished
    file of the same detector is left alone."""
    job = jobs.read_job(job_id)
    if job is None or job.get("kind") != "ball_place":
        raise P.PanelError(f"{job_id!r} is not a ball_place job")
    require_heldout_job(job)
    if suggester is None:
        suggester = PS.RtDetrSuggester(weights, device=device, batch=batch, detector=detector)
    out = suggest_path(job_id, suggester.ident, root)
    if out.is_file():
        return {"job_id": job_id, "status": "exists", "path": str(out), "detector_sha8": suggester.ident}
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    boxes: dict[str, dict | None] = json.loads(partial.read_text()) if partial.is_file() else {}
    pre = sorted(PS.frame_key(c, f) for (c, f) in job_human(job))
    frames = job_frames(job)
    total = sum(len(v) for v in frames.values())
    t0, done_now = time.monotonic(), 0
    for clip, fl in frames.items():
        todo = [f for f in fl if PS.frame_key(clip, f) not in boxes]
        if not todo:
            continue
        got = suggester.suggest(clip, _video(job, clip), todo)
        for f in todo:
            boxes[PS.frame_key(clip, f)] = got.get(f)
        done_now += len(todo)
        P._atomic_write(partial, boxes, exclusive=False)  # noqa: SLF001
        el = time.monotonic() - t0
        log(f"[job-suggest] {job_id} {len(boxes)}/{total} frames, {el:.0f}s ({el / max(done_now, 1):.2f} s/frame)")
    elapsed = time.monotonic() - t0
    from footy_track.ball_eval import tracklets as T  # noqa: N812, PLC0415

    size = T.video_size(_video(job, next(iter(frames)))) if frames else T.DEFAULT_SIZE
    wins = [[(it["clip"], int(f)) for f in jobs._item_units(job["kind"], it)] for it in job["items"]]  # noqa: SLF001
    T.add_tracklets(boxes, wins, size)
    doc = {
        "schema": SCHEMA,
        "job_id": job_id,
        "created_at": P.now_iso(),
        "detector": suggester.info,
        "size": list(size),
        "pre_labelled": pre,
        "note": "pre_labelled = frames already labelled when this file was written; no suggestion is served for them",
        "n_frames": total,
        "n_with_suggestion": sum(1 for v in boxes.values() if v),
        "seconds_this_run": round(elapsed, 1),
        "seconds_per_frame": round(elapsed / done_now, 3) if done_now else None,
        "boxes": boxes,
    }
    P._atomic_write(out, doc, exclusive=True, sort_keys=False)  # noqa: SLF001
    partial.unlink(missing_ok=True)
    return {k: v for k, v in doc.items() if k not in ("boxes", "pre_labelled")} | {
        "status": "created", "detector_sha8": suggester.ident, "path": str(out), "n_pre_labelled": len(pre)}


def coverage(job: dict, doc: dict) -> dict[str, dict]:
    """Per stratum: frames, share with a suggestion, mean conf of those."""
    acc: dict[str, dict] = defaultdict(lambda: {"frames": 0, "n": 0, "conf": 0.0})
    for it in job["items"]:
        for f in jobs._item_units(job["kind"], it):  # noqa: SLF001
            a = acc[str(it.get("stratum") or "")]
            a["frames"] += 1
            if box := doc["boxes"].get(PS.frame_key(it["clip"], int(f))):
                a["n"] += 1
                a["conf"] += float(box.get("conf", 0.0))
    return {s: {"frames": a["frames"], "coverage": round(a["n"] / a["frames"], 4),
                "mean_conf": round(a["conf"] / a["n"], 4) if a["n"] else None} for s, a in sorted(acc.items())}


def report(job_id: str, *, root: Path | None = None, r_px: float = 10.0, size: tuple[int, int] = (1920, 1080)) -> dict:
    """Per stratum and ALL, on the job's labels: share of ball labels that were accepted suggestions, and agreement
    (label centre within ``r_px`` of RT-DETR's top-1) overall and among hand-drawn labels only, plus RT-DETR's miss rate, and how often an accept was #1 vs #2-5 (``rank1`` / ``rank2_5``: cheap evidence of how often the
    right ball is among the candidates but not the pick).
    Accepted boxes are ordinary tier-1 labels and stay in every number. No blind control, so no bias comparison."""
    job = jobs.read_job(job_id)
    sf = find_suggest_file(job_id, root)
    if job is None or sf is None:
        raise P.PanelError(f"no job or no suggest file for {job_id!r}")
    doc = json.loads(sf.read_text())
    human = job_human(job)
    cells: dict[str, dict] = defaultdict(lambda: dict.fromkeys(
        ("ball_labels", "accepted", "hand", "agree", "agree_hand", "rtdetr_miss", "no_ball", "rank1", "rank2_5"), 0))
    for it in job["items"]:
        for f in (int(x) for x in jobs._item_units(job["kind"], it)):  # noqa: SLF001
            h = human.get((it["clip"], f))
            if h is None:
                continue
            for key in (str(it.get("stratum") or ""), "ALL"):
                c = cells[key]
                if h.state != BALL:
                    c["no_ball"] += h.state == "no_ball"
                    continue
                c["ball_labels"] += 1
                c["accepted"] += h.accepted
                c["rank1"] += h.accepted and h.rank in (None, 1)  # old accepts (no rank marker) were the single #1
                c["rank2_5"] += h.accepted and h.rank not in (None, 1)
                c["hand"] += not h.accepted
                box = doc["boxes"].get(PS.frame_key(it["clip"], f))
                if box is None:
                    c["rtdetr_miss"] += 1
                    continue
                bx, by = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
                d = min(math.hypot((bx - cx) * size[0], (by - cy) * size[1]) for cx, cy in h.centres)
                c["agree"] += d <= r_px
                c["agree_hand"] += (d <= r_px) and not h.accepted

    def share(a, n):
        return round(a / n, 4) if n else None

    return {"job_id": job_id, "suggest_file": sf.name, "r_px": r_px, "image_size": list(size),
            "coverage": coverage(job, doc),
            "strata": {st: {**c, "share_accepted": share(c["accepted"], c["ball_labels"]),
                            "share_rank1_of_accepts": share(c["rank1"], c["accepted"]),
                            "agree_rtdetr": share(c["agree"], c["ball_labels"]),
                            "agree_rtdetr_hand_only": share(c["agree_hand"], c["hand"]),
                            "rtdetr_miss_rate": share(c["rtdetr_miss"], c["ball_labels"])}
                       for st, c in sorted(cells.items())}}
