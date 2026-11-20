"""Pre-fill machine player / referee boxes on labelled panel frames, and queue Player Check for them.

George has hand-labelled the ball on the eval-panel frames; this makes those same frames *fully* labelled
(``docs/design/full_frame_labelling.md`` sections 1 and 4c). Two steps, both CLI subcommands::

    python -m footy_track.ball_eval.panel_players prefill --dataset ball_eval_v1 --match M [--match M2 ...]
    python -m footy_track.ball_eval.panel_players job     --dataset ball_eval_v1 --match M

``prefill`` runs the production RT-DETR (1280 px, CPU) on exactly the panel's window frames and appends
``player`` / ``referee`` boxes (confidence >= ``PERSON_CONF`` = 0.3, the floor ``ball_mining.candidates`` uses
for person boxes) to the dataset's sidecars as MACHINE rows::

    {"frame_index": 500, "bbox": {x,y,w,h}, "center": {x,y}, "tags": ["player", "yolo", "prefill:rtdetr"]}

* Provenance tag is ``yolo``: the machine tier every reader already understands (ingest: source ``yolo``,
  unreviewed; the labeller restores it as machine, not GT). ``rtdetr`` is NOT a recognised provenance tag:
  ``Session`` would restore such a row as ``labeller`` ground truth. The trailing ``prefill:rtdetr`` marker
  names the writer; ingest and the labeller ignore unknown trailing markers, and this module uses it to find
  its own rows again.
* Append-or-replace-own-rows only. Ball boxes, ``no_ball`` / ``not_broadcast`` / ``players_complete`` /
  ``accepted_suggestion`` lines and every other box are copied byte for byte. Re-running replaces only the rows
  carrying the marker, on the target frames.
* A frame is skipped (reported, untouched) when it is ``not_broadcast``, carries a ``players_complete`` marker,
  or already has a hand-labelled (``labeller``) player-class box: that is George's work.
* QA alias clips (``<clip>__b``, the blind ball-repeat cards) are NOT included: they are separate sidecars
  that exist to measure ball-labelling noise, and a second player label of the same frame adds nothing.
* Writes use the same temp file + fsync + ``os.replace`` as ``/review/correct`` (the labeller takes no file
  lock on sidecars), after re-checking that the file did not change while we merged (retry otherwise).

HELDOUT RULE: the frames are eval-only inference; nothing is trained or cached. ``prefill`` refuses a match
that is not registered heldout.

``job`` submits one ``player_check`` labelling job per match (one item per panel window) to the labeller
(``jobs_client``); see ``labeller/README.md`` section 19l.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from footy_track.ball_eval import panel as P  # noqa: N812
from footy_track.ball_eval import panel_suggest as PS  # noqa: N812
from footy_track.labeller.constants import (
    NOT_BROADCAST_TAG,
    PLAYERS_COMPLETE_TAG,
    PROV_LABELLER,
    PROV_YOLO,
)

PREFILL_TAG = "prefill:rtdetr"
PERSON_CONF = 0.3
LABELS = ("player", "referee")
IMGSZ = 1280
BATCH = 4
MAX_RETRIES = 5

DetectFn = Callable[[str, Path, list[int]], dict[int, list[dict]]]


# ---- pure merge ------------------------------------------------------------------------------ #


def _parse(line: str) -> dict | None:
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def box_line(frame: int, label: str, b: dict) -> str:
    x, y = max(0.0, min(1.0, b["x"])), max(0.0, min(1.0, b["y"]))
    w, h = max(0.0, min(1.0 - x, b["w"])), max(0.0, min(1.0 - y, b["h"]))
    return json.dumps(
        {
            "frame_index": int(frame),
            "bbox": {"x": round(x, 6), "y": round(y, 6), "w": round(w, 6), "h": round(h, 6)},
            "center": {"x": round(x + w / 2, 6), "y": round(y + h / 2, 6)},
            "tags": [label, PROV_YOLO, PREFILL_TAG],
        }
    )


def merge_prefill(
    lines: list[str], boxes_by_frame: dict[int, list[dict]]
) -> tuple[list[str], dict]:
    """``(new lines, report)``: drop the old prefill rows of the target frames, append the new ones.

    Every other line is returned unchanged and in order. ``boxes_by_frame`` maps frame -> ``[{label,x,y,w,h,conf}]``.
    """
    targets = set(boxes_by_frame)
    skip: dict[int, str] = {}
    for raw in lines:
        d = _parse(raw)
        if d is None or d.get("frame_index") not in targets:
            continue
        tags = d.get("tags") or []
        f = int(d["frame_index"])
        if NOT_BROADCAST_TAG in tags:
            skip[f] = "not_broadcast"
        elif PLAYERS_COMPLETE_TAG in tags:
            skip.setdefault(f, "players_complete")
        elif (
            d.get("bbox") is not None
            and PROV_LABELLER in tags
            and any(t in ("player", "referee", "player_sub", "coach", "person") for t in tags)
        ):
            skip.setdefault(f, "hand_player_box")
    work = targets - set(skip)
    kept: list[str] = []
    removed = 0
    for raw in lines:
        d = _parse(raw)
        if d is not None and d.get("frame_index") in work and PREFILL_TAG in (d.get("tags") or []):
            removed += 1
            continue
        kept.append(raw)
    added = [box_line(f, b["label"], b) for f in sorted(work) for b in boxes_by_frame[f]]
    counts = {f: Counter(b["label"] for b in boxes_by_frame[f]) for f in sorted(work)}
    return kept + added, {
        "frames": len(work),
        "added": len(added),
        "replaced": removed,
        "skipped": {str(f): why for f, why in sorted(skip.items())},
        "per_frame": {str(f): dict(c) for f, c in counts.items()},
    }


def _sig(path: Path) -> tuple | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def write_prefill(path: Path, boxes_by_frame: dict[int, list[dict]]) -> dict:
    """Merge ``boxes_by_frame`` into the sidecar ``path`` (atomic, re-checked against concurrent writers)."""
    from footy_track.labeller.review import _write_text_atomic  # noqa: PLC0415

    for _ in range(MAX_RETRIES):
        before = _sig(path)
        lines = path.read_text().splitlines() if before else []
        new, rep = merge_prefill(lines, boxes_by_frame)
        if new == lines:
            return rep
        if _sig(path) != before:
            time.sleep(0.2)
            continue
        _write_text_atomic(path, "\n".join(new) + "\n")
        return rep
    raise RuntimeError(f"{path} kept changing under us; not written")


# ---- detection ------------------------------------------------------------------------------- #


class RtDetrPersons:
    """RT-DETR at 1280 px: ``{frame: [{label, x, y, w, h, conf}]}`` of player / referee boxes with conf >= 0.3."""

    def __init__(self, weights: Path | None = None, *, device: str = "cpu", imgsz: int = IMGSZ, batch: int = BATCH):
        from footy_track.ball_mining import candidates as C  # noqa: N812, PLC0415

        if weights is None:
            from footy_track.detectors.ultralytics import current_best_detector_path  # noqa: I001, PLC0415

            weights = current_best_detector_path()
        self._C = C
        self.batch = batch
        self.weights = Path(weights)
        self.sha8 = C.sha8(C.sha256_file(weights))
        self.detector = C.make_detector(weights, imgsz=imgsz, conf=PERSON_CONF, device=device)
        self.dropped: Counter = Counter()

    def __call__(self, clip: str, video: Path, frames: list[int]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        pending: list[tuple[int, object]] = []

        def flush() -> None:
            if not pending:
                return
            results = self.detector.predict_frames([b for _, b in pending])
            self._C.limit_threads()
            for (f, _), res in zip(pending, results, strict=True):
                rows = []
                for d in res.detections:
                    if d.label in LABELS and d.confidence >= PERSON_CONF:
                        rows.append({"label": d.label, "x": d.x, "y": d.y, "w": d.w, "h": d.h, "conf": float(d.confidence)})
                    elif d.label not in ("ball", "in_play_ball", "out_of_play_ball"):
                        self.dropped[d.label] += 1
                out[f] = rows
            pending.clear()

        for f, bgr in PS._read_frames(video, frames):  # noqa: SLF001 - sequential, frame-exact decode
            pending.append((f, bgr))
            if len(pending) >= self.batch:
                flush()
        flush()
        return out


def find_video(clips_dir: Path, clip: str) -> Path:
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        p = clips_dir / f"{clip}{ext}"
        if p.exists():
            return p
    raise P.PanelError(f"no video for {clip} in {clips_dir}")


def prefill_match(
    match: str,
    *,
    panel_version: str = P.PANEL_VERSION,
    marks_dir: Path,
    clips_dir: Path,
    detect: DetectFn,
    root: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Detect + write for every window frame of ``match``'s panel. Returns a per-clip report."""
    P.require_heldout(match)
    doc = P.read_panel(P.panel_path(match, panel_version, root))
    targets = PS.panel_frames(doc)  # windows only: the ``__b`` QA aliases are not in here
    report: dict = {"match": match, "clips": {}, "frames": 0, "players": 0, "referees": 0}
    for clip, frames in targets.items():
        sidecar = marks_dir / f"{clip}.jsonl"
        if not sidecar.exists():
            log(f"{clip}: no sidecar, skipped (frames not labelled)")
            report["clips"][clip] = {"skipped": "no sidecar"}
            continue
        t0 = time.time()
        boxes = detect(clip, find_video(clips_dir, clip), frames)
        rep = write_prefill(sidecar, boxes)
        n_p = sum(c.get("player", 0) for c in rep["per_frame"].values())
        n_r = sum(c.get("referee", 0) for c in rep["per_frame"].values())
        report["clips"][clip] = rep
        report["frames"] += rep["frames"]
        report["players"] += n_p
        report["referees"] += n_r
        log(
            f"{clip}: {rep['frames']} frames, +{rep['added']} boxes ({n_p} player, {n_r} referee), "
            f"replaced {rep['replaced']}, skipped {len(rep['skipped'])} in {time.time() - t0:.0f}s"
        )
    return report


# ---- job ------------------------------------------------------------------------------------- #


def not_broadcast_frames(marks_dir: Path, clip: str) -> set[int]:
    path = marks_dir / f"{clip}.jsonl"
    if not path.exists():
        return set()
    out = set()
    for raw in path.read_text().splitlines():
        d = _parse(raw)
        if d is not None and NOT_BROADCAST_TAG in (d.get("tags") or []):
            out.add(int(d["frame_index"]))
    return out


def job_body(doc: dict, not_broadcast: dict[str, set[int]] | None = None) -> dict:
    """One ``player_check`` job per panel: one item per window (``frame_start..frame_end`` step), ``eval: true``.

    A window whose frames are ALL ``not_broadcast`` (George marked them so) is left out: those frames can never be
    completed, and the job would sit below 100 % forever. ``not_broadcast`` is ``{clip: {frames}}``."""
    nb = not_broadcast or {}
    windows = [w for w in doc["windows"] if not set(w["frames"]) <= nb.get(w["clip"], set())]
    n_frames = sum(len(w["frames"]) for w in windows)
    items = [
        {
            "clip": w["clip"],
            "frame_start": w["frame_start"],
            "frame_end": w["frame_end"],
            "step": w["step"],
            "eval": True,
            "stratum": doc["match"],
            "sampling": {"regime": "eval", "panel": doc["panel_version"], "match": doc["match"], "window_id": w["window_id"]},
        }
        for w in windows
    ]
    return {
        "title": f"Player check (eval): {doc['match']}",
        "kind": "player_check",
        "dataset": P.dataset_name(doc["panel_version"]),
        "items": items,
        "why": (
            f"Box every player and referee on the {n_frames} ball-panel frames of {doc['match']} "
            f"({len(windows)} windows). Machine boxes (RT-DETR, conf >= 0.3) are pre-filled: swipe right "
            "if each person has exactly one box, left to fix. Eval data: never training."
        ),
        "device": "laptop",  # ~15 tiny boxes per full 1920 px frame: tap-to-fix needs a laptop (swipe-ok alone works on a phone)
        "priority": 2,
        "key": f"player-check/{doc['panel_version']}/{doc['match']}",
    }


# ---- CLI ------------------------------------------------------------------------------------- #


def _dataset_dirs(name: str) -> tuple[Path, Path]:
    from footy_track.labeller import datasets  # noqa: PLC0415

    ds = datasets.get(name)
    return ds.marks_dir, ds.clips_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prefill", "job"):
        sp = sub.add_parser(name)
        sp.add_argument("--dataset", default=P.dataset_name())
        sp.add_argument("--panel-version", default=P.PANEL_VERSION)
        sp.add_argument("--match", action="append", required=True)
        sp.add_argument("--root", type=Path, default=None)
        if name == "prefill":
            sp.add_argument("--weights", type=Path, default=None)
            sp.add_argument("--device", default="cpu")
            sp.add_argument("--report", type=Path, default=None, help="write the JSON report here")
        else:
            sp.add_argument("--url", default=None, help="labeller URL (default $FOOTY_LABELLER_URL)")
    a = ap.parse_args(argv)

    if a.cmd == "prefill":
        marks, clips = _dataset_dirs(a.dataset)
        os.nice(10)
        det = RtDetrPersons(a.weights, device=a.device)
        reports = []
        for m in a.match:
            rep = prefill_match(m, panel_version=a.panel_version, marks_dir=marks, clips_dir=clips, detect=det, root=a.root)
            rep["detector_sha8"] = det.sha8
            rep["dropped_other_classes"] = dict(det.dropped)
            reports.append(rep)
            print(json.dumps({k: v for k, v in rep.items() if k != "clips"}))
        if a.report:
            a.report.write_text(json.dumps(reports, indent=1))
        return 0

    from footy_track.labeller import jobs_client  # noqa: PLC0415

    if a.url:
        os.environ["FOOTY_LABELLER_URL"] = a.url
        jobs_client.DEFAULT_URL = a.url
    for m in a.match:
        doc = P.read_panel(P.panel_path(m, a.panel_version, a.root))
        marks, _clips = _dataset_dirs(a.dataset)
        body = job_body(doc, {c: not_broadcast_frames(marks, c) for c in {w["clip"] for w in doc["windows"]}})
        job = jobs_client.submit(
            body["title"], body["kind"], body["items"], dataset=a.dataset, why=body["why"],
            device=body["device"], priority=body["priority"], key=body["key"],
        )
        print(json.dumps({"match": m, "id": job.get("id"), "status": job.get("status"), "url": job.get("url")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
