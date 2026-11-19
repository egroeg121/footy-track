"""Propose machine labels for one window (docs/design/full_frame_labelling.md section 4, stage 2).

For a window from :mod:`footy_track.autolabel.windows` run, over its frames:

1. the **detector** (RT-DETR via ``get_current_best_detector``): players, referees, ball
   candidates -> store rows ``source='yolo'``;
2. the **ball tracker forward AND backward** over the detector's ball candidates, using the
   existing ``ball_trackers`` functions (``run_online``, ``run_reversed``, ``merge_passes``);
   the merged track -> rows ``source=detector_track`` (true origin: detection-based, no ViTTrack ran;
   ``vittrack`` is reserved for a real ViTTrack adapter behind ``TrackerPasses``, via ``ball_source``). The two passes are kept apart for the disagreement score;
3. **SAM3**, only on frames where the detector's person count differs from the median of its
   neighbours (radius ``neighbour_radius``), to add person boxes the detector missed (a SAM3
   box is added only when it does not overlap a detector box, IoU < ``dup_iou``) -> rows
   ``source='sam3'``.

Everything heavy is behind small injectable interfaces (:class:`FrameReader`,
:class:`FrameDetector`, :class:`PersonFiller`, :class:`TrackerPasses`), so tests use fakes and
the real adapters import their models lazily.

Store output (all rows ``reviewed=False``, ``needs_review=True``): one ``run`` row
``autolabel:<window_id>`` (the *proposing run*; its id is what later ``frame_label_state`` rows
cite) plus one run row per source ``autolabel:<window_id>:<source>`` (run ids are the primary
key, detection rows carry the per-source one). Re-proposing a window replaces its earlier rows
(``delete_run_rows`` on the ``autolabel:<window_id>:`` prefix); it refuses if any of them was
already reviewed.

Disagreement scores are stored beside the rows as ``<out_dir>/<window_id>.npz`` (the layout the
design doc names, ``proposals/<window>.npz``; not a store table because they are per-run
derived data the queue reads whole, and the store has no float-array column convention).
Keys, one entry per window frame in order:

``frame_index`` int; ``ball_gap_px`` float (forward/backward centre distance, NaN unless both
passes have a ball); ``ball_presence_mismatch`` bool (exactly one pass has a ball);
``det_person_count`` int; ``sam3_person_count`` int (-1 where SAM3 was not run);
``person_flag`` bool (detector count disagrees with neighbours or with SAM3);
``ball_component`` / ``person_component`` / ``score`` float in 0..1 (``score`` is the max of the
two components; higher = the teacher members disagree more = more worth a human's time).

CLI (what a Prefect job calls)::

    python -m footy_track.autolabel.propose --window-file windows.json [--window-id ID | --all]
        [--db STORE.duckdb] [--out-dir proposals/] [--no-sam3]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from footy_track.autolabel.windows import Window, eligible_match_key, read_windows
from footy_track.ball_trackers.bidirectional import (
    merge_passes,
    run_online,
    run_reversed,
)
from footy_track.ball_trackers.detection_tracker import NONE, BallCandidate, BallPoint
from footy_track.constants import (
    BALL_TAG,
    IN_PLAY_BALL_TAG,
    OUT_OF_PLAY_BALL_TAG,
)
from footy_track.feature_store.schema import (
    DetectionRow,
    FrameRow,
    GameRow,
    RunRow,
    Source,
    Stage,
)
from footy_track.schema import FrameDetections, ObjectDetection

BALL_LABELS = frozenset({BALL_TAG, IN_PLAY_BALL_TAG, OUT_OF_PLAY_BALL_TAG})
DEFAULT_AGREE_PX = 60.0  # same default as the bidirectional tracker's merge
RUN_PREFIX = "autolabel:"


# --------------------------------------------------------------------------------------------
# Injectable interfaces
# --------------------------------------------------------------------------------------------
class FrameReader(Protocol):
    def fps(self, video: str) -> float: ...

    def read(
        self, video: str, start: int, end: int
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, BGR image)`` for ``start <= i < end``."""
        ...


class FrameDetector(Protocol):
    @property
    def model_tag(self) -> str: ...

    def detect(self, frames: list[np.ndarray]) -> list[FrameDetections]: ...


class PersonFiller(Protocol):
    """SAM3 (or a fake): person-group boxes for one frame."""

    @property
    def model_tag(self) -> str: ...

    def people(self, frame: np.ndarray) -> list[ObjectDetection]: ...


@dataclass
class TrackPasses:
    forward: list[BallPoint]
    backward: list[BallPoint]
    merged: list[BallPoint]


class TrackerPasses(Protocol):
    def run(
        self, frames: list[tuple[int, list[BallCandidate]]], frame_size: tuple[int, int]
    ) -> TrackPasses: ...


class DetectionTrackerPasses:
    """Default: the existing detection-based tracker run forward and reversed, then merged."""

    def __init__(self, agree_px: float = DEFAULT_AGREE_PX, **tracker_kwargs) -> None:
        self.agree_px = agree_px
        # same tuned defaults BidirectionalBallTracker applies
        tracker_kwargs.setdefault("max_missed", 6)
        tracker_kwargs.setdefault("growth_px", 15.0)
        self.kwargs = tracker_kwargs

    def run(self, frames, frame_size):
        kw = dict(frame_size=frame_size, **self.kwargs)
        fwd = run_online(frames, **kw)
        bwd = run_reversed(frames, **kw)
        merged = merge_passes(
            fwd, bwd, frame_size, agree_px=self.agree_px,
            start_conf=self.kwargs.get("start_conf", 0.6),
        )  # fmt: skip
        return TrackPasses(fwd, bwd, merged)


# --------------------------------------------------------------------------------------------
# Real adapters (lazy heavy imports)
# --------------------------------------------------------------------------------------------
class Cv2FrameReader:
    def fps(self, video: str) -> float:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(video)
        try:
            return float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        finally:
            cap.release()

    def read(self, video: str, start: int, end: int):
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(video)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(start, end):
                ok, img = cap.read()
                if not ok:
                    return
                yield i, img
        finally:
            cap.release()


class ObjectDetectorAdapter:
    """Wraps an ``ObjectDetector`` (``predict_frames`` when it has it, else per-frame files)."""

    def __init__(self, detector) -> None:
        self.detector = detector

    @property
    def model_tag(self) -> str:
        return self.detector.model_tag

    def detect(self, frames):
        if hasattr(self.detector, "predict_frames"):
            return self.detector.predict_frames(frames)
        with tempfile.TemporaryDirectory() as td:
            return [
                self.detector.predict_from_path(_write_jpg(Path(td) / f"{i}.jpg", f))
                for i, f in enumerate(frames)
            ]


class Sam3PersonFiller:
    """SAM3 via ``UltralyticsSam3Detector``; ball detections are dropped."""

    def __init__(self, detector=None) -> None:
        if detector is None:
            from footy_track.detectors.ultralytics import UltralyticsSam3Detector  # noqa: PLC0415

            detector = UltralyticsSam3Detector()
        self.detector = detector

    @property
    def model_tag(self) -> str:
        return "sam3"

    def people(self, frame):
        with tempfile.TemporaryDirectory() as td:
            res = self.detector.predict_from_path(_write_jpg(Path(td) / "f.jpg", frame))
        return [d for d in res.detections if d.label not in BALL_LABELS]


def _write_jpg(path: Path, frame: np.ndarray) -> Path:
    import cv2  # noqa: PLC0415

    cv2.imwrite(str(path), frame)
    return path


# --------------------------------------------------------------------------------------------
# Disagreement scores (pure)
# --------------------------------------------------------------------------------------------
def person_count_flags(counts: Sequence[int], radius: int = 5) -> list[bool]:
    """True where a frame's person count differs from the median of its neighbours (``radius``
    frames each side, itself excluded, rounded to the nearest integer)."""
    flags = []
    for i, c in enumerate(counts):
        nb = [
            counts[j]
            for j in range(max(0, i - radius), min(len(counts), i + radius + 1))
            if j != i
        ]
        flags.append(bool(nb) and round(statistics.median(nb)) != c)
    return flags


def _centre_px(p: BallPoint, frame_size: tuple[int, int]) -> tuple[float, float] | None:
    if p.source == NONE or p.x is None:
        return None
    return (p.x + p.w / 2) * frame_size[0], (p.y + p.h / 2) * frame_size[1]


def disagreement_scores(  # noqa: PLR0917
    frame_indices: Sequence[int],
    fwd: Sequence[BallPoint],
    bwd: Sequence[BallPoint],
    det_person_counts: Sequence[int],
    sam3_person_counts: Sequence[int] | None,
    frame_size: tuple[int, int],
    *,
    agree_px: float = DEFAULT_AGREE_PX,
    neighbour_radius: int = 5,
) -> dict[str, np.ndarray]:
    """Per-frame disagreement between teacher members (see the module docstring for keys).

    ball_component: 0 when neither pass has a ball or both agree within ``agree_px``; grows
    linearly to 1 at ``4 * agree_px`` apart; 1 when exactly one pass has a ball.
    person_component: 1 - 1/(1+d) where d is |detector count - SAM3 count| where SAM3 ran,
    else 0.5 for a frame whose detector count disagrees with its neighbours' median (SAM3
    could not weigh in), else 0.
    """
    n = len(frame_indices)
    if not (len(fwd) == len(bwd) == len(det_person_counts) == n):
        raise ValueError("per-frame inputs must all have one entry per frame")
    sam3 = list(sam3_person_counts) if sam3_person_counts is not None else [-1] * n
    gap = np.full(n, np.nan)
    mismatch = np.zeros(n, dtype=bool)
    ball_c = np.zeros(n)
    for i, (a, b) in enumerate(zip(fwd, bwd, strict=True)):
        ca, cb = _centre_px(a, frame_size), _centre_px(b, frame_size)
        if ca is None and cb is None:
            continue
        if ca is None or cb is None:
            mismatch[i] = True
            ball_c[i] = 1.0
            continue
        gap[i] = math.dist(ca, cb)
        ball_c[i] = min(max((gap[i] - agree_px) / (3 * agree_px), 0.0), 1.0)
    flags = person_count_flags(det_person_counts, neighbour_radius)
    person_c = np.zeros(n)
    person_flag = np.zeros(n, dtype=bool)
    for i in range(n):
        if sam3[i] >= 0:
            d = abs(det_person_counts[i] - sam3[i])
            person_c[i] = 1.0 - 1.0 / (1.0 + d)
            person_flag[i] = d > 0 or flags[i]
        elif flags[i]:
            person_c[i], person_flag[i] = 0.5, True
    return {
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        "ball_gap_px": gap,
        "ball_presence_mismatch": mismatch,
        "det_person_count": np.asarray(det_person_counts, dtype=np.int64),
        "sam3_person_count": np.asarray(sam3, dtype=np.int64),
        "person_flag": person_flag,
        "ball_component": ball_c,
        "person_component": person_c,
        "score": np.maximum(ball_c, person_c),
    }


# --------------------------------------------------------------------------------------------
# Propose
# --------------------------------------------------------------------------------------------
@dataclass
class ProposeResult:
    window_id: str
    run_id: str  # the proposing run (cite this from frame_label_state.run_id)
    run_ids: dict[str, str] = field(default_factory=dict)  # source -> per-source run id
    rows_by_source: dict[str, int] = field(default_factory=dict)
    frames: int = 0
    sam3_frames: list[int] = field(default_factory=list)
    scores_path: Path | None = None


def _iou(a: ObjectDetection, b: ObjectDetection) -> float:
    from footy_track.detectors.utils import calculate_iou  # noqa: PLC0415

    return calculate_iou(a, b)


def _clamp(v: float) -> float:
    return min(max(float(v), 0.0), 1.0)


def _rows(  # noqa: PLR0917
    game_id, frame_index, ct, source, run_id, dets, *, track_ids=None, interpolated=None
):
    out = []
    for i, d in enumerate(dets):
        out.append(
            DetectionRow(
                game_id=game_id, frame_index=frame_index, continuous_time_s=ct,
                detection_id=i, source=source, run_id=run_id, label=d.label,
                confidence=d.confidence, bbox_x=_clamp(d.x), bbox_y=_clamp(d.y),
                bbox_w=_clamp(d.w), bbox_h=_clamp(d.h),
                track_id=track_ids[i] if track_ids else None,
                is_interpolated=bool(interpolated[i]) if interpolated else False,
                needs_review=True, reviewed=False,
            )
        )  # fmt: skip
    return out


def propose_window(  # noqa: PLR0913, PLR0914, PLR0915
    window: Window,
    *,
    reader: FrameReader,
    detector: FrameDetector,
    tracker: TrackerPasses | None = None,
    person_filler: PersonFiller | None = None,
    store=None,
    out_dir: Path | None = None,
    ball_source: str = Source.DETECTOR_TRACK.value,
    batch_size: int = 8,
    neighbour_radius: int = 5,
    dup_iou: float = 0.5,
) -> ProposeResult:
    """Run the proposers over ``window``; write rows to ``store`` and scores to ``out_dir``."""
    # Rule zero (ball_sampling.md): never propose for anything but a train-split segment clip,
    # whatever windows.json says. Raises RegisterError when the split register is missing.
    if eligible_match_key(window.game_id) is None:
        raise ValueError(
            f"{window.game_id}: not a train-split _segNNN clip; refusing to propose"
        )
    tracker = tracker or DetectionTrackerPasses()
    base = f"{RUN_PREFIX}{window.window_id}"
    prefix = base + ":"
    video = window.video or ""
    fps = reader.fps(video)
    if store is not None:
        done = store.query(
            "SELECT count(*) AS n FROM detection WHERE game_id = ? "
            "AND starts_with(run_id, ?) AND reviewed",
            [window.game_id, prefix],
        )
        if int(done["n"][0]):
            raise RuntimeError(f"{base}: proposals already reviewed; not overwriting")

    # pass 1: detector over the window (frames are not retained)
    indices: list[int] = []
    dets: list[FrameDetections] = []
    buf: list[tuple[int, np.ndarray]] = []

    def flush() -> None:
        if buf:
            res = detector.detect([f for _, f in buf])
            indices.extend(i for i, _ in buf)
            dets.extend(res)
            buf.clear()

    for i, img in reader.read(video, window.start_frame, window.end_frame):
        buf.append((i, img))
        if len(buf) >= batch_size:
            flush()
    flush()
    if not indices:
        raise RuntimeError(
            f"{window.window_id}: no frames could be read from {video!r}"
        )
    frame_size = (dets[0].width, dets[0].height)

    # ball: forward + backward + merged over detector candidates
    cand_frames = [
        (
            i,
            [
                BallCandidate(d.x, d.y, d.w, d.h, d.confidence)
                for d in fd.detections
                if d.label in BALL_LABELS
            ],
        )
        for i, fd in zip(indices, dets, strict=True)
    ]
    passes = tracker.run(cand_frames, frame_size)

    # persons: SAM3 only where the detector count disagrees with its neighbours
    person_dets = [
        [d for d in fd.detections if d.label not in BALL_LABELS] for fd in dets
    ]
    det_counts = [len(p) for p in person_dets]
    sam3_counts = [-1] * len(indices)
    sam3_extra: dict[int, list[ObjectDetection]] = {}
    sam3_frames: list[int] = []
    if person_filler is not None:
        flagged = {
            indices[k]
            for k, f in enumerate(person_count_flags(det_counts, neighbour_radius))
            if f
        }
        if flagged:
            pos = {fi: k for k, fi in enumerate(indices)}
            for fi, img in reader.read(video, min(flagged), max(flagged) + 1):
                if fi not in flagged:
                    continue
                k = pos[fi]
                found = person_filler.people(img)
                sam3_counts[k] = len(found)
                sam3_extra[fi] = [
                    s
                    for s in found
                    if all(_iou(s, p) < dup_iou for p in person_dets[k])
                ]
                sam3_frames.append(fi)

    scores = disagreement_scores(
        indices, passes.forward, passes.backward, det_counts, sam3_counts, frame_size,
        neighbour_radius=neighbour_radius,
    )  # fmt: skip

    result = ProposeResult(
        window.window_id, base, frames=len(indices), sam3_frames=sam3_frames
    )
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result.scores_path = out_dir / f"{window.window_id}.npz"
        np.savez_compressed(result.scores_path, **scores)
    if store is not None:
        _write_store(store, window, result, fps, frame_size, indices, dets, passes,
                     sam3_extra, ball_source, detector, person_filler)  # fmt: skip
    return result


def _write_store(  # noqa: PLR0917
    store, window, result, fps, frame_size, indices, dets, passes,
    sam3_extra, ball_source, detector, person_filler,
) -> None:  # fmt: skip
    base, gid = result.run_id, window.game_id
    sources = {
        Source.YOLO.value: detector.model_tag,
        ball_source: "ball-tracker-fwd-bwd",
    }
    if sam3_extra:
        sources[Source.SAM3.value] = person_filler.model_tag
    result.run_ids = {s: f"{base}:{s}" for s in sources}
    merged = {p.frame_index: p for p in passes.merged}
    rows: dict[str, list[DetectionRow]] = {s: [] for s in sources}
    for fi, fd in zip(indices, dets, strict=True):
        ct = fi / fps
        rows[Source.YOLO.value] += _rows(
            gid,
            fi,
            ct,
            Source.YOLO.value,
            result.run_ids[Source.YOLO.value],
            fd.detections,
        )
        p = merged.get(fi)
        if p is not None and p.source != NONE and p.x is not None:
            box = ObjectDetection(
                label=BALL_TAG, confidence=p.confidence if p.confidence is not None else 0.0,
                x=_clamp(p.x), y=_clamp(p.y), w=_clamp(p.w), h=_clamp(p.h),
            )  # fmt: skip
            rows[ball_source] += _rows(
                gid, fi, ct, ball_source, result.run_ids[ball_source], [box],
                track_ids=[p.track_id], interpolated=[p.source != "detected"],
            )  # fmt: skip
        if fi in sam3_extra:
            rows[Source.SAM3.value] += _rows(
                gid,
                fi,
                ct,
                Source.SAM3.value,
                result.run_ids[Source.SAM3.value],
                sam3_extra[fi],
            )
    params = json.dumps(
        {"window_id": window.window_id, "start_frame": window.start_frame,
         "end_frame": window.end_frame, "runs": result.run_ids}, sort_keys=True,
    )  # fmt: skip
    with store.transaction():
        store.delete_run_rows(gid, base + ":")
        if store.query("SELECT 1 FROM game WHERE game_id = ?", [gid]).empty:
            store.upsert_games(
                [
                    GameRow(
                        game_id=gid, fps=fps, width=frame_size[0], height=frame_size[1]
                    )
                ]
            )
        known = set(
            store.query("SELECT frame_index FROM frame WHERE game_id = ?", [gid])[
                "frame_index"
            ]
        )
        store.upsert_frames([
            FrameRow(game_id=gid, frame_index=fi, frame_uri=f"{gid}_frame_{fi:06d}",
                     width=frame_size[0], height=frame_size[1], continuous_time_s=fi / fps)
            for fi in indices if fi not in known
        ])  # fmt: skip
        store.upsert_runs(
            [RunRow(run_id=base, stage=Stage.DETECTION, source="autolabel",
                    model_name="autolabel-propose", params_json=params)]
            + [RunRow(run_id=result.run_ids[s], stage=Stage.DETECTION, source=s, model_name=m)
               for s, m in sources.items()]
        )  # fmt: skip
        for s, rs in rows.items():
            store.upsert_detections(rs)
            result.rows_by_source[s] = len(rs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window-file", type=Path, required=True)
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--window-id")
    sel.add_argument(
        "--all", action="store_true", help="every window in the file (default)"
    )
    ap.add_argument(
        "--db",
        type=Path,
        help="feature store to write rows to (default: $FOOTY_FEATURE_STORE)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("proposals"))
    ap.add_argument("--no-sam3", action="store_true")
    args = ap.parse_args(argv)

    windows = read_windows(args.window_file)
    if args.window_id:
        windows = [w for w in windows if w.window_id == args.window_id]
        if not windows:
            ap.error(f"window {args.window_id!r} not in {args.window_file}")

    from footy_track.detectors.ultralytics import get_current_best_detector  # noqa: PLC0415
    from footy_track.feature_store.lock import default_db_path, store_lock  # noqa: PLC0415
    from footy_track.feature_store.store import FeatureStore  # noqa: PLC0415

    detector = ObjectDetectorAdapter(get_current_best_detector())
    filler = None if args.no_sam3 else Sam3PersonFiller()
    db = args.db or default_db_path()
    with store_lock(db, exclusive=True), FeatureStore.open(db) as store:
        for w in windows:
            r = propose_window(w, reader=Cv2FrameReader(), detector=detector,
                               person_filler=filler, store=store, out_dir=args.out_dir)  # fmt: skip
            print(json.dumps({"window_id": r.window_id, "run_id": r.run_id,
                              "rows": r.rows_by_source, "sam3_frames": len(r.sam3_frames),
                              "scores": str(r.scores_path)}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
