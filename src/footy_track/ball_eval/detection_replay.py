"""Ball-tracking measurement: cache detector output once, replay trackers, score vs GT.

Three steps, so a tracker change costs seconds instead of a GPU run:

1. ``cache``  run the detector once per clip and write every ball candidate
   (confidence >= ``--detect-conf``, nothing filtered) to ``<cache>/<clip>.jsonl``.
2. ``replay`` feed a cache to any tracker from ``ball_trackers.factory`` (CPU only).
3. ``score``  compare the tracker output with the human ball marks.

    python -m footy_track.scripts.ball_tracking_eval cache  --videos V --out C
    python -m footy_track.scripts.ball_tracking_eval score  --cache C --gt G \\
        --trackers online,bidirectional,viterbi --split split.json

See ``docs/ball-tracking-eval.md`` for the metric definitions and the GT rules.

Cache format (JSONL): the first line is ``{"meta": {...}}`` (clip, frame size, fps,
weights, imgsz, detect_conf); then one row per processed frame
``{"f": frame_index, "c": [[x, y, w, h, conf], ...]}`` with normalised top-left
boxes, ``c`` empty when the detector found no ball.

Ground truth. A ball mark is a sidecar row whose first tag is ``in_play_ball`` or
``out_of_play_ball``. Only ``labeller`` (human-placed) marks are GT. Marks that a
machine propagated and a human then checked (``vittrack`` + ``human_checked``) are
scored separately as ``propagated``; ``yolo``/``sam3`` marks and rows with no
provenance are never GT. A ``no_ball`` row says the ball is not visible in that
frame; ``not_broadcast`` frames and frames with no row are not scored. Sidecars
are merged by canonical clip id (``footy_track.clip_id``), so the several names of
one clip are one clip.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from footy_track.ball_trackers.detection_tracker import (
    DETECTED,
    INTERPOLATED,
    BallCandidate,
    BallPoint,
)
from footy_track.clip_id import canonical_clip_id

BALL_TAGS = ("in_play_ball", "out_of_play_ball")
SKIP_TAGS = ("no_ball", "not_broadcast")
HUMAN = "human"
PROPAGATED = "propagated"
DEFAULT_RADII_PX = (10.0, 25.0)

# --------------------------------------------------------------------------- cache


@dataclass
class CandidateCache:
    """Every ball candidate the detector produced for one clip."""

    clip: str
    frame_size: tuple[int, int]
    frames: list[tuple[int, list[BallCandidate]]]
    meta: dict = field(default_factory=dict)

    def candidates(self, min_conf: float = 0.0) -> dict[int, list[BallCandidate]]:
        return {f: [c for c in cs if c.confidence >= min_conf] for f, cs in self.frames}


def write_cache(path: Path, cache: CandidateCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(cache.meta, clip=cache.clip, frame_size=list(cache.frame_size))
    with path.open("w") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for f, cands in cache.frames:
            rows = [
                [round(c.x, 5), round(c.y, 5), round(c.w, 5), round(c.h, 5)]
                + [round(c.confidence, 4)]
                for c in cands
            ]
            fh.write(json.dumps({"f": f, "c": rows}) + "\n")


def read_cache(path: Path) -> CandidateCache:
    with path.open() as fh:
        meta = json.loads(fh.readline())["meta"]
        frames = []
        for line in fh:
            row = json.loads(line)
            frames.append((row["f"], [BallCandidate(*c) for c in row["c"]]))
    w, h = meta["frame_size"]
    return CandidateCache(meta["clip"], (w, h), frames, meta)


def cache_from_detections_jsonl(
    path: Path, frame_size: tuple[int, int], clip: str | None = None
) -> CandidateCache:
    """Build a cache from a pipeline ``detections.jsonl`` (machine-label rows)."""
    by_frame: dict[int, list[BallCandidate]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        by_frame.setdefault(d["frame_index"], [])
        if d["tags"][0] in ("in_play_ball", "ball"):
            b = d["bbox"]
            by_frame[d["frame_index"]].append(
                BallCandidate(b["x"], b["y"], b["w"], b["h"], d["confidence"])
            )
    frames = sorted(by_frame.items())
    return CandidateCache(clip or path.parent.name, frame_size, frames, {})


def build_cache(
    video: Path, detector, clip: str, *, batch_size: int = 8
) -> CandidateCache:
    """Run ``detector`` over every frame of ``video`` and collect ball candidates."""
    import cv2  # noqa: PLC0415

    from footy_track.inference.pipeline import (  # noqa: PLC0415
        ball_candidates,
        iter_frames,
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[tuple[int, list[BallCandidate]]] = []
    batch: list[tuple[int, object]] = []

    def flush() -> None:
        results = detector.predict_frames([f for _, f in batch])
        for (index, _), res in zip(batch, results, strict=True):
            frames.append((index, ball_candidates(res)))
        batch.clear()

    for item in iter_frames(cap, 0, None, 1):
        batch.append(item)
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()
    cap.release()
    return CandidateCache(clip, size, frames, {"fps": fps})


# ------------------------------------------------------------------------ ground truth


@dataclass
class ClipGT:
    """Ball marks for one clip (normalised centres), by provenance tier."""

    clip: str
    marks: dict[str, dict[int, list[tuple[float, float]]]] = field(
        default_factory=lambda: {HUMAN: {}, PROPAGATED: {}}
    )
    no_ball: set[int] = field(default_factory=set)
    not_broadcast: set[int] = field(default_factory=set)
    ignored_machine_marks: int = 0  # yolo / sam3 / no-provenance ball rows

    def scored(
        self, tiers: Iterable[str]
    ) -> tuple[dict[int, list[tuple[float, float]]], set[int]]:
        """(ball frames -> centres, no-ball frames) for the given provenance tiers."""
        ball: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for tier in tiers:
            for f, cs in self.marks[tier].items():
                ball[f].extend(cs)
        no_ball = {f for f in self.no_ball if f not in ball}
        return dict(ball), no_ball - self.not_broadcast


def _centre(d: dict) -> tuple[float, float] | None:
    c = d.get("center")
    if isinstance(c, dict):
        return float(c["x"]), float(c["y"])
    if isinstance(c, list | tuple) and len(c) == 2:
        return float(c[0]), float(c[1])
    b = d.get("bbox")
    if isinstance(b, dict):
        return b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    return None


def load_gt(gt_dir: Path) -> dict[str, ClipGT]:
    """All sidecars in ``gt_dir``, merged by canonical clip id."""
    out: dict[str, ClipGT] = {}
    for path in sorted(gt_dir.glob("*.jsonl")):
        gt = out.setdefault(
            canonical_clip_id(path.stem), ClipGT(canonical_clip_id(path.stem))
        )
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            tags = d.get("tags") or []
            if not tags:
                continue
            f = int(d["frame_index"])
            if tags[0] == "no_ball":
                gt.no_ball.add(f)
            elif tags[0] == "not_broadcast":
                gt.not_broadcast.add(f)
            elif tags[0] in BALL_TAGS:
                c = _centre(d)
                if c is None:
                    continue
                if "labeller" in tags:
                    gt.marks[HUMAN].setdefault(f, []).append(c)
                elif "vittrack" in tags and "human_checked" in tags:
                    gt.marks[PROPAGATED].setdefault(f, []).append(c)
                else:
                    gt.ignored_machine_marks += 1
    return out


# ----------------------------------------------------------------------------- scoring


@dataclass
class Counts:
    """Counts for one radius over one clip or a pool of clips."""

    ball_frames: int = 0
    hit_detected: int = 0
    hit_interp: int = 0
    wrong_detected: int = 0
    wrong_interp: int = 0
    miss: int = 0
    no_ball_frames: int = 0
    false_detected: int = 0
    false_interp: int = 0
    det_any: int = 0  # detector upper bound: any candidate within R
    det_strong: int = 0  # any candidate >= strong_conf within R

    def add(self, other: Counts) -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))

    @property
    def hit(self) -> int:
        return self.hit_detected + self.hit_interp

    @property
    def wrong(self) -> int:
        return self.wrong_detected + self.wrong_interp

    @property
    def false(self) -> int:
        return self.false_detected + self.false_interp

    def ratios(self) -> dict[str, float | None]:
        def r(a: float, b: float) -> float | None:
            return a / b if b else None

        return {
            "precision": r(self.hit, self.hit + self.wrong + self.false),
            "recall": r(self.hit, self.ball_frames),
            "wrong_per_ball_frame": r(self.wrong, self.ball_frames),
            "wrong_per_prediction": r(self.wrong, self.hit + self.wrong),
            "det_upper_bound": r(self.det_any, self.ball_frames),
        }


def _nearest_px(point: tuple[float, float], marks, size: tuple[int, int]) -> float:
    w, h = size
    return min(math.hypot((point[0] - mx) * w, (point[1] - my) * h) for mx, my in marks)


def score_clip(  # noqa: PLR0912
    points: list[BallPoint],
    cache: CandidateCache,
    gt: ClipGT,
    *,
    tiers: Iterable[str] = (HUMAN,),
    radii: Iterable[float] = DEFAULT_RADII_PX,
    strong_conf: float = 0.6,
) -> dict[float, Counts]:
    """Score tracker ``points`` against the GT of one clip, per radius in px."""
    ball, no_ball = gt.scored(tiers)
    pred = {p.frame_index: p for p in points}
    cands = dict(cache.frames)
    size = cache.frame_size
    out = {r: Counts() for r in radii}
    for f, marks in ball.items():
        if f not in pred:  # frame outside the processed range
            continue
        p = pred[f]
        has_pred = p.source in (DETECTED, INTERPOLATED)
        dist = None
        if has_pred:
            dist = _nearest_px((p.x + p.w / 2, p.y + p.h / 2), marks, size)
        cdist = [_nearest_px(c.center, marks, size) for c in cands.get(f, [])]
        cstrong = [
            d
            for d, c in zip(cdist, cands.get(f, []), strict=True)
            if c.confidence >= strong_conf
        ]
        for r, cnt in out.items():
            cnt.ball_frames += 1
            cnt.det_any += bool(cdist) and min(cdist) <= r
            cnt.det_strong += bool(cstrong) and min(cstrong) <= r
            if not has_pred:
                cnt.miss += 1
            elif dist <= r:
                if p.source == DETECTED:
                    cnt.hit_detected += 1
                else:
                    cnt.hit_interp += 1
            elif p.source == DETECTED:
                cnt.wrong_detected += 1
            else:
                cnt.wrong_interp += 1
    for f in no_ball:
        if f not in pred:
            continue
        p = pred[f]
        for cnt in out.values():
            cnt.no_ball_frames += 1
            if p.source == DETECTED:
                cnt.false_detected += 1
            elif p.source == INTERPOLATED:
                cnt.false_interp += 1
    return out


def replay(tracker, cache: CandidateCache) -> list[BallPoint]:
    """Run a tracker (``update`` per frame, then ``points``) over a cache."""
    for f, cands in cache.frames:
        tracker.update(f, cands)
    return tracker.points()


def evaluate(
    caches: dict[str, CandidateCache],
    gts: dict[str, ClipGT],
    make_tracker: Callable[[tuple[int, int]], object],
    *,
    tiers: Iterable[str] = (HUMAN,),
    radii: Iterable[float] = DEFAULT_RADII_PX,
    clips: Iterable[str] | None = None,
) -> dict[str, dict[float, Counts]]:
    """Per-clip counts for every clip that has both a cache and GT."""
    radii = tuple(radii)
    out: dict[str, dict[float, Counts]] = {}
    for clip in sorted(clips if clips is not None else caches):
        if clip not in caches or clip not in gts:
            continue
        points = replay(make_tracker(caches[clip].frame_size), caches[clip])
        out[clip] = score_clip(
            points, caches[clip], gts[clip], tiers=tiers, radii=radii
        )
    return out


def pool(
    per_clip: dict[str, dict[float, Counts]], clips: Iterable[str]
) -> dict[float, Counts]:
    pooled: dict[float, Counts] = {}
    for clip in clips:
        for r, cnt in per_clip.get(clip, {}).items():
            pooled.setdefault(r, Counts()).add(cnt)
    return pooled


def format_counts(label: str, counts: dict[float, Counts]) -> list[str]:
    lines = []
    for r, c in sorted(counts.items()):
        x = c.ratios()

        def f(v: float | None) -> str:
            return "-" if v is None else f"{v:.3f}"

        lines.append(
            f"| {label} | {r:g} | {c.ball_frames} | {c.hit} ({c.hit_detected}d/{c.hit_interp}i) "
            f"| {c.wrong} ({c.wrong_detected}d/{c.wrong_interp}i) | {c.miss} "
            f"| {c.no_ball_frames} | {c.false} | {f(x['precision'])} | {f(x['recall'])} "
            f"| {f(x['wrong_per_prediction'])} | {c.det_any} ({f(x['det_upper_bound'])}) |"
        )
    return lines


TABLE_HEADER = [
    "| set | R px | ball frames | HIT | WRONG | MISS | no-ball frames | FALSE "
    "| precision | recall | wrong/pred | detector any-cand |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
