"""Run a tracker over precomputed detections and emit tracklets for review.

Tracking-by-detection: reads existing per-frame detection JSONL (the TIER-3
RT-DETR output in ``machine_labels/``) and assigns ``track_id``, so it needs no
GPU and no re-inference. ~650 fps on a laptop CPU.

    python -m footy_track.scripts.track_detections \
        --detections machine_labels/seg000.jsonl \
        --out tracklets/seg000.jsonl

Output is the format ``/identity/tracklets`` reads, so the review page picks it
up directly.

MEASURED DEFAULTS (arsenal_mancity_20250925_seg000, 38k player/referee
detections over 2534 frames). The library defaults (iou 0.3, max_age 30)
fragment badly on broadcast football — median tracklet 5 frames, 22% singletons:

    conf  iou   max_age | tracklets  median  >=25f  singletons
    0.5   0.3   30      |       397       5    106         22%   <- library default
    0.5   0.15  90      |       157      31     87          9%   <- default here

Association is NOT the limiting factor. Consecutive-frame best IoU is median
0.72 with only 4-6% below 0.3, and the tracker's spawn rate (4.2% of detections)
matches that tail almost exactly. Fragmentation comes from detection dropouts,
so the looser IoU and longer max_age exist to bridge missing detections rather
than to fix bad matching.

CAVEAT: none of the above measures tracklet PURITY. Fragmentation is measurable
without ground truth; correctness is not. Longer tracklets may mean better
tracking OR more ID switches, and switches are the failure mode that poisons
training data while fragments are harmless. Treat these defaults as a starting
point to be re-tuned once the human verification pass gives real purity numbers.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import time
from pathlib import Path

from footy_track.schema import FrameDetections, ObjectDetection
from footy_track.trackers.lap import LapTracker

# People only: identity work does not apply to the ball, and including it adds a
# fast-moving 16px object that association handles badly.
DEFAULT_LABELS = ("player", "referee")
SOURCE_TAGS = ("rtdetr", "yolo", "lap", "bytetrack")


def _clamp01(v: float) -> float:
    """RT-DETR emits boxes fractionally outside the frame (observed x=-1.9e-05).

    The schema requires [0, 1], so clamp as the feature-store importer does
    rather than dropping otherwise-valid detections.
    """
    return max(0.0, min(1.0, float(v)))


def label_of(tags: list[str]) -> str:
    return next((t for t in tags if t not in SOURCE_TAGS), "?")


def load_detections(
    path: Path, *, min_confidence: float, keep_labels: tuple[str, ...]
) -> dict[int, list[tuple[str, dict, float]]]:
    by_frame: dict[int, list[tuple[str, dict, float]]] = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn line must not abort a whole clip
        label = label_of(row.get("tags") or [])
        conf = float(row.get("confidence", 1.0))
        bbox = row.get("bbox")
        if label not in keep_labels or conf < min_confidence or not isinstance(bbox, dict):
            continue
        by_frame[int(row["frame_index"])].append((label, bbox, conf))
    return by_frame


def track_clip(
    by_frame: dict[int, list[tuple[str, dict, float]]],
    *,
    iou_threshold: float,
    max_age: int,
    fps: float = 25.0,
    width: int = 1920,
    height: int = 1080,
) -> list[dict]:
    """Assign track ids. Returns rows ready to write as tracklet JSONL."""
    if not by_frame:
        return []
    tracker = LapTracker(max_age=max_age, iou_threshold=iou_threshold)
    out: list[dict] = []
    for frame_index in range(max(by_frame) + 1):
        dets = [
            ObjectDetection(
                label=label,
                confidence=conf,
                x=_clamp01(b["x"]),
                y=_clamp01(b["y"]),
                w=_clamp01(b["w"]),
                h=_clamp01(b["h"]),
                model="rtdetr",
            )
            for label, b, conf in by_frame.get(frame_index, [])
        ]
        frame = FrameDetections(
            uri=Path(f"frame_{frame_index:06d}"),
            width=width,
            height=height,
            detections=dets,
        )
        for td in tracker.update(frame, frame_index / fps):
            out.append(
                {
                    "frame_index": td.frame_index,
                    "track_id": td.track_id,
                    "tags": [td.label, "lap"],
                    "bbox": {"x": td.x, "y": td.y, "w": td.w, "h": td.h},
                    "confidence": td.confidence,
                    "is_interpolated": bool(td.is_interpolated),
                }
            )
    return out


def summarise(rows: list[dict]) -> dict:
    lengths = collections.Counter(r["track_id"] for r in rows)
    values = sorted(lengths.values())
    if not values:
        return {"tracklets": 0}
    return {
        "tracklets": len(values),
        "detections": len(rows),
        "median_length": st.median(values),
        "p90_length": values[int(len(values) * 0.9)],
        "max_length": values[-1],
        "usable_ge_25f": sum(1 for v in values if v >= 25),
        "singleton_pct": round(100 * sum(1 for v in values if v == 1) / len(values), 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.15, help="see measured defaults")
    ap.add_argument("--max-age", type=int, default=90, help="see measured defaults")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args(argv)

    by_frame = load_detections(
        args.detections, min_confidence=args.conf, keep_labels=DEFAULT_LABELS
    )
    t0 = time.time()
    rows = track_clip(
        by_frame, iou_threshold=args.iou, max_age=args.max_age, fps=args.fps
    )
    elapsed = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))

    stats = summarise(rows)
    n_frames = max(by_frame) + 1 if by_frame else 0
    print(f"{args.detections.name}: {n_frames} frames, {stats.get('detections', 0)} tracked")
    print(f"  {elapsed:.1f}s ({n_frames / max(elapsed, 0.01):.0f} fps, CPU)")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  -> {args.out}")
    print("  NOTE: purity is NOT measured here — verify tracklets at /identity")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
