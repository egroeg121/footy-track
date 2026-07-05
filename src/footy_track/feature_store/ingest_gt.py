"""Initial-flush driver: GT/labeller marks + the Roboflow hand-label dataset
into the DuckDB feature store (ft-lzx).

CLI:
    uv run python -m footy_track.feature_store.ingest_gt \\
        --gt-dir /mnt/storage/footy_data/ball_gt_marks \\
        --video-dir /home/george/code/footy_track/refinery/rig/eval_data/clips \\
        --db data/feature_store.duckdb \\
        [--dry-run] [--clip <stem>] [--roboflow-dir <path>]

Each ``<clip_stem>.jsonl`` in ``--gt-dir`` holds one JSON object per line:
    {"frame_index": int, "bbox": {x,y,w,h} | null, "center": {...} | null,
     "tags": [<object-class-or-skip-tag>, <provenance>]}

Object-class tags observed: player, referee, coach, person, player_sub,
in_play_ball, out_of_play_ball. Skip markers: no_ball, not_broadcast (no
detection emitted, but the frame is still recorded on the spine).
Provenance tags: labeller -> hand_label, yolo -> yolo, sam3 -> sam3.

For this initial flush every imported label gets ``reviewed=True`` and
``dataset_tag='ball_gt_marks'`` (per user instruction: treat as hand-reviewed).

Frame width/height/fps come from the matching video in ``--video-dir`` (named
``<clip_stem>.mp4``) via OpenCV when available. If no matching video is found
locally, we degrade gracefully: fall back to ``DEFAULT_FPS``/
``DEFAULT_WIDTH``/``DEFAULT_HEIGHT`` and note the frame as best-effort (a
documented limitation — see the bead comment).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from footy_track.feature_store.importers import import_roboflow
from footy_track.feature_store.schema import DetectionRow, FrameRow, GameRow, RunRow, Stage
from footy_track.feature_store.store import FeatureStore

DATASET_TAG = "ball_gt_marks"

# Degrade-gracefully defaults when no local video is found for a clip.
DEFAULT_FPS = 25.0
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

_SKIP_TAGS = {"no_ball", "not_broadcast"}
_PROVENANCE_TAGS = {"labeller", "yolo", "sam3"}
_PROVENANCE_TO_SOURCE = {"labeller": "hand_label", "yolo": "yolo", "sam3": "sam3"}
_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")


@dataclass
class GtImportReport:
    games: set[str] = field(default_factory=set)
    frames_written: int = 0
    detections_written: int = 0
    by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    clips_missing_video: set[str] = field(default_factory=set)


def _find_video(video_dir: Path, stem: str) -> Path | None:
    for suffix in _VIDEO_SUFFIXES:
        candidate = video_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _video_meta(video_path: Path) -> tuple[float, int, int]:
    """Return (fps, width, height) for a video via OpenCV."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or DEFAULT_WIDTH
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or DEFAULT_HEIGHT
    finally:
        cap.release()
    return float(fps), width, height


def _split_tags(tags: list[str]) -> tuple[str | None, str | None]:
    """Return (object_class_tag, provenance_tag) from a GT-mark tags list."""
    obj_tag = None
    prov_tag = None
    for t in tags:
        if t in _PROVENANCE_TAGS:
            prov_tag = t
        elif t not in _SKIP_TAGS:
            obj_tag = t
    return obj_tag, prov_tag


def ingest_gt_jsonl(
    store: FeatureStore,
    jsonl_path: Path,
    *,
    video_dir: Path,
    dry_run: bool = False,
) -> GtImportReport:
    """Import one ``<clip_stem>.jsonl`` GT-marks file into the store."""
    stem = jsonl_path.stem
    report = GtImportReport()

    video_path = _find_video(video_dir, stem)
    if video_path is not None:
        fps, width, height = _video_meta(video_path)
    else:
        fps, width, height = DEFAULT_FPS, DEFAULT_WIDTH, DEFAULT_HEIGHT
        report.clips_missing_video.add(stem)

    # Group raw records by frame_index, and per (frame_index, source) assign
    # sequential detection_id.
    frame_indices: set[int] = set()
    dets_by_frame_source: dict[tuple[int, str], list[DetectionRow]] = defaultdict(list)

    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        frame_index = int(rec["frame_index"])
        frame_indices.add(frame_index)

        tags = rec.get("tags", [])
        if any(t in _SKIP_TAGS for t in tags):
            continue  # frame recorded on the spine, no detection

        obj_tag, prov_tag = _split_tags(tags)
        bbox = rec.get("bbox")
        if obj_tag is None or prov_tag is None or bbox is None:
            continue

        source = _PROVENANCE_TO_SOURCE[prov_tag]
        ct = frame_index / fps
        run_id = f"gt_import_{prov_tag}"
        key = (frame_index, source)
        det_id = len(dets_by_frame_source[key])
        dets_by_frame_source[key].append(
            DetectionRow(
                game_id=stem,
                frame_index=frame_index,
                continuous_time_s=ct,
                detection_id=det_id,
                source=source,
                run_id=run_id,
                label=obj_tag,
                confidence=None,
                bbox_x=float(bbox["x"]),
                bbox_y=float(bbox["y"]),
                bbox_w=float(bbox["w"]),
                bbox_h=float(bbox["h"]),
                reviewed=True,
                dataset_tag=DATASET_TAG,
            )
        )

    if dry_run:
        report.games.add(stem)
        report.frames_written = len(frame_indices)
        for (_, source), rows in dets_by_frame_source.items():
            report.detections_written += len(rows)
            report.by_source[source] += len(rows)
        return report

    store.upsert_games([GameRow(game_id=stem, fps=fps, width=width, height=height)])
    report.games.add(stem)

    store.upsert_frames(
        [
            FrameRow(
                game_id=stem,
                frame_index=fi,
                frame_uri=f"{stem}_frame_{fi:06d}",
                width=width,
                height=height,
                continuous_time_s=fi / fps,
            )
            for fi in sorted(frame_indices)
        ]
    )
    report.frames_written = len(frame_indices)

    # One run row per provenance actually seen in this file.
    seen_sources = {source for (_, source) in dets_by_frame_source}
    for prov_tag, source in _PROVENANCE_TO_SOURCE.items():
        if source not in seen_sources:
            continue
        store.upsert_runs(
            [
                RunRow(
                    run_id=f"gt_import_{prov_tag}",
                    stage=Stage.DETECTION,
                    source=source,
                    model_name=prov_tag,
                )
            ]
        )

    for (_, source), rows in dets_by_frame_source.items():
        store.upsert_detections(rows)
        report.detections_written += len(rows)
        report.by_source[source] += len(rows)

    return report


def ingest_gt_dir(
    store: FeatureStore,
    gt_dir: Path,
    *,
    video_dir: Path,
    clip: str | None = None,
    dry_run: bool = False,
) -> GtImportReport:
    total = GtImportReport()
    paths = sorted(gt_dir.glob("*.jsonl"))
    if clip:
        paths = [p for p in paths if p.stem == clip]
    for path in paths:
        r = ingest_gt_jsonl(store, path, video_dir=video_dir, dry_run=dry_run)
        total.games |= r.games
        total.frames_written += r.frames_written
        total.detections_written += r.detections_written
        for k, v in r.by_source.items():
            total.by_source[k] += v
        total.clips_missing_video |= r.clips_missing_video
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clip", type=str, default=None, help="only ingest this clip stem")
    parser.add_argument(
        "--roboflow-dir",
        type=Path,
        default=None,
        help="Roboflow YOLO dataset dir to import as source=hand_label (skipped if omitted)",
    )
    args = parser.parse_args(argv)

    store = FeatureStore.open(":memory:" if args.dry_run else args.db)

    roboflow_report = None
    if args.roboflow_dir is not None:
        roboflow_report = import_roboflow(store, args.roboflow_dir)

    gt_report = ingest_gt_dir(
        store, args.gt_dir, video_dir=args.video_dir, clip=args.clip, dry_run=args.dry_run
    )

    print("=== ingest_gt import report ===")
    if roboflow_report is not None:
        print(
            f"roboflow: games={len(roboflow_report.games)} "
            f"frames={roboflow_report.frames_written} "
            f"detections={roboflow_report.detections_written}"
        )
    print(f"gt-marks games={len(gt_report.games)} frames={gt_report.frames_written}")
    print(f"gt-marks detections={gt_report.detections_written} by_source={dict(gt_report.by_source)}")
    if gt_report.clips_missing_video:
        print(
            f"WARNING: {len(gt_report.clips_missing_video)} clip(s) had no local video "
            f"(used default fps={DEFAULT_FPS}, {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}): "
            f"{sorted(gt_report.clips_missing_video)}"
        )

    if not args.dry_run:
        print(f"store totals: games={store.count('game')} frames={store.count('frame')} "
              f"detections={store.count('detection')} runs={store.count('run')}")


if __name__ == "__main__":
    main()
