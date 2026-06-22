"""Ingest GT (ground-truth) label marks and Roboflow datasets into the feature store.

GT marks live as JSONL sidecars produced by the web labeller / YOLO / SAM3.
Each line is one detected object with a bbox (top-left xywh, normalised) and
a ``tags`` list whose contents are decomposed as:

- **Provenance** tags: ``labeller``, ``yolo``, ``sam3``  → ``Source`` enum
- **Label** tags: everything else except modifiers       → ``DetectionRow.label``
- **Modifier** tags: ``no_ball``, ``not_broadcast``      → skip row (frame-level signals)

One run is created per unique provenance found in a JSONL file so different
sources stay independently queryable.

Frame spine
-----------
If ``--video-dir`` is supplied, the ingester walks all .mp4/.mov/.avi/.mkv
files under that directory, reads width/height/fps via cv2 (or falls back to
1920×1080 @ 25 fps), and creates ``game`` + ``frame`` rows for every frame
index that appears in the corresponding JSONL file.  Frames that do not appear
in any GT mark file are skipped (they would have an empty frame spine entry
which is wasteful).

CLI
---
    uv run python -m footy_track.feature_store.ingest_gt \\
        --gt-dir ~/Library/.../footy_data/ball_gt_marks \\
        --video-dir ~/Library/.../footy_data \\
        --db ~/Library/.../footy_data/feature_store.duckdb

    # dry-run
    uv run python -m footy_track.feature_store.ingest_gt \\
        --gt-dir ... --dry-run

    # single clip
    uv run python -m footy_track.feature_store.ingest_gt \\
        --gt-dir ... --clip arsenal_mancity_seg010
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from footy_track.feature_store.schema import (
    DetectionRow,
    FrameRow,
    GameRow,
    RunRow,
    Source,
    Stage,
)

log = logging.getLogger(__name__)

# Tags that signal provenance (which tool produced this detection).
_PROV_TAGS: frozenset[str] = frozenset({"labeller", "yolo", "sam3"})
# Tags that are frame-level signals, not object labels — skip these rows as detections.
_MODIFIER_TAGS: frozenset[str] = frozenset({"no_ball", "not_broadcast"})

_PROV_TO_SOURCE: dict[str, str] = {
    "labeller": Source.HAND_LABEL,
    "yolo": Source.YOLO,
    "sam3": Source.SAM3,
}

_VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


@dataclass
class GtImportReport:
    """Outcome of one :func:`ingest_gt_dir` call."""

    clips_processed: int = 0
    clips_skipped: int = 0
    games_written: int = 0
    frames_written: int = 0
    detections_written: int = 0
    runs_written: int = 0
    modifier_rows_skipped: int = 0
    games: set[str] = field(default_factory=set)

    def __str__(self) -> str:
        return (
            f"clips={self.clips_processed} games={self.games_written} "
            f"frames={self.frames_written} detections={self.detections_written} "
            f"runs={self.runs_written} modifier_skipped={self.modifier_rows_skipped}"
        )


# --------------------------------------------------------------------------- #
# Video metadata helper                                                       #
# --------------------------------------------------------------------------- #


def _video_meta(video_path: Path) -> tuple[int, int, float, int]:
    """Return (width, height, fps, total_frames).  Falls back to 1920×1080@25 on error."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        log.debug("cv2 not available; using default video dims for %s", video_path.name)
        return 1920, 1080, 25.0, 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Cannot open video %s; using defaults", video_path)
        return 1920, 1080, 25.0, 0
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return w, h, fps, n


def _find_video(video_dir: Path, stem: str) -> Path | None:
    """Search *video_dir* recursively for a clip whose stem matches *stem*."""
    for suffix in _VIDEO_SUFFIXES:
        for candidate in video_dir.rglob(f"{stem}{suffix}"):
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Parse one JSONL mark file                                                   #
# --------------------------------------------------------------------------- #


def _parse_mark(
    record: dict,
    *,
    game_id: str,
    continuous_time_s: float,
    run_ids: dict[str, str],
    detection_counters: dict[str, int],
    now: datetime,
) -> DetectionRow | None:
    """Parse one GT mark record.  Returns None for modifier-only rows."""
    tags = set(record.get("tags", []))
    prov_tags = tags & _PROV_TAGS
    modifier_tags = tags & _MODIFIER_TAGS
    label_tags = tags - _PROV_TAGS - _MODIFIER_TAGS

    # Frame-level signal (no_ball / not_broadcast) — not a detection row.
    if modifier_tags and not label_tags and not prov_tags:
        return None

    # Determine provenance; default to "labeller" if ambiguous / missing.
    if "labeller" in prov_tags:
        prov = "labeller"
    elif prov_tags:
        prov = next(iter(sorted(prov_tags)))
    else:
        prov = "labeller"

    source = _PROV_TO_SOURCE[prov]
    run_id = run_ids[prov]
    label = next(iter(sorted(label_tags))) if label_tags else "unknown"

    frame_index: int = record["frame_index"]
    bbox = record["bbox"]
    key = (prov, frame_index)
    det_id = detection_counters.get(key, 0)
    detection_counters[key] = det_id + 1

    return DetectionRow(
        game_id=game_id,
        frame_index=frame_index,
        continuous_time_s=continuous_time_s,
        detection_id=det_id,
        source=source,
        run_id=run_id,
        label=label,
        confidence=None,
        bbox_x=float(bbox["x"]),
        bbox_y=float(bbox["y"]),
        bbox_w=float(bbox["w"]),
        bbox_h=float(bbox["h"]),
        created_at=now,
    )


# --------------------------------------------------------------------------- #
# Ingest one JSONL file                                                       #
# --------------------------------------------------------------------------- #


def _collect_provenances(records: list[dict]) -> set[str]:
    """Return the set of provenance labels seen in *records*."""
    provenances: set[str] = set()
    for rec in records:
        tags = set(rec.get("tags", []))
        prov_tags = tags & _PROV_TAGS
        label_tags = tags - _PROV_TAGS - _MODIFIER_TAGS
        if not (label_tags or (prov_tags and not (tags & _MODIFIER_TAGS))):
            continue
        if "labeller" in prov_tags:
            provenances.add("labeller")
        elif prov_tags:
            provenances.update(prov_tags)
        else:
            provenances.add("labeller")
    return provenances


def _resolve_video_meta(
    game_id: str, video_dir: Path | None, fps_override: float | None
) -> tuple[int, int, float, str]:
    """Return (width, height, fps, video_uri_prefix) for a clip."""
    width, height, fps = 1920, 1080, 25.0
    video_uri_prefix = game_id
    if video_dir is not None:
        video_path = _find_video(video_dir, game_id)
        if video_path:
            width, height, fps, _ = _video_meta(video_path)
            video_uri_prefix = str(video_path)
        else:
            log.warning("No video found for clip %s under %s", game_id, video_dir)
    if fps_override is not None:
        fps = fps_override
    return width, height, fps, video_uri_prefix


def _write_clip(
    store,
    game_id: str,
    records: list[dict],
    provenances: set[str],
    video_dir: Path | None,
    fps_override: float | None,
    now: datetime,
) -> GtImportReport:
    """Write one clip's rows to the store and return a partial report."""
    report = GtImportReport()
    run_ids: dict[str, str] = {prov: f"gt_import_{prov}" for prov in provenances}
    width, height, fps, video_uri_prefix = _resolve_video_meta(
        game_id, video_dir, fps_override
    )
    frame_indices: set[int] = {rec["frame_index"] for rec in records}

    run_rows = [
        RunRow(
            run_id=run_ids[prov],
            stage=Stage.DETECTION,
            source=_PROV_TO_SOURCE[prov],
            model_name="human" if prov == "labeller" else prov,
            model_version=None,
            created_at=now,
        )
        for prov in sorted(provenances)
    ]
    store.upsert_runs(run_rows)
    report.runs_written += len(run_rows)

    store.upsert_games(
        [
            GameRow(
                game_id=game_id,
                fps=fps,
                width=width,
                height=height,
                source_video_uri=video_uri_prefix if video_dir else None,
            )
        ]
    )
    report.games_written += 1

    frame_rows = [
        FrameRow(
            game_id=game_id,
            frame_index=fi,
            frame_uri=f"{video_uri_prefix}_frame_{fi:06d}",
            width=width,
            height=height,
            continuous_time_s=fi / fps,
        )
        for fi in sorted(frame_indices)
    ]
    store.upsert_frames(frame_rows)
    report.frames_written += len(frame_rows)

    detection_counters: dict[tuple, int] = {}
    det_rows: list[DetectionRow] = []
    for rec in records:
        row = _parse_mark(
            rec,
            game_id=game_id,
            continuous_time_s=rec["frame_index"] / fps,
            run_ids=run_ids,
            detection_counters=detection_counters,
            now=now,
        )
        if row is None:
            report.modifier_rows_skipped += 1
            continue
        det_rows.append(row)

    store.upsert_detections(det_rows)
    report.detections_written += len(det_rows)

    log.info(
        "Ingested %s: %d detections, %d frames, provenances=%s",
        game_id,
        len(det_rows),
        len(frame_rows),
        sorted(provenances),
    )
    return report


def ingest_gt_file(
    store,
    jsonl_path: Path,
    *,
    video_dir: Path | None = None,
    fps_override: float | None = None,
    dry_run: bool = False,
) -> GtImportReport:
    """Ingest one GT marks JSONL file into *store*.

    Returns a :class:`GtImportReport` describing what was written (or would be
    written in ``dry_run`` mode).
    """
    report = GtImportReport()
    game_id = jsonl_path.stem
    now = datetime.now(tz=UTC)

    records = [
        json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()
    ]
    if not records:
        log.info("Empty GT file: %s", jsonl_path.name)
        return report

    provenances = _collect_provenances(records)
    if not provenances:
        log.info("No actionable records in %s", jsonl_path.name)
        return report

    report.games.add(game_id)
    report.clips_processed += 1

    if dry_run:
        for rec in records:
            tags = set(rec.get("tags", []))
            is_modifier_only = (
                bool(tags & _MODIFIER_TAGS)
                and not (tags - _PROV_TAGS - _MODIFIER_TAGS)
                and not (tags & _PROV_TAGS)
            )
            if is_modifier_only:
                report.modifier_rows_skipped += 1
            else:
                report.frames_written += 1
                report.detections_written += 1
        report.games_written += 1
        report.runs_written += len(provenances)
        return report

    partial = _write_clip(
        store, game_id, records, provenances, video_dir, fps_override, now
    )
    report.games_written += partial.games_written
    report.frames_written += partial.frames_written
    report.detections_written += partial.detections_written
    report.runs_written += partial.runs_written
    report.modifier_rows_skipped += partial.modifier_rows_skipped
    return report


# --------------------------------------------------------------------------- #
# Ingest all JSONL files in a directory                                       #
# --------------------------------------------------------------------------- #


def ingest_gt_dir(
    store,
    gt_dir: Path,
    *,
    video_dir: Path | None = None,
    clip: str | None = None,
    fps_override: float | None = None,
    dry_run: bool = False,
) -> GtImportReport:
    """Ingest all (or one) GT marks JSONL files under *gt_dir*.

    Parameters
    ----------
    store:
        An open :class:`~footy_track.feature_store.store.FeatureStore`.
    gt_dir:
        Directory containing ``<clip_stem>.jsonl`` files.
    video_dir:
        If given, searched recursively for clip videos to extract frame metadata.
    clip:
        If given, process only the JSONL file whose stem matches this value.
    fps_override:
        Override the FPS used for ``continuous_time_s`` computation (useful when
        cv2 is not available or the video is absent).
    dry_run:
        Print counts without writing anything.
    """
    gt_dir = Path(gt_dir)
    total = GtImportReport()

    if clip is not None:
        jsonl_files = [gt_dir / f"{clip}.jsonl"]
        if not jsonl_files[0].exists():
            raise FileNotFoundError(f"GT marks file not found: {jsonl_files[0]}")
    else:
        jsonl_files = sorted(gt_dir.glob("*.jsonl"))

    for jsonl_path in jsonl_files:
        if not jsonl_path.exists():
            log.warning("GT file missing: %s", jsonl_path)
            total.clips_skipped += 1
            continue
        r = ingest_gt_file(
            store,
            jsonl_path,
            video_dir=video_dir,
            fps_override=fps_override,
            dry_run=dry_run,
        )
        total.clips_processed += r.clips_processed
        total.clips_skipped += r.clips_skipped
        total.games_written += r.games_written
        total.frames_written += r.frames_written
        total.detections_written += r.detections_written
        total.runs_written += r.runs_written
        total.modifier_rows_skipped += r.modifier_rows_skipped
        total.games.update(r.games)

    return total


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

_DEFAULT_GT_DIR = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "com~apple~CloudDocs"
    / "footy_data"
    / "ball_gt_marks"
)
_DEFAULT_DB = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "com~apple~CloudDocs"
    / "footy_data"
    / "feature_store.duckdb"
)


def _build_parser():
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(
        prog="python -m footy_track.feature_store.ingest_gt",
        description="Ingest GT label marks into the DuckDB feature store.",
    )
    p.add_argument(
        "--gt-dir",
        type=Path,
        default=_DEFAULT_GT_DIR,
        help="Directory of <stem>.jsonl GT mark files (default: iCloud footy_data/ball_gt_marks)",
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Root directory searched recursively for clip videos (for width/height/fps).",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DB,
        help="Path to the DuckDB feature store file (default: iCloud footy_data/feature_store.duckdb)",
    )
    p.add_argument(
        "--clip",
        default=None,
        help="Process only this clip stem (e.g. arsenal_mancity_seg010).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS for continuous_time_s (useful without cv2 or video files).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing to the database.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    from footy_track.feature_store.store import FeatureStore  # noqa: PLC0415

    if args.dry_run:
        print(f"[dry-run] GT dir  : {args.gt_dir}")
        print(f"[dry-run] Video dir: {args.video_dir}")
        print(f"[dry-run] DB       : {args.db}")
        store = FeatureStore.open(":memory:")
    else:
        print(f"Opening feature store: {args.db}")
        store = FeatureStore.open(args.db)

    with store:
        report = ingest_gt_dir(
            store,
            args.gt_dir,
            video_dir=args.video_dir,
            clip=args.clip,
            fps_override=args.fps,
            dry_run=args.dry_run,
        )

    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}Done: {report}")
    print(f"{prefix}Games: {sorted(report.games)}")


if __name__ == "__main__":
    main()
