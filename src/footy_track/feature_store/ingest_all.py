"""Ingest every label source into the one feature store, idempotently.

    python -m footy_track.feature_store.ingest_all [--db PATH] [--data-root DIR]
        [--marks-dir DIR ...] [--dry-run]

Default store: ``/mnt/storage/footy_data/feature_store/feature_store.duckdb`` (env
``FOOTY_FEATURE_STORE`` overrides; ``--db`` wins). Everything under the labeller's marks
directories is read; the existing per-source ingesters do the parsing:

==================================  ===========================================================
``<marks>/*.jsonl``                 ball GT sidecars -> ``detection`` (+ ``gt_frame`` for skip markers)
``<marks>/ball_checks/*.jsonl``     Ball Check verdicts -> ``detection_review`` / ``reviewed``
``<marks>/calibration_gt/*.json``   human pitch calibration -> ``calibration`` / ``pitch_keypoint``
``<marks>/calibration_proposals``   machine calibration proposals (same tables, other source)
``<marks>/frame_type/*.json``       not-broadcast / too-few-points frame flags (``calibration``)
``<marks>/broadcast_review/*.json`` broadcast-review verdicts -> ``broadcast_label``
==================================  ===========================================================

``<marks>`` is every ``marks_dir`` of ``labeller_datasets.json`` (the live labeller's registry:
the calib set, the held-out Nottingham Forest sets, the broadcast-review set), so backup copies
next to them (``ball_gt_marks.bak-*``, ``ball_gt_marks_MERGED``) are never read.

Idempotent: every ingester replaces its own rows per clip / file, so a re-run with unchanged
sources leaves the store byte-for-byte equal in content; the run reports a content fingerprint
before and after and says ``unchanged`` when they match. A GT sidecar stem present in two marks
dirs is ingested from the first only (a warning is printed); Ball Check logs of one clip in
several dirs are merged by timestamp (the labeller's own "latest verdict wins" rule).

Single writer: an exclusive lock on ``<db>.lock`` is taken for the whole run (a second ingester
fails fast with ``StoreLocked`` instead of queueing on DuckDB's file lock).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from footy_track import splits
from footy_track.feature_store.ball_checks import import_ball_checks
from footy_track.feature_store.broadcast_labels import import_broadcast_reviews
from footy_track.feature_store.ingest_calibration import ingest_calibration_dir
from footy_track.feature_store.ingest_gt import (
    GT_RUN_PREFIX,  # noqa: F401 - re-exported for tests
    _find_video,
    ingest_gt_jsonl,
)
from footy_track.feature_store.lock import default_db_path, store_lock
from footy_track.feature_store.store import FeatureStore

log = logging.getLogger(__name__)

#: Tables whose content the "unchanged" check fingerprints (everything an ingest writes).
FINGERPRINT_TABLES = (
    "game",
    "frame",
    "run",
    "detection",
    "detection_review",
    "calibration",
    "pitch_keypoint",
    "gt_frame",
    "broadcast_label",
)

#: Clip directories searched for the video behind a sidecar (fps / size), after the registry's.
_EXTRA_VIDEO_DIRS = (
    "eval_clips_kf10",
    "ballcheck_clips_kf10",
    "clips",
    "eval_data/clips",
    "arsenal_nottinghamforest/split_video_broadcast_frames_kf10",
)


@dataclass
class IngestAllReport:
    db: str = ""
    marks_dirs: list[str] = field(default_factory=list)
    gt_clips: int = 0
    gt_detections: int = 0
    gt_frames: int = 0
    ball_check_clips: int = 0
    ball_check_verdicts: int = 0
    calibration_games: int = 0
    calibration_gt_frames: int = 0
    calibration_proposal_frames: int = 0
    broadcast_files: int = 0
    broadcast_labels: int = 0
    warnings: list[str] = field(default_factory=list)
    fingerprint_before: str = ""
    fingerprint_after: str = ""
    totals: dict[str, int] = field(default_factory=dict)

    @property
    def unchanged(self) -> bool:
        return self.fingerprint_before == self.fingerprint_after

    def lines(self) -> list[str]:
        out = [
            f"store: {self.db}",
            f"marks dirs: {', '.join(self.marks_dirs) or '(none)'}",
            f"ball GT: clips={self.gt_clips} frames={self.gt_frames} detections={self.gt_detections}",
            f"ball checks: clips={self.ball_check_clips} verdicts={self.ball_check_verdicts}",
            f"calibration: games={self.calibration_games} gt_frames={self.calibration_gt_frames} "
            f"proposal/flag_frames={self.calibration_proposal_frames}",
            f"broadcast review: files={self.broadcast_files} labels={self.broadcast_labels}",
            "store totals: " + " ".join(f"{k}={v}" for k, v in self.totals.items()),
            f"content fingerprint: {self.fingerprint_after[:16]} "
            + ("(unchanged)" if self.unchanged else f"(was {self.fingerprint_before[:16] or 'empty'})"),
        ]
        out += [f"WARNING: {w}" for w in self.warnings]
        return out


# --------------------------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------------------------
def _registry_path(root: Path) -> Path:
    return Path(os.environ.get("FOOTY_LABELLER_DATASETS") or root / "labeller_datasets.json")


def discover_sources(
    root: Path, marks_dirs: list[Path] | None = None
) -> tuple[list[Path], list[Path]]:
    """``(marks dirs, video dirs)``: from ``--marks-dir`` when given, else the labeller dataset
    registry (``labeller_datasets.json``), else ``<root>/ball_gt_marks``."""
    video: list[Path] = []
    marks: list[Path] = list(marks_dirs or [])
    reg = _registry_path(root)
    datasets: dict = {}
    try:
        datasets = json.loads(reg.read_text()).get("datasets") or {}
    except (OSError, ValueError):
        pass
    for spec in datasets.values():
        if not marks_dirs and spec.get("marks_dir"):
            marks.append(Path(spec["marks_dir"]))
        if spec.get("clips_dir"):
            video.append(Path(spec["clips_dir"]))
        video +=[Path(p) for p in spec.get("extra_clips_dirs") or []]
    if not marks:
        marks = [root / "ball_gt_marks"]
    video += [root / d for d in _EXTRA_VIDEO_DIRS]
    uniq_marks = list(dict.fromkeys(p.resolve() for p in marks if p.is_dir()))
    uniq_video = list(dict.fromkeys(p for p in video if p.is_dir()))
    return uniq_marks, uniq_video


def _have_cv2() -> bool:
    try:
        import cv2  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001 - ImportError or a broken native build
        return False
    return True


# --------------------------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------------------------
def _sidecar_markers(path: Path) -> dict[int, str | None]:
    """``{frame_index: marker}`` for every frame a sidecar mentions (marker None = has boxes)."""
    out: dict[int, str | None] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            fi = int(rec["frame_index"])
        except (ValueError, KeyError, TypeError):
            continue
        tags = rec.get("tags") or []
        cur = out.get(fi)
        if "not_broadcast" in tags:
            out[fi] = "not_broadcast"
        elif "no_ball" in tags:
            out[fi] = cur if cur == "not_broadcast" else "no_ball"
        else:
            out.setdefault(fi, None)
    return out


def _ingest_sidecars(
    store: FeatureStore,
    marks: list[Path],
    video_dirs: list[Path],
    rep: IngestAllReport,
) -> None:
    seen: dict[str, Path] = {}
    for d in marks:
        for path in sorted(d.glob("*.jsonl")):
            if path.stem in seen:
                rep.warnings.append(
                    f"GT sidecar {path.stem!r} is in {seen[path.stem].parent} and {d}; "
                    "only the first is ingested"
                )
                continue
            seen[path.stem] = path
    use_video = _have_cv2()
    if not use_video:
        rep.warnings.append("cv2 unavailable: GT games use default fps / frame size")
    empty = Path(tempfile.gettempdir()) / "footy-no-videos"
    for stem, path in sorted(seen.items()):
        vdir = next(
            (v for v in video_dirs if use_video and _find_video(v, stem)), empty
        )
        r = ingest_gt_jsonl(store, path, video_dir=vdir)
        rep.gt_clips += 1
        rep.gt_frames += r.frames_written
        rep.gt_detections += r.detections_written
        markers = _sidecar_markers(path)
        with store.transaction():
            store._conn.execute("DELETE FROM gt_frame WHERE game_id = ?", [stem])  # noqa: SLF001
            if markers:
                store._conn.executemany(  # noqa: SLF001
                    "INSERT INTO gt_frame VALUES (?,?,?)",
                    [(stem, fi, m) for fi, m in sorted(markers.items())],
                )


def _merge_ball_check_logs(marks: list[Path], out_dir: Path) -> None:
    """One log per clip across all marks dirs, lines ordered by their timestamp."""
    per_clip: dict[str, list[tuple[float, int, str]]] = {}
    n = 0
    for d in marks:
        for path in sorted((d / "ball_checks").glob("*.jsonl")):
            for raw in path.read_text().splitlines():
                if not raw.strip():
                    continue
                try:
                    ts = float(json.loads(raw).get("ts") or 0.0)
                except (ValueError, TypeError, AttributeError):
                    ts = 0.0
                per_clip.setdefault(path.stem, []).append((ts, n, raw))
                n += 1
    for clip, lines in per_clip.items():
        lines.sort(key=lambda t: (t[0], t[1]))
        (out_dir / f"{clip}.jsonl").write_text("\n".join(t[2] for t in lines) + "\n")


def fingerprint(store: FeatureStore) -> str:
    """sha256 over the sorted content of every table an ingest writes."""
    h = hashlib.sha256()
    for table in FINGERPRINT_TABLES:
        h.update(table.encode())
        rows = store._conn.execute(  # noqa: SLF001
            f"SELECT * FROM {table} ORDER BY ALL"
        ).fetchall()
        h.update(repr(rows).encode())
    return h.hexdigest()


def ingest_all(
    store: FeatureStore,
    *,
    data_root: Path | None = None,
    marks_dirs: list[Path] | None = None,
    db_label: str = "",
) -> IngestAllReport:
    """Run every ingester into ``store``. Caller holds the write lock (see :func:`run`)."""
    root = Path(data_root) if data_root else splits.data_root()
    marks, video_dirs = discover_sources(root, marks_dirs)
    rep = IngestAllReport(db=db_label, marks_dirs=[str(m) for m in marks])
    rep.fingerprint_before = fingerprint(store)

    _ingest_sidecars(store, marks, video_dirs, rep)

    with tempfile.TemporaryDirectory(prefix="ball-checks-") as tmp:
        _merge_ball_check_logs(marks, Path(tmp))
        bc = import_ball_checks(store, Path(tmp))
    rep.ball_check_clips, rep.ball_check_verdicts = bc.clips, bc.verdicts

    cal_files: dict[str, Path] = {}
    for d in marks:
        for sub in ("calibration_gt", "frame_type"):
            for p in sorted((d / sub).glob("*.json")) if (d / sub).is_dir() else []:
                key = f"{sub}/{p.stem}"
                if key in cal_files:
                    rep.warnings.append(
                        f"{key} exists in {cal_files[key].parent.parent} and {d}: "
                        "the later import replaces the earlier"
                    )
                cal_files[key] = p
    games: set[str] = set()
    for d in marks:
        cr = ingest_calibration_dir(store, d)
        games |= cr.games
        rep.calibration_gt_frames += cr.gt_frames
        rep.calibration_proposal_frames += cr.proposal_frames
    rep.calibration_games = len(games)

    br = import_broadcast_reviews(store, [d / "broadcast_review" for d in marks])
    rep.broadcast_files, rep.broadcast_labels = br.files, br.labels

    rep.fingerprint_after = fingerprint(store)
    rep.totals = {
        t: store.count(t)
        for t in (
            "game",
            "frame",
            "detection",
            "detection_review",
            "calibration",
            "pitch_keypoint",
            "gt_frame",
            "broadcast_label",
        )
    }
    return rep


def run(
    db: Path | None = None,
    *,
    data_root: Path | None = None,
    marks_dirs: list[Path] | None = None,
    dry_run: bool = False,
) -> IngestAllReport:
    """Lock, open (creating) the store, ingest everything. ``dry_run`` uses an in-memory store."""
    path = Path(db) if db else default_db_path()
    if dry_run:
        store = FeatureStore.open(":memory:")
        try:
            return ingest_all(
                store, data_root=data_root, marks_dirs=marks_dirs, db_label=":memory: (dry run)"
            )
        finally:
            store.close()
    with store_lock(path, exclusive=True):
        store = FeatureStore.open(path)
        try:
            return ingest_all(
                store, data_root=data_root, marks_dirs=marks_dirs, db_label=str(path)
            )
        finally:
            store.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=None, help="store path (default: $FOOTY_FEATURE_STORE or /mnt/storage/...)")
    ap.add_argument("--data-root", type=Path, default=None, help="footy data dir (default: $FOOTY_DATA)")
    ap.add_argument("--marks-dir", type=Path, action="append", default=None, help="marks dir(s) instead of the labeller registry")
    ap.add_argument("--dry-run", action="store_true", help="parse everything into an in-memory store")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    rep = run(args.db, data_root=args.data_root, marks_dirs=args.marks_dir, dry_run=args.dry_run)
    print("\n".join(rep.lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
