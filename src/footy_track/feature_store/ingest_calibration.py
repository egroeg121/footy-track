"""Import camera-calibration sidecars into the feature store.

CLI:
    PYTHONPATH=src python -m footy_track.feature_store.ingest_calibration \\
        --gt-dir /mnt/storage/footy_data/ball_gt_marks \\
        --db data/feature_store.duckdb [--dry-run] [--clip <name>]

Reads, under ``--gt-dir``:

* ``calibration_gt/<canonical clip id>.json``        -> human GT
  (``source=hand_label``, ``review_status=hand``, run ``calibration_gt_import``)
* ``calibration_proposals/<canonical clip id>.json`` -> machine proposals
  (``source`` = the run's source, ``review_status`` = the frame's status)

into ``calibration`` + ``pitch_keypoint`` (one ``run`` row per proposal run,
``stage=calibration``). The file format is the shared contract described in
``docs/design/calibration.md``; H is normalised image -> pitch metres.

Clip -> game_id
---------------
``ingest_gt`` keys the ball GT by the *sidecar stem* (``game_id`` is NOT
canonicalised, see ``clip_id`` module docstring), so one physical clip can be
several games (``arsenal_mancity_seg010``, ``arsenal_mancity_20250925_seg010``,
``mancity_seg010``). Calibration files are keyed by ``canonical_clip_id``, so
each file is written to **every** game_id whose canonical id matches it and
that exists either as a ``<stem>.jsonl`` in ``--gt-dir`` or as a ``game`` row
in the store (frame indices agree across aliases: same video). With no such
alias the canonical id itself is the game_id. This makes
``detections_on_pitch`` work for the ball GT of every alias.

Re-import semantics (T-04 style): per game, the file is the source of truth.
In one transaction every ``calibration``/``pitch_keypoint`` row of that game
tagged with the importer's ``dataset_tag`` is deleted, then the parsed rows are
inserted, so deleted/edited frames leave no ghosts. Rows from any other writer
are untouched. Malformed frames are skipped with a warning, never a crash.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from footy_track.clip_id import canonical_clip_id
from footy_track.feature_store.schema import (
    H_CONVENTION,
    CalibrationRow,
    PitchKeypointRow,
    RunRow,
    Stage,
)
from footy_track.feature_store.store import FeatureStore

log = logging.getLogger(__name__)

GT_DATASET_TAG = "calibration_gt"
PROPOSALS_DATASET_TAG = "calibration_proposals"
#: run_id of every human-GT import (like ``gt_import_*`` for ball GT).
GT_RUN_ID = "calibration_gt_import"
GT_SOURCE = "hand_label"
_H_CONVENTIONS = {H_CONVENTION, "norm_image->pitch_m"}
_PROPOSAL_STATUSES = ("proposed", "accepted", "rejected", "failed")


@dataclass
class CalibrationImportReport:
    games: set[str] = field(default_factory=set)
    gt_frames: int = 0
    proposal_frames: int = 0
    keypoints: int = 0
    runs: set[str] = field(default_factory=set)
    by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_status: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rows_deleted: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    #: ``{file clip name: [game_id, ...]}`` -- where each file landed.
    game_map: dict[str, list[str]] = field(default_factory=dict)

    def warn(self, msg: str) -> None:
        log.warning(msg)
        self.warnings.append(msg)
        self.skipped += 1


# --------------------------------------------------------------------------- #
# parsing                                                                      #
# --------------------------------------------------------------------------- #


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _parse_ts(v: Any) -> datetime | None:
    if not isinstance(v, str) or not v:
        return None
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _parse_h(raw: Any) -> list[float]:
    """3x3 nested list (or flat 9) -> flat row-major list; raises ValueError."""
    if not isinstance(raw, list):
        raise ValueError("H missing")
    flat = [x for r in raw for x in r] if raw and isinstance(raw[0], list) else raw
    vals = [_num(x) for x in flat]
    if len(vals) != 9 or any(v is None for v in vals):
        raise ValueError("H is not a finite 3x3")
    return [float(v) for v in vals]  # type: ignore[arg-type]


def _parse_frame(
    rec: dict[str, Any],
    *,
    frame_index: int,
    game_id: str,
    source: str,
    run_id: str,
    dataset_tag: str,
    status: str,
    created_default: datetime | None,
) -> tuple[CalibrationRow, list[PitchKeypointRow]]:
    """One sidecar frame record -> rows. Raises ValueError when unusable."""
    is_gt = source == GT_SOURCE
    failed = status == "failed"
    H: list[float] | None = None
    if not failed:
        H = _parse_h(rec.get("H"))
        conv = rec.get("H_convention", H_CONVENTION)
        if conv not in _H_CONVENTIONS:
            raise ValueError(f"unsupported H_convention {conv!r}")

    corr = rec.get("correspondences") or []
    if not isinstance(corr, list):
        raise ValueError("correspondences is not a list")
    inliers = rec.get("inliers")
    err_px = rec.get("err_px")
    kps: list[PitchKeypointRow] = []
    for i, c in enumerate(corr):
        try:
            u, v = c["image"]
            px, py = c["pitch_xy"]
            vals = [_num(u), _num(v), _num(px), _num(py)]
            if any(x is None for x in vals):
                raise ValueError("non-numeric coordinate")
        except (TypeError, KeyError, ValueError) as e:
            log.warning(
                "%s frame %s: dropping malformed correspondence #%d (%s)",
                game_id,
                frame_index,
                i,
                e,
            )
            continue
        inl = inliers[i] if isinstance(inliers, list) and i < len(inliers) else None
        kps.append(
            PitchKeypointRow(
                game_id=game_id,
                source=source,
                run_id=run_id,
                frame_index=frame_index,
                point_idx=i,
                keypoint_id=c.get("pitch_id"),
                u=vals[0],  # type: ignore[arg-type]
                v=vals[1],  # type: ignore[arg-type]
                pitch_x=vals[2],  # type: ignore[arg-type]
                pitch_y=vals[3],  # type: ignore[arg-type]
                is_inlier=bool(inl) if isinstance(inl, bool) else None,
                err_px=_num(err_px[i]) if isinstance(err_px, list) and i < len(err_px) else None,
                dataset_tag=dataset_tag,
            )
        )

    errs = [k.err_px for k in kps if k.err_px is not None]
    n_inl = sum(1 for k in kps if k.is_inlier) if any(k.is_inlier is not None for k in kps) else None
    dims = rec.get("pitch_dims") or [None, None]
    size = rec.get("image_size") or [None, None]
    score = _num(rec.get("score"))
    if is_gt and score is None:
        score = 1.0  # human-verified H: always above the projector gate
    if score is not None:
        score = min(1.0, max(0.0, score))
    origin = rec.get("origin") if isinstance(rec.get("origin"), dict) else {}
    view = rec.get("view") if isinstance(rec.get("view"), dict) else {}
    w, h = (_num(size[0]), _num(size[1])) if len(size) == 2 else (None, None)

    row = CalibrationRow(
        game_id=game_id,
        source=source,
        run_id=run_id,
        frame_index=frame_index,
        H=H,
        quality=None if failed else score,
        mean_err_px=sum(errs) / len(errs) if errs else None,
        mean_err_m=_num(rec.get("mean_err_m")),
        distortion_k1=_k1(rec.get("distortion")),
        n_points=len(kps),
        n_inliers=n_inl,
        pitch_length=_num(dims[0]) if len(dims) == 2 else None,
        pitch_width=_num(dims[1]) if len(dims) == 2 else None,
        image_width=int(w) if w else None,
        image_height=int(h) if h else None,
        method=rec.get("method"),
        review_status=status,
        reviewed=is_gt or status in ("accepted", "rejected"),
        origin_source=origin.get("source"),
        origin_run_id=origin.get("run_id"),
        origin_status=origin.get("proposal_status"),
        failure=rec.get("failure") if failed else None,
        view_end=view.get("end"),
        camera_side=view.get("camera_side"),
        created_at=_parse_ts(rec.get("ts")) or created_default,
        dataset_tag=dataset_tag,
    )
    return row, kps


def _k1(dist: Any) -> float | None:
    """k1 of a record's ``distortion`` (``None`` = none / zero). The store (and the SQL projection
    view) only model a lens centred on the image centre."""
    if not isinstance(dist, dict) or not _num(dist.get("k1")):
        return None
    if dist.get("model", "division") != "division":
        raise ValueError(f"unknown distortion model {dist.get('model')!r}")
    c = dist.get("centre") or (0.5, 0.5)
    if abs(float(c[0]) - 0.5) > 1e-9 or abs(float(c[1]) - 0.5) > 1e-9:
        raise ValueError("distortion centre must be the image centre (0.5, 0.5)")
    return float(dist["k1"])


def _frame_index(key: Any) -> int:
    try:
        i = int(key)
    except (TypeError, ValueError):
        raise ValueError(f"bad frame key {key!r}") from None
    if i < 0:
        raise ValueError(f"negative frame key {key!r}")
    return i


@dataclass
class _Parsed:
    calib: list[CalibrationRow] = field(default_factory=list)
    kps: list[PitchKeypointRow] = field(default_factory=list)
    runs: list[RunRow] = field(default_factory=list)


def _load_json(path: Path, report: CalibrationImportReport) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        report.warn(f"{path.name}: unreadable ({e}); skipped")
        return None
    if not isinstance(data, dict):
        report.warn(f"{path.name}: top level is not an object; skipped")
        return None
    return data


def parse_gt_file(
    path: Path, game_id: str, report: CalibrationImportReport
) -> _Parsed | None:
    data = _load_json(path, report)
    if data is None:
        return None
    out = _Parsed()
    frames = data.get("frames")
    if not isinstance(frames, dict):
        report.warn(f"{path.name}: no 'frames' object")
        return out
    for key, rec in frames.items():
        try:
            if not isinstance(rec, dict):
                raise ValueError("frame record is not an object")
            fi = _frame_index(key)
            row, kps = _parse_frame(
                rec,
                frame_index=fi,
                game_id=game_id,
                source=GT_SOURCE,
                run_id=GT_RUN_ID,
                dataset_tag=GT_DATASET_TAG,
                status="hand",
                created_default=None,
            )
        except (ValueError, TypeError, KeyError) as e:
            report.warn(f"{path.name} frame {key}: {e}; skipped")
            continue
        out.calib.append(row)
        out.kps.extend(kps)
    if out.calib:
        out.runs.append(
            RunRow(
                run_id=GT_RUN_ID,
                stage=Stage.CALIBRATION,
                source=GT_SOURCE,
                model_name="manual_dlt",
                params_json=json.dumps({"importer": "ingest_calibration"}),
            )
        )
    return out


def parse_proposals_file(
    path: Path, game_id: str, report: CalibrationImportReport
) -> _Parsed | None:
    data = _load_json(path, report)
    if data is None:
        return None
    out = _Parsed()
    runs = data.get("runs")
    if not isinstance(runs, dict):
        report.warn(f"{path.name}: no 'runs' object")
        return out
    for run_id, run in runs.items():
        if not isinstance(run, dict) or not isinstance(run.get("frames"), dict):
            report.warn(f"{path.name} run {run_id}: malformed run; skipped")
            continue
        source = run.get("source")
        if not isinstance(source, str) or not source or source == GT_SOURCE:
            report.warn(f"{path.name} run {run_id}: bad source {source!r}; skipped")
            continue
        params = run.get("params") if isinstance(run.get("params"), dict) else {}
        created = _parse_ts(run.get("created"))
        n_before = len(out.calib)
        for key, rec in run["frames"].items():
            try:
                if not isinstance(rec, dict):
                    raise ValueError("frame record is not an object")
                status = rec.get("status", "proposed")
                if status not in _PROPOSAL_STATUSES:
                    raise ValueError(f"unknown status {status!r}")
                row, kps = _parse_frame(
                    rec,
                    frame_index=_frame_index(key),
                    game_id=game_id,
                    source=source,
                    run_id=run_id,
                    dataset_tag=PROPOSALS_DATASET_TAG,
                    status=status,
                    created_default=created,
                )
            except (ValueError, TypeError, KeyError) as e:
                report.warn(f"{path.name} run {run_id} frame {key}: {e}; skipped")
                continue
            out.calib.append(row)
            out.kps.extend(kps)
        if len(out.calib) > n_before:
            out.runs.append(
                RunRow(
                    run_id=run_id,
                    stage=Stage.CALIBRATION,
                    source=source,
                    model_name=str(params.get("model") or source),
                    model_version=params.get("prompt_version"),
                    params_json=json.dumps(params, sort_keys=True, default=str),
                    created_at=created,
                )
            )
    return out


# --------------------------------------------------------------------------- #
# clip -> game_id                                                              #
# --------------------------------------------------------------------------- #


def resolve_game_ids(
    clip: str, gt_dir: Path, store: FeatureStore | None = None
) -> list[str]:
    """game_ids (ball-GT sidecar stems / existing games) that are *clip*."""
    cid = canonical_clip_id(clip)
    names = {p.stem for p in gt_dir.glob("*.jsonl")}
    if store is not None:
        names |= set(store.query("SELECT game_id FROM game")["game_id"])
    found = sorted(n for n in names if canonical_clip_id(n) == cid)
    return found or [cid]


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #


def _import_dir(
    store: FeatureStore,
    directory: Path,
    parser: Any,
    dataset_tag: str,
    gt_dir: Path,
    report: CalibrationImportReport,
    *,
    clip: str | None,
    dry_run: bool,
    replace: bool,
    is_gt: bool,
) -> None:
    if not directory.is_dir():
        return
    cid_filter = canonical_clip_id(clip) if clip else None
    for path in sorted(directory.glob("*.json")):
        if cid_filter and canonical_clip_id(path.stem) != cid_filter:
            continue
        game_ids = resolve_game_ids(path.stem, gt_dir, store)
        report.game_map.setdefault(path.stem, [])
        skipped_before = report.skipped
        base = parser(path, game_ids[0], report)  # parse (and warn) once per file
        if base is None:
            continue
        if not base.calib and report.skipped > skipped_before:
            # Everything in the file was malformed: keep the earlier import
            # rather than wiping it for nothing.
            log.warning("%s: no usable frames; existing rows kept", path.name)
            continue
        for game_id in game_ids:
            parsed = _Parsed(
                calib=[r.model_copy(update={"game_id": game_id}) for r in base.calib],
                kps=[r.model_copy(update={"game_id": game_id}) for r in base.kps],
                runs=base.runs,
            )
            if game_id not in report.game_map[path.stem]:
                report.game_map[path.stem].append(game_id)
            report.games.add(game_id)
            if not dry_run:
                with store.transaction():
                    if replace:
                        d = store.delete_calibration_rows(game_id, dataset_tag)
                        report.rows_deleted += d["calibration"]
                    store.upsert_runs(parsed.runs)
                    store.upsert_calibrations(parsed.calib)
                    store.upsert_pitch_keypoints(parsed.kps)
            if is_gt:
                report.gt_frames += len(parsed.calib)
            else:
                report.proposal_frames += len(parsed.calib)
            report.keypoints += len(parsed.kps)
            report.runs |= {r.run_id for r in parsed.runs}
            for c in parsed.calib:
                report.by_source[c.source] += 1
                report.by_status[c.review_status] += 1


def ingest_calibration_dir(
    store: FeatureStore,
    gt_dir: Path,
    *,
    clip: str | None = None,
    dry_run: bool = False,
    replace: bool = True,
) -> CalibrationImportReport:
    """Import ``calibration_gt/`` and ``calibration_proposals/`` under *gt_dir*.

    With ``dry_run`` nothing is written (parse + report only).
    """
    report = CalibrationImportReport()
    common = dict(clip=clip, dry_run=dry_run, replace=replace)
    _import_dir(
        store,
        gt_dir / "calibration_gt",
        parse_gt_file,
        GT_DATASET_TAG,
        gt_dir,
        report,
        is_gt=True,
        **common,
    )
    _import_dir(
        store,
        gt_dir / "calibration_proposals",
        parse_proposals_file,
        PROPOSALS_DATASET_TAG,
        gt_dir,
        report,
        is_gt=False,
        **common,
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clip", type=str, default=None, help="only this clip (any alias)")
    parser.add_argument(
        "--replace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default: a clip's file replaces that importer's earlier rows for the game.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    # Dry run reads an existing store (alias resolution) but writes nothing.
    if args.dry_run and not args.db.exists():
        store = FeatureStore.open(":memory:")
    else:
        store = FeatureStore.open(args.db)
    r = ingest_calibration_dir(
        store, args.gt_dir, clip=args.clip, dry_run=args.dry_run, replace=args.replace
    )
    print("=== ingest_calibration report" + (" (DRY RUN)" if args.dry_run else "") + " ===")
    for name, gids in r.game_map.items():
        print(f"clip {name} -> game_id {gids}")
    print(
        f"games={len(r.games)} gt_frames={r.gt_frames} "
        f"proposal_frames={r.proposal_frames} keypoints={r.keypoints} "
        f"runs={sorted(r.runs)}"
    )
    print(f"by_source={dict(r.by_source)} by_status={dict(r.by_status)}")
    if r.rows_deleted:
        print(f"replaced: deleted {r.rows_deleted} existing calibration rows")
    if r.skipped:
        print(f"WARNING: {r.skipped} item(s) skipped (see log)")
    if not args.dry_run:
        print(
            f"store totals: calibration={store.count('calibration')} "
            f"pitch_keypoint={store.count('pitch_keypoint')}"
        )


if __name__ == "__main__":
    main()
