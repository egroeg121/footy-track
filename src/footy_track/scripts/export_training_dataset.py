"""Export a leakage-free YOLO training dataset from the feature store (ft-n2o.1).

Complete frames only (docs/design/full_frame_labelling.md sections 1-2). In multi-class mode
(``--classes``) a frame reaches the export iff it is in the ``frame_complete`` view; in the default
ball-only mode only the BALL group must be ``complete``/``absent`` (player state is not required), so ball frames with partial/unreviewed ball state are skipped. ``frame_complete`` means every class group (``ball``, ``player``) has state ``complete`` or
``absent`` in ``frame_label_state``. A missing label never implies background. Frames that are partial or
unreviewed are skipped, and the manifest counts them by reason (``frames.skipped_by_reason``).

NOTE: player completeness is written by the ``P`` "All players boxed" marker, the series review and the
legacy backfill (``--legacy-complete-clips``). Multi-class (``--classes``) exports still gate on ``frame_complete``,
so they yield only frames complete for both groups. Under the 2026-11-21 decision (two detectors) each model should
gate on its own group; a player-only mode is TODO (see docs/design/full_frame_labelling.md). The default ball-only
export already does this for the ball.

Negatives:

* a frame whose ball group is ``absent`` is written with no ball box (an empty label file when it has no
  other requested boxes): a true negative for the ball;
* boxes the user marked ``not_ball`` in Ball Check are never positives and never a class. They are written
  to ``hard_negatives.json`` in the export dir (for exported frames, with the image path) for hard-negative
  mining. Skipped when ``include_rejected`` is set (the boxes are then positives).

Classes. By default (no ``--classes``) the export is ball-only: ``ball`` / ``in_play_ball`` /
``out_of_play_ball`` collapse to ONE class ``ball`` with id 0 (the original single-class contract; ball
trainers expect nc=1). With ``--classes`` (comma list of store labels, or ``all`` = every frozen class
present in the store) each label is its own class with a FROZEN id from
``footy_track.constants.FROZEN_CLASS_NAMES`` (player=0, ball=1, referee=2, coach=3, player_sub=4,
person=5, in_play_ball=6, out_of_play_ball=7). ``data.yaml`` lists the full frozen names (nc=8) and the
manifest records the mapping, so ids never depend on what happens to be in the store. A frame carrying
boxes of an unrequested label in a requested group (e.g. a coach when only ``player,referee`` is asked)
is skipped (reason ``unrequested_labels``) rather than written with a missing box.

Assembles a YOLOv8-format dataset (``images/{train,val[,test]}``,
``labels/{train,val[,test]}``, ``data.yaml``). The train/val/test assignment is NOT computed
here: it comes from the global split authority (:mod:`footy_track.splits`, docs/data.md
"Train/eval partitioning"), which keeps it stable as clips and matches are added:

* ``train``              -> ``train/``
* ``eval_in_match``      -> ``val/``   (the ~1-in-5 clips of train matches registered as eval)
* ``eval_heldout_match`` -> left out (never in a training export); with ``--heldout-as-test``
  they become ``test/`` (evaluation only, for a trainer that reports a test split).

The split is by clip (a clip never straddles two splits). Missing/invalid registers fail closed
before anything is queried or written. The manifest reports emitted frames by label state, split and
class.

Leakage guard (the critical requirement): every clip used by the ball_eval
harness is excluded from every split. The harness eval set is the set of clips
that have a GT ``.jsonl`` sidecar under ``--eval-dir`` (default
``eval_data/clips``). Extra clips can be excluded with repeated ``--exclude-clip``.

Clip identity is canonical (T-03, ``footy_track.clip_id``): one physical clip
can sit in the store under several ``game_id`` aliases (``mancity_seg010``,
``arsenal_mancity_seg010``, ``arsenal_mancity_20250925_seg010``). Any alias of
an eval/excluded clip is excluded, and all aliases of a clip land in the same
split. The split is computed over alias groups keyed by their smallest
``game_id``, so a store without aliases splits exactly as before.

``--sources`` (comma list). Default is ``hand_label,reviewed``: human boxes plus machine boxes a human
accepted (``reviewed=true``). Other tokens are ``source`` values (e.g. ``hand_label,vittrack``);
``all`` is the explicit unrestricted pull of every canonical source (machine boxes included).

Images are sourced two ways:
  - detections whose ``frame_uri`` points at a real image file (Roboflow
    hand-label frames) are copied directly;
  - detections from GT-mark clips (synthetic ``frame_uri``) have their frame
    extracted on demand from ``<clip_stem>.mp4`` under ``--video-dir``.

The exporter tags every exported detection in the store with
``dataset_tag='<tag>:<split>'`` so the exact composition stays queryable.

CLI:
    uv run python -m footy_track.scripts.export_training_dataset \\
        --db data/feature_store.duckdb \\
        --video-dir eval_data/clips \\
        --out data/training_datasets/ball_v1 \\
        [--eval-dir eval_data/clips] [--exclude-clip <stem> ...] \\
        [--heldout-as-test] [--tag ball_v1] \\
        [--classes player,referee,person | --classes all] \\
        [--sources hand_label,reviewed | --sources all]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from footy_track import splits as authority
from footy_track.clip_id import canonical_clip_id
from footy_track.constants import FROZEN_CLASS_IDS, FROZEN_CLASS_NAMES
from footy_track.feature_store.store import FeatureStore
from footy_track.match_splits import is_heldout_clip, load_register  # noqa: F401
from footy_track.splits import EVAL_HELDOUT

log = logging.getLogger(__name__)

BALL_LABELS = ("ball", "in_play_ball", "out_of_play_ball")
PLAYER_LABELS = ("player", "player_sub", "referee", "coach", "person")
BALL_CLASS_ID = 0  # ball-only mode (no --classes): one collapsed class, id 0
#: Default ``--sources``: human boxes plus machine boxes a human accepted.
DEFAULT_SOURCES: tuple[str, ...] = ("hand_label", "reviewed")
ALL_SOURCES = "all"

# A box the user marked not_ball in Ball Check (feature_store/ball_checks.py)
# must not be exported as a positive. No verdict -> row unaffected.
_NOT_REJECTED = """
          AND NOT EXISTS (
              SELECT 1 FROM detection_review rv
              WHERE rv.verdict = 'not_ball'
                AND rv.game_id = detections_enriched.game_id
                AND rv.frame_index = detections_enriched.frame_index
                AND rv.matched_source = detections_enriched.source
                AND rv.matched_run_id = detections_enriched.run_id
                AND rv.matched_detection_id = detections_enriched.detection_id
          )"""


# Held-out (evaluation-only) matches come from the per-match split register
# (footy_track.match_splits; /mnt/storage/footy_data/match_splits.json, falling back to the
# built-in seed when absent). A held-out match never reaches a training export, checked on the
# canonical clip id in addition to the eval-dir / --exclude-clip mechanism.
# Training-pool matches keep the clip-level train/val split below.


def _eval_clip_stems(eval_dir: Path) -> set[str]:
    """Clips the ball_eval harness scores on = those with a GT .jsonl sidecar."""
    return {p.stem for p in eval_dir.glob("*.jsonl")}


def _distinct_labels(store: FeatureStore) -> list[str]:
    """Every distinct canonical label present in the store, sorted."""
    df = store.query(
        "SELECT DISTINCT label FROM detections_enriched WHERE canonical ORDER BY label"
    )
    return sorted(df["label"].tolist())


def _source_clause(
    sources: list[str] | None, *, alias: str = ""
) -> tuple[str, list[object]]:
    """SQL fragment + params restricting boxes by ``--sources`` tokens.

    ``None`` -> the default (hand_label + reviewed machine boxes); ``["all"]`` -> unrestricted; the token
    ``reviewed`` matches ``reviewed = TRUE``; anything else is a ``source`` value."""
    tokens = list(DEFAULT_SOURCES if sources is None else sources)
    if ALL_SOURCES in tokens:
        return "", []
    p = f"{alias}." if alias else ""
    names = [t for t in tokens if t != "reviewed"]
    parts: list[str] = []
    params: list[object] = []
    if names:
        parts.append(f"{p}source IN ({', '.join('?' for _ in names)})")
        params += names
    if "reviewed" in tokens:
        parts.append(f"{p}reviewed")
    if not parts:
        return " AND FALSE", []
    return f" AND ({' OR '.join(parts)})", params


def _query_detections(
    store: FeatureStore,
    labels: list[str] | tuple[str, ...],
    *,
    sources: list[str] | None = None,
    include_rejected: bool = False,
    collapse_ball: bool = False,
):
    placeholders = ", ".join("?" for _ in labels)
    params: list[object] = list(labels)
    source_sql, source_params = _source_clause(sources)
    params += source_params
    rejected_clause = "" if include_rejected else _NOT_REJECTED
    label_expr = "lower(label)" if collapse_ball else "label"
    sql = f"""
        SELECT game_id, frame_index, detection_id, source, run_id, label,
               bbox_x, bbox_y, bbox_w, bbox_h, frame_uri, dataset_tag
        FROM detections_enriched
        WHERE canonical
          AND {label_expr} IN ({placeholders}){source_sql}{rejected_clause}
        ORDER BY game_id, frame_index, detection_id
    """
    return store.query(sql, params)


def _query_ball_detections(
    store: FeatureStore,
    *,
    sources: list[str] | None = None,
    include_rejected: bool = False,
):
    """Ball boxes (collapsed labels). Unlike :func:`export`, ``sources=None`` here means unrestricted."""
    return _query_detections(
        store,
        BALL_LABELS,
        sources=[ALL_SOURCES] if sources is None else sources,
        include_rejected=include_rejected,
        collapse_ball=True,
    )


def _query_frame_states(store: FeatureStore):
    """One row per frame with any ``frame_label_state`` row: both group states, ``frame_uri`` and
    ``is_complete`` (membership of the ``frame_complete`` view, the one definition of fully labelled)."""
    return store.query(
        """
        SELECT s.game_id, s.frame_index, f.frame_uri,
               max(CASE WHEN s.cls_group = 'ball'
                        THEN CASE WHEN s.reason IS NULL THEN s.state ELSE 'partial' END END) AS ball_state,
               max(CASE WHEN s.cls_group = 'player' THEN s.state END) AS player_state,
               bool_or(c.game_id IS NOT NULL) AS is_complete
        FROM frame_label_state s
        LEFT JOIN frame f USING (game_id, frame_index)
        LEFT JOIN frame_complete c USING (game_id, frame_index)
        GROUP BY s.game_id, s.frame_index, f.frame_uri
        ORDER BY s.game_id, s.frame_index
        """
    )


def _query_stateless_frames(
    store: FeatureStore, *, sources: list[str] | None
) -> list[tuple[str, int]]:
    """Frames with in-scope canonical boxes but no ``frame_label_state`` row at all."""
    source_sql, params = _source_clause(sources, alias="d")
    df = store.query(
        f"""
        SELECT DISTINCT d.game_id, d.frame_index
        FROM detections_enriched d
        WHERE d.canonical{source_sql}
          AND NOT EXISTS (SELECT 1 FROM frame_label_state s
                          WHERE s.game_id = d.game_id AND s.frame_index = d.frame_index)
        """,
        params,
    )
    return [(r.game_id, int(r.frame_index)) for r in df.itertuples()]


def _query_frames_with_labels(
    store: FeatureStore,
    labels: list[str] | tuple[str, ...],
    *,
    sources: list[str] | None,
    exclude_rejected: bool = False,
) -> set[tuple[str, int]]:
    """Frames with canonical in-scope boxes of ``labels`` (distinct ``(game_id, frame_index)``)."""
    if not labels:
        return set()
    source_sql, params = _source_clause(sources)
    rejected = _NOT_REJECTED if exclude_rejected else ""
    df = store.query(
        f"""
        SELECT DISTINCT game_id, frame_index FROM detections_enriched
        WHERE canonical AND label IN ({", ".join("?" for _ in labels)}){source_sql}{rejected}
        """,
        [*labels, *params],
    )
    return {(r.game_id, int(r.frame_index)) for r in df.itertuples()}


def _query_not_ball_boxes(store: FeatureStore):
    """Ball Check ``not_ball`` boxes: hard negatives, never a class."""
    return store.query(
        """
        SELECT game_id, frame_index, vsource, box_index, label,
               bbox_x, bbox_y, bbox_w, bbox_h
        FROM detection_review WHERE verdict = 'not_ball'
        ORDER BY game_id, frame_index, vsource, box_index
        """
    )


def _resolve_classes(
    classes: list[str] | None,
) -> tuple[list[str], dict[str, int], bool]:
    """``(class_names, label -> id, ball_only)``. Ids come from the frozen list, never from sort order,
    except ball-only mode (no ``classes``) where the collapsed ``ball`` is the single class id 0."""
    if classes is None:
        return ["ball"], defaultdict(lambda: BALL_CLASS_ID), True
    unknown = sorted(set(classes) - set(FROZEN_CLASS_IDS))
    if unknown:
        raise ValueError(
            f"--classes {unknown} not in the frozen class list {list(FROZEN_CLASS_NAMES)}; "
            "append them to footy_track.constants.FROZEN_CLASS_NAMES first"
        )
    return list(FROZEN_CLASS_NAMES), dict(FROZEN_CLASS_IDS), False


def _split_by_clip_identity(
    game_ids: list[str], *, heldout_as_test: bool = False
) -> tuple[set[str], set[str], set[str], dict[str, set[str]], set[str]]:
    """Partition ``game_id``s with the split authority (:mod:`footy_track.splits`).

    Every alias ``game_id`` of one physical clip (T-03) gets the clip's split, and an alias of a
    held-out match is held out whichever alias it is. Returns ``(train, val, test, aliases_by_id,
    heldout)``: ``val`` is the in-match eval tier, ``test`` the heldout-match tier when
    ``heldout_as_test`` (else ``test`` is empty and those clips are returned in ``heldout`` for
    the caller to drop)."""
    aliases_by_id: dict[str, set[str]] = defaultdict(set)
    for game_id in game_ids:
        aliases_by_id[canonical_clip_id(game_id)].add(game_id)
    train: set[str] = set()
    val: set[str] = set()
    test: set[str] = set()
    heldout: set[str] = set()
    for cid, group in aliases_by_id.items():
        if any(authority.split_of_name(g) == EVAL_HELDOUT for g in group):
            split = EVAL_HELDOUT
        else:
            split = authority.split_of_name(cid)
        target = authority.export_dir(split, heldout_as_test=heldout_as_test)
        {"train": train, "val": val, "test": test, None: heldout}[target].update(group)
    return train, val, test, aliases_by_id, heldout


# Fallback source-video resolution (ft-n2o.1): the `eval_data/clips/*.mp4`
# files are, on this machine, broken symlinks into a macOS-only iCloud path
# that never synced here. The real source footage lives on
# ``--data-root`` (default ``/mnt/storage/footy_data``) under per-match
# directories with a *different* filename convention than the GT-mark clip
# stems. This table maps a clip-stem *prefix* -> (subdir, real-file suffix
# transform) so we can locate the actual video. Prefixes not covered here
# (bare ``mancity_seg*``, ``bournemouth_seg*``, ``bournemouth_broadcast_*``)
# are aliases of clips that do exist on disk (see ``footy_track.clip_id``);
# they resolve through ``--video-dir`` symlinks (``eval_data/clips``,
# ``ballcheck_clips``) and are otherwise reported as unresolved.
_FALLBACK_MATCH_DIRS: tuple[tuple[str, str, str], ...] = (
    # (clip_stem_prefix, data_root_subdir, real_filename_prefix)
    (
        "arsenal_mancity_",
        "arsenal_mancity/split_video_broadcast_frames",
        "arsenal_mancity_20250925_",
    ),
    (
        "mancity_part",
        "arsenal_mancity/split_video_10s",  # parts are 10 s cuts, not broadcast segs
        "arsenal_mancity_20250925_part",
    ),
    (
        "astonvilla_",
        "arsenal_astonvilla/split_video_broadcast_frames",
        "Arsenal - Aston Villa_",
    ),
    (
        "bournemouth_1st_",
        "arsenal_bournmouth_1st_half/split_video_broadcast_frames",
        "Bournemouth vs Arsenal 1_",
    ),
)


def _resolve_video_path(
    clip: str, video_dir: Path, data_root: Path | None
) -> Path | None:
    """Find the real video file for *clip*, following the local ``video_dir``
    first and falling back to the raw match-footage tree on ``data_root``."""
    direct = video_dir / f"{clip}.mp4"
    try:
        if direct.exists() and direct.stat().st_size > 0:
            return direct
    except OSError:
        pass  # broken symlink -> fall through to data_root

    if data_root is None:
        return None
    for prefix, subdir, real_prefix in _FALLBACK_MATCH_DIRS:
        if not clip.startswith(prefix):
            continue
        suffix = clip[len(prefix) :]  # e.g. "seg010" or "050"
        candidate = data_root / subdir / f"{real_prefix}{suffix}.mp4"
        if candidate.is_file():
            return candidate
    return None


def _extract_frame(video_path: Path, frame_index: int, out_path: Path) -> bool:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = cap.read()
        if not ok:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), bgr)
        return True
    finally:
        cap.release()


def _yolo_line(class_id: int, x: float, y: float, w: float, h: float) -> str:
    cx = min(max(x + w / 2, 0.0), 1.0)
    cy = min(max(y + h / 2, 0.0), 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _materialise_image(
    clip: str,
    frame_index: int,
    frame_uri: str | None,
    img_out: Path,
    video_dir: Path,
    data_root: Path | None,
) -> bool:
    """Put real pixels at *img_out* for one frame, or return False if none exist."""
    uri_path = Path(frame_uri) if frame_uri else None
    if uri_path and uri_path.is_file():
        shutil.copyfile(uri_path, img_out)
        return True
    video_path = _resolve_video_path(clip, video_dir, data_root)
    if video_path is None:
        return False
    return _extract_frame(video_path, frame_index, img_out)


def _split_for_clip(clip: str, val_clips: set[str], test_clips: set[str]) -> str:
    if clip in val_clips:
        return "val"
    if clip in test_clips:
        return "test"
    return "train"


def _reset_output_dirs(out_dir: Path, splits: tuple[str, ...]) -> None:
    """Remove any previous export and recreate empty per-split directories."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _tag_exported_rows(store: FeatureStore, tagged: list[tuple], tag: str) -> None:
    """Tag exported rows in the store (dataset_tag = "<tag>:<split>")."""
    for game_id, source, run_id, fi, det_id, split in tagged:
        store.query(
            "UPDATE detection SET dataset_tag = ? "
            "WHERE game_id=? AND source=? AND run_id=? AND frame_index=? AND detection_id=?",
            [f"{tag}:{split}", game_id, source, run_id, fi, det_id],
        )


def _new_split_counts() -> dict:
    return {
        "images": 0,
        "boxes": 0,
        "empty_frames": 0,
        "by_source": defaultdict(int),
        "by_class": defaultdict(int),
        "by_state": defaultdict(int),  # "<group>:<state>" -> frames
    }


def _export_clip(
    clip: str,
    frames: list[dict],
    split: str,
    out_dir: Path,
    video_dir: Path,
    data_root: Path | None,
    counts: dict,
    tagged: list,
    class_id_for_label: dict[str, int],
    class_names: list[str],
    skipped: dict[str, int],
    exported_frames: dict[tuple[str, int], str],
) -> tuple[int, int]:
    """Write images+labels for one clip's complete frames. Returns (n_images, n_boxes).

    A frame with no requested boxes (e.g. ball ``absent``) gets an empty label file: a true negative."""
    clip_images = clip_boxes = 0
    for fr in frames:
        fi = fr["frame_index"]
        frows = fr["rows"]
        base = f"{clip}_{fi:06d}"
        img_out = out_dir / "images" / split / f"{base}.jpg"
        if not _materialise_image(
            clip, fi, fr["frame_uri"], img_out, video_dir, data_root
        ):
            skipped["no_pixels"] += 1
            continue  # no pixels -> skip this frame entirely
        lines = [
            _yolo_line(
                class_id_for_label[r.label], r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h
            )
            for r in frows
        ]
        (out_dir / "labels" / split / f"{base}.txt").write_text(
            "\n".join(lines) + "\n" if lines else ""
        )
        exported_frames[(clip, fi)] = f"images/{split}/{base}.jpg"
        clip_images += 1
        clip_boxes += len(frows)
        c = counts[split]
        c["images"] += 1
        c["boxes"] += len(frows)
        c["empty_frames"] += not frows
        c["by_state"][f"ball:{fr['ball_state']}"] += 1
        c["by_state"][f"player:{fr['player_state'] or 'unreviewed'}"] += 1
        for r in frows:
            c["by_source"][r.source] += 1
            c["by_class"][class_names[class_id_for_label[r.label]]] += 1
            tagged.append(
                (clip, r.source, r.run_id, int(fi), int(r.detection_id), split)
            )
    return clip_images, clip_boxes


def _write_hard_negatives(
    store: FeatureStore,
    out_dir: Path,
    exported_frames: dict[tuple[str, int], str],
    split_of_clip: dict[str, str],
) -> int:
    """``hard_negatives.json``: Ball Check ``not_ball`` boxes on exported frames. Never a class."""
    negs = []
    for r in _query_not_ball_boxes(store).itertuples():
        key = (r.game_id, int(r.frame_index))
        if key not in exported_frames:
            continue
        negs.append(
            {
                "clip": r.game_id,
                "frame_index": int(r.frame_index),
                "split": split_of_clip[r.game_id],
                "image": exported_frames[key],
                "label": r.label,
                "bbox_xywh": [r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h],
                "review": {"vsource": r.vsource, "box_index": int(r.box_index)},
            }
        )
    (out_dir / "hard_negatives.json").write_text(
        json.dumps(
            {"note": "Ball Check not_ball boxes; never a class", "negatives": negs},
            indent=1,
        )
    )
    return len(negs)


def _write_data_yaml(out_dir: Path, class_names: list[str], has_test: bool) -> None:
    """Write data.yaml (YOLOv8 format; paths relative to the dataset root)."""
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    if has_test:
        data_yaml["test"] = "images/test"
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))


def _split_summary(counts: dict, split: str) -> dict:
    c = counts[split]
    return {
        "images": c["images"],
        "boxes": c["boxes"],
        "empty_frames": c["empty_frames"],
        "by_source": dict(c["by_source"]),
        "by_class": dict(sorted(c["by_class"].items())),
        "frames_by_state": dict(sorted(c["by_state"].items())),
    }


def _build_manifest(
    *,
    tag: str,
    class_names: list[str],
    ball_only: bool,
    sources: list[str],
    eval_clips: set[str],
    extra_exclude: set[str],
    excluded_boxes: int,
    train_clips: set[str],
    val_clips: set[str],
    test_clips: set[str],
    has_test: bool,
    counts: dict,
    per_clip: dict,
    unresolved_clips: set[str],
    aliases_by_id: dict[str, set[str]],
    skipped: dict[str, int],
    hard_negatives: int,
    heldout_boxes: int = 0,
    heldout_as_test: bool = False,
) -> dict:
    split_names = ["train", "val", *(["test"] if has_test else [])]
    frames_by_state: dict[str, int] = defaultdict(int)
    for s in split_names:
        for k, v in counts[s]["by_state"].items():
            frames_by_state[k] += v
    manifest = {
        "tag": tag,
        "split_authority": {
            "module": "footy_track.splits",
            "train": "train",
            "val": "eval_in_match",
            "test": "eval_heldout_match" if heldout_as_test else None,
            "heldout_excluded_boxes": heldout_boxes,
        },
        "classes": class_names,
        "class_mode": "ball_only" if ball_only else "frozen_multiclass",
        "class_ids": {"ball": BALL_CLASS_ID} if ball_only else dict(FROZEN_CLASS_IDS),
        "frozen_class_ids": dict(FROZEN_CLASS_IDS),
        "sources": sorted(sources),
        "frames": {
            "emitted": sum(counts[s]["images"] for s in split_names),
            "by_state": dict(sorted(frames_by_state.items())),
            "by_split": {s: counts[s]["images"] for s in split_names},
            "skipped_by_reason": dict(sorted(skipped.items())),
        },
        "hard_negatives": hard_negatives,
        "excluded_eval_clips": sorted(eval_clips),
        "extra_excluded_clips": sorted(extra_exclude),
        "excluded_ball_boxes": excluded_boxes,
        "train_clips": sorted(train_clips),
        "val_clips": sorted(val_clips),
        "train": _split_summary(counts, "train"),
        "val": _split_summary(counts, "val"),
        "per_clip": per_clip,
        "clip_aliases": {
            cid: sorted(g) for cid, g in sorted(aliases_by_id.items()) if len(g) > 1
        },
        "unresolved_clips_no_source_video": sorted(unresolved_clips),
    }
    if has_test:
        manifest["test_clips"] = sorted(test_clips)
        manifest["test"] = _split_summary(counts, "test")
    return manifest


def _frame_skip_reason(ball: str | None, player: str | None) -> str:
    """Why a frame with state rows is not in ``frame_complete``."""
    if "partial" in (ball, player):
        return "partial"
    missing = [
        g
        for g, st in (("ball", ball), ("player", player))
        if st not in ("complete", "absent")
    ]
    return "_and_".join(missing) + "_unreviewed"


def export(  # noqa: PLR0912, PLR0915
    store: FeatureStore,
    *,
    out_dir: Path,
    video_dir: Path,
    eval_dir: Path,
    extra_exclude: set[str],
    tag: str,
    data_root: Path | None = None,
    classes: list[str] | None = None,
    sources: list[str] | None = None,
    heldout_as_test: bool = False,
    include_rejected: bool = False,
) -> dict:
    """Export a leakage-free YOLO dataset of fully labelled (``frame_complete``) frames.

    Raises ``match_splits.RegisterError`` up front (before any query or write) when the match
    split register is missing/invalid.

    ``classes`` is ``None`` for the original ball-only mode (single ``ball`` class, id 0). Otherwise
    it is an explicit list of store labels (already resolved from ``--classes all``); each is its own
    class with its FROZEN id (``footy_track.constants.FROZEN_CLASS_NAMES``), and ``data.yaml`` names
    the whole frozen list.

    ``sources``: ``None`` -> default ``["hand_label", "reviewed"]`` (human boxes + machine boxes a
    human accepted); ``["all"]`` -> every canonical source; otherwise ``source`` values, where the
    token ``reviewed`` means ``reviewed = TRUE``.

    The train/val/test assignment comes from :mod:`footy_track.splits` (clip-level, sticky): train
    -> ``train``, in-match eval -> ``val``, held-out-match eval -> dropped (or ``test`` with
    ``heldout_as_test=True``). There are no fractions to choose.

    Frames not in ``frame_complete`` are skipped and counted by reason in the manifest. Boxes the user
    marked ``not_ball`` in Ball Check are dropped from the positives and written to
    ``hard_negatives.json`` (unless ``include_rejected=True``, which keeps them as positives).
    """
    load_register()  # fail closed before touching the store or out_dir
    authority.load_clip_register()
    class_names, class_id_for_label, ball_only = _resolve_classes(classes)
    eff_sources = list(DEFAULT_SOURCES if sources is None else sources)
    if ball_only:
        requested = set(BALL_LABELS)
        df = _query_detections(
            store,
            BALL_LABELS,
            sources=sources,
            include_rejected=include_rejected,
            collapse_ball=True,
        )
        unrequested_labels: list[str] = []
    else:
        requested = set(classes or [])
        df = _query_detections(
            store, sorted(requested), sources=sources, include_rejected=include_rejected
        )
        touched = (BALL_LABELS if requested & set(BALL_LABELS) else ()) + (
            PLAYER_LABELS if requested & set(PLAYER_LABELS) else ()
        )
        unrequested_labels = [lb for lb in touched if lb not in requested]
    unrequested = _query_frames_with_labels(store, unrequested_labels, sources=sources)
    # Boxes of requested labels that exist but are filtered out by --sources: a frame with only those
    # would be written as a false negative, so it is skipped instead.
    out_of_scope = (
        set()
        if ALL_SOURCES in eff_sources
        else _query_frames_with_labels(
            store,
            sorted(requested),
            sources=[ALL_SOURCES],
            exclude_rejected=not include_rejected,
        )
    )
    eval_clips = _eval_clip_stems(eval_dir)
    # Compare canonical ids, so an eval clip stored under another alias is
    # still excluded (T-03).
    excluded_ids = {canonical_clip_id(c) for c in eval_clips | extra_exclude}

    dets: dict[tuple[str, int], list] = defaultdict(list)
    excluded_boxes = 0
    for r in df.itertuples():
        if canonical_clip_id(r.game_id) in excluded_ids:
            excluded_boxes += 1
            continue
        dets[(r.game_id, int(r.frame_index))].append(r)

    states = _query_frame_states(store)
    stateless = _query_stateless_frames(store, sources=sources)
    candidate_clips = sorted(
        {g for g in states["game_id"] if canonical_clip_id(g) not in excluded_ids}
        | {g for g, _ in stateless if canonical_clip_id(g) not in excluded_ids}
    )
    has_test = heldout_as_test
    train_clips, val_clips, test_clips, aliases_by_id, heldout_clips = (
        _split_by_clip_identity(candidate_clips, heldout_as_test=heldout_as_test)
    )
    heldout_boxes = sum(len(v) for (g, _), v in dets.items() if g in heldout_clips)
    excluded_boxes += heldout_boxes

    skipped: dict[str, int] = defaultdict(int)
    frames_by_clip: dict[str, list[dict]] = defaultdict(list)
    for r in states.itertuples():
        clip = r.game_id
        if clip in heldout_clips or canonical_clip_id(clip) in excluded_ids:
            continue
        key = (clip, int(r.frame_index))
        if ball_only:
            # ball-only: the ball group alone qualifies (player state is not required)
            qualifies = r.ball_state in ("complete", "absent")
            reason = _frame_skip_reason(r.ball_state, "complete")
        else:
            qualifies = bool(r.is_complete)
            reason = _frame_skip_reason(r.ball_state, r.player_state)
        if not qualifies:
            skipped[reason] += 1
        elif key in unrequested:
            skipped["unrequested_labels"] += 1
        elif key in out_of_scope and key not in dets:
            skipped["boxes_out_of_scope"] += 1
        else:
            frames_by_clip[clip].append(
                {
                    "frame_index": int(r.frame_index),
                    "frame_uri": r.frame_uri,
                    "ball_state": r.ball_state,
                    "player_state": r.player_state,
                    "rows": dets.get(key, []),
                }
            )
    for g, _ in stateless:
        if g not in heldout_clips and canonical_clip_id(g) not in excluded_ids:
            skipped["no_state"] += 1

    # Only clips that actually have complete frames are split members.
    train_clips &= set(frames_by_clip)
    val_clips &= set(frames_by_clip)
    test_clips &= set(frames_by_clip)
    aliases_by_id = {
        cid: g for cid, g in aliases_by_id.items() if g & set(frames_by_clip)
    }

    # Sanity: no clip in more than one split.
    assert not (train_clips & val_clips), "clip leakage between splits"
    assert not (train_clips & test_clips), "clip leakage between splits"
    assert not (val_clips & test_clips), "clip leakage between splits"

    splits = ("train", "val", "test") if has_test else ("train", "val")
    counts = {split: _new_split_counts() for split in splits}
    per_clip = {}

    _reset_output_dirs(out_dir, splits)

    tagged: list[
        tuple
    ] = []  # (game_id, source, run_id, frame_index, detection_id, split)
    unresolved_clips: set[str] = set()
    exported_frames: dict[tuple[str, int], str] = {}
    split_of_clip: dict[str, str] = {}

    for clip, frames in frames_by_clip.items():
        split = _split_for_clip(clip, val_clips, test_clips)
        split_of_clip[clip] = split
        clip_images, clip_boxes = _export_clip(
            clip,
            frames,
            split,
            out_dir,
            video_dir,
            data_root,
            counts,
            tagged,
            class_id_for_label,
            class_names,
            skipped,
            exported_frames,
        )
        per_clip[clip] = {"split": split, "images": clip_images, "boxes": clip_boxes}
        if frames and not clip_images:
            unresolved_clips.add(clip)

    hard_negatives = (
        0
        if include_rejected
        else _write_hard_negatives(store, out_dir, exported_frames, split_of_clip)
    )
    _tag_exported_rows(store, tagged, tag)
    _write_data_yaml(out_dir, class_names, has_test)
    manifest = _build_manifest(
        tag=tag,
        class_names=class_names,
        ball_only=ball_only,
        sources=eff_sources,
        eval_clips=eval_clips,
        extra_exclude=extra_exclude,
        excluded_boxes=excluded_boxes,
        train_clips=train_clips,
        val_clips=val_clips,
        test_clips=test_clips,
        has_test=has_test,
        counts=counts,
        per_clip=per_clip,
        unresolved_clips=unresolved_clips,
        aliases_by_id=aliases_by_id,
        skipped=skipped,
        hard_negatives=hard_negatives,
        heldout_boxes=heldout_boxes,
        heldout_as_test=heldout_as_test,
    )
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("data/training_datasets/ball_v1")
    )
    parser.add_argument("--eval-dir", type=Path, default=Path("eval_data/clips"))
    parser.add_argument(
        "--exclude-clip", action="append", default=[], dest="exclude_clip"
    )
    parser.add_argument(
        "--heldout-as-test",
        action="store_true",
        help="Write held-out-match clips as a test/ split (evaluation only). Default: "
        "they are left out of the export entirely. val/ is always the in-match "
        "eval tier (footy_track.splits).",
    )
    parser.add_argument("--tag", type=str, default="ball_v1")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Fallback root (e.g. /mnt/storage/footy_data) searched for source "
        "video when --video-dir has a broken/missing symlink for a clip.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated store label(s) to export as their own classes with FROZEN ids "
        "(e.g. 'player,referee,person'), or 'all' for every frozen class present in the "
        "store. Omit for the ball-only single-class export (back-compat default).",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated detection 'source' values; the token 'reviewed' means machine "
        "boxes with reviewed=true. Default 'hand_label,reviewed'. 'all' = every canonical "
        "source (unreviewed machine boxes included).",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Keep boxes the user marked not_ball in Ball Check as positives (default: drop "
        "them and write them to hard_negatives.json).",
    )
    args = parser.parse_args(argv)

    store = FeatureStore.open(args.db)

    classes: list[str] | None = None
    if args.classes is not None:
        if args.classes.strip().lower() == "all":
            present = set(_distinct_labels(store))
            classes = [n for n in FROZEN_CLASS_NAMES if n in present]
            ignored = sorted(present - set(FROZEN_CLASS_NAMES))
            if ignored:
                log.warning(
                    "--classes all: labels outside the frozen list ignored: %s", ignored
                )
        else:
            classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    manifest = export(
        store,
        out_dir=args.out,
        video_dir=args.video_dir,
        eval_dir=args.eval_dir,
        extra_exclude=set(args.exclude_clip),
        heldout_as_test=args.heldout_as_test,
        data_root=args.data_root,
        tag=args.tag,
        classes=classes,
        sources=sources,
        include_rejected=args.include_rejected,
    )

    print("=== export_training_dataset manifest ===")
    print(f"classes: {manifest['classes']}")
    print(f"sources: {manifest['sources']}")
    fr = manifest["frames"]
    print(
        f"frames emitted: {fr['emitted']} by_state={fr['by_state']} by_split={fr['by_split']}"
    )
    print(f"frames skipped by reason: {fr['skipped_by_reason']}")
    print(f"hard negatives: {manifest['hard_negatives']}")
    print(
        f"excluded eval clips ({len(manifest['excluded_eval_clips'])}): "
        f"{manifest['excluded_eval_clips']}"
    )
    print(f"excluded ball boxes (in eval clips): {manifest['excluded_ball_boxes']}")
    for name in ("train", "val", "test"):
        if name not in manifest:
            continue
        m = manifest[name]
        print(
            f"{name}: clips={len(manifest[f'{name}_clips'])} images={m['images']} "
            f"boxes={m['boxes']} by_class={m['by_class']} by_source={m['by_source']}"
        )
    overlap = set(manifest["train_clips"]) & set(manifest["val_clips"])
    if "test" in manifest:
        overlap |= set(manifest["train_clips"]) & set(manifest["test_clips"])
        overlap |= set(manifest["val_clips"]) & set(manifest["test_clips"])
    print(f"train/val(/test) clip overlap (must be empty): {overlap}")
    unresolved = manifest["unresolved_clips_no_source_video"]
    if unresolved:
        print(
            f"WARNING: {len(unresolved)} clip(s) had complete frames but no locatable "
            f"source video (skipped, no images/labels emitted): {unresolved}"
        )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
