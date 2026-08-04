"""Export a leakage-free YOLO training dataset from the feature store (ft-n2o.1).

By default (no ``--classes``), queries the ``detections_enriched`` canonical
view for ball-class detections (``ball`` / ``in_play_ball`` / ``out_of_play_ball``
-> single ``ball`` class) — this is the original, back-compat behavior.

Passing ``--classes`` switches to multi-class mode: a comma-separated list of
store label strings (e.g. ``player,referee,person``), or the literal ``all``
for every distinct label present. In this mode each store label maps 1:1 to
its own YOLO class (no aliasing/collapsing, except the ball-mode collapse
above, which is exclusive to the default/no-``--classes`` path). Class ids are
assigned by sorting the requested (or discovered, for ``all``) label strings
alphabetically and taking their index — deterministic and recorded in
``data.yaml["names"]``.

Assembles a YOLOv8-format dataset (``images/{train,val[,test]}``,
``labels/{train,val[,test]}``, ``data.yaml``) with a **by-clip** split (a clip
never straddles a split, preventing near-duplicate frame leakage) in all modes.

By default only train/val are produced (``--test-fraction 0`` — the original,
back-compat two-way split). Passing ``--test-fraction`` > 0 switches to a
three-way by-clip split (train/val/test); clips are still never split across
any pair of splits, and the same deterministic assignment approach as the
two-way split is used. The recommended convention is a 70/20/10 split:
``--val-fraction 0.2 --test-fraction 0.1``.

Leakage guard (the critical requirement): every clip used by the ball_eval
harness is excluded from both splits. The harness eval set is the set of clips
that have a GT ``.jsonl`` sidecar under ``--eval-dir`` (default
``eval_data/clips``). Extra clips can be excluded with repeated ``--exclude-clip``.

``--sources`` (comma list, e.g. ``hand_label`` or ``hand_label,vittrack``)
optionally restricts detections to specific ``source`` values. Default (omitted)
is unrestricted — the original behavior of pulling every canonical source.

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
        [--val-fraction 0.2] [--test-fraction 0.1] [--tag ball_v1] \\
        [--classes player,referee,person | --classes all] \\
        [--sources hand_label,vittrack]

    # Recommended 70/20/10 split:
    uv run python -m footy_track.scripts.export_training_dataset \\
        --db data/feature_store.duckdb \\
        --video-dir eval_data/clips \\
        --out data/training_datasets/ball_v1 \\
        --val-fraction 0.2 --test-fraction 0.1 --tag ball_v1
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from footy_track.feature_store.store import FeatureStore

BALL_LABELS = ("ball", "in_play_ball", "out_of_play_ball")
BALL_CLASS_ID = 0


def _eval_clip_stems(eval_dir: Path) -> set[str]:
    """Clips the ball_eval harness scores on = those with a GT .jsonl sidecar."""
    return {p.stem for p in eval_dir.glob("*.jsonl")}


def _distinct_labels(store: FeatureStore) -> list[str]:
    """Every distinct canonical label present in the store, sorted."""
    df = store.query(
        "SELECT DISTINCT label FROM detections_enriched WHERE canonical ORDER BY label"
    )
    return sorted(df["label"].tolist())


def _query_ball_detections(store: FeatureStore, *, sources: list[str] | None = None):
    placeholders = ", ".join("?" for _ in BALL_LABELS)
    params: list[object] = list(BALL_LABELS)
    source_clause = ""
    if sources:
        source_placeholders = ", ".join("?" for _ in sources)
        source_clause = f" AND source IN ({source_placeholders})"
        params += sources
    sql = f"""
        SELECT game_id, frame_index, detection_id, source, run_id, label,
               bbox_x, bbox_y, bbox_w, bbox_h, frame_uri, dataset_tag
        FROM detections_enriched
        WHERE canonical
          AND lower(label) IN ({placeholders}){source_clause}
        ORDER BY game_id, frame_index, detection_id
    """
    return store.query(sql, params)


def _query_multiclass_detections(
    store: FeatureStore, labels: list[str], *, sources: list[str] | None = None
):
    placeholders = ", ".join("?" for _ in labels)
    params: list[object] = list(labels)
    source_clause = ""
    if sources:
        source_placeholders = ", ".join("?" for _ in sources)
        source_clause = f" AND source IN ({source_placeholders})"
        params += sources
    sql = f"""
        SELECT game_id, frame_index, detection_id, source, run_id, label,
               bbox_x, bbox_y, bbox_w, bbox_h, frame_uri, dataset_tag
        FROM detections_enriched
        WHERE canonical
          AND label IN ({placeholders}){source_clause}
        ORDER BY game_id, frame_index, detection_id
    """
    return store.query(sql, params)


def _pick_spread(clips: list[str], n: int) -> set[str]:
    """Pick *n* clips spread across the sorted list (every k-th) for game
    diversity. ``clips`` must already be sorted."""
    if not n:
        return set()
    step = max(1, len(clips) // n)
    picked = {clips[i] for i in range(0, len(clips), step)}
    return set(sorted(picked)[:n])


def _split_clips(clips: list[str], val_fraction: float) -> tuple[set[str], set[str]]:
    """Deterministic by-clip split: assign whole clips to val until the target
    fraction of clips is reached. Sorted for reproducibility."""
    clips = sorted(clips)
    if not clips:
        return set(), set()
    n_val = max(1, round(len(clips) * val_fraction)) if len(clips) > 1 else 0
    val = _pick_spread(clips, n_val)
    train = set(clips) - val
    return train, val


def _split_clips_three_way(
    clips: list[str], val_fraction: float, test_fraction: float
) -> tuple[set[str], set[str], set[str]]:
    """Deterministic by-clip three-way split: train/val/test, no clip in more
    than one split. Same spread-picking approach as the two-way split, applied
    first to carve out val, then test, from what remains.

    ``test_fraction == 0`` reduces exactly to ``_split_clips``'s train/val
    behavior (test is simply empty)."""
    clips = sorted(clips)
    if not clips:
        return set(), set(), set()
    n_val = max(1, round(len(clips) * val_fraction)) if len(clips) > 1 else 0
    val = _pick_spread(clips, n_val)

    remaining = sorted(set(clips) - val)
    n_test = 0
    if test_fraction > 0 and len(clips) > 1 and remaining:
        n_test = max(1, round(len(clips) * test_fraction))
        n_test = min(n_test, len(remaining))
    test = _pick_spread(remaining, n_test)

    train = set(clips) - val - test
    return train, val, test


# Fallback source-video resolution (ft-n2o.1): the `eval_data/clips/*.mp4`
# files are, on this machine, broken symlinks into a macOS-only iCloud path
# that never synced here. The real source footage lives on
# ``--data-root`` (default ``/mnt/storage/footy_data``) under per-match
# directories with a *different* filename convention than the GT-mark clip
# stems. This table maps a clip-stem *prefix* -> (subdir, real-file suffix
# transform) so we can locate the actual video. Prefixes not covered here
# (e.g. bare ``mancity_seg*``, ``bournemouth_seg*``, ``bournemouth_broadcast_*``)
# have no matching footage on this machine at all (verified: the relevant
# match directories are empty or absent) and are reported as unresolved
# rather than guessed at.
_FALLBACK_MATCH_DIRS: tuple[tuple[str, str, str], ...] = (
    # (clip_stem_prefix, data_root_subdir, real_filename_prefix)
    (
        "arsenal_mancity_",
        "arsenal_mancity/split_video_broadcast_frames",
        "arsenal_mancity_20250925_",
    ),
    (
        "mancity_part",
        "arsenal_mancity/split_video_broadcast_frames",
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


def _export_clip(
    clip: str,
    rows: list,
    split: str,
    out_dir: Path,
    video_dir: Path,
    data_root: Path | None,
    counts: dict,
    tagged: list,
    class_id_for_label: dict[str, int],
) -> tuple[int, int]:
    """Write images+labels for one clip's frames. Returns (n_images, n_boxes)."""
    by_frame: dict[int, list] = defaultdict(list)
    frame_uri: dict[int, str] = {}
    for r in rows:
        by_frame[r.frame_index].append(r)
        frame_uri[r.frame_index] = r.frame_uri
    clip_images = clip_boxes = 0
    for fi, frows in sorted(by_frame.items()):
        base = f"{clip}_{fi:06d}"
        img_out = out_dir / "images" / split / f"{base}.jpg"
        if not _materialise_image(
            clip, fi, frame_uri[fi], img_out, video_dir, data_root
        ):
            continue  # no pixels -> skip this frame entirely
        lines = [
            _yolo_line(
                class_id_for_label[r.label], r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h
            )
            for r in frows
        ]
        (out_dir / "labels" / split / f"{base}.txt").write_text("\n".join(lines) + "\n")
        clip_images += 1
        clip_boxes += len(frows)
        counts[split]["images"] += 1
        counts[split]["boxes"] += len(frows)
        for r in frows:
            counts[split]["by_source"][r.source] += 1
            tagged.append(
                (clip, r.source, r.run_id, int(fi), int(r.detection_id), split)
            )
    return clip_images, clip_boxes


def export(
    store: FeatureStore,
    *,
    out_dir: Path,
    video_dir: Path,
    eval_dir: Path,
    extra_exclude: set[str],
    val_fraction: float,
    tag: str,
    data_root: Path | None = None,
    classes: list[str] | None = None,
    sources: list[str] | None = None,
    test_fraction: float = 0.0,
) -> dict:
    """Export a leakage-free YOLO dataset.

    ``classes`` is ``None`` for the original ball-only mode (single ``ball``
    class, id 0). Otherwise it's an explicit list of store label strings to
    export (already resolved from ``--classes all`` if that was requested);
    each label becomes its own class, ids assigned by sorted label order.

    ``sources`` optionally restricts detections to specific ``source`` column
    values (e.g. ``["hand_label"]``); ``None``/empty means unrestricted.

    ``test_fraction`` defaults to ``0.0``: the original two-way (train/val)
    by-clip split, output unchanged. Set > 0 for a three-way (train/val/test)
    by-clip split (recommended: ``val_fraction=0.2, test_fraction=0.1`` for a
    70/20/10 split); a clip never straddles any pair of splits.
    """
    multiclass = classes is not None
    if multiclass:
        class_names = sorted(set(classes))
        class_id_for_label = {name: i for i, name in enumerate(class_names)}
        df = _query_multiclass_detections(store, class_names, sources=sources)
    else:
        class_names = ["ball"]
        class_id_for_label = defaultdict(lambda: BALL_CLASS_ID)
        df = _query_ball_detections(store, sources=sources)
    eval_clips = _eval_clip_stems(eval_dir)
    excluded = eval_clips | extra_exclude

    # Group rows by clip (game_id), skipping excluded clips.
    rows_by_clip: dict[str, list] = defaultdict(list)
    excluded_boxes = 0
    for r in df.itertuples():
        if r.game_id in excluded:
            excluded_boxes += 1
            continue
        rows_by_clip[r.game_id].append(r)

    clips = list(rows_by_clip)
    has_test = test_fraction > 0
    if has_test:
        train_clips, val_clips, test_clips = _split_clips_three_way(
            clips, val_fraction, test_fraction
        )
    else:
        train_clips, val_clips = _split_clips(clips, val_fraction)
        test_clips = set()

    # Sanity: no clip in more than one split.
    assert not (train_clips & val_clips), "clip leakage between splits"
    assert not (train_clips & test_clips), "clip leakage between splits"
    assert not (val_clips & test_clips), "clip leakage between splits"

    splits = ("train", "val", "test") if has_test else ("train", "val")
    counts = {
        split: {"images": 0, "boxes": 0, "by_source": defaultdict(int)}
        for split in splits
    }
    per_clip = {}

    # Reset output dirs.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    tagged: list[
        tuple
    ] = []  # (game_id, source, run_id, frame_index, detection_id, split)
    unresolved_clips: set[str] = set()

    for clip, rows in rows_by_clip.items():
        if clip in val_clips:
            split = "val"
        elif clip in test_clips:
            split = "test"
        else:
            split = "train"
        clip_images, clip_boxes = _export_clip(
            clip,
            rows,
            split,
            out_dir,
            video_dir,
            data_root,
            counts,
            tagged,
            class_id_for_label,
        )
        per_clip[clip] = {"split": split, "images": clip_images, "boxes": clip_boxes}
        if clip_boxes and not clip_images:
            unresolved_clips.add(clip)

    # Tag exported rows in the store (dataset_tag = "<tag>:<split>").
    for game_id, source, run_id, fi, det_id, split in tagged:
        store.query(
            "UPDATE detection SET dataset_tag = ? "
            "WHERE game_id=? AND source=? AND run_id=? AND frame_index=? AND detection_id=?",
            [f"{tag}:{split}", game_id, source, run_id, fi, det_id],
        )

    # data.yaml (YOLOv8 format; paths relative to the dataset root).
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

    manifest = {
        "tag": tag,
        "classes": class_names,
        "sources": sorted(sources) if sources else None,
        "excluded_eval_clips": sorted(eval_clips),
        "extra_excluded_clips": sorted(extra_exclude),
        "excluded_ball_boxes": excluded_boxes,
        "train_clips": sorted(train_clips),
        "val_clips": sorted(val_clips),
        "train": {
            "images": counts["train"]["images"],
            "boxes": counts["train"]["boxes"],
            "by_source": dict(counts["train"]["by_source"]),
        },
        "val": {
            "images": counts["val"]["images"],
            "boxes": counts["val"]["boxes"],
            "by_source": dict(counts["val"]["by_source"]),
        },
        "per_clip": per_clip,
        "unresolved_clips_no_source_video": sorted(unresolved_clips),
    }
    if has_test:
        manifest["test_clips"] = sorted(test_clips)
        manifest["test"] = {
            "images": counts["test"]["images"],
            "boxes": counts["test"]["boxes"],
            "by_source": dict(counts["test"]["by_source"]),
        }
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
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Fraction of clips held out as a third (test) split, by-clip like "
        "--val-fraction. Default 0.0 keeps the original train/val-only "
        "behavior exactly. Recommended convention for a 70/20/10 split: "
        "--val-fraction 0.2 --test-fraction 0.1.",
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
        help="Comma-separated store label(s) to export as their own YOLO classes "
        "(e.g. 'player,referee,person'), or 'all' for every distinct label in "
        "the store. Omit for the original ball-only single-class export "
        "(back-compat default).",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated detection 'source' values to restrict the export to "
        "(e.g. 'hand_label' or 'hand_label,vittrack'). Omit for unrestricted "
        "(all canonical sources, the original behavior).",
    )
    args = parser.parse_args(argv)

    store = FeatureStore.open(args.db)

    classes: list[str] | None = None
    if args.classes is not None:
        if args.classes.strip().lower() == "all":
            classes = _distinct_labels(store)
        else:
            classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    sources: list[str] | None = None
    if args.sources is not None:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    manifest = export(
        store,
        out_dir=args.out,
        video_dir=args.video_dir,
        eval_dir=args.eval_dir,
        extra_exclude=set(args.exclude_clip),
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        data_root=args.data_root,
        tag=args.tag,
        classes=classes,
        sources=sources,
    )

    print("=== export_training_dataset manifest ===")
    print(f"classes: {manifest['classes']}")
    print(
        f"sources: {manifest['sources'] if manifest['sources'] else 'all (unrestricted)'}"
    )
    print(
        f"excluded eval clips ({len(manifest['excluded_eval_clips'])}): "
        f"{manifest['excluded_eval_clips']}"
    )
    print(f"excluded ball boxes (in eval clips): {manifest['excluded_ball_boxes']}")
    print(
        f"train: clips={len(manifest['train_clips'])} "
        f"images={manifest['train']['images']} boxes={manifest['train']['boxes']} "
        f"by_source={manifest['train']['by_source']}"
    )
    print(
        f"val:   clips={len(manifest['val_clips'])} "
        f"images={manifest['val']['images']} boxes={manifest['val']['boxes']} "
        f"by_source={manifest['val']['by_source']}"
    )
    if "test" in manifest:
        print(
            f"test:  clips={len(manifest['test_clips'])} "
            f"images={manifest['test']['images']} boxes={manifest['test']['boxes']} "
            f"by_source={manifest['test']['by_source']}"
        )
    overlap = set(manifest["train_clips"]) & set(manifest["val_clips"])
    if "test" in manifest:
        overlap |= set(manifest["train_clips"]) & set(manifest["test_clips"])
        overlap |= set(manifest["val_clips"]) & set(manifest["test_clips"])
    print(f"train/val(/test) clip overlap (must be empty): {overlap}")
    unresolved = manifest["unresolved_clips_no_source_video"]
    if unresolved:
        print(
            f"WARNING: {len(unresolved)} clip(s) had ball labels but no locatable "
            f"source video (skipped, no images/labels emitted): {unresolved}"
        )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
