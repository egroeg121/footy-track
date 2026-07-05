"""Export a leakage-free ball-detection YOLO dataset from the feature store (ft-n2o.1).

Queries the ``detections_enriched`` canonical view for ball-class detections
(``ball`` / ``in_play_ball`` / ``out_of_play_ball`` -> single ``ball`` class),
assembles a YOLOv8-format dataset (``images/{train,val}``, ``labels/{train,val}``,
``data.yaml``) with a **by-clip** train/val split (a clip never straddles the
split, preventing near-duplicate frame leakage).

Leakage guard (the critical requirement): every clip used by the ball_eval
harness is excluded from both splits. The harness eval set is the set of clips
that have a GT ``.jsonl`` sidecar under ``--eval-dir`` (default
``eval_data/clips``). Extra clips can be excluded with repeated ``--exclude-clip``.

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
        [--val-fraction 0.2] [--tag ball_v1]
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


def _query_ball_detections(store: FeatureStore):
    placeholders = ", ".join("?" for _ in BALL_LABELS)
    sql = f"""
        SELECT game_id, frame_index, detection_id, source, run_id,
               bbox_x, bbox_y, bbox_w, bbox_h, frame_uri, dataset_tag
        FROM detections_enriched
        WHERE canonical
          AND lower(label) IN ({placeholders})
        ORDER BY game_id, frame_index, detection_id
    """
    return store.query(sql, list(BALL_LABELS))


def _split_clips(clips: list[str], val_fraction: float) -> tuple[set[str], set[str]]:
    """Deterministic by-clip split: assign whole clips to val until the target
    fraction of clips is reached. Sorted for reproducibility."""
    clips = sorted(clips)
    if not clips:
        return set(), set()
    n_val = max(1, round(len(clips) * val_fraction)) if len(clips) > 1 else 0
    # Spread val picks across the sorted list (every k-th) for game diversity.
    val: set[str] = set()
    if n_val:
        step = max(1, len(clips) // n_val)
        val = {clips[i] for i in range(0, len(clips), step)}
        # trim to exactly n_val
        val = set(sorted(val)[:n_val])
    train = set(clips) - val
    return train, val


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
    ("arsenal_mancity_", "arsenal_mancity/split_video_broadcast_frames", "arsenal_mancity_20250925_"),
    ("mancity_part", "arsenal_mancity/split_video_broadcast_frames", "arsenal_mancity_20250925_part"),
    ("astonvilla_", "arsenal_astonvilla/split_video_broadcast_frames", "Arsenal - Aston Villa_"),
    ("bournemouth_1st_", "arsenal_bournmouth_1st_half/split_video_broadcast_frames", "Bournemouth vs Arsenal 1_"),
)


def _resolve_video_path(clip: str, video_dir: Path, data_root: Path | None) -> Path | None:
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
        suffix = clip[len(prefix):]  # e.g. "seg010" or "050"
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


def _yolo_line(x: float, y: float, w: float, h: float) -> str:
    cx = min(max(x + w / 2, 0.0), 1.0)
    cy = min(max(y + h / 2, 0.0), 1.0)
    return f"{BALL_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


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
        if not _materialise_image(clip, fi, frame_uri[fi], img_out, video_dir, data_root):
            continue  # no pixels -> skip this frame entirely
        lines = [_yolo_line(r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h) for r in frows]
        (out_dir / "labels" / split / f"{base}.txt").write_text("\n".join(lines) + "\n")
        clip_images += 1
        clip_boxes += len(frows)
        counts[split]["images"] += 1
        counts[split]["boxes"] += len(frows)
        for r in frows:
            counts[split]["by_source"][r.source] += 1
            tagged.append((clip, r.source, r.run_id, int(fi), int(r.detection_id), split))
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
) -> dict:
    df = _query_ball_detections(store)
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
    train_clips, val_clips = _split_clips(clips, val_fraction)

    # Sanity: no clip in both splits.
    assert not (train_clips & val_clips), "clip leakage between splits"

    counts = {
        "train": {"images": 0, "boxes": 0, "by_source": defaultdict(int)},
        "val": {"images": 0, "boxes": 0, "by_source": defaultdict(int)},
    }
    per_clip = {}

    # Reset output dirs.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    tagged: list[tuple] = []  # (game_id, source, run_id, frame_index, detection_id, split)
    unresolved_clips: set[str] = set()

    for clip, rows in rows_by_clip.items():
        split = "val" if clip in val_clips else "train"
        clip_images, clip_boxes = _export_clip(
            clip, rows, split, out_dir, video_dir, data_root, counts, tagged
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
        "nc": 1,
        "names": ["ball"],
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    manifest = {
        "tag": tag,
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
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/training_datasets/ball_v1"))
    parser.add_argument("--eval-dir", type=Path, default=Path("eval_data/clips"))
    parser.add_argument("--exclude-clip", action="append", default=[], dest="exclude_clip")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tag", type=str, default="ball_v1")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Fallback root (e.g. /mnt/storage/footy_data) searched for source "
        "video when --video-dir has a broken/missing symlink for a clip.",
    )
    args = parser.parse_args(argv)

    store = FeatureStore.open(args.db)
    manifest = export(
        store,
        out_dir=args.out,
        video_dir=args.video_dir,
        eval_dir=args.eval_dir,
        extra_exclude=set(args.exclude_clip),
        val_fraction=args.val_fraction,
        data_root=args.data_root,
        tag=args.tag,
    )

    print("=== export_training_dataset manifest ===")
    print(f"excluded eval clips ({len(manifest['excluded_eval_clips'])}): "
          f"{manifest['excluded_eval_clips']}")
    print(f"excluded ball boxes (in eval clips): {manifest['excluded_ball_boxes']}")
    print(f"train: clips={len(manifest['train_clips'])} "
          f"images={manifest['train']['images']} boxes={manifest['train']['boxes']} "
          f"by_source={manifest['train']['by_source']}")
    print(f"val:   clips={len(manifest['val_clips'])} "
          f"images={manifest['val']['images']} boxes={manifest['val']['boxes']} "
          f"by_source={manifest['val']['by_source']}")
    overlap = set(manifest["train_clips"]) & set(manifest["val_clips"])
    print(f"train/val clip overlap (must be empty): {overlap}")
    unresolved = manifest["unresolved_clips_no_source_video"]
    if unresolved:
        print(
            f"WARNING: {len(unresolved)} clip(s) had ball labels but no locatable "
            f"source video (skipped, no images/labels emitted): {unresolved}"
        )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
