"""Adapter: convert hand-label ball GT marks into ball_eval harness sidecars.

The GT marks live as JSONL files with one row *per labelled object per frame*
(player, referee, coach, ball, ...), e.g.::

    {"frame_index": 16, "bbox": {"x": .., "y": .., "w": .., "h": ..},
     "center": {"x": .., "y": ..}, "tags": ["in_play_ball", "labeller"]}

The ball_eval harness (``footy_track.ball_eval.dataset``) expects one sidecar
row *per frame*, with ``bbox`` as a plain ``[x, y, w, h]`` list (or ``null``
when the ball isn't visible)::

    {"frame_index": 16, "bbox": [x, y, w, h], "tags": ["in_play_ball", "labeller"]}

This script bridges the two: for each ``<stem>.jsonl`` GT mark file with a
matching ``<stem>.mp4`` in the clips directory, it filters to ball-only rows,
converts the bbox to list form, and writes ``eval_data/clips/<stem>.jsonl``.

Usage::

    uv run python scripts/build_eval_sidecars.py \\
        --gt-dir ~/code/footy_data/ball_gt_marks \\
        --clips-dir eval_data/clips
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

BALL_TAGS = {"in_play_ball", "out_of_play_ball", "ball"}
_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")

PROJECT_ROOT = pathlib.Path(__file__).parents[1]
GT_DIR_DEFAULT = pathlib.Path.home() / "code" / "footy_data" / "ball_gt_marks"
CLIPS_DIR_DEFAULT = PROJECT_ROOT / "eval_data" / "clips"


def _is_ball_row(tags: list[str]) -> bool:
    return any(tag in BALL_TAGS for tag in tags)


def _bbox_dict_to_list(bbox: dict | list | None) -> list[float] | None:
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        return [float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])]
    return [float(v) for v in bbox]


def convert_gt_file(gt_path: pathlib.Path) -> list[dict]:
    """Read a raw GT marks JSONL file and return harness-format frame rows.

    Groups rows by frame_index, keeps only ball-tagged rows per frame, and
    resolves multiple ball candidates in a frame by preferring
    provenance == "labeller" over any other (e.g. "yolo").

    Frames with no ball row at all are omitted (matches harness convention:
    absence of a frame means "no ground-truth label", not "ball absent" —
    the harness only tracks frames that were actually labelled).
    """
    # frame_index -> list of raw ball rows for that frame
    by_frame: dict[int, list[dict]] = {}
    with gt_path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            tags = row.get("tags") or []
            if not _is_ball_row(tags):
                continue
            by_frame.setdefault(int(row["frame_index"]), []).append(row)

    out_rows: list[dict] = []
    for frame_index in sorted(by_frame):
        candidates = by_frame[frame_index]
        chosen = next(
            (c for c in candidates if "labeller" in (c.get("tags") or [])),
            candidates[0],
        )
        out_rows.append(
            {
                "frame_index": frame_index,
                "bbox": _bbox_dict_to_list(chosen.get("bbox")),
                "tags": list(chosen.get("tags") or []),
            }
        )
    return out_rows


def _find_video(clips_dir: pathlib.Path, stem: str) -> pathlib.Path | None:
    for suffix in _VIDEO_SUFFIXES:
        candidate = clips_dir / (stem + suffix)
        if candidate.exists():
            return candidate
    return None


def build_sidecars(
    gt_dir: pathlib.Path, clips_dir: pathlib.Path
) -> list[tuple[str, int, int]]:
    """Convert every matched (GT, video) pair. Returns (stem, n_frames, n_boxes) list."""
    usable: list[tuple[str, int, int]] = []
    for gt_path in sorted(gt_dir.glob("*.jsonl")):
        stem = gt_path.stem
        video_path = _find_video(clips_dir, stem)
        if video_path is None:
            continue  # No matching .mp4 — skip per spec.

        rows = convert_gt_file(gt_path)
        out_path = clips_dir / f"{stem}.jsonl"
        with out_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        n_boxes = sum(1 for r in rows if r["bbox"] is not None)
        usable.append((stem, len(rows), n_boxes))
    return usable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=GT_DIR_DEFAULT)
    parser.add_argument("--clips-dir", type=pathlib.Path, default=CLIPS_DIR_DEFAULT)
    args = parser.parse_args()

    if not args.gt_dir.exists():
        print(f"ERROR: GT dir not found: {args.gt_dir}", file=sys.stderr)
        raise SystemExit(1)
    if not args.clips_dir.exists():
        print(f"ERROR: clips dir not found: {args.clips_dir}", file=sys.stderr)
        raise SystemExit(1)

    usable = build_sidecars(args.gt_dir, args.clips_dir)

    total_boxes = sum(n_boxes for _, _, n_boxes in usable)
    total_frames = sum(n_frames for _, n_frames, _ in usable)
    print(f"Converted {len(usable)} clips (matched .mp4 + GT marks):")
    for stem, n_frames, n_boxes in usable:
        print(f"  {stem}: {n_frames} labelled frames, {n_boxes} ball boxes")
    print(
        f"\nTotal: {len(usable)} clips, {total_frames} labelled frames, {total_boxes} ball boxes"
    )


if __name__ == "__main__":
    main()
