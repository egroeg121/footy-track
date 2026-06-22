"""Bootstrap the feature store from local video clips.

Reads every .mp4 / .mov under --video-dir, extracts video metadata via
OpenCV (fps, resolution, frame count), and writes one ``game`` row plus
one ``frame`` row per frame index.  All operations are idempotent — re-running
adds new clips and skips frames that already exist.

Usage
-----
    # Populate:
    uv run python -m footy_track.feature_store.bootstrap --video-dir /path/to/clips --db /path/to/store.duckdb

    # Verify:
    uv run python -m footy_track.feature_store.bootstrap --verify --db /path/to/store.duckdb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from footy_track.feature_store.schema import FrameRow, GameRow
from footy_track.feature_store.store import FeatureStore

_DEFAULT_DB = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/footy_data/feature_store.duckdb"
_VIDEO_SUFFIXES = {".mp4", ".mov"}
_FRAME_BATCH = 5000  # rows per executemany call to keep memory bounded


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    return cap


def _video_meta(cap: cv2.VideoCapture) -> tuple[float, int, int, int]:
    """Return (fps, width, height, frame_count)."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frame_count


def bootstrap(video_dir: Path, store: FeatureStore) -> None:
    videos = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in _VIDEO_SUFFIXES)
    if not videos:
        print(f"No video files found in {video_dir}", file=sys.stderr)
        return

    for video_path in tqdm(videos, desc="Clips", unit="clip"):
        game_id = video_path.stem
        cap = _open_video(video_path)
        try:
            fps, width, height, frame_count = _video_meta(cap)
        finally:
            cap.release()

        if fps <= 0:
            print(f"  Warning: {video_path.name} has fps={fps}, skipping", file=sys.stderr)
            continue
        if frame_count <= 0:
            print(f"  Warning: {video_path.name} has frame_count={frame_count}, skipping", file=sys.stderr)
            continue

        game_row = GameRow(
            game_id=game_id,
            source_video_uri=str(video_path),
            fps=fps,
            width=width,
            height=height,
        )
        store.upsert_games([game_row])

        frame_uri_base = str(video_path)
        batch: list[FrameRow] = []
        for frame_index in tqdm(range(frame_count), desc=f"  {game_id}", unit="frame", leave=False):
            frame_uri = f"{frame_uri_base}?frame={frame_index}"
            continuous_time_s = frame_index / fps
            batch.append(
                FrameRow(
                    game_id=game_id,
                    frame_index=frame_index,
                    frame_uri=frame_uri,
                    width=width,
                    height=height,
                    continuous_time_s=continuous_time_s,
                )
            )
            if len(batch) >= _FRAME_BATCH:
                store.upsert_frames(batch)
                batch.clear()
        if batch:
            store.upsert_frames(batch)


def verify(store: FeatureStore) -> None:
    game_df = store.query("SELECT game_id FROM game ORDER BY game_id")
    game_count = len(game_df)

    frame_df = store.query(
        "SELECT game_id, count(*) AS n FROM frame GROUP BY game_id ORDER BY game_id"
    )
    total_frames = int(frame_df["n"].sum()) if not frame_df.empty else 0
    avg_frames = total_frames / game_count if game_count else 0

    print(f"Games:  {game_count}")
    print(f"Frames: {total_frames} total  (avg {avg_frames:.0f} per clip)")
    print("Clips:")
    for game_id in game_df["game_id"].tolist():
        n = int(frame_df.loc[frame_df["game_id"] == game_id, "n"].iloc[0]) if not frame_df.empty else 0
        print(f"  {game_id}  ({n} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the feature store from local video clips.")
    parser.add_argument("--video-dir", type=Path, help="Directory containing .mp4 / .mov clips")
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB, help="Path to DuckDB feature store")
    parser.add_argument("--verify", action="store_true", help="Print a summary instead of populating")
    args = parser.parse_args()

    if not args.verify and args.video_dir is None:
        parser.error("--video-dir is required unless --verify is set")

    with FeatureStore.open(args.db) as store:
        if args.verify:
            verify(store)
        else:
            bootstrap(args.video_dir, store)
            print("Done.")
            verify(store)


if __name__ == "__main__":
    main()
