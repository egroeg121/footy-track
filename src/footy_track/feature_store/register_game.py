"""Seed script: register a game and its clips in the feature store.

Usage
-----
    uv run python -m footy_track.feature_store.register_game

This registers the arsenal_mancity game in the feature store and populates the
clip table from the split_video_10s directory. It is idempotent — re-running
upserts in place without creating duplicate rows.

After registration, verify with:

    SELECT g.home_team, g.away_team, g.match_date, count(f.frame_index)
    FROM game g
    JOIN clip c USING(game_id)
    JOIN frame f USING(clip_id)
    GROUP BY 1, 2, 3;
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from footy_track.feature_store.schema import ClipRow, GameRow
from footy_track.feature_store.store import FeatureStore

GAME_ID = "arsenal_mancity"
HOME_TEAM = "Arsenal"
AWAY_TEAM = "Manchester City"
MATCH_DATE = date(2025, 9, 25)
COMPETITION = "Premier League 2025/26"
SOURCE_VIDEO_URI = (
    "~/Library/Mobile Documents/com~apple~CloudDocs"
    "/footy_data/arsenal_mancity/original_video/arsenal_mancity_20250925.webm"
)
CLIPS_DIR = Path(
    "~/Library/Mobile Documents/com~apple~CloudDocs"
    "/footy_data/arsenal_mancity/split_video_10s"
).expanduser()

# Clips are named arsenal_mancity_20250925_part000.mp4 etc.
_CLIP_STEM = re.compile(r"^(?P<stem>.+_part(?P<idx>\d+))$")

# Default feature store location (relative to repo root)
_DEFAULT_DB = Path(__file__).parents[5] / "data" / "feature_store.duckdb"


def build_game_row() -> GameRow:
    return GameRow(
        game_id=GAME_ID,
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
        match_date=MATCH_DATE,
        competition=COMPETITION,
        source_video_uri=SOURCE_VIDEO_URI,
    )


def build_clip_rows(clips_dir: Path = CLIPS_DIR) -> list[ClipRow]:
    rows: list[ClipRow] = []
    for path in sorted(clips_dir.glob("*.mp4")):
        m = _CLIP_STEM.match(path.stem)
        if not m:
            continue
        clip_id = m.group("stem")
        segment_index = int(m.group("idx"))
        rows.append(
            ClipRow(
                clip_id=clip_id,
                game_id=GAME_ID,
                local_path=str(path),
                segment_index=segment_index,
            )
        )
    return rows


def register(
    db_path: str | Path = _DEFAULT_DB,
    *,
    clips_dir: Path = CLIPS_DIR,
    verbose: bool = True,
) -> None:
    """Register the arsenal_mancity game and its clips in the feature store."""
    db_path = Path(db_path)
    with FeatureStore.open(db_path) as store:
        game_row = build_game_row()
        store.upsert_games([game_row])
        if verbose:
            print(f"Registered game: {GAME_ID}")

        clip_rows = build_clip_rows(clips_dir)
        store.upsert_clips(clip_rows)
        if verbose:
            print(f"Registered {len(clip_rows)} clips from {clips_dir}")

        # Verify join query from the bead spec
        df = store.query(
            """
            SELECT g.home_team, g.away_team, g.match_date, count(f.frame_index) AS frame_count
            FROM game g
            JOIN clip c USING (game_id)
            JOIN frame f USING (clip_id)
            GROUP BY 1, 2, 3
            """
        )
        if verbose:
            print("\nVerification query (frame_count expected 0 until bootstrap runs):")
            print(df.to_string(index=False) if not df.empty else "  (no frames yet — run bootstrap to populate)")

        game_count = store.count("game")
        clip_count = store.count("clip")
        if verbose:
            print(f"\nStore state: {game_count} game(s), {clip_count} clip(s)")


if __name__ == "__main__":
    register()
