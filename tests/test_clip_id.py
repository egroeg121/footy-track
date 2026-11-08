"""Tests for the canonical clip id (T-03). Names are real ones listed from
/mnt/storage/footy_data/{clips,ball_gt_marks,ballcheck_clips,machine_labels,
ball_candidates}, the match footage dirs and eval_data/clips."""

from __future__ import annotations

import duckdb
import pytest

from footy_track.clip_id import (
    aliases_of,
    canonical_clip_id,
    collisions,
    group_by_clip,
    main,
    same_clip,
    store_game_id_collisions,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Man City broadcast segments: raw (clips/, machine_labels/, candidates/),
        # labeller/eval, and the short GT form.
        ("arsenal_mancity_20250925_seg010.mp4", "arsenal_mancity_seg010"),
        ("arsenal_mancity_20250925_seg010.jsonl", "arsenal_mancity_seg010"),
        ("arsenal_mancity_seg010", "arsenal_mancity_seg010"),
        ("mancity_seg010.jsonl", "arsenal_mancity_seg010"),
        ("mancity_seg104", "arsenal_mancity_seg104"),
        (
            "/mnt/storage/footy_data/clips/arsenal_mancity_20250925_seg000.mp4",
            "arsenal_mancity_seg000",
        ),
        # Man City 10 s parts: a different cut, never merged with segNNN.
        ("arsenal_mancity_20250925_part050.mp4", "arsenal_mancity_part050"),
        ("mancity_part050", "arsenal_mancity_part050"),
        # Aston Villa.
        ("astonvilla_seg020.jsonl", "arsenal_astonvilla_seg020"),
        ("Arsenal - Aston Villa_seg020.mp4", "arsenal_astonvilla_seg020"),
        # Bournemouth first half: four spellings of the same footage.
        ("Bournemouth vs Arsenal 1_seg015.mp4", "arsenal_bournemouth_1st_seg015"),
        ("bournemouth_broadcast_seg015", "arsenal_bournemouth_1st_seg015"),
        ("bournemouth_1st_seg005", "arsenal_bournemouth_1st_seg005"),
        ("bournemouth_seg009.jsonl", "arsenal_bournemouth_1st_seg009"),
        # The example video.
        ("arsenal_example.mp4", "arsenal_example"),
        ("arsenal_mancity_example_video.mp4", "arsenal_example"),
        # Unknown names pass through (minus extension); match-level ids
        # (Roboflow game_id) are not clips and are left alone.
        ("arsenal_mancity_20250925", "arsenal_mancity_20250925"),
        ("clip_7.mp4", "clip_7"),
        ("newclip", "newclip"),
    ],
)
def test_canonical_clip_id_real_names(name: str, expected: str) -> None:
    assert canonical_clip_id(name) == expected


def test_canonical_is_idempotent() -> None:
    for name in (
        "mancity_seg010",
        "bournemouth_seg009",
        "Arsenal - Aston Villa_seg020",
    ):
        cid = canonical_clip_id(name)
        assert canonical_clip_id(cid) == cid


def test_seg_and_part_with_same_number_are_different_clips() -> None:
    assert not same_clip("mancity_seg050", "mancity_part050")


def test_aliases_and_collisions() -> None:
    names = [
        "arsenal_mancity_20250925_seg010.jsonl",
        "arsenal_mancity_seg010.jsonl",
        "mancity_seg010.jsonl",
        "mancity_seg031.jsonl",
        "astonvilla_seg020.jsonl",
    ]
    assert aliases_of("arsenal_mancity_20250925_seg010", names) == [
        "arsenal_mancity_20250925_seg010",
        "arsenal_mancity_seg010",
        "mancity_seg010",
    ]
    assert group_by_clip(names)["arsenal_astonvilla_seg020"] == ["astonvilla_seg020"]
    assert collisions(names) == {
        "arsenal_mancity_seg010": [
            "arsenal_mancity_20250925_seg010",
            "arsenal_mancity_seg010",
            "mancity_seg010",
        ]
    }


def test_store_collision_report_is_read_only(tmp_path, capsys) -> None:
    db = tmp_path / "fs.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE game (game_id VARCHAR)")
    con.execute("CREATE TABLE detection (game_id VARCHAR)")
    con.execute(
        "INSERT INTO game VALUES ('mancity_seg050'), ('arsenal_mancity_seg050')"
    )
    con.execute("INSERT INTO detection VALUES ('arsenal_mancity_20250925_seg010')")
    con.close()
    before = db.read_bytes()

    assert store_game_id_collisions(str(db)) == {
        "arsenal_mancity_seg050": ["arsenal_mancity_seg050", "mancity_seg050"]
    }
    main(["--db", str(db)])
    assert "arsenal_mancity_seg050" in capsys.readouterr().out
    assert db.read_bytes() == before
