"""Tests for scripts/build_eval_sidecars.py (ft-5hd.1).

Uses a small synthetic fixture GT-marks file to check ball-only filtering,
bbox dict->list conversion, provenance preference, and idempotency, without
touching real GT data or video files.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from build_eval_sidecars import build_sidecars, convert_gt_file  # noqa: E402

FIXTURE_ROWS = [
    # frame 0: a player row (should be dropped) and a labeller ball row (kept)
    {
        "frame_index": 0,
        "bbox": {"x": 0.1, "y": 0.2, "w": 0.02, "h": 0.03},
        "tags": ["player", "labeller"],
    },
    {
        "frame_index": 0,
        "bbox": {"x": 0.5, "y": 0.6, "w": 0.01, "h": 0.01},
        "tags": ["in_play_ball", "labeller"],
    },
    # frame 1: no ball row at all -> frame omitted from output
    {
        "frame_index": 1,
        "bbox": {"x": 0.3, "y": 0.3, "w": 0.02, "h": 0.02},
        "tags": ["referee", "labeller"],
    },
    # frame 2: two ball candidates (yolo + labeller) -> labeller wins
    {
        "frame_index": 2,
        "bbox": {"x": 0.2, "y": 0.2, "w": 0.02, "h": 0.02},
        "tags": ["out_of_play_ball", "yolo"],
    },
    {
        "frame_index": 2,
        "bbox": {"x": 0.25, "y": 0.25, "w": 0.02, "h": 0.02},
        "tags": ["out_of_play_ball", "labeller"],
    },
    # frame 3: ball absent (bbox null) still counts as a ball-tagged row
    {"frame_index": 3, "bbox": None, "tags": ["ball", "labeller"]},
]


@pytest.fixture
def gt_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    gt_dir = tmp_path / "ball_gt_marks"
    gt_dir.mkdir()
    gt_path = gt_dir / "fixture_clip.jsonl"
    with gt_path.open("w") as f:
        for row in FIXTURE_ROWS:
            f.write(json.dumps(row) + "\n")
    return gt_dir


def test_convert_gt_file_filters_and_converts(gt_fixture: pathlib.Path):
    rows = convert_gt_file(gt_fixture / "fixture_clip.jsonl")
    frame_indices = [r["frame_index"] for r in rows]

    # frame 1 (no ball row) is dropped; frames 0, 2, 3 are kept.
    assert frame_indices == [0, 2, 3]

    # bbox converted from dict to plain list.
    frame0 = rows[0]
    assert frame0["bbox"] == [0.5, 0.6, 0.01, 0.01]
    assert isinstance(frame0["bbox"], list)
    assert "in_play_ball" in frame0["tags"]

    # provenance preference: labeller wins over yolo for frame 2.
    frame2 = rows[1]
    assert frame2["bbox"] == pytest.approx([0.25, 0.25, 0.02, 0.02])
    assert "labeller" in frame2["tags"]

    # null bbox (ball absent) preserved as None.
    frame3 = rows[2]
    assert frame3["bbox"] is None


def test_build_sidecars_writes_matching_clips_only(
    gt_fixture: pathlib.Path, tmp_path: pathlib.Path
):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "fixture_clip.mp4").write_bytes(b"not a real video")
    # A GT file with no matching mp4 should be skipped entirely.
    (gt_fixture / "orphan_clip.jsonl").write_text(
        json.dumps({"frame_index": 0, "bbox": None, "tags": ["ball"]}) + "\n"
    )

    usable = build_sidecars(gt_fixture, clips_dir)

    assert [u[0] for u in usable] == ["fixture_clip"]
    stem, n_frames, n_boxes = usable[0]
    assert n_frames == 3
    assert n_boxes == 2  # frame 3 has bbox=None, doesn't count as a box

    out_path = clips_dir / "fixture_clip.jsonl"
    assert out_path.exists()
    written = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(written) == 3

    # No sidecar written for the orphan (no matching video).
    assert not (clips_dir / "orphan_clip.jsonl").exists()


def test_build_sidecars_is_idempotent(gt_fixture: pathlib.Path, tmp_path: pathlib.Path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "fixture_clip.mp4").write_bytes(b"not a real video")

    build_sidecars(gt_fixture, clips_dir)
    first = (clips_dir / "fixture_clip.jsonl").read_text()

    build_sidecars(gt_fixture, clips_dir)
    second = (clips_dir / "fixture_clip.jsonl").read_text()

    assert first == second
