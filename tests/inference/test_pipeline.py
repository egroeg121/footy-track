"""Pipeline tests with a fake detector and a tiny synthetic video (no weights)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from footy_track.inference.fake_detector import GAP, STRAY_FRAME, FakeDetector
from footy_track.inference.pipeline import (
    InferenceConfig,
    iter_frames,
    run_inference,
    select_device,
)
from footy_track.scripts.run_inference import main as run_inference_cli
from footy_track.scripts.track_detections import label_of

W, H, N = 320, 180, 30


def _make_video(path: Path) -> None:
    """Grey video; frame i is painted with its index so the fake can read it."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    for i in range(N):
        writer.write(np.full((H, W, 3), i * 8, dtype=np.uint8))
    writer.release()


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    _make_video(path)
    return path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_writes_expected_files_and_formats(video, tmp_path):
    out = tmp_path / "out"
    summary = run_inference(
        video, out, FakeDetector(), InferenceConfig(batch_size=4, overlay=True)
    )
    for name in (
        "detections.jsonl",
        "tracklets.jsonl",
        "ball_track.jsonl",
        "summary.json",
        "overlay.mp4",
    ):
        assert (out / name).exists(), name

    dets = _read(out / "detections.jsonl")
    # machine-label format: tags are [label, source] and label_of() recovers the label
    assert {
        "frame_index",
        "bbox",
        "center",
        "tags",
        "confidence",
        "model_id",
        "run_id",
        "reviewed",
    } <= set(dets[0])
    assert {label_of(d["tags"]) for d in dets} == {"player", "in_play_ball"}
    assert len(dets) == summary["detections"]

    ball = _read(out / "ball_track.jsonl")
    assert len(ball) == N
    assert summary["frames_processed"] == N
    assert json.loads((out / "summary.json").read_text())["config"]["stride"] == 1


def test_ball_gap_is_interpolated_and_stray_ignored(video, tmp_path):
    out = tmp_path / "out"
    run_inference(video, out, FakeDetector())
    ball = _read(out / "ball_track.jsonl")
    assert {b["track_id"] for b in ball} == {1}
    assert [b["source"] for b in ball if b["frame_index"] in GAP] == [
        "interpolated"
    ] * len(GAP)
    stray = ball[STRAY_FRAME]
    assert stray["source"] == "detected" and stray["center"]["x"] < 0.8


def test_player_ids_are_stable(video, tmp_path):
    out = tmp_path / "out"
    summary = run_inference(video, out, FakeDetector())
    rows = _read(out / "tracklets.jsonl")
    assert {r["track_id"] for r in rows} == {1, 2}
    assert len(rows) == 2 * N
    assert summary["players"]["tracklets"] == 2


def test_start_stride_and_max_frames(video, tmp_path):
    out = tmp_path / "out"
    cfg = InferenceConfig(start=4, stride=3, max_frames=5)
    summary = run_inference(video, out, FakeDetector(), cfg)
    assert [b["frame_index"] for b in _read(out / "ball_track.jsonl")] == [
        4,
        7,
        10,
        13,
        16,
    ]
    assert summary["frames_processed"] == 5


def test_iter_frames_stops_at_end(video):
    cap = cv2.VideoCapture(str(video))
    assert len(list(iter_frames(cap, N - 2, 10, 1))) == 2


def test_select_device_is_valid():
    assert select_device() in {"cuda", "mps", "cpu"}


def test_cli_rejects_missing_weights(video, tmp_path):
    with pytest.raises(SystemExit):
        run_inference_cli(
            [
                "--video",
                str(video),
                "--out",
                str(tmp_path),
                "--weights",
                str(tmp_path / "nope.pt"),
            ]
        )


REAL_WEIGHTS = Path("/mnt/storage/models/rtdetr_1920_v11/best.pt")


@pytest.mark.slow
@pytest.mark.skipif(
    not REAL_WEIGHTS.exists(), reason="RT-DETR weights not on this machine"
)
def test_real_detector_smoke(video, tmp_path):
    exit_code = run_inference_cli(
        [
            "--video",
            str(video),
            "--out",
            str(tmp_path / "out"),
            "--weights",
            str(REAL_WEIGHTS),
            "--max-frames",
            "2",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "out" / "summary.json").exists()
