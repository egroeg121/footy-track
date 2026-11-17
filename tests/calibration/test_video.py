"""Frame reader: cv2 first, ffmpeg fallback, frame-accurate, cached per video."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from footy_track.calibration import video as vid

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

N, FPS = 30, 25


@pytest.fixture
def numbered_clip(tmp_path):
    """30 frames; frame i is a flat grey of level 20 + 7*i (decodable by value)."""
    path = tmp_path / "n.mp4"
    frames = tmp_path / "f"
    frames.mkdir()
    import cv2

    for i in range(N):
        cv2.imwrite(str(frames / f"{i:03d}.png"), np.full((64, 96, 3), 20 + 7 * i, np.uint8))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(frames / "%03d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", str(path)],
        check=True,
    )
    vid.clear_cache()
    return path


def _level(frame):
    return float(frame.mean())


def _reference(path):
    """Per-frame mean levels from a sequential cv2 decode (works on this clip)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(float(f.mean()))
    return out


def test_ffmpeg_reader_is_frame_accurate(numbered_clip):
    ref = _reference(numbered_clip)
    assert len(ref) == N and len(set(round(r) for r in ref)) == N
    for i in (0, 1, 7, 15, N - 1):  # incl. first and last frame
        f = vid._ffmpeg_frame(numbered_clip, i)
        assert abs(_level(f) - ref[i]) < 1.5, i


def test_cv2_path_used_and_cached_when_it_works(numbered_clip):
    f = vid.read_frame(numbered_clip, 10)
    assert abs(_level(f) - _reference(numbered_clip)[10]) < 1.5
    assert vid.reader_for(numbered_clip) == "cv2"


def test_black_cv2_frames_fall_back_to_ffmpeg_and_are_cached(numbered_clip, monkeypatch):
    calls = {"cv2": 0}

    def black(video, idx):
        calls["cv2"] += 1
        return np.zeros((64, 96, 3), np.uint8)

    monkeypatch.setattr(vid, "_cv2_frame", black)
    f = vid.read_frame(numbered_clip, 12)
    assert abs(_level(f) - _reference(numbered_clip)[12]) < 1.5
    assert vid.reader_for(numbered_clip) == "ffmpeg"
    assert calls["cv2"] == 1
    # known-bad video skips cv2 from now on
    f2 = vid.read_frame(numbered_clip, 20)
    assert abs(_level(f2) - _reference(numbered_clip)[20]) < 1.5
    assert calls["cv2"] == 1


def test_cv2_failure_falls_back(numbered_clip, monkeypatch):
    monkeypatch.setattr(vid, "_cv2_frame", lambda v, i: None)
    assert abs(_level(vid.read_frame(numbered_clip, 3)) - _reference(numbered_clip)[3]) < 1.5


def test_missing_video_returns_none(tmp_path):
    vid.clear_cache()
    assert vid.read_frame(tmp_path / "nope.mp4", 0) is None


def test_is_black():
    assert vid.is_black(None)
    assert vid.is_black(np.zeros((4, 4, 3), np.uint8))
    assert not vid.is_black(np.full((4, 4, 3), 100, np.uint8))
