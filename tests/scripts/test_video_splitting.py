import shutil
import subprocess
from pathlib import Path

import pytest

from footy_track.scripts.split_video import split_video


@pytest.fixture(scope="session")
def split_frames_path(repo_root: Path) -> Path:
    path = repo_root / "tests" / "data" / "split_videos"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    # Cleanup after tests
    if path.exists():
        for file in path.glob("*"):
            file.unlink()
        path.rmdir()


def get_video_duration(video_path: Path) -> float:
    """Returns the duration of a video in seconds."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout)


def test_split_30s_video_into_10s_chunks(video_path, split_frames_path):
    """Tests that a 30s video is split into three 10s chunks."""
    expected_chunks = 3
    expected_duration = 10
    duration_tolerance = 1

    # Split the 30s video into 10s chunks
    split_video(
        input_path=video_path,
        output_dir=split_frames_path,
        chunk_seconds=expected_duration,
        re_encode=False,
        vcodec=None,
        acodec=None,
        bitrate=None,
        extra_ffmpeg_args=[],
    )

    # Check that three chunks were created
    chunks = list(split_frames_path.glob("*.mp4"))
    assert len(chunks) == expected_chunks

    # Check that each chunk is approximately 10 seconds long
    for chunk in chunks:
        duration = get_video_duration(chunk)
        assert (
            expected_duration - duration_tolerance
            < duration
            < expected_duration + duration_tolerance
        )
