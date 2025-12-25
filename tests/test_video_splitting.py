import shutil
import subprocess
import tempfile
from pathlib import Path

from footy_track.scripts.split_video import split_video


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


def test_split_30s_video_into_10s_chunks():
    """Tests that a 30s video is split into three 10s chunks."""
    expected_chunks = 3
    expected_duration = 10
    duration_tolerance = 1

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        video_path = Path(
            "data/arsenal_mancity/temp_30s_chunks_for_review/arsenal_mancity_20250925_part192.mp4"
        )

        # Split the 30s video into 10s chunks
        split_video(
            input_path=video_path,
            output_dir=temp_dir_path,
            chunk_seconds=expected_duration,
            re_encode=False,
            vcodec=None,
            acodec=None,
            bitrate=None,
            extra_ffmpeg_args=[],
        )

        # Check that three chunks were created
        chunks = list(temp_dir_path.glob("*.mp4"))
        assert len(chunks) == expected_chunks

        # Check that each chunk is approximately 10 seconds long
        for chunk in chunks:
            duration = get_video_duration(chunk)
            assert (
                expected_duration - duration_tolerance
                < duration
                < expected_duration + duration_tolerance
            )
