import tempfile
from pathlib import Path

from footy_track.scripts.extract_frames import extract_frames


def test_extract_frames_correct_number_of_frames():
    """Tests that extract_frames extracts the correct number of frames."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        video_path = Path(
            "data/arsenal_mancity/temp_30s_chunks_for_review/arsenal_mancity_20250925_part192.mp4"
        )

        # Extract frames at 1 FPS
        extract_frames(
            input_path=video_path,
            output_dir=temp_dir_path,
            fps=1,
            img_format="png",
            quality=None,
            start=None,
            duration=None,
            width=None,
            height=None,
            prefix=None,
            start_number=0,
            keyframes_only=False,
            extra_ffmpeg_args=[],
        )

        # Check that 30 frames were created
        frames = list(temp_dir_path.glob("*.png"))
        assert len(frames) == 30
