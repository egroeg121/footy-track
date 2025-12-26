from pathlib import Path

EXPECTED_FRAMES = 30


def test_extract_frames_correct_number_of_frames(frames_path, extracted_frames: list[Path]):
    """Tests that extract_frames extracts the correct number of frames."""
    frames = list(frames_path.glob("*.png"))
    assert len(frames) == EXPECTED_FRAMES
    assert len(extracted_frames) == EXPECTED_FRAMES
    assert len(frames) == EXPECTED_FRAMES
