"""``read_frame_exact``: frame N is the N-th frame of a sequential decode, also for variable-frame-rate files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from footy_track.calibration import video as vid

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg + ffprobe")


@pytest.fixture
def vfr_video(tmp_path) -> Path:
    """64 distinct frames with holes in the timeline (variable frame rate), one long GOP."""
    out = tmp_path / "vfr.mp4"
    vf = "select='lt(mod(n,10),7)',setpts=N/FRAME_RATE/TB+floor(N/7)*0.2/TB,drawtext=text='%{n}':fontsize=20"
    base = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=25:duration=4"]
    for f in (vf, vf.split(",drawtext")[0]):  # drawtext needs libfreetype; fall back to the bare timeline
        r = subprocess.run([*base, "-vf", f, "-fps_mode", "vfr", "-g", "250", "-pix_fmt", "yuv420p", str(out)],
                           capture_output=True, check=False)
        if r.returncode == 0:
            return out
    pytest.skip("cannot synthesise a test video")


def _sequential(path: Path) -> list[np.ndarray]:
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(f)


def test_exact_matches_sequential_decode_on_vfr(vfr_video):
    vid.clear_cache()
    seq = _sequential(vfr_video)
    assert len(seq) > 20
    table = vid._pts_table(vfr_video)
    assert table is not None and len(table[1]) == len(seq)
    assert np.diff(table[1]).max() > 2 * np.diff(table[1]).min()  # really variable frame rate
    for k in (0, 1, 6, 7, 8, len(seq) // 2, len(seq) - 1):
        got = vid.read_frame_exact(vfr_video, k)
        assert got is not None and got.shape == seq[k].shape
        assert np.abs(got.astype(int) - seq[k].astype(int)).max() <= 2, k


def test_out_of_range_and_cache(vfr_video):
    vid.clear_cache()
    n = len(_sequential(vfr_video))
    assert vid.read_frame_exact(vfr_video, n) is None
    assert vid.read_frame_exact(vfr_video, -1) is None
    a = vid.read_frame_exact(vfr_video, 3)
    a[:] = 0  # callers get a private copy
    b = vid.read_frame_exact(vfr_video, 3)
    assert b.max() > 0 and len(vid._EXACT_CACHE) == 1


def test_missing_file_is_none(tmp_path):
    assert vid.read_frame_exact(tmp_path / "nope.mp4", 0) is None
