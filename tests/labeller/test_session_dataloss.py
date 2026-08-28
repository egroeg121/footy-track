"""Regression tests for the GT-sidecar data-loss paths.

``Session._do_flush`` rewrites the WHOLE sidecar from the in-memory timeline
every couple of seconds. That makes two silent data-loss routes possible, both
of which were observed destroying real human labels on 2026-08-28:

1. ``_load_existing_marks`` drops any mark whose ``frame_index`` is beyond
   ``total_frames``. If the video is unreadable or its length is misreported
   (a dangling symlink, a different encode), EVERY mark is dropped and the next
   flush writes the survivors back — erasing the file. ``arsenal_example.jsonl``
   went from 3348 rows to 0 exactly this way.
2. An empty timeline flushes an empty string over a populated sidecar.

The fixes: refuse to flush a sidecar that did not fully load, keep a ``.bak``
copy before emptying a non-empty file, and write atomically.
"""

from __future__ import annotations

import json

import pytest

from footy_track.labeller import server as labeller_server
from footy_track.labeller.server import PROV_LABELLER, Session
from footy_track.schema import ObjectDetection


@pytest.fixture
def gt_marks_dir(tmp_path, monkeypatch):
    d = tmp_path / "ball_gt_marks"
    monkeypatch.setattr(labeller_server, "_GT_MARKS_DIR", d)
    return d


class _StemPath:
    def __init__(self, stem: str) -> None:
        self.stem = stem


def _make_session(video_stem: str, total_frames: int) -> Session:
    session = Session()
    session.video_path = _StemPath(video_stem)
    session.total_frames = total_frames
    session.timeline = [None] * total_frames
    session.no_ball_frames = set()
    session.not_broadcast_frames = set()
    return session


def _box(x: float = 0.1) -> ObjectDetection:
    return ObjectDetection(
        label="in_play_ball",
        confidence=1.0,
        x=x,
        y=0.2,
        w=0.05,
        h=0.08,
        model=PROV_LABELLER,
    )


def _write_sidecar(gt_dir, stem: str, frame_indices: list[int]) -> None:
    gt_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "frame_index": i,
                "bbox": {"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.08},
                "center": {"x": 0.125, "y": 0.24},
                "tags": ["in_play_ball", PROV_LABELLER],
            }
        )
        for i in frame_indices
    ]
    (gt_dir / f"{stem}.jsonl").write_text("\n".join(lines) + "\n")


def test_flush_refuses_when_marks_failed_to_load(gt_marks_dir):
    """A too-short total_frames must not silently delete the dropped marks."""
    _write_sidecar(gt_marks_dir, "clip_a", [0, 1, 2, 900])
    path = gt_marks_dir / "clip_a.jsonl"
    before = path.read_text()

    # total_frames=10 => the mark at frame 900 cannot be loaded.
    session = _make_session("clip_a", total_frames=10)
    session._load_existing_marks()
    assert session._load_dropped == 1

    session._do_flush()

    # The sidecar is untouched: the unloadable mark still exists on disk.
    assert path.read_text() == before
    assert "900" in path.read_text()


def test_flush_proceeds_normally_when_everything_loaded(gt_marks_dir):
    """The guard must not block ordinary saves."""
    _write_sidecar(gt_marks_dir, "clip_b", [0, 1])
    session = _make_session("clip_b", total_frames=10)
    session._load_existing_marks()
    assert session._load_dropped == 0

    session.timeline[3] = [_box(x=0.4)]
    session._do_flush()

    rows = [
        json.loads(line)
        for line in (gt_marks_dir / "clip_b.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert {r["frame_index"] for r in rows} == {0, 1, 3}


def test_emptying_a_sidecar_keeps_a_backup(gt_marks_dir):
    """Clearing all marks is allowed, but the old content must survive."""
    _write_sidecar(gt_marks_dir, "clip_c", [0, 1, 2])
    path = gt_marks_dir / "clip_c.jsonl"
    original = path.read_text()

    session = _make_session("clip_c", total_frames=10)  # empty timeline
    session._do_flush()

    assert path.read_text() == ""
    backups = list(gt_marks_dir.glob("clip_c.jsonl.bak-*"))
    assert len(backups) == 1, "previous content must be backed up before truncation"
    assert backups[0].read_text() == original


def test_flush_is_atomic_and_leaves_no_tmp_file(gt_marks_dir):
    session = _make_session("clip_d", total_frames=5)
    session.timeline[1] = [_box()]
    session._do_flush()

    assert (gt_marks_dir / "clip_d.jsonl").exists()
    assert not list(gt_marks_dir.glob("*.tmp")), "temp file must be renamed away"
