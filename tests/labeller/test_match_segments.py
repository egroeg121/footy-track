"""Calibration editor fallback to the full-match broadcast segments (``labeller.match_segments``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from footy_track.labeller import calibration as cal
from footy_track.labeller import calibration_match as cm
from footy_track.labeller import match_segments as ms

from ..calibration.test_geometry import H, W
from .test_calibration import KP, _body, _corr  # noqa: F401

BOURN = "Bournemouth vs Arsenal 1_seg087.mp4"


def _match(root: Path, name: str, files: list[str], *, touch: bool = True) -> Path:
    d = root / name / "split_video_broadcast_frames"
    d.mkdir(parents=True)
    (d / "segments.json").write_text(json.dumps({
        "fps": 50.0, "segments": [{"index": i, "file": f, "frame_count": 10} for i, f in enumerate(files)],
    }))
    if touch:
        for f in files:
            if f == Path(f).name:
                (d / f).touch()
    return d


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "footy_data"
    root.mkdir()
    monkeypatch.setenv("FOOTY_DATA_DIR", str(root))
    ms.clear_cache()
    cm._EDITOR_CACHE.clear()
    yield root
    ms.clear_cache()
    cm._EDITOR_CACHE.clear()


def test_index_maps_canonical_id_to_segment_file(data):
    bd = _match(data, "arsenal_bournmouth_1st_half", [BOURN, "Bournemouth vs Arsenal 1_seg003.mp4"])
    vd = _match(data, "arsenal_astonvilla", ["Arsenal - Aston Villa_seg110.mp4"])
    idx = ms.segment_index()
    assert idx["arsenal_bournemouth_1st_seg087"] == bd / BOURN
    assert idx["arsenal_bournemouth_1st_seg003"] == bd / "Bournemouth vs Arsenal 1_seg003.mp4"
    assert idx["arsenal_astonvilla_seg110"] == vd / "Arsenal - Aston Villa_seg110.mp4"
    assert ms.ambiguous() == {}


def test_find_accepts_any_alias_of_the_clip(data):
    bd = _match(data, "arsenal_bournmouth_1st_half", [BOURN])
    for alias in ("arsenal_bournemouth_1st_seg087", "bournemouth_seg087", "Bournemouth vs Arsenal 1_seg087"):
        assert ms.find_segment_video(alias) == bd / BOURN


def test_unknown_missing_and_hostile_clips_resolve_to_none(data):
    d = _match(data, "m", ["m_seg000.mp4", "m_seg001.mp4"])
    (d / "m_seg001.mp4").unlink()  # listed but gone
    (data / "secret.mp4").touch()
    assert ms.find_segment_video("m_seg000") == d / "m_seg000.mp4"
    for bad in ("m_seg001", "nope_seg000", "", "..", "../secret", "../../secret", "..%2Fsecret", "/etc/passwd",
                "m_seg000/../../secret", "secret"):
        assert ms.find_segment_video(bad) is None, bad


def test_segments_json_entries_must_be_bare_video_names(data):
    (data / "secret.mp4").touch()
    _match(data, "m", ["../../secret.mp4", "/etc/passwd", "sub/x.mp4", ".hidden.mp4", "notes.txt", "ok_seg005.mp4"],
           touch=True)
    assert set(ms.segment_index()) == {"ok_seg005"}


def test_malformed_segments_json_is_skipped(data):
    _match(data, "good", ["good_seg000.mp4"])
    bad = data / "bad" / "split_video_broadcast_frames"
    bad.mkdir(parents=True)
    (bad / "segments.json").write_text("{not json")
    odd = data / "odd" / "split_video_broadcast_frames"
    odd.mkdir(parents=True)
    (odd / "segments.json").write_text(json.dumps({"segments": [None, 3, {"file": 5}]}))
    assert set(ms.segment_index()) == {"good_seg000"}


def test_ambiguous_canonical_id_is_deterministic_and_reported(data):
    a = _match(data, "a_match", ["Arsenal - Aston Villa_seg001.mp4"])
    _match(data, "b_match", ["astonvilla_seg001.mp4"])
    assert ms.find_segment_video("arsenal_astonvilla_seg001") == a / "Arsenal - Aston Villa_seg001.mp4"
    assert list(ms.ambiguous()) == ["arsenal_astonvilla_seg001"]


def test_index_refreshes_when_segments_json_changes(data, monkeypatch):
    _match(data, "m", ["m_seg000.mp4"])
    assert set(ms.segment_index()) == {"m_seg000"}
    d = data / "m" / "split_video_broadcast_frames"
    (d / "m_seg001.mp4").touch()
    (d / "segments.json").write_text(json.dumps({"segments": [{"file": "m_seg000.mp4"}, {"file": "m_seg001.mp4"}]}))
    monkeypatch.setattr(ms, "_TTL_S", 0.0)
    assert set(ms.segment_index()) == {"m_seg000", "m_seg001"}


def test_missing_data_root_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("FOOTY_DATA_DIR", str(tmp_path / "nope"))
    ms.clear_cache()
    assert ms.segment_index() == {} and ms.find_segment_video("x_seg000") is None


# ---------------------------------------------------------------------------
# Editor integration
# ---------------------------------------------------------------------------


@pytest.fixture
def ed(client, clips_dir, gt_marks_dir, data, monkeypatch):
    """Editor with one dataset clip and a full-match segment dir; reads record which file they decode."""
    (clips_dir / "arsenal_mancity_20250925_seg010.mp4").touch()
    seg = _match(data, "arsenal_bournmouth_1st_half", [BOURN])
    _match(data, "arsenal_mancity", ["arsenal_mancity_20250925_seg010.mp4"])  # also a full-match segment
    frame = np.full((H, W, 3), 90, np.uint8)
    reads: list[tuple[str, int]] = []

    def read(video, idx):
        reads.append((str(video), idx))
        return frame

    monkeypatch.setattr(cal, "_read_frame", read)
    monkeypatch.setattr(cm, "_read_frame", read)
    monkeypatch.setattr(cal, "_video_meta", lambda v: {"frames": 50, "fps": 25.0, "width": W, "height": H})
    client.reads, client.seg, client.clips, client.gt = reads, seg, clips_dir, gt_marks_dir
    return client


def test_editor_opens_clip_from_match_segments(ed):
    r = ed.get("/calibration/api/clips/arsenal_bournemouth_1st_seg087/meta")
    assert r.status_code == 200 and r.json()["width"] == W
    assert ed.get("/calibration/api/frame/arsenal_bournemouth_1st_seg087/7.jpg").status_code == 200
    assert ed.reads == [(str(ed.seg / BOURN), 7)]


def test_dataset_clip_dirs_keep_priority(ed):
    """A clip that is in the dataset dir resolves there, not to the full-match segment of the same canonical id."""
    v = cal._find_video("arsenal_mancity_seg010")
    assert v == ed.clips / "arsenal_mancity_20250925_seg010.mp4"
    assert ms.find_segment_video("arsenal_mancity_seg010") != v  # the fallback would pick another file


def test_unknown_and_traversal_clips_404(ed, data):
    (data / "secret.mp4").touch()
    for clip in ("nope_seg001", "..", "%2E%2E%2Fsecret", "..%2Fsecret"):
        assert ed.get(f"/calibration/api/clips/{clip}/meta").status_code == 404, clip
        assert ed.get(f"/calibration/api/frame/{clip}/1.jpg").status_code == 404, clip
    assert ed.reads == []


def test_gt_for_segment_clip_is_keyed_by_canonical_id(ed):
    r = ed.put("/calibration/api/gt/Bournemouth vs Arsenal 1_seg087/12", json=_body())
    assert r.status_code == 200
    files = sorted(p.name for p in (ed.gt / "calibration_gt").iterdir())
    assert files == ["arsenal_bournemouth_1st_seg087.json"]
    # any alias reads and extends the same sidecar
    assert ed.put("/calibration/api/gt/arsenal_bournemouth_1st_seg087/13", json=_body()).status_code == 200
    assert ed.get("/calibration/api/gt/bournemouth_seg087/12").status_code == 200
    assert sorted(p.name for p in (ed.gt / "calibration_gt").iterdir()) == ["arsenal_bournemouth_1st_seg087.json"]
    meta = ed.get("/calibration/api/clips/arsenal_bournemouth_1st_seg087/meta").json()
    assert meta["gt_frames"] == [12, 13]


def test_dataset_clip_gt_reuses_its_sidecar_no_duplicate(ed):
    assert ed.put("/calibration/api/gt/arsenal_mancity_20250925_seg010/3", json=_body()).status_code == 200
    assert ed.put("/calibration/api/gt/arsenal_mancity_seg010/4", json=_body()).status_code == 200
    assert [p.name for p in (ed.gt / "calibration_gt").iterdir()] == ["arsenal_mancity_seg010.json"]


def test_match_page_editor_ok_now_true_for_segment_clips(ed):
    assert cm._editor_has_clip("arsenal_bournemouth_1st_seg087") is True
    assert cm._editor_has_clip("arsenal_mancity_seg010") is True
    assert cm._editor_has_clip("arsenal_bournemouth_1st_seg999") is False


def test_editor_and_match_page_decode_the_same_file_and_index(ed):
    """The editor link ``?clip=<canonical stem>&frame=<local>`` decodes ``read_frame(<match page's video>, local)``."""
    match_video = cm._video_path("arsenal_bournmouth_1st_half", BOURN)
    assert cal._find_video("arsenal_bournemouth_1st_seg087") == match_video
    assert ed.get("/calibration/api/frame/arsenal_bournemouth_1st_seg087/321.jpg").status_code == 200
    assert ed.reads[-1] == (str(match_video), 321)


def test_editor_reads_segments_by_pts_but_dataset_clips_by_seek(ed, monkeypatch):
    """Frame numbering differs: a segment is addressed by decoded index (pts), a dataset clip by the cv2 seek."""
    calls: list[str] = []
    monkeypatch.undo()  # drop the recording fake; patch the two real readers instead
    monkeypatch.setattr(cal, "read_frame_exact", lambda v, i: calls.append("exact"))
    monkeypatch.setattr(cal, "read_frame", lambda v, i: calls.append("seek"))
    cal._read_frame(ed.seg / BOURN, 5)
    cal._read_frame(ed.clips / "arsenal_mancity_20250925_seg010.mp4", 5)
    assert calls == ["exact", "seek"]


def test_match_page_reads_by_pts(monkeypatch):
    seen = []
    monkeypatch.setattr(cm, "read_frame_exact", lambda v, i: seen.append((v, i)))
    cm._read_frame(Path("x.mp4"), 9)
    assert seen == [(Path("x.mp4"), 9)]
