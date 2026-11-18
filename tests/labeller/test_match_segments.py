"""Calibration editor fallback to the full-match broadcast segments (``labeller.match_segments``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from footy_track.feature_store.ingest_calibration import (
    CalibrationImportReport,
    parse_gt_file,
)
from footy_track.labeller import calibration as cal
from footy_track.labeller import calibration_match as cm
from footy_track.labeller import datasets, jobs
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


def test_match_page_links_only_clips_the_editor_numbers_by_decoded_index(ed):
    assert cm._editor_has_clip("arsenal_bournemouth_1st_seg087") is True
    assert cm._editor_mode("arsenal_bournemouth_1st_seg087") == "decoded"
    # opens (dataset dir) but by the cv2 seek: not linked
    assert cm._editor_has_clip("arsenal_mancity_seg010") is False
    assert cm._editor_mode("arsenal_mancity_seg010") == "cv2_seek"
    assert cm._editor_has_clip("arsenal_bournemouth_1st_seg999") is False
    assert cm._editor_mode("arsenal_bournemouth_1st_seg999") is None


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


# ---------------------------------------------------------------------------
# Frame-index convention: per clip, independent of the active dataset
# ---------------------------------------------------------------------------

SEG_CLIP = "arsenal_bournemouth_1st_seg087"


@pytest.fixture
def multi(client, tmp_path, data, monkeypatch):
    """Two registered datasets; ``calib`` (default) has no clips, ``other`` holds the Bournemouth clip; the same clip
    is also a full-match segment. Reads record which reader was used."""
    for var in ("FOOTY_CLIPS_DIR", "FOOTY_GT_MARKS_DIR", "FOOTY_BALL_CANDIDATES_DIR"):
        monkeypatch.delenv(var, raising=False)
    dirs = {n: {"clips": tmp_path / n / "clips", "marks": tmp_path / n / "marks"} for n in ("calib", "other")}
    for d in dirs.values():
        d["clips"].mkdir(parents=True)
        d["marks"].mkdir(parents=True)
    reg = tmp_path / "reg.json"
    monkeypatch.setenv("FOOTY_LABELLER_DATASETS", str(reg))
    datasets.write_registry(reg, "calib", {
        n: {"clips_dir": str(d["clips"]), "marks_dir": str(d["marks"])} for n, d in dirs.items()})
    seg = _match(data, "arsenal_bournmouth_1st_half", [BOURN])
    calls: list[tuple[str, str, int]] = []
    monkeypatch.setattr(cal, "read_frame_exact", lambda v, i: calls.append(("exact", str(v), i)) or np.zeros((H, W, 3), np.uint8))
    monkeypatch.setattr(cal, "read_frame", lambda v, i: calls.append(("seek", str(v), i)) or np.zeros((H, W, 3), np.uint8))
    monkeypatch.setattr(cal, "_video_meta", lambda v: {"frames": 50, "fps": 25.0, "width": W, "height": H})
    client.dirs, client.seg, client.calls, client.reg = dirs, seg, calls, reg
    cm._EDITOR_CACHE.clear()
    cal._WARNED_SEEK.clear()
    yield client
    datasets.invalidate()


def _gt_file(multi, ds="calib"):
    return multi.dirs[ds]["marks"] / "calibration_gt" / f"{SEG_CLIP}.json"


def test_clip_in_another_dataset_keeps_cv2_seek_whichever_dataset_is_active(multi):
    other_video = multi.dirs["other"]["clips"] / BOURN
    other_video.touch()
    # active dataset is calib, which has no such clip: still the other dataset's file, still the seek
    assert cal._find_video(SEG_CLIP) == other_video
    assert cal.frame_convention(SEG_CLIP) == "cv2_seek"
    assert multi.get(f"/calibration/api/frame/{SEG_CLIP}/5.jpg").status_code == 200
    assert multi.calls == [("seek", str(other_video), 5)]
    # and with the owning dataset active
    assert multi.get(f"/calibration/api/frame/{SEG_CLIP}/6.jpg?dataset=other").status_code == 200
    assert multi.calls[-1] == ("seek", str(other_video), 6)


def test_current_dataset_is_searched_before_the_others(multi):
    (multi.dirs["other"]["clips"] / BOURN).touch()
    mine = multi.dirs["calib"]["clips"] / BOURN
    mine.touch()
    assert cal._find_video(SEG_CLIP) == mine


def test_extra_clips_dirs_of_other_datasets_count(multi, tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / BOURN).touch()
    datasets.write_registry(multi.reg, "calib", {
        "calib": {"clips_dir": str(multi.dirs["calib"]["clips"]), "marks_dir": str(multi.dirs["calib"]["marks"])},
        "other": {"clips_dir": str(multi.dirs["other"]["clips"]), "marks_dir": str(multi.dirs["other"]["marks"]),
                  "extra_clips_dirs": [str(extra)]}})
    assert cal._find_video(SEG_CLIP) == extra / BOURN
    assert cal.frame_convention(SEG_CLIP) == "cv2_seek"


def test_clip_found_only_as_a_segment_is_read_by_decoded_index(multi):
    assert cal._find_video(SEG_CLIP) == multi.seg / BOURN
    assert cal.frame_convention(SEG_CLIP) == "decoded"
    multi.get(f"/calibration/api/frame/{SEG_CLIP}/9.jpg")
    assert multi.calls == [("exact", str(multi.seg / BOURN), 9)]


def test_new_saves_record_convention_and_video(multi):
    r = multi.put(f"/calibration/api/gt/{SEG_CLIP}/12", json=_body())
    assert r.status_code == 200
    rec = json.loads(_gt_file(multi).read_text())["frames"]["12"]
    assert rec["frame_index_convention"] == "decoded"
    assert rec["video"] == f"arsenal_bournmouth_1st_half/split_video_broadcast_frames/{BOURN}"
    # the next frame of the same clip stays decoded (the sidecar holds only decoded frames)
    assert multi.put(f"/calibration/api/gt/{SEG_CLIP}/13", json=_body()).status_code == 200
    assert json.loads(_gt_file(multi).read_text())["frames"]["13"]["frame_index_convention"] == "decoded"


def test_dataset_clip_save_is_marked_cv2_seek(multi):
    (multi.dirs["other"]["clips"] / BOURN).touch()
    assert multi.put(f"/calibration/api/gt/{SEG_CLIP}/4", json=_body()).status_code == 200
    rec = json.loads(_gt_file(multi).read_text())["frames"]["4"]
    assert rec["frame_index_convention"] == "cv2_seek"
    assert rec["video"].endswith(f"other/clips/{BOURN}") or rec["video"].endswith(BOURN)


def test_legacy_gt_sidecar_forces_the_seek_on_the_segment_file(multi, caplog):
    gt = _gt_file(multi)
    gt.parent.mkdir(parents=True)
    legacy = {"frame_index": 802, "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "correspondences": []}  # no marker
    gt.write_text(json.dumps({"clip": SEG_CLIP, "frames": {"802": legacy}}))
    cm._EDITOR_CACHE.clear()
    with caplog.at_level("WARNING"):
        assert cal.frame_convention(SEG_CLIP) == "cv2_seek"
        multi.get(f"/calibration/api/frame/{SEG_CLIP}/802.jpg")
    assert multi.calls == [("seek", str(multi.seg / BOURN), 802)]
    assert any("frame_index_convention" in r.message for r in caplog.records)
    assert cm._editor_has_clip(SEG_CLIP) is False and cm._editor_mode(SEG_CLIP) == "cv2_seek"
    # a save into that sidecar is marked cv2_seek: the conventions never mix in one file
    assert multi.put(f"/calibration/api/gt/{SEG_CLIP}/810", json=_body()).status_code == 200
    frames = json.loads(gt.read_text())["frames"]
    assert frames["810"]["frame_index_convention"] == "cv2_seek" and "frame_index_convention" not in frames["802"]
    assert cal.frame_convention(SEG_CLIP) == "cv2_seek"


def test_sidecar_of_only_decoded_frames_stays_decoded(multi):
    gt = _gt_file(multi)
    gt.parent.mkdir(parents=True)
    gt.write_text(json.dumps({"clip": SEG_CLIP, "frames": {"3": {"frame_index_convention": "decoded"}}}))
    assert cal.frame_convention(SEG_CLIP) == "decoded"


def test_gt_readers_ignore_the_new_keys(multi):
    multi.put(f"/calibration/api/gt/{SEG_CLIP}/12", json=_body())
    path = _gt_file(multi)
    report = CalibrationImportReport()
    parsed = parse_gt_file(path, SEG_CLIP, report)
    assert parsed is not None and len(parsed.calib) == 1
    assert report.warnings == [] and report.skipped == 0
    doc = json.loads(path.read_text())
    assert cal._read_gt(SEG_CLIP)["frames"]["12"] == doc["frames"]["12"]
    assert cal._gt_frames(SEG_CLIP) == [12]
    assert jobs  # imported: its calibration reader picks named fields only (_CALIB_FIELDS)


def test_unsafe_clip_names_find_nothing(multi):
    for bad in ("../x", "a/b", ".hidden", "x*", "x[0]", ""):
        assert cal._find_video(bad) is None, bad


def test_match_page_dims_the_link_for_seek_clips_and_says_why(multi, monkeypatch):
    """frame_info reports editor_ok False + editor_mode cv2_seek for a clip that lives in a dataset dir."""
    (multi.dirs["other"]["clips"] / BOURN).touch()
    cm._EDITOR_CACHE.clear()
    assert cm._editor_has_clip(SEG_CLIP) is False
    assert cm._editor_mode(SEG_CLIP) == "cv2_seek"
    html = (Path(cm.__file__).parent / "web" / "calibration_match.html").read_text()
    assert "editor numbers frames differently for this clip (cv2 seek); see VFR note" in html


def test_clip_only_in_another_dataset_and_not_a_segment_stays_isolated(multi):
    (multi.dirs["other"]["clips"] / "private_clip.mp4").touch()
    assert cal._find_video("private_clip") is None
    assert multi.get("/calibration/api/clips/private_clip/meta").status_code == 404
    assert multi.get("/calibration/api/clips/private_clip/meta?dataset=other").status_code == 200
