"""--classify-only / --cut-from split of split_broadcast_segments: same segments.json as the one-shot run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track.scripts import split_broadcast_segments as sbs

FPS = 25.0


@pytest.fixture
def patched(monkeypatch, tmp_path):
    src = tmp_path / "match.mp4"
    src.write_bytes(b"x")
    mask = np.zeros(400, dtype=bool)
    mask[10:120] = True
    mask[122:300] = True  # 2-frame gap is bridged
    mask[340:350] = True  # 10 frames < 2 s: dropped
    monkeypatch.setattr(sbs, "classify_broadcast_mask", lambda *a, **k: (mask, FPS, len(mask)))
    monkeypatch.setattr(sbs, "_resolve_device", lambda d: "cpu")
    monkeypatch.setattr(sbs, "_check_ffmpeg", lambda: "ffmpeg")
    calls = []

    def fake_cut(ffmpeg, inp, out, start_s, dur_s):
        calls.append((out.name, start_s, dur_s))
        out.write_bytes(b"seg")
        return 0

    monkeypatch.setattr(sbs, "cut_segment", fake_cut)
    return src, calls


def test_split_phases_match_one_shot(patched, tmp_path):
    src, calls = patched
    one = tmp_path / "one"
    assert sbs.main([str(src), "--outdir", str(one)]) == 0
    n_one = len(calls)
    assert n_one == 1  # one merged segment survives

    two = tmp_path / "two"
    draft = tmp_path / "segments.draft.json"
    calls.clear()
    assert sbs.main([str(src), "--outdir", str(two), "--classify-only", "--draft-path", str(draft)]) == 0
    assert calls == []  # no encoding in the classify phase
    assert not (two / "segments.json").exists()
    assert json.loads(draft.read_text())["segments"] == [[10, 299]]

    assert sbs.main(["--cut-from", str(draft), "--outdir", str(two)]) == 0
    assert len(calls) == n_one
    assert (one / "segments.json").read_text() == (two / "segments.json").read_text()


def test_cut_from_never_classifies(patched, tmp_path, monkeypatch):
    src, _ = patched
    draft = tmp_path / "d.json"
    sbs.main([str(src), "--outdir", str(tmp_path / "o"), "--classify-only", "--draft-path", str(draft)])

    def boom(*a, **k):
        raise AssertionError("classifier must not run")

    monkeypatch.setattr(sbs, "classify_broadcast_mask", boom)
    assert sbs.main(["--cut-from", str(draft), "--outdir", str(tmp_path / "o2")]) == 0


def test_bad_draft_and_flag_conflicts(patched, tmp_path):
    src, _ = patched
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert sbs.main(["--cut-from", str(bad)]) == 2
    bad.write_text(json.dumps({"version": 1}))
    assert sbs.main(["--cut-from", str(bad)]) == 2
    assert sbs.main([str(src), "--classify-only", "--cut-from", str(bad)]) == 2
    assert sbs.main([]) == 2
