"""Stage 1: segments, streaming sampled decode, detection -> CSR arrays, resumable per-clip run with meta.json last."""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from footy_track.ball_mining import candidates as C  # noqa: N812


def write_video(path, n_frames, size=(64, 48)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, size)
    for i in range(n_frames):
        img = np.full((size[1], size[0], 3), (i * 7) % 255, np.uint8)
        vw.write(img)
    vw.release()
    return path


def det(label, x, y, w, h, conf):
    return SimpleNamespace(label=label, x=x, y=y, w=w, h=h, confidence=conf)


class FakeDetector:
    """One ball per frame whose x encodes the frame position in the batch, plus a player; counts calls."""

    def __init__(self):
        self.calls = []

    def predict_frames(self, frames):
        self.calls.append(len(frames))
        out = []
        for f in frames:
            m = float(f.mean()) / 255.0
            out.append(
                SimpleNamespace(
                    detections=[
                        det("in_play_ball", 0.1, 0.2, 0.01, 0.02, 0.5 + 0.4 * m),
                        det("player", 0.5, 0.5, 0.1, 0.3, 0.9),
                        det(
                            "player", 0.2, 0.2, 0.1, 0.3, 0.1
                        ),  # below PERSON_CONF: dropped
                        det(
                            "out_of_play_ball", 0.9, 0.9, 0.01, 0.02, 0.9
                        ),  # not a candidate
                    ]
                )
            )
        return out


def make_match(tmp_path, n_clips=3):
    seg = tmp_path / "m1" / C.SEGMENTS_DIR
    seg.mkdir(parents=True)
    segs = []
    for i in range(n_clips):
        write_video(seg / f"m1_seg{i:03d}.mp4", 23)
        segs.append({"index": i, "file": f"m1_seg{i:03d}.mp4", "frame_count": 23})
    segs.append(
        {"index": 9, "file": "m1_seg009.mp4", "frame_count": 5}
    )  # listed but the file is missing
    (seg / "segments.json").write_text(json.dumps({"fps": 25.0, "segments": segs}))
    return tmp_path


def test_segments_of_keeps_existing_files_and_filters(tmp_path):
    root = make_match(tmp_path)
    segs = C.segments_of(root, "m1")
    assert [s.clip for s in segs] == ["m1_seg000", "m1_seg001", "m1_seg002"]
    assert [s.clip for s in C.segments_of(root, "m1", only=["m1_seg001"])] == [
        "m1_seg001"
    ]
    assert len(C.segments_of(root, "m1", limit=2)) == 2


def test_iter_sampled_frames_indices_and_stride(tmp_path):
    v = write_video(tmp_path / "v.mp4", 23)
    idx = [i for i, _ in C.iter_sampled_frames(v, 5)]
    assert idx == [0, 5, 10, 15, 20]
    with pytest.raises(FileNotFoundError):
        list(C.iter_sampled_frames(tmp_path / "missing.mp4", 5))


def test_rows_of_splits_ball_person_and_filters_confidence():
    b, p = C.rows_of(
        [det("in_play_ball", 0.1, 0.2, 0.01, 0.02, 0.5), det("ball", 0.3, 0.3, 0.01, 0.02, 0.12), det("referee", 0.4, 0.4, 0.1, 0.3, 0.31),
         det("player", 0.5, 0.5, 0.1, 0.3, 0.29), det("out_of_play_ball", 0.9, 0.9, 0.01, 0.02, 0.9)]
    )  # fmt: skip
    assert b.shape == (2, 5) and p.shape == (1, 5)
    assert np.isclose(p[0, 4], 0.31)


def test_detect_clip_streams_batches_and_roundtrips(tmp_path):
    v = write_video(tmp_path / "v.mp4", 23)
    d = FakeDetector()
    c = C.detect_clip(v, d, stride=5, batch=2)
    assert list(c.frames) == [0, 5, 10, 15, 20]
    assert d.calls == [2, 2, 1]  # streamed in batches, never the whole clip at once
    assert c.balls(5).shape == (1, 5) and c.persons(5).shape == (1, 5)
    assert c.balls(7).shape == (0, 5) and not c.has(7)
    assert c.n_src_frames == 23 and c.stride == 5 and (c.width, c.height) == (64, 48)
    px = c.persons_px(5, 1280, 720)
    assert np.allclose(px[0], [(0.5 + 0.05) * 1280, (0.5 + 0.15) * 720, 128, 216])
    assert (
        c.nearest_sampled(7) == 5
        and c.nearest_sampled(8) == 10
        and c.nearest_sampled(-3) == 0
    )
    C.save_clip(c, tmp_path / "c.npz")
    c2 = C.load_clip(tmp_path / "c.npz")
    assert (
        np.array_equal(c2.frames, c.frames)
        and np.allclose(c2.ball, c.ball)
        and c2.fps == c.fps
    )
    assert not list(tmp_path.glob(".*tmp*"))  # atomic write leaves no temp file


def test_run_match_resumable_meta_last_and_complete_flag(tmp_path):
    root = make_match(tmp_path)
    segs = C.segments_of(root, "m1")
    w = tmp_path / "w.pt"
    w.write_bytes(b"weights")
    out = tmp_path / "out" / "candidates.x"
    d = FakeDetector()
    st = C.run_match(
        segs, out, detector=d, weights=w, batch=4, max_seconds=-1
    )  # budget already spent: stops before the first clip
    assert st["stopped_early"] and not st["complete"] and not C.stage_done(out)
    st = C.run_match(
        segs[:2], out, detector=d, weights=w, batch=4, expected=[s.clip for s in segs]
    )
    assert (
        st["done"] == ["m1_seg000", "m1_seg001"]
        and not st["complete"]
        and not C.stage_done(out)
    )
    st = C.run_match(segs, out, detector=d, weights=w, batch=4)
    assert (
        st["skipped"] == ["m1_seg000", "m1_seg001"]
        and st["done"] == ["m1_seg002"]
        and st["complete"]
    )
    assert C.stage_done(out)
    meta = json.loads((out / C.META).read_text())
    assert (
        meta["detector"]["sha8"] == C.sha8(C.sha256_file(w))
        and meta["stride"] == 5
        and set(meta["clips"]) == set(meta["expected"])
    )


def test_run_match_refuses_other_weights(tmp_path):
    root = make_match(tmp_path, 1)
    segs = C.segments_of(root, "m1")
    w = tmp_path / "w.pt"
    w.write_bytes(b"a")
    out = tmp_path / "o"
    C.run_match(segs, out, detector=FakeDetector(), weights=w)
    w.write_bytes(b"b")
    with pytest.raises(RuntimeError, match="other settings"):
        C.run_match(segs, out, detector=FakeDetector(), weights=w)


def test_failed_clip_is_reported_and_retried(tmp_path):
    root = make_match(tmp_path, 2)
    segs = C.segments_of(root, "m1")
    (tmp_path / "w.pt").write_bytes(b"w")

    class Flaky(FakeDetector):
        boom = True

        def predict_frames(self, frames):
            if self.boom and len(self.calls) == 0:
                self.calls.append(0)
                raise RuntimeError("cuda oom")
            return super().predict_frames(frames)

    d = Flaky()
    out = tmp_path / "o"
    st = C.run_match(segs, out, detector=d, weights=tmp_path / "w.pt")
    assert list(st["failed"]) == ["m1_seg000"] and not st["complete"]
    d.boom = False
    st = C.run_match(segs, out, detector=d, weights=tmp_path / "w.pt")
    assert st["complete"] and st["skipped"] == ["m1_seg001"]


# ---------------------------------------------------------------------------------------------- device (merlin iGPU / ROCm)
def test_make_detector_follows_pick_device_and_runs_fp16_on_rocm_cuda(monkeypatch):
    """ROCm torch reports as cuda: with no flags the detector must land on ``cuda`` and predict with ``half=True``."""
    import numpy as np
    import torch

    from footy_track.detectors import ultralytics as ud

    for k in ("FOOTY_DEVICE", "FOOTY_HALF"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.2", raising=False)
    calls = []

    class FakeYOLO:
        names = {0: "ball"}

        def __init__(self, uri):
            pass

        def predict(self, frames, **kw):
            calls.append(kw)
            return []

    monkeypatch.setattr(ud, "YOLO", FakeYOLO)
    det = C.make_detector("w.pt")
    assert det.device == "cuda"
    det.predict_frames([np.zeros((8, 8, 3), np.uint8)])
    assert calls[0]["device"] == "cuda" and calls[0]["half"] is True and calls[0]["imgsz"] == C.IMGSZ

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    det = C.make_detector("w.pt")
    det.predict_frames([np.zeros((8, 8, 3), np.uint8)])
    assert det.device == "cpu" and calls[1]["half"] is False
