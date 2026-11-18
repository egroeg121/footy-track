"""Stage 2: the motion-blob source (blob filtering, params key, end-to-end on a synthetic moving square, resumable stage)."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from footy_track.ball_mining import candidates as C  # noqa: N812
from footy_track.ball_mining import motion as M  # noqa: N812

H, W = M.HH, M.WW


def diff_with(blobs, base=0.0):
    d = np.full((H, W), base, np.float32)
    for x0, y0, x1, y1, v in blobs:
        d[y0:y1, x0:x1] = v
    return d, np.ones((H, W), bool)


def test_params_key_stable_and_sensitive():
    assert M.MParams().key() == M.MParams().key()
    assert M.MParams().key() != M.MParams(thr=20).key()
    p = M.MParams()
    assert (p.k, p.thr, p.topn, p.mask_persons) == (
        3,
        14.0,
        20,
        True,
    )  # the tuned production parameters


def test_blob_filter_keeps_ball_sized_compact_blobs_only():
    ball = (400, 300, 410, 310, 120)  # 10x10 compact: kept
    big = (600, 200, 660, 260, 120)  # 60x60: too big
    thin = (800, 400, 840, 403, 120)  # 40x3: elongated
    rows = M.blobs_from_diffs([diff_with([ball, big, thin])], M.MParams())
    assert rows.shape[1] == len(M.COLS) and len(rows) == 1
    assert (
        abs(rows[0, 0] - 404.5) < 2 and abs(rows[0, 1] - 304.5) < 2
    )  # +-1 px: the even 2x2 opening shifts the centroid
    assert rows[0, M.COLS.index("area")] == 100 and rows[0, M.COLS.index("fill")] == 1.0


def test_two_sided_min_and_single_side_fallback():
    a, b = (
        diff_with([(400, 300, 410, 310, 120)]),
        diff_with([(100, 100, 110, 110, 120)]),
    )
    assert (
        len(M.blobs_from_diffs([a, b], M.MParams())) == 0
    )  # a pixel must differ from BOTH neighbours
    assert len(M.blobs_from_diffs([a], M.MParams())) == 1


def test_person_boxes_mask_blobs_and_flag_in_person():
    d = diff_with([(400, 300, 410, 310, 120), (900, 500, 910, 510, 120)])
    persons = np.array([[405, 305, 60, 100]], np.float32)
    kept = M.blobs_from_diffs([d], M.MParams(), persons)
    assert (
        len(kept) == 1 and kept[0, 0] > 800
    )  # the blob inside the person box is masked
    both = M.blobs_from_diffs([d], M.MParams(mask_persons=False), persons)
    assert len(both) == 2 and sorted(both[:, M.COLS.index("in_person")]) == [0.0, 1.0]


def test_topn_caps_by_score():
    blobs = [(100 + 40 * i, 300, 110 + 40 * i, 310, 60 + 3 * i) for i in range(10)]
    rows = M.blobs_from_diffs([diff_with(blobs)], M.MParams(topn=3))
    assert len(rows) == 3 and np.all(np.diff(rows[:, 2]) <= 0)


def moving_square_video(path, n=30, step=12):
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur(rng.integers(40, 200, (H, W), dtype=np.uint8), (0, 0), 2)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
    pos = {}
    for i in range(n):
        g = bg.copy()
        x, y = 300 + step * i, 400
        g[y : y + 10, x : x + 10] = 255
        pos[i] = (x + 5, y + 5)
        vw.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    vw.release()
    return pos


def test_motion_clip_finds_the_moving_square(tmp_path):
    pos = moving_square_video(tmp_path / "v.mp4")
    cand = C.build_clip(
        [0, 5, 10, 15, 20],
        [np.zeros((0, 5), np.float32)] * 5,
        [np.zeros((0, 5), np.float32)] * 5,
        size=(W, H),
        fps=25.0,
        n_src_frames=30,
        stride=5,
    )
    m = M.motion_clip(tmp_path / "v.mp4", cand, M.MParams())
    assert list(m.frames) == [0, 5, 10, 15, 20]
    assert m.at(99) is None
    assert int(m.status[2]) == 2  # both sides aligned on a static textured background
    for t in (10, 15):
        rows = m.at(t)
        d = np.hypot(rows[:, 0] - pos[t][0], rows[:, 1] - pos[t][1])
        assert len(rows) and d.min() < 4, (t, rows[:, :2], pos[t])
        assert rows[np.argmin(d), M.COLS.index("score")] > 0


def test_motion_save_load_roundtrip(tmp_path):
    m = M.ClipMotion(
        np.array([0, 5], np.int32),
        np.array([0, 1, 1], np.int32),
        np.ones((1, 11), np.float32),
        np.array([2, 1], np.int8),
        {"k": 3},
    )
    M.save_motion(m, tmp_path / "m.npz")
    m2 = M.load_motion(tmp_path / "m.npz")
    assert (
        m2.at(0).shape == (1, 11)
        and m2.at(5).shape == (0, 11)
        and m2.params == {"k": 3}
    )


def test_run_match_needs_finished_candidates_and_is_resumable(tmp_path):
    pos = moving_square_video(tmp_path / "m1_seg000.mp4")  # noqa: F841
    seg = C.Segment("m1_seg000", tmp_path / "m1_seg000.mp4", 0, 30, 25.0)
    cdir = tmp_path / "candidates.abc"
    with pytest.raises(RuntimeError, match="not complete"):
        M.run_match([seg], cdir, tmp_path / "motion.abc")
    cdir.mkdir()
    cand = C.build_clip(
        [0, 5, 10],
        [np.zeros((0, 5), np.float32)] * 3,
        [np.zeros((0, 5), np.float32)] * 3,
        size=(W, H),
        fps=25.0,
        n_src_frames=30,
        stride=5,
    )
    C.save_clip(cand, cdir / "m1_seg000.npz")
    (cdir / C.META).write_text(
        json.dumps(
            {
                "complete": True,
                "expected": ["m1_seg000"],
                "stride": 5,
                "detector": {"sha8": "abc"},
            }
        )
    )
    out = tmp_path / "motion.abc"
    st = M.run_match([seg], cdir, out)
    assert st["complete"] and st["done"] == ["m1_seg000"] and C.stage_done(out)
    st2 = M.run_match([seg], cdir, out)
    assert st2["skipped"] == ["m1_seg000"] and st2["done"] == []
    # a different parameter set restarts the stage
    st3 = M.run_match([seg], cdir, out, params=M.MParams(thr=18))
    assert st3["done"] == ["m1_seg000"]
    assert json.loads((out / C.META).read_text())["params"]["thr"] == 18
