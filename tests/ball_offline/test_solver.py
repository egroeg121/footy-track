"""Synthetic-trajectory tests for the offline ball solver and metrics."""

from __future__ import annotations

import numpy as np

from footy_track.ball_offline import baselines as bl
from footy_track.ball_offline.data import ClipData, GTFrame
from footy_track.ball_offline.metrics import score_clip
from footy_track.ball_offline.solver import SolverParams, fill_gaps, solve


def make_clip(
    cands: dict[int, list[tuple[float, float, float]]], n: int, gt=None
) -> ClipData:
    xs, cs, ptr = [], [], [0]
    for t in range(n):
        for x, y, c in cands.get(t, []):
            xs.append((x, y))
            cs.append(c)
        ptr.append(len(cs))
    return ClipData(
        clip_id="synthetic",
        n_frames=n,
        ptr=np.array(ptr),
        xy=np.array(xs, float).reshape(-1, 2),
        conf=np.array(cs, float),
        wh=np.full((len(cs), 2), 10.0),
        gt={t: GTFrame(p, "human") for t, p in (gt or {}).items()},
        no_ball=set(),
        cam_shift=np.zeros((n, 2)),
        cam_cut=np.zeros(n, bool),
    )


def line(t):
    return (100.0 + 8.0 * t, 500.0 + 2.0 * t)


P = SolverParams(min_conf=0.1, kappa=2.0, fill_gap=25, fill_min_p=0.5)


def test_bridges_dropout_with_interpolation():
    n = 60
    cands = {t: [(*line(t), 0.9)] for t in range(n) if not 20 <= t < 35}
    clip = make_clip(cands, n)
    pred, nodes = bl.b1_solver(clip, P)
    assert len(nodes) == n - 15
    assert set(pred) == set(range(n))  # gap of 15 frames filled
    for t in range(20, 35):
        assert np.hypot(pred[t][0] - line(t)[0], pred[t][1] - line(t)[1]) < 1e-6


def test_long_gap_not_filled():
    n = 80
    cands = {t: [(*line(t), 0.9)] for t in range(n) if not 10 <= t < 60}
    clip = make_clip(cands, n)
    pred, _ = bl.b1_solver(clip, P)
    assert not any(10 <= t < 60 for t in pred)  # 50-frame gap > fill_gap


def test_kalman_fill_matches_linear_on_constant_velocity():
    n = 40
    cands = {t: [(*line(t), 0.9)] for t in range(n) if not 15 <= t < 25}
    clip = make_clip(cands, n)
    pred, _ = bl.b1_solver(clip, P.replace(fill_mode="kalman"))
    for t in range(15, 25):
        assert np.hypot(pred[t][0] - line(t)[0], pred[t][1] - line(t)[1]) < 1.0


def test_rejects_static_false_positive_and_isolated_low_conf():
    n = 50
    cands = {}
    for t in range(n):
        cands[t] = [(*line(t), 0.8), (900.0, 200.0, 0.55)]  # static distractor
    cands[7].append((1500.0, 900.0, 0.2))  # isolated low-conf clutter
    clip = make_clip(cands, n)
    pred, _ = bl.b1_solver(clip, P.replace(min_conf=0.1))
    for t in range(n):
        assert np.hypot(pred[t][0] - line(t)[0], pred[t][1] - line(t)[1]) < 1e-6


def test_low_conf_on_trajectory_accepted_off_trajectory_rejected():
    n = 50
    cands = {t: [(*line(t), 0.9)] for t in range(n)}
    cands[25] = [(*line(25), 0.25), (700.0, 100.0, 0.45)]  # true ball only at 0.25
    clip = make_clip(cands, n)
    # lower the strict-confidence drop: pass min_conf 0.1 so the DP may use it
    pred, nodes = bl.b1_solver(clip, P.replace(min_conf=0.1))
    # frame 25 is filled by interpolation or the low-conf candidate: either way correct
    assert np.hypot(pred[25][0] - line(25)[0], pred[25][1] - line(25)[1]) < 1e-6
    assert all(abs(nd.xy[0] - 700.0) > 1 for nd in nodes)


def test_start_of_clip_without_ball():
    n = 40
    cands = {t: [(*line(t), 0.9)] for t in range(15, n)}
    clip = make_clip(cands, n)
    pred, nodes = bl.b1_solver(clip, P)
    assert min(pred) == 15 and nodes[0].t == 15


def test_empty_clip():
    clip = make_clip({}, 10)
    assert solve(clip, P) == []
    assert fill_gaps([], P) == {}


def test_velocity_term_picks_continuing_direction():
    # two equally confident candidates one frame after a fast straight run;
    # one continues the motion, the other is a nearby reversal.
    n = 30
    cands = {t: [(*line(t), 0.7)] for t in range(15)}
    cands[15] = [(line(15)[0], line(15)[1], 0.6), (line(14)[0] - 8.0, line(14)[1], 0.6)]
    for t in range(16, n):
        cands[t] = [(*line(t), 0.7)]
    clip = make_clip(cands, n)
    pred, _ = bl.b1_solver(clip, P.replace(lam_vel=1.0))
    assert np.hypot(pred[15][0] - line(15)[0], pred[15][1] - line(15)[1]) < 1e-6


def test_anchor_forces_path_and_overrides_candidates():
    n = 40
    cands = {t: [(900.0, 100.0, 0.40)] for t in range(n)}  # weak static distractor
    clip = make_clip(cands, n)
    anchors = {10: line(10), 30: line(30)}
    pred, nodes = bl.b1_solver(clip, P, anchors=anchors)
    assert pred[10] == line(10) and pred[30] == line(30)
    # interpolation between the two anchors (gap 20 <= fill_gap)
    assert np.hypot(pred[20][0] - line(20)[0], pred[20][1] - line(20)[1]) < 1e-6


def test_camera_compensation_follows_pan():
    # camera pans 30 px/frame; ball moves 4 px/frame in world coordinates.  Without
    # compensation the apparent speed (34 px/f) exceeds vmax=10 and links die.
    n = 40
    cands = {t: [(300.0 + 34.0 * t, 500.0, 0.55)] for t in range(n)}
    clip = make_clip(cands, n)
    clip.cam_shift[1:, 0] = 30.0
    q = P.replace(vmax=10.0, r0=5.0, dmax=1.5, bias=2.0, kappa=3.0, min_conf=0.1)
    # w = logit(0.74)-2 < 0, so a lone detection is rejected; only a link rescues it
    nodes_plain = solve(clip, q)
    nodes_cam = solve(clip, q.replace(camera=True))
    assert len(nodes_cam) > len(nodes_plain)
    assert all(n_.linked for n_ in nodes_cam[1:])


def test_metrics_counts_and_denominators():
    gt = {1: (100.0, 100.0), 2: (200.0, 200.0), 3: (300.0, 300.0)}
    pred = {1: (103.0, 104.0), 2: (230.0, 200.0), 5: (5.0, 5.0), 9: (1.0, 1.0)}
    c = score_clip(pred, gt, {9}, 20)
    s = c.summary()
    assert s["n_gt_ball"] == 3 and s["n_no_ball"] == 1
    assert s["tp@10"] == 1 and s["fp@10"] == 2  # frame 2 wrong + no_ball frame 9
    assert abs(s["recall@10"] - 1 / 3) < 1e-9
    assert abs(s["precision@10"] - 1 / 3) < 1e-9
    assert abs(s["coverage"] - 4 / 20) < 1e-9  # frame 5 (unscored) counts for coverage
    assert s["tp@5"] == 1 and s["tp@20"] == 1 and s["tp@10"] == 1


def test_b0_threshold():
    clip = make_clip({0: [(1.0, 1.0, 0.3), (2.0, 2.0, 0.8)], 1: [(3.0, 3.0, 0.2)]}, 3)
    assert bl.b0_per_frame(clip, 0.5) == {0: (2.0, 2.0)}
    assert set(bl.b0_per_frame(clip, 0.1)) == {0, 1}
