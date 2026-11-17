"""PTZ V3 anchor recovery: pose prior, trimming of non-pitch tails, motion confirmation, multi-start lines-only search."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from footy_track.calibration import ptz
from footy_track.calibration import ptz_recover as pr
from footy_track.calibration import ptz_temporal as pt
from footy_track.calibration.pitch_model import PitchModel
from footy_track.scripts import ptz_temporal_segment as seg
from tests.calibration.test_ptz_temporal import C_TRUE, _evidence, _pairs, _sweep

RIGS = Path(__file__).resolve().parents[2] / "src/footy_track/calibration/camera_rigs"
PRIOR = {"tilt_a": 96.825, "tilt_b": -10.106, "tilt_sigma_deg": 1.12, "pan_deg": [40.0, 145.0], "f_px": [1800.0, 8000.0]}


def _on_band(pan_deg, f):
    return ptz.Camera(np.deg2rad(pan_deg), np.deg2rad(PRIOR["tilt_a"] + PRIOR["tilt_b"] * np.log(f)), float(np.log(f)),
                      np.deg2rad(0.2), C_TRUE, 0.014)


# ---------------------------------------------------------------- pose prior


def test_emirates_rig_carries_the_pose_prior_and_vitality_does_not():
    em = ptz.Rig.from_json(RIGS / "emirates.json")
    assert pr.pose_prior(em)["tilt_sigma_deg"] > 0
    assert pr.pose_prior(ptz.Rig.from_json(RIGS / "vitality.json")) is None  # no band measured: search unrestricted


def test_rig_variants_add_the_two_penalty_line_camera_positions_only_for_the_emirates_rig():
    em = ptz.Rig.from_json(RIGS / "emirates.json")
    v = pr.rig_variants(em)
    assert [n for n, _ in v] == ["centre", "x+32.7", "x-32.7"]
    assert np.allclose(v[1][1].C - em.C, [32.7, 0, 0]) and v[1][1].k1 == em.k1 and v[2][1].roll == em.roll
    assert [n for n, _ in pr.rig_variants(ptz.Rig.from_json(RIGS / "vitality.json"))] == ["centre"]


def test_in_prior_accepts_the_band_and_refuses_off_band_poses():
    c = _on_band(90.0, 4000.0)
    assert pr.in_prior(PRIOR, c.pan, c.tilt, c.logf)
    assert not pr.in_prior(PRIOR, c.pan, c.tilt + np.deg2rad(6.0), c.logf)  # 5 sigma above the band
    assert not pr.in_prior(PRIOR, np.deg2rad(20.0), c.tilt, c.logf)  # pan outside the stand's range
    assert not pr.in_prior(PRIOR, c.pan, c.tilt, np.log(1200.0))  # a zoom the operator never uses
    assert pr.in_prior(None, 0.0, 3.0, 20.0)  # no prior: everything passes


def test_prior_grid_lies_on_the_band():
    g = pr.prior_grid(PRIOR)
    assert g.shape[1] == 3 and 4000 < len(g) < 8000
    resid = np.array([pr.tilt_resid_deg(PRIOR, t, lf) for _, t, lf in g])
    assert np.abs(resid).max() <= 4.0 + 1e-9
    assert np.exp(g[:, 2]).min() >= 1799 and np.exp(g[:, 2]).max() <= 8001


# ---------------------------------------------------------------- trimming


def _fake_cache(n=240, cut=None):
    hd = np.full(n, 0.05)
    if cut is not None:
        hd[cut] = 0.8
    return SimpleNamespace(ev_frames=list(range(0, n, 5)), hd=hd)


def _pm(cache, bad_from, p_bad=0.05):
    return {t: (p_bad if t >= bad_from else 0.9) for t in cache.ev_frames}


def test_trim_cuts_a_tail_at_the_histogram_jump():
    cache = _fake_cache(240, cut=233)  # logo wipe: the first bad evidence frame is 235, the true cut is at 233
    assert pr.trim_runs([(0, 239)], cache, _pm(cache, 235)) == [(0, 232)]


def test_trim_smooth_transition_cuts_at_the_gap_start():
    cache = _fake_cache(240)  # no jump: a slow zoom into a close-up
    assert pr.trim_runs([(0, 239)], cache, _pm(cache, 200)) == [(0, 195)]


def test_trim_drops_a_middle_stretch_and_keeps_both_pieces():
    cache = _fake_cache(240)
    pm = {t: (0.05 if 100 <= t < 130 else 0.9) for t in cache.ev_frames}
    assert pr.trim_runs([(0, 239)], cache, pm) == [(0, 95), (130, 239)]


def test_trim_ignores_a_single_bad_evidence_frame_and_short_pieces():
    cache = _fake_cache(240)
    pm = dict.fromkeys(cache.ev_frames, 0.9)
    pm[100] = 0.1  # a player in front of the camera
    assert pr.trim_runs([(0, 239)], cache, pm) == [(0, 239)]
    pm2 = {t: (0.05 if t >= 5 else 0.9) for t in cache.ev_frames}  # 5 good frames then rubbish: too short to keep
    assert pr.trim_runs([(0, 239)], cache, pm2) == []


def test_trim_clean_run_is_untouched():
    cache = _fake_cache(320)
    cache.ev_frames = list(range(0, 240, 5))
    assert pr.trim_runs([(0, 239), (250, 300)], cache, dict.fromkeys(cache.ev_frames, 0.9)) == [(0, 239), (250, 300)]  # a run with no evidence frame is left alone


# ---------------------------------------------------------------- confirmation


def test_confirm_keeps_agreeing_candidates_and_drops_a_lone_wrong_one():
    cams = _sweep(9)
    rig = ptz.Rig(C_TRUE, cams[0].roll, cams[0].k1, (1.0, 3.0, 1.0))
    sh = pt.Shared.from_rig(rig)
    truth = np.array([[c.pan, c.tilt, c.logf] for c in cams])
    pairs_by = {(p.i, p.j): p for p in _pairs(cams, np.random.default_rng(2))}
    cands = {0: truth[0] + [0.001, 0.0, 0.0], 4: truth[4] + [0.0, 0.001, 0.005]}
    kept, partners = pr.confirm_anchors(cands, pairs_by, sh)
    assert sorted(kept) == [0, 4] and partners[0] == [4]
    wrong = {**cands, 8: truth[8] + [np.deg2rad(6.0), 0.0, 0.0]}  # a confident wrong frame no neighbour supports
    kept, _ = pr.confirm_anchors(wrong, pairs_by, sh)
    assert sorted(kept) == [0, 4]
    kept, _ = pr.confirm_anchors(wrong, pairs_by, sh, strong={8: True})  # unless it is very strong by itself
    assert sorted(kept) == [0, 4, 8]


def test_confirm_does_not_cross_a_broken_chain():
    cams = _sweep(9)
    sh = pt.Shared.from_rig(ptz.Rig(C_TRUE, cams[0].roll, cams[0].k1, (1.0, 3.0, 1.0)))
    truth = np.array([[c.pan, c.tilt, c.logf] for c in cams])
    pairs = [p for p in _pairs(cams, np.random.default_rng(3)) if p.i != 3]  # no pair 3 -> 4
    kept, partners = pr.confirm_anchors({0: truth[0], 8: truth[8]}, {(p.i, p.j): p for p in pairs}, sh)
    assert kept == {} and partners == {0: [], 8: []}


# ---------------------------------------------------------------- search


def test_multistart_prior_search_finds_a_band_camera_from_its_line_mask():
    cam = _on_band(88.0, 4800.0)  # centre circle + halfway line + far touchline: the seg057 kind of frame
    hyps = pr.search_lines_ms(_evidence(cam), ptz.Rig(C_TRUE, cam.roll, cam.k1, (1.0, 3.0, 1.0)), PitchModel(), prior=PRIOR,
                              n_cells=8, n_refine=3, n_ms_cells=5)
    best = hyps[0]
    assert best["f1"] > 0.7 and best["prior_ok"]
    assert ptz._cam_distance(best["cam"], cam) < 1.0


def test_lines_candidate_json_roundtrips_and_refuses_off_band_answers():
    cam = _on_band(88.0, 4800.0)
    rig = ptz.Rig(C_TRUE, cam.roll, cam.k1, (1.0, 3.0, 1.0))
    rec = pr.lines_candidate(_evidence(cam), rig, PitchModel(), PRIOR, n_cells=6, n_refine=2, n_ms_cells=4)
    assert rec["ok"] is False and rec["cand"] and rec["prior_ok"] and rec["f1_fixed"] > 0.7
    assert json.loads(json.dumps(rec, default=float))["refit"]["fixed"][2] == rec["refit"]["fixed"][2]


def test_trim_splits_at_a_hard_cut_and_drops_the_sliver_after_it():
    cache = _fake_cache(240, cut=237)  # a 3-frame close-up flash at the end, after the last evidence frame (235)
    cache.hd[237] = 0.8
    assert pr.trim_runs([(0, 239)], cache, dict.fromkeys(cache.ev_frames, 0.9)) == [(0, 236)]


def test_trim_cuts_a_soft_jump_after_the_last_evidence_frame():
    cache = _fake_cache(240)
    cache.hd[238] = 0.4  # a wipe starting in the last frames (below the hard-cut level)
    assert pr.trim_runs([(0, 239)], cache, dict.fromkeys(cache.ev_frames, 0.9)) == [(0, 237)]


# ---------------------------------------------------------------- runner helpers


def test_candidate_frames_spread_over_the_runs_and_usable_candidates_filter():
    cache = SimpleNamespace(ev_frames=list(range(0, 300, 5)))
    fr = seg.candidate_frames(cache, [(0, 99), (150, 299)], 6)
    assert len(fr) == 6 and fr == sorted(fr) and all(t < 100 or t >= 150 for t in fr) and fr[0] == 0
    assert seg.candidate_frames(cache, [(0, 20)], 12) == [0, 5, 10, 15, 20]  # fewer evidence frames than asked for
    good = {"prior_ok": True, "f1_fixed": 0.7, "f1": 0.7}
    cands = {"0": {"lines": good, "finder": {**good, "f1_fixed": 0.8}}, "5": {"lines": {**good, "f1_fixed": 0.3}},
             "10": {"lines": {**good, "prior_ok": False}}, "15": {"lines": None}}
    use = seg.usable_candidates(cands, [0, 5, 10, 15, 20])
    assert list(use) == [0] and use[0]["f1_fixed"] == 0.8  # the better of lines / finder; weak, off-band, absent: dropped
