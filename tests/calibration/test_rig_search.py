"""Rig search helpers (no video): hash canonical form, abort rule, rig document, per-segment diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track.calibration import match_diagnostic as md
from footy_track.calibration import ptz
from footy_track.calibration import rig_search as rs


def test_rig_hash_is_content_not_key_order_and_matches_the_jobs_formula():
    a = {"C": [0.1, -66.5, 15.7], "k1": 0.03, "info": {"x": "é"}}
    b = {"info": {"x": "é"}, "k1": 0.03, "C": [0.1, -66.5, 15.7]}
    assert rs.rig_hash(a) == rs.rig_hash(b) and len(rs.rig_hash(a)) == 8
    assert rs.rig_hash({**a, "k1": 0.031}) != rs.rig_hash(a)
    # pinned value: the jobs repo (footy_ptz_temporal.rig_hash) must produce the same for the same document
    assert rs.rig_tag("stem", a) == f"stem-{rs.rig_hash(a)}"


def test_assess_search_aborts_on_few_units_or_wild_finalists():
    ok = [[0.0, -66.0, 15.0], [1.0, -65.0, 16.0], [0.5, -67.0, 14.0]]
    assert rs.assess_search(89, 162, ok) == []
    assert "anchor" in rs.assess_search(10, 162, ok)[0]
    assert rs.assess_search(40, 162, ok) and rs.assess_search(41, 162, ok) == []  # 25% of 162 = 40.5
    wild = [[0.0, -66.0, 15.0], [12.0, -66.0, 15.0]]
    assert any("disagree" in w for w in rs.assess_search(89, 162, wild + ok))
    assert rs.finalist_spread(wild) == pytest.approx(12.0)


def test_rig_document_roundtrips_through_ptz_rig(tmp_path):
    prior = json.loads(rs.GENERIC_PRIOR.read_text())
    pp = {"pan_deg": [40.0, 145.0], "side_cx_m": [32.7, -32.7]}
    doc = rs.rig_document([0.2, -66.1, 15.9], -0.002, 0.02, prior, source="test", pose_prior=pp)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(doc))
    rig = ptz.Rig.from_json(p)
    assert np.allclose(rig.C, [0.2, -66.1, 15.9]) and rig.k1 == pytest.approx(0.02)
    assert rig.info["pose_prior"]["side_cx_m"] == [32.7, -32.7]
    assert "pose_prior" not in rs.rig_document([0, 0, 0], 0, 0.03, prior, source="t")["info"]
    assert rs.rig_hash(doc) == rs.rig_hash(json.loads(p.read_text()))  # deterministic


def test_generic_prior_has_no_gt_derived_content():
    prior = json.loads(rs.GENERIC_PRIOR.read_text())
    assert prior["C"] == [0.0, -63.0, 14.0] and prior["k1"] == 0.03 and "pose_prior" not in prior["info"]


def _meta(d, cal, acc, **kw):
    d.mkdir(parents=True)
    np.savez(d / "calib_full.npz", frame=np.arange(2))
    (d / "meta.json").write_text(json.dumps({"frames_decoded": 100, "calibrated_frac": cal, "accepted_frac": acc,
                                             "anchors_ok": 3, "anchors_tried": 4, "cpu_s": 36.0, "code_sha": "abcdef0123456789",
                                             **kw}))


def test_diagnostics_flags(tmp_path):
    t = tmp_path
    segs = [{"index": i, "frame_count": 100} for i in range(8)]
    for i in range(5):  # five good segments
        _meta(t / f"m_seg{i:03d}@r-aaaaaaaa", 0.99, 0.95)
    _meta(t / "m_seg005@r-aaaaaaaa", 0.0, 0.0)  # uncalibrated
    _meta(t / "m_seg006@r-aaaaaaaa", 0.9, 0.8, rig_variant="x+32.7", recovery={"used": True})  # other camera
    # seg007 missing
    costs = [{"segment": i, "cost": 100.0 + i} for i in range(5)] + [{"segment": 6, "cost": 5000.0}]
    rows = {r["segment"]: r for r in md.segment_diagnostics(segs, t, "m", "r-aaaaaaaa", costs)}
    assert rows[0]["flags"] == [] and rows[0]["anchors"] == "3/4"
    assert rows[5]["flags"] == ["uncalibrated"]
    assert set(rows[6]["flags"]) == {"recovered", "other_camera", "high_cost"}
    assert rows[7]["flags"] == ["missing"]
    s = md.summarise(list(rows.values()))
    assert s["done"] == 7 and s["missing"] == 1 and s["flags"]["uncalibrated"]["segments"] == [5]
    assert s["cpu_h"] == pytest.approx(7 * 36 / 3600, abs=1e-3)
    assert "other_camera" in md.render_markdown({"match": "m", "run_id": "r", "segments_summary": s, "segments": list(rows.values())})
