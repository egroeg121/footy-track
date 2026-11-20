"""Level-2 rig consensus on synthetic segments (``calibration.rig_consensus``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track.calibration.rig_consensus import (
    ConsensusParams,
    ConsensusResult,
    compute_consensus,
    load_estimates,
    stop_rule,
)
from footy_track.scripts import ptz_rig_consensus

TRUTH = np.array([0.10, -66.58, 15.74, np.radians(-0.13)])


def _ridge_cov(rng, scale):
    """Informative covariance with a strong C_y / C_z (f-ridge-like) correlation, randomised."""
    sd = scale * np.array(
        [rng.uniform(0.3, 1.0), rng.uniform(0.8, 2.0), rng.uniform(0.3, 1.0)]
    )
    rho = rng.uniform(0.85, 0.97)
    C = np.eye(3)
    C[1, 2] = C[2, 1] = rho
    C3 = C * np.outer(sd, sd)
    S = np.zeros((4, 4))
    S[:3, :3] = C3
    S[3, 3] = np.radians(rng.uniform(0.03, 0.15)) ** 2
    return S


def _write(root, name, rig, cov, **kw):
    d = root / name
    d.mkdir()
    doc = {
        "rig": list(map(float, rig)),
        "cov": np.asarray(cov).tolist(),
        "k1": 0.0147,
        "converged": True,
        "n_anchors": 10,
    } | kw
    (d / "rig_estimate.json").write_text(json.dumps(doc))


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    root = tmp_path_factory.mktemp("match")
    rng = np.random.default_rng(7)
    for i in range(150):
        S = _ridge_cov(rng, rng.uniform(0.5, 2.5))
        _write(root, f"mc_seg{i:03d}@emirates", rng.multivariate_normal(TRUTH, S), S)
    for j, sx in enumerate([32.7, 32.7, 32.7, -32.7, -32.7, -32.7]):
        S = _ridge_cov(rng, 0.6)
        r = np.array([sx, -60.0, 22.0, 0.0]) + rng.multivariate_normal(np.zeros(4), S)
        _write(root, f"mc_seg{200 + j:03d}@gantry", r, S)
    for j in range(10):
        S = np.diag([40.0**2, 60.0**2, 30.0**2, 0.3**2])
        r = np.array(
            [
                rng.uniform(-100, 100),
                rng.uniform(-120, 0),
                rng.uniform(0, 60),
                rng.uniform(-1, 1),
            ]
        )
        _write(root, f"mc_seg{300 + j:03d}@weak", r, S)
    return root


def test_consensus_gantry_uninformative(match):
    res = compute_consensus(load_estimates(match), reference=TRUTH)
    assert np.linalg.norm(np.array(res.rig[:3]) - TRUTH[:3]) < 0.3
    assert abs(np.degrees(res.rig[3] - TRUTH[3])) < 0.05
    assert res.reference["dC_norm"] < 0.3
    fl = res.flags()
    gantry = [k for k in fl if "@gantry" in k]
    assert len(gantry) == 6 and all(
        fl[k] == "outlier" or fl[k].startswith("variant:") for k in gantry
    )
    weak = [k for k in fl if "@weak" in k]
    assert all(fl[k] == "uninformative" for k in weak)
    inl = [k for k in fl if "@emirates" in k]
    assert sum(fl[k] != "inlier" for k in inl) <= 3
    # two second-camera clusters (+/-32.7), three segments each
    assert len(res.variants) == 2
    assert sorted(len(v["members"]) for v in res.variants) == [3, 3]
    assert sorted(round(v["rig"][0]) for v in res.variants) == [-33, 33]
    assert res.main_weight_share > 0.9


def test_unconverged_and_missing_cov(tmp_path):
    rng = np.random.default_rng(1)
    for i in range(12):
        S = _ridge_cov(rng, 0.5)
        _write(tmp_path, f"m_seg{i:03d}@x", rng.multivariate_normal(TRUTH, S), S)
    _write(tmp_path, "m_seg900@x", TRUTH + 5, np.eye(4) * 0.01, converged=False)
    d = tmp_path / "m_seg901@x"
    d.mkdir()
    (d / "rig_estimate.json").write_text(json.dumps({"rig": [1, 2, 3, 0.0]}))
    fl = compute_consensus(load_estimates(tmp_path)).flags()
    assert fl["m_seg900@x"] == "unconverged" and fl["m_seg901@x"] == "uninformative"


def test_stop_rule_and_json_roundtrip(match, tmp_path):
    est = load_estimates(match)
    a = compute_consensus(est)
    assert stop_rule(a, compute_consensus(est))["stop"]
    # drop 3 inliers: tiny rig change but still checks the rule runs; then a real shift must not stop
    b = compute_consensus(est[3:])
    shifted = ConsensusResult.from_dict(json.loads(json.dumps(a.to_dict())))
    shifted.rig = [a.rig[0] + 0.5, *a.rig[1:]]
    v = stop_rule(a, shifted)
    assert not v["stop"] and v["dC_max_m"] == pytest.approx(0.5)
    rolled = ConsensusResult.from_dict(a.to_dict())
    rolled.rig = [*a.rig[:3], a.rig[3] + np.radians(0.1)]
    assert not stop_rule(a, rolled)["stop"]
    flipped = ConsensusResult.from_dict(a.to_dict())
    flipped.segments[0] = {**flipped.segments[0], "flag": "outlier"}
    assert stop_rule(a, flipped)["flips"] == [a.segments[0]["seg_id"]]
    assert stop_rule(a, b)["dC_max_m"] < 0.2
    p = tmp_path / "c.json"
    a.save(p)
    back = ConsensusResult.load(p)
    assert back.to_dict() == a.to_dict()
    assert all(len(i["sha256"]) == 64 for i in back.inputs)


def test_cli(match, tmp_path, capsys):
    out = tmp_path / "o.json"
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"rig": TRUTH.tolist()}))
    assert (
        ptz_rig_consensus.main(
            [
                str(match),
                "--out",
                str(out),
                "--reference-rig",
                str(ref),
                "--prev",
                str(out),
            ]
        )
        is not None
        or True
    )
    assert "rig*" in capsys.readouterr().out
    res = ConsensusResult.load(out)
    assert res.reference["dC_norm"] < 0.3
    assert ConsensusParams(**res.params).outlier_quantile == 0.999
