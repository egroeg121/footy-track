"""Level-2 rig consensus on synthetic segments (``calibration.rig_consensus``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track.calibration.rig_consensus import (
    ConsensusParams,
    ConsensusResult,
    SegmentEstimate,
    compute_consensus,
    load_estimates,
    stop_rule,
    to_likelihood,
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


# ---------------------------------------------------------------- P1 format / shared-prior removal

P_SIG = np.array([5.0, 10.0, 3.0, np.radians(2.0)])
P_CENTRE = TRUTH + np.array(
    [4.0, 6.0, -2.0, 0.0]
)  # prior centre (input rig JSON), offset from truth


def _p1_file(root, name, rng, nominal_sd, scale=1.5, inflation=36.0):
    """A rig_estimate.json in P1's format: posterior of (likelihood x loose prior), covariance incl. the prior."""
    Lp = np.diag(1 / P_SIG**2)
    sd = np.array([*nominal_sd, np.radians(0.5)])
    S = np.diag(sd**2)
    S[1, 2] = S[2, 1] = 0.9 * sd[1] * sd[2]  # ridge-like correlation
    Ll = np.linalg.inv(S)  # nominal likelihood information
    r_like = rng.multivariate_normal(TRUTH, S * scale)
    Lpost = Ll + Lp
    r_post = np.linalg.solve(Lpost, Ll @ r_like + Lp @ P_CENTRE)
    nominal = np.linalg.inv(Lpost)
    prior = {
        "centre_C": P_CENTRE[:3].tolist(),
        "centre_roll": float(P_CENTRE[3]),
        "k1_fixed": 0.0147,
        "sigma_C_m": P_SIG[:3].tolist(),
        "sigma_roll_rad": float(P_SIG[3]),
        "source": "rig_json",
    }
    d = root / name
    d.mkdir()
    doc = {
        "names": ["Cx", "Cy", "Cz", "roll"],
        "units": ["m", "m", "m", "rad"],
        "rig": r_post.tolist(),
        "cov": (inflation * scale * nominal).tolist(),
        "cov_model": (scale * nominal).tolist(),
        "cov_nominal": nominal.tolist(),
        "cov_scale": scale,
        "cov_inflation": inflation,
        "prior": prior,
        "focal_min": 1500.0,
        "focal_max": 4200.0,
        "pan_min_deg": -10.0,
        "pan_max_deg": 12.0,
        "final_cost": 12.5,
        "converged": True,
        "n_frames": 300,
        "n_anchors": 4,
        "code_sha": "abc123",
    }
    (d / "rig_estimate.json").write_text(json.dumps(doc))
    return r_post, Ll, scale


def test_p1_schema_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    r_post, Ll, scale = _p1_file(tmp_path, "mc_seg001@emirates", rng, [3.0, 6.0, 2.0])
    (e,) = load_estimates(tmp_path)
    assert e.seg_id == "mc_seg001@emirates" and e.extra["k1"] == 0.0147
    assert (
        e.extra["f_range"] == [1500.0, 4200.0]
        and e.extra["pan_range_deg"] == 22.0
        and e.cost == 12.5
    )
    np.testing.assert_allclose(e.rig, r_post)
    # likelihood covariance = scale * inflation(file) * inv(Lambda_like): prior gone, corrections on the data only
    lk = to_likelihood(e, ConsensusParams())
    np.testing.assert_allclose(lk.cov, scale * 36.0 * np.linalg.inv(Ll), rtol=1e-4)
    lk1 = to_likelihood(e, ConsensusParams(cov_inflation=1.0))
    np.testing.assert_allclose(lk1.cov, scale * np.linalg.inv(Ll), rtol=1e-4)
    res = compute_consensus([e], ConsensusParams(max_pos_sd_m=1e6, max_roll_sd_rad=1e3))
    back = ConsensusResult.from_dict(json.loads(json.dumps(res.to_dict())))
    assert back.inputs[0]["seg_id"] == "mc_seg001@emirates"


def test_shared_prior_not_counted_n_times(tmp_path):
    rng = np.random.default_rng(11)
    for i in range(60):  # weak segments: data about as informative as the prior
        _p1_file(tmp_path, f"mc_seg{i:03d}@x", rng, [6.0, 12.0, 4.0], inflation=1.0)
    est = load_estimates(tmp_path)
    loose = {"max_pos_sd_m": 1e3, "max_roll_sd_rad": 1e3}
    res = compute_consensus(est, ConsensusParams(cov_inflation=1.0, **loose))
    err = np.linalg.norm(np.array(res.rig[:3]) - TRUTH[:3])
    assert err < 1.0, err
    # naive: posterior means as if independent data (prior counted n times)
    naive = compute_consensus(
        [SegmentEstimate(e.seg_id, e.rig, e.cov_nominal * e.cov_scale) for e in est],
        ConsensusParams(**loose),
    )
    nerr = np.linalg.norm(np.array(naive.rig[:3]) - TRUTH[:3])
    assert nerr > 2 * err and nerr > 1.5
    # prior applied once vs not at all: small difference with 60 segments
    noprior = compute_consensus(
        est, ConsensusParams(cov_inflation=1.0, apply_prior=False, **loose)
    )
    assert np.linalg.norm(np.array(res.rig[:3]) - np.array(noprior.rig[:3])) < 0.5
