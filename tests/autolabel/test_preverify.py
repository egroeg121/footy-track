import json

import numpy as np

from footy_track.autolabel.audit_stats import (
    binom_tail_ge,
    pick_threshold,
    wilson_lower,
)
from footy_track.autolabel.preverify import (
    BallProposal,
    PreverifyConfig,
    VerifierAnswer,
    preverify_window,
    write_preverify,
)


class FakeVerifier:
    model, prompt_version = "fake", "verify-v2.2"

    def __init__(self, conf=0.99, verdict="ball", per_frame=None):
        self.conf, self.verdict, self.per = conf, verdict, per_frame or {}
        self.called = []

    def verify(self, proposals):
        self.called += [p.frame_index for p in proposals]
        default = VerifierAnswer(self.verdict, self.conf)
        return {p.frame_index: self.per.get(p.frame_index, default) for p in proposals}


def make(n=100, **over):
    s = {
        "frame_index": np.arange(n),
        "ball_gap_px": np.full(n, 5.0),
        "ball_presence_mismatch": np.zeros(n, bool),
        "det_person_count": np.full(n, 10),
        "person_flag": np.zeros(n, bool),
        "person_component": np.zeros(n),
        "score": np.zeros(n),
    }
    s.update(over)
    props = [BallProposal(i, 100 + i, 100) for i in range(n)]
    return s, props


def run(scores, props, ver=None, **cfg):
    return preverify_window(
        "w", scores, props, ver or FakeVerifier(), PreverifyConfig(**cfg)
    )


def test_ordering_deterministic_and_tiebreak():
    sc = np.zeros(20)
    sc[[3, 7, 5]] = [0.9, 0.9, 0.5]
    s, p = make(20, score=sc, ball_presence_mismatch=sc > 0)
    assert run(s, p).flagged_order == run(s, p).flagged_order == [3, 7, 5]


def test_audit_fraction_seeded_stratified():
    s, p = make(200)
    r = run(s, p, seed=1)
    assert len(r.audit) == 20 and r.audit == run(s, p, seed=1).audit
    assert r.audit != run(s, p, seed=2).audit
    assert all(r.audit[i] // 10 == i for i in range(20))  # one per stratum
    assert set(r.audit) <= {f.frame_index for f in r.frames if f.verdict == "auto_ok"}


def test_threshold_param_and_calibration(tmp_path):
    s, p = make(10)
    assert run(s, p, FakeVerifier(conf=0.9)).flagged_order == list(
        range(10)
    )  # default 0.95
    assert run(s, p, FakeVerifier(conf=0.9), conf_threshold=0.85).flagged_order == []
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"ball_conf_threshold": 0.85}))
    assert run(s, p, FakeVerifier(conf=0.9), calibration_path=cal).flagged_order == []
    r = run(s, p, FakeVerifier(conf=0.9), calibration_path=cal, conf_threshold=0.99)
    assert r.params["conf_threshold"] == 0.99 and len(r.flagged_order) == 10


def test_player_never_auto_ok_by_default_and_experimental_policy():
    s, p = make(10)
    assert all(
        f.player_verdict == "flagged" and f.player_reasons == ["player_not_trusted"]
        for f in run(s, p).frames
    )
    pf = np.zeros(10, bool)
    pf[2] = True
    s2, p2 = make(10, person_flag=pf)
    r2 = run(s2, p2, player_auto_ok_policy="stable_count")
    assert r2.frames[2].player_reasons == ["player_count_unstable"]
    assert r2.frames[0].player_verdict == "auto_ok"


def test_reason_codes_and_verifier_only_on_tracker_ok_frames():
    s, p = make(6, ball_gap_px=np.array([5, 500, 5, 5, 5, 5.0]))
    p[0] = None
    p[4] = BallProposal(4, 900, 900)  # jump
    ver = FakeVerifier(
        per_frame={5: VerifierAnswer("ball", 0.5), 2: VerifierAnswer("unsure", 0.99)}
    )
    r = run(s, p, ver)
    by = {f.frame_index: f.reasons for f in r.frames}
    assert by[0] == ["no_ball_proposal"] and by[1] == ["tracker_disagree"]
    assert by[2] == ["verifier_not_ball"] and by[3] == ["track_not_smooth"]
    assert by[4] == ["track_not_smooth"] and by[5] == ["track_not_smooth"]
    assert 1 not in ver.called and 0 not in ver.called and 4 not in ver.called


def test_low_conf_reason():
    s, p = make(3)
    r = run(s, p, FakeVerifier(per_frame={1: VerifierAnswer("ball", 0.5)}))
    assert r.frames[1].reasons == ["verifier_low_conf"]


def test_json_provenance(tmp_path):
    s, p = make(10)
    path = write_preverify(run(s, p), tmp_path / "preverify")
    d = json.loads(path.read_text())
    assert d["verifier"] == {"model": "fake", "prompt_version": "verify-v2.2"}
    assert path.name == "w.json" and d["counts"]["auto_ok"] == 10
    assert "priority" in d["frames"][0]


def test_wilson_bound():
    assert wilson_lower(0, 0) == 0.0
    lo = wilson_lower(95, 100)
    assert 0.89 < lo < 0.95
    assert wilson_lower(950, 1000) > lo


def test_pick_threshold():
    data = (
        [(0.99, True)] * 60
        + [(0.97, True)] * 60
        + [(0.5, i % 10 < 7) for i in range(60)]
    )
    assert pick_threshold(data, target=0.95, delta=0.1) == 0.97
    assert pick_threshold([(0.9, True)] * 2, target=0.95, delta=0.1) is None
    assert abs(binom_tail_ge(0, 5, 0.5) - 1.0) < 1e-9
