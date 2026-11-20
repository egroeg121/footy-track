"""rank_candidates (Python) and rankCandidates (the page's JS, run under node) agree and behave (LAB-1479)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from footy_track.ball_eval.suggest_rank import rank_candidates

HTML = Path(__file__).resolve().parents[2] / "src/footy_track/labeller/web/index.html"
SIZE = (1920, 1080)


def box(px, py, conf, tid=0):  # centre in px at 1080p
    return [px / 1920 - 0.005, py / 1080 - 0.005, 0.01, 0.01, conf, tid]


FAR_HIGH, NEAR_MED = box(1500, 500, 0.9), box(520, 500, 0.4)
A = {"f": 100, "cx": 500 / 1920, "cy": 500 / 1080}


def order(cands, anchors, frame):
    return [c["row"] for c in rank_candidates(cands, anchors, frame, SIZE)]


def test_near_medium_beats_far_high_conf_at_small_gap():
    assert order([FAR_HIGH, NEAR_MED], [A], 105)[0] == NEAR_MED


def test_conf_wins_at_large_gap_and_with_no_anchor():
    mid = box(800, 500, 0.9)  # 300 px away
    assert order([NEAR_MED, mid], [A], 102)[0] == NEAR_MED  # sigma 20 px: the motion term dominates
    assert order([NEAR_MED, mid], [A], 160)[0] == mid  # sigma 368 px: it falls off gracefully, conf wins
    assert order([NEAR_MED, FAR_HIGH], [], 105) == [FAR_HIGH, NEAR_MED]


def test_constant_velocity_and_tracklet_bonus():
    a0, a1 = {"f": 90, "cx": 400 / 1920, "cy": 500 / 1080}, {"f": 100, "cx": 500 / 1920, "cy": 500 / 1080}
    ahead, lag = box(600, 500, 0.4), box(500, 500, 0.4)  # moving 10 px/frame: at 110 the ball is near 600
    assert order([lag, ahead], [a0, a1], 110)[0] == ahead
    r = rank_candidates([box(800, 500, 0.5, 7), box(800, 500, 0.5, 8)], [A], 101, SIZE, anchor_tid=7)
    assert r[0]["row"][5] == 7 and r[0]["tid_bonus"] == 0.5


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_rankCandidates_matches_python():
    js = HTML.read_text()
    m = re.search(r"const SUG_RANK = .*?\nfunction rankCandidates\(.*?\n}\n", js, re.S)
    assert m, "rankCandidates not found in the page"
    cases = [([FAR_HIGH, NEAR_MED], [A], 105, 0), ([NEAR_MED, box(800, 500, 0.9)], [A], 160, 0), ([NEAR_MED, FAR_HIGH], [], 105, 0),
             ([box(800, 500, 0.5, 7), box(800, 500, 0.5, 8)], [A], 101, 7),
             ([box(600, 500, 0.4), box(500, 500, 0.4)], [{"f": 90, "cx": 400 / 1920, "cy": 500 / 1080}, {"f": 100, "cx": 500 / 1920, "cy": 500 / 1080}], 110, 0)]
    prog = m.group(0) + f"\nconst cases={json.dumps(cases)};console.log(JSON.stringify(cases.map(c=>rankCandidates(c[0],c[1],c[2],[1920,1080],c[3]).map(o=>[o.row,+o.score.toFixed(6)]))));"
    out = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    for (cands, anchors, frame, tid), got in zip(cases, out, strict=True):
        want = rank_candidates(cands, anchors, frame, SIZE, tid)
        assert [g[0] for g in got] == [w["row"] for w in want]
        assert [g[1] for g in got] == pytest.approx([round(w["score"], 6) for w in want], abs=1e-5)
