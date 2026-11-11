"""Tests for the Claude verify pilot (synthetic data, no GPU, no network)."""

from __future__ import annotations

import json

import pytest

from footy_track.autolabel import claude_cli
from footy_track.autolabel.verify_pilot import (
    CELL_PX,
    Cand,
    cell_centre,
    crop_origin,
    expected_position,
    merge_candidates,
    normalise_answer,
    sample_frames,
    score_question,
    summarise,
)

SIZE = (1920, 1080)


def _clips():
    held = {
        "ball": {f: [(500.0, 500.0)] for f in range(10)},
        "no_ball": {20, 21},
        "cands": {f: [(500.0, 500.0)] for f in range(30)},
    }
    seen = {
        "ball": {f: [(900.0, 400.0)] for f in range(200)},
        "no_ball": set(range(300, 340)),
        # candidate near the mark only on even frames
        "cands": {
            f: ([(905.0, 400.0)] if f % 2 == 0 else [(100.0, 100.0)])
            for f in range(400)
        },
    }
    return {"h": held, "s": seen}


def test_sampling_composition_and_determinism():
    split = {"h": "held_out", "s": "seen"}
    a = sample_frames(
        _clips(), split, seed=3, seen_with_cand=10, seen_without_cand=10, seen_no_ball=5
    )
    b = sample_frames(
        _clips(), split, seed=3, seen_with_cand=10, seen_without_cand=10, seen_no_ball=5
    )
    assert a == b
    strata = {}
    for s in a:
        strata[s.stratum] = strata.get(s.stratum, 0) + 1
    assert strata == {
        "held_out": 12,
        "seen_cand": 10,
        "seen_nocand": 10,
        "seen_no_ball": 5,
    }
    assert len({(s.clip, s.frame) for s in a}) == len(a)
    c = sample_frames(
        _clips(), split, seed=4, seen_with_cand=10, seen_without_cand=10, seen_no_ball=5
    )
    assert c != a


def test_expected_position_never_needs_gt():
    pts = {10: (100.0, 200.0), 30: (900.0, 900.0)}
    assert expected_position(10, pts, SIZE) == (100.0, 200.0, "tracker")
    assert expected_position(15, pts, SIZE)[2] == "near-5"
    assert expected_position(25, pts, SIZE)[:2] == (900.0, 900.0)
    assert expected_position(500, pts, SIZE) == (960.0, 540.0, "centre")


def test_crop_origin_is_clamped():
    assert crop_origin(5, 5, SIZE) == (0, 0)
    assert crop_origin(1915, 1075, SIZE) == (1600, 760)
    assert crop_origin(960, 540, SIZE) == (800, 380)


def test_merge_dedupes_within_6px_and_keeps_original_provenance():
    orig = [Cand(100, 100, 10, 10, 0.3, "orig")]
    re = [
        Cand(103, 100, 10, 10, 0.9, "redetect"),
        Cand(200, 200, 10, 10, 0.5, "redetect"),
    ]
    m = merge_candidates(orig, re)
    assert [(c.cx, c.source) for c in m] == [(200, "redetect"), (100, "orig")]
    many = [Cand(i * 50, 0, 5, 5, i / 100, "redetect") for i in range(1, 12)]
    assert len(merge_candidates([], many)) == 6


def test_cell_centre():
    assert cell_centre("A1", 100, 200) == (100 + CELL_PX / 2, 200 + CELL_PX / 2)
    assert cell_centre("C5", 0, 0) == (2.5 * CELL_PX, 4.5 * CELL_PX)


def test_normalise_answer_rejects_off_list():
    assert (
        normalise_answer(
            {"verdict": "candidate", "candidate": 2, "confidence": 0.9}, 3
        )["candidate"]
        == 2
    )
    assert (
        normalise_answer(
            {"verdict": "candidate", "candidate": 4, "confidence": 0.9}, 3
        )["verdict"]
        == "invalid"
    )
    assert (
        normalise_answer({"verdict": "cell", "cell": "I9"}, 3)["verdict"] == "invalid"
    )
    assert (
        normalise_answer({"verdict": "cell", "cell": "C5", "confidence": 2}, 3)[
            "confidence"
        ]
        == 1.0
    )
    assert (
        normalise_answer({"verdict": "none", "confidence": 0.5}, 0)["verdict"] == "none"
    )
    assert normalise_answer("nope", 1)["verdict"] == "invalid"


def _q(cands):
    return {"crop": [800, 400], "candidates": [c.__dict__ for c in cands]}


def test_scoring_ball_and_no_ball():
    cands = [
        Cand(900, 500, 10, 10, 0.5, "redetect"),
        Cand(850, 450, 10, 10, 0.4, "orig"),
    ]
    q = _q(cands)
    gt = {"kind": "ball", "gt": [[905, 505]]}
    pick = {"verdict": "candidate", "candidate": 1, "cell": None, "confidence": 0.9}
    r = score_question(q, gt, pick, 25)
    assert (
        r["outcome"] == "correct"
        and r["redetect_only"]
        and r["picked_near"]
        and r["gt_in_crop"]
    )
    assert score_question(q, gt, pick, 10)["outcome"] == "correct"  # 7.07 px away
    assert score_question(q, gt, pick, 5)["outcome"] == "wrong_pick"
    wrong = score_question(q, gt, {**pick, "candidate": 2}, 25)
    assert wrong["outcome"] == "wrong_pick"
    cell = {"verdict": "cell", "candidate": None, "cell": "D4", "confidence": 0.5}
    # D4 centre = 800+3.5*40, 400+3.5*40 = (940, 540); GT (905,505) is 49 px away
    assert score_question(q, gt, cell, 25)["outcome"] == "wrong_pick"
    none = {"verdict": "none", "candidate": None, "cell": None, "confidence": 0.5}
    assert score_question(q, gt, none, 25)["outcome"] == "false_none"
    nb = {"kind": "no_ball", "gt": []}
    assert score_question(q, nb, none, 25)["outcome"] == "correct"
    assert score_question(q, nb, pick, 25)["outcome"] == "false_ball"


def test_summarise_rates():
    rows = [
        {
            "kind": "ball",
            "outcome": "correct",
            "cand_within": True,
            "redetect_only": True,
            "picked_near": True,
            "gt_in_crop": True,
        },
        {
            "kind": "ball",
            "outcome": "false_none",
            "cand_within": True,
            "redetect_only": False,
            "picked_near": False,
            "gt_in_crop": True,
        },
        {
            "kind": "ball",
            "outcome": "unsure",
            "cand_within": False,
            "redetect_only": False,
            "picked_near": False,
            "gt_in_crop": False,
        },
        {
            "kind": "no_ball",
            "outcome": "false_ball",
            "cand_within": False,
            "redetect_only": False,
            "picked_near": False,
            "gt_in_crop": False,
        },
    ]
    s = summarise(rows)
    assert s["agreement"] == 0.25
    assert s["agreement_cand_available"] == 0.5
    assert s["false_none_rate"] == pytest.approx(1 / 3)
    assert s["false_ball_rate"] == 0.25 and s["unsure_rate"] == 0.25
    assert (
        s["redetect_only"] == 1
        and s["redetect_only_picked"] == 1
        and s["gt_outside_crop"] == 1
    )


def test_parse_array_tolerates_fences_and_reordering():
    ids = ["q1", "q2"]
    a = [{"id": "q2", "verdict": "none"}, {"id": "q1", "verdict": "unsure"}]
    got = claude_cli.parse_array("```json\n" + json.dumps(a) + "\n```", ids)
    assert [g["id"] for g in got] == ids
    assert claude_cli.parse_array("sorry", ids) is None
    assert claude_cli.parse_array(json.dumps(a[:1]), ids) is None


def test_prompt_lists_every_image():
    items = [
        {"id": "q1", "image": "/x/q1.png", "n_candidates": 3},
        {"id": "q2", "image": "/x/q2.png", "n_candidates": 0},
    ]
    p = claude_cli.build_prompt(items)
    assert "/x/q1.png" in p and "numbered 1..3" in p and "no boxes drawn" in p


def test_render_smoke(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np  # noqa: PLC0415

    from footy_track.autolabel.render import (  # noqa: PLC0415
        contact_sheet,
        render_question,
    )

    frame = np.full((1080, 1920, 3), 90, np.uint8)
    cands = [
        Cand(900, 500, 12, 12, 0.5, "orig"),
        Cand(905, 505, 12, 12, 0.4, "redetect"),
    ]
    out = tmp_path / "q.png"
    render_question(frame, (800, 400), cands, out)
    img = cv2.imread(str(out))
    assert img.shape == (960, 960 + 480, 3)
    sheet = tmp_path / "s.jpg"
    contact_sheet([(out, "cap", (900.0, 500.0))], sheet, origin_of=[(800, 400)])
    assert sheet.exists()
