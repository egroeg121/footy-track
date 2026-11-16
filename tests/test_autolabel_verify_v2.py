"""Tests for the Claude verify v2 logic (synthetic data, no GPU, no network)."""

from __future__ import annotations

import json

import pytest

from footy_track.autolabel import claude_cli
from footy_track.autolabel.prompts_v2 import PROMPTS_V2, build_prompt_v2
from footy_track.autolabel.verify_pilot import Cand
from footy_track.autolabel.verify_v2 import (
    apply_decisions,
    build_review_items,
    find_disagreements,
    normalise_answer,
    patch_origin,
    question_id,
    score_candidates,
    score_frames,
    select_candidates,
    summarise,
    tracker_anchor,
)
from footy_track.scripts import claude_verify_v2 as cv


def _c(x, y, conf, source="orig"):
    return Cand(x, y, 10, 14, conf, source)


def test_select_tracker_top_and_redetect():
    cached = [
        _c(100, 100, 0.9),
        _c(300, 300, 0.8),
        _c(500, 100, 0.7),
        _c(700, 100, 0.6),
    ]
    redet = [
        _c(110, 105, 0.5, "redetect"),  # within 25 px of a cached cand: dropped
        _c(900, 500, 0.4, "redetect"),
        _c(950, 520, 0.3, "redetect"),
        _c(1100, 600, 0.2, "redetect"),
    ]
    out = select_candidates(cached, (300.0, 301.0), redet, top_others=2, max_redetect=2)
    assert [(o.source, o.cx) for o in out] == [
        ("tracker", 300),
        ("top", 100),
        ("top", 500),
        ("redetect", 900),
        ("redetect", 950),
    ]


def test_select_no_tracker_detection_and_dedupe():
    cached = [_c(100, 100, 0.9), _c(102, 101, 0.5)]
    out = select_candidates(cached, None, [], top_others=2)
    assert [o.source for o in out] == ["top"]  # the near duplicate is dropped
    # tracker point far from every cached box selects nothing
    out = select_candidates(cached, (400.0, 400.0), [], top_others=1)
    assert [o.source for o in out] == ["top"]


def test_tracker_anchor_and_patch_origin():
    pts = {10: (50.0, 60.0), 40: (70.0, 80.0)}
    assert tracker_anchor(10, pts) == (50.0, 60.0)
    assert tracker_anchor(15, pts) == (50.0, 60.0)
    assert tracker_anchor(100, pts) is None
    cand = select_candidates([_c(20, 30, 0.9)], None)[0]
    assert patch_origin(cand) == (-44, -34)  # patch may leave the frame, it is padded


def test_normalise_answer():
    ok = normalise_answer({"verdict": "ball", "confidence": 1.4, "reason": "x"})
    assert ok["verdict"] == "ball" and ok["confidence"] == 1.0
    assert normalise_answer({"verdict": "candidate"})["verdict"] == "invalid"
    assert (
        normalise_answer({"verdict": "ball", "confidence": "high"})["verdict"]
        == "invalid"
    )
    assert normalise_answer(None)["verdict"] == "invalid"


def _setup():
    def q(fk, k, x, source="top", split="seen"):
        return {
            "id": question_id(fk.split("|")[0], int(fk.split("|")[1]), k),
            "fkey": fk,
            "split": split,
            "cand": {
                "cx": x,
                "cy": 100.0,
                "w": 10,
                "h": 10,
                "conf": 0.5,
                "source": source,
            },
        }

    questions = [
        q("a|1", 0, 100, "tracker"), q("a|1", 1, 400),  # ball frame, one near one far
        q("a|2", 0, 400), q("a|2", 1, 500),  # ball frame, nothing near (unanswerable)
        q("a|3", 0, 100), q("a|3", 1, 400),  # no-ball frame
        q("a|4", 0, 100),  # ball frame, near candidate
    ]  # fmt: skip
    truth = {
        "a|1": {"kind": "ball", "gt": [[105.0, 100.0]]},
        "a|2": {"kind": "ball", "gt": [[105.0, 100.0]]},
        "a|3": {"kind": "no_ball", "gt": []},
        "a|4": {"kind": "ball", "gt": [[105.0, 100.0]]},
    }
    frames = [{"fkey": k, "split": "seen", "kind": t["kind"]} for k, t in truth.items()]
    return questions, truth, frames


def _ans(verdict, conf=0.9):
    return {"verdict": verdict, "confidence": conf, "reason": ""}


def test_frame_and_candidate_scoring():
    questions, truth, frames = _setup()
    answers = {
        questions[0]["id"]: _ans("ball"),  # a|1 near: TP
        questions[1]["id"]: _ans("not_ball"),  # a|1 far: TN -> frame correct
        questions[2]["id"]: _ans("ball"),  # a|2 far, false ball, frame unanswerable
        questions[3]["id"]: _ans("not_ball"),
        questions[4]["id"]: _ans("not_ball"),  # a|3 no-ball
        questions[5]["id"]: _ans("ball"),  # a|3 false ball
        questions[6]["id"]: _ans("unsure"),  # a|4 near but unsure
    }
    rows = score_candidates(questions, truth, answers, 25.0)
    frs = score_frames(frames, rows, questions)
    by = {f["fkey"]: f for f in frs}
    assert by["a|1"]["correct"] and by["a|1"]["answerable"]
    assert not by["a|2"]["answerable"]  # must not count against Claude
    assert not by["a|3"]["correct"] and by["a|3"]["false_ball_frame"]
    assert not by["a|4"]["correct"] and by["a|4"]["missed_near"]
    s = summarise(frames, rows, frs)
    assert s["ball_frames_answerable"] == 2
    assert s["ball_frames_no_cand_within_R"] == 1
    assert s["frame_agreement"] == 0.5  # a|1 correct, a|4 not
    assert s["near_cands"] == 2 and s["ball_recall"] == 0.5
    assert s["ball_precision"] == pytest.approx(1 / 3)  # 3 said ball, 1 near
    assert s["false_not_ball_rate"] == 0.0 and s["near_unsure_rate"] == 0.5
    assert s["false_ball_on_no_ball_cands"] == 0.5
    assert s["unanswerable_false_ball_cands"] == 1
    assert s["by_source"]["tracker"]["precision"] == 1.0


def test_partial_answers_frame_not_scored():
    questions, truth, frames = _setup()
    answers = {questions[0]["id"]: _ans("ball")}  # a|1 has two questions, one answered
    rows = score_candidates(questions, truth, answers, 25.0)
    assert score_frames(frames, rows, questions) == []


def test_radius_changes_near():
    questions, truth, _ = _setup()
    answers = {questions[0]["id"]: _ans("ball")}
    assert score_candidates(questions, truth, answers, 25.0)[0]["near"]
    assert not score_candidates(questions, truth, answers, 2.0)[0]["near"]


def test_prompts_versioned_and_built():
    assert {"verify-v2", "verify-v2.1", "verify-v2.2"} <= set(PROMPTS_V2)
    items = [{"id": "q1", "image": "/x/q1.png"}]
    p = build_prompt_v2(items, "verify-v2")
    assert "id=q1 image=/x/q1.png" in p and "not_ball" in p
    assert claude_cli.build_prompt(items, "verify-v2.1").startswith("You are helping")
    assert "candidate" in claude_cli.build_prompt(
        [{"id": "q", "image": "i", "n_candidates": 2}]
    )


def test_failure_classification_and_env(monkeypatch):
    assert (
        claude_cli.classify_failure("Claude AI usage limit reached|1700")
        == "usage_limit"
    )
    assert claude_cli.classify_failure("Invalid API key · Please run /login") == "auth"
    assert claude_cli.classify_failure("boom") == "other"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    env = claude_cli.subscription_env()
    assert "ANTHROPIC_API_KEY" not in env and "PATH" in env


def test_run_once_reports_limit(tmp_path, monkeypatch):
    fake = tmp_path / "claude"
    fake.write_text(
        "#!/bin/sh\ncat >/dev/null\n"
        'echo \'{"is_error": true, "result": "Claude usage limit reached"}\'\nexit 1\n'
    )
    fake.chmod(0o755)
    monkeypatch.setenv("CLAUDE_BIN", str(fake))
    with pytest.raises(claude_cli.ClaudeCallError) as e:
        claude_cli.run_once(
            [{"id": "q", "image": "i"}], "opus", tmp_path, prompt_version="verify-v2"
        )
    assert e.value.kind == "usage_limit"


def test_ask_resumes_and_stops_on_limit(tmp_path, monkeypatch):
    (tmp_path / "images").mkdir()
    qs = [
        {"id": f"q{i}", "image": f"q{i}.png", "fkey": "a|1", "cand": {}}
        for i in range(4)
    ]
    (tmp_path / "questions.json").write_text(json.dumps(qs))
    calls = []

    def fake_ask(items, model, cwd, version):
        calls.append([i["id"] for i in items])
        if len(calls) == 2:
            raise claude_cli.ClaudeCallError("limit", "usage_limit")
        res = claude_cli.CallResult(
            model,
            [i["id"] for i in items],
            1.0,
            1,
            False,
            0.1,
            None,
            1,
            1,
            "",
            [
                {"id": i["id"], "verdict": "ball", "confidence": 0.5, "reason": ""}
                for i in items
            ],
        )
        return [res]

    monkeypatch.setattr(claude_cli, "ask_batch", fake_ask)
    argv = [
        "ask",
        "--run",
        str(tmp_path),
        "--model",
        "opus",
        "--batch",
        "2",
        "--parallel",
        "1",
    ]
    assert cv.main(argv) == 3
    done = json.loads((tmp_path / "answers_verify-v2_opus.json").read_text())
    assert sorted(done) == ["q0", "q1"]
    status = json.loads((tmp_path / "ask_status_verify-v2_opus.json").read_text())
    assert status["kind"] == "usage_limit"

    def ask_not_ball(items, model, cwd, version):
        calls.append([i["id"] for i in items])
        ids = [i["id"] for i in items]
        arr = [
            {"id": i, "verdict": "not_ball", "confidence": 0.5, "reason": ""}
            for i in ids
        ]
        return [
            claude_cli.CallResult(model, ids, 1.0, 1, False, 0.1, None, 1, 1, "", arr)
        ]

    calls.clear()
    monkeypatch.setattr(claude_cli, "ask_batch", ask_not_ball)
    assert cv.main(argv) == 0  # resumed: only q2, q3 asked
    done = json.loads((tmp_path / "answers_verify-v2_opus.json").read_text())
    assert calls == [["q2", "q3"]]
    assert done["q0"]["verdict"] == "ball" and done["q3"]["verdict"] == "not_ball"
    assert cv.main([*argv[:-2], "--backend", "api"]) == 2


def test_api_backend_not_wired(tmp_path):
    assert cv.main(["ask", "--run", str(tmp_path), "--backend", "api"]) == 2


# ------------------------------------------------------------------ recheck (disagreements)


def _recheck_fixture():
    questions = [
        {
            "id": "c_000010_0",
            "fkey": "c|10",
            "clip": "c",
            "frame": 10,
            "split": "held_out",
            "cand": {
                "cx": 100,
                "cy": 100,
                "w": 10,
                "h": 10,
                "conf": 0.9,
                "source": "tracker",
            },
        },
        {
            "id": "c_000010_1",
            "fkey": "c|10",
            "clip": "c",
            "frame": 10,
            "split": "held_out",
            "cand": {
                "cx": 800,
                "cy": 400,
                "w": 10,
                "h": 10,
                "conf": 0.6,
                "source": "top",
            },
        },
        {
            "id": "c_000020_0",
            "fkey": "c|20",
            "clip": "c",
            "frame": 20,
            "split": "held_out",
            "cand": {
                "cx": 50,
                "cy": 50,
                "w": 10,
                "h": 10,
                "conf": 0.7,
                "source": "top",
            },
        },
    ]
    truth = {
        "c|10": {"kind": "ball", "gt": [[102, 101]]},
        "c|20": {"kind": "no_ball", "gt": []},
    }
    ans = {
        "c_000010_0": {"verdict": "not_ball", "confidence": 0.8, "reason": ""},
        "c_000010_1": {"verdict": "ball", "confidence": 0.9, "reason": ""},
        "c_000020_0": {"verdict": "ball", "confidence": 0.6, "reason": ""},
    }
    return questions, truth, ans


def test_find_disagreements_types():
    q, t, a = _recheck_fixture()
    d = find_disagreements(q, t, a)
    assert {x["qid"]: x["type"] for x in d} == {
        "c_000010_0": "false_not_ball",
        "c_000010_1": "false_ball",
        "c_000020_0": "false_ball_no_ball_frame",
    }


def test_review_items_and_decisions():
    q, t, a = _recheck_fixture()
    items = build_review_items(
        find_disagreements(q, t, a), {"c": (1000, 500)}, {"c": "c_vid"}
    )
    # frame 10: GT card first, then its two Claude cards; frame 20: one Claude card
    assert [i["role"] for i in items] == ["gt", "claude", "claude", "claude"]
    assert all(i["clip"] == "c_vid" and i["confidence"] is None for i in items)
    assert items[0]["bbox"]["x"] == pytest.approx((102 - 6) / 1000)
    ids = [i["id"] for i in items]

    # George: the GT mark is not the ball, Claude's far box is; frame 20's box is a spare
    dec = {
        ids[0]: {"verdict": "not_ball"},
        ids[2]: {"verdict": "ball"},
        ids[3]: {"verdict": "not_ball"},
    }
    new, stats = apply_decisions(t, q, items, dec)
    assert new["c|10"] == {"kind": "ball", "gt": [[800, 400]]}
    assert new["c|20"]["kind"] == "no_ball"
    assert stats["frames_changed"] == 1 and stats["gt_not_ball"] == 1
    rows = {r["id"]: r for r in score_candidates(q, new, a)}
    assert rows["c_000010_1"]["near"] and not rows["c_000010_0"]["near"]

    # GT right, Claude wrong; and a dragged correction of the mark
    new, _ = apply_decisions(t, q, items, {ids[0]: {"verdict": "ball"}})
    assert new == t
    corr = {
        "verdict": "corrected",
        "corrected_bbox": {"x": 0.3, "y": 0.2, "w": 0.01, "h": 0.02},
    }
    new, _ = apply_decisions(t, q, items, {ids[0]: corr})
    assert new["c|10"]["gt"] == [pytest.approx([305, 105])]
    # no decisions -> untouched
    assert apply_decisions(t, q, items, {})[0] == t
