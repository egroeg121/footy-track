"""Detector suggestions on the eval panel: the random blind control, the precompute, the endpoint logic and the
blind-vs-shown scorer (footy_track.ball_eval.panel_suggest / panel.blind_windows / panel_score.suggestion_report)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from footy_track.ball_eval import panel as P
from footy_track.ball_eval import panel_score as S
from footy_track.ball_eval import panel_suggest as PS
from footy_track.labeller import suggestions

from .test_panel import HELD, HELD2, build, fd  # noqa: F401  (fd is a fixture)

SHA = "deadbeef"


def test_blind_windows_are_a_seeded_quarter_and_differ_per_match(fd):  # noqa: F811
    a, b = build(fd), build(fd, HELD2)
    ba, bb = P.blind_windows(a), P.blind_windows(b)
    assert len(ba) == round(0.25 * a["n_windows"]) and ba <= {w["window_id"] for w in a["windows"]}
    assert P.blind_windows(build(fd)) == ba  # reproducible from the panel file alone
    assert {w.split(":")[1] for w in ba} != {w.split(":")[1] for w in bb}


def test_job_items_carry_the_suggest_flag_and_qa_is_always_blind(fd):  # noqa: F811
    d = build(fd)
    blind = P.blind_windows(d)
    items = P.job_items(d)
    wins = [i for i in items if i["sampling"]["qa"] is None]
    assert {i["sampling"]["window_id"] for i in wins if i["sampling"]["suggest"] == "blind"} == blind
    assert all(i["sampling"]["suggest"] == "shown" for i in wins if i["sampling"]["window_id"] not in blind)
    assert all(i["sampling"]["suggest"] == "blind" for i in items if i["sampling"]["qa"])
    assert "suggestion" not in json.dumps(items)  # the box itself never rides in the job


class FakeDet:
    def __init__(self):
        self.calls = 0

    def predict_frames(self, frames):
        self.calls += len(frames)
        out = []
        for i, _ in enumerate(frames):
            dets = [
                SimpleNamespace(label="in_play_ball", x=0.1, y=0.2, w=0.01, h=0.02, confidence=0.3),
                SimpleNamespace(label="in_play_ball", x=0.5, y=0.6, w=0.01, h=0.02, confidence=0.9),
                SimpleNamespace(label="player", x=0.0, y=0.0, w=0.1, h=0.3, confidence=0.99),
            ]
            out.append(SimpleNamespace(detections=dets if i % 3 else []))
        return out


def _suggest(fd, monkeypatch, labelled=None):  # noqa: F811
    d = build(fd)
    P.write_panel(d, root=fd)
    w = fd / "weights.pt"
    w.write_bytes(b"weights")
    monkeypatch.setattr(PS, "_read_frames", lambda video, wanted: ((f, object()) for f in wanted))
    det = FakeDet()
    r = PS.suggest_match(HELD, root=fd, weights=w, detector=det, labelled_keys=labelled, log=lambda m: None)
    return d, r, det


def test_suggest_writes_top1_per_panel_frame_next_to_the_panel(fd, monkeypatch):  # noqa: F811
    d, r, det = _suggest(fd, monkeypatch)
    path = P.suggest_path(HELD, r["detector_sha8"], root=fd)
    tab = json.loads(path.read_text())
    frames = {f"{w['clip']}:{f}" for w in d["windows"] for f in w["frames"]}
    assert set(tab) == frames and det.calls == len(frames)  # one key per frame, QA cards excluded
    got = [v for v in tab.values() if v]
    assert got and all(v["conf"] == 0.9 and v["x"] == 0.5 for v in got)  # top-1 ball, never the player
    assert any(v is None for v in tab.values())  # no candidate -> null, not a guess
    assert r["n_with_suggestion"] == len(got) and P.suggest_meta_path(path).is_file()
    assert PS.suggest_match(HELD, root=fd, weights=fd / "weights.pt", detector=det)["status"] == "exists"
    assert not path.with_name(path.name + ".partial").exists()


def test_endpoint_logic_shows_only_on_shown_cards_and_never_for_blind_or_qa(fd, monkeypatch):  # noqa: F811
    d, r, _ = _suggest(fd, monkeypatch)
    blind = P.blind_windows(d)
    tab = json.loads(P.find_suggest_file(HELD, root=fd).read_text())
    job = {"kind": "ball_place", "items": P.job_items(d)}
    seen_blind = seen_shown = 0
    for w in d["windows"]:
        f = w["frames"][1]  # a frame with a candidate in FakeDet (index 1 of each batch is non-empty)
        res = suggestions.suggestion_for(job, w["clip"], f, root=fd)
        want = tab[f"{w['clip']}:{f}"]
        if w["window_id"] in blind:
            assert res["suggestion"] is None and res["suggest"] == "blind"
            seen_blind += 1
        else:
            seen_shown += 1
            assert res["suggest"] == "shown"
            assert (res["suggestion"] is None) == (want is None)
            if want:
                assert res["suggestion"]["detector"] == r["detector_sha8"] and res["suggestion"]["x"] == want["x"]
    assert seen_blind and seen_shown
    qa = d["qa"][0]
    assert suggestions.suggestion_for(job, qa["clip"], qa["frame"], root=fd)["suggestion"] is None


def test_live_jobs_without_the_flag_derive_blind_from_the_seed(fd, monkeypatch):  # noqa: F811
    d, _, _ = _suggest(fd, monkeypatch)
    items = P.job_items(d)
    for it in items:
        it["sampling"].pop("suggest")  # a job created before the flag existed
    job = {"kind": "ball_place", "items": items}
    blind = P.blind_windows(d)
    for w in d["windows"]:
        res = suggestions.suggestion_for(job, w["clip"], w["frames"][1], root=fd)
        assert res["suggest"] == ("blind" if w["window_id"] in blind else "shown")


def test_non_panel_jobs_and_missing_suggest_file_answer_null(fd):  # noqa: F811
    d = build(fd)
    job = {"kind": "ball_place", "items": P.job_items(d)}
    w = next(w for w in d["windows"] if w["window_id"] not in P.blind_windows(d))
    assert suggestions.suggestion_for(job, w["clip"], w["frames"][0], root=fd)["suggestion"] is None
    plain = {"kind": "ball_place", "items": [{"clip": "c", "frame": 3}]}
    assert suggestions.suggestion_for(plain, "c", 3, root=fd)["suggestion"] is None


# ---- scorer ---------------------------------------------------------------------------------------
def _scored_panel(fd, monkeypatch, pre=()):  # noqa: F811
    d, r, _ = _suggest(fd, monkeypatch, labelled=lambda: set(pre))
    return d


def _human(d, shown_shift_px=0.0, accept=True):
    """Every window labelled; the ball at the centre, but in shown windows the label is shifted (suggestion bias)."""
    blind = P.blind_windows(d)
    human = {}
    for w in d["windows"]:
        for f in w["frames"]:
            shown = w["window_id"] not in blind
            c = (0.5 + (shown_shift_px / 1000 if shown else 0.0), 0.5)
            human[(w["clip"], f)] = S.Human("ball", [c], accepted=bool(shown and accept))
    return human


def test_accepted_marker_is_read_and_is_still_a_tier1_label():
    rows = {"frame_index": 3, "bbox": {"x": 0.4, "y": 0.4, "w": 0.02, "h": 0.02}, "tags": ["in_play_ball", "labeller", "accepted_suggestion", "suggest_det:deadbeef"]}
    out: dict = {}
    S._add_row(out, "c", rows)
    assert out[("c", 3)].state == "ball" and out[("c", 3)].accepted is True
    plain: dict = {}
    S._add_row(plain, "c", {**rows, "tags": ["in_play_ball", "labeller"]})
    assert plain[("c", 3)].accepted is False and plain[("c", 3)].state == "ball"


def test_blind_vs_shown_comparison_and_blind_only(fd, monkeypatch):  # noqa: F811
    d = _scored_panel(fd, monkeypatch)
    d = {**d, "width": 1000, "height": 1000}
    human = _human(d, shown_shift_px=0.0)
    preds = dict.fromkeys(human, (500.0, 500.0, 0.9))  # the model is exactly right on the unshifted labels
    rep = S.score([d], human, preds, draws=200, seed=1, root=fd)
    sg = rep["suggestions"]["pooled"]
    n_blind = len(P.blind_windows(d))
    assert sg["comparison"]["n_windows_blind"] == n_blind
    assert sg["comparison"]["n_windows_shown"] == d["n_windows"] - n_blind
    assert sg["accepted"] == sg["shown_balls"] == 5 * (d["n_windows"] - n_blind)  # accepted share of shown cards
    assert sg["share_accepted_of_shown_ball_labels"] == 1.0
    diff = sg["comparison"]["diff_blind_minus_shown"]["r10"]["recall"]
    assert diff["est"] == 0.0  # no bias: nothing to see
    # labels pulled 30 px by the suggestion: shown recall@10 collapses, blind stays 1.0 -> a clear positive gap
    biased = _human(d, shown_shift_px=30.0)
    rep2 = S.score([d], biased, preds, draws=200, seed=1, root=fd)
    gap = rep2["suggestions"]["pooled"]["comparison"]["diff_blind_minus_shown"]["r10"]["recall"]
    assert gap["est"] == 1.0 and gap["lo"] > 0.5
    # --blind-only scores the detector on blind windows only: untouched by the shift
    only = S.score([d], biased, preds, draws=200, seed=1, root=fd, blind_only=True)
    assert only["blind_only"] is True
    assert only["pooled"]["frame_weighted"]["r10"]["recall"]["est"] == 1.0
    assert rep2["pooled"]["frame_weighted"]["r10"]["recall"]["est"] < 1.0
    # accepted boxes stay in the estimate (no separate tier): the full score counts the shown windows
    assert rep2["matches"][HELD]["n_windows_scored"] == d["n_windows"]


def test_labels_placed_before_suggestions_existed_are_blind_equivalent(fd, monkeypatch):  # noqa: F811
    d0 = build(fd)
    shown = [w for w in d0["windows"] if w["window_id"] not in P.blind_windows(d0)]
    pre = {f"{w['clip']}:{f}" for w in shown[:3] for f in w["frames"]}
    d = _scored_panel(fd, monkeypatch, pre=pre)
    human = _human(d, accept=False)
    g = S.window_groups(d, human, fd)
    assert sorted(v for v in g.values()).count("blind_equiv") == 3
    assert all(g[w["window_id"]] == "blind_equiv" for w in shown[:3])
    d = {**d, "width": 1000, "height": 1000}
    rep = S.score([d], human, dict.fromkeys(human, (500.0, 500.0, 0.9)), draws=50, root=fd)
    assert rep["suggestions"]["matches"][HELD]["windows"] == {"blind": len(P.blind_windows(d)), "blind_equiv": 3, "shown": len(shown) - 3}


def test_no_suggest_file_means_every_label_is_blind_equivalent(fd):  # noqa: F811
    d = build(fd)
    human = _human(d, accept=False)
    assert set(S.window_groups(d, human, fd).values()) == {"blind", "blind_equiv"}
