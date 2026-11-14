"""Auto-labelling loop (labelling mode): pool guard, static filter, decision rule,
objects, blind cards, ingest and provenance, circuit breaker. Synthetic data only."""

from __future__ import annotations

import json

import pytest

from footy_track import eval_registry
from footy_track.autolabel import loop
from footy_track.autolabel.loop import StaticParams, Thresholds
from footy_track.labeller.constants import card_box_index
from footy_track.scripts import autolabel_loop as cli

SIZE = [1920, 1080]


# ------------------------------------------------------------------------- pool guard


def _split(tmp_path, held=("arsenal_mancity_seg031",), seen=("arsenal_example",)):
    p = tmp_path / "split.json"
    p.write_text(
        json.dumps({**dict.fromkeys(held, "held_out"), **dict.fromkeys(seen, "seen")})
    )
    return p


def _videos(tmp_path, names):
    d = tmp_path / "clips"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / f"{n}.mp4").touch()
    return d


def test_eval_clip_is_refused_held_out_seen_and_alias(tmp_path):
    ids = loop.eval_clip_ids(_split(tmp_path))
    for bad in (
        "arsenal_mancity_seg031",  # held out
        "arsenal_example",  # seen eval clip
        "arsenal_mancity_20250925_seg031",  # alias of the held-out clip
        "mancity_seg031.mp4",
    ):
        with pytest.raises(loop.EvalClipError):
            loop.assert_not_eval([bad], ids)
    loop.assert_not_eval(["arsenal_mancity_20250925_seg012"], ids)  # fine


def test_eval_ids_include_gt_sidecars_and_missing_split_fails(tmp_path):
    gt = tmp_path / "gt"
    (gt / "ball_checks").mkdir(parents=True)
    (gt / "bournemouth_1st_seg005.jsonl").touch()
    (gt / "ball_checks" / "astonvilla_seg020.jsonl").touch()
    ids = loop.eval_clip_ids(_split(tmp_path), gt_dir=gt)
    assert {"arsenal_bournemouth_1st_seg005", "arsenal_astonvilla_seg020"} <= ids
    with pytest.raises(loop.EvalClipError):
        loop.eval_clip_ids(tmp_path / "nope.json")


def test_pick_clips_hard_fails_on_explicit_eval_clip_and_random_avoids_them(tmp_path):
    ids = loop.eval_clip_ids(_split(tmp_path))
    d = _videos(
        tmp_path,
        [
            "arsenal_mancity_20250925_seg031",
            "arsenal_mancity_20250925_seg040",
            "arsenal_mancity_20250925_seg041",
            "arsenal_example",
        ],
    )
    with pytest.raises(loop.EvalClipError):
        loop.pick_clips(d, ids, names=["arsenal_mancity_seg031"])
    with pytest.raises(loop.EvalClipError):
        loop.pick_clips(d, ids, glob="*seg031*")
    for seed in range(5):
        got = loop.pick_clips(d, ids, n_random=2, seed=seed)
        assert got == ["arsenal_mancity_seg040", "arsenal_mancity_seg041"]
    with pytest.raises(ValueError, match="only 2"):
        loop.pick_clips(d, ids, n_random=3)
    assert loop.pick_clips(d, ids, n_random=1, exclude=["arsenal_mancity_seg040"]) == [
        "arsenal_mancity_seg041"
    ]


def test_cli_pool_and_check_exit_codes(tmp_path):
    sp = _split(tmp_path)
    d = _videos(tmp_path, ["arsenal_mancity_20250925_seg031", "x_seg001"])
    run = tmp_path / "run"
    rc = cli.main(
        ["pool", "--run", str(run), "--split", str(sp), "--clips-dir", str(d)]
        + ["--clips", "arsenal_mancity_seg031"]
    )
    assert rc == 4 and not (run / "clips.json").exists()
    rc = cli.main(
        ["pool", "--run", str(run), "--split", str(sp), "--clips-dir", str(d)]
        + ["--random", "1"]
    )
    assert rc == 0
    assert json.loads((run / "clips.json").read_text())["clips"] == ["x_seg001"]
    assert cli.main(["check", "--split", str(sp), "--clips", "arsenal_example"]) == 4


# -------------------------------------------------------------------------- static


def test_static_filter_flags_image_static_while_camera_moves():
    frames = range(0, 40)
    cands = {f: [(500.0, 300.0), (100.0 + 20 * f, 400.0)] for f in frames}
    moving = {f: (3.0 * f, 0.0) for f in frames}  # 117 px of pan
    flagged = loop.static_candidates(cands, moving)
    assert {(f, 0) for f in frames} <= flagged  # the logo at (500, 300)
    assert not any(i == 1 for _, i in flagged)  # a moving object is kept


def test_static_filter_keeps_resting_ball_under_still_camera_and_unknown_camera():
    cands = {f: [(500.0, 300.0)] for f in range(40)}
    assert loop.static_candidates(cands, dict.fromkeys(range(40), (0.0, 0.0))) == set()
    assert loop.static_candidates(cands, {}) == set()  # no estimate: keep


def test_static_filter_params_respected():
    cands = {f: [(500.0, 300.0)] for f in range(6)}
    cam = {f: (10.0 * f, 0.0) for f in range(6)}
    assert loop.static_candidates(cands, cam) == set()  # needs 8 frames
    p = StaticParams(min_frames=5, min_cam_px=20.0)
    assert len(loop.static_candidates(cands, cam, p)) == 6


# ---------------------------------------------------------------------------- rule


@pytest.mark.parametrize(
    ("source", "verdict", "conf", "want"),
    [
        ("tracker", "ball", 0.97, "auto"),
        ("tracker", "ball", 0.95, "auto"),
        ("tracker", "ball", 0.9, "review"),  # second tier
        ("tracker", "ball", 0.5, "review"),
        ("tracker", "not_ball", 0.99, "review"),  # tracker and Claude disagree
        ("tracker", "unsure", 0.5, "review"),
        ("top", "ball", 0.99, "review"),  # other candidates never auto
        ("redetect", "ball", 0.99, "review"),
        ("top", "not_ball", 0.9, "drop"),
        ("top", "not_ball", 0.5, "review"),  # not sure it is not a ball
        ("top", "unsure", 0.9, "review"),
    ],
)
def test_decision_rule(source, verdict, conf, want):
    assert loop.classify(source, verdict, conf)[0] == want


def test_decision_rule_thresholds_configurable_and_breaker_open():
    assert loop.classify("tracker", "ball", 0.9, Thresholds(auto_conf=0.9))[0] == "auto"
    assert loop.classify("top", "not_ball", 0.6, Thresholds(drop_conf=0.6))[0] == "drop"
    assert loop.classify("tracker", "ball", 0.99, breaker_open=True) == ("review", True)


# -------------------------------------------------------------------------- objects


def _c(qid, frame, cx, cy, source="top", track=None, **kw):
    return {
        "id": qid,
        "clip": "clipA",
        "frame": frame,
        "cx": cx,
        "cy": cy,
        "w": 12.0,
        "h": 12.0,
        "det_conf": 0.5,
        "source": source,
        "track_id": track,
        "verdict": "ball",
        "claude_conf": 0.9,
        "status": "asked",
        **kw,
    }


def test_objects_link_by_proximity_and_track_id_and_never_two_per_frame():
    cands = [
        _c("a0", 0, 100, 100),
        _c("a1", 5, 110, 100),  # same object: near
        _c("a2", 10, 118, 102),
        _c("b0", 0, 900, 500),  # a second, distant object
        _c("b1", 5, 905, 500),
        _c("t0", 20, 300, 300, "tracker", track=7),
        _c("t1", 25, 380, 300, "tracker", track=7),  # fast ball: linked by track id
        _c("z", 5, 400, 102),  # on a1's frame: cannot join a1's object
    ]
    objs = loop.link_objects(cands)
    groups = sorted(sorted(m["id"] for m in o) for o in objs)
    assert ["a0", "a1", "a2"] in groups
    assert ["b0", "b1"] in groups
    assert ["t0", "t1"] in groups
    assert ["z"] in groups
    for o in objs:
        assert len({m["frame"] for m in o}) == len(o)


def test_object_id_is_stable():
    a = [_c("q", 5, 100.2, 100.4)]
    assert loop.object_id("c", a) == loop.object_id("c", [dict(a[0], id="other")])


# --------------------------------------------------------------------- build round


def _round_items():
    items = []
    # a tracked ball, 12 frames, Claude sure: auto-accepted object
    for k in range(12):
        items.append(
            _c(f"t{k}", 5 * k, 100 + 30 * k, 200, "tracker", 1, claude_conf=0.97)
        )
    # a spare ball seen by top candidates, Claude says ball: review object
    for k in range(6):
        items.append(_c(f"s{k}", 5 * k, 1500, 800, "top", claude_conf=0.9))
    # a hoarding logo Claude is sure is not a ball: dropped
    for k in range(4):
        items.append(
            _c(f"d{k}", 5 * k, 700, 100, "top", verdict="not_ball", claude_conf=0.95)
        )
    # filtered by the static filter, never asked
    items.append(_c("f0", 0, 50, 50, "top", status="filtered", verdict=None))
    # unanswered (budget stop)
    items.append(_c("u0", 0, 60, 60, "top", verdict=None))
    return items


def _build(**kw):
    return loop.build_round(
        _round_items(),
        round_id="r1",
        sizes={"clipA": SIZE},
        stems={"clipA": "clipA_stem"},
        thr=kw.pop("thr", Thresholds(control_rate=0.0)),
        **kw,
    )


def test_build_round_counts_and_classes():
    b = _build()
    c = b["counts"]
    assert c["candidates"] == len(_round_items())
    assert c["filtered_static"] == 1
    assert c["auto_candidates"] == 12
    assert c["review_candidates"] == 6
    assert c["dropped_candidates"] == 5  # 4 not_ball + the filtered one
    assert c["answered"] == 22
    assert b["decisions"]["f0"]["reason"] == "static"
    assert "u0" not in b["decisions"]
    by = {o["class"]: o for o in b["objects"].values() if o["clip"] == "clipA"}
    assert len(by["auto"]["members"]) == 12 and len(by["review"]["members"]) == 6


def test_cards_are_blind_stable_and_cover_roles():
    b = _build()
    for card in b["cards"]:
        assert set(card) == {
            "id",
            "clip",
            "run_clip",
            "frame_index",
            "frame_size",
            "bbox",
            "label",
            "provenance",
            "confidence",
        }
        assert card["provenance"] == "list"
        assert card["clip"] == "clipA_stem"
    roles = {m["role"] for m in b["meta"].values()}
    assert {"review", "audit", "spot"} <= roles
    # one audit card (10% of one auto object, at least one), spot cards for big objects
    assert b["counts"]["cards_by_role"]["audit"] == 1
    assert b["counts"]["cards_by_role"]["spot"] == 4  # 2 for each of the 2 big objects
    # regenerating gives the same ids; reordering does not change them
    b2 = _build()
    assert [c["id"] for c in b["cards"]] == [c["id"] for c in b2["cards"]]
    assert len({c["id"] for c in b["cards"]}) == len(b["cards"])
    # distinct card ids -> distinct stable box indices
    assert len({card_box_index(c["id"]) for c in b["cards"]}) == len(b["cards"])


def test_controls_sample_dropped_objects_and_breaker_sends_all_to_queue():
    b = _build(thr=Thresholds(control_rate=1.0))
    assert b["counts"]["cards_by_role"]["control"] == 2  # the logo + the filtered one
    b = _build(breaker_open=True)
    assert b["counts"]["auto_candidates"] == 0
    assert b["counts"]["objects_by_class"]["auto"] == 0
    assert b["counts"]["would_auto_objects"] == 1
    assert "audit" not in b["counts"]["cards_by_role"]


# ----------------------------------------------------------------------- ingest


def _verdicts(b, by_role: dict[str, str]):
    out = {}
    for cid, m in b["meta"].items():
        v = by_role.get(m["role"])
        if v:
            out[cid] = {
                "verdict": v,
                "corrected_bbox": {"x": 0.1, "y": 0.2, "w": 0.01, "h": 0.01},
            }
    return out


def _ingest(b, verdicts, eval_ids=frozenset()):
    return loop.plan_ingest(
        _round_items(), b, verdicts, sizes={"clipA": SIZE}, eval_ids=set(eval_ids)
    )


def _tags(plan):
    return sorted(tuple(ln["tags"]) for ln in plan.labels.get("clipA", []))


def test_ingest_provenance_tiers():
    b = _build()
    p = _ingest(b, _verdicts(b, {"review": "ball", "audit": "ball", "spot": "ball"}))
    tags = _tags(p)
    # audited auto object confirmed: tier 2 (12 frames); review object confirmed: tier 2 (6)
    assert tags.count(("in_play_ball", "claude", "human_checked")) == 18
    assert ("in_play_ball", "claude_auto") not in tags
    assert p.negatives == []
    # no verdicts at all: auto-accepted stay machine tier 3, the review object is not written
    p = _ingest(b, {})
    assert _tags(p) == [("in_play_ball", "claude_auto")] * 12
    assert p.audit["objects_decided"] == 0 and p.audit["precision_objects"] is None


def test_ingest_not_ball_corrected_and_box_off():
    b = _build()
    p = _ingest(b, _verdicts(b, {"review": "not_ball", "audit": "not_ball"}))
    assert _tags(p) == []  # audited object wrong: its auto labels are withdrawn
    assert len(p.negatives) == 12 + 6
    assert p.audit["precision_objects"] == 0.0 and p.audit["precision_frames"] == 0.0
    p = _ingest(b, _verdicts(b, {"review": "corrected"}))
    assert ("in_play_ball", "labeller") in _tags(p)  # one dragged box, tier 1
    assert sum(t == ("in_play_ball", "labeller") for t in _tags(p)) == 1
    p = _ingest(b, _verdicts(b, {"review": "box_off"}))
    assert ("in_play_ball", "claude", "human_checked") not in _tags(p)


def test_ingest_inconsistent_spot_check_writes_nothing_for_the_object():
    b = _build()
    v = {}
    for cid, m in b["meta"].items():
        if m["role"] == "review":
            v[cid] = {"verdict": "ball"}
        if m["role"] == "spot" and m["n_members"] == 6:
            v[cid] = {"verdict": "not_ball"}
    p = _ingest(b, v)
    assert len(p.inconsistent) == 1
    assert ("in_play_ball", "claude", "human_checked") not in _tags(p)


def test_ingest_refuses_eval_clip():
    b = _build()
    with pytest.raises(loop.EvalClipError):
        _ingest(b, {}, eval_ids={"clipA"})


def test_audit_precision_counts_frames():
    b = _build()
    p = _ingest(b, _verdicts(b, {"audit": "ball"}))
    assert p.audit["ball"] == 1 and p.audit["precision_frames"] == 1.0


def test_merge_label_lines_keeps_strongest_tier_and_dedupes():
    box = {"x": 0.1, "y": 0.1, "w": 0.01, "h": 0.01}
    near = {"x": 0.1005, "y": 0.1, "w": 0.01, "h": 0.01}
    far = {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.01}
    lines = [
        loop._line(3, box, ["in_play_ball", "claude_auto"]),
        loop._line(3, near, ["in_play_ball", "claude", "human_checked"]),
        loop._line(3, far, ["in_play_ball", "claude_auto"]),
        loop._line(3, box, ["in_play_ball", "claude_auto"]),
    ]
    out = loop.merge_label_lines(lines)
    assert len(out) == 2
    assert out[0]["tags"] == ["in_play_ball", "claude", "human_checked"]
    assert loop.dedupe_points([[10, 10], [12, 11], [40, 40]]) == [[10, 10], [40, 40]]


# ---------------------------------------------------------------------- the breaker


def _rep(name, ts, decided, precision):
    return {
        "round": name,
        "created_at": ts,
        "audit": {
            "objects_decided": decided,
            "precision_objects": precision,
            "precision_frames": precision,
        },
    }


def test_circuit_breaker():
    thr = Thresholds()
    assert loop.breaker_state([], thr)[0] is False
    assert loop.breaker_state([_rep("a", 1, 10, 1.0)], thr)[0] is False
    assert loop.breaker_state([_rep("a", 1, 10, 0.96)], thr)[0] is True
    # too few audited objects: no judgement, the earlier state stands
    assert loop.breaker_state([_rep("a", 1, 10, 0.5), _rep("b", 2, 2, 1.0)], thr)[0]
    # a later round (measured through would_auto while open) closes it again
    assert not loop.breaker_state([_rep("a", 1, 10, 0.5), _rep("b", 2, 8, 1.0)], thr)[0]
    assert (
        loop.breaker_state([_rep("a", 1, 10, 0.96)], Thresholds(breaker_precision=0.9))[
            0
        ]
        is False
    )


# --------------------------------------------------------- decide/ingest end to end


def _make_run(tmp_path):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    items = _round_items()
    for it in items:
        it.pop("verdict")
        it.pop("claude_conf")
    (run / "items.json").write_text(json.dumps(items))
    (run / "clips.json").write_text(
        json.dumps({"clips": ["clipA"], "videos": {"clipA": "/x/clipA_stem.mp4"}})
    )
    (run / "sizes.json").write_text(json.dumps({"clipA": SIZE}))
    (run / "prepare.done.json").write_text(json.dumps({"n_frames": 40}))
    ans = {}
    for it in _round_items():
        if it["verdict"] is not None:
            ans[it["id"]] = {"verdict": it["verdict"], "confidence": it["claude_conf"]}
    (run / "answers.json").write_text(json.dumps(ans))
    return run


def _write_verdicts(vdir, cards, meta, role_verdicts):
    vdir.mkdir(parents=True, exist_ok=True)
    with (vdir / "clipA_stem.jsonl").open("a") as fh:
        for c in cards:
            v = role_verdicts.get(meta[c["id"]]["role"])
            if v:
                rec = {
                    "clip": c["clip"],
                    "frame_index": c["frame_index"],
                    "box_index": card_box_index(c["id"]),
                    "source": "list",
                    "verdict": v,
                }
                fh.write(json.dumps(rec) + "\n")


def test_decide_then_ingest_roundtrip_and_breaker_next_round(tmp_path):
    run = _make_run(tmp_path)
    sp = _split(tmp_path)
    assert (
        cli.main(["decide", "--run", str(run), "--answers", str(run / "answers.json")])
        == 0
    )
    cards = json.loads((run / "review_list.json").read_text())["items"]
    meta = json.loads((run / "cards_meta.json").read_text())
    # the reviewer answers in a different order than the list: keyed by id, not position
    vdir = tmp_path / "review_gt" / "ball_checks"
    _write_verdicts(
        vdir, list(reversed(cards)), meta, {"review": "ball", "audit": "not_ball"}
    )
    labels = tmp_path / "labels"
    args = ["ingest", "--run", str(run), "--split", str(sp)]
    args += ["--answers", str(run / "answers.json"), "--verdicts-dir", str(vdir)]
    args += ["--labels-dir", str(labels), "--rounds-dir", str(run.parent)]
    assert cli.main(args) == 0
    out = [json.loads(x) for x in (labels / "clipA.jsonl").read_text().splitlines()]
    assert len(out) == 6 and all("human_checked" in x["tags"] for x in out)
    rep = json.loads((run / "round_report.json").read_text())
    assert rep["audit"]["precision_objects"] == 0.0
    # idempotent: a second ingest changes nothing and makes no new backup
    assert cli.main(args) == 0
    assert (labels / "clipA.jsonl").read_text() == "".join(
        json.dumps(x) + "\n" for x in out
    )
    # propagated verdicts for the feature store
    bc = [
        json.loads(x)
        for x in (labels / "ball_checks" / "clipA.jsonl").read_text().splitlines()
    ]
    assert {r["verdict"] for r in bc} == {"ball", "not_ball"}
    # one audited object, below min_audit_objects: the breaker keeps its state ...
    assert loop.breaker_state([rep])[0] is False
    # ... and with a trip forced, the next round sends everything to the queue
    run2 = tmp_path / "runs" / "r2"
    run2.mkdir()
    for f in ("items.json", "clips.json", "sizes.json", "prepare.done.json"):
        (run2 / f).write_text((run / f).read_text())
    rep["audit"]["objects_decided"] = 9
    (run / "round_report.json").write_text(json.dumps(rep))
    rc = cli.main(
        ["decide", "--run", str(run2), "--answers", str(run / "answers.json")]
        + ["--rounds-dir", str(run.parent)]
    )
    assert rc == 0
    r2 = json.loads((run2 / "round_report.json").read_text())
    assert r2["breaker"]["open"] is True
    assert r2["counts"]["auto_candidates"] == 0


def test_ingest_cli_refuses_eval_clip(tmp_path):
    run = _make_run(tmp_path)
    cli.main(["decide", "--run", str(run), "--answers", str(run / "answers.json")])
    sp = _split(tmp_path, held=("clipA",))
    rc = cli.main(
        [
            "ingest",
            "--run",
            str(run),
            "--split",
            str(sp),
            "--answers",
            str(run / "answers.json"),
        ]
        + ["--verdicts-dir", str(tmp_path / "v"), "--labels-dir", str(tmp_path / "l")]
    )
    assert rc == 4 and not (tmp_path / "l").exists()


def test_apply_budget_caps_asked_and_prefers_tracker_share():
    items = [_c(f"t{i}", i, 1, 1, "tracker") for i in range(50)]
    items += [_c(f"o{i}", i, 1, 1, "top") for i in range(200)]
    items.append(_c("f", 0, 1, 1, status="filtered"))
    out = cli.apply_budget(items, 100)
    asked = [i for i in out if i["status"] == "asked"]
    assert len(asked) == 100
    assert sum(i["source"] == "tracker" for i in asked) == 50  # all 50: 60% of 100 > 50
    assert out[-1]["status"] == "filtered"
    assert cli.apply_budget(items, 0) == items


# ------------------------------------------------------- queue priority, missed balls


def test_gap_items_only_where_the_tracker_has_a_path_but_no_candidate_near():
    path = {
        0: (100.0, 100.0),
        5: (200.0, 100.0),
        10: (300.0, 100.0),
        15: (400.0, 100.0),
    }
    det = {0: ((100.0, 100.0), 1)}
    cands = {5: [(210.0, 100.0)], 10: [(600.0, 600.0)], 15: []}
    got = cli.gap_items("c", [0, 5, 10, 15, 20], cands, path, det)
    # 0 was detected, 5 has a candidate within 25 px, 20 has no path
    assert [g["frame"] for g in got] == [10, 15]
    assert got[0]["status"] == "gap" and got[0]["cx"] == 300.0


def _gap_item(frame, x):
    return _c(
        f"g{frame}",
        frame,
        x,
        500,
        "gap",
        status="gap",
        verdict=None,
        claude_conf=None,
        det_conf=0.0,
    )


def _items_with_gaps():
    items = _round_items()
    items += [_gap_item(f, 700 + 10 * f) for f in (100, 105, 110)]  # one gap run
    items += [_gap_item(f, 100) for f in (300, 305)]  # a second, far in time
    return items


def test_missed_ball_cards_come_first_one_per_gap_and_never_consecutive_frames():
    b = loop.build_round(
        _items_with_gaps(),
        round_id="r1",
        sizes={"clipA": SIZE},
        stems={"clipA": "s"},
        thr=Thresholds(control_rate=0.0),
    )
    assert b["counts"]["gap_frames"] == 5 and b["counts"]["gap_objects"] == 2
    prios = [b["meta"][c["id"]]["priority"] for c in b["cards"]]
    # missed balls, spare balls, tracker, spot; the blind audit cards are mixed in
    main = [
        b["meta"][c["id"]]["priority"]
        for c in b["cards"]
        if b["meta"][c["id"]]["role"] not in loop.MIXED_ROLES
    ]
    assert main == sorted(main)
    first = [c for c in b["cards"] if b["meta"][c["id"]]["role"] == "gap"]
    assert len(first) == 2 and prios[:2] == [1, 1]
    # the middle frame of the run of three, not an edge
    assert 105 in [c["frame_index"] for c in first]
    # a spare ball (non-tracker object) outranks a tracker-only one
    roles = {
        b["meta"][c["id"]]["role"]: b["meta"][c["id"]]["priority"] for c in b["cards"]
    }
    assert roles["review"] == 2 and roles["audit"] == 4 and roles["spot"] == 5
    capped = loop.build_round(
        _items_with_gaps(),
        round_id="r1",
        sizes={"clipA": SIZE},
        stems={"clipA": "s"},
        thr=Thresholds(control_rate=0.0, max_gap_cards=1),
    )
    assert capped["counts"]["cards_by_role"]["gap"] == 1


def test_card_frame_is_the_fastest_motion_frame_and_spots_are_the_ends():
    ms = [_c(f"m{k}", 5 * k, 100 + (200 if k == 3 else 5) * k, 100) for k in range(6)]
    # displacements: k=3 jumps ahead the most -> the frame after the jump is shown
    assert loop._rep(ms)["id"] in {"m3", "m4"}
    assert loop._rep(ms[:2])["id"] == "m0"
    assert loop._rep(ms, "gap")["id"] == "m3"


def test_cards_interleave_clips_inside_a_priority():
    items = []
    for clip in ("clipA", "clipB"):
        for o in range(3):
            items.append(
                _c(
                    f"{clip}{o}",
                    5 * o,
                    100 + 600 * o,
                    300,
                    "top",
                    clip=clip,
                    claude_conf=0.9,
                )
            )
    b = loop.build_round(
        items,
        round_id="r",
        sizes={"clipA": SIZE, "clipB": SIZE},
        stems={"clipA": "a", "clipB": "b"},
        thr=Thresholds(control_rate=0.0),
    )
    order = [
        c["run_clip"] for c in b["cards"] if b["meta"][c["id"]]["role"] == "review"
    ]
    assert len(order) == 6 and all(order[i] != order[i + 1] for i in range(5))


def test_ingest_missed_ball_cards_become_hard_positives_or_no_ball():
    items = _items_with_gaps()
    b = loop.build_round(
        items,
        round_id="r1",
        sizes={"clipA": SIZE},
        stems={"clipA": "s"},
        thr=Thresholds(control_rate=0.0),
    )
    gap_cards = [cid for cid, m in b["meta"].items() if m["role"] == "gap"]
    drag = {"x": 0.3, "y": 0.4, "w": 0.01, "h": 0.01}
    v = {
        gap_cards[0]: {"verdict": "corrected", "corrected_bbox": drag},
        gap_cards[1]: {"verdict": "not_ball"},
    }
    p = loop.plan_ingest(items, b, v, sizes={"clipA": SIZE}, eval_ids=set())
    lines = [ln for ln in p.labels["clipA"] if "labeller" in ln["tags"]]
    assert len(lines) == 1 and lines[0]["bbox"] == drag
    assert [n["reason"] for n in p.negatives] == ["gap_no_ball"]
    assert p.counts["gap_hard_positives_tier1"] == 1


# ======================================================================================
# gate audits: floor, decay, state file, interleaving
# ======================================================================================


def _gate_items(n_auto=20, n_claude=10, n_static=8, n_over=9):
    """One object per candidate (frames 100 apart): every gate has a known size."""
    items, k = [], 0

    def add(prefix, n, **kw):
        nonlocal k
        for i in range(n):
            items.append(
                _c(f"{prefix}{i}", 100 * k, 40 + 50 * (k % 30), 40 + 7 * k, **kw)
            )
            k += 1

    add("a", n_auto, source="tracker", track=None, claude_conf=0.97)
    for it in items:  # distinct tracker ids so nothing links
        it["track_id"] = int(it["id"][1:]) + 1
    add("c", n_claude, source="top", verdict="not_ball", claude_conf=0.95)
    add("s", n_static, source="top", status="filtered", verdict=None)
    add("o", n_over, source="top", status="over_budget", verdict=None)
    return items


def _gate_build(items=None, **kw):
    return loop.build_round(
        items or _gate_items(),
        round_id="g1",
        sizes={"clipA": SIZE},
        stems={"clipA": "clipA_stem"},
        **kw,
    )


def _cards_of(b, gate):
    return [
        c
        for c, m in b["meta"].items()
        if m.get("gate") == gate and m["role"] in ("audit", "control")
    ]


def test_gate_sample_size_rate_floor_and_cap():
    assert loop.gate_sample_size(100, 0.20, 6) == 20
    assert loop.gate_sample_size(10, 0.20, 6) == 6  # floor
    assert loop.gate_sample_size(4, 0.20, 6) == 4  # never more than the gate has
    assert loop.gate_sample_size(0, 0.20, 6) == 0
    assert loop.gate_sample_size(100, 0.05, 3) == 5
    assert loop.gate_sample_size(10, 0.05, 3) == 3


def test_every_gate_is_sampled_with_the_floor_and_capped_at_its_size():
    b = _gate_build()
    g = b["counts"]["cards_by_gate"]
    # 20 auto objects: 20% is 4, the floor is 6; 10 claude drops: floor; 8 static: floor;
    # 9 never asked: floor
    assert {k: v["objects"] for k, v in g.items()} == {
        "auto": 20,
        "claude_drop": 10,
        "static_drop": 8,
        "over_budget": 9,
    }
    assert {k: v["cards"] for k, v in g.items()} == dict.fromkeys(g, 6)
    small = _gate_build(_gate_items(3, 2, 1, 4))
    assert {k: v["cards"] for k, v in small["counts"]["cards_by_gate"].items()} == {
        "auto": 3,
        "claude_drop": 2,
        "static_drop": 1,
        "over_budget": 4,
    }
    big = _gate_build(_gate_items(60, 0, 0, 0))
    assert big["counts"]["cards_by_gate"]["auto"]["cards"] == 12  # 20% of 60
    assert big["counts"]["cards_by_role"]["audit"] == 12


def test_settled_gates_use_the_lower_rate_and_floor_and_the_plan_is_configurable():
    thr = Thresholds()
    state = {"gates": {}}
    plan = loop.gate_params(thr, state)
    assert all(
        not v["settled"] and v["rate"] == 0.20 and v["min"] == 6 for v in plan.values()
    )
    plan = {g: {"rate": 0.05, "min": 3} for g in loop.GATES}
    b = _gate_build(_gate_items(60, 10, 8, 9), gate_plan=plan)
    cards = {k: v["cards"] for k, v in b["counts"]["cards_by_gate"].items()}
    assert cards == {"auto": 3, "claude_drop": 3, "static_drop": 3, "over_budget": 3}
    b = _gate_build(
        _gate_items(60, 10, 8, 9), thr=Thresholds(audit_rate=0.5, audit_min=1)
    )
    assert b["counts"]["cards_by_gate"]["auto"]["cards"] == 30


def _hist(*rows):
    return [
        {"round": f"r{i}", "decided": d, "correct": c, "precision": c / d, "at": i}
        for i, (d, c) in enumerate(rows)
    ]


def test_gate_decays_after_three_passing_rounds_and_a_miss_resets_it():
    thr = Thresholds()  # auto passes at 0.97, the drops at 0.95
    state = lambda h: {"gates": {"auto": {"history": h}, "claude_drop": {"history": h}}}  # noqa: E731
    ok = _hist((6, 6), (6, 6))
    assert not loop.gate_params(thr, state(ok))["auto"]["settled"]  # two rounds
    ok3 = _hist((6, 6), (6, 6), (7, 7))
    p = loop.gate_params(thr, state(ok3))
    assert p["auto"]["settled"] and p["auto"]["rate"] == 0.05 and p["auto"]["min"] == 3
    assert not p["static_drop"]["settled"]  # no history of its own
    # a failing round (judged enough, below the threshold) ends the streak
    bad = _hist((6, 6), (6, 5), (6, 6), (6, 6))
    assert loop.gate_params(thr, state(bad))["auto"]["streak"] == 2
    # a round that judged too few objects is no evidence either way
    thin = _hist((6, 6), (6, 6), (2, 0), (6, 6))
    assert loop.gate_params(thr, state(thin))["auto"]["settled"]
    # re-deciding a round ignores that round's own earlier ingest
    p = loop.gate_params(thr, state(ok3), exclude_round="r2")
    assert not p["auto"]["settled"]
    # a gate with no pass level (never asked) is never settled
    allg = {"gates": {"over_budget": {"history": _hist((6, 6), (6, 6), (6, 6))}}}
    assert not loop.gate_params(thr, allg)["over_budget"]["settled"]


def test_gate_state_file_replaces_a_round_and_orders_by_time(tmp_path):
    path = tmp_path / "runs" / "gate_state.json"
    g = {"decided": 6, "correct": 6, "precision": 1.0}
    loop.update_gate_state(path, "r1", {"auto": g, "claude_drop": {"decided": 0}}, 10.0)
    loop.update_gate_state(
        path, "r2", {"auto": {**g, "correct": 5, "precision": 5 / 6}}, 20.0
    )
    loop.update_gate_state(
        path, "r1", {"auto": {**g, "decided": 8, "correct": 8}}, 10.0
    )
    st = loop.load_gate_state(path)
    assert [h["round"] for h in st["gates"]["auto"]["history"]] == ["r1", "r2"]
    assert (
        st["gates"]["auto"]["history"][0]["decided"] == 8
    )  # r1 replaced, not appended
    assert "claude_drop" not in st["gates"]  # nothing decided: not recorded
    assert loop.load_gate_state(tmp_path / "none.json") == {"version": 1, "gates": {}}


def test_audit_cards_are_blind_and_mixed_through_the_whole_queue():
    items = _gate_items() + [
        _c(f"r{i}", 5000 + 100 * i, 800, 100 + 9 * i, "top", claude_conf=0.9)
        for i in range(30)
    ]
    b = _gate_build(items)
    keys = {frozenset(c) for c in b["cards"] if c["bbox"] is not None}
    assert keys == {
        frozenset(
            {"id", "clip", "run_clip", "frame_index", "frame_size", "bbox", "label"}
            | {"provenance", "confidence"}
        )
    }
    n = len(b["cards"])
    pos = [
        i
        for i, c in enumerate(b["cards"])
        if b["meta"][c["id"]]["role"] in loop.MIXED_ROLES
    ]
    assert len(pos) == 24
    # spread over the list: some in the first quarter, some in the last quarter
    assert min(pos) < n / 4 and max(pos) > 3 * n / 4
    assert sum(p < n / 2 for p in pos) not in (0, len(pos))
    # the rest keeps its priority order
    main = [
        b["meta"][c["id"]]["priority"]
        for c in b["cards"]
        if b["meta"][c["id"]]["role"] not in loop.MIXED_ROLES
    ]
    assert main == sorted(main)
    # deterministic
    assert [c["id"] for c in _gate_build(items)["cards"]] == [
        c["id"] for c in b["cards"]
    ]


def test_gate_objects_are_not_mixed_between_static_and_claude_drops():
    items = [
        _c("c0", 0, 100, 100, "top", verdict="not_ball", claude_conf=0.95),
        _c("s0", 5, 104, 100, "top", status="filtered", verdict=None),  # same place
    ]
    b = _gate_build(items)
    assert sorted(o["gate"] for o in b["objects"].values()) == [
        "claude_drop",
        "static_drop",
    ]
    assert b["decisions"]["s0"]["reason"] == "static"


def test_over_budget_candidates_are_decided_only_in_the_new_scheme():
    items = [_c("o0", 0, 100, 100, "top", status="over_budget", verdict=None)]
    new = _gate_build(items)
    assert new["decisions"]["o0"]["class"] == "unasked"
    assert new["counts"]["over_budget"] == 1 and new["counts"]["cards"] == 1
    old = _gate_build(items, thr=Thresholds(scheme=1))
    assert "o0" not in old["decisions"] and old["counts"]["cards"] == 0


def test_scheme_one_reproduces_the_first_rounds_sampling():
    b = _gate_build(
        _gate_items(20, 10, 8, 9),
        thr=Thresholds(scheme=1, audit_rate=0.10, control_rate=0.0),
    )
    assert b["counts"]["cards_by_role"] == {"audit": 2}  # 10% of 20, no controls at 5%
    assert all("gate" not in m for m in b["meta"].values())


# ------------------------------------------------------------------ gate ingest numbers


def _verdict_map(b, by_gate, extra=None):
    out = {}
    for cid, m in b["meta"].items():
        v = by_gate.get(m.get("gate")) if m["role"] in ("audit", "control") else None
        if callable(v):
            v = v(cid)
        if v:
            out[cid] = {"verdict": v}
    return {**out, **(extra or {})}


def test_wilson_interval():
    lo, hi = loop.wilson(9, 10)
    assert 0.59 < lo < 0.60 and 0.97 < hi < 0.99
    assert loop.wilson(0, 0) is None
    lo, hi = loop.wilson(6, 6)
    assert 0.60 < lo < 0.62 and hi == pytest.approx(1.0)


def test_ingest_reports_precision_per_gate_with_counts_and_interval():
    items = _gate_items()
    b = _gate_build(items)
    auto_cards = sorted(_cards_of(b, "auto"))
    wrong = auto_cards[0]
    v = _verdict_map(
        b,
        {
            "auto": lambda cid: "not_ball" if cid == wrong else "ball",
            "claude_drop": "not_ball",
            "static_drop": lambda cid: "ball"
            if cid == sorted(_cards_of(b, "static_drop"))[0]
            else "not_ball",
            "over_budget": "unsure",
        },
    )
    p = loop.plan_ingest(items, b, v, sizes={"clipA": SIZE}, eval_ids=set())
    a = p.gates["auto"]
    assert (a["audited"], a["ball"], a["not_ball"], a["decided"]) == (6, 5, 1, 6)
    assert a["precision"] == pytest.approx(5 / 6) and a["passed"] is False
    assert a["ci95"][0] < 5 / 6 < a["ci95"][1] and a["threshold"] == 0.97
    c = p.gates["claude_drop"]
    assert c["precision"] == 1.0 and c["passed"] is True and c["ball_rate"] == 0.0
    s = p.gates["static_drop"]
    assert (s["correct"], s["decided"]) == (5, 6) and s["ball_rate"] == pytest.approx(
        1 / 6
    )
    o = p.gates["over_budget"]
    assert o["decided"] == 0 and o["unsure"] == 6 and o["precision"] is None
    assert o["threshold"] is None and o["passed"] is None
    # the breaker still reads the auto-accept audit only
    assert p.audit["ball"] == 5 and p.audit["not_ball"] == 1


# ------------------------------------------------------------- random no-candidate frames


def _random_frames():
    return [
        {"id": f"clipA_{f:06d}_r", "clip": "clipA", "frame": f, "cands": cands}
        for f, cands in (
            (1000, [[960.0, 540.0, 0.4]]),  # a cached candidate at the centre
            (1005, []),
            (1010, [[100.0, 100.0, 0.2]]),
            (1015, []),
        )
    ]


def test_random_frames_become_blind_place_cards_without_a_box():
    b = _gate_build(random_frames=_random_frames())
    rc = [c for c in b["cards"] if b["meta"][c["id"]]["role"] == "random"]
    assert len(rc) == 4 and b["counts"]["random_cards"] == 4
    for c in rc:
        assert c["bbox"] is None and c["place"] is True
        assert c["provenance"] == "list" and c["clip"] == "clipA_stem"
        assert "note" not in c and "role" not in c and "source" not in c
    assert {c["frame_index"] for c in rc} == {1000, 1005, 1010, 1015}
    assert len({card_box_index(c["id"]) for c in b["cards"]}) == len(b["cards"])
    # they sit among the other cards, not at the end
    pos = [i for i, c in enumerate(b["cards"]) if c in rc]
    assert min(pos) < len(b["cards"]) / 2


def test_random_frames_measure_the_miss_rate_and_write_tapped_balls():
    items = _gate_items() + [
        # the loop asked about a candidate on frame 1005 (the tapped one)
        _c("q1", 1005, 300, 300, "top", claude_conf=0.9),
    ]
    rf = _random_frames()
    b = _gate_build(items, random_frames=rf)
    by_frame = {
        m["frame"]: cid for cid, m in b["meta"].items() if m["role"] == "random"
    }

    def tap(cx, cy):
        return {
            "verdict": "corrected",
            "corrected_bbox": {
                "x": cx / 1920 - 0.005,
                "y": cy / 1080 - 0.005,
                "w": 0.01,
                "h": 0.01,
            },
        }

    v = {
        by_frame[1000]: tap(962, 541),  # cached candidate (never selected): not asked
        by_frame[1005]: tap(305, 300),  # a candidate the loop asked about
        by_frame[1010]: tap(1500, 900),  # nothing near: missed
        by_frame[1015]: {"verdict": "not_ball"},  # no ball visible
    }
    p = loop.plan_ingest(
        items, b, v, sizes={"clipA": SIZE}, eval_ids=set(), random_frames=rf
    )
    r = p.random
    assert (r["frames"], r["ball_visible"], r["no_ball"]) == (4, 3, 1)
    assert (r["found_selected"], r["found_not_selected"], r["missed"]) == (1, 1, 1)
    assert r["miss_rate"] == pytest.approx(1 / 3)
    assert r["unexamined_rate"] == pytest.approx(2 / 3)
    assert r["miss_ci95"][0] < 1 / 3 < r["miss_ci95"][1]
    assert r["ball_visible_rate"] == pytest.approx(3 / 4)
    # the three tapped balls are tier 1 labels (the person drew them), the empty one nothing
    tier1 = [ln for ln in p.labels["clipA"] if "labeller" in ln["tags"]]
    assert sorted(ln["frame_index"] for ln in tier1) == [1000, 1005, 1010]
    assert p.counts["random_hard_positives_tier1"] == 3
    # not answered yet: pending, never a miss
    p0 = loop.plan_ingest(
        items, b, {}, sizes={"clipA": SIZE}, eval_ids=set(), random_frames=rf
    )
    assert p0.random["pending"] == 4 and p0.random["miss_rate"] is None


def test_pick_random_frames_spreads_over_clips_and_is_deterministic():
    frames = {"a": list(range(0, 100, 5)), "b": [0, 5], "c": []}
    got = loop.pick_random_frames(frames, 6, round_id="r1", seed=1)
    assert len(got) == 6 and got == loop.pick_random_frames(
        frames, 6, round_id="r1", seed=1
    )
    assert {c for c, _ in got[:2]} == {"a", "b"}  # round-robin, not all from one clip
    assert got != loop.pick_random_frames(frames, 6, round_id="r2", seed=1)
    assert len(loop.pick_random_frames(frames, 99, round_id="r1")) == 22


# ------------------------------------------------------------------- CLI round trip


def test_cli_decide_and_ingest_write_gate_state_report_and_use_stored_plan(tmp_path):
    run = _make_run(tmp_path)
    rf = {
        "frames": [{"id": "clipA_000100_r", "clip": "clipA", "frame": 100, "cands": []}]
    }
    (run / "random_frames.json").write_text(json.dumps(rf))
    assert (
        cli.main(["decide", "--run", str(run), "--answers", str(run / "answers.json")])
        == 0
    )
    rep = json.loads((run / "round_report.json").read_text())
    assert rep["thresholds"]["scheme"] == 2 and rep["random_frames"] == 1
    assert set(rep["gates_plan"]) == set(loop.GATES)
    cards = json.loads((run / "review_list.json").read_text())["items"]
    meta = json.loads((run / "cards_meta.json").read_text())
    assert sum(m["role"] == "random" for m in meta.values()) == 1
    place = [c for c in cards if c.get("place")]
    assert len(place) == 1 and place[0]["bbox"] is None
    md = (run / "round_report.md").read_text()
    assert "Gate audits" in md and "claude_drop" in md
    vdir = tmp_path / "gt" / "ball_checks"
    tapped = {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.01}
    vdir.mkdir(parents=True)
    with (vdir / "clipA_stem.jsonl").open("w") as fh:
        for c in cards:
            m = meta[c["id"]]
            v = {"audit": "ball", "control": "not_ball", "random": "corrected"}.get(
                m["role"]
            )
            if v:
                fh.write(
                    json.dumps(
                        {
                            "clip": c["clip"],
                            "frame_index": c["frame_index"],
                            "box_index": card_box_index(c["id"]),
                            "source": "list",
                            "verdict": v,
                            "corrected_bbox": tapped if v == "corrected" else None,
                        }
                    )
                    + "\n"
                )
    sp = _split(tmp_path)
    args = [
        "ingest",
        "--run",
        str(run),
        "--split",
        str(sp),
        "--verdicts-dir",
        str(vdir),
    ]
    args += [
        "--answers",
        str(run / "answers.json"),
        "--labels-dir",
        str(tmp_path / "labels"),
    ]
    args += ["--audit-rate", "0.9"]  # ignored: the round's own thresholds decide
    assert cli.main(args) == 0
    rep = json.loads((run / "round_report.json").read_text())
    assert rep["gates"]["auto"]["precision"] == 1.0
    assert rep["gates"]["claude_drop"]["precision"] == 1.0
    assert rep["random"]["missed"] == 1 and rep["random"]["miss_rate"] == 1.0
    md = (run / "round_report.md").read_text()
    assert "Wilson 95%" in md and "Random no-candidate frames" in md
    state = loop.load_gate_state(run.parent / "gate_state.json")
    assert state["gates"]["auto"]["history"][0]["round"] == "r1"
    assert state["gates"]["auto"]["history"][0]["precision"] == 1.0
    assert cli.main(args) == 0  # idempotent: the history still has one entry for r1
    assert (
        len(
            loop.load_gate_state(run.parent / "gate_state.json")["gates"]["auto"][
                "history"
            ]
        )
        == 1
    )
    # three more passing rounds in the history: the next decide samples at the settled rate
    st = loop.load_gate_state(run.parent / "gate_state.json")
    hist = _hist((6, 6), (6, 6), (6, 6))
    for i, h in enumerate(hist):
        h["round"] = f"old{i}"
    st["gates"]["auto"]["history"] = hist
    (run.parent / "gate_state.json").write_text(json.dumps(st))
    run2 = tmp_path / "runs" / "r2"
    run2.mkdir()
    for f in ("items.json", "clips.json", "sizes.json", "prepare.done.json"):
        (run2 / f).write_text((run / f).read_text())
    assert (
        cli.main(["decide", "--run", str(run2), "--answers", str(run / "answers.json")])
        == 0
    )
    plan2 = json.loads((run2 / "round_report.json").read_text())["gates_plan"]
    assert plan2["auto"]["settled"] and plan2["auto"]["rate"] == 0.05
    assert not plan2["claude_drop"]["settled"]


def test_ingest_of_a_round_decided_before_the_gate_audits_still_works(tmp_path):
    """A report without ``scheme`` is a round of the first scheme: ingest rebuilds the
    cards with that sampling (10% audit, 5% controls), so its stored ids still match."""
    run = _make_run(tmp_path)
    thr = Thresholds(scheme=1)
    old = loop.build_round(
        _round_items(),
        round_id=run.name,
        sizes={"clipA": SIZE},
        stems={"clipA": "clipA_stem"},
        thr=thr,
    )
    (run / "cards_meta.json").write_text(json.dumps(old["meta"]))
    stored = {
        k: v
        for k, v in loop.asdict_thr(thr).items()
        if k
        not in (
            "scheme",
            "audit_min",
            "settled_rate",
            "settled_min",
            "settled_rounds",
            "drop_precision",
            "random_frames",
        )
    }
    stored["audit_rate"] = 0.10
    (run / "round_report.json").write_text(
        json.dumps(
            {
                "round": "r1",
                "created_at": 1.0,
                "thresholds": stored,
                "breaker": {"open": False, "why": "x"},
                "counts": old["counts"],
                "clips": 1,
                "frames": 40,
            }
        )
    )
    assert (
        cli._round_thr(json.loads((run / "round_report.json").read_text()), None).scheme
        == 1
    )
    sp = _split(tmp_path)
    args = [
        "ingest",
        "--run",
        str(run),
        "--split",
        str(sp),
        "--answers",
        str(run / "answers.json"),
    ]
    args += ["--verdicts-dir", str(tmp_path / "v"), "--labels-dir", str(tmp_path / "l")]
    assert cli.main(args) == 0
    assert not (
        run.parent / "gate_state.json"
    ).exists()  # legacy rounds leave no gate history


# ------------------------------------------------------------ eval registry, sparse eval


def test_registered_eval_clips_are_refused_by_the_loop_guard(tmp_path):
    reg = tmp_path / "reg.json"
    eval_registry.register_clips(reg, ["arsenal_mancity_20250925_seg044"])
    ids = loop.eval_clip_ids(_split(tmp_path), registry=reg)
    assert "arsenal_mancity_seg044" in ids
    for alias in ("arsenal_mancity_seg044", "mancity_seg044.mp4"):
        with pytest.raises(loop.EvalClipError):
            loop.assert_not_eval([alias], ids)
    d = _videos(tmp_path, ["arsenal_mancity_20250925_seg044", "x_seg001", "x_seg002"])
    assert loop.pick_clips(d, ids, n_random=2) == ["x_seg001", "x_seg002"]
    with pytest.raises(loop.EvalClipError):
        loop.pick_clips(d, ids, names=["arsenal_mancity_seg044"])
    # through the CLI (exit code 4) and through $FOOTY_EVAL_REGISTRY
    sp = _split(tmp_path)
    run = tmp_path / "run"
    rc = cli.main(
        ["pool", "--run", str(run), "--split", str(sp), "--clips-dir", str(d)]
        + ["--clips", "arsenal_mancity_seg044", "--eval-registry", str(reg)]
    )
    assert rc == 4
    assert (
        cli.main(
            [
                "check",
                "--split",
                str(sp),
                "--clips",
                "mancity_seg044",
                "--eval-registry",
                str(reg),
            ]
        )
        == 4
    )
    assert (
        cli.main(
            [
                "check",
                "--split",
                str(sp),
                "--clips",
                "x_seg001",
                "--eval-registry",
                str(reg),
            ]
        )
        == 0
    )


def test_registry_from_environment(tmp_path, monkeypatch):
    reg = tmp_path / "reg.json"
    eval_registry.register_clips(reg, ["x_seg009"])
    monkeypatch.setenv("FOOTY_EVAL_REGISTRY", str(reg))
    assert "x_seg009" in loop.eval_clip_ids(_split(tmp_path))
    assert eval_registry.register_clips(reg, ["x_seg009"]) == []  # idempotent
    assert eval_registry.read_registry(None) == {}


def test_eval_register_refuses_a_clip_an_earlier_round_used(tmp_path, capsys):
    d = _videos(tmp_path, ["a_seg001", "a_seg002"])
    runs = tmp_path / "runs"
    (runs / "r1").mkdir(parents=True)
    (runs / "r1" / "clips.json").write_text(json.dumps({"clips": ["a_seg001"]}))
    reg = tmp_path / "reg.json"
    base = ["eval-register", "--registry", str(reg), "--clips-dir", str(d)]
    assert cli.main([*base, "--clips", "a_seg001", "--rounds-dir", str(runs)]) == 5
    assert not reg.exists()
    assert cli.main([*base, "--clips", "nosuch"]) == 2
    assert cli.main([*base, "--clips", "a_seg002", "--rounds-dir", str(runs)]) == 0
    assert set(eval_registry.read_registry(reg)) == {"a_seg002"}


def test_eval_list_makes_one_place_card_per_second_and_finalise_writes_no_ball(
    tmp_path, monkeypatch
):
    d = _videos(tmp_path, ["a_seg002"])
    reg = tmp_path / "reg.json"
    eval_registry.register_clips(reg, ["a_seg002"])
    monkeypatch.setattr(cli, "_video_info", lambda p: (25.0, 100, (640, 360)))
    out = tmp_path / "list.json"
    rc = cli.main(
        ["eval-list", "--registry", str(reg), "--clips-dir", str(d), "--out", str(out)]
    )
    assert rc == 0
    cards = json.loads(out.read_text())["items"]
    assert [c["frame_index"] for c in cards] == [0, 25, 50, 75]
    assert all(
        c["bbox"] is None and c["place"] and c["clip"] == "a_seg002" for c in cards
    )
    assert len({c["id"] for c in cards}) == 4
    # a clip that is not registered is refused
    assert (
        cli.main(
            [
                "eval-list",
                "--registry",
                str(reg),
                "--clips-dir",
                str(d),
                "--clips",
                "zzz",
                "--out",
                str(out),
            ]
        )
        == 5
    )
    gt = tmp_path / "gt"
    (gt / "ball_checks").mkdir(parents=True)
    with (gt / "ball_checks" / "a_seg002.jsonl").open("w") as fh:
        for c, v in zip(cards, ("not_ball", "corrected", "unsure", None), strict=True):
            if v:
                fh.write(
                    json.dumps(
                        {
                            "clip": "a_seg002",
                            "frame_index": c["frame_index"],
                            "box_index": card_box_index(c["id"]),
                            "source": "list",
                            "verdict": v,
                        }
                    )
                    + "\n"
                )
    (gt / "a_seg002.jsonl").write_text(
        json.dumps(
            {
                "frame_index": 25,
                "bbox": {"x": 0.1, "y": 0.1, "w": 0.01, "h": 0.01},
                "tags": ["in_play_ball", "labeller"],
            }
        )
        + "\n"
    )
    args = ["eval-finalise", "--list", str(out), "--gt-dir", str(gt)]
    assert cli.main(args) == 0
    lines = [json.loads(x) for x in (gt / "a_seg002.jsonl").read_text().splitlines()]
    assert [ln["frame_index"] for ln in lines] == [25, 0]
    assert lines[1]["tags"] == ["no_ball"] and lines[1]["bbox"] is None
    assert cli.main(args) == 0  # idempotent
    assert len((gt / "a_seg002.jsonl").read_text().splitlines()) == 2
