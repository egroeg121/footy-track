"""Series review API (LAB-1460..1468): accept-all + exceptions, reject/re-propose, heldout refusal.

No video, no GPU: the proposal is fabricated straight into a tmp feature store and ``proposals/*.npz``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track import match_splits as ms
from footy_track import splits as sp
from footy_track.feature_store.schema import (
    DetectionRow,
    FrameRow,
    GameRow,
    RunRow,
    Stage,
)
from footy_track.feature_store.store import FeatureStore
from footy_track.labeller import series_review as sr

MATCHES = {
    "arsenal_mancity": {"split": "train", "aliases": ["mancity"]},
    "arsenal_nottinghamforest": {"split": "heldout", "aliases": ["nottinghamforest"]},
}
TRAIN = "arsenal_mancity_seg000"
HELD = "arsenal_nottinghamforest_seg000"
START, END = 0, 40  # 40-frame windows: limit 4 exceptions, 4 audit frames


@pytest.fixture
def env(tmp_path, monkeypatch):
    mp = tmp_path / "match_splits.json"
    mp.write_text(json.dumps(MATCHES))
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(mp))
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED", raising=False)
    monkeypatch.setenv("FOOTY_FULL_MATCHES_MANIFEST", str(tmp_path / "none.jsonl"))
    clips = {
        f"{m}_seg000": {"split": "train"}
        for m in ("arsenal_mancity", "arsenal_nottinghamforest")
    }
    cp = tmp_path / "clip_splits.json"
    cp.write_text(
        json.dumps(
            {"version": 1, "rule": {"modulus": 5, "eval_remainder": 0}, "clips": clips}
        )
    )
    monkeypatch.setenv("FOOTY_CLIP_SPLITS", str(cp))
    sp.clear_cache()
    ms.clear_cache()
    monkeypatch.setenv("FOOTY_AUTOLABEL_DIR", str(tmp_path / "autolabel"))
    monkeypatch.setenv("FOOTY_FEATURE_STORE", str(tmp_path / "fs.duckdb"))
    yield tmp_path
    sp.clear_cache()
    ms.clear_cache()


def _window(game: str) -> dict:
    return {
        "window_id": f"{game}_f{START:06d}", "game_id": game, "clip_id": game, "match_key": "x",
        "video": None, "start_frame": START, "end_frame": END, "labelled_frames": 0,
        "agreement": None, "rank": 0,
    }  # fmt: skip


def propose(env, game=TRAIN, *, ball_frames=None, player_frames=None, flagged=()):
    """Fabricate windows.json, a proposal in the store and the disagreement npz."""
    wid = f"{game}_f{START:06d}"
    al = env / "autolabel"
    (al / "proposals").mkdir(parents=True, exist_ok=True)
    (al / "windows.json").write_text(
        json.dumps(
            {"version": 1, "window_frames": END - START, "windows": [_window(game)]}
        )
    )
    score = np.zeros(END - START)
    for f in flagged:
        score[f - START] = 0.9
    np.savez(
        al / "proposals" / f"{wid}.npz",
        frame_index=np.arange(START, END),
        score=score,
        person_flag=np.zeros(END - START, bool),
        ball_presence_mismatch=np.zeros(END - START, bool),
    )
    ball_frames = range(START, END) if ball_frames is None else ball_frames
    player_frames = range(START, END) if player_frames is None else player_frames
    base = f"autolabel:{wid}"
    with FeatureStore.open(env / "fs.duckdb") as st:
        st.upsert_games([GameRow(game_id=game, fps=50.0, width=1920, height=1080)])
        st.upsert_frames(
            [
                FrameRow(game_id=game, frame_index=i, frame_uri=f"{game}_{i}", width=1920, height=1080,
                         continuous_time_s=i / 50)
                for i in range(START, END)
            ]
        )  # fmt: skip
        st.upsert_runs(
            [
                RunRow(
                    run_id=base,
                    stage=Stage.DETECTION,
                    source="autolabel",
                    model_name="m",
                ),
                RunRow(
                    run_id=f"{base}:yolo",
                    stage=Stage.DETECTION,
                    source="yolo",
                    model_name="m",
                ),
                RunRow(
                    run_id=f"{base}:detector_track",
                    stage=Stage.DETECTION,
                    source="detector_track",
                    model_name="m",
                ),
            ]
        )

        def row(fi, src, label):
            return DetectionRow(
                game_id=game, frame_index=fi, continuous_time_s=fi / 50, detection_id=0, source=src,
                run_id=f"{base}:{src}", label=label, confidence=0.9, bbox_x=0.1, bbox_y=0.1,
                bbox_w=0.05, bbox_h=0.1, needs_review=True,
            )  # fmt: skip

        st.upsert_detections(
            [row(f, "yolo", "player") for f in player_frames]
            + [row(f, "detector_track", "ball") for f in ball_frames]
        )
    return wid


def detail(client, wid):
    r = client.get(f"/series/window/{wid}")
    assert r.status_code == 200, r.text
    return r.json()


def audit_ok(d, ok=True):
    return [{"frame_index": f, "ok": ok} for f in d["audit_frames"]]


def non_audit(d, n=1):
    return [x["frame_index"] for x in d["frames"] if not x["audit"]][:n]


def states(env, game=TRAIN):
    with FeatureStore.open(env / "fs.duckdb") as st:
        df = st.read_frame_label_states(game)
    return {
        (int(r.frame_index), r.cls_group): (r.state, r.method, r.run_id)
        for r in df.itertuples()
    }


def test_get_window_flagged_first_and_audit_boxes_hidden(client, env):
    wid = propose(env, flagged=[7, 30])
    d = detail(client, wid)
    assert {f["frame_index"] for f in d["frames"][:2]} == {7, 30}
    assert all(f["flagged"] for f in d["frames"][:2])
    audit = [f for f in d["frames"] if f["audit"]]
    assert len(audit) == len(d["audit_frames"]) == 4  # 10% of 40
    assert all(f["boxes"] == [] and not f["flagged"] for f in audit)
    assert all(f["boxes"] for f in d["frames"] if not f["audit"])
    assert detail(client, wid)["audit_frames"] == d["audit_frames"]  # deterministic
    assert client.get("/series/windows").json()["windows"][0]["status"] == "proposed"


def test_accept_all_writes_complete_hand_series_for_both_groups(client, env):
    wid = propose(env)
    d = detail(client, wid)
    r = client.post(f"/series/window/{wid}/accept", json={"audit_results": audit_ok(d)})
    assert r.status_code == 200 and r.json()["status"] == "accepted", r.text
    s = states(env)
    assert len(s) == 2 * (END - START)
    assert all(v == ("complete", "hand_series", f"autolabel:{wid}") for v in s.values())
    with FeatureStore.open(env / "fs.duckdb") as st:
        n = st.query(
            "SELECT count(*) n FROM detection WHERE reviewed AND starts_with(run_id, 'autolabel:')"
        )["n"][0]
    assert n == 2 * (END - START)
    assert client.get("/series/windows").json()["windows"][0]["status"] == "accepted"


def test_no_ball_or_players_gives_absent(client, env):
    wid = propose(
        env, ball_frames=range(START + 1, END), player_frames=range(START, END - 1)
    )
    d = detail(client, wid)
    client.post(f"/series/window/{wid}/accept", json={"audit_results": audit_ok(d)})
    s = states(env)
    assert s[(START, "ball")][0] == "absent"
    assert s[(END - 1, "player")][0] == "absent"


def test_one_exception_demotes_only_that_frame(client, env):
    wid = propose(env)
    d = detail(client, wid)
    (f,) = non_audit(d)
    body = {
        "exceptions": [{"frame_index": f, "kind": "players_incomplete"}],
        "audit_results": audit_ok(d),
    }
    assert (
        client.post(f"/series/window/{wid}/accept", json=body).json()["status"]
        == "accepted"
    )
    s = states(env)
    assert s[(f, "player")][0] == "partial"
    assert s[(f, "ball")][0] == "complete"
    assert all(v[0] == "complete" for k, v in s.items() if k != (f, "player"))
    with FeatureStore.open(env / "fs.duckdb") as st:
        assert st.read_series().empty  # not every frame complete -> no series
        done = st.query("SELECT count(*) n FROM frame_complete")["n"][0]
    assert done == END - START - 1


def test_ball_not_ball_unreviews_ball_and_records_rejection(client, env):
    wid = propose(env)
    d = detail(client, wid)
    (f,) = non_audit(d)
    body = {
        "exceptions": [{"frame_index": f, "kind": "ball_not_ball"}],
        "audit_results": audit_ok(d),
    }
    client.post(f"/series/window/{wid}/accept", json=body)
    assert states(env)[(f, "ball")][0] == "unreviewed"
    with FeatureStore.open(env / "fs.duckdb") as st:
        ball = st.query(
            "SELECT reviewed FROM detection WHERE frame_index = ? AND label = 'ball'",
            [f],
        )
        player = st.query(
            "SELECT reviewed FROM detection WHERE frame_index = ? AND label = 'player'",
            [f],
        )
        rv = st.query(
            "SELECT verdict, matched_source FROM detection_review WHERE game_id = ? AND frame_index = ?",
            [TRAIN, f],
        )
    assert not ball["reviewed"][0]
    assert player["reviewed"][0]
    assert rv["verdict"].tolist() == ["not_ball"]
    assert rv["matched_source"][0] == "detector_track"


def test_ball_absent_marks_ball_absent(client, env):
    wid = propose(env)
    d = detail(client, wid)
    (f,) = non_audit(d)
    body = {
        "exceptions": [{"frame_index": f, "kind": "ball_absent"}],
        "audit_results": audit_ok(d),
    }
    client.post(f"/series/window/{wid}/accept", json=body)
    assert states(env)[(f, "ball")][0] == "absent"
    with FeatureStore.open(env / "fs.duckdb") as st:
        # absent ball + complete players is a complete frame, so the whole window is still a series
        assert len(st.read_series()) == 1


def test_fix_box_goes_to_sidecar_as_hand_label(client, env, gt_marks_dir):
    wid = propose(env)
    d = detail(client, wid)
    (f,) = non_audit(d)
    fix = {
        "frame_index": f,
        "kind": "fix_box",
        "label": "ball",
        "bbox": {"x": 0.2, "y": 0.3, "w": 0.01, "h": 0.02},
    }
    body = {"exceptions": [fix], "audit_results": audit_ok(d)}
    assert (
        client.post(f"/series/window/{wid}/accept", json=body).json()["status"]
        == "accepted"
    )
    lines = [
        json.loads(x)
        for x in (gt_marks_dir / f"{TRAIN}.jsonl").read_text().splitlines()
    ]
    assert lines[0]["frame_index"] == f
    assert lines[0]["tags"] == ["ball", "labeller"]
    assert states(env)[(f, "ball")] == ("complete", "hand_frame", f"autolabel:{wid}")
    with FeatureStore.open(env / "fs.duckdb") as st:
        r = st.query(
            "SELECT reviewed FROM detection WHERE frame_index = ? AND label = 'ball'",
            [f],
        )
    assert not r["reviewed"][0]  # the machine box is superseded by the hand label


def test_too_many_exceptions_rejects_without_state_and_requeues(client, env):
    wid = propose(env)
    d = detail(client, wid)
    fs = non_audit(d, 5)  # limit is 4 of 40
    body = {
        "exceptions": [{"frame_index": f, "kind": "ball_absent"} for f in fs],
        "audit_results": audit_ok(d),
    }
    r = client.post(f"/series/window/{wid}/accept", json=body).json()
    assert r["status"] == "rejected"
    assert r["reproposal_queued"]
    assert states(env) == {}
    with FeatureStore.open(env / "fs.duckdb") as st:
        assert st.query("SELECT 1 FROM detection WHERE reviewed").empty
        assert st.read_series().empty
    q = sr.reproposal_queue()
    assert [x["window_id"] for x in q] == [wid]
    assert {a["frame_index"] for a in q[0]["anchors"]} == set(fs)
    saved = json.loads(
        (env / "autolabel" / "series_review" / f"{wid}.json").read_text()
    )
    assert saved["status"] == "rejected"
    # a rejected window cannot be accepted again until reopened
    assert (
        client.post(
            f"/series/window/{wid}/accept", json={"audit_results": audit_ok(d)}
        ).status_code
        == 409
    )
    assert client.post(f"/series/window/{wid}/reopen").status_code == 200
    assert sr.reproposal_queue() == []


def test_exactly_the_limit_is_accepted(client, env):
    wid = propose(env)
    d = detail(client, wid)
    body = {
        "exceptions": [
            {"frame_index": f, "kind": "ball_absent"} for f in non_audit(d, 4)
        ],
        "audit_results": audit_ok(d),
    }
    assert (
        client.post(f"/series/window/{wid}/accept", json=body).json()["status"]
        == "accepted"
    )


def test_audit_failure_rejects_and_keeps_human_anchor(client, env):
    wid = propose(env)
    d = detail(client, wid)
    res = audit_ok(d)
    res[0] = {
        "frame_index": res[0]["frame_index"],
        "ok": False,
        "ball_bbox": {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.01},
    }
    r = client.post(f"/series/window/{wid}/accept", json={"audit_results": res}).json()
    assert r["status"] == "rejected"
    assert r["audit_failed"] == [res[0]["frame_index"]]
    assert states(env) == {}
    anchor = sr.reproposal_queue()[0]["anchors"][0]
    assert anchor["kind"] == "audit_failed"
    assert anchor["bbox"]["x"] == 0.5


def test_audit_results_must_cover_audit_frames(client, env):
    wid = propose(env)
    assert client.post(f"/series/window/{wid}/accept", json={}).status_code == 422
    d = detail(client, wid)
    bad = {
        "exceptions": [{"frame_index": d["audit_frames"][0], "kind": "ball_absent"}],
        "audit_results": audit_ok(d),
    }
    assert client.post(f"/series/window/{wid}/accept", json=bad).status_code == 422


def test_heldout_window_refused_everywhere(client, env):
    wid = propose(env, game=HELD)
    assert client.get(f"/series/window/{wid}").status_code == 403
    assert client.post(f"/series/window/{wid}/accept", json={}).status_code == 403
    assert client.get("/series/windows").json()["windows"] == []
    assert states(env, HELD) == {}


def test_full_window_registers_series_and_reaccept_is_idempotent(client, env):
    wid = propose(env)
    d = detail(client, wid)
    r1 = client.post(
        f"/series/window/{wid}/accept", json={"audit_results": audit_ok(d)}
    ).json()
    assert r1["series_registered"]
    assert not r1["idempotent"]
    with FeatureStore.open(env / "fs.duckdb") as st:
        rows = st.read_series()[
            ["game_id", "start_frame", "end_frame", "run_id"]
        ].values.tolist()
        before = st.query("SELECT * FROM frame_label_state ORDER BY ALL")
    assert rows == [[TRAIN, START, END, f"autolabel:{wid}"]]
    r2 = client.post(
        f"/series/window/{wid}/accept", json={"audit_results": audit_ok(d, ok=False)}
    )
    assert r2.status_code == 200
    assert r2.json()["idempotent"]
    assert r2.json()["status"] == "accepted"
    with FeatureStore.open(env / "fs.duckdb") as st:
        assert len(st.read_series()) == 1
        assert st.query("SELECT * FROM frame_label_state ORDER BY ALL").equals(before)


def test_page_is_served(client):
    r = client.get("/series")
    assert r.status_code == 200
    assert "Series review" in r.text
