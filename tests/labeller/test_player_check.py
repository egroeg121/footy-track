"""Player Check API (README section 19i, LAB-1451..): queue, verdicts, completeness, split guard."""

from __future__ import annotations

import json

import pytest

from footy_track import splits
from footy_track.feature_store import FeatureStore
from footy_track.labeller import player_check

CLIP = "pc_clip"
TRIO = [(0.10, 0.5), (0.12, 0.5), (0.14, 0.5)]
LONER = (0.80, 0.5)


def _line(fi, x, y, tags=("player", "yolo")):
    return json.dumps(
        {
            "frame_index": fi,
            "bbox": {"x": x, "y": y, "w": 0.05, "h": 0.15},
            "center": {"x": x + 0.025, "y": y + 0.075},
            "tags": list(tags),
        }
    )


@pytest.fixture
def env(tmp_path, monkeypatch, gt_marks_dir):
    monkeypatch.setattr(player_check, "_db_path", lambda: tmp_path / "fs.duckdb")
    monkeypatch.setattr(splits, "split_of", lambda *a, **k: splits.TRAIN)
    lines = []
    for f in range(4):
        pts = TRIO[:2] if f == 2 else TRIO  # frame 2 flickers 3,3,2,3
        lines += [_line(f, x, y) for x, y in [*pts, LONER]]
    (gt_marks_dir / f"{CLIP}.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path / "fs.duckdb"


def _queue(client, **kw):
    return client.get("/player-check/queue", params={"clip": CLIP, **kw})


def _post(client, **body):
    return client.post("/player-check/verdict", json={"clip": CLIP, **body})


def _states(db):
    with FeatureStore.open(db) as s:
        df = s.read_frame_label_states(CLIP, "player")
    return {int(r.frame_index): (r.state, r.method) for r in df.itertuples()}


def _cluster_ids(q):
    big = next(c for c in q["clusters"] if max(t["count"] for t in c["timeline"]) == 3)
    lone = next(c for c in q["clusters"] if max(t["count"] for t in c["timeline"]) == 1)
    return big, lone


def test_queue_flags_flicker_first_and_audits_everything_by_default(client, env):
    q = _queue(client).json()
    assert q["policy"] == "audit_all" and q["auto_ok"] == []
    assert len(q["clusters"]) == 2 and len(q["sweeps"]) == 4
    big, _lone = _cluster_ids(q)
    assert q["clusters"][0]["cluster_id"] == big["cluster_id"]  # worst first
    assert big["flicker_frames"] == [2]
    assert [t["count"] for t in big["timeline"]] == [3, 3, 2, 3]
    # blind audit subset is deterministic and carries no boxes
    assert _queue(client).json()["audits"] == q["audits"]
    assert all(a["boxes"] == [] for a in q["audits"])


def test_permissive_policy_drops_isolated_stable_track(client, env):
    q = _queue(client, policy="permissive").json()
    assert len(q["auto_ok"]) == 1 and len(q["clusters"]) == 1


def test_frame_complete_only_when_all_clusters_and_sweep_verified(client, env):
    big, lone = _cluster_ids(_queue(client).json())
    r = _post(client, cluster_id=big["cluster_id"], verdict="ok").json()
    assert r["complete_frames"] == [] and r["partial_frames"] == [0, 1, 2, 3]
    assert {s for s, _ in _states(env).values()} == {"partial"}
    _post(client, cluster_id=lone["cluster_id"], verdict="ok")
    assert {s for s, _ in _states(env).values()} == {"partial"}  # sweep still missing
    r = _post(client, sweep=0, verdict="ok").json()
    assert r["complete_frames"] == [0]
    assert _states(env)[0] == ("complete", "hand_frame")
    assert _states(env)[1][0] == "partial"


def test_unsure_never_completes(client, env):
    big, lone = _cluster_ids(_queue(client).json())
    _post(client, cluster_id=big["cluster_id"], verdict="ok")
    _post(client, cluster_id=lone["cluster_id"], verdict="unsure")
    r = _post(client, sweep=0, verdict="ok").json()
    assert r["complete_frames"] == [] and _states(env)[0][0] == "partial"
    r = _post(client, sweep=1, verdict="unsure").json()
    assert r["complete_frames"] == []


def test_count_is_n_resolves_a_run(client, env):
    big, lone = _cluster_ids(_queue(client).json())
    _post(client, cluster_id=lone["cluster_id"], verdict="ok")
    r = _post(
        client, cluster_id=big["cluster_id"], verdict="count_is_3", start=0, end=1
    ).json()
    assert r["frames_written"] == [0, 1] and r["needs_fix"] == []
    _post(client, sweep=0, verdict="ok")
    assert _states(env)[0][0] == "complete"
    with FeatureStore.open(env) as s:
        df = s.read_player_checks(CLIP, [0, 1])
    rows = df[df.cluster_id == big["cluster_id"]]
    assert set(rows["verdict"]) == {"count_n"} and set(rows["count"]) == {3}
    assert set(rows["run_id"]) == {f"{big['cluster_id']}:0-1"}
    # asserted count 3 disagrees with frame 2's two boxes: flagged for a fix, not completed
    r = _post(
        client, cluster_id=big["cluster_id"], verdict="count_is_3", start=2, end=2
    ).json()
    assert r["needs_fix"] == [2] and r["complete_frames"] == []


def test_fixed_writes_hand_label_boxes(client, env, gt_marks_dir):
    big, _lone = _cluster_ids(_queue(client).json())
    add = {"label": "player", "bbox": {"x": 0.14, "y": 0.5, "w": 0.05, "h": 0.15}}
    fix = [{"frame_index": 2, "delete": [], "add": [add]}]
    r = _post(
        client, cluster_id=big["cluster_id"], verdict="fixed", start=2, end=2, fix=fix
    ).json()
    assert r["ok"] and r["frames_written"] == [2]
    recs = [
        json.loads(line)
        for line in (gt_marks_dir / f"{CLIP}.jsonl").read_text().splitlines()
    ]
    f2 = [x for x in recs if x["frame_index"] == 2]
    assert len(f2) == 4 and f2[-1]["tags"] == ["player", "labeller"]  # hand_label GT
    assert _states(env)[2][0] == "partial"  # loner and sweep still pending


def test_fixed_without_boxes_is_rejected(client, env):
    big, _ = _cluster_ids(_queue(client).json())
    assert (
        _post(client, cluster_id=big["cluster_id"], verdict="fixed").status_code == 422
    )


def test_heldout_refused_unless_eval_flag(client, env, monkeypatch):
    monkeypatch.setattr(splits, "split_of", lambda *a, **k: splits.EVAL_HELDOUT)
    assert _queue(client).status_code == 403
    assert _post(client, sweep=0, verdict="ok").status_code == 403
    ok = _queue(client, eval="true")
    assert ok.status_code == 200 and ok.json()["split"] == splits.EVAL_HELDOUT
    assert _post(client, sweep=0, verdict="ok", eval=True).status_code == 200
