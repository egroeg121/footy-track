"""Reconciler: each detection on fixtures, the clean case, read-only behaviour and the CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from footy_track.ball_mining import reconcile
from footy_track.feature_store.schema import SelectedFrameRow
from footy_track.feature_store.store import FeatureStore
from footy_track.scripts import ball_reconcile

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
OLD = (NOW - timedelta(days=10)).isoformat()
FRESH = (NOW - timedelta(hours=1)).isoformat()
TRAIN_KEY, EVAL_KEY = "ball-round/2026-11-20/1", "ball-eval/p1/m9"


def split_fn(clip: str) -> str:
    return "eval_heldout_match" if clip.startswith("e_") else "train"


class FakeClient:
    """Double of the jobs client: ``jobs`` is ``{key: {"created", "frames": [(clip, frame)], "verdicts": {(clip, frame): verdict}}}``."""

    def __init__(self, jobs: dict) -> None:
        self.jobs = jobs

    def list_jobs(self, **_kw):
        return [{"id": f"id{i}", "key": k, "kind": "ball_check", "status": "open", "created": j["created"]} for i, (k, j) in enumerate(self.jobs.items())]

    def results(self, ref: str) -> dict:
        key = list(self.jobs)[int(ref[2:])]
        j = self.jobs[key]
        items = [{"index": i, "item": {"clip": c, "frame_index": f, "bbox": {"x": 0.1, "y": 0.2, "w": 0.01, "h": 0.01}}} for i, (c, f) in enumerate(j["frames"])]
        labels = [{"clip": c, "frame_index": f, "verdict": v} for (c, f), v in j.get("verdicts", {}).items()]
        return {"job": {"id": ref}, "items": items, "labels": labels}


def sel(job, clip, frame, split="train", at=None):
    return SelectedFrameRow(job_key=job, game_id=clip, frame_index=frame, window_id="w", split=split, selected_at=at or NOW - timedelta(hours=1))


def make_store(rows, terminals=()):
    st = FeatureStore.open(":memory:")
    st.upsert_selected_frames(rows)
    for clip, frame in terminals:
        st.write_ball_terminal_state(clip, frame, "ball", method="verdict")
    return st


def run(store, jobs, **kw):
    return reconcile.reconcile(store, FakeClient(jobs), now=NOW, split_fn=split_fn, **kw)


def clean_setup():
    store = make_store([sel(TRAIN_KEY, "m1", 1), sel(TRAIN_KEY, "m1", 2), sel(EVAL_KEY, "e_m9", 5, "eval_heldout_match")], terminals=[("m1", 1)])
    jobs = {
        TRAIN_KEY: {"created": FRESH, "frames": [("m1", 1), ("m1", 2)], "verdicts": {("m1", 1): "ball"}},
        EVAL_KEY: {"created": FRESH, "frames": [("e_m9", 5)]},
    }
    return store, jobs


def kinds(rep, fam):
    return {k: n for k, n in rep["families"][fam]["counts"].items() if n}


def test_clean():
    rep = run(*clean_setup())
    assert rep["ok"] and rep["violations"] == 0
    assert kinds(rep, "train") == {} and kinds(rep, "eval") == {}
    assert rep["families"]["train"]["frames_expected"] == 2


def test_orphan_card_and_verdict():
    store, jobs = clean_setup()
    jobs[TRAIN_KEY]["frames"].append(("m1", 99))
    jobs[TRAIN_KEY]["verdicts"][("m1", 99)] = "not_ball"
    rep = run(store, jobs)
    types = {o["type"] for o in rep["families"]["train"]["orphans"]}
    assert types == {"card", "verdict"} and not rep["ok"]
    assert kinds(rep, "eval") == {}  # families checked independently


def test_orphan_job_without_selection():
    store, jobs = clean_setup()
    jobs["ball-round/2026-11-20/2"] = {"created": FRESH, "frames": [("m1", 7)]}
    rep = run(store, jobs)
    assert rep["families"]["train"]["orphans"][0]["type"] == "job_without_selection"


def test_missing_no_card_and_no_job():
    store, jobs = clean_setup()
    store.upsert_selected_frames([sel(TRAIN_KEY, "m1", 3), sel("ball-round/2026-11-20/9", "m1", 4)])
    rep = run(store, jobs)
    assert {(m["frame"], m["why"]) for m in rep["families"]["train"]["missing"]} == {(3, "no_card"), (4, "no_job")}


def test_stuck_open_card_and_non_terminal_frame():
    store = make_store([sel(TRAIN_KEY, "m1", 1, at=NOW - timedelta(days=9)), sel(TRAIN_KEY, "m1", 2, at=NOW - timedelta(days=9))])
    jobs = {TRAIN_KEY: {"created": OLD, "frames": [("m1", 1), ("m1", 2)]}}
    rep = run(store, jobs, stuck_days=3)
    assert {s["type"] for s in rep["families"]["train"]["stuck"]} == {"open_card", "non_terminal"}
    # fresh job and fresh selection: not stuck
    assert run(*clean_setup(), stuck_days=3)["ok"]


def test_store_lag():
    store, jobs = clean_setup()
    jobs[TRAIN_KEY]["verdicts"][("m1", 2)] = "corrected"  # verdict, but the store has no terminal state yet
    rep = run(store, jobs)
    assert [(x["frame"], x["terminal"]) for x in rep["families"]["train"]["store_lag"]] == [(2, None)]
    jobs[TRAIN_KEY]["verdicts"][("m1", 2)] = "not_ball"  # a not_ball verdict leaves no terminal state: not lag
    assert run(store, jobs)["ok"]


def test_split_violations():
    store, jobs = clean_setup()
    jobs[TRAIN_KEY]["frames"].append(("e_x", 2))  # eval clip in a train job, on an F frame of that job
    store.upsert_selected_frames([sel(TRAIN_KEY, "e_x", 2, "train")])
    store.upsert_selected_frames([sel(EVAL_KEY, "m1", 1, "train")])  # train-split frame in the eval family
    store.upsert_selected_frames([sel(EVAL_KEY, "m1", 2, "eval_heldout_match")])  # F row split disagrees with split_of; also in both families
    rep = run(store, jobs)
    t = {v["type"] for f in rep["families"].values() for v in f["split_violations"]}
    assert {"card_split", "selected_split", "wrong_family", "in_both_families"} <= t


def test_duplicate_cards():
    store, jobs = clean_setup()
    jobs[TRAIN_KEY]["frames"].append(("m1", 2))
    jobs["ball-round/2026-11-20/2"] = {"created": FRESH, "frames": [("m1", 1)]}
    store.upsert_selected_frames([sel("ball-round/2026-11-20/2", "m1", 1)])
    rep = run(store, jobs)
    assert {d["scope"] for d in rep["families"]["train"]["duplicate_cards"]} == {"job", "family"}


def test_read_only_and_family_filter():
    store, jobs = clean_setup()
    before = store.read_selected_frames().to_dict("records")
    run(store, jobs)
    assert store.read_selected_frames().to_dict("records") == before
    rep = run(store, jobs, families=("eval",))
    assert list(rep["families"]) == ["eval"]


def test_cli(tmp_path, capsys):
    store, jobs = clean_setup()
    out = tmp_path / "r.json"
    assert ball_reconcile.main(["--json", str(out)], client=FakeClient(jobs), store=store, split_fn=split_fn) == 0
    assert json.loads(out.read_text())["schema"] == "ball-reconcile/v1"
    jobs[TRAIN_KEY]["frames"].append(("m1", 99))
    assert ball_reconcile.main(["--json", str(out)], client=FakeClient(jobs), store=store, split_fn=split_fn) == 1
    assert ball_reconcile.main(["--apply"], client=FakeClient(jobs), store=store, split_fn=split_fn) == 2
    assert "report-only" in capsys.readouterr().err
