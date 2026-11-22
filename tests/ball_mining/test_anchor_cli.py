"""Anchor search CLI (design task 5): ``run`` / ``sweep`` over a fake jobs client (the in-process job store), idempotent, shares
the job mutex with refit / close. Synthetic candidates and a fake feature store only."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.ball_mining import refit
from footy_track.ball_mining.round import ROUND_HOOKS
from footy_track.labeller import jobs
from footy_track.scripts import ball_anchor_search as cli

from .test_anchor_client import DAY, KEY, AppendApi, seed, submit  # noqa: F401
from .test_anchor_search import TC, verdict

AX, AY = 0.41, 0.52
FRAMES = list(range(0, 100, 10))


class Store:
    def read_selected_frames(self, job_key=None, **_):
        return pd.DataFrame(
            {"job_key": job_key, "game_id": TC, "frame_index": FRAMES, "split": "train"}
        )

    def read_ball_terminal_states(self, game, frames=None):
        return {}


@pytest.fixture
def setup(ds, monkeypatch):
    cs = A.CandidateSet("job_suggest", "abcd1234")
    for f in FRAMES:
        cs.by_frame[(TC, f)] = [A.Cand(AX + 0.001 * f / 10, AY, 0.02, 0.04, 0.8, 7)]
    monkeypatch.setattr(A, "load_candidates", lambda j, **k: cs)
    jid = submit([seed(0, "R")])
    verdict(ds, TC, 0, "ball", bbox={"x": 0.4, "y": 0.5, "w": 0.02, "h": 0.04})
    return jid, AppendApi()


def run(jid, cl, ds, **kw):
    return cli.run_job(jid, client=cl, marks_dir=ds, store=Store(), today=DAY, **kw)


def test_run_appends_search_cards_and_a_second_run_adds_nothing(setup, ds):
    jid, cl = setup
    first = run(jid, cl, ds)
    assert (
        first["status"] == "ok" and first["push"]["added"] == 3 and not first["noop"]
    )  # hop cap 3
    n = len(jobs.read_job(jid)["items"])
    assert {it["sampling"]["bucket"] for it in jobs.read_job(jid)["items"][1:]} == {"S"}
    second = run(jid, cl, ds)
    assert second["push"]["added"] == 0 and len(jobs.read_job(jid)["items"]) == n
    third = run(jid, cl, ds)
    assert (
        third["noop"] and third["new_files"] == 0 and third["push"]["added"] == 0
    )  # same digests: nothing written


def test_dry_run_writes_nothing(setup, ds):
    jid, cl = setup
    out = run(jid, cl, ds, dry_run=True)
    assert out["dry_run"] and out["push"]["would_add"] == 3
    assert len(jobs.read_job(jid)["items"]) == 1 and not (ds / "search").exists()


def test_busy_job_is_skipped_cleanly_and_unlocks_after(setup, ds):
    jid, cl = setup
    with refit.job_lock(ds, jid):
        out = run(jid, cl, ds)
    assert (
        out["status"] == "skipped_busy"
        and out["noop"]
        and len(jobs.read_job(jid)["items"]) == 1
    )
    assert run(jid, cl, ds)["push"]["added"] == 3


def test_overlapping_runs_converge(setup, ds):
    jid, cl = setup
    a = run(jid, cl, ds)
    b = run(jid, cl, ds)
    assert a["push"]["added"] + b["push"]["added"] == 3
    frames = [it["frame_index"] for it in jobs.read_job(jid)["items"]]
    assert len(frames) == len(set(frames))


def test_sweep_covers_live_round_jobs_and_is_a_noop_when_nothing_changed(setup, ds):
    jid, cl = setup
    other = submit([seed(5, "R")], key="not-a-round/x")  # not a ball-round job: ignored
    first = cli.sweep(client=cl, marks_dir=ds, store=Store(), today=DAY)
    assert first["n_jobs"] == 1 and first["jobs"][0]["job"] == jid and not first["noop"]
    cli.sweep(
        client=cl, marks_dir=ds, store=Store(), today=DAY
    )  # appended frames become pending: one more converging file
    again = cli.sweep(client=cl, marks_dir=ds, store=Store(), today=DAY)
    assert again["noop"] and again["n_changed"] == 0 and again["status"] == "ok"
    assert other not in [r["job"] for r in again["jobs"]]


def test_a_done_job_is_skipped(setup, ds):
    jid, cl = setup
    out = cli.run_job(
        jid,
        client=type(
            "C",
            (),
            {
                "results": lambda self, ref: {
                    **cl.results(ref),
                    "job": {**cl.results(ref)["job"], "status": "done"},
                }
            },
        )(),
        marks_dir=ds,
    )
    assert out["status"] == "skipped" and out["noop"]


def test_main_prints_one_json_line(setup, ds, monkeypatch, capsys):
    jid, cl = setup
    monkeypatch.setattr(cli, "jobs_client", cl)
    monkeypatch.setattr(cli, "_open_store", lambda p: Store())
    assert (
        cli.main(
            ["run", "--job", jid, "--marks-dir", str(ds), "--store", "x", "--dry-run"]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok" and out["dry_run"] is True


def test_the_search_hook_is_in_round_hooks():
    (h,) = [h for h in ROUND_HOOKS if h["name"] == "anchor-search"]
    assert (
        h["trigger"] == {"every_n": 10, "min_interval_s": 60}
        and h["action"]["deployment"] == "ball-anchor-search/ball-anchor-search"
    )
