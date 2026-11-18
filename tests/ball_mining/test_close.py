"""Close of a round (library + ``ball_round close``, outside the labeller server): final refit, tier-3 auto-accept files, closed marker."""

from __future__ import annotations

import json

import pytest

from footy_track.ball_mining import close, miner, refit
from footy_track.ball_mining.schema import write_pool
from footy_track.labeller import jobs, jobs_client
from footy_track.scripts import ball_round

from .conftest import make_pool
from .test_refit import (
    answer,
    card,
    round_job,  # noqa: F401 (fixture)
    submit,
)

KEY2 = "ball-round/2026-11-18/2"


@pytest.fixture
def finished(ds, round_job):  # noqa: F811
    """The round with every card answered (the tail too), so the job is ``done``."""
    job = jobs.read_job(round_job["id"])
    for pos in range(710, len(job["items"])):
        answer(
            ds,
            job["id"],
            pos,
            "ball"
            if job["items"][pos]["p_real"] and job["items"][pos]["p_real"] > 0.8
            else "not_ball",
        )
    return job


def pool_rows():
    r = lambda **k: {"p_real": 0.97, "scorer_version": "raw_conf", **k}  # noqa: E731
    return make_pool([
        r(match="m1", clip="m1_seg002", frame=10),
        r(match="m1", clip="m1_seg002", frame=60, p_real=0.9),
        r(match="m1", clip="m1_seg002", frame=120, p_real=0.5),  # below the threshold
        r(match="m1", clip="m1_seg001", frame=12),  # window of a human verdict (frame 0): left to the humans
        r(match="m1", clip="m1_seg009", frame=10),  # an in-match eval clip
        r(match="m1", clip="m1_seg002", frame=200, scorer_version="other"),  # another scorer: not this round's
        r(match="held1", clip="held1_seg001", frame=10),  # a heldout match
        r(match="m2", clip="m2_seg001", frame=10, p_real=0.2),  # a train match with nothing above the threshold
    ])  # fmt: skip


def kw(splitter, tmp_path, **more):
    splitter.eval_clips = {"m1_seg009"}
    return dict(pool=pool_rows(), root=tmp_path / "ball_auto", data_root="/nonexistent", split_of=splitter, match_of=lambda c: c.split("_seg")[0],
                eval_clips=[("m1", "m1_seg009")], **more)  # fmt: skip


def rows_of(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_close_writes_the_tier3_set_for_the_scorer_and_threshold_and_marks_the_round_closed(
    api, ds, finished, splitter, tmp_path
):
    out = close.close_round(finished["id"], client=api, **kw(splitter, tmp_path))
    state = refit.latest_state(ds)
    assert (
        out["status"] == "closed"
        and state["closed"] is True
        and state["n_verdicts"] == out["n_verdicts"]
    )
    thr = state["auto_accept"]["threshold"]
    assert state["auto_accept"]["scale"] == "p_real" and 0.80 <= thr <= 0.85
    root = tmp_path / "ball_auto" / "raw_conf"
    assert sorted(p.name for p in root.iterdir()) == [
        "m1.jsonl",
        "m2.jsonl",
    ]  # the heldout match has no file
    rows = rows_of(root / "m1.jsonl")
    assert [(r["clip"], r["frame_index"]) for r in rows] == [
        ("m1_seg002", 10),
        ("m1_seg002", 60),
    ]
    assert all(
        r["provenance"] == "ball_pool_auto"
        and r["tier"] == 3
        and r["p_real"] >= thr
        and r["scorer_version"] == "raw_conf"
        for r in rows
    )
    assert {r["round"] for r in rows} == {finished["key"]} and rows[0]["bbox"] == {
        "x": 0.5,
        "y": 0.5,
        "w": 0.01,
        "h": 0.02,
    }
    assert (
        rows_of(root / "m2.jsonl") == []
    )  # a train match with nothing above t gets an empty file (a stale one is overwritten)
    aa = out["auto_accept"]
    assert (
        aa["written"]
        and aa["n_accepted"] == 2
        and aa["refused"]["auto_accept"]["rows"]
        == {"eval_heldout_match": 1, "eval_in_match": 1}
    )
    marker = json.loads(close.closed_marker(ds, finished["id"]).read_text())
    assert (
        marker["state_version"] == state["version"]
        and marker["auto_accept"]["files"]["m1"]["n"] == 2
    )


def test_close_writes_nothing_into_the_marks_or_gt_sidecars(
    api, ds, finished, splitter, tmp_path
):
    before = {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}
    close.close_round(finished["id"], client=api, **kw(splitter, tmp_path))
    changed = {
        p for p in ds.rglob("*") if p.is_file() and before.get(p) != p.read_bytes()
    }
    assert {p.relative_to(ds).parts[:2] for p in changed} == {
        ("ball_mining", "state"),
        ("ball_mining", "closed"),
    }
    assert not any(
        "ball_pool_auto" in p.read_text() for p in ds.rglob("*") if p.is_file()
    )


def test_close_is_idempotent_per_job_and_force_redoes_it(
    api, ds, finished, splitter, tmp_path
):
    first = close.close_round(finished["id"], client=api, **kw(splitter, tmp_path))
    f = tmp_path / "ball_auto" / "raw_conf" / "m1.jsonl"
    f.write_text("tampered\n")
    again = close.close_round(finished["id"], client=api, **kw(splitter, tmp_path))
    assert (
        first["status"] == "closed"
        and again["status"] == "already_closed"
        and f.read_text() == "tampered\n"
    )
    forced = close.close_round(
        finished["id"], client=api, force=True, **kw(splitter, tmp_path)
    )
    assert forced["status"] == "closed" and len(rows_of(f)) == 2
    assert [p.name for p in (ds / "ball_mining" / "state").iterdir()] == [
        "v001.json"
    ]  # same verdicts: the state version is reused


def test_a_job_that_is_not_done_is_not_closed(api, ds, round_job, splitter, tmp_path):  # noqa: F811
    with pytest.raises(RuntimeError, match="not done"):
        close.close_round(round_job["id"], client=api, **kw(splitter, tmp_path))
    assert not (tmp_path / "ball_auto").exists() and not close.closed_dir(ds).exists()


def test_without_a_valid_threshold_nothing_is_written_but_the_round_is_closed(
    api, ds, splitter, tmp_path
):
    job = submit([card(i * 100, "A", 0.99, pi=0.1) for i in range(10)])
    for i in range(10):
        answer(ds, job["id"], i, "ball")
    out = close.close_round(job["id"], client=api, **kw(splitter, tmp_path))
    aa = out["auto_accept"]
    assert (
        aa["written"] is False
        and aa["fallback"] is True
        and "claude_verify_conf" in aa["reason"]
    )
    assert (
        not (tmp_path / "ball_auto").exists()
        and close.closed_marker(ds, job["id"]).is_file()
    )


def test_a_flagged_gold_check_blocks_the_auto_accept_unless_allowed(
    api, ds, finished, splitter, tmp_path
):
    other = submit([card(900000 + i * 100, "G", None) for i in range(8)], key=KEY2)
    for i in range(8):
        answer(
            ds, other["id"], i, "ball"
        )  # a labeller who accepts every planted-wrong box
    out = close.close_round(finished["id"], client=api, **kw(splitter, tmp_path))
    assert (
        out["auto_accept"]["written"] is False
        and "gold check flagged" in out["auto_accept"]["reason"]
    )
    forced = close.close_round(
        finished["id"],
        client=api,
        force=True,
        allow_flagged=True,
        **kw(splitter, tmp_path),
    )
    assert forced["auto_accept"]["written"] is True


def test_no_pool_means_nothing_written(api, ds, finished, splitter, tmp_path):
    a = kw(splitter, tmp_path, params=None)
    a["pool"] = tmp_path / "missing_pool"
    out = close.close_round(finished["id"], client=api, **a)
    assert (
        out["auto_accept"]["written"] is False
        and "no pool" in out["auto_accept"]["reason"]
    )


def test_unsafe_scorer_or_match_names_never_become_paths(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        close.write_auto_accept(
            {},
            scorer="../x",
            threshold=0.9,
            root=tmp_path,
            key=None,
            job_id="j",
            version=1,
        )
    with pytest.raises(ValueError, match="unsafe"):
        close.write_auto_accept(
            {"../m": make_pool([{}])},
            scorer="s",
            threshold=0.9,
            root=tmp_path,
            key=None,
            job_id="j",
            version=1,
        )


def test_cli_close_with_a_mocked_labeller_api(
    monkeypatch, api, ds, finished, splitter, tmp_path, capsys
):
    for name in ("results", "list_jobs", "reorder"):
        monkeypatch.setattr(jobs_client, name, getattr(api, name))
    pool = tmp_path / "pool"
    (pool / "m1").mkdir(parents=True)
    write_pool(pool_rows(), pool / "m1" / "pool.raw_conf.parquet")
    monkeypatch.setattr(miner, "default_split_of", splitter)
    monkeypatch.setattr(miner, "default_eval_clips", lambda: [("m1", "m1_seg009")])
    monkeypatch.setattr(miner, "default_match_of", lambda c: c.split("_seg")[0])
    splitter.eval_clips = {"m1_seg009"}
    args = [
        "close",
        "--job",
        finished["id"],
        "--labeller-url",
        "http://x:1",
        "--pool",
        str(pool),
        "--out-root",
        str(tmp_path / "auto"),
        "--data-root",
        "/nonexistent",
    ]
    assert ball_round.main(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert (
        doc["status"] == "closed"
        and doc["auto_accept"]["written"]
        and doc["auto_accept"]["n_accepted"] == 2
    )
    assert [
        r["frame_index"] for r in rows_of(tmp_path / "auto" / "raw_conf" / "m1.jsonl")
    ] == [10, 60]
    assert (
        ball_round.main(args) == 0
        and json.loads(capsys.readouterr().out)["status"] == "already_closed"
    )

    def down(*a, **k):
        raise jobs_client.JobsError(503, "labeller down")

    monkeypatch.setattr(jobs_client, "results", down)
    assert (
        ball_round.main(
            [
                "close",
                "--job",
                "nope",
                "--pool",
                str(pool),
                "--out-root",
                str(tmp_path / "auto2"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "auto2").exists()
