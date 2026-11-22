"""Round logic: the tick rule, idempotent keys, the job body, create / plan / status and the CLI."""

from __future__ import annotations

import json

import pytest

from footy_track.ball_mining import miner
from footy_track.ball_mining import round as rnd
from footy_track.labeller import jobs, jobs_client
from footy_track.scripts import ball_round

DAY = "2026-11-18"


def listed(key, status="open", total=300, done=0):
    return {"id": key.replace("/", "-"), "key": key, "kind": "ball_check", "status": status, "n_items": total,
            "progress": {"supported": True, "total": total, "done": done, "remaining": total - done}}  # fmt: skip


def test_tick_rule_open_cards_and_daily_cap():
    assert rnd.tick_decision([], DAY)["allowed"]
    assert rnd.tick_decision([listed(f"ball-round/{DAY}/1", total=300, done=201)], DAY)[
        "allowed"
    ]  # 99 open, 300 today
    busy = rnd.tick_decision([listed(f"ball-round/{DAY}/1", total=300, done=200)], DAY)
    assert (
        not busy["allowed"]
        and busy["open_cards"] == 100
        and "open training cards" in busy["reason"]
    )
    capped = rnd.tick_decision(
        [
            listed(f"ball-round/{DAY}/1", "done", 300, 300),
            listed(f"ball-round/{DAY}/2", "done", 300, 300),
        ],
        DAY,
    )
    assert (
        not capped["allowed"]
        and capped["today_cards"] == 600
        and "today" in capped["reason"]
    )
    # yesterday's cards do not count towards today's cap; other jobs and cancelled rounds never count
    other = [
        listed("ball-round/2026-11-17/1", "done", 500, 500),
        listed("autolabel/x", total=900),
        listed(f"ball-round/{DAY}/3", "cancelled", 400),
    ]
    assert rnd.tick_decision(other, DAY)["allowed"]


def test_keys_are_per_day_and_increment():
    assert rnd.next_key([], DAY) == f"ball-round/{DAY}/1"
    js = [
        listed(f"ball-round/{DAY}/1"),
        listed(f"ball-round/{DAY}/2"),
        listed("ball-round/2026-11-17/9"),
    ]
    assert rnd.next_key(js, DAY) == f"ball-round/{DAY}/3"


@pytest.fixture
def splits_patched(monkeypatch, splitter):
    monkeypatch.setattr(miner, "default_split_of", splitter)
    monkeypatch.setattr(miner, "default_eval_clips", lambda: [])
    monkeypatch.setattr(rnd, "default_match_of", lambda c: c.split("_seg")[0])
    return splitter


class FakeClient:
    def __init__(self, jobs_=()):
        self.jobs = list(jobs_)
        self.submitted = []

    def list_jobs(self, **kw):
        return self.jobs

    def get(self, ref):
        return {
            "id": ref,
            "items": next(
                (j.get("_items", []) for j in self.jobs if j["id"] == ref), []
            ),
        }

    def submit(self, title, kind, items, **kw):
        self.submitted.append({"title": title, "kind": kind, "items": items, **kw})
        return {"id": "abc", "status": "open", "created": True}


def cfg(**kw):
    return rnd.RoundConfig(
        dataset="train", data_root="/nonexistent", use_gold=False, link_clips=False, **kw
    )


def test_job_body_is_a_valid_ball_check_job_with_the_refit_and_close_hooks(
    big_pool, splits_patched, ds
):
    res = rnd.plan_round(
        big_pool, cfg(budget=100), marks=ds, round_key=f"ball-round/{DAY}/1"
    )
    body = rnd.job_body(res, f"ball-round/{DAY}/1", cfg(budget=100))
    assert (
        body["kind"] == "ball_check"
        and body["key"] == f"ball-round/{DAY}/1"
        and len(body["items"]) == 100
    )
    assert [(h["name"], h["trigger"].get("every_n"), h["trigger"].get("on_done", False), h["action"]["type"], h["action"]["deployment"]) for h in body["hooks"]] == [
        ("refit", 50, False, "prefect", "ball-refit/ball-refit"),
        ("anchor-search", 10, False, "prefect", "ball-anchor-search/ball-anchor-search"),
        ("close", None, True, "prefect", "ball-round-close/ball-round-close")]  # fmt: skip
    job, created = jobs.submit(body, "train")  # the real validator: items, hooks, key
    again, created2 = jobs.submit(body, "train")
    assert (
        created and not created2 and again["id"] == job["id"]
    )  # idempotent on the key
    assert [h["name"] for h in job["hooks"]] == ["refit", "anchor-search", "close"]
    assert all(jobs.is_inline(it) for it in job["items"])


def test_create_submits_once_then_the_tick_rule_blocks_the_next_round(
    big_pool, splits_patched, ds
):
    client = FakeClient()
    out = rnd.create_round(big_pool, cfg(budget=60), client=client, today=DAY, marks=ds)
    assert out["created"] and out["key"] == f"ball-round/{DAY}/1"
    (sub,) = client.submitted
    assert (
        sub["key"] == f"ball-round/{DAY}/1"
        and sub["kind"] == "ball_check"
        and len(sub["items"]) == 60
    )
    assert sub["device"] == "phone"  # Ball Check cards are a phone task on the todo board
    assert [h["action"]["deployment"] for h in sub["hooks"]] == [
        "ball-refit/ball-refit",
        "ball-anchor-search/ball-anchor-search",
        "ball-round-close/ball-round-close",
    ]  # prefect actions: the labeller only triggers the flows
    log = json.loads(
        (ds / "ball_mining" / "rounds" / f"ball-round_{DAY}_1.json").read_text()
    )
    assert (
        log["report"]["inclusion_probabilities"]["R"] and log["job"] == "abc"
    )  # the exact pi are logged with the round
    # the labeller now lists the round with 60 open cards: not below 100 only once it is nearly answered
    client.jobs = [
        listed(f"ball-round/{DAY}/1", total=60, done=0) | {"_items": sub["items"]}
    ]
    assert (
        rnd.create_round(big_pool, cfg(budget=60), client=client, today=DAY, marks=ds)[
            "created"
        ]
        is True
    )  # 60 < 100 open
    client.jobs = [listed(f"ball-round/{DAY}/1", total=300, done=0)]
    blocked = rnd.create_round(
        big_pool, cfg(budget=60), client=client, today=DAY, marks=ds
    )
    assert (
        blocked["created"] is False
        and not blocked["tick"]["allowed"]
        and len(client.submitted) == 2
    )
    forced = rnd.create_round(
        big_pool, cfg(budget=60), client=client, today=DAY, marks=ds, force=True
    )
    assert forced["created"] and forced["key"] == f"ball-round/{DAY}/2"


def test_a_new_round_does_not_re_mine_frames_pending_in_an_open_round(
    big_pool, splits_patched, ds
):
    client = FakeClient()
    first = rnd.create_round(
        big_pool, cfg(budget=60), client=client, today=DAY, marks=ds
    )
    cards = client.submitted[0]["items"]
    client.jobs = [listed(f"ball-round/{DAY}/1", total=60, done=0) | {"_items": cards}]
    rnd.create_round(big_pool, cfg(budget=60), client=client, today=DAY, marks=ds)
    old = {(c["clip"], c["frame_index"] // 50) for c in cards}  # a 2 s window at 25 fps
    new = {(c["clip"], c["frame_index"] // 50) for c in client.submitted[1]["items"]}
    assert first["created"] and not (old & new)


def test_dry_run_creates_nothing(big_pool, splits_patched, ds):
    client = FakeClient()
    out = rnd.create_round(
        big_pool, cfg(budget=30), client=client, today=DAY, marks=ds, dry_run=True
    )
    assert (
        out["dry_run"] and client.submitted == [] and not (ds / "ball_mining").exists()
    )


def test_status_reports_rounds_counts_and_latest_state(ds, splits_patched):
    client = FakeClient([listed(f"ball-round/{DAY}/1", total=300, done=250)])
    st = rnd.status(cfg(), client=client, today=DAY)
    assert st["tick"]["open_cards"] == 50 and st["rounds"] == [
        {
            "key": f"ball-round/{DAY}/1",
            "id": f"ball-round-{DAY}-1",
            "status": "open",
            "cards": 300,
            "remaining": 50,
        }
    ]
    assert st["state"] is None


def _ml(path, clip, rows):
    path.mkdir(exist_ok=True)
    with (path / f"{clip}.jsonl").open("w") as fh:
        for f, c, x in rows:
            fh.write(
                json.dumps(
                    {
                        "frame_index": f,
                        "bbox": {"x": x, "y": 0.3, "w": 0.01, "h": 0.02},
                        "tags": ["in_play_ball", "rtdetr"],
                        "confidence": c,
                        "run_id": "r1",
                    }
                )
                + "\n"
            )


def test_cli_plan_prints_counts_and_writes_the_cards(tmp_path, splits_patched, capsys):
    d = tmp_path / "ml"
    for k in range(3):
        _ml(
            d,
            f"m1_seg00{k}",
            [
                (f * 40, 0.3 + 0.01 * (f % 60), 0.05 + 0.01 * (f % 80))
                for f in range(120)
            ],
        )
    out = tmp_path / "plan.json"
    rc = ball_round.main(["plan", "--machine-labels", str(d), "--match", "m1", "--dataset", "train", "--marks-dir", str(tmp_path / "marks"), "--data-root", "/nonexistent",
                          "--budget", "40", "--no-gold", "--no-link", "--out", str(out)])  # fmt: skip
    assert rc == 0
    txt = capsys.readouterr().out
    assert (
        "cards 40 / budget 40" in txt and "per match x bucket" in txt and "m1:" in txt
    )
    doc = json.loads(out.read_text())
    assert len(doc["cards"]) == 40 and doc["report"]["by_match_bucket"]["m1"]["U"] > 0
    assert {c["sampling"]["regime"] for c in doc["cards"]} == {"train"}


def test_cli_create_dry_run_and_tick_block(
    tmp_path, splits_patched, monkeypatch, capsys
):
    d = tmp_path / "ml"
    _ml(d, "m1_seg001", [(f * 40, 0.4, 0.05 + 0.01 * (f % 80)) for f in range(120)])
    monkeypatch.setattr(
        jobs_client,
        "list_jobs",
        lambda **kw: [
            listed(
                f"ball-round/{rnd.date_cls.today().isoformat()}/1", total=500, done=0
            )
        ],
    )
    monkeypatch.setattr(jobs_client, "get", lambda ref, **kw: {"id": ref, "items": []})
    submitted = []
    monkeypatch.setattr(
        jobs_client, "submit", lambda *a, **k: submitted.append(a) or {"id": "x"}
    )
    args = [
        "create",
        "--machine-labels",
        str(d),
        "--match",
        "m1",
        "--dataset",
        "train",
        "--marks-dir",
        str(tmp_path / "marks"),
        "--data-root",
        "/nonexistent",
        "--no-gold",
        "--no-link",
        "--budget",
        "20",
    ]
    assert (
        ball_round.main(args) == 0
    )  # blocked by the tick rule: nothing created, not an error
    assert '"created": false' in capsys.readouterr().out and submitted == []
    assert ball_round.main([*args, "--force", "--dry-run"]) == 0 and submitted == []
