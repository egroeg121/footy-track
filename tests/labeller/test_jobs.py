"""Labelling jobs API: submit, list, fetch by id/title, idempotency, computed progress,
results filtered to the job's items, task-board visibility, and the stdlib client.

Fixtures ``two`` / ``web`` (conftest) give two datasets (calib default, nottinghamforest) in tmp dirs.
"""

from __future__ import annotations

import ast
import json
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from footy_track.labeller import datasets, jobs, jobs_client, server
from footy_track.scripts import labelling_jobs as cli

CLIP = "arsenal_nottinghamforest_seg003"


def _row(frame: int, tags: list[str], x: float = 0.5) -> str:
    box = tags[0] not in ("no_ball", "not_broadcast")
    return json.dumps(
        {
            "frame_index": frame,
            "bbox": {"x": x, "y": 0.4, "w": 0.01, "h": 0.02} if box else None,
            "center": {"x": x + 0.005, "y": 0.41} if box else None,
            "tags": tags,
        }
    )


def _sidecar(two, ds: str, clip: str, rows: list[str]) -> None:
    (two[ds] / "marks" / f"{clip}.jsonl").write_text("\n".join(rows) + "\n")


def _submit(web, **over):
    body = {
        "title": "Nottingham Forest windows",
        "kind": "ball_place",
        "dataset": "nottinghamforest",
        "items": [{"clip": CLIP, "frame_start": 10, "frame_end": 19}],
        **over,
    }
    return web.post("/api/jobs", json=body)


# ---------------------------------------------------------------------------
# Submit / list / fetch
# ---------------------------------------------------------------------------


def test_submit_returns_id_url_status_and_is_stored_one_file_per_job(web, tmp_path):
    r = _submit(web, why="held-out windows", priority=1, device="laptop", requester={"branch": "exp/x", "commit": "abc"})
    assert r.status_code == 201
    j = r.json()
    assert j["status"] == "open" and j["created"] is True and j["id"].startswith("j-")
    assert j["url"].startswith("http://testserver/labeller?")
    assert "dataset=nottinghamforest" in j["url"] and f"clip={CLIP}" in j["url"] and "frame=10" in j["url"]
    files = list((tmp_path / "labelling_jobs").glob("*.json"))
    assert [f.stem for f in files] == [j["id"]]
    stored = json.loads(files[0].read_text())
    assert stored["requester"] == {"branch": "exp/x", "commit": "abc"} and stored["device"] == "laptop"
    assert not list((tmp_path / "labelling_jobs").glob("*.tmp"))


def test_dataset_defaults_to_the_request_dataset(web):
    assert web.post("/api/jobs", json={"title": "a", "kind": "other"}).json()["dataset"] == "calib"
    assert web.post("/api/jobs?dataset=nottinghamforest", json={"title": "b", "kind": "other"}).json()["dataset"] == "nottinghamforest"


def test_get_by_id_and_by_exact_title_and_listing_filters(web):
    a = _submit(web, title="Nottingham Forest windows", requester={"branch": "exp/a"}).json()
    b = _submit(web, title="Calib recheck", dataset="calib", kind="calibration",
                items=[{"clip": "x", "frame": 3}], requester={"branch": "exp/b"}).json()  # fmt: skip
    assert web.get(f"/api/jobs/{a['id']}").json()["title"] == "Nottingham Forest windows"
    by_title = web.get("/api/jobs/Nottingham Forest windows").json()
    assert by_title["id"] == a["id"]
    assert web.get("/api/jobs/Calib%20recheck").json()["id"] == b["id"]
    assert web.get("/api/jobs/nope").status_code == 404

    ids = lambda **q: sorted(j["id"] for j in web.get("/api/jobs", params=q).json()["jobs"])  # noqa: E731
    assert ids() == sorted([a["id"], b["id"]])
    assert ids(title="Nottingham Forest windows") == [a["id"]]  # exact
    assert ids(title="recheck") == [b["id"]]  # substring, case-insensitive
    assert ids(kind="calibration") == [b["id"]]
    assert ids(dataset="nottinghamforest") == [a["id"]]
    assert ids(**{"requester.branch": "exp/b"}) == [b["id"]]
    assert ids(status="open") == sorted([a["id"], b["id"]]) and ids(status="done") == []


def test_title_with_a_slash_and_an_ambiguous_title(web):
    j = _submit(web, title="nottinghamforest/windows r1").json()
    assert web.get("/api/jobs/nottinghamforest%2Fwindows%20r1").json()["id"] == j["id"]
    assert web.get("/api/jobs/nottinghamforest%2Fwindows%20r1/results").json()["job"]["id"] == j["id"]
    j2 = _submit(web, title="nottinghamforest/windows r1").json()
    r = web.get("/api/jobs/nottinghamforest%2Fwindows%20r1")
    assert r.status_code == 409 and sorted(r.json()["ids"]) == sorted([j["id"], j2["id"]])
    web.post(f"/api/jobs/{j['id']}/cancel")  # a cancelled duplicate no longer makes the title ambiguous
    assert web.get("/api/jobs/nottinghamforest%2Fwindows%20r1").json()["id"] == j2["id"]


@pytest.mark.parametrize(
    "over, text",
    [
        ({"title": ""}, "title"),
        ({"kind": "nope"}, "kind"),
        ({"dataset": "nope"}, "unknown dataset"),
        ({"device": "tv"}, "device"),
        ({"items": []}, "at least one item"),
        ({"items": [{"clip": CLIP, "frame_start": 9, "frame_end": 3}]}, "frame_end"),
        ({"items": [{"clip": "../etc", "frame": 1}]}, "clip"),
        ({"items": [{"clip": CLIP}]}, "frame"),
        ({"items": "x"}, "list of objects"),
    ],
)
def test_validation_errors_are_422(web, over, text):
    r = _submit(web, **over)
    assert r.status_code == 422 and text in r.json()["detail"]


def test_title_is_required_and_other_kind_needs_no_items(web):
    assert web.post("/api/jobs", json={"kind": "other"}).status_code == 422
    assert web.post("/api/jobs", json={"title": "free form"}).status_code == 201


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_same_key_returns_the_existing_job_not_a_duplicate(web, tmp_path):
    a = _submit(web, key="exp/x/nottinghamforest-r1")
    b = _submit(web, key="exp/x/nottinghamforest-r1", title="a different title is ignored")
    assert (a.status_code, b.status_code) == (201, 200)
    assert a.json()["id"] == b.json()["id"] and b.json()["created"] is False
    assert b.json()["title"] == "Nottingham Forest windows"
    assert len(list((tmp_path / "labelling_jobs").glob("*.json"))) == 1
    c = _submit(web, key="exp/x/nottinghamforest-r2")
    assert c.json()["id"] != a.json()["id"]
    assert len(web.get("/api/jobs").json()["jobs"]) == 2


def test_same_key_concurrent_submits_create_one_job(two, tmp_path):
    body = {"title": "t", "kind": "other", "key": "k"}
    out: list = []

    def go():
        out.append(TestClient(server.app).post("/api/jobs", json=body).json()["id"])

    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(out)) == 1 and len(list((tmp_path / "labelling_jobs").glob("*.json"))) == 1


# ---------------------------------------------------------------------------
# Progress from the real label store
# ---------------------------------------------------------------------------


def test_progress_counts_only_human_labels_in_the_jobs_frames(web, two):
    _sidecar(
        two, "nottinghamforest", CLIP,
        [
            _row(10, ["in_play_ball", "labeller"]),  # human: counts
            _row(11, ["in_play_ball", "vittrack"]),  # machine only (tier 3): does not
            _row(12, ["in_play_ball", "vittrack", "human_checked"]),  # reviewed machine box: counts
            _row(13, ["no_ball"]),  # human "no ball here": counts
            _row(14, ["not_broadcast"]),  # counts
            _row(15, ["player", "labeller"]),  # a player label is not a ball label
            _row(99, ["in_play_ball", "labeller"]),  # outside the window
        ],
    )  # fmt: skip
    j = _submit(web).json()
    p = web.get(f"/api/jobs/{j['id']}").json()["progress"]
    assert (p["done"], p["total"], p["remaining"]) == (4, 10, 6)
    assert p["fraction"] == 0.4 and p["supported"] is True
    assert p["next"] == {"clip": CLIP, "frame": 11}  # first unlabelled frame
    assert web.get(f"/api/jobs/{j['id']}").json()["status"] == "in_progress"


def test_progress_follows_the_label_files_not_a_counter(web, two):
    j = _submit(web).json()
    assert web.get(f"/api/jobs/{j['id']}").json()["progress"]["done"] == 0
    _sidecar(two, "nottinghamforest", CLIP, [_row(10, ["in_play_ball", "labeller"])])
    assert web.get(f"/api/jobs/{j['id']}").json()["progress"]["done"] == 1
    _sidecar(two, "nottinghamforest", CLIP, [_row(10, ["in_play_ball", "labeller"]), _row(11, ["no_ball"])])
    assert web.get(f"/api/jobs/{j['id']}").json()["progress"]["done"] == 2


def test_window_step_and_start_end_aliases_and_extra_columns(web, two):
    items = [{"clip": CLIP, "start_frame": 10, "end_frame": 20, "step": 5, "stratum": "easy"}]
    j = _submit(web, items=items).json()
    got = web.get(f"/api/jobs/{j['id']}").json()
    assert got["items"][0]["frame_start"] == 10 and got["items"][0]["stratum"] == "easy"
    assert got["progress"]["total"] == 3  # frames 10, 15, 20 (both ends inclusive)


def test_labels_in_another_dataset_do_not_count(web, two):
    _sidecar(two, "calib", CLIP, [_row(f, ["in_play_ball", "labeller"]) for f in range(10, 20)])
    j = _submit(web).json()  # dataset nottinghamforest
    assert web.get(f"/api/jobs/{j['id']}").json()["progress"]["done"] == 0
    c = _submit(web, title="same in calib", dataset="calib").json()
    got = web.get(f"/api/jobs/{c['id']}").json()
    assert got["progress"]["done"] == 10 and got["status"] == "done"


def test_alias_sidecar_stem_is_found(web, two):
    """A clip whose GT lives under an alias stem (canonical clip id) still counts."""
    _sidecar(two, "nottinghamforest", "arsenal_mancity_seg010", [_row(5, ["in_play_ball", "labeller"])])
    j = _submit(web, kind="ball_place", items=[{"clip": "arsenal_mancity_20250925_seg010", "frame": 5}]).json()
    assert web.get(f"/api/jobs/{j['id']}").json()["progress"]["done"] == 1


def test_auto_done_at_full_progress_and_not_reopened_or_cancel_overridden(web, two):
    _sidecar(two, "nottinghamforest", CLIP, [_row(f, ["in_play_ball", "labeller"]) for f in range(10, 20)])
    j = _submit(web).json()
    got = web.get(f"/api/jobs/{j['id']}").json()
    assert got["status"] == "done" and got["progress"]["remaining"] == 0 and got["closed_by"] == "auto"
    assert web.get("/api/jobs", params={"status": "done"}).json()["jobs"][0]["id"] == j["id"]
    # a cancelled job stays cancelled even when its labels exist
    c = _submit(web, title="cancelled one").json()
    web.post(f"/api/jobs/{c['id']}/cancel")
    assert web.get(f"/api/jobs/{c['id']}").json()["status"] == "cancelled"


def test_done_cancel_reopen_endpoints(web):
    j = _submit(web).json()
    assert web.post(f"/api/jobs/{j['id']}/done").json()["status"] == "done"
    assert web.get("/api/jobs/Nottingham Forest windows").json()["status"] == "done"
    assert web.post(f"/api/jobs/{j['id']}/reopen").json()["status"] == "open"
    r = web.post("/api/jobs/Nottingham Forest windows/cancel").json()
    assert r["status"] == "cancelled" and r["closed_by"] == "api"
    assert web.post("/api/jobs/nope/done").status_code == 404


def test_other_kind_has_no_computed_progress(web):
    j = web.post("/api/jobs", json={"title": "free", "kind": "other"}).json()
    got = web.get(f"/api/jobs/{j['id']}").json()
    assert got["progress"]["supported"] is False and got["status"] == "open"
    res = web.get(f"/api/jobs/{j['id']}/results").json()
    assert res["labels"] == []


def test_ball_check_progress_and_results_from_the_verdict_log(web, two):
    vdir = two["nottinghamforest"] / "marks" / "ball_checks"
    vdir.mkdir()
    rec = lambda f, b, v, **kw: json.dumps(  # noqa: E731
        {"clip": CLIP, "frame_index": f, "box_index": b, "source": "list", "verdict": v,
         "bbox": {"x": 0.1, "y": 0.2, "w": 0.01, "h": 0.01}, "label": "in_play_ball",
         "provenance": "rtdetr", "confidence": None, "corrected_bbox": None, "ts": 1.5, **kw}
    )  # fmt: skip
    (vdir / f"{CLIP}.jsonl").write_text(
        "\n".join(
            [
                rec(4, 0, "ball"),
                rec(5, 1, "not_ball"),
                rec(5, 1, None),  # undo tombstone: card 5/1 is open again
                rec(9, 2, "box_off"),  # not one of the job's cards
            ]
        )
        + "\n"
    )
    cards = [{"clip": CLIP, "frame_index": 4, "box_index": 0, "source": "list"},
             {"clip": CLIP, "frame_index": 5, "box_index": 1, "source": "list"}]  # fmt: skip
    j = _submit(web, kind="ball_check", items=cards).json()
    assert "/ball_check?" in j["url"] and "source=list" in j["url"] and f"clip={CLIP}" in j["url"]
    got = web.get(f"/api/jobs/{j['id']}").json()
    assert (got["progress"]["done"], got["progress"]["total"]) == (1, 2)
    res = web.get(f"/api/jobs/{j['id']}/results").json()
    assert [(r["frame_index"], r["box_index"], r["verdict"]) for r in res["labels"]] == [(4, 0, "ball")]
    assert res["labels"][0]["provenance"] == "rtdetr" and res["labels"][0]["ts"] == 1.5


def test_calibration_and_frame_type_progress(web, two):
    marks = two["nottinghamforest"] / "marks"
    (marks / "calibration_gt").mkdir()
    (marks / "frame_type").mkdir()
    (marks / "calibration_gt" / f"{CLIP}.json").write_text(
        json.dumps({"clip": CLIP, "frames": {"7": {"frame_index": 7, "H": [[1, 0, 0]] * 3, "method": "dlt",
                                                      "ts": "2026-11-14T00:00:00+0000", "origin": {"source": "human"}}}})
    )  # fmt: skip
    (marks / "frame_type" / f"{CLIP}.json").write_text(
        json.dumps({"clip": CLIP, "frames": {"8": {"type": "not_broadcast", "ts": "2026-11-14T01:00:00+0000"}}})
    )
    items = [{"clip": CLIP, "frame": 7}, {"clip": CLIP, "frame": 8}, {"clip": CLIP, "frame": 9}]
    cal = _submit(web, kind="calibration", items=items, title="cal").json()
    got = web.get(f"/api/jobs/{cal['id']}").json()
    assert (got["progress"]["done"], got["progress"]["total"]) == (2, 3)  # calibrated + not-broadcast
    assert got["progress"]["next"] == {"clip": CLIP, "frame": 9} and "/calibration?" in cal["url"]
    rows = {r["frame_index"]: r for r in web.get(f"/api/jobs/{cal['id']}/results").json()["labels"]}
    assert rows[7]["status"] == "calibrated" and rows[7]["method"] == "dlt" and rows[7]["origin"]["source"] == "human"
    assert rows[8]["status"] == "not_broadcast"
    ft = _submit(web, kind="frame_type", items=items, title="ft").json()
    assert web.get(f"/api/jobs/{ft['id']}").json()["progress"]["done"] == 1  # only frame 8 is flagged


def test_identity_progress(web, two):
    d = two["nottinghamforest"].parent / "nottinghamforest" / "marks" / "identity_labels"
    d.mkdir()
    ref = lambda c, t: {"clip": c, "track_id": t, "source": "bytetrack", "run_id": None}  # noqa: E731
    (d / "identity_pairs.jsonl").write_text(
        json.dumps({"kind": "pair", "a": ref(CLIP, 1), "b": ref("other_clip", 2), "verdict": "same", "ts": 3.0}) + "\n"
    )
    j = _submit(web, kind="identity", items=[{"clip": CLIP}, {"clip": "untouched"}]).json()
    got = web.get(f"/api/jobs/{j['id']}").json()
    assert (got["progress"]["done"], got["progress"]["total"]) == (1, 2)
    labels = web.get(f"/api/jobs/{j['id']}/results").json()["labels"]
    assert len(labels) == 1 and labels[0]["verdict"] == "same" and labels[0]["clip"] == CLIP


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_results_contain_exactly_the_jobs_labels_with_provenance(web, two):
    _sidecar(
        two, "nottinghamforest", CLIP,
        [
            _row(9, ["in_play_ball", "labeller"]),  # before the window
            _row(10, ["in_play_ball", "labeller"], x=0.25),
            _row(11, ["in_play_ball", "vittrack"]),  # machine only
            _row(12, ["in_play_ball", "vittrack", "human_checked"], x=0.31),
            _row(13, ["no_ball"]),
            _row(14, ["player", "labeller"]),
            _row(20, ["in_play_ball", "labeller"]),  # after the window
        ],
    )  # fmt: skip
    _sidecar(two, "nottinghamforest", "other_clip", [_row(10, ["in_play_ball", "labeller"])])
    j = _submit(web).json()
    res = web.get(f"/api/jobs/{j['id']}/results").json()
    assert res["schema"] == "labelling-job-results/v1"
    assert res["job"]["id"] == j["id"] and res["job"]["dataset"] == "nottinghamforest"
    assert res["progress"]["done"] == 3 and res["progress"]["total"] == 10
    rows = {r["frame_index"]: r for r in res["labels"]}
    assert sorted(rows) == [10, 12, 13] and {r["clip"] for r in res["labels"]} == {CLIP}
    assert rows[10]["status"] == "ball" and rows[10]["provenance"] == "labeller" and rows[10]["tier"] == 1
    assert rows[10]["bbox"]["x"] == 0.25 and rows[10]["label"] == "in_play_ball"
    assert rows[12]["provenance"] == "machine+human_checked" and rows[12]["tier"] == 2
    assert rows[13]["status"] == "no_ball" and "bbox" not in rows[13]
    store = res["stores"][0]
    assert store["clip"] == CLIP and store["store"].endswith(f"{CLIP}.jsonl") and store["store_mtime"]
    assert res["items"][0]["done"] == 3 and res["items"][0]["total"] == 10
    # by title works too, and nothing was written to the label store
    assert web.get("/api/jobs/Nottingham Forest windows/results").json()["labels"] == res["labels"]


# ---------------------------------------------------------------------------
# Task board
# ---------------------------------------------------------------------------


def test_open_jobs_appear_on_the_task_board_and_manual_tasks_still_work(web, two, tmp_path, monkeypatch):
    tasks_file = tmp_path / "t.json"
    tasks_file.write_text(json.dumps({"tasks": [{"id": "manual", "title": "A manual task", "priority": 2}]}))
    monkeypatch.setenv("FOOTY_TASKS_FILE", str(tasks_file))
    assert [t["id"] for t in web.get("/api/tasks").json()["tasks"]] == ["manual"]

    j = _submit(web, priority=1, why="held out", requester={"branch": "exp/x"}).json()
    got = web.get("/api/tasks").json()
    assert [t["id"] for t in got["tasks"]] == [j["id"], "manual"]  # priority 1 sorts first
    t = got["tasks"][0]
    assert t["job"] is True and t["title"] == "Nottingham Forest windows" and t["owner"] == "exp/x"
    assert t["url"].startswith("/labeller?") and "dataset=nottinghamforest" in t["url"] and f"clip={CLIP}" in t["url"]
    assert t["count_url"] == f"/api/jobs/{j['id']}" and t["count_path"] == "progress.remaining"
    assert web.get(t["count_url"]).json()["progress"]["remaining"] == 10  # what the board's chip fetches
    assert tasks_file.read_text() == json.dumps({"tasks": [{"id": "manual", "title": "A manual task", "priority": 2}]})

    web.post(f"/api/jobs/{j['id']}/done")  # closed jobs leave the board
    assert [t["id"] for t in web.get("/api/tasks").json()["tasks"]] == ["manual"]


def test_board_link_points_at_the_next_unlabelled_frame(web, two):
    _sidecar(two, "nottinghamforest", CLIP, [_row(f, ["in_play_ball", "labeller"]) for f in (10, 11, 12)])
    j = _submit(web).json()
    t = next(t for t in web.get("/api/tasks").json()["tasks"] if t["id"] == j["id"])
    assert "frame=13" in t["url"]


def test_a_broken_job_file_does_not_break_the_board_or_listing(web, tmp_path):
    _submit(web)
    (tmp_path / "labelling_jobs" / "junk.json").write_text("{not json")
    assert len(web.get("/api/tasks").json()["tasks"]) >= 1
    assert len(web.get("/api/jobs").json()["jobs"]) == 1


def test_main_page_renders_job_chip_code(web):
    assert "t.job" in web.get("/").text


# ---------------------------------------------------------------------------
# Client (stdlib urllib) against a real local uvicorn
# ---------------------------------------------------------------------------


@pytest.fixture
def live(two):
    cfg = uvicorn.Config(server.app, host="127.0.0.1", port=0, log_level="error")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    port = srv.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=10)


def test_client_submit_get_wait_results_roundtrip(live, two, monkeypatch):
    monkeypatch.setenv("FOOTY_LABELLER_URL", live)
    job = jobs_client.submit(
        "client job", "ball_place", [{"clip": CLIP, "frame_start": 10, "frame_end": 12}],
        dataset="nottinghamforest", key="client/1", why="from a test",
    )  # fmt: skip
    assert job["status"] == "open" and job["created"] and job["url"].startswith(live)
    again = jobs_client.submit("ignored", "ball_place", [{"clip": "x", "frame": 1}], dataset="nottinghamforest", key="client/1")
    assert again["id"] == job["id"] and again["created"] is False

    # requester auto-filled from git (this checkout), host from the hostname
    req = jobs_client.get(job["id"])["requester"]
    assert req["branch"] and req["commit"] and req["host"]
    assert jobs_client.get("client job")["id"] == job["id"]
    assert [j["id"] for j in jobs_client.list_jobs(kind="ball_place", dataset="nottinghamforest")] == [job["id"]]

    with pytest.raises(TimeoutError):
        jobs_client.wait(job["id"], poll=0.05, timeout=0.2)

    def label_later():
        time.sleep(0.3)
        _sidecar(two, "nottinghamforest", CLIP, [_row(f, ["in_play_ball", "labeller"]) for f in (10, 11, 12)])

    threading.Thread(target=label_later).start()
    seen = []
    final = jobs_client.wait(job["id"], poll=0.1, timeout=20, on_poll=lambda j: seen.append(j["progress"]["done"]))
    assert final["status"] == "done" and seen[-1] == 3
    res = jobs_client.results(job["id"])
    assert [r["frame_index"] for r in res["labels"]] == [10, 11, 12]


def test_client_errors_are_typed(live, monkeypatch):
    monkeypatch.setenv("FOOTY_LABELLER_URL", live)
    with pytest.raises(jobs_client.JobsError) as e:
        jobs_client.get("no such job")
    assert e.value.status == 404
    with pytest.raises(jobs_client.JobsError) as e:
        jobs_client.submit("bad", "ball_place", [])
    assert e.value.status == 422 and "at least one item" in str(e.value)


def test_cli_submits_a_csv_of_windows_and_reports_status(live, two, tmp_path, capsys):
    csv_path = tmp_path / "windows.csv"
    csv_path.write_text(
        "clip,start_frame,end_frame,stratum,reason\n"
        f'{CLIP},10,259,easy,"B1 covers 100% of frames, mean p 0.96"\n'
        "arsenal_nottinghamforest_seg022,10,259,easy,x\n"
    )
    rc = cli.main(["--url", live, "submit", "--title", "Nottingham Forest windows r1", "--kind", "ball_place",
                   "--dataset", "nottinghamforest", "--items", str(csv_path), "--step", "5", "--key", "cli/nottinghamforest-1"])  # fmt: skip
    assert rc == 0
    job = json.loads(capsys.readouterr().out)
    got = jobs_client.get(job["id"], url=live)
    assert got["items"][0] == {"clip": CLIP, "frame_start": 10, "frame_end": 259, "step": 5, "stratum": "easy",
                               "reason": "B1 covers 100% of frames, mean p 0.96"}  # fmt: skip
    assert got["progress"]["total"] == 2 * 50  # 10..259 step 5 = 50 frames per window
    assert got["requester"]["branch"]
    assert cli.main(["--url", live, "status", "Nottingham Forest windows r1"]) == 0
    assert "0/100" in capsys.readouterr().out
    out = tmp_path / "labels.json"
    assert cli.main(["--url", live, "results", "Nottingham Forest windows r1", "-o", str(out)]) == 0
    assert json.loads(out.read_text())["labels"] == []
    assert cli.main(["--url", live, "cancel", "Nottingham Forest windows r1"]) == 0
    assert cli.main(["--url", live, "wait", "Nottingham Forest windows r1", "--poll", "0.05"]) == 1  # cancelled, not done


def test_client_is_a_standalone_stdlib_only_file_and_served_by_the_server(web):
    src = Path(jobs_client.__file__).read_text()
    stdlib = set(__import__("sys").stdlib_module_names)
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    assert mods <= stdlib, mods - stdlib
    assert web.get("/api/jobs_client.py").text == src


def test_jobs_dir_and_registry_untouched_when_nothing_is_submitted(web, tmp_path):
    web.get("/api/jobs")
    web.get("/api/tasks")
    assert not (tmp_path / "labelling_jobs").exists()  # created lazily, on first submit
    assert datasets.registry_path().exists()  # (the test registry, written by the fixture)
    assert jobs.jobs_dir() == tmp_path / "labelling_jobs"
