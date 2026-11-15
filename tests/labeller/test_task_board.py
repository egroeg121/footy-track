"""Labelling home page: task board (/api/tasks), CLI, /api/server_info, page smoke."""

from __future__ import annotations

import json

import pytest

from footy_track.labeller import tasks as T
from footy_track.scripts import labelling_tasks as cli
from footy_track.scripts import warm_ball_check as W


@pytest.fixture
def tasks_file(tmp_path, monkeypatch):
    p = tmp_path / "sub" / "labelling_tasks.json"
    monkeypatch.setenv("FOOTY_TASKS_FILE", str(p))
    return p


def test_missing_file_is_seeded(client, tasks_file):
    j = client.get("/api/tasks").json()
    assert "error" not in j
    assert len(j["tasks"]) == len(T.SEED_TASKS) == 4
    assert tasks_file.exists()  # seeded once
    assert all(t["id"] and t["title"] and t["url"] for t in j["tasks"])
    assert {t["device"] for t in j["tasks"]} <= set(T.DEVICES)


def test_existing_file_is_not_reseeded_and_reread_each_request(client, tasks_file):
    tasks_file.parent.mkdir(parents=True)
    tasks_file.write_text(json.dumps({"tasks": [{"id": "a", "title": "A", "priority": 3}]}))
    assert [t["id"] for t in client.get("/api/tasks").json()["tasks"]] == ["a"]
    tasks_file.write_text(json.dumps({"tasks": [{"id": "b", "title": "B"}, {"id": "c", "title": "C", "priority": 1}]}))
    assert [t["id"] for t in client.get("/api/tasks").json()["tasks"]] == ["c", "b"]  # priority order


@pytest.mark.parametrize("body", ["{not json", '{"tasks": "nope"}', "[1, 2"])
def test_malformed_file_returns_200_with_error(client, tasks_file, body):
    tasks_file.parent.mkdir(parents=True)
    tasks_file.write_text(body)
    r = client.get("/api/tasks")
    assert r.status_code == 200
    j = r.json()
    assert j["tasks"] == [] and j["error"]
    assert tasks_file.read_text() == body  # never overwritten


def test_cli_add_done_roundtrip(client, tasks_file, capsys):
    assert cli.main(["add", "Check X", "--url", "/ball_check", "--why", "because", "--device", "phone",
                     "--priority", "0", "--count-url", "/api/x", "--count-label", "left", "--owner", "me"]) == 0
    tid = capsys.readouterr().out.strip()
    assert tid.startswith("t-")
    tasks = client.get("/api/tasks").json()["tasks"]
    mine = next(t for t in tasks if t["id"] == tid)
    assert mine["title"] == "Check X" and mine["device"] == "phone" and mine["owner"] == "me"
    assert mine["count_url"] == "/api/x" and mine["added"]
    assert tasks[0]["id"] == tid  # lowest priority number sorts first

    assert cli.main(["done", tid]) == 0
    assert tid not in {t["id"] for t in client.get("/api/tasks").json()["tasks"]}
    assert cli.main(["done", tid]) == 1  # already gone
    assert cli.main(["list"]) == 0
    assert not list(tasks_file.parent.glob("*.tmp"))  # atomic write left no temp files


def test_cli_ids_are_unique_when_added_in_the_same_second(tasks_file):
    ids = []
    for _ in range(3):
        tasks, _ = T.load_tasks()
        t = {"id": T.new_task_id(tasks), "title": "x"}
        T.write_tasks_atomic(T.tasks_path(), [*tasks, t])
        ids.append(t["id"])
    assert len(set(ids)) == 3


def test_cli_refuses_to_edit_malformed_file(tasks_file):
    tasks_file.parent.mkdir(parents=True)
    tasks_file.write_text("{broken")
    with pytest.raises(SystemExit):
        cli.main(["add", "T"])
    assert tasks_file.read_text() == "{broken"


def test_server_info_shape(client, monkeypatch):
    monkeypatch.setenv("FOOTY_SERVICE_NAME", "calib")
    monkeypatch.setenv("FOOTY_SERVICE_PORT", "8011")
    monkeypatch.setenv("FOOTY_GT_MARKS_DIR", "/x/gt")
    r = client.get("/api/server_info")
    assert r.status_code == 200
    j = r.json()
    assert j["name"] == "calib" and j["port"] == 8011
    assert {"sha", "short_sha", "branch", "describe", "dirty", "started", "uptime_s", "env", "repo"} <= set(j)
    assert j["uptime_s"] >= 0
    assert j["env"]["FOOTY_GT_MARKS_DIR"] == "/x/gt"
    assert r.headers["access-control-allow-origin"] == "*"  # cross-origin reads from the other server's home page


def test_home_page_smoke(client):
    for route in ("/", "/main"):
        r = client.get(route)
        assert r.status_code == 200
        assert "no-store" in r.headers["cache-control"]
        for needle in ("To do now", "/api/tasks", "/api/server_info", "/ball_check", "/calibration/review"):
            assert needle in r.text


def test_warm_helper_urls_and_walks_only_undone_cards(monkeypatch):
    assert W.card_urls({"clip": "c 1", "frame_index": 7, "box_index": 3}) == [
        "/ball_check/crop/c%201/7/3.jpg?src=list&pad=6&reticle=false",
        "/ball_check/clip/c%201/7/3.mp4?pad=6&src=list",
    ]
    seen: list[str] = []

    def fake_get(url, timeout):
        if "/ball_check/queue" in url:
            return json.dumps({"items": [{"clip": "a", "frame_index": 1, "box_index": 4},
                                         {"clip": "b", "frame_index": 2, "box_index": 5}]}).encode()
        seen.append(url)
        return b""

    monkeypatch.setattr(W, "_get", fake_get)
    assert W.main(["--base", "http://x:1", "--sleep", "0"]) == 0
    assert [u.split("?")[0] for u in seen] == [
        "http://x:1/ball_check/crop/a/1/4.jpg", "http://x:1/ball_check/clip/a/1/4.mp4",
        "http://x:1/ball_check/crop/b/2/5.jpg", "http://x:1/ball_check/clip/b/2/5.mp4",
    ]
