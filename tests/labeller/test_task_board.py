"""Labelling home page: task board (/api/tasks), CLI, /api/server_info, page smoke."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from footy_track.calibration.frame_type import update_frame_types
from footy_track.calibration.prelabel.proposals import write_proposals
from footy_track.labeller import calibration as cal
from footy_track.labeller import task_done as TD
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
    tasks_file.write_text(
        json.dumps({"tasks": [{"id": "a", "title": "A", "priority": 3}]})
    )
    assert [t["id"] for t in client.get("/api/tasks").json()["tasks"]] == ["a"]
    tasks_file.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "b", "title": "B"},
                    {"id": "c", "title": "C", "priority": 1},
                ]
            }
        )
    )
    assert [t["id"] for t in client.get("/api/tasks").json()["tasks"]] == [
        "c",
        "b",
    ]  # priority order


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
    assert (
        cli.main(
            [
                "add",
                "Check X",
                "--url",
                "/ball_check",
                "--why",
                "because",
                "--device",
                "phone",
                "--priority",
                "0",
                "--count-url",
                "/api/x",
                "--count-label",
                "left",
                "--owner",
                "me",
            ]
        )
        == 0
    )
    tid = capsys.readouterr().out.strip()
    assert tid.startswith("t-")
    tasks = client.get("/api/tasks").json()["tasks"]
    mine = next(t for t in tasks if t["id"] == tid)
    assert (
        mine["title"] == "Check X"
        and mine["device"] == "phone"
        and mine["owner"] == "me"
    )
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
    assert {
        "sha",
        "short_sha",
        "branch",
        "describe",
        "dirty",
        "started",
        "uptime_s",
        "env",
        "repo",
    } <= set(j)
    assert j["uptime_s"] >= 0
    assert j["env"]["FOOTY_GT_MARKS_DIR"] == "/x/gt"
    assert (
        r.headers["access-control-allow-origin"] == "*"
    )  # cross-origin reads from the other server's home page


def test_home_page_smoke(client):
    for route in ("/", "/main"):
        r = client.get(route)
        assert r.status_code == 200
        assert "no-store" in r.headers["cache-control"]
        for needle in (
            "To do now",
            "/api/tasks",
            "/api/server_info",
            "/ball_check",
            "/calibration/review",
        ):
            assert needle in r.text


def test_warm_helper_urls_and_walks_only_undone_cards(monkeypatch):
    assert W.card_urls({"clip": "c 1", "frame_index": 7, "box_index": 3}) == [
        "/ball_check/crop/c%201/7/3.jpg?src=list&pad=6&reticle=false",
        "/ball_check/clip/c%201/7/3.mp4?pad=6&src=list",
    ]
    seen: list[str] = []

    def fake_get(url, timeout):
        if "/ball_check/queue" in url:
            return json.dumps(
                {
                    "items": [
                        {"clip": "a", "frame_index": 1, "box_index": 4},
                        {"clip": "b", "frame_index": 2, "box_index": 5},
                    ]
                }
            ).encode()
        seen.append(url)
        return b""

    monkeypatch.setattr(W, "_get", fake_get)
    assert W.main(["--base", "http://x:1", "--sleep", "0"]) == 0
    assert [u.split("?")[0] for u in seen] == [
        "http://x:1/ball_check/crop/a/1/4.jpg",
        "http://x:1/ball_check/clip/a/1/4.mp4",
        "http://x:1/ball_check/crop/b/2/5.jpg",
        "http://x:1/ball_check/clip/b/2/5.mp4",
    ]


# ------------------------------------------------------------------ self-completing tasks (done_when)

CLIP = "arsenal_bournemouth_1st_seg030"
ALIAS = "bournemouth_broadcast_seg030"  # the same clip: canonical_clip_id resolves both to CLIP


@pytest.fixture(autouse=True)
def _no_done_cache(monkeypatch):
    TD._cache.clear()
    monkeypatch.setattr(TD, "CACHE_TTL", 0.0)
    yield
    TD._cache.clear()


def _post(tasks_file, *tasks):
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text(json.dumps({"tasks": list(tasks)}))


def _board(client):
    j = client.get("/api/tasks").json()
    assert "error" not in j
    return {t["id"]: t for t in j["tasks"]}, [t["id"] for t in j["tasks"]]


def _save_gt(clip, frame, ts="2026-11-16T10:00:00+0000"):
    doc = cal._read_gt(clip)
    doc["frames"][str(frame)] = {"frame_index": frame, "ts": ts}
    cal._write_gt(clip, doc)


def test_task_without_done_when_keeps_todays_behaviour(
    client, tasks_file, gt_marks_dir
):
    _post(tasks_file, {"id": "a", "title": "A", "url": "/ball_check"})
    by_id, _ = _board(client)
    assert by_id["a"]["done"] is False
    assert "done_when" not in by_id["a"] and "progress" not in by_id["a"]


def test_gt_done_when_resolves_clip_aliases(client, tasks_file, gt_marks_dir):
    _post(
        tasks_file,
        {"id": "a", "title": "A", "done_when": {"gt": {"clip": CLIP, "frame": 205}}},
    )
    assert _board(client)[0]["a"]["done"] is False
    _save_gt(ALIAS, 205)  # saved under another alias of the clip
    assert _board(client)[0]["a"]["done"] is True
    _post(
        tasks_file,
        {"id": "b", "title": "B", "done_when": {"gt": {"clip": ALIAS, "frame": 205}}},
        {"id": "c", "title": "C", "done_when": {"gt": {"clip": CLIP, "frame": 206}}},
    )
    by_id, _ = _board(client)
    assert (
        by_id["b"]["done"] is True and by_id["c"]["done"] is False
    )  # other frame: untouched


def test_proposal_done_when_status_in(client, tasks_file, gt_marks_dir):
    rec = lambda st: {"runs": {"r1": {"frames": {"205": {"status": st}}}}}  # noqa: E731
    dw = {"proposal": {"clip": ALIAS, "run": "r1", "frame": 205}}
    _post(
        tasks_file,
        {"id": "a", "title": "A", "done_when": dw},
        {"id": "b", "title": "B", "done_when": {**dw, "status_in": ["rejected"]}},
    )
    write_proposals(gt_marks_dir, CLIP, {"clip": CLIP, **rec("proposed")})
    by_id, _ = _board(client)
    assert not by_id["a"]["done"] and not by_id["b"]["done"]
    write_proposals(gt_marks_dir, CLIP, {"clip": CLIP, **rec("edited")})
    by_id, _ = _board(client)
    assert (
        by_id["a"]["done"] is True and by_id["b"]["done"] is False
    )  # default: accepted/edited
    write_proposals(gt_marks_dir, CLIP, {"clip": CLIP, **rec("rejected")})
    by_id, _ = _board(client)
    assert by_id["a"]["done"] is False and by_id["b"]["done"] is True


def test_frames_done_when_counts_gt_and_frame_type_marks_with_progress(
    client, tasks_file, gt_marks_dir
):
    frames = [[CLIP, 1], [CLIP, 2], [CLIP, 3], ["arsenal_mancity_seg010", 4]]
    _post(tasks_file, {"id": "a", "title": "A", "done_when": {"frames": frames}})
    t = _board(client)[0]["a"]
    assert t["done"] is False and t["progress"] == {"done": 0, "total": 4}
    _save_gt(ALIAS, 1)
    update_frame_types(
        gt_marks_dir,
        CLIP,
        lambda d: d["frames"].update({"2": {"type": "not_broadcast"}}),
    )
    update_frame_types(
        gt_marks_dir,
        CLIP,
        lambda d: d["frames"].update({"3": {"type": "too_few_points"}}),
    )
    t = _board(client)[0]["a"]
    assert t["done"] is False and t["progress"] == {"done": 3, "total": 4}
    update_frame_types(
        gt_marks_dir,
        "mancity_seg010",
        lambda d: d["frames"].update({"4": {"type": "not_broadcast"}}),
    )
    t = _board(client)[0]["a"]
    assert t["done"] is True and t["progress"] == {"done": 4, "total": 4}


def test_count_zero_uses_count_url_and_unreachable_is_not_done(
    client, tasks_file, monkeypatch
):
    left = {"total": 3}
    seen = []

    def fake(url):
        seen.append(url)
        if left is None:
            raise OSError("down")
        return left

    monkeypatch.setattr(TD, "fetch_json", fake)
    _post(
        tasks_file,
        {
            "id": "a",
            "title": "A",
            "count_url": "http://mac-mini-i7:8011/x?limit=1",
            "done_when": {"count_zero": True},
        },
        {
            "id": "b",
            "title": "B",
            "count_url": "/y",
            "count_path": "progress.remaining",
            "done_when": {"count_zero": True},
        },
    )
    left = {"total": 3, "progress": {"remaining": 0}}
    by_id, _ = _board(client)
    assert by_id["a"]["done"] is False and by_id["a"]["progress"] == {"remaining": 3}
    assert by_id["b"]["done"] is True
    assert (
        "http://testserver/x?limit=1" in seen
    )  # our own host is fetched through the request base
    left = {"total": 0, "progress": {"remaining": 1}}
    by_id, _ = _board(client)
    assert by_id["a"]["done"] is True and by_id["b"]["done"] is False
    left = None
    by_id, _ = _board(client)
    assert by_id["a"]["done"] is False  # a failing count url never completes a task


def test_inference_from_url_and_count_url_without_touching_the_file(
    client, tasks_file, gt_marks_dir, monkeypatch
):
    monkeypatch.setattr(TD, "fetch_json", lambda url: {"total": 0})
    url = (
        f"http://mac-mini-i7:8011/calibration?clip={CLIP}&frame=205&proposal=bmth_pre1"
    )
    _post(
        tasks_file,
        {"id": "gt", "title": "G", "url": url, "added": "2026-11-15"},
        {
            "id": "q",
            "title": "Q",
            "url": "/calibration/review?run=x",
            "count_url": "/api/q",
        },
        {"id": "plain", "title": "P", "url": "/ball_check"},
    )
    before = tasks_file.read_text()
    by_id, _ = _board(client)
    assert by_id["gt"]["done_when"] == {
        "gt": {"clip": CLIP, "frame": 205, "since": "added"}
    }
    assert by_id["gt"]["done_when_inferred"] is True and by_id["gt"]["done"] is False
    assert (
        by_id["q"]["done_when"] == {"count_zero": True} and by_id["q"]["done"] is True
    )
    assert "done_when" not in by_id["plain"] and by_id["plain"]["done"] is False
    _save_gt(CLIP, 205, ts="2026-11-16T09:00:00+0000")
    assert _board(client)[0]["gt"]["done"] is True  # the Bournemouth seg030 f205 case
    assert tasks_file.read_text() == before  # inferred at read time, never written back


def test_inferred_gt_ignores_gt_older_than_the_task(client, tasks_file, gt_marks_dir):
    """A recheck task opens a frame that already has GT: only a save after it was added completes it."""
    _save_gt(CLIP, 87, ts="2026-11-10T09:00:00+0000")
    _post(
        tasks_file,
        {
            "id": "r",
            "title": "R",
            "added": "2026-11-15",
            "url": f"/calibration?clip={CLIP}&frame=87",
        },
    )
    assert _board(client)[0]["r"]["done"] is False
    _save_gt(CLIP, 87, ts="2026-11-16T09:00:00+0000")
    assert _board(client)[0]["r"]["done"] is True


def test_seed_tasks_are_not_done_on_an_empty_dataset(
    client, tasks_file, gt_marks_dir, monkeypatch
):
    monkeypatch.setattr(TD, "fetch_json", lambda url: {"total": 5})
    by_id, _ = _board(client)
    assert len(by_id) == 4 and not any(t["done"] for t in by_id.values())
    assert by_id["t-seed-bmth-seg030"]["done_when"]["gt"]["frame"] == 205
    assert by_id["t-seed-bmth-nonbroadcast"]["done_when"] == {"count_zero": True}


def test_done_tasks_sort_last_age_off_after_two_days_and_reopen(
    client, tasks_file, gt_marks_dir, monkeypatch
):
    clock = [datetime(2026, 11, 16, 12, 0, tzinfo=UTC)]
    monkeypatch.setattr(TD, "now_utc", lambda: clock[0])
    _post(
        tasks_file,
        {
            "id": "first",
            "title": "F",
            "priority": 1,
            "done_when": {"gt": {"clip": CLIP, "frame": 1}},
        },
        {"id": "mid", "title": "M", "priority": 2},
        {"id": "last", "title": "L", "priority": 3},
    )
    _save_gt(CLIP, 1)
    by_id, order = _board(client)
    assert order == ["mid", "last", "first"]  # done last despite priority 1
    assert by_id["first"]["done"] and by_id["first"]["done_at"].startswith("2026-11-16")
    clock[0] += timedelta(days=1, hours=23)
    assert _board(client)[1] == ["mid", "last", "first"]
    clock[0] += timedelta(hours=2)
    assert _board(client)[1] == ["mid", "last"]  # past 2 days: dropped from the board
    assert "first" in [
        t["id"] for t in json.loads(tasks_file.read_text())["tasks"]
    ]  # ...but never deleted from the file
    # GT removed -> the task is active again and its done timestamp is forgotten
    doc = cal._read_gt(CLIP)
    doc["frames"].clear()
    cal._write_gt(CLIP, doc)
    by_id, order = _board(client)
    assert order == ["first", "mid", "last"] and by_id["first"]["done"] is False
    assert json.loads(TD.sidecar_path(tasks_file).read_text()) == {}


def test_done_evaluation_is_cached_for_a_few_seconds(
    client, tasks_file, gt_marks_dir, monkeypatch
):
    monkeypatch.setattr(TD, "CACHE_TTL", 60.0)
    calls = []
    real = TD._evaluate
    monkeypatch.setattr(TD, "_evaluate", lambda *a: calls.append(1) or real(*a))
    _post(
        tasks_file,
        {"id": "a", "title": "A", "done_when": {"gt": {"clip": CLIP, "frame": 5}}},
    )
    assert not _board(client)[0]["a"]["done"]
    _save_gt(CLIP, 5)
    assert not _board(client)[0]["a"]["done"]  # still the cached answer
    assert len(calls) == 1
    TD._cache.clear()
    assert _board(client)[0]["a"]["done"]


def test_bad_done_when_never_breaks_the_board(client, tasks_file, gt_marks_dir):
    _post(
        tasks_file,
        {"id": "a", "title": "A", "done_when": {"gt": {"clip": CLIP}}},
        {"id": "b", "title": "B", "done_when": {"frames": "nope"}},
        {"id": "c", "title": "C", "done_when": {"count_zero": True}},  # no count_url
        {"id": "d", "title": "D", "done_when": {"wat": 1}},
    )
    by_id, _ = _board(client)
    assert len(by_id) == 4 and not any(t["done"] for t in by_id.values())


def test_cli_add_done_when_flags(client, tasks_file, capsys):
    assert cli.main(["add", "G", "--done-when-gt", f"{CLIP}:205"]) == 0
    assert (
        cli.main(
            ["add", "F", "--done-when-frames", f"{CLIP}:1,arsenal_mancity_seg010:4"]
        )
        == 0
    )
    assert (
        cli.main(["add", "Z", "--count-url", "/api/z", "--done-when-count-zero"]) == 0
    )
    capsys.readouterr()
    by_title = {t["title"]: t for t in json.loads(tasks_file.read_text())["tasks"]}
    assert by_title["G"]["done_when"] == {
        "gt": {"clip": CLIP, "frame": 205, "since": "added"}
    }
    assert by_title["F"]["done_when"] == {
        "frames": [[CLIP, 1], ["arsenal_mancity_seg010", 4]]
    }
    assert by_title["Z"]["done_when"] == {"count_zero": True}
    with pytest.raises(SystemExit):
        cli.main(["add", "bad", "--done-when-gt", "no-frame-here"])
    with pytest.raises(SystemExit):
        cli.main(["add", "bad", "--done-when-count-zero"])  # needs --count-url
    # manual removal still works
    assert cli.main(["done", by_title["G"]["id"]]) == 0


def test_home_page_has_a_collapsed_done_section(client):
    html = client.get("/").text
    assert "donebox" in html and "t.done" in html
