"""/matches dashboard: page serves, API renders rows from a fake data tree (no Prefect, no real mounts)."""

from __future__ import annotations

import json

from footy_track.labeller import matches_status


def _build(root):
    fm = root / "full_matches"
    (fm / "work" / "locks").mkdir(parents=True)
    (fm / "work" / "pixeldrain").mkdir()
    (fm / "links").mkdir()
    done_mp4 = fm / "che.v.mun.full.mp4"
    done_mp4.write_bytes(b"x" * 10)
    post = "https://example.test/chelsea-vs-manchester-united-full-match-2026-27/"
    live = "https://example.test/tottenham-hotspur-vs-coventry-city-full-match-2026-27/"
    queued = "https://example.test/arsenal-vs-chelsea-full-match-2026-27/"
    recs = [
        {
            "post_url": post,
            "title": "Chelsea v Man Utd",
            "status": "done",
            "mp4": str(done_mp4),
            "size": 10,
        },
        {
            "post_url": live,
            "title": "Spurs v Cov",
            "host": "pixeldrain",
            "status": "downloading",
        },
    ]
    (fm / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (fm / "links" / "l.jsonl").write_text(
        json.dumps({"post_url": queued, "title": "Arsenal v Chelsea"}) + "\n"
    )
    (fm / "work" / "locks" / "a.lock").write_text(
        json.dumps({"post_url": live, "host": "pixeldrain", "run_id": "run-123"})
    )
    (fm / "work" / "pixeldrain" / "tot.v.cov.full.rar.part").write_bytes(b"y" * 4096)
    # an old-style match with segments + detections
    (root / "arsenal_x" / "original_video").mkdir(parents=True)
    (root / "arsenal_x" / "original_video" / "x.mp4").write_bytes(b"v")
    (root / "clips").mkdir()
    for i in range(3):
        (root / "clips" / f"arsenal_x_seg00{i}.mp4").write_bytes(b"c")
    (root / "machine_labels").mkdir()
    (root / "machine_labels" / "arsenal_x_seg000.jsonl").write_text("{}\n")


def test_matches_page_and_api(client, tmp_path, monkeypatch):
    _build(tmp_path)
    monkeypatch.setenv("FOOTY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FOOTY_RUNS_DIR", str(tmp_path / "no_runs"))
    monkeypatch.setenv(
        "MATCHES_PREFECT_API", "http://127.0.0.1:9/api"
    )  # unreachable -> no links
    monkeypatch.setattr(matches_status, "_api_cache", {"t": 0.0, "runs": []})
    reg = tmp_path / "splits.json"
    reg.write_text(
        '{"arsenal_x": {"split": "train"},'
        ' "chelsea-vs-manchester-united-2026-27": {"split": "heldout"}}'
    )
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(reg))
    from footy_track import match_splits

    match_splits.clear_cache()

    page = client.get("/matches")
    assert page.status_code == 200 and "Matches" in page.text
    assert "/matches" in client.get("/").text  # linked from the hub

    r = client.get("/api/matches?refresh=true")
    assert r.status_code == 200
    d = r.json()
    by = {m["key"]: m for m in d["matches"]}
    assert (
        by["chelsea-vs-manchester-united-2026-27"]["stages"]["download"]["state"]
        == "done"
    )
    live = by["tottenham-hotspur-vs-coventry-city-2026-27"]["stages"]["download"]
    assert (
        live["state"] == "running"
        and "4 KB" in live["text"]
        and live["run_id"] == "run-123"
    )
    assert live["link"].endswith("/runs/flow-run/run-123")
    assert by["arsenal-vs-chelsea-2026-27"]["stages"]["download"]["state"] == "queued"
    x = by["arsenal_x"]["stages"]
    assert x["download"]["state"] == "done"
    assert x["segments"]["text"] == "3 segments"
    assert x["detection"]["text"].startswith("1/3")
    assert x["projection"]["state"] == "na"
    assert by["arsenal_x"]["split"] == "train"
    assert by["chelsea-vs-manchester-united-2026-27"]["split"] == "heldout"
    assert by["arsenal-vs-chelsea-2026-27"]["split"] == "unassigned"
    assert (
        d["matches"][0]["stages"]["download"]["state"] == "running"
    )  # active rows first


def test_matches_api_split_unknown_when_register_missing(client, tmp_path, monkeypatch):
    _build(tmp_path)
    monkeypatch.setenv("FOOTY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FOOTY_RUNS_DIR", str(tmp_path / "no_runs"))
    monkeypatch.setenv("MATCHES_PREFECT_API", "http://127.0.0.1:9/api")
    monkeypatch.setattr(matches_status, "_api_cache", {"t": 0.0, "runs": []})
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(tmp_path / "absent.json"))
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED")
    from footy_track import match_splits

    match_splits.clear_cache()
    r = client.get("/api/matches?refresh=true")
    assert r.status_code == 200
    assert {m["split"] for m in r.json()["matches"]} == {"unknown"}


def _seg_run(rid, state, match, name=""):
    return {
        "id": rid,
        "state": state,
        "state_name": name or state.title(),
        "params": {"match": match, "source": "/x.mp4"},
        "flow": "match-segment",
    }


_n = iter(range(1000))


def _seg_cells(tmp_path, monkeypatch, runs):
    tmp_path = tmp_path / f"case{next(_n)}"
    tmp_path.mkdir()
    _build(tmp_path)
    monkeypatch.setenv("FOOTY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FOOTY_RUNS_DIR", str(tmp_path / "no_runs"))
    monkeypatch.setenv("MATCHES_PREFECT_API", "http://127.0.0.1:9/api")
    monkeypatch.setattr(
        matches_status, "_api_cache", {"t": 9e12, "runs": runs}
    )  # fresh cache -> no network
    monkeypatch.setattr(matches_status, "_cache", {"t": 0.0, "data": None})
    d = matches_status.get_status(force=True)
    return {m["key"]: m for m in d["matches"]}


def test_segments_cell_queued_running_failed(tmp_path, monkeypatch):
    k = "arsenal-vs-chelsea-2026-27"
    cases = [
        ([_seg_run("q1", "SCHEDULED", k, "Scheduled")], "queued", "q1"),
        ([_seg_run("q1", "SCHEDULED", k, "Late")], "queued", "q1"),
        ([_seg_run("p1", "PENDING", k)], "queued", "p1"),
        ([_seg_run("r1", "RUNNING", k)], "running", "r1"),
        ([_seg_run("f1", "FAILED", k)], "failed", "f1"),
        # newest first: the queued retry beats the older failure
        ([_seg_run("q2", "SCHEDULED", k), _seg_run("f1", "FAILED", k)], "queued", "q2"),
    ]
    for runs, state, rid in cases:
        by = _seg_cells(tmp_path, monkeypatch, runs)
        seg = by[k]["stages"]["segments"]
        assert (seg["state"], seg["run_id"]) == (state, rid), runs
        assert seg["link"].endswith("/runs/flow-run/" + rid)


def test_segments_cell_other_match_ignored_and_download_unaffected(
    tmp_path, monkeypatch
):
    k = "arsenal-vs-chelsea-2026-27"
    runs = [_seg_run("q1", "SCHEDULED", "someone-else-2026-27")]
    by = _seg_cells(tmp_path, monkeypatch, runs)
    assert by[k]["stages"]["segments"]["state"] == "none"
    # a segment run for the key must not become the download run link
    live = "tottenham-hotspur-vs-coventry-city-2026-27"
    runs = [_seg_run("q1", "SCHEDULED", live)]
    by = _seg_cells(tmp_path, monkeypatch, runs)
    assert by[live]["stages"]["download"]["run_id"] == "run-123"  # from the lock
    assert by[live]["stages"]["segments"]["state"] == "queued"
    done = "chelsea-vs-manchester-united-2026-27"
    by = _seg_cells(tmp_path, monkeypatch, [_seg_run("q1", "SCHEDULED", done)])
    assert by[done]["stages"]["download"]["run_id"] is None


def test_segments_exist_link_completing_run(tmp_path, monkeypatch):
    runs = [
        _seg_run("new", "SCHEDULED", "arsenal_x"),
        _seg_run("done", "COMPLETED", "arsenal_x"),
        _seg_run("old", "FAILED", "arsenal_x"),
    ]
    by = _seg_cells(tmp_path, monkeypatch, runs)
    seg = by["arsenal_x"]["stages"]["segments"]
    assert seg["state"] == "done" and seg["run_id"] == "done"


def test_find_run_ignores_segment_runs():
    runs = [
        _seg_run("s", "SCHEDULED", "k"),
        {"id": "d", "state": "COMPLETED", "params": {"match": "k"}},
    ]
    assert matches_status._find_run(runs, match="k") == "d"
