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
