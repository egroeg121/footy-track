"""Broadcast-review job kind: submit, verdict round-trip, computed progress, frame serving, export."""

from __future__ import annotations

import json

from footy_track.scripts import export_broadcast_review as exp

MATCH = "che.v.mun.full.epl.31.10.2026"


def _frames(two, frames=(100, 200, 300)):
    d = two["nottinghamforest"] / "clips" / MATCH
    d.mkdir(parents=True)
    for f in frames:
        (d / f"f{f}.jpg").write_bytes(b"\xff\xd8jpeg")


def _submit(web):
    items = [{"clip": MATCH, "frame": f, "ts_s": f / 10, "p_yes": 0.5, "pred": "Yes", "reason": "edge"} for f in (100, 200, 300)]
    r = web.post("/api/jobs", json={"title": "bc review", "kind": "broadcast_review", "dataset": "nottinghamforest",
                                    "items": items, "key": "br-test"})
    assert r.status_code == 201, r.text
    return r.json()


def test_job_progress_and_verdicts(two, web):
    _frames(two)
    job = _submit(web)
    assert "/broadcast_review?" in job["url"] and "job=" in job["url"]
    g = web.get(f"/api/jobs/{job['id']}?progress=true").json()
    assert g["progress"]["total"] == 3 and g["progress"]["done"] == 0

    r = web.put(f"/broadcast_review/api/verdict/{MATCH}/100?dataset=nottinghamforest", json={"verdict": "no", "p_yes": 0.5})
    assert r.status_code == 200
    web.put(f"/broadcast_review/api/verdict/{MATCH}/200?dataset=nottinghamforest", json={"verdict": "yes"})
    assert web.put(f"/broadcast_review/api/verdict/{MATCH}/300?dataset=nottinghamforest", json={"verdict": "maybe"}).status_code == 422

    g = web.get(f"/api/jobs/{job['id']}?progress=true").json()
    assert (g["progress"]["done"], g["progress"]["remaining"]) == (2, 1)
    view = web.get(f"/broadcast_review/api/job/{job['id']}").json()
    assert [f["verdict"] for f in view["frames"]] == ["no", "yes", None]

    res = web.get(f"/api/jobs/{job['id']}/results").json()
    assert {(r["frame"], r["verdict"]) for r in res["labels"]} == {(100, "no"), (200, "yes")}

    web.delete(f"/broadcast_review/api/verdict/{MATCH}/100?dataset=nottinghamforest")
    assert web.get(f"/api/jobs/{job['id']}?progress=true").json()["progress"]["done"] == 1


def test_frame_serving_and_traversal(two, web):
    _frames(two)
    ok = web.get(f"/broadcast_review/frame/{MATCH}/100.jpg?dataset=nottinghamforest")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg"
    assert web.get(f"/broadcast_review/frame/{MATCH}/999.jpg?dataset=nottinghamforest").status_code == 404
    assert web.get("/broadcast_review/frame/..%2Fx/1.jpg?dataset=nottinghamforest").status_code == 404
    assert web.get("/broadcast_review").status_code == 200


def _manifest(tmp_path, monkeypatch, key="chelsea-vs-manchester-united-2026-27"):
    from footy_track import splits

    m = tmp_path / "manifest.jsonl"
    m.write_text(json.dumps({"post_url": f"https://x/{key}/", "mp4": f"/x/{MATCH}.mp4"}) + "\n")
    monkeypatch.setenv("FOOTY_FULL_MATCHES_MANIFEST", str(m))
    splits.clear_cache()


def test_export_layout(two, web, tmp_path, monkeypatch):
    from footy_track import splits
    from footy_track.labeller import datasets

    _frames(two)
    _manifest(tmp_path, monkeypatch)
    for f, v in ((100, "yes"), (200, "no"), (300, "unsure")):
        web.put(f"/broadcast_review/api/verdict/{MATCH}/{f}?dataset=nottinghamforest", json={"verdict": v})
    out = tmp_path / "out"
    counts = exp.export(datasets.get("nottinghamforest"), out)
    assert counts == {"Yes": 1, "No": 1, "unsure": 1, "missing_image": 0, "heldout_skipped": 0}
    # frames 100 and 200 share one window, so they land on the same side (authority decides which)
    key = "chelsea-vs-manchester-united-2026-27"
    side = {"train": "train", "val": "valid"}[
        splits.export_dir(splits.split_of(key, splits.broadcast_window_id(key, 100)))
    ]
    assert (out / side / "Yes" / f"{MATCH}_f100.jpg").is_file()
    assert (out / side / "No" / f"{MATCH}_f200.jpg").is_file()
    # explicit override still wins
    out2 = tmp_path / "out2"
    assert exp.export(datasets.get("nottinghamforest"), out2, {MATCH})["Yes"] == 1
    assert (out2 / "valid" / "Yes").is_dir()


def test_export_unmappable_match_is_an_error(two, web, tmp_path):
    import pytest

    from footy_track.labeller import datasets

    _frames(two)
    web.put(f"/broadcast_review/api/verdict/{MATCH}/100?dataset=nottinghamforest", json={"verdict": "yes"})
    with pytest.raises(ValueError, match="cannot map"):
        exp.export(datasets.get("nottinghamforest"), tmp_path / "o")
    assert exp.export(datasets.get("nottinghamforest"), tmp_path / "o", allow_unmapped=True)["Yes"] == 1


def test_export_heldout_match_frames_are_left_out(two, web, tmp_path, monkeypatch):
    from footy_track.labeller import datasets

    _frames(two)
    reg = tmp_path / "ms.json"
    reg.write_text(json.dumps({"chelsea-vs-manchester-united-2026-27": {"split": "heldout"}}))
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(reg))
    _manifest(tmp_path, monkeypatch)
    web.put(f"/broadcast_review/api/verdict/{MATCH}/100?dataset=nottinghamforest", json={"verdict": "yes"})
    ds = datasets.get("nottinghamforest")
    c = exp.export(ds, tmp_path / "o", allow_unmapped=False)
    assert c["Yes"] == 0 and c["heldout_skipped"] == 1
    c = exp.export(ds, tmp_path / "o2", heldout_as_test=True)
    assert c["Yes"] == 1 and (tmp_path / "o2" / "test" / "Yes").is_dir()


# ---------------------------------------------------------------------------
# Per-match evaluation + job trim
# ---------------------------------------------------------------------------


def test_wilson():
    from footy_track.labeller.broadcast_review import wilson

    assert wilson(0, 0) is None
    lo, hi = wilson(35, 35)
    assert hi == 1.0 and 0.9 < lo < 0.95
    lo, hi = wilson(5, 10)
    assert round(lo, 3) == 0.237 and round(hi, 3) == 0.763


def test_eval_match_separates_random_and_edge():
    from footy_track.labeller.broadcast_review import eval_match

    rows = [
        {"verdict": "yes", "pred": "Yes", "reason": "random"},
        {"verdict": "no", "pred": "Yes", "reason": "random"},  # wrong
        {"verdict": "no", "p_yes": 0.1, "reason": "random"},  # pred from p_yes
        {"verdict": "unsure", "pred": "Yes", "reason": "random"},
        {"verdict": "no", "pred": "Yes", "reason": "edge_mid"},  # edge, false yes
        {"verdict": "yes", "pred": "No", "reason": "edge_boundary"},  # edge, false no
        {"verdict": "yes", "pred": "Yes", "reason": "edge_mid"},
        {"verdict": None, "pred": "Yes", "reason": "random"},  # not labelled
    ]
    e = eval_match(rows)
    assert (e["random"]["n"], e["random"]["correct"]) == (3, 2) and abs(e["random"]["acc"] - 2 / 3) < 1e-9
    assert e["edge"] == {"n": 3, "wrong": 2, "false_yes": 1, "false_no": 1}
    assert e["unsure"] == 1
    assert eval_match([])["random"]["acc"] is None


def _submit_mixed(web, key, match, n_random=4, n_edge=4):
    items = [{"clip": match, "frame": 1000 + k, "ts_s": 100 + k, "p_yes": 0.9, "pred": "Yes", "reason": "random"} for k in range(n_random)]
    items += [{"clip": match, "frame": 2000 + k, "ts_s": 200 + k, "p_yes": 0.5, "pred": "Yes", "reason": "edge_mid"} for k in range(n_edge)]
    r = web.post("/api/jobs", json={"title": f"Broadcast review: {match}", "kind": "broadcast_review", "dataset": "nottinghamforest",
                                    "items": items, "key": key})
    assert r.status_code == 201, r.text
    return r.json()


def test_summary_ranks_worst_first(two, web):
    a, b = "aaa.match", "bbb.match"
    ja, jb = _submit_mixed(web, "k-a", a), _submit_mixed(web, "k-b", b)
    put = lambda m, f, v: web.put(f"/broadcast_review/api/verdict/{m}/{f}?dataset=nottinghamforest", json={"verdict": v})  # noqa: E731
    for f in range(1000, 1004):  # a: 2/4 right on random, 1 wrong edge
        put(a, f, "yes" if f < 1002 else "no")
    put(a, 2000, "no")
    for f in range(1000, 1004):  # b: 4/4 right
        put(b, f, "yes")
    r = web.get("/broadcast_review/api/summary?dataset=nottinghamforest").json()
    ms = r["matches"]
    assert [m["match"] for m in ms] == [a, b] and ms[0]["split"] in ("train", "unknown") and r["pooled"]
    assert (ms[0]["labelled"], ms[0]["total"]) == (5, 8)
    assert ms[0]["random"]["acc"] == 0.5 and ms[0]["random"]["n"] == 4 and ms[0]["edge"]["wrong"] == 1
    assert ms[1]["random"]["acc"] == 1.0 and ms[0]["job_id"] == ja["id"] and ms[1]["job_id"] == jb["id"]


def test_trim_keeps_answered_and_listed(two, web):
    m = "ccc.match"
    j = _submit_mixed(web, "k-c", m)  # frames 1000-1003 random, 2000-2003 edge
    web.put(f"/broadcast_review/api/verdict/{m}/1003?dataset=nottinghamforest", json={"verdict": "yes"})  # answered, not in keep
    keep = [{"clip": m, "frame": 1000}, {"clip": m, "frame": 2001}]
    r = web.post(f"/api/jobs/{j['id']}/trim", json={"keep": keep, "by": "test"})
    assert r.status_code == 200, r.text
    assert r.json()["removed"] == 5 and r.json()["total"] == 3 and r.json()["answered_kept"] == 1
    view = web.get(f"/broadcast_review/api/job/{j['id']}").json()
    assert [f["frame"] for f in view["frames"]] == [1000, 1003, 2001] and view["rev"] == 1
    assert web.post(f"/api/jobs/{j['id']}/trim", json={"keep": keep}).json()["removed"] == 0  # idempotent
    assert web.post(f"/api/jobs/{j['id']}/trim", json={"keep": "x"}).status_code == 422
    assert web.post(f"/api/jobs/{j['id']}/trim", json={"keep": [{"clip": m, "frame": 5}], "if_rev": 0}).status_code == 409


def test_pool_by_split():
    from footy_track.labeller.broadcast_review import pool_by_split

    mk = lambda split, n, k, ew, en: {"split": split, "total": 50, "labelled": n + en, "random": {"n": n, "correct": k}, "edge": {"n": en, "wrong": ew}}  # noqa: E731
    p = {r["split"]: r for r in pool_by_split([mk("train", 10, 9, 1, 5), mk("train", 30, 30, 0, 5), mk("heldout", 50, 40, 0, 0)])}
    assert p["train"]["random"]["n"] == 40 and p["train"]["random"]["correct"] == 39 and p["train"]["edge"] == {"n": 10, "wrong": 1}
    assert p["heldout"]["random"]["acc"] == 0.8 and p["heldout"]["matches"] == 1
