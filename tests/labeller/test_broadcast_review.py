"""Broadcast-review job kind: submit, verdict round-trip, computed progress, frame serving, export."""

from __future__ import annotations

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


def test_export_layout(two, web, tmp_path):
    from footy_track.labeller import datasets

    _frames(two)
    for f, v in ((100, "yes"), (200, "no"), (300, "unsure")):
        web.put(f"/broadcast_review/api/verdict/{MATCH}/{f}?dataset=nottinghamforest", json={"verdict": v})
    out = tmp_path / "out"
    counts = exp.export(datasets.get("nottinghamforest"), out)
    assert counts == {"Yes": 1, "No": 1, "unsure": 1, "missing_image": 0}
    assert (out / "train" / "Yes" / f"{MATCH}_f100.jpg").is_file()
    assert (out / "train" / "No" / f"{MATCH}_f200.jpg").is_file()
    assert exp.export(datasets.get("nottinghamforest"), out, {MATCH})["Yes"] == 1  # valid/ split
    assert (out / "valid" / "Yes").is_dir()
