"""Calibration proposal review API + page (src/footy_track/labeller/README.md §16, LAB-125x)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.proposals import read_proposals, write_proposals
from footy_track.labeller import calibration as cal

from ..calibration.test_geometry import H, W, _camera_h
from .test_calibration import KP

CLIP = "arsenal_mancity_seg010"


def _record(score, status="proposed", shift=0.0, with_outlier=False):
    pitch = PitchModel()
    img, _ = geo.apply_homography(np.linalg.inv(_camera_h()), np.array([pitch.keypoints()[k] for k in KP]))
    corr = [{"image": [float(p[0]) + shift, float(p[1])], "pitch_id": k, "pitch_xy": list(pitch.keypoints()[k]),
             "confidence": 0.9, "candidate_id": i + 1} for i, (p, k) in enumerate(zip(img, KP, strict=True))]
    inl = [True] * len(corr)
    if with_outlier:
        corr[0]["image"] = [0.9, 0.9]
        inl[0] = False
    return {
        "frame_index": 0, "correspondences": corr, "H": _camera_h().tolist(),
        "H_convention": "normalised_image_to_pitch_m", "pitch_dims": [105, 68], "image_size": [W, H],
        "method": "ransac", "inliers": inl, "mean_err_m": 0.0, "err_px": [0.0] * len(corr), "ts": "t",
        "status": status, "score": score, "failure": None if status != "failed" else "unusable: close-up",
        "view": {"end": "L", "camera_side": "main_wide", "notes": "left end"},
    }


@pytest.fixture
def prop_env(client, clips_dir, gt_marks_dir, monkeypatch):
    (clips_dir / "arsenal_mancity_20250925_seg010.mp4").touch()
    frame = np.full((H, W, 3), 90, np.uint8)
    monkeypatch.setattr(cal, "_read_frame", lambda v, i: frame)
    doc = {"clip": CLIP, "runs": {
        "claude-x-p1-20260101T000000": {"source": "claude", "created": "2026-01-01", "params": {"model": "x"},
            "frames": {"0": _record(0.9), "1": _record(0.2), "2": _record(0.5), "3": _record(0.0, "failed")}},
        "mock-20260102T000000": {"source": "classical", "created": "2026-01-02", "params": {},
            "frames": {"10": _record(0.7)}},
    }}
    write_proposals(gt_marks_dir, CLIP, doc)
    return gt_marks_dir


def test_pages_served(client):
    r = client.get("/calibration/review")
    assert r.status_code == 200 and "no-store" in r.headers["cache-control"]
    assert 'name="viewport"' in r.text and "width=device-width" in r.text
    assert "/calibration/review" in client.get("/main").text


def test_list_filters_and_sorts_by_score(client, prop_env):
    d = client.get("/calibration/api/proposals").json()  # default: status=proposed, worst first
    assert [i["score"] for i in d["items"]] == [0.2, 0.5, 0.7, 0.9]
    assert {r["run_id"] for r in d["runs"]} == {"claude-x-p1-20260101T000000", "mock-20260102T000000"}
    d = client.get("/calibration/api/proposals?sort=score_desc").json()
    assert [i["score"] for i in d["items"]] == [0.9, 0.7, 0.5, 0.2]
    d = client.get("/calibration/api/proposals?run=mock-20260102T000000").json()
    assert [i["frame_index"] for i in d["items"]] == [10]
    d = client.get("/calibration/api/proposals?status=failed").json()
    assert [i["frame_index"] for i in d["items"]] == [3] and d["items"][0]["failure"]
    assert client.get("/calibration/api/proposals?status=all").json()["total"] == 5
    assert client.get("/calibration/api/proposals?status=bogus").status_code == 400
    assert client.get("/calibration/api/proposals?sort=zig").status_code == 400
    run = next(r for r in client.get("/calibration/api/proposals").json()["runs"] if r["source"] == "claude")
    assert run["counts"]["proposed"] == 3 and run["counts"]["failed"] == 1


def test_get_one_has_overlay_and_404(client, prop_env):
    r = client.get(f"/calibration/api/proposals/{CLIP}/claude-x-p1-20260101T000000/0").json()
    assert r["status"] == "proposed" and r["view"]["notes"] == "left end" and "HALFWAY_LINE" in r["overlay"]
    assert client.get(f"/calibration/api/proposals/{CLIP}/nope/0").status_code == 404
    assert client.get(f"/calibration/api/proposals/{CLIP}/claude-x-p1-20260101T000000/99").status_code == 404
    # alias of the clip resolves to the same file
    assert client.get("/calibration/api/proposals/mancity_seg010/claude-x-p1-20260101T000000/0").status_code == 200


def test_accept_copies_to_gt_with_origin_and_flips_status(client, prop_env):
    run = "claude-x-p1-20260101T000000"
    r = client.post(f"/calibration/api/proposals/{CLIP}/{run}/0/accept")
    assert r.status_code == 200 and r.json()["status"] == "accepted"
    gt = json.loads((prop_env / "calibration_gt" / f"{CLIP}.json").read_text())["frames"]["0"]
    assert gt["origin"] == {"source": "claude", "run_id": run, "proposal_status": "accepted"}
    assert gt["H_convention"] == "normalised_image_to_pitch_m" and len(gt["correspondences"]) == len(KP)
    assert read_proposals(prop_env, CLIP)["runs"][run]["frames"]["0"]["status"] == "accepted"
    # the GT record loads through the normal GT route and the accepted item leaves the queue
    assert client.get(f"/calibration/api/gt/{CLIP}/0").json()["origin"]["run_id"] == run
    assert 0.9 not in [i["score"] for i in client.get("/calibration/api/proposals").json()["items"]]
    # a second accept would clobber GT: refused unless overwrite
    assert client.post(f"/calibration/api/proposals/{CLIP}/{run}/0/accept").status_code == 409
    assert client.post(f"/calibration/api/proposals/{CLIP}/{run}/0/accept", json={"overwrite": True}).status_code == 200


def test_accept_drops_outliers_and_does_not_touch_real_gt_of_other_frames(client, prop_env):
    run = "claude-x-p1-20260101T000000"
    doc = read_proposals(prop_env, CLIP)
    doc["runs"][run]["frames"]["1"] = _record(0.4, with_outlier=True)
    write_proposals(prop_env, CLIP, doc)
    r = client.post(f"/calibration/api/proposals/{CLIP}/{run}/1/accept").json()
    assert r["status"] == "accepted" and len(r["record"]["correspondences"]) == len(KP) - 1


def test_accept_with_changed_points_is_edited(client, prop_env):
    run = "claude-x-p1-20260101T000000"
    rec = client.get(f"/calibration/api/proposals/{CLIP}/{run}/2").json()
    pts = [{"image": c["image"], "pitch_id": c["pitch_id"]} for c in rec["correspondences"]]
    same = client.post(f"/calibration/api/proposals/{CLIP}/{run}/2/accept", json={"correspondences": pts})
    assert same.json()["status"] == "accepted"
    pts[0]["image"] = [pts[0]["image"][0] + 0.01, pts[0]["image"][1]]
    r = client.post(f"/calibration/api/proposals/{CLIP}/{run}/1/accept", json={"correspondences": pts, "overwrite": True})
    assert r.json()["status"] == "edited"
    assert r.json()["record"]["origin"]["proposal_status"] == "edited"
    assert read_proposals(prop_env, CLIP)["runs"][run]["frames"]["1"]["status"] == "edited"


def test_accept_failed_needs_points_and_bad_points_400(client, prop_env):
    run = "claude-x-p1-20260101T000000"
    assert client.post(f"/calibration/api/proposals/{CLIP}/{run}/3/accept").status_code == 409
    assert client.post(f"/calibration/api/proposals/{CLIP}/{run}/1/accept",
                       json={"correspondences": [{"image": [0.1, 0.1], "pitch_id": "CENTRE_SPOT"}]}).status_code == 400
    assert not (prop_env / "calibration_gt").exists() or not (prop_env / "calibration_gt" / f"{CLIP}.json").exists()


def test_reject_flips_status_and_writes_no_gt(client, prop_env):
    run = "claude-x-p1-20260101T000000"
    assert client.post(f"/calibration/api/proposals/{CLIP}/{run}/1/reject", json={"reason": "mirrored"}).status_code == 200
    f = read_proposals(prop_env, CLIP)["runs"][run]["frames"]["1"]
    assert f["status"] == "rejected" and f["reject_reason"] == "mirrored"
    assert not (prop_env / "calibration_gt" / f"{CLIP}.json").exists()
    assert client.post(f"/calibration/api/proposals/{CLIP}/{run}/77/reject").status_code == 404
    rej = client.get("/calibration/api/proposals?status=rejected").json()
    assert [i["frame_index"] for i in rej["items"]] == [1]


def _human_gt(idx, marker):
    """A hand-labelled GT record (no ``origin``) for ``idx``; ``marker`` makes it recognisable."""
    pts = _record(0.9)["correspondences"]
    res = cal._fit({"correspondences": pts, "pitch_dims": [105, 68], "image_size": [W, H], "method": "dlt"})
    rec = cal._gt_record(idx, res, {"image_size": [W, H], "method": "dlt"})
    rec["note"] = marker
    doc = cal._read_gt(CLIP)
    doc["frames"][str(idx)] = rec
    cal._write_gt(CLIP, doc)


def test_accept_never_silently_overwrites_hand_gt(client, prop_env):
    run = "claude-x-p1-20260101T000000"
    _human_gt(0, "hand-labelled")
    r = client.post(f"/calibration/api/proposals/{CLIP}/{run}/0/accept")
    assert r.status_code == 409
    body = r.json()
    assert body["ok"] is False and body["existing_gt"] == "human" and "hand-labelled" in body["error"]
    assert "overwrite=true" in body["error"]
    # the hand GT is untouched and the proposal still waits for review
    assert cal._read_gt(CLIP)["frames"]["0"]["note"] == "hand-labelled"
    assert "origin" not in cal._read_gt(CLIP)["frames"]["0"]
    assert read_proposals(prop_env, CLIP)["runs"][run]["frames"]["0"]["status"] == "proposed"
    # operator-edited points do not bypass it either
    pts = _record(0.9)["correspondences"]
    assert client.post(f"/calibration/api/proposals/{CLIP}/{run}/0/accept", json={"correspondences": pts}).status_code == 409
    assert cal._read_gt(CLIP)["frames"]["0"]["note"] == "hand-labelled"
    # only an explicit overwrite replaces it, and the result is marked as machine-origin
    r = client.post(f"/calibration/api/proposals/{CLIP}/{run}/0/accept", json={"overwrite": True})
    assert r.status_code == 200
    assert cal._read_gt(CLIP)["frames"]["0"]["origin"]["run_id"] == run


def test_accept_conflict_with_machine_gt_is_labelled_as_such(client, prop_env):
    run = "claude-x-p1-20260101T000000"
    assert client.post(f"/calibration/api/proposals/{CLIP}/{run}/0/accept").status_code == 200
    r = client.post(f"/calibration/api/proposals/{CLIP}/mock-20260102T000000/10/accept")
    assert r.status_code == 200  # a different frame is fine
    _record_again = client.post(f"/calibration/api/proposals/{CLIP}/{run}/0/accept")
    assert _record_again.status_code == 409 and _record_again.json()["existing_gt"] == "machine"
    assert "hand-labelled" not in _record_again.json()["error"]


def test_review_page_has_an_explicit_replace_flow(client):
    html = client.get("/calibration/review").text
    assert "already has hand GT" in html and 'id="btnReplace"' in html and "existing_gt" in html


def test_review_page_defaults_to_warp_view(client):
    html = client.get("/calibration/review").text
    assert "showWarp = true, warpMode = true" in html            # warp on at load
    assert "warpMode = true; showWarp = true" in html            # and reset to warp on every item
    assert "setWarp(!warpMode)" in html and "setWarp(warpMode)" in html   # hold flips, release restores
    assert 'class="small on" id="btnWarp"' in html


# Real pilot1 frame 575 of arsenal_mancity_seg010: three of the four stored points
# (centre-circle top/bottom, halfway top) are collinear on the halfway line, so a plain
# refit is refused as degenerate although the proposal's refined H is fine.
_H575 = [[0.30689886890777746, -0.0032974663368638937, -0.04531593329474465],
         [-0.06717557418107739, -0.7736163471902385, 0.5482814194446377],
         [-0.00042308278220137564, 0.011401708985898248, 0.003342736441704121]]
_P575 = [("CENTRE_CIRCLE_TOP", (0.0, 9.15), (0.15387401734159936, 0.5785740889480612)),
         ("CENTRE_CIRCLE_BOTTOM", (0.0, -9.15), (0.1567715987628685, 0.8482552228581172)),
         ("CENTRE_CIRCLE_RIGHT", (9.15, 0.0), (0.475382351302383, 0.6674462592675481)),
         ("HALFWAY_TOP", (0.0, 34.0), (0.1516048116935509, 0.3673765889403585))]


def _degenerate_record():
    rec = _record(0.57)
    rec["correspondences"] = [{"image": list(uv), "pitch_id": k, "pitch_xy": list(xy), "confidence": 0.8,
                               "candidate_id": None} for k, xy, uv in _P575]
    rec.update(H=_H575, inliers=[True] * 4, err_px=[0.0] * 4)
    return rec


def test_accept_pads_collinear_points_and_reproduces_the_stored_h(client, prop_env):
    run = "claude-x-p1-20260101T000000"
    doc = read_proposals(prop_env, CLIP)
    doc["runs"][run]["frames"]["575"] = _degenerate_record()
    write_proposals(prop_env, CLIP, doc)
    r = client.post(f"/calibration/api/proposals/{CLIP}/{run}/575/accept")
    assert r.status_code == 200, r.text
    rec = r.json()["record"]
    assert len(rec["correspondences"]) > 4 and rec["origin"]["padded_synthetic"]
    grid = np.array([(u, v) for u in np.linspace(0.05, 0.95, 5) for v in np.linspace(0.05, 0.95, 5)])
    a, _ = geo.apply_homography(np.array(_H575), grid)
    b, _ = geo.apply_homography(np.array(rec["H"]), grid)
    assert np.abs(a - b).max() < 1e-2  # metres
    assert read_proposals(prop_env, CLIP)["runs"][run]["frames"]["575"]["status"] == "accepted"


def test_edited_points_are_still_refused_when_degenerate(client, prop_env):
    run = "claude-x-p1-20260101T000000"
    pts = [{"image": list(uv), "pitch_id": k} for k, _xy, uv in _P575[:2] + _P575[3:]] + [
        {"image": [0.3, 0.5], "pitch_id": "HALFWAY_BOTTOM"}]
    r = client.post(f"/calibration/api/proposals/{CLIP}/{run}/0/accept", json={"correspondences": pts})
    assert r.status_code == 400 and "degenerate configuration" in r.json()["error"]


def test_pad_points_breaks_collinearity_in_the_pipeline():
    from footy_track.calibration.prelabel.fit import pad_points, refit_gap_m  # noqa: PLC0415

    pitch = PitchModel()
    uv = np.array([p[2] for p in _P575])
    xy = np.array([p[1] for p in _P575])
    assert refit_gap_m(np.array(_H575), uv, xy) == float("inf")
    add = pad_points(np.array(_H575), [tuple(u) for u in uv], [p[0] for p in _P575], [tuple(x) for x in xy], pitch)
    assert add and all(0.03 < u < 0.97 and 0.03 < v < 0.97 for _k, (u, v), _xy in add)
    assert refit_gap_m(np.array(_H575), np.vstack([uv, [a[1] for a in add]]),
                       np.vstack([xy, [a[2] for a in add]])) < 0.25
