"""rerefine: only ``proposed`` items change, the old record is kept, reviewed items are untouched."""

from __future__ import annotations

import copy

from footy_track.calibration.prelabel import rerefine as rr
from footy_track.calibration.prelabel.proposals import read_proposals, write_proposals

CLIP = "arsenal_mancity_seg010"


def _rec(status, score, idx=0):
    return {"frame_index": idx, "status": status, "score": score, "ts": f"t{idx}", "H": [[1]],
            "correspondences": [
                {"image": [0.5, 0.5], "image_label": [0.4, 0.4], "pitch_id": "CENTRE_SPOT", "confidence": 0.9},
                {"image": [0.1, 0.1], "pitch_id": None, "synthetic": True}],
            "line_labels": [{"line_id": "HALFWAY_LINE", "points": [[0.5, 0.1], [0.5, 0.9]], "confidence": 0.9}]}


def test_labels_rebuilt_from_label_positions_without_synthetic():
    labels, lines = rr.labels_from_record(_rec("proposed", 0.5))
    assert [(lb.keypoint_id, lb.uv) for lb in labels] == [("CENTRE_SPOT", (0.4, 0.4))]
    assert lines[0].line_id == "HALFWAY_LINE" and lines[0].points == [(0.5, 0.1), (0.5, 0.9)]


def test_only_proposed_change_and_previous_kept(tmp_path, monkeypatch):
    old_prev = _rec("proposed", 0.3)
    old_prev["previous"] = {"score": 0.1}  # an older re-refine: must not nest
    doc = {"clip": CLIP, "runs": {"r": {"frames": {
        "0": old_prev, "1": _rec("accepted", 0.9, 1), "2": _rec("rejected", 0.4, 2),
        "3": _rec("edited", 0.6, 3), "4": _rec("failed", 0.0, 4)}}}}
    write_proposals(tmp_path, CLIP, doc)
    before = copy.deepcopy(doc)

    def fake(job):
        clip, run_id, idx, old, *_ = job
        return clip, run_id, idx, {**_rec("proposed", 0.8, int(idx)), "distortion": {"k1": -0.1}}, None

    monkeypatch.setattr(rr, "_work", fake)
    s = rr.rerefine(tmp_path, [tmp_path], workers=1, log=lambda *_: None)
    assert (s.rerefined, s.now_failed, s.unchanged) == (1, 0, 4)
    after = read_proposals(tmp_path, CLIP)["runs"]["r"]["frames"]
    assert after["0"]["score"] == 0.8 and after["0"]["rerefined"]["fit_k1"] is True
    assert after["0"]["previous"]["score"] == 0.3 and "previous" not in after["0"]["previous"]
    for k in "1234":
        assert after[k] == before["runs"]["r"]["frames"][k]


def test_status_changed_meanwhile_is_left_alone(tmp_path, monkeypatch):
    write_proposals(tmp_path, CLIP, {"clip": CLIP, "runs": {"r": {"frames": {"0": _rec("proposed", 0.3)}}}})

    def fake(job):  # the reviewer accepts it while we compute
        d = read_proposals(tmp_path, CLIP)
        d["runs"]["r"]["frames"]["0"]["status"] = "accepted"
        write_proposals(tmp_path, CLIP, d)
        return job[0], job[1], job[2], _rec("proposed", 0.8), None

    monkeypatch.setattr(rr, "_work", fake)
    s = rr.rerefine(tmp_path, [tmp_path], workers=1, log=lambda *_: None)
    assert s.rerefined == 0
    assert read_proposals(tmp_path, CLIP)["runs"]["r"]["frames"]["0"]["status"] == "accepted"


def test_dry_run_writes_nothing(tmp_path):
    write_proposals(tmp_path, CLIP, {"clip": CLIP, "runs": {"r": {"frames": {"0": _rec("proposed", 0.3)}}}})
    p = tmp_path / "calibration_proposals" / f"{CLIP}.json"
    b = p.read_bytes()
    rr.rerefine(tmp_path, [tmp_path], dry_run=True, log=lambda *_: None)
    assert p.read_bytes() == b
