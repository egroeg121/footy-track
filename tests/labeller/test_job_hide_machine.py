"""Job mode hides machine boxes (LAB-1444) without ever losing them on save.

The page keeps hidden machine boxes in ``hiddenMachine`` and appends them
unchanged to every ``/edit`` body. These tests replay that contract against the
real server: /timeline tells the page which boxes are human, and a save that
carries the hidden boxes back leaves their sidecar rows byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from footy_track.labeller.server import Session

INDEX_HTML = Path(__file__).resolve().parents[2] / "src/footy_track/labeller/web/index.html"


class _Stem:
    def __init__(self, stem: str) -> None:
        self.stem = stem


def _line(idx, label, *tags, x=0.1):
    return {
        "frame_index": idx,
        "bbox": {"x": x, "y": 0.2, "w": 0.01, "h": 0.02},
        "center": {"x": x + 0.005, "y": 0.21},
        "tags": [label, *tags],
    }


def test_save_in_job_mode_keeps_hidden_machine_rows(client, fresh_session, gt_marks_dir):
    rows = [
        _line(2, "in_play_ball", "vittrack", x=0.30),
        _line(2, "player", "yolo", x=0.40),
        _line(2, "referee", "yolo", x=0.50),
        _line(2, "in_play_ball", "vittrack", "human_checked", x=0.60),
        _line(2, "in_play_ball", "labeller", x=0.70),
    ]
    (gt_marks_dir / "clip_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s: Session = fresh_session
    s.video_path = _Stem("clip_a")
    s.total_frames = 5
    s.timeline = [None] * 5
    s.no_ball_frames = set()
    s.not_broadcast_frames = set()
    s._load_existing_marks()

    boxes = client.get("/timeline/2").json()["boxes"]
    human = [b for b in boxes if b["source"] == "labeller" or b["human_checked"]]
    hidden = [b for b in boxes if b not in human]
    assert len(human) == 2 and len(hidden) == 3
    assert {b["source"] for b in hidden} == {"vittrack", "yolo"}

    # Client in job mode: draws only `human`, user adds a new ball, saves
    # visible + hiddenMachine (what boxesNorm() sends).
    new = {"label": "in_play_ball", "model": "labeller", "conf": 1.0, "x": 0.8, "y": 0.2, "w": 0.01, "h": 0.02}
    visible = [
        {"label": b["label"], "model": b["source"], "conf": b["conf"], "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}
        for b in human
    ]
    hid = [
        {"label": b["label"], "model": b["source"], "conf": b["conf"], "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}
        for b in hidden
    ]
    assert client.post("/edit", json={"idx": 2, "objects": visible + [new] + hid}).status_code == 200
    s._do_flush()

    out = [json.loads(x) for x in (gt_marks_dir / "clip_a.jsonl").read_text().splitlines()]
    key = lambda r: (r["frame_index"], tuple(r["tags"]), tuple(r["bbox"].values()))  # noqa: E731
    got = {key(r) for r in out}
    for r in rows:
        assert key(r) in got, f"row lost or changed: {r}"  # incl. the human_checked marker
    assert len(out) == len(rows) + 1


def test_page_merges_hidden_boxes_on_save_and_has_toggle():
    html = INDEX_HTML.read_text()
    assert 'id="jobShowMachine"' in html
    assert "vis.concat(hiddenMachine)" in html
    assert '(e.key === "m" || e.key === "M")' in html


def test_hidden_machine_boxes_never_undo_no_ball_or_not_broadcast():
    """A no_ball / not_broadcast frame keeps no boxes in the sidecar, and a non-empty /edit would make the server
    drop the mark: hidden machine boxes are not sent for a skip-marked frame, and they do not clear the job's mark."""
    from pathlib import Path

    html = (Path(__file__).resolve().parents[2] / "src/footy_track/labeller/web/index.html").read_text()
    assert "skipMarked" in html and "(visibleOnly || skipMarked) ? vis : vis.concat(hiddenMachine)" in html
    assert 'if (boxLayer.find(".box").length || hiddenMachine.length)' not in html


def test_timeline_bar_marks_job_frames_of_loaded_clip():
    html = INDEX_HTML.read_text()
    bar = html.split("function renderTimelineBar()")[1].split("\n}\n")[0]
    assert "job.frames" in bar and "stemOf(loadedPath)" in bar and "jobDone.has" in bar
    render = html.split("function jobRender()")[1].split("\n}\n")[0]
    assert "renderTimelineBar()" in render
