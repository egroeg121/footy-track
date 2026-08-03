"""WebSocket run-protocol tests (docs/labeller_requirements.md §4).

The BackgroundLabeller is replaced with a scripted fake whose ``frames`` are
fully populated at submit time, so the streamer drains them deterministically.
No real propagation, videos, or inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from footy_track.labeller.server import (
    PROV_LABELLER,
    PROV_VITTRACK,
    Session,
)
from footy_track.schema import FrameDetections, ObjectDetection

from .conftest import make_box, patch_labeller_attr


def _fd(idx: int, boxes: list[tuple[str, float]]) -> FrameDetections:
    """FrameDetections at absolute index ``idx`` with (label, x) tracker boxes."""
    return FrameDetections(
        uri=Path(f"clip_frame_{idx:06d}"),
        width=640,
        height=360,
        detections=[
            ObjectDetection(
                label=label, confidence=0.9, x=x, y=0.2, w=0.05, h=0.08, model="raw"
            )
            for label, x in boxes
        ],
    )


class FakeBackgroundLabeller:
    """Scripted stand-in: submit() installs pre-baked frames, no thread."""

    def __init__(
        self,
        frames: list[FrameDetections | None],
        anomaly: tuple[int, str] | None = None,
    ):
        self._script_frames = frames
        self._script_anomaly = anomaly
        self.frames: list[FrameDetections | None] = []
        self.last_completed_frame = -1
        self.running = False
        self.anomaly_frame: int | None = None
        self.anomaly_reason: str | None = None
        self.submissions: list[dict] = []
        self.paused = 0

    def pause(self) -> None:
        self.paused += 1
        self.running = False

    def submit(
        self, video_path, objects, model_uri, conf, start_frame=0, imgsz=512
    ) -> None:
        self.submissions.append(
            {
                "video_path": video_path,
                "objects": objects,
                "model_uri": model_uri,
                "conf": conf,
                "start_frame": start_frame,
                "imgsz": imgsz,
            }
        )
        self.frames = list(self._script_frames)
        self.last_completed_frame = max(
            (i for i, f in enumerate(self.frames) if f is not None), default=-1
        )
        if self._script_anomaly is not None:
            self.anomaly_frame, self.anomaly_reason = self._script_anomaly
        self.running = False  # everything already "completed"

    def frame_at(self, idx: int) -> FrameDetections | None:
        if 0 <= idx < len(self.frames):
            return self.frames[idx]
        return None


@pytest.fixture
def ws_session(monkeypatch) -> Session:
    """A session with video metadata and a 8-frame timeline, no real video."""
    session = Session()
    session.video_path = Path("/fake/clip.mp4")
    session.fps = 25.0
    session.total_frames = 8
    session.width = 640
    session.height = 360
    session.timeline = [None] * 8
    patch_labeller_attr(monkeypatch, "SESSION", session)
    return session


def _drain_until(ws, mtype: str, limit: int = 50) -> list[dict]:
    """Receive messages until one of type ``mtype`` (inclusive)."""
    out = []
    for _ in range(limit):
        m = ws.receive_json()
        out.append(m)
        if m["type"] == mtype:
            return out
    raise AssertionError(f"never received {mtype!r}: {out}")


# ---------------------------------------------------------------------------


def test_run_with_no_seed_boxes_sends_error(client, ws_session):
    ws_session.bg = FakeBackgroundLabeller(frames=[None] * 8)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run", "start_frame": 0})
        msg = ws.receive_json()
    assert msg == {"type": "error", "message": "No boxes on frame 0 to seed from."}
    assert ws_session.bg.submissions == []  # nothing was started


def test_run_streams_compiling_then_running_then_frames_then_done(client, ws_session):
    ws_session.timeline[0] = [make_box("player", PROV_LABELLER)]
    fake = FakeBackgroundLabeller(
        frames=[
            _fd(0, [("player", 0.10)]),
            _fd(1, [("player", 0.11)]),
            _fd(2, [("player", 0.12)]),
        ]
    )
    ws_session.bg = fake

    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "run",
                "start_frame": 0,
                "conf": 0.4,
                "imgsz": 1024,
                "model_uri": "m.pt",
            }
        )
        msgs = _drain_until(ws, "done")
        idle = ws.receive_json()

    # Compiling is announced BEFORE submit blocks (ft-wkc), then again by the
    # streamer, then running before the first frame.
    assert [m["type"] for m in msgs[:3]] == ["status", "status", "status"]
    assert [m["state"] for m in msgs[:3]] == ["compiling", "compiling", "running"]
    frames = [m for m in msgs if m["type"] == "frame"]
    assert [f["idx"] for f in frames] == [0, 1, 2]
    assert msgs[-1] == {"type": "done", "last_frame": 2}
    assert idle == {"type": "status", "state": "idle"}
    # Submit received the seed + run parameters.
    (sub,) = fake.submissions
    assert sub["start_frame"] == 0
    assert sub["conf"] == 0.4
    assert sub["imgsz"] == 1024
    assert sub["model_uri"] == "m.pt"
    assert len(sub["objects"]) == 1  # seeded from the timeline, not the client


def test_run_seed_frame_kept_verbatim_and_downstream_stamped_vittrack(
    client, ws_session
):
    gt = make_box("player", PROV_LABELLER, x=0.42)
    ws_session.timeline[0] = [gt]
    ws_session.bg = FakeBackgroundLabeller(
        frames=[_fd(0, [("player", 0.99)]), _fd(1, [("player", 0.5)])]
    )

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run", "start_frame": 0})
        msgs = _drain_until(ws, "done")

    frames = {m["idx"]: m for m in msgs if m["type"] == "frame"}
    # Seed frame: the timeline GT is emitted verbatim — the tracker's
    # re-detection (x=0.99) must NOT overwrite it.
    assert frames[0]["boxes"][0]["x"] == pytest.approx(0.42)
    assert frames[0]["boxes"][0]["source"] == PROV_LABELLER
    assert frames[0]["gt_kept"] is False
    # Downstream frames are stamped vittrack and written to the timeline.
    assert frames[1]["boxes"][0]["source"] == PROV_VITTRACK
    assert ws_session.timeline[1][0].model == PROV_VITTRACK


def test_mid_clip_run_streams_frames_from_start_frame(client, ws_session):
    """Regression pin for the completed_frames()-from-0 bug: a run seeded at
    frame N (frames 0..N-1 still None) must stream and ingest frames N..M —
    the old contiguous-from-0 scan returned nothing and every frame was
    silently skipped."""
    ws_session.timeline[3] = [make_box("player", PROV_LABELLER, x=0.30)]
    ws_session.bg = FakeBackgroundLabeller(
        frames=[
            None,
            None,
            None,
            _fd(3, [("player", 0.30)]),
            _fd(4, [("player", 0.31)]),
            _fd(5, [("player", 0.32)]),
        ]
    )

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run", "start_frame": 3})
        msgs = _drain_until(ws, "done")

    frames = [m for m in msgs if m["type"] == "frame"]
    assert [f["idx"] for f in frames] == [3, 4, 5]
    # Downstream frames actually landed in the timeline (ingest happened).
    assert ws_session.timeline[4] is not None
    assert ws_session.timeline[4][0].model == PROV_VITTRACK
    assert ws_session.timeline[5][0].model == PROV_VITTRACK
    assert msgs[-1]["last_frame"] == 5


def test_run_reports_gt_kept_frames(client, ws_session):
    ws_session.timeline[0] = [make_box("player", PROV_LABELLER, x=0.10)]
    kept_gt = make_box("player", PROV_LABELLER, x=0.55)
    ws_session.timeline[1] = [kept_gt]  # pre-existing hand marks mid-run
    ws_session.bg = FakeBackgroundLabeller(
        frames=[
            _fd(0, [("player", 0.10)]),
            _fd(1, [("player", 0.11)]),
            _fd(2, [("player", 0.12)]),
        ]
    )

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run", "start_frame": 0})
        msgs = _drain_until(ws, "done")

    frames = {m["idx"]: m for m in msgs if m["type"] == "frame"}
    assert frames[1]["gt_kept"] is True
    # The GT frame is completely untouched — tracker output discarded.
    assert frames[1]["boxes"][0]["x"] == pytest.approx(0.55)
    assert ws_session.timeline[1] == [kept_gt]
    assert frames[2]["gt_kept"] is False


def test_anomaly_handback_pauses_run(client, ws_session):
    ws_session.timeline[0] = [make_box("player", PROV_LABELLER)]
    ws_session.bg = FakeBackgroundLabeller(
        frames=[
            _fd(0, [("player", 0.10)]),
            _fd(1, [("player", 0.11)]),
            _fd(2, [("player", 0.12)]),
        ],
        anomaly=(2, "VitTrack confidence dropped to 0.31 for 'player' (threshold 0.5)"),
    )

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run", "start_frame": 0})
        msgs = _drain_until(ws, "anomaly")
        paused = ws.receive_json()

    anomaly = msgs[-1]
    assert anomaly["idx"] == 2
    assert "confidence dropped" in anomaly["reason"]
    assert paused == {"type": "status", "state": "paused"}
    # Frames up to the anomaly were still streamed and ingested.
    assert [m["idx"] for m in msgs if m["type"] == "frame"] == [0, 1, 2]
    # The anomaly marker is cleared server-side after handback.
    assert ws_session.bg.anomaly_frame is None


def test_restart_message_behaves_like_run(client, ws_session):
    ws_session.timeline[2] = [make_box("player", PROV_LABELLER, x=0.2)]
    fake = FakeBackgroundLabeller(frames=[None, None, _fd(2, [("player", 0.2)])])
    ws_session.bg = fake

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "restart", "start_frame": 2})
        msgs = _drain_until(ws, "done")

    assert fake.submissions[0]["start_frame"] == 2
    assert [m["idx"] for m in msgs if m["type"] == "frame"] == [2]


def test_pause_message_pauses_and_acknowledges(client, ws_session):
    fake = FakeBackgroundLabeller(frames=[None] * 8)
    ws_session.bg = fake

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "pause"})
        msg = ws.receive_json()

    assert msg == {"type": "status", "state": "paused"}
    assert fake.paused >= 1
