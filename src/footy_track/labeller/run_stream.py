"""Live frame streaming for propagation runs over the websocket.

Implements the server side of the run protocol in
``docs/labeller_requirements.md`` §4: polling the ``BackgroundLabeller``,
ingesting completed frames into the session timeline (GT-authoritative), and
pushing status/frame/anomaly/done messages to the client. The ``/ws`` message
handler itself lives in ``server.py``.
"""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

from footy_track.labeller.constants import PROV_VITTRACK
from footy_track.labeller.session import Session, boxes_payload
from footy_track.schema import FrameDetections, ObjectDetection


def ingest_completed_frame(
    session: Session, idx: int, fd: FrameDetections, start_idx: int
) -> tuple[list[ObjectDetection], bool]:
    """Write a propagated frame into the timeline and return (boxes, gt_kept).

    The seed frame (idx == start_idx) is the ground-truth seed — its timeline
    entry is already correct, so the tracker's re-detection is not merged into
    it. Downstream frames get their detections stamped vittrack and merged in
    (keeping labeller ground truth); ``gt_kept`` is True when existing GT made
    the frame skip the merge.
    """
    if idx == start_idx:
        return session.get_frame(idx), False
    vittrack_boxes = [
        ObjectDetection(
            label=d.label,
            confidence=d.confidence,
            x=d.x,
            y=d.y,
            w=d.w,
            h=d.h,
            model=PROV_VITTRACK,
        )
        for d in fd.detections
    ]
    gt_kept = session.merge_propagated(idx, vittrack_boxes)
    return session.get_frame(idx), gt_kept


async def stream_frames(websocket: WebSocket, session: Session, start_idx: int) -> None:
    """Push each newly-completed frame to the client until the run stops."""
    sent = start_idx - 1
    await websocket.send_json({"type": "status", "state": "compiling"})
    announced_running = False
    while True:
        cur_bg = session.bg  # may swap on a fresh load
        while sent < cur_bg.last_completed_frame:
            sent += 1
            # frame_at handles mid-clip runs: completed_frames() only scans the
            # contiguous run from frame 0, so a run seeded at frame N (with
            # frames 0..N-1 still None) would silently skip every frame —
            # nothing ingested into the timeline, nothing streamed (the bug
            # behind "ran to frame 30 but 28-29 have no boxes").
            fd = cur_bg.frame_at(sent)
            if fd is not None:
                if not announced_running:
                    await websocket.send_json({"type": "status", "state": "running"})
                    announced_running = True
                boxes, gt_kept = ingest_completed_frame(session, sent, fd, start_idx)
                await websocket.send_json(
                    {
                        "type": "frame",
                        "idx": sent,
                        "boxes": boxes_payload(boxes),
                        "gt_kept": gt_kept,
                    }
                )
        if cur_bg.anomaly_frame is not None:
            await websocket.send_json(
                {
                    "type": "anomaly",
                    "idx": cur_bg.anomaly_frame,
                    "reason": cur_bg.anomaly_reason or "implausible track motion",
                }
            )
            cur_bg.anomaly_frame = None
            await websocket.send_json({"type": "status", "state": "paused"})
            return
        if not cur_bg.running:
            await websocket.send_json(
                {"type": "done", "last_frame": cur_bg.last_completed_frame}
            )
            await websocket.send_json({"type": "status", "state": "idle"})
            return
        await asyncio.sleep(0.1)
