"""Pins the per-box GT promotion semantics landed in 940ab23.

Client boxes that carry a ``model`` field keep that provenance through /edit;
boxes without one fall back to the endpoint default (``labeller``) — so saving
a frame promotes only the boxes the user actually touched, and older clients
that omit ``model`` are unaffected.
"""

from __future__ import annotations

from footy_track.labeller.server import PROV_LABELLER, PROV_VITTRACK, PROV_YOLO
from footy_track.labeller.session import boxes_from_payload


def test_edit_keeps_per_box_model_tags(client, fresh_session, monkeypatch):
    fresh_session.total_frames = 3
    fresh_session.timeline = [None] * 3
    monkeypatch.setattr(fresh_session, "schedule_flush", lambda: None)

    r = client.post(
        "/edit",
        json={
            "idx": 1,
            "objects": [
                # Hand-touched box: frontend stamps it labeller.
                {
                    "label": "player",
                    "x": 0.1,
                    "y": 0.1,
                    "w": 0.1,
                    "h": 0.1,
                    "model": "labeller",
                },
                # Untouched machine boxes keep their tags.
                {
                    "label": "player",
                    "x": 0.4,
                    "y": 0.1,
                    "w": 0.1,
                    "h": 0.1,
                    "model": "vittrack",
                },
                {
                    "label": "referee",
                    "x": 0.7,
                    "y": 0.1,
                    "w": 0.1,
                    "h": 0.1,
                    "model": "yolo",
                },
                # Legacy client without a model field: falls back to labeller.
                {"label": "coach", "x": 0.1, "y": 0.5, "w": 0.1, "h": 0.1},
            ],
        },
    )
    sources = [b["source"] for b in r.json()["boxes"]]
    assert sources == [PROV_LABELLER, PROV_VITTRACK, PROV_YOLO, PROV_LABELLER]
    assert [b.model for b in fresh_session.timeline[1]] == sources


def test_boxes_from_payload_model_fallback_rules():
    items = [
        {"label": "player", "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "model": "yolo"},
        {"label": "player", "x": 0.2, "y": 0.1, "w": 0.1, "h": 0.1, "model": None},
        {"label": "player", "x": 0.3, "y": 0.1, "w": 0.1, "h": 0.1, "model": ""},
        {"label": "player", "x": 0.4, "y": 0.1, "w": 0.1, "h": 0.1},
    ]
    out = boxes_from_payload(items, PROV_LABELLER)
    # Explicit tag kept; None/empty/absent all fall back to the default.
    assert [b.model for b in out] == [
        PROV_YOLO,
        PROV_LABELLER,
        PROV_LABELLER,
        PROV_LABELLER,
    ]
