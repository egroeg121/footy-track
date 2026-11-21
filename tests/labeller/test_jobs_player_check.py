"""``player_check`` labelling jobs (README 19l, LAB-1480..1482): window items, progress from the feature store's
player frame_label_state, and the deep link into /player-check."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pytest

from footy_track.feature_store import FeatureStore
from footy_track.feature_store.schema import (
    ClsGroup,
    FrameLabelStateRow,
    LabelMethod,
    LabelState,
)
from footy_track.labeller import jobs

CLIP = "arsenal_nottinghamforest_seg003"
ITEMS = [
    {"clip": CLIP, "frame_start": 500, "frame_end": 540, "step": 10, "eval": True},
    {"clip": CLIP, "frame_start": 900, "frame_end": 940, "step": 10, "eval": True},
]


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = tmp_path / "fs.duckdb"
    monkeypatch.setenv("FOOTY_FEATURE_STORE", str(p))
    return p


def _complete(db, frames, state=LabelState.COMPLETE):
    with FeatureStore.open(db) as s:
        s.upsert_frame_label_states(
            [
                FrameLabelStateRow(
                    game_id=CLIP, frame_index=f, cls_group=ClsGroup.PLAYER, state=state,
                    method=LabelMethod.HAND_FRAME, run_id="player_check", verified_by="t",
                    verified_at=datetime(2026, 1, 1),
                )
                for f in frames
            ]
        )


def _submit(web, **over):
    return web.post(
        "/api/jobs",
        json={"title": "Player check", "kind": "player_check", "dataset": "nottinghamforest", "items": ITEMS, **over},
    )


def test_submit_keeps_window_items_and_eval_flag(web, db):
    r = _submit(web)
    assert r.status_code == 201
    j = web.get(f"/api/jobs/{r.json()['id']}").json()
    assert j["kind"] == "player_check" and j["progress"]["total"] == 10 and j["progress"]["done"] == 0
    assert all(i["eval"] is True and i["step"] == 10 for i in j["items"])


def test_progress_counts_only_complete_player_frames_and_next_is_the_first_open_window(web, db):
    jid = _submit(web).json()["id"]
    _complete(db, [500, 510, 520, 530, 540, 900])
    _complete(db, [910], LabelState.PARTIAL)  # partial does not count
    j = web.get(f"/api/jobs/{jid}").json()
    assert (j["progress"]["done"], j["progress"]["total"]) == (6, 10)
    q = parse_qs(urlparse(j["url"]).query)
    assert urlparse(j["url"]).path == "/player-check"
    assert q["clip"] == [CLIP] and q["start"] == ["900"] and q["end"] == ["940"] and q["eval"] == ["true"] and q["view"] == ["frame"]
    assert q["dataset"] == ["nottinghamforest"] and q["job"] == [jid]


def test_job_goes_done_when_every_frame_is_complete(web, db):
    jid = _submit(web).json()["id"]
    _complete(db, [*range(500, 541, 10), *range(900, 941, 10)])
    assert web.get(f"/api/jobs/{jid}").json()["status"] == "done"


def test_board_task_links_to_player_check_with_a_count(web, db):
    _submit(web, device="phone")
    t = next(t for t in web.get("/api/tasks").json()["tasks"] if t.get("kind") == "player_check")
    assert t["url"].startswith("/player-check?") and t["device"] == "phone" and t["remaining"] == 10


def test_missing_store_means_nothing_done_not_an_error(web, db):
    assert not db.exists()
    assert jobs._player_complete_frames(CLIP) == set()
