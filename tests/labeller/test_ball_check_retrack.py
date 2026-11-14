"""Ball Check ``source=list`` with re-tracking after each verdict (labeller/retrack_service)."""

from __future__ import annotations

import json
import threading

import pytest

from footy_track.autolabel import retrack as rt
from footy_track.ball_eval.detection_replay import write_cache
from footy_track.labeller import ball_check, server
from tests.test_autolabel_retrack import (  # noqa: E402,I001
    ball_xy,
    card,
    make_cache,
)

from .conftest import *  # noqa: F403 - fixtures


@pytest.fixture
def rig(tmp_path, monkeypatch, clips_dir, gt_marks_dir):
    """One clip, review cards every 10 frames (frames 0..50), 2 of them blind audits."""
    (clips_dir / "stemA.mp4").touch()
    cache_dir = tmp_path / "cache"
    write_cache(cache_dir / "A.jsonl", make_cache())
    cards = []
    for k, f in enumerate(range(0, 60, 10)):
        x, y = (
            (1500, 900) if k % 2 == 0 else ball_xy(f)
        )  # alternate spare ball / real ball
        cards.append(card(f, x, y, cid=f"card{f}"))
    meta = {c["id"]: {"role": "review"} for c in cards}
    meta["card20"]["role"] = "audit"
    meta["card50"]["role"] = "audit"
    run = tmp_path / "round-x"
    run.mkdir()
    (run / "review_list.json").write_text(json.dumps({"items": cards}))
    (run / "cards_meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(server, "_BALL_CHECK_LIST", str(run / "review_list.json"))
    monkeypatch.setattr(server, "_RETRACK_CACHE", str(cache_dir))
    monkeypatch.setattr(server, "_RETRACK_META", str(run / "cards_meta.json"))
    monkeypatch.setattr(server, "_RETRACK_AUTO_DONE", None)
    ball_check._RETRACK.clear()
    return cards


def _queue(client):
    return client.get("/ball_check/queue?source=list&limit=500").json()


def _ids(data):
    return [i["id"] for i in data["items"]]


def _post(client, c, verdict, **extra):
    return client.post(
        "/ball_check/verdict",
        json={
            "clip": c["clip"],
            "frame_index": c["frame_index"],
            "box_index": c["box_index"],
            "source": "list",
            "verdict": verdict,
            **extra,
        },
    ).json()


def _with_index(c):
    from footy_track.labeller.constants import card_box_index  # noqa: PLC0415

    return {**c, "box_index": card_box_index(c["id"])}


def test_queue_unchanged_until_a_retrack_lands_then_followups_come_first(client, rig):
    base = _ids(_queue(client))
    assert base == [c["id"] for c in rig]
    svc = ball_check._retrack()
    gate = threading.Event()
    real = rt.solve

    def slow(cache, cons):
        gate.wait(5)
        return real(cache, cons)

    rt.solve, saved = slow, rt.solve
    try:
        out = _post(
            client, _with_index(rig[1]), "ball"
        )  # returns at once, solve blocked
        assert out["ok"] is True
        data = _queue(client)
        assert data["retrack"]["pending"] == 1
        assert _ids(data) == [
            i for i in base if i != rig[1]["id"]
        ]  # next card as it was
        gate.set()
        assert svc.wait()
    finally:
        rt.solve = saved
    data = _queue(client)
    ids = _ids(data)
    assert data["retrack"]["pending"] == 0 and data["retrack"]["version"] >= 1
    assert sorted(ids) == sorted(set(ids)) and len(ids) >= len(base) - 1
    pinned = svc.status()["pinned"]
    assert 1 <= pinned <= 2
    # the audit cards are never pinned: they keep their place among the rest
    assert not ({"card20", "card50"} & set(ids[:pinned]))
    st = client.get("/ball_check/retrack/status").json()
    assert (
        st["enabled"]
        and st["clips"]["A"]["constraints"] == 1
        and st["clips"]["A"]["ms"] >= 0
    )


def test_all_audits_are_served_however_many_followups(client, rig):
    served = []
    guard = 0
    while guard < 100:
        guard += 1
        data = _queue(client)
        if not data["items"]:
            break
        head = data["items"][0]
        served.append(head["id"])
        c = (
            next(c for c in rig if c["id"] == head["id"])
            if head["id"].startswith("card")
            else head
        )
        _post(client, _with_index(c), "ball")
        assert ball_check._retrack().wait()
    assert guard < 100
    assert len(served) == len(set(served))  # nothing served twice
    assert {"card20", "card50"} <= set(served)  # audits all reached
    assert {c["id"] for c in rig} <= set(served)


def test_dynamic_cards_are_blind_id_keyed_and_judgeable(
    client, rig, tmp_path, monkeypatch
):
    # sparse list: only frame 0 has a card, so a follow-up has to be a new card
    path = tmp_path / "round-x" / "review_list.json"
    path.write_text(json.dumps({"items": [rig[0], rig[5]]}))
    ball_check._LIST_CACHE.clear()
    ball_check._RETRACK.clear()
    r0 = _with_index(rig[0])
    _post(client, r0, "ball")
    svc = ball_check._retrack()
    assert svc.wait()
    dyn = svc.dynamic_cards()
    assert dyn, "a follow-up card was made"
    top = _queue(client)["items"][0]
    assert top["id"] == dyn[0]["id"] or any(d["id"] == top["id"] for d in dyn)
    assert top["blind"] is True and not {"note", "type", "other_bbox", "role"} & set(
        top
    )
    assert top["image_url"].startswith(
        f"/ball_check/crop/{top['clip']}/{top['frame_index']}/"
    )
    # it is a real list card: it can be judged, and a restart finds it again
    assert _post(client, top, "not_ball")["ok"] is True
    assert svc.wait()
    ball_check._RETRACK.clear()
    svc2 = ball_check._retrack()
    assert {d["id"] for d in svc2.dynamic_cards()} >= {top["id"]}
    assert top["id"] not in _ids(_queue(client))  # judged: not served again


def test_auto_done_hides_confident_cards_never_audits(client, rig, monkeypatch):
    monkeypatch.setattr(server, "_RETRACK_AUTO_DONE", "0.4")
    ball_check._RETRACK.clear()
    _post(client, _with_index(rig[1]), "ball")  # anchor on the real ball in frame 10
    assert ball_check._retrack().wait()
    data = _queue(client)
    ids = _ids(data)
    assert data["auto_done"] >= 1
    assert "card20" in ids and "card50" in ids  # audits are never hidden
    assert (
        "card30" not in ids
    )  # confident track on its box (real ball, conf 0.5 >= 0.4)


def test_retrack_off_keeps_the_plain_list(client, rig, monkeypatch):
    monkeypatch.setattr(server, "_RETRACK_CACHE", None)
    data = _queue(client)
    assert "retrack" not in data and _ids(data) == [c["id"] for c in rig]
    assert _post(client, _with_index(rig[0]), "ball")["ok"] is True
    assert client.get("/ball_check/retrack/status").json() == {"enabled": False}


def test_a_reviewer_who_always_takes_the_top_card_still_reaches_the_base_order(
    client, rig
):
    """However many follow-ups there are, a base card comes up after every two pinned."""
    from footy_track.labeller.retrack_service import PINS_PER_BASE  # noqa: PLC0415

    svc = ball_check._retrack()
    todo = [_with_index(c) for c in rig]
    base_ids = [c["id"] for c in todo]
    served = []
    n = 0
    while todo and n < 40:
        n += 1
        # the re-tracker keeps producing follow-ups: top the pinned list up every time
        with svc._lock:
            fake = [
                {**todo[-1], "id": f"fake{n}{k}", "box_index": 9000 + 10 * n + k}
                for k in range(3)
            ]
            todo = todo + fake
            svc._pinned = [i for i in svc._pinned if any(c["id"] == i for c in todo)]
            svc._pinned += [c["id"] for c in fake][: 6 - len(svc._pinned)]
        order, _ = svc.arrange(todo)
        head = order[0]
        served.append(head["id"])
        svc.on_verdict(None, head["id"])  # no clip: only the streak is counted
        todo = [c for c in todo if c["id"] != head["id"]]
        if all(i in served for i in base_ids):
            break
    assert all(
        i in served for i in base_ids
    )  # every base card (audits included) got served
    run, longest = 0, 0
    for i in served:
        run = 0 if i in base_ids else run + 1
        longest = max(longest, run)
    assert longest <= PINS_PER_BASE
