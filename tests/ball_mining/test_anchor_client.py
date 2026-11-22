"""Anchor search client (design ``ball_anchor_iteration.md`` task 4): cards -> ``ball_check`` items appended to the round job
within the open / daily caps, idempotent, rev-conflict retry, crash convergence, and the reorder by generator rank. A fake
jobs client over the real in-process job store (the same one the labeller API serves)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from footy_track.ball_mining import anchor_search as A  # noqa: N812
from footy_track.ball_mining import round as rnd
from footy_track.labeller import job_edit, jobs, jobs_client

from .conftest import LocalApi
from .test_anchor_search import HC, TC, register_splits  # noqa: F401

DAY = datetime.now(UTC).strftime("%Y-%m-%d")  # added_by.ts is stamped with the real UTC clock
KEY = f"ball-round/{DAY}/1"
BOX = {"x": 0.4, "y": 0.5, "w": 0.02, "h": 0.04}


def seed(frame, bucket="R"):
    return {"clip": TC, "frame_index": frame, "bbox": dict(BOX), "p_real": 0.5, "sources": ["rtdetr"],
            "sampling": {"regime": "train", "bucket": bucket, "split": "train"}}  # fmt: skip


def submit(items, key=KEY):
    job, created = jobs.submit({"title": "round", "kind": "ball_check", "dataset": "train", "items": items, "key": key,
                                "max_cards": 2000}, "train")  # fmt: skip
    assert created
    return job["id"]


def spec(frame, rank, *, hop=1, margin=0.5, rank_conf=0.8, window="w1", clip=TC):
    return {"clip": clip, "frame_index": frame, "bbox": {"x": 0.4 + frame / 10000, "y": 0.5, "w": 0.02, "h": 0.04},
            "conf": 0.6, "raw_conf": 0.8, "p_cal": 0.7, "tid": 3, "cand_source": "detected", "hop": hop, "margin": margin,
            "direction": "fwd", "anchor_id": f"j:{clip}:0", "source": "search", "generator_version": "det:ab|trk:cd|code:anchor-gen/1",
            "teacher_version": "offline_v1", "run_id": "run1", "search_round": 1, "window_id": window, "rank_conf": rank_conf,
            "rank": rank}  # fmt: skip


def doc(cards, digest="dig1", window="w1", split="train", version=1):
    return {"schema": A.SEARCH_SCHEMA, "digest": digest, "window_id": window, "split": split, "version": version, "cards": cards}


class AppendApi(LocalApi):
    """``LocalApi`` plus ``append`` and a job list with progress; hooks simulate a concurrent change or a crash."""

    def __init__(self):
        super().__init__()
        self.before_append = None
        self.after_append = None  # called after the server applied the append (a crash before the client sees the reply)

    def list_jobs(self, **f):
        self.calls.append(("list", f))
        out = []
        for j in jobs.filter_jobs(kind=f.get("kind"), dataset=f.get("dataset")):
            full, c = jobs.job_with_progress(j)
            out.append(jobs.public(full, c, items=False))
        return out

    def append(self, ref, items, *, by="api", note=None, if_rev=None, url=None):
        self.calls.append(("append", ref, len(items), by, note, if_rev))
        if self.before_append:
            self.before_append(ref, items)
        try:
            r = job_edit.append_items(jobs.resolve(ref)["id"], items, by=by, note=note, if_rev=if_rev)
        except jobs.JobError as e:
            raise jobs_client.JobsError(e.status, {"detail": e.detail, **e.extra}) from None
        if self.after_append:
            self.after_append(ref, r)
        return r


@pytest.fixture
def cl():
    return AppendApi()


def frames_of(jid):
    return [(it["frame_index"], (it.get("sampling") or {}).get("bucket")) for it in jobs.read_job(jid)["items"]]


def push(cl, jid, docs, **kw):
    kw.setdefault("today", DAY)
    return A.push_search(jid, docs, client=cl, **kw)


# ---- content ----------------------------------------------------------------------------------------
def test_item_carries_card_geometry_and_the_hidden_search_block(ds, cl):
    jid = submit([seed(0)])
    out = push(cl, jid, [doc([spec(10, 0)])])
    assert out["added"] == 1
    it = jobs.read_job(jid)["items"][1]
    assert (it["clip"], it["frame_index"]) == (TC, 10) and it["bbox"] == spec(10, 0)["bbox"]
    assert it["sampling"]["bucket"] == "S" and it["sampling"]["regime"] == "train" and it["sampling"]["source"] == "search"
    s = it["search"]
    assert {k: s[k] for k in ("anchor_id", "hop", "direction", "source", "generator_version", "teacher_version", "run_id",
                              "search_round", "margin", "p_cal", "window_id", "rank")} == {
        "anchor_id": f"j:{TC}:0", "hop": 1, "direction": "fwd", "source": "search", "generator_version": "det:ab|trk:cd|code:anchor-gen/1",
        "teacher_version": "offline_v1", "run_id": "run1", "search_round": 1, "margin": 0.5, "p_cal": 0.7, "window_id": "w1", "rank": 0}  # fmt: skip
    assert s["idem"] == A.idempotency_key("dig1", spec(10, 0)) and s["digest"] == "dig1" and it["p_real"] == 0.7
    assert it["added_by"]["by"] == A.APPEND_BY


def test_idempotency_key_depends_on_digest_and_card():
    a, b = spec(10, 0), spec(20, 1)
    assert A.idempotency_key("d1", a) == A.idempotency_key("d1", dict(a))
    assert len({A.idempotency_key("d1", a), A.idempotency_key("d2", a), A.idempotency_key("d1", b)}) == 3


# ---- caps -------------------------------------------------------------------------------------------
def test_append_stops_at_the_open_card_headroom_best_card_first(ds, cl):
    jid = submit([seed(i) for i in range(90)])  # 90 open cards, cap 100 -> room 10
    cards = [spec(1000 + i, i, hop=1 + i // 5, margin=0.1 * (i % 5)) for i in range(25)][::-1]  # offered worst first
    out = push(cl, jid, [doc(cards)])
    assert out["added"] == 10 and out["headroom"]["room"] == 10 and out["offered"] == 25
    got = [f for f, b in frames_of(jid) if b == "S"]
    assert got == [1000 + i for i in range(10)]  # the ten best (rank 0..9), in rank order


def test_no_room_means_nothing_is_appended(ds, cl):
    jid = submit([seed(i) for i in range(100)])
    out = push(cl, jid, [doc([spec(1000, 0)])])
    assert out["added"] == 0 and out["headroom"]["room"] == 0
    assert not any(c[0] == "append" for c in cl.calls)


def test_daily_cap_counts_cards_created_today(ds, cl):
    jid = submit([seed(i) for i in range(10)])
    out = push(cl, jid, [doc([spec(1000 + i, i) for i in range(30)])], daily_max=14)  # 10 created today, room 4
    assert out["added"] == 4 and out["headroom"]["today_cards"] == 10
    out = push(cl, jid, [doc([spec(1000 + i, i) for i in range(30)])], daily_max=14)  # now 14 today
    assert out["added"] == 0 and out["headroom"]["room"] == 0


def test_cards_added_today_to_a_round_of_an_earlier_day_count_against_the_daily_cap(ds, cl):
    jid = submit([seed(i) for i in range(10)], key="ball-round/2020-01-01/1")
    out = push(cl, jid, [doc([spec(1000 + i, i) for i in range(30)])], daily_max=14)  # tick sees 0 today; +0 added yet
    assert out["added"] == 14
    out = push(cl, jid, [doc([spec(2000 + i, i) for i in range(30)])], daily_max=14)
    assert out["added"] == 0 and out["headroom"]["today_cards"] == 14


def test_default_caps_are_the_tick_rules():
    assert (rnd.OPEN_CARDS_MAX, rnd.DAILY_CARDS_MAX) == (100, 600)


# ---- idempotency ------------------------------------------------------------------------------------
def test_a_retried_push_adds_nothing(ds, cl):
    jid = submit([seed(0)])
    docs = [doc([spec(10 + i, i) for i in range(5)])]
    assert push(cl, jid, docs)["added"] == 5
    snapshot = jobs.read_job(jid)["items"]
    again = push(cl, jid, docs)
    assert again["added"] == 0 and again["already_in_job"] == 5
    assert jobs.read_job(jid)["items"] == snapshot


def test_server_side_dedupe_backs_up_the_client_filter(ds, cl):
    jid = submit([seed(0)])
    d = doc([spec(10, 0)])
    assert push(cl, jid, [d])["added"] == 1
    r = job_edit.append_items(jid, [A.item_from_card(spec(10, 0), digest="dig1", split="train")])  # raw replay
    assert r["added"] == 0 and r["skipped_duplicates"] == 1


# ---- 409 --------------------------------------------------------------------------------------------
def test_a_rev_conflict_rereads_and_retries(ds, cl):
    jid = submit([seed(0), seed(1)])
    fired = []

    def racing(ref, items):
        if not fired:
            fired.append(1)
            job_edit.reorder_unanswered(jid, [1, 0], by="someone")  # bumps rev between our read and write

    cl.before_append = racing
    out = push(cl, jid, [doc([spec(10, 0), spec(11, 1)])])
    assert out["added"] == 2 and out["append_attempts"] == 2
    ifrevs = [c[5] for c in cl.calls if c[0] == "append"]
    assert ifrevs[1] == ifrevs[0] + 1
    assert [f for f, b in frames_of(jid) if b == "S"] == [10, 11]


def test_a_persistent_rev_conflict_gives_up_cleanly(ds, cl):
    jid = submit([seed(0), seed(1)])

    def busy(ref, items):  # every attempt finds the job reordered since its read
        job_edit.reorder_unanswered(jid, [1, 0], by="busy")

    cl.before_append = busy
    out = push(cl, jid, [doc([spec(10, 0)])])
    assert out["append"] == "rev_conflict" and out["added"] == 0 and out["append_attempts"] == A.APPEND_TRIES
    assert all(b != "S" for _, b in frames_of(jid))


def test_other_server_errors_propagate_and_write_nothing(ds, cl):
    jid = submit([seed(0)])
    with pytest.raises(jobs_client.JobsError) as e:  # a heldout card into a train job: the split guard (422)
        push(cl, jid, [doc([spec(10, 0, clip=HC)], split="train")])
    assert e.value.status == 422 and len(jobs.read_job(jid)["items"]) == 1


# ---- atomic append / kill test ---------------------------------------------------------------------
def test_kill_between_append_and_reorder_converges_without_duplicates_or_loss(ds, cl):
    jid = submit([seed(0), seed(1, "U"), seed(2)])
    docs = [doc([spec(10 + i, i, margin=0.1 * i) for i in range(4)])]

    class Crash(BaseException):  # a kill, not an error the code could catch
        pass

    def kill(ref, order, **kw):
        raise Crash

    real = cl.reorder
    cl.reorder = kill
    with pytest.raises(Crash):
        push(cl, jid, docs)
    after_crash = jobs.read_job(jid)["items"]
    assert [it["frame_index"] for it in after_crash][-4:] == [10, 11, 12, 13]  # appended, order is the old one
    cl.reorder = real
    out = push(cl, jid, docs)  # the rerun
    assert out["added"] == 0 and out["already_in_job"] == 4
    final = jobs.read_job(jid)["items"]
    keys = [(it["clip"], it["frame_index"]) for it in final]
    assert len(keys) == len(set(keys)) == 7  # no duplicates, no lost card
    assert {it["frame_index"] for it in final} == {0, 1, 2, 10, 11, 12, 13}


def test_kill_after_the_server_applied_the_append_but_before_the_reply(ds, cl):
    jid = submit([seed(0)])
    docs = [doc([spec(10 + i, i) for i in range(3)])]

    class Crash(BaseException):
        pass

    def die(ref, r):
        raise Crash

    cl.after_append = die
    with pytest.raises(Crash):
        push(cl, jid, docs)
    cl.after_append = None
    out = push(cl, jid, docs)
    assert out["added"] == 0 and len(jobs.read_job(jid)["items"]) == 4


def test_kill_before_the_append_reaches_the_server_loses_nothing(ds, cl):
    jid = submit([seed(0)])
    docs = [doc([spec(10 + i, i) for i in range(3)])]

    def die(ref, items):
        raise KeyboardInterrupt

    cl.before_append = die
    with pytest.raises(KeyboardInterrupt):
        push(cl, jid, docs)
    assert len(jobs.read_job(jid)["items"]) == 1  # all or nothing
    cl.before_append = None
    assert push(cl, jid, docs)["added"] == 3


# ---- reorder ----------------------------------------------------------------------------------------
def test_search_cards_are_re_dealt_by_rank_inside_their_slots_others_keep_theirs(ds, cl):
    jid = submit([seed(0, "R"), seed(1, "A"), seed(2, "G")])
    push(cl, jid, [doc([spec(10, 0, hop=1, margin=0.5), spec(11, 1, hop=2)], window="wa", digest="da")])
    # a later window whose best card outranks everything appended so far
    push(cl, jid, [doc([spec(20, 0, hop=1, margin=0.1), spec(21, 1, hop=3)], window="wb", digest="db")])
    items = jobs.read_job(jid)["items"]
    assert [it["frame_index"] for it in items[:3]] == [0, 1, 2]  # R / A / G never move
    assert [it["frame_index"] for it in items[3:]] == [20, 10, 11, 21]  # rank, then hop, margin


def test_answered_cards_never_move(ds, cl):
    jid = submit([seed(0), seed(1)])
    d = ds / "ball_checks"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{TC}.jsonl").write_text(json.dumps({"clip": TC, "frame_index": 0, "box_index": 0, "source": "list", "verdict": "ball"}) + "\n")
    push(cl, jid, [doc([spec(10, 0), spec(11, 1)])])
    assert [it["frame_index"] for it in jobs.read_job(jid)["items"]][:2] == [0, 1]


def test_search_order_is_a_permutation_and_a_noop_without_search_cards():
    un = [(5, {"sampling": {"bucket": "U"}, "frame_index": 0}), (6, {"sampling": {"bucket": "R"}, "frame_index": 1})]
    assert A.search_order(un) == [5, 6]


def test_a_job_that_is_done_is_left_alone(ds, cl):
    jid = submit([seed(0)])
    jobs._set_status(jid, "cancelled", by="test")
    out = push(cl, jid, [doc([spec(10, 0)])])
    assert out["added"] == 0 and "skipped" in out["append"]
    assert len(jobs.read_job(jid)["items"]) == 1


def test_dry_run_writes_nothing(ds, cl):
    jid = submit([seed(0)])
    out = push(cl, jid, [doc([spec(10, 0)])], dry_run=True)
    assert out["would_add"] == 1 and out["added"] == 0 and len(jobs.read_job(jid)["items"]) == 1
