"""The round's own labeller dataset: registration, clip linking, gold clip resolution and naming, and the verdict join
(plan-time exclusion, refit / gold check, feature-store ingest) across ``ball_train_v1`` and the older datasets."""

from __future__ import annotations

import json
import os
import stat

import pytest

from footy_track.ball_mining import miner, refit, traindata
from footy_track.ball_mining import round as rnd
from footy_track.ball_mining.clips import read_verdict_frames
from footy_track.ball_mining.gold import load_gold_multi
from footy_track.feature_store import ingest_all
from footy_track.feature_store.store import FeatureStore
from footy_track.labeller import datasets, jobs
from tests.ball_mining.conftest import synthetic_big_pool

DAY = "2026-11-18"
NAME = traindata.TRAIN_DATASET
BOX = {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.02}


class FakeClient:
    def __init__(self):
        self.submitted = []

    def list_jobs(self, **kw):
        return []

    def get(self, ref):
        return {"id": ref, "items": []}

    def submit(self, title, kind, items, **kw):
        self.submitted.append({"title": title, "kind": kind, "items": items, **kw})
        return {"id": "abc", "status": "open", "created": True}


@pytest.fixture
def splits_patched(monkeypatch, splitter):
    monkeypatch.setattr(miner, "default_split_of", splitter)
    monkeypatch.setattr(miner, "default_eval_clips", lambda: [])
    monkeypatch.setattr(rnd, "default_match_of", lambda c: c.split("_seg")[0])
    return splitter


def touch(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A footy_data root: pool segment mp4s of m1..m3, a ``calib`` dataset (GT sidecar ``mancity_seg010`` whose video is the
    date-stamped ``arsenal_mancity_20250925_seg010.mp4`` of its extra clips dir), and a registry that does not list the
    training dataset yet."""
    root = tmp_path / "footy_data"
    for clip in sorted(set(synthetic_big_pool()["clip"])):
        touch(root / clip.split("_seg")[0] / traindata.SEGMENTS_DIR / f"{clip}.mp4")
    calib_clips, calib_extra, calib_marks = (root / "eval_clips", root / "ballcheck_clips", root / "ball_gt_marks")
    calib_clips.mkdir(parents=True)
    touch(calib_extra / "arsenal_mancity_20250925_seg010.mp4")
    calib_marks.mkdir(parents=True)
    rows = [{"frame_index": 4000 + i * 200, "bbox": BOX, "tags": ["ball", "labeller"]} for i in range(40)]
    (calib_marks / "mancity_seg010.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    reg = root / "labeller_datasets.json"
    datasets.write_registry(reg, "calib", {"calib": {"label": "Calib", "clips_dir": str(calib_clips), "marks_dir": str(calib_marks),
                                                     "extra_clips_dirs": [str(calib_extra)]}})  # fmt: skip
    reg.chmod(0o664)
    monkeypatch.setenv("FOOTY_LABELLER_DATASETS", str(reg))
    datasets.invalidate()
    yield root
    datasets.invalidate()


def cfg(root, **kw):
    return rnd.RoundConfig(data_root=str(root), **kw)


# ------------------------------------------------------------------ registration


def test_register_adds_the_dataset_once_and_keeps_the_rest_of_the_registry(env):
    reg = env / "labeller_datasets.json"
    before = json.loads(reg.read_text())
    assert traindata.register_dataset(env) is True
    assert traindata.register_dataset(env) is False  # idempotent: nothing rewritten
    doc = json.loads(reg.read_text())
    assert doc["default"] == "calib" and doc["datasets"]["calib"] == before["datasets"]["calib"]
    spec = doc["datasets"][NAME]
    assert spec["clips_dir"] == str(env / "ball_train" / "clips")
    assert spec["marks_dir"] == str(env / f"ball_gt_marks_{NAME}")
    assert "ingest" not in spec  # these verdicts are training data: ingest_all must read them
    assert (env / "ball_train" / "clips").is_dir() and (env / f"ball_gt_marks_{NAME}").is_dir()
    assert stat.S_IMODE(os.stat(reg).st_mode) == 0o664  # the atomic rewrite keeps the registry's mode
    ds = datasets.get(NAME)  # the labeller re-reads it on mtime
    assert ds.clips_dir == env / "ball_train" / "clips" and not ds.is_default


def test_register_never_overwrites_an_existing_entry_and_needs_a_registry(env, tmp_path):
    assert datasets.register(NAME, {"clips_dir": "/a", "marks_dir": "/b"}) is True
    assert datasets.register(NAME, {"clips_dir": "/other", "marks_dir": "/other"}) is False
    assert datasets.get(NAME).clips_dir.as_posix() == "/a"
    with pytest.raises(FileNotFoundError):
        datasets.register(NAME, {"clips_dir": "/a", "marks_dir": "/b"}, path=tmp_path / "missing.json")
    with pytest.raises(ValueError):
        datasets.register("bad name", {"clips_dir": "/a", "marks_dir": "/b"})


def test_ingest_all_reads_the_training_dataset_marks_and_its_verdicts(env):
    traindata.register_dataset(env)
    marks = env / f"ball_gt_marks_{NAME}"
    (marks / "ball_checks").mkdir()
    rec = {"clip": "m1_seg001", "frame_index": 100, "box_index": 2**30 + 5, "source": "list", "verdict": "not_ball", "bbox": BOX, "ts": 1.0}
    (marks / "ball_checks" / "m1_seg001.jsonl").write_text(json.dumps(rec) + "\n")
    found, _ = ingest_all.discover_sources(env)
    assert marks.resolve() in found  # not "ingest": false, unlike ball_eval_v1
    rep = ingest_all.ingest_all(FeatureStore.open(":memory:"), data_root=env)
    assert rep.ball_check_clips == 1 and rep.ball_check_verdicts == 1  # recorded in detection_review even without a detection


# ------------------------------------------------------------------ linking


def pool_card(clip="m1_seg001", frame=100):
    return {"clip": clip, "frame_index": frame, "bbox": BOX, "match": clip.split("_seg")[0], "sampling": {"bucket": "U"}}


def gold_card(clip="mancity_seg010", frame=4000):
    return {"clip": clip, "frame_index": frame, "bbox": BOX, "match": "mancity", "sampling": {"bucket": "G"}}


def test_pool_cards_link_to_their_segment_mp4s_and_it_is_idempotent(env, tmp_path):
    dst = tmp_path / "clips"
    cards = [pool_card("m1_seg001", 100), pool_card("m1_seg001", 900), pool_card("m2_seg003", 50)]
    out, rep = traindata.link_cards(cards, dst, data_root=env)
    assert [c["clip"] for c in out] == ["m1_seg001", "m1_seg001", "m2_seg003"]
    assert rep["clips"] == 2 and rep["linked_new"] == 2
    link = dst / "m1_seg001.mp4"
    assert link.is_symlink() and link.resolve() == (env / "m1" / traindata.SEGMENTS_DIR / "m1_seg001.mp4").resolve()
    out2, rep2 = traindata.link_cards(cards, dst, data_root=env)
    assert out2 == out and rep2["linked_new"] == 0 and rep2["already_linked"] == 2
    assert cards[0]["clip"] == "m1_seg001"  # inputs are not modified


def test_gold_cards_resolve_from_the_gt_dataset_and_are_named_canonically(env, tmp_path):
    dst = tmp_path / "clips"
    resolver = traindata.ClipResolver(traindata.gold_dataset_dirs(["calib"]))
    out, rep = traindata.link_cards([gold_card(), pool_card()], dst, data_root=env, resolver=resolver)
    g, p = out
    assert g["clip"] == "arsenal_mancity_seg010" and g["source_clip"] == "mancity_seg010"  # sidecar alias -> canonical
    assert "source_clip" not in p and rep["gold_renamed"] == 1
    link = dst / "arsenal_mancity_seg010.mp4"  # the card's clip name IS the linked file
    assert link.resolve() == (env / "ballcheck_clips" / "arsenal_mancity_20250925_seg010.mp4").resolve()
    # every alias of the clip lands on the same name, so the verdicts share one ball_checks file
    for alias in ("arsenal_mancity_20250925_seg010", "mancity_seg010", "arsenal_mancity_seg010"):
        o, _ = traindata.link_cards([gold_card(alias)], dst, data_root=env, resolver=resolver)
        assert o[0]["clip"] == "arsenal_mancity_seg010"


def test_a_gold_clip_already_in_the_dataset_is_not_linked_to_itself(env, tmp_path):
    dst = tmp_path / "clips"
    traindata.link_cards([pool_card("m1_seg001")], dst, data_root=env)
    resolver = traindata.ClipResolver([dst])  # a not_ball verdict of an earlier round: the clip is a pool clip
    out, rep = traindata.link_cards([gold_card("m1_seg001")], dst, data_root=env, resolver=resolver)
    assert out[0]["clip"] == "m1_seg001" and rep["linked_new"] == 0
    assert (dst / "m1_seg001.mp4").resolve() == (env / "m1" / traindata.SEGMENTS_DIR / "m1_seg001.mp4").resolve()


def test_unresolvable_clips_fail_before_anything_is_linked_and_conflicts_are_refused(env, tmp_path):
    dst = tmp_path / "clips"
    with pytest.raises(traindata.ClipLinkError, match="no video for 1"):
        traindata.link_cards([pool_card("m1_seg001"), pool_card("m9_seg001")], dst, data_root=env)
    assert not dst.exists() or not list(dst.iterdir())
    dst.mkdir()
    touch(dst / "m1_seg001.mp4", b"another video")
    with pytest.raises(traindata.ClipLinkError, match="not a link"):
        traindata.link_cards([pool_card("m1_seg001")], dst, data_root=env)
    (dst / "m1_seg001.mp4").unlink()
    (dst / "m1_seg001.mp4").symlink_to(dst / "gone.mp4")  # dangling
    with pytest.raises(traindata.ClipLinkError, match="not a link"):
        traindata.link_cards([pool_card("m1_seg001")], dst, data_root=env)


# ------------------------------------------------------------------ create_round end to end


def test_create_registers_links_and_submits_cards_that_resolve(env, splits_patched):
    client = FakeClient()
    out = rnd.create_round(big(), cfg(env, budget=200, gold_datasets=("calib",)), client=client, today=DAY)
    assert out["created"] and out["links"]["linked_new"] > 0
    ds = datasets.get(NAME)  # registered by create_round, before the submit
    assert client.submitted[0]["dataset"] == NAME
    cards = client.submitted[0]["items"]
    gold = [c for c in cards if c["sampling"]["bucket"] == "G"]
    assert gold and all(c["clip"] == "arsenal_mancity_seg010" and c["source_clip"] == "mancity_seg010" for c in gold)
    for c in cards:
        assert (ds.clips_dir / f"{c['clip']}.mp4").is_file(), c["clip"]
    assert (ds.marks_dir / "ball_mining" / "rounds" / f"ball-round_{DAY}_1.json").is_file()  # the round log is in the new marks
    again = rnd.create_round(big(), cfg(env, budget=200), client=FakeClient(), today=DAY, force=True)
    assert again["links"]["linked_new"] == 0  # linking is idempotent


def big():
    return synthetic_big_pool()


def test_dry_run_links_and_returns_the_cards_but_submits_nothing(env, splits_patched):
    client = FakeClient()
    out = rnd.create_round(big(), cfg(env, budget=60, use_gold=False), client=client, today=DAY, dry_run=True)
    assert out["dry_run"] and client.submitted == [] and out["links"]["clips"] > 0
    assert all((datasets.get(NAME).clips_dir / f"{c['clip']}.mp4").is_file() for c in out["cards"])


def test_a_gold_clip_with_no_video_is_left_out_of_the_plan(env, splits_patched):
    (env / "ballcheck_clips" / "arsenal_mancity_20250925_seg010.mp4").unlink()
    res = rnd.plan_round(big(), cfg(env, budget=200))
    assert res.report["gold_unresolved_clips"] == ["mancity_seg010"]
    assert not [c for c in res.cards if c["sampling"]["bucket"] == "G"]


def test_plan_before_the_dataset_exists_uses_the_path_create_will_register(env, splits_patched):
    assert NAME not in datasets.names()
    res = rnd.plan_round(big(), cfg(env, budget=60, use_gold=False))
    assert res.cards and NAME not in datasets.names()  # plan registers nothing
    assert rnd.marks_dir(NAME, env) == env / f"ball_gt_marks_{NAME}"


def test_the_default_round_dataset_is_the_training_dataset():
    assert rnd.DEFAULT_DATASET == "ball_train_v1" == rnd.RoundConfig().dataset


# ------------------------------------------------------------------ verdict join


def verdict(marks, clip, frame, box_index, v="ball", ts=1.0):
    d = marks / "ball_checks"
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{clip}.jsonl").open("a") as fh:
        fh.write(json.dumps({"clip": clip, "frame_index": frame, "box_index": box_index, "source": "list", "verdict": v, "bbox": BOX, "ts": ts}) + "\n")


def test_verdicts_in_other_datasets_on_the_same_canonical_clip_are_not_asked_again(env, splits_patched):
    first = rnd.plan_round(big(), cfg(env, budget=200, use_gold=False))
    c = next(c for c in first.cards if c["sampling"]["bucket"] != "G")
    # an old Ball Check verdict in the calib marks, under an alias stem of the clip the pool names
    calib_marks = datasets.get("calib").marks_dir
    stem = c["clip"]
    verdict(calib_marks, stem, c["frame_index"], 7)
    assert (stem, c["frame_index"]) in read_verdict_frames(calib_marks)
    again = rnd.plan_round(big(), cfg(env, budget=200, use_gold=False))
    assert (c["clip"], c["frame_index"] // 50) not in {(x["clip"], x["frame_index"] // 50) for x in again.cards}
    # a verdict in the round dataset's own marks (and a tombstoned one, which does not count) behaves the same
    own = rnd.marks_dir(NAME, env)
    verdict(own, "m3_seg000", 150, 9)
    verdict(own, "m3_seg000", 150, 9, v=None, ts=2.0)
    assert ("m3_seg000", 150) not in read_verdict_frames(own)


def test_gold_is_read_from_where_the_gt_lives_and_from_the_round_dataset(env):
    traindata.register_dataset(env)
    own = datasets.get(NAME).marks_dir
    verdict(own, "arsenal_mancity_seg010", 4400, 2**30 + 1, v="not_ball")  # a gold card's verdict in the new marks
    g = load_gold_multi([own, datasets.get("calib").marks_dir], match_of=lambda c: c)
    assert set(g["kind"]) == {"gt_shifted", "known_not_ball"}
    assert set(g[g.kind == "gt_shifted"]["clip"]) == {"mancity_seg010"}  # the sidecar stem
    # the same physical frame in the new marks and the calib sidecar is one row per kind
    verdict(own, "arsenal_mancity_seg010", 4000, 2**30 + 2, v="not_ball")
    g = load_gold_multi([own, datasets.get("calib").marks_dir], match_of=lambda c: c)
    assert len(g) == len(g.drop_duplicates(["frame", "kind"]))


def test_refit_and_the_gold_check_join_verdicts_in_the_new_marks_dir(env, splits_patched, api):
    client = FakeClient()
    rnd.create_round(big(), cfg(env, budget=200), client=client, today=DAY)
    cards = client.submitted[0]["items"]
    job, _ = jobs.submit({"title": "r", "kind": "ball_check", "dataset": NAME, "items": cards, "key": f"ball-round/{DAY}/1", "max_cards": 2000}, NAME)
    own = datasets.get(NAME).marks_dir
    stored = jobs.read_job(job["id"])["items"]
    gold_pos = [i for i, c in enumerate(cards) if c["sampling"]["bucket"] == "G"]
    assert len(gold_pos) >= 5
    for pos in gold_pos:  # the labeller accepts every planted-wrong gold card: the round must be flagged
        it = stored[pos]
        verdict(own, it["clip"], it["frame_index"], it["box_index"], v="ball")
    out = refit.refit_round(job["id"], client=api, reorder=False)
    state = refit.latest_state(own)  # written to the NEW marks dir, found by the dataset name
    assert state is not None and state["job_id"] == job["id"]
    assert out["gold"]["n"] == len(gold_pos) and out["gold"]["accepted_wrong"] == len(gold_pos) and out["gold"]["flagged"]
    assert refit.resolve_marks(NAME, None) == own
    rows = refit.collect_rows(api, NAME)
    assert {r["clip"] for r in rows if r["gold_kind"]} == {"arsenal_mancity_seg010"}  # joined on the linked clip name


def test_two_alias_files_of_one_gold_clip_share_one_link_across_cards_and_rounds(env, tmp_path):
    clips = env / "eval_clips"  # the real calib clips dir holds both ``mancity_seg010.mp4`` and ``arsenal_mancity_seg010.mp4``
    touch(clips / "mancity_seg010.mp4", b"alias one")
    touch(clips / "arsenal_mancity_seg010.mp4", b"alias two")
    dst = tmp_path / "clips"
    resolver = traindata.ClipResolver(traindata.gold_dataset_dirs(["calib"]))
    cards = [gold_card("mancity_seg010", 4000), gold_card("arsenal_mancity_seg010", 4200)]
    out, rep = traindata.link_cards(cards, dst, data_root=env, resolver=resolver)
    assert {c["clip"] for c in out} == {"arsenal_mancity_seg010"} and rep["clips"] == 1
    target = os.path.realpath(dst / "arsenal_mancity_seg010.mp4")
    # a later round meets the other alias first: it keeps the link an earlier round made
    out2, rep2 = traindata.link_cards(cards[::-1], dst, data_root=env, resolver=resolver)
    assert rep2["linked_new"] == 0 and os.path.realpath(dst / "arsenal_mancity_seg010.mp4") == target


# ------------------------------------------------------------------ spaced pool clip names (jobs._clip rule)

SPACED = {
    "arsenal_astonvilla": ("Arsenal - Aston Villa_seg123", "arsenal_astonvilla_seg123"),
    "arsenal_bournmouth_1st_half": ("Bournemouth vs Arsenal 1_seg096", "arsenal_bournemouth_1st_seg096"),
}


def spaced_env(env):
    for match, (clip, _) in SPACED.items():
        touch(env / match / traindata.SEGMENTS_DIR / f"{clip}.mp4")
    return env


def spaced_pool():
    from tests.ball_mining.conftest import make_pool

    rows = [{"match": m, "clip": c, "frame": 50 * i + k, "p_real": 0.3 + 0.1 * k} for m, (c, _) in SPACED.items() for i in range(40) for k in range(3)]
    rows += [{"match": "m1", "clip": "m1_seg001", "frame": 50 * i, "p_real": 0.5} for i in range(40)]
    return make_pool(rows)


def test_link_name_keeps_safe_clips_and_slugs_unsafe_ones_deterministically():
    assert traindata.link_name("m1_seg001") == "m1_seg001"
    assert traindata.link_name("arsenal_nottinghamforest_seg003") == "arsenal_nottinghamforest_seg003"
    assert traindata.link_name("Arsenal - Aston Villa_seg123") == "arsenal_astonvilla_seg123"
    assert traindata.link_name("Bournemouth vs Arsenal 1_seg096") == "arsenal_bournemouth_1st_seg096"
    assert traindata.link_name("Some Other  Match (2)_seg005") == "Some_Other_Match_2_seg005"  # no rule: sanitised, collapsed


def test_spaced_pool_clips_link_under_a_safe_name_and_keep_pool_clip(env, tmp_path):
    spaced_env(env)
    dst = tmp_path / "clips"
    cards = [pool_card(c, 10) | {"match": m} for m, (c, _) in SPACED.items()] + [pool_card("m1_seg001")]
    out, rep = traindata.link_cards(cards, dst, data_root=env)
    for card, (m, (clip, safe)) in zip(out, SPACED.items(), strict=False):
        assert card["clip"] == safe and card["pool_clip"] == clip
        jobs._clip(card, 0)  # the labeller job API's rule
        assert (dst / f"{safe}.mp4").resolve() == (env / m / traindata.SEGMENTS_DIR / f"{clip}.mp4").resolve()
    assert out[2]["clip"] == "m1_seg001" and "pool_clip" not in out[2]  # safe clips keep their name
    again, rep2 = traindata.link_cards(cards, dst, data_root=env)
    assert again == out and rep2["linked_new"] == 0


def test_two_sources_for_one_safe_name_are_refused(env, tmp_path):
    spaced_env(env)
    touch(env / "arsenal_astonvilla" / traindata.SEGMENTS_DIR / "arsenal_astonvilla_seg123.mp4", b"another video")
    cards = [pool_card("Arsenal - Aston Villa_seg123") | {"match": "arsenal_astonvilla"}]
    cards.append(pool_card("arsenal_astonvilla_seg123") | {"match": "arsenal_astonvilla"})
    with pytest.raises(traindata.ClipLinkError, match="two different videos"):
        traindata.link_cards(cards, tmp_path / "clips", data_root=env)


def test_a_verdict_under_the_safe_name_joins_the_pool_clip(env, tmp_path):
    """Exclusion / auto-accept / eval-buffer joins are keyed by canonical clip id: safe name and pool clip are one id."""
    from footy_track.clip_id import canonical_clip_id

    for _, (clip, safe) in SPACED.items():
        assert canonical_clip_id(safe) == canonical_clip_id(clip)
    assert canonical_clip_id("Some Other Match_seg005") == canonical_clip_id("Some_Other_Match_seg005")
    marks = tmp_path / "marks"
    (marks / "ball_checks").mkdir(parents=True)
    rec = {"clip": "arsenal_astonvilla_seg123", "frame_index": 100, "box_index": 1, "source": "list", "verdict": "not_ball", "bbox": BOX, "ts": 1.0}
    (marks / "ball_checks" / "arsenal_astonvilla_seg123.jsonl").write_text(json.dumps(rec) + "\n")
    assert (canonical_clip_id("Arsenal - Aston Villa_seg123"), 100) in read_verdict_frames(marks)


def test_refit_rows_report_the_pool_clip():
    doc = {"job": {"id": "j"}, "labels": [{"clip": "arsenal_astonvilla_seg123", "frame_index": 5, "verdict": "ball",
           "meta": {"pool_clip": "Arsenal - Aston Villa_seg123", "match": "arsenal_astonvilla", "p_real": 0.4, "sampling": {"regime": "train", "bucket": "R"}}}]}  # fmt: skip
    (row,) = refit.rows_from_results(doc)
    assert row["clip"] == "Arsenal - Aston Villa_seg123" and row["bucket"] == "R"


def test_create_round_submits_valid_clip_names_for_a_spaced_pool_and_excludes_by_verdict(env, splits_patched):
    spaced_env(env)
    client = FakeClient()
    pool = spaced_pool()
    out = rnd.create_round(pool, cfg(env, budget=60, use_gold=False), client=client, today=DAY, dry_run=True)
    cards = out["cards"]
    assert any("pool_clip" in c for c in cards)
    ds = datasets.get(NAME)
    for c in cards:
        jobs._clip(c, 0)
        assert (ds.clips_dir / f"{c['clip']}.mp4").is_file()
        if "pool_clip" in c:
            assert c["pool_clip"] in {v[0] for v in SPACED.values()}
    # a verdict filed under the safe name keeps that frame out of the next plan
    first = next(c for c in cards if "pool_clip" in c)
    marks = ds.marks_dir / "ball_checks"
    marks.mkdir(parents=True, exist_ok=True)
    rec = {"clip": first["clip"], "frame_index": first["frame_index"], "box_index": 1, "source": "list", "verdict": "not_ball", "bbox": BOX, "ts": 1.0}
    (marks / f"{first['clip']}.jsonl").write_text(json.dumps(rec) + "\n")
    out2 = rnd.create_round(pool, cfg(env, budget=60, use_gold=False), client=client, today=DAY, dry_run=True)
    assert not any(c["pool_clip" if "pool_clip" in c else "clip"] == first["pool_clip"] and c["frame_index"] == first["frame_index"] for c in out2["cards"])
