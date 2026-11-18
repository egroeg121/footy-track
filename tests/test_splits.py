"""The split authority (footy_track.splits): tiers, stickiness, monotone growth, overrides,
fail-closed, atomic + concurrent register writes, name mapping and the contamination check."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from footy_track import match_splits as ms
from footy_track import splits as sp
from footy_track.clip_id import canonical_clip_id  # noqa: F401

MATCHES = {
    "arsenal_mancity": {
        "split": "train",
        "aliases": ["mancity", "arsenal_mancity_20250925"],
    },
    "arsenal_nottinghamforest": {
        "split": "heldout",
        "aliases": ["arsenal_norwich", "nottinghamforest"],
    },
    "chelsea-vs-hull-city-2026-27": {
        "split": "train",
        "aliases": ["chelsea_vs_hull_city_2026_27"],
    },
}


@pytest.fixture(autouse=True)
def regs(tmp_path, monkeypatch):
    mpath = tmp_path / "match_splits.json"
    mpath.write_text(json.dumps(MATCHES))
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(mpath))
    monkeypatch.delenv("FOOTY_CLIP_SPLITS", raising=False)
    monkeypatch.delenv(
        "FOOTY_MATCH_SPLITS_ALLOW_SEED"
    )  # exercise the real fail-closed paths
    monkeypatch.setenv(
        "FOOTY_FULL_MATCHES_MANIFEST", str(tmp_path / "no-manifest.jsonl")
    )
    sp.clear_cache()
    yield tmp_path
    sp.clear_cache()


def clips(match: str, n: int) -> list[str]:
    return [f"{match}_seg{i:03d}" for i in range(n)]


def make_register(tmp_path: Path, **kw) -> Path:
    p = sp.clip_register_path()
    sp.register_clips(kw.get("ids", []), p, dry_run=False)
    return p


# ---- tiers -----------------------------------------------------------------------------------
def test_clip_register_is_a_sibling_of_the_match_register(regs):
    assert sp.clip_register_path() == regs / "clip_splits.json"


def test_three_tiers(regs):
    make_register(regs)
    assert (
        sp.split_of("arsenal_nottinghamforest", "arsenal_nottinghamforest_seg001")
        == sp.EVAL_HELDOUT
    )
    assert (
        sp.split_of(None, "arsenal_norwich_seg012") == sp.EVAL_HELDOUT
    )  # alias of a heldout match
    ids = clips("chelsea-vs-hull-city-2026-27", 200)
    got = {sp.split_of("chelsea-vs-hull-city-2026-27", c) for c in ids}
    assert got == {sp.TRAIN, sp.EVAL_IN_MATCH}
    # match-level query: the match is in the pool
    assert sp.split_of("chelsea-vs-hull-city-2026-27") == sp.TRAIN
    assert sp.split_of("arsenal_nottinghamforest") == sp.EVAL_HELDOUT


def test_rate_is_about_one_in_five():
    ids = [f"m_seg{i:04d}" for i in range(5000)]
    n = sum(sp.hash_clip_split(c) == sp.CLIP_EVAL for c in ids)
    assert 0.17 < n / len(ids) < 0.23
    assert sp.IN_MATCH_MODULUS == 5


def test_hash_is_on_the_canonical_clip_id(regs):
    """All aliases of one physical clip share a split (and one register entry)."""
    a, b = "arsenal_mancity_20250925_seg010", "mancity_seg010"
    assert sp.hash_clip_split(a) == sp.hash_clip_split(b)
    make_register(regs)
    plan = sp.register_clips([a, b], dry_run=True)
    assert list(plan) == ["arsenal_mancity_seg010"]


def test_unregistered_match_is_train_pool_but_flagged(regs):
    make_register(regs)
    assert sp.unregistered_match(None, "arsenal_example")
    assert sp.split_of(None, "arsenal_example") in (sp.TRAIN, sp.EVAL_IN_MATCH)
    assert not sp.unregistered_match(None, "arsenal_mancity_seg001")


def test_longest_alias_prefix_wins(regs):
    assert (
        sp.match_key_of("chelsea_vs_hull_city_2026_27_seg004")
        == "chelsea-vs-hull-city-2026-27"
    )
    assert sp.match_key_of("mancity_seg010") == "arsenal_mancity"
    assert sp.match_key_of("nobody_seg001") is None
    # boundary: a longer name is not a prefix match
    assert sp.match_key_of("arsenal_mancityX_seg001") is None


# ---- stickiness / monotone growth ------------------------------------------------------------
def test_assignment_is_sticky(regs):
    ids = clips("arsenal_mancity", 100)
    first = sp.register_clips(ids, dry_run=False)
    assert len(first) == 100
    snapshot = json.loads(sp.clip_register_path().read_text())
    # the second run adds nothing and rewrites nothing
    assert sp.register_clips(ids, dry_run=False) == {}
    assert json.loads(sp.clip_register_path().read_text()) == snapshot


def test_rule_in_the_register_beats_the_code_constant(regs, monkeypatch):
    """Changing the code constant must not reshuffle: the register carries its own rule."""
    ids = clips("arsenal_mancity", 100)
    sp.register_clips(ids, dry_run=False)
    before = {c: sp.split_of(None, c) for c in ids}
    monkeypatch.setattr(sp, "IN_MATCH_MODULUS", 2)
    sp.clear_cache()
    assert {c: sp.split_of(None, c) for c in ids} == before
    # and unregistered clips follow the register's rule (5), not the constant
    new = "arsenal_mancity_seg999"
    rule = sp.load_clip_register()["rule"]
    assert rule["modulus"] == 5
    assert sp.clip_tier(new) == sp.hash_clip_split(new, 5)


def test_monotone_growth_clips_and_matches(regs):
    a = clips("arsenal_mancity", 60)
    sp.register_clips(a, dry_run=False)
    old = sp.eval_set(a)
    old_all = {c: sp.split_of(None, c) for c in a}
    # add clips of existing matches, a new train match and a new heldout match
    mpath = ms.register_path()
    reg = json.loads(mpath.read_text())
    reg["new-vs-new-2026-27"] = {"split": "heldout", "aliases": ["new_vs_new_2026_27"]}
    reg["other-vs-other-2026-27"] = {
        "split": "train",
        "aliases": ["other_vs_other_2026_27"],
    }
    mpath.write_text(json.dumps(reg))
    sp.clear_cache()
    b = (
        a
        + clips("arsenal_mancity", 90)[60:]
        + clips("other_vs_other_2026_27", 40)
        + clips("new_vs_new_2026_27", 10)
    )
    sp.register_clips(b, dry_run=False)
    assert {c: sp.split_of(None, c) for c in a} == old_all  # nothing moved
    new = sp.eval_set(b)
    assert set(old) <= set(new)
    assert all(new[c] == s for c, s in old.items())
    assert all(
        sp.split_of(None, c) == sp.EVAL_HELDOUT for c in clips("new_vs_new_2026_27", 10)
    )
    # heldout clips are decided by their match and never enter the clip register
    assert not any(c.startswith("new_vs_new") for c in sp.load_clip_register()["clips"])


def test_adding_clips_never_moves_others_unlike_a_positional_pick(regs):
    make_register(regs)
    base = clips("arsenal_mancity", 50)
    before = {c: sp.clip_tier(c) for c in base}
    extra = [f"arsenal_mancity_seg{i:03d}" for i in range(50, 120)]
    after = {c: sp.clip_tier(c) for c in base + extra}
    assert all(after[c] == before[c] for c in base)


# ---- override precedence ---------------------------------------------------------------------
def test_manual_clip_entry_wins_and_is_never_overwritten(regs):
    cid = "arsenal_mancity_seg005"
    auto = sp.hash_clip_split(cid)
    forced = sp.CLIP_TRAIN if auto == sp.CLIP_EVAL else sp.CLIP_EVAL
    p = sp.clip_register_path()
    doc = {
        "version": 1,
        "rule": {"modulus": 5, "eval_remainder": 0},
        "clips": {
            cid: {"split": forced, "match": "arsenal_mancity", "reason": "manual"}
        },
    }
    p.write_text(json.dumps(doc))
    sp.clear_cache()
    assert sp.clip_tier(cid) == forced
    assert sp.register_clips([cid], dry_run=False) == {}
    assert json.loads(p.read_text())["clips"][cid]["reason"] == "manual"


def test_heldout_match_beats_a_clip_entry(regs):
    cid = "arsenal_nottinghamforest_seg001"
    p = sp.clip_register_path()
    p.write_text(
        json.dumps({"rule": {"modulus": 5}, "clips": {cid: {"split": "train"}}})
    )
    sp.clear_cache()
    assert sp.split_of(None, cid) == sp.EVAL_HELDOUT


def test_manual_match_override_flips_clip_to_heldout(regs):
    cid = "arsenal_mancity_seg001"
    make_register(regs)
    assert sp.split_of(None, cid) != sp.EVAL_HELDOUT
    mpath = ms.register_path()
    reg = json.loads(mpath.read_text())
    reg["arsenal_mancity"]["split"] = "heldout"
    mpath.write_text(json.dumps(reg))
    sp.clear_cache()
    assert sp.split_of(None, cid) == sp.EVAL_HELDOUT


# ---- fail closed -----------------------------------------------------------------------------
def test_missing_clip_register_fails_closed(regs):
    with pytest.raises(ms.RegisterError, match="clip split register not found"):
        sp.split_of(None, "arsenal_mancity_seg001")
    # a match-level or heldout answer needs no clip register
    assert sp.split_of("arsenal_nottinghamforest") == sp.EVAL_HELDOUT
    # the explicit tests/preview escape hatches
    assert sp.load_clip_register(allow_seed=True)["clips"] == {}


def test_preview_env_is_the_only_way_to_read_without_a_register(regs, monkeypatch):
    monkeypatch.setenv("FOOTY_CLIP_SPLITS_PREVIEW", "1")
    assert sp.split_of(None, "arsenal_mancity_seg001") in (sp.TRAIN, sp.EVAL_IN_MATCH)


@pytest.mark.parametrize(
    "text",
    [
        "nope",
        "[]",
        '{"clips": {}}',
        '{"rule": {"modulus": 0}, "clips": {}}',
        '{"rule": {"modulus": 5}, "clips": {"a": {"split": "val"}}}',
    ],
)
def test_invalid_clip_register_always_raises(regs, text, monkeypatch):
    sp.clip_register_path().write_text(text)
    monkeypatch.setenv("FOOTY_MATCH_SPLITS_ALLOW_SEED", "1")
    with pytest.raises(ms.RegisterError, match="invalid"):
        sp.load_clip_register()


def test_missing_match_register_fails_closed(regs, monkeypatch):
    make_register(regs)
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(regs / "absent.json"))
    sp.clear_cache()
    with pytest.raises(ms.RegisterError):
        sp.split_of(None, "arsenal_mancity_seg001")


# ---- atomic + concurrent writes --------------------------------------------------------------
def _writer(args):
    path, ids = args
    os.environ.pop("FOOTY_MATCH_SPLITS_ALLOW_SEED", None)
    return len(sp.register_clips(ids, Path(path), dry_run=False))


def test_concurrent_writers_lose_nothing(regs):
    p = sp.clip_register_path()
    batches = [
        [f"arsenal_mancity_seg{w * 1000 + i:04d}" for i in range(40)] for w in range(6)
    ]
    # overlapping batch to force contention on the same ids
    batches.append(batches[0][:20] + batches[1][:20])
    with mp.get_context("fork").Pool(4) as pool:
        pool.map(_writer, [(str(p), b) for b in batches])
    doc = json.loads(p.read_text())  # never torn
    expected = {canonical_clip_id(c) for b in batches for c in b}
    assert set(doc["clips"]) == expected
    assert [f.name for f in p.parent.iterdir() if f.name.endswith(".tmp")] == []


def test_write_is_atomic(regs, monkeypatch):
    ids = clips("arsenal_mancity", 20)
    sp.register_clips(ids, dry_run=False)
    p = sp.clip_register_path()
    good = p.read_text()

    def boom(*a, **k):
        raise OSError("disk went away mid-rename")

    monkeypatch.setattr(sp.os, "replace", boom)
    with pytest.raises(OSError):
        sp.register_clips(clips("arsenal_mancity", 40), dry_run=False)
    assert p.read_text() == good  # untouched
    assert [f.name for f in p.parent.iterdir() if f.name.endswith(".tmp")] == []


def test_dry_run_writes_nothing(regs):
    p = sp.clip_register_path()
    plan = sp.register_clips(clips("arsenal_mancity", 10), p, dry_run=True)
    assert len(plan) == 10 and not p.exists()


def test_cli_assign_defaults_to_dry_run(regs, capsys):
    root = regs / "fd"
    (root / "clips").mkdir(parents=True)
    for i in range(10):
        (root / "clips" / f"arsenal_mancity_20250925_seg{i:03d}.mp4").write_bytes(b"")
    assert sp.main(["assign", "--data-root", str(root)]) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert not sp.clip_register_path().exists()
    assert sp.main(["assign", "--data-root", str(root), "--apply"]) == 0
    assert len(sp.load_clip_register()["clips"]) == 10
    assert sp.main(["stats", "--data-root", str(root)]) == 0
    assert "tracking" in capsys.readouterr().out


# ---- names, Roboflow mapping, contamination --------------------------------------------------
@pytest.mark.parametrize(
    "fname,match,clip",
    [
        (
            "astonvilla_seg080_000029_jpg.rf.3eff4710f55aef83f587c21ffd025e73.jpg",
            None,
            "arsenal_astonvilla_seg080",
        ),
        (
            "arsenal_mancity_20250925_000472_png.rf.2fc88d470adac256787b21e1b0f38de7.jpg",
            "arsenal_mancity",
            None,
        ),
        (
            "arsenal_norwich_000010_jpg.rf.abcdefabcdefabcdefabcdefabcdefab.jpg",
            "arsenal_nottinghamforest",
            None,
        ),
        (
            "arsenal_mancity_seg010_000123.jpg",
            "arsenal_mancity",
            "arsenal_mancity_seg010",
        ),
        (
            "bournemouth_seg009_000002_jpg.rf.0123456789abcdef0123456789abcdef.jpg",
            None,
            "arsenal_bournemouth_1st_seg009",
        ),
        (
            "arsenal_example_000000_jpg.rf.0123456789abcdef0123456789abcdef.jpg",
            None,
            "arsenal_example",
        ),
        ("mystery_000001.jpg", None, None),
    ],
)
def test_parse_image_name(fname, match, clip):
    ref = sp.parse_image_name(fname)
    assert ref.clip_id == clip
    if match:
        assert ref.match_key == match


def test_parse_image_name_legacy_roboflow_prefixes(regs):
    m = json.loads(ms.register_path().read_text())
    m["arsenal_astonvilla"] = {"split": "train", "aliases": ["astonvilla"]}
    ms.register_path().write_text(json.dumps(m))
    sp.clear_cache()
    ref = sp.parse_image_name(
        "Arsenal-Aston-Villa_000022_jpg.rf.13f35ca44dc730519f089580d2a182b1.jpg"
    )
    assert ref.match_key == "arsenal_astonvilla" and ref.clip_id is None


def test_broadcast_names_resolve_through_the_manifest(regs, monkeypatch):
    stem = "nfo.v.ars.full.epl.18.10.2026.fullmatchsports.com"
    man = regs / "manifest.jsonl"
    man.write_text(
        json.dumps(
            {
                "post_url": "https://x/nottingham-forest-vs-arsenal-full-match-2026-27/",
                "mp4": f"/x/{stem}.mp4",
            }
        )
        + "\n"
    )
    monkeypatch.setenv("FOOTY_FULL_MATCHES_MANIFEST", str(man))
    sp.clear_cache()
    ref = sp.parse_image_name(f"{stem}_f12345.jpg")
    assert ref.match_key == "nottingham-forest-vs-arsenal-2026-27"
    assert ref.clip_id == "nottingham-forest-vs-arsenal-2026-27_w0002"  # 12345 // 6000
    assert (
        sp.broadcast_window_id("m", 5999) == "m_w0000"
        and sp.broadcast_window_id("m", 6000) == "m_w0001"
    )


def _touch(d: Path, names: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x")


def test_contamination_report(regs, tmp_path):
    make_register(regs)
    ds = tmp_path / "ds"
    eval_clip = next(
        c for c in clips("arsenal_mancity", 300) if sp.clip_tier(c) == sp.CLIP_EVAL
    )
    train_clip = next(
        c for c in clips("arsenal_mancity", 300) if sp.clip_tier(c) == sp.CLIP_TRAIN
    )
    _touch(
        ds / "train" / "images",
        [
            f"{eval_clip}_000001_jpg.rf.{'a' * 32}.jpg",  # in-match eval -> contaminated
            f"{train_clip}_000001_jpg.rf.{'b' * 32}.jpg",  # fine
            f"arsenal_norwich_000007_jpg.rf.{'c' * 32}.jpg",  # heldout match (full-video frame)
            f"arsenal_mancity_20250925_000472_jpg.rf.{'d' * 32}.jpg",  # match known, clip unknown
            "mystery_000001.jpg",  # unmapped
        ],
    )
    _touch(
        ds / "valid" / "images", [f"{eval_clip}_000009.jpg"]
    )  # eval in valid is fine
    rep = sp.contamination(ds)
    assert rep["train_images"] == 5
    assert rep["train_contaminated"] == {sp.EVAL_HELDOUT: 1, sp.EVAL_IN_MATCH: 1}
    assert rep["train_clip_unknown"] == 1 and rep["train_unmapped"] == 1
    assert rep["by_subset"]["val"][sp.EVAL_IN_MATCH] == 1
    assert rep["contaminating_clips"][sp.EVAL_IN_MATCH] == {eval_clip: 1}
    assert "contaminated" in sp.format_contamination(rep)
    assert sp.main(["contamination", str(ds), "--strict"]) == 1


def test_contamination_cli_clean_dataset(regs, tmp_path, capsys):
    make_register(regs)
    train_clip = next(
        c for c in clips("arsenal_mancity", 300) if sp.clip_tier(c) == sp.CLIP_TRAIN
    )
    _touch(tmp_path / "d" / "train" / "Yes", [f"{train_clip}_f10.jpg"])
    assert sp.main(["contamination", str(tmp_path / "d"), "--strict"]) == 0


def test_split_maps(regs):
    make_register(regs)
    ids = [
        "arsenal_nottinghamforest_seg001",
        "arsenal_mancity_seg001",
        "arsenal_mancity_seg002",
    ]
    m = sp.split_map(ids)
    assert m["arsenal_nottinghamforest_seg001"] == sp.EVAL_HELDOUT
    legacy = sp.legacy_split_map(ids)
    assert legacy["arsenal_nottinghamforest_seg001"] == "held_out"
    assert set(legacy.values()) <= {"seen", "held_out"}
    assert sp.load_split_arg(None, ids) == legacy
    assert sp.export_dir(sp.EVAL_HELDOUT) == "test"  # test/ is the default
    assert sp.export_dir(sp.EVAL_HELDOUT, include_test=False) is None
    assert sp.export_dir(sp.EVAL_HELDOUT, heldout_as_test=False) is None  # deprecated spelling
    assert sp.export_dir("test") == "test" and sp.export_dir("eval") == "val"
    assert (
        sp.export_dir(sp.EVAL_IN_MATCH) == "val" and sp.export_dir(sp.TRAIN) == "train"
    )


# --- canonical train / eval / test partitions ----------------------------------------------
def test_partition_vocabulary_and_aliases():
    assert sp.PARTITIONS == ("train", "eval", "test")
    assert sp.PARTITION_OF_SPLIT == {
        "train": "train", "eval_in_match": "eval", "eval_heldout_match": "test",
    }
    for old, new in [
        ("eval_in_match", "eval"), ("valid", "eval"), ("val", "eval"), ("validation", "eval"),
        ("eval_heldout_match", "test"), ("heldout", "test"), ("held_out", "test"),
        ("seen", "train"), ("train", "train"), ("eval", "eval"), ("test", "test"),
    ]:
        assert sp.to_partition(old) == new
    assert sp.to_split("eval") == sp.EVAL_IN_MATCH and sp.to_split("test") == sp.EVAL_HELDOUT
    with pytest.raises(ValueError):
        sp.to_partition("bogus")
    # the old values still round-trip, so nothing stored has to change
    for s in sp.SPLITS:
        assert sp.to_split(sp.to_partition(s)) == s


def test_partition_of_matches_split_of(tmp_path, monkeypatch):
    names = ["arsenal_nottinghamforest_seg001", "bournemouth_1st_seg070", "mancity_seg010", "clip_a_seg003"]
    for n in names:
        assert sp.partition_of_name(n) == sp.PARTITION_OF_SPLIT[sp.split_of_name(n)]
        assert sp.partition_label(n) == sp.PARTITION_OF_SPLIT[sp.split_label(n)]
    assert sp.partition_of_name("arsenal_nottinghamforest_seg001") == "test"
    assert sum(sp.partition_counts(names).values()) == len(names)


def test_match_partition():
    assert sp.match_partition("arsenal_nottinghamforest") == "test"
    assert sp.match_partition("no_such_match_xyz") == "unassigned"


def test_image_partition_flags_unmapped_and_keeps_it_out_of_test():
    part, flag = sp.image_partition("totally_unknown_thing_000012_jpg.rf." + "a" * 32 + ".jpg")
    assert (part, flag) == ("unmapped", "unmapped")
    part, flag = sp.image_partition("arsenal_nottinghamforest_seg001_000010.jpg")
    assert (part, flag) == ("test", None)
    # a known train-pool match with no clip in the name: train, flagged (may be in-match eval)
    part, flag = sp.image_partition("arsenal_mancity_20250925_000472_jpg.rf." + "b" * 32 + ".jpg")
    assert part in ("train", "test") and (part != "train" or flag == "clip_unknown")
