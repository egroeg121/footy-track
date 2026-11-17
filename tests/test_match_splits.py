import json

import pytest

from footy_track import match_splits as ms


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(tmp_path / "match_splits.json"))
    ms.clear_cache()
    yield
    ms.clear_cache()


def test_missing_file_falls_back_to_seed():
    assert ms.is_heldout_clip("arsenal_nottinghamforest_seg012.mp4")
    assert ms.is_heldout_clip("arsenal_norwich_seg012")
    assert not ms.is_heldout_clip("arsenal_mancity_20250925_seg012")
    assert ms.split_for_match("arsenal_nottinghamforest") == "heldout"
    assert ms.split_for_match("arsenal_mancity") == "unassigned"


def test_register_file_wins(tmp_path):
    (tmp_path / "match_splits.json").write_text(
        json.dumps(
            {
                "chelsea-vs-x-2026-27": {
                    "split": "heldout",
                    "aliases": ["chelsea_vs_x_2026_27"],
                },
                "arsenal_mancity": {"split": "train", "aliases": ["mancity"]},
            }
        )
    )
    assert ms.is_heldout_clip("chelsea_vs_x_2026_27_seg003")
    assert not ms.is_heldout_clip("chelsea_vs_x_2026_270_seg003")  # boundary
    assert not ms.is_heldout_clip(
        "arsenal_nottinghamforest_seg012"
    )  # file replaces seed
    assert ms.split_for_match("x", ["mancity"]) == "train"
    assert ms.split_for_match("chelsea-vs-x-2026-27") == "heldout"


def test_missing_file_fails_closed(monkeypatch):
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED")
    ms.clear_cache()
    with pytest.raises(ms.RegisterError, match="not found"):
        ms.is_heldout_clip("arsenal_nottinghamforest_seg001")
    assert ms.load_register(allow_seed=True) == ms.DEFAULT_REGISTER


@pytest.mark.parametrize("text", ["not json", "[]", '{"a": {"split": "bogus"}}'])
def test_invalid_file_always_raises(tmp_path, text):
    (tmp_path / "match_splits.json").write_text(text)
    with pytest.raises(ms.RegisterError, match="invalid"):
        ms.is_heldout_clip("arsenal_nottinghamforest_seg001")
    with pytest.raises(ms.RegisterError):
        ms.load_register(allow_seed=True)


def test_default_split_deterministic_and_about_quarter():
    assert ms.default_split("a-vs-b-2026-27") == ms.default_split("a-vs-b-2026-27")
    n = sum(
        ms.default_split(f"team{i}-vs-other-2026-27") == "heldout" for i in range(400)
    )
    assert 60 < n < 140
