"""Guard: export / eval code must not grow its own train/eval split logic.

Two layers. A lint-style scan of the source for the patterns the old, drifting implementations
used, and behavioural checks that each consumer really asks :mod:`footy_track.splits` (stub the
authority; the consumer's output must follow the stub).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from footy_track import splits as authority

SRC = Path(authority.__file__).parent

#: Patterns that mean "a script is deciding the split itself".
FORBIDDEN = {
    r"\b_pick_spread\b": "position-based clip picking (reshuffles when a clip is added)",
    r"\b_split_clips(_three_way)?\b": "own by-clip splitter",
    r"\bval_fraction\b|\btest_fraction\b": "fraction-based splitting",
    r"json\.loads\(\s*args\.split\b": "hand-made split.json (use splits.load_split_arg)",
    r"\bhash\w*\(.*%\s*[0-9]+\s*==": "ad-hoc hash-modulo split",
}
#: Files allowed to contain split logic: the authority and the older register it builds on.
ALLOWED = {"splits.py", "match_splits.py"}


def _py_files():
    return [p for p in SRC.rglob("*.py") if p.name not in ALLOWED]


@pytest.mark.parametrize("pattern", sorted(FORBIDDEN))
def test_no_private_split_logic(pattern):
    rx = re.compile(pattern)
    hits = [
        f"{p.relative_to(SRC)}:{i + 1}: {line.strip()}"
        for p in _py_files()
        for i, line in enumerate(p.read_text().splitlines())
        if rx.search(line) and not line.lstrip().startswith("#")
    ]
    assert not hits, (
        f"{FORBIDDEN[pattern]}: route through footy_track.splits\n" + "\n".join(hits)
    )


def test_consumers_import_the_authority():
    for rel in (
        "scripts/export_training_dataset.py",
        "scripts/export_broadcast_review.py",
        "scripts/ball_tracking_eval.py",
        "calibration/prelabel/evaluate.py",
        "calibration/prelabel/sweep.py",
        "scripts/projection_eval.py",
        "feature_store/ingest_gt.py",
        "feature_store/ingest_calibration.py",
        "feature_store/ball_checks.py",
    ):
        text = (SRC / rel).read_text()
        assert re.search(r"from footy_track(\.splits| import splits)", text), rel


@pytest.fixture
def stub_authority(monkeypatch, tmp_path):
    """Make every clip an eval-in-match clip via the authority; return the call log."""
    calls: list[str] = []

    def fake(name):
        calls.append(name)
        return authority.EVAL_IN_MATCH

    monkeypatch.setattr(authority, "split_of_name", fake)
    return calls


def test_training_export_routes_through_the_authority(stub_authority):
    from footy_track.scripts import export_training_dataset as etd

    train, val, test, _, held = etd._split_by_clip_identity(["a_seg001", "b_seg002"])
    assert (train, test, held) == (set(), set(), set())
    assert val == {"a_seg001", "b_seg002"}
    assert stub_authority  # asked, not computed


def test_broadcast_export_routes_through_the_authority(monkeypatch, tmp_path):
    from footy_track.labeller import broadcast_review as br
    from footy_track.scripts import export_broadcast_review as exp

    seen = []
    monkeypatch.setattr(
        authority,
        "split_of",
        lambda key, clip=None: seen.append((key, clip)) or authority.EVAL_IN_MATCH,
    )
    monkeypatch.setattr(authority, "match_key_of", lambda name: "key")

    class DS:
        marks_dir = tmp_path
        clips_dir = tmp_path

    (tmp_path / "broadcast_review").mkdir()
    (tmp_path / "broadcast_review" / "m.json").write_text(
        json.dumps({"frames": {"7": {"verdict": "yes"}}})
    )
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "f7.jpg").write_bytes(b"x")
    monkeypatch.setattr(br, "frame_path", lambda ds, m, f: tmp_path / m / f"f{f}.jpg")
    exp.export(DS, tmp_path / "out")
    assert seen == [("key", "key_w0000")]
    assert (tmp_path / "out" / "valid" / "Yes" / "m_f7.jpg").is_file()


def test_ball_tracking_eval_routes_through_the_authority(stub_authority):
    from footy_track.scripts import ball_tracking_eval as bte

    groups, labels = bte.split_groups(["x_seg001", "y_seg002"])
    assert set(labels.values()) == {authority.EVAL_IN_MATCH}
    assert groups[authority.EVAL_IN_MATCH] == ["x_seg001", "y_seg002"]
    assert groups[authority.TRAIN] == []
    # the legacy json is an explicit override and keeps the old vocabulary
    legacy, _ = bte.split_groups(["x_seg001"], {"x_seg001": "held_out"})
    assert legacy["held-out"] == ["x_seg001"]


def test_calibration_eval_reports_the_authority_split(monkeypatch):
    from footy_track.calibration.prelabel import evaluate as ev

    monkeypatch.setattr(ev, "split_label", lambda clip: "eval_in_match")
    out = ev._by_split(
        [{"split": "eval_in_match", "clip": "c", "mean_px": 3.0, "status": "ok"}], 10.0
    )
    assert out["eval_in_match"]["n_success"] == 1
