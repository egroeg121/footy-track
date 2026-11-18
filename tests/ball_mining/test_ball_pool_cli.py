"""``scripts/ball_pool.py``: the stage CLI end to end on synthetic candidates (no detector, no GPU): pool -> score -> status."""

from __future__ import annotations

import json

import pytest

from footy_track import splits
from footy_track.ball_mining import candidates as C  # noqa: N812
from footy_track.ball_mining import schema
from footy_track.ball_mining import scorer as S  # noqa: N812
from footy_track.scripts import ball_pool as cli

from .test_pool import synthetic_clip
from .test_scorer import split_of, synth_table


@pytest.fixture
def world(tmp_path, monkeypatch, capsys):
    """Match ``m1`` (train) with finished candidates for detector sha ``abcd1234`` and a saved scorer; ``hm`` is heldout."""
    monkeypatch.setattr(
        splits, "unregistered_match", lambda m, c=None: m not in ("m1", "hm")
    )
    monkeypatch.setattr(
        splits,
        "split_of",
        lambda m, c=None: "eval_heldout_match" if m == "hm" else "train",
    )
    root = tmp_path / "fd"
    cdir = root / "ball_pool" / "m1" / "candidates.abcd1234"
    cdir.mkdir(parents=True)
    C.save_clip(
        synthetic_clip([0.9, 0.85, 0.9, None, 0.8, 0.9]), cdir / "m1_seg000.npz"
    )
    (cdir / C.META).write_text(
        json.dumps(
            {
                "complete": True,
                "expected": ["m1_seg000"],
                "stride": 5,
                "clips": {"m1_seg000": {}},
                "detector": {"sha8": "abcd1234"},
            }
        )
    )
    sc, _ = S.train(synth_table(), split_of=split_of)
    S.save(sc, tmp_path / "scorers")

    def run(*args):
        capsys.readouterr()
        rc = cli.main(["--root", str(root), *args])
        out = capsys.readouterr().out.strip().splitlines()
        return rc, json.loads(out[-1])

    return run, root, tmp_path / "scorers", sc.version


def test_heldout_and_unregistered_matches_are_refused(world):
    run, *_ = world
    for m in ("hm", "ghost"):
        with pytest.raises(SystemExit, match="refusing|heldout"):
            run("pool", "--match", m, "--detector-sha8", "abcd1234")


def test_pool_then_score_writes_a_valid_pool_the_miner_can_read(world):
    run, root, scorers, version = world
    _, pooled = run("pool", "--match", "m1", "--detector-sha8", "abcd1234")
    assert (
        pooled["status"] == "built"
        and pooled["gap_nodes"] == 1
        and pooled["motion"] is False
    )
    assert (
        run("pool", "--match", "m1")[1]["status"] == "exists"
    )  # detector found from the single candidates dir; motion absent but not staged either
    _, scored = run("score", "--match", "m1", "--scorer-root", str(scorers))
    assert (
        scored["status"] == "scored"
        and scored["scorer_version"] == version
        and scored["n_gap_nodes"] == 1
    )
    path = root / "ball_pool" / "m1" / f"pool.{version}.parquet"
    pool = schema.read_pool(path)  # validates against the miner's schema
    assert (pool["scorer_version"] == version).all() and (
        pool["detector_sha"] == "abcd1234"
    ).all()
    gap = pool[pool["gap_frame"].eq(True)]
    assert (
        len(gap) == 1
        and gap["sources"].iloc[0] == ["tracker_gap"]
        and 0 < gap["p_real"].iloc[0] <= 1
    )
    assert pool["p_real"].between(0, 1).all()
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["detector_sha8"] == "abcd1234" and meta["scorer_version"] == version
    assert (
        run("score", "--match", "m1", "--scorer-root", str(scorers))[1]["status"]
        == "exists"
    )
    assert (
        run("score", "--match", "m1", "--scorer-root", str(scorers), "--force")[1][
            "status"
        ]
        == "scored"
    )


def test_a_pool_made_from_another_detector_is_rewritten_not_skipped(world):
    run, root, scorers, version = world
    run("pool", "--match", "m1", "--detector-sha8", "abcd1234")
    run("score", "--match", "m1", "--scorer-root", str(scorers))
    meta = root / "ball_pool" / "m1" / f"pool.{version}.json"
    d = json.loads(meta.read_text())
    meta.write_text(json.dumps({**d, "detector_sha8": "ffffffff"}))
    assert (
        run("score", "--match", "m1", "--scorer-root", str(scorers))[1]["status"]
        == "scored"
    )
    assert json.loads(meta.read_text())["detector_sha8"] == "abcd1234"


def test_score_needs_nodes_and_a_scorer(world):
    run, _, scorers, _ = world
    with pytest.raises(SystemExit, match="run the pool stage"):
        run("score", "--match", "m1", "--scorer-root", str(scorers))
    with pytest.raises(FileNotFoundError):
        run("score", "--match", "m1", "--scorer-root", str(scorers / "nowhere"))


def test_status_reports_every_stage(world):
    run, root, scorers, version = world
    seg = root / "m1" / "split_video_broadcast_frames"
    seg.mkdir(parents=True)
    (seg / "segments.json").write_text(json.dumps({"fps": 50, "segments": []}))
    _, st = run("status", "--match", "m1", "--scorer-root", str(scorers))
    m = st["matches"][0]
    assert (
        st["scorer"] == version
        and m["candidates"]
        == {"abcd1234": {"complete": True, "clips": 1, "expected": 1}}
        and m["motion"] == {}
        and not m["current_pool"]
    )
    run("pool", "--match", "m1", "--detector-sha8", "abcd1234")
    run("score", "--match", "m1", "--scorer-root", str(scorers))
    assert (
        run("status", "--match", "m1", "--scorer-root", str(scorers))[1]["matches"][0][
            "current_pool"
        ]
        is True
    )
    _, allm = run("status", "--all", "--scorer-root", str(scorers))
    assert [x["match"] for x in allm["matches"]] == ["m1"]


def test_probe_frames_lists_every_n_frames_and_gap_frames(world):
    run, root, *_ = world
    run("pool", "--match", "m1", "--detector-sha8", "abcd1234")
    _, res = run(
        "probe-frames", "--match", "m1", "--detector-sha8", "abcd1234", "--every", "10"
    )
    doc = json.loads(
        (root / "ball_pool" / "m1" / "sam3_probe_frames.abcd1234.json").read_text()
    )
    frames = doc["frames"]["m1_seg000"]
    assert {0, 10, 20}.issubset(
        frames
    ) and 15 in frames  # every 10th sample, plus the gap sample at frame 15
    assert res["n_gap_frames"] == 1 and doc["every"] == 10


def test_detector_sha_needs_a_hint_when_ambiguous(world):
    run, root, *_ = world
    (root / "ball_pool" / "m1" / "candidates.11112222").mkdir()
    with pytest.raises(SystemExit, match="cannot tell which detector"):
        run("pool", "--match", "m1")
