# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408, B007, SIM105
"""Snapshot / pluggable-GT / tracker-scoring tests on a synthetic eval set."""

from __future__ import annotations

import json
import stat

import numpy as np
import pytest

from footy_track.ball_offline import baselines as bl
from footy_track.ball_offline import cli
from footy_track.ball_offline import evaluate as E
from footy_track.ball_offline import score as S
from footy_track.ball_offline.data import MarksDirGT
from footy_track.ball_offline.metrics import score_clip
from footy_track.ball_offline.snapshot import (
    create_snapshot,
    load_snapshot,
    resolve_snapshot,
)
from footy_track.ball_offline.sources import eval_set

W, H = 1920.0, 1080.0
N = 60


def _row(t, cx, cy, tags, conf=None, w=0.006, h=0.011):
    r = {
        "frame_index": t,
        "bbox": {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h},
        "center": {"x": cx, "y": cy},
        "tags": tags,
    }
    if conf is not None:
        r["confidence"] = conf
    return json.dumps(r)


def _pos(t):
    return (0.2 + 0.005 * t, 0.5)


@pytest.fixture
def world(tmp_path):
    """Two clips.  ``arsenal_norwich_seg001``: GT frames 10..19 (labeller) + 30 (no_ball);
    ``arsenal_norwich_seg002``: GT frames 5..9, all of them near a 'training' image."""
    gt = tmp_path / "ball_gt_marks_norwich"
    cand = tmp_path / "ball_candidates_norwich"
    gt.mkdir()
    cand.mkdir()
    (cand / "_manifest.json").write_text(
        json.dumps(
            {"clips": {f"arsenal_norwich_seg00{i}": {"frames": N} for i in (1, 2)}}
        )
    )
    for seg in ("1", "2"):
        rows = []
        for t in range(N):
            x, y = _pos(t)
            rows.append(_row(t, x, y, ["in_play_ball", "rtdetr"], 0.9))
            rows.append(
                _row(t, 0.9, 0.1, ["in_play_ball", "rtdetr"], 0.35)
            )  # distractor
        (cand / f"arsenal_norwich_seg00{seg}.jsonl").write_text("\n".join(rows) + "\n")
    g1 = [_row(t, *_pos(t), ["in_play_ball", "labeller"]) for t in range(10, 20)]
    g1.append(
        json.dumps(
            {"frame_index": 30, "bbox": None, "center": None, "tags": ["no_ball"]}
        )
    )
    g1.append(
        json.dumps(
            {"frame_index": 40, "bbox": None, "center": None, "tags": ["not_broadcast"]}
        )
    )
    g1.append(_row(25, 0.5, 0.5, ["in_play_ball", "yolo"]))  # machine-only: not GT
    g1.append(_row(26, 0.5, 0.5, ["player", "yolo"]))
    (gt / "arsenal_norwich_seg001.jsonl").write_text("\n".join(g1) + "\n")
    g2 = [
        _row(t, *_pos(t), ["in_play_ball", "vittrack", "human_checked"])
        for t in range(5, 10)
    ]
    (gt / "arsenal_norwich_seg002.jsonl").write_text("\n".join(g2) + "\n")
    spec = eval_set("norwich", root=tmp_path)
    rf = {("11", "train"): {"arsenal_norwich_seg002": {7}}}
    return tmp_path, spec, rf


def test_marks_dir_gt_reads_labeller_sidecars(world):
    _, spec, _ = world
    src = MarksDirGT(spec.gt_dir)
    assert src.clip_ids() == ["arsenal_norwich_seg001", "arsenal_norwich_seg002"]
    gt, no_ball, aliases, conflicts = src.load("arsenal_norwich_seg001")
    assert sorted(gt) == list(range(10, 20))
    assert gt[10].tier == "human" and gt[10].xy == pytest.approx((0.250 * W, 0.5 * H))
    assert no_ball == {30}  # not_broadcast and machine-only rows ignored
    assert aliases == ["arsenal_norwich_seg001"] and conflicts == 0
    gt2, *_ = src.load("arsenal_norwich_seg002")
    assert {g.tier for g in gt2.values()} == {"vittrack_checked"}


def test_snapshot_roles_contamination_and_hash(world, tmp_path):
    _, spec, rf = world
    out = tmp_path / "snaps"
    snap = create_snapshot(spec, out, rf=rf)
    assert (
        snap.path.name
        == f"norwich-{snap.manifest['created_utc'][:10].replace('-', '')}-{snap.short}"
    )
    assert snap.clips["arsenal_norwich_seg001"].role == "test"
    # seg002: frame 7 +- 5 covers every GT frame -> majority contaminated -> dev
    assert snap.clips["arsenal_norwich_seg002"].role == "dev"
    assert (
        snap.manifest["n_test_clean_ball"] == 10
        and snap.manifest["n_test_no_ball"] == 1
    )
    # frozen files are read-only
    assert not (snap.path / "manifest.json").stat().st_mode & stat.S_IWUSR
    # idempotent: same labels -> same snapshot, nothing new written
    again = create_snapshot(spec, out, rf=rf)
    assert again.path == snap.path and len(list(out.iterdir())) == 1
    # a label change gives a new hash
    p = spec.gt_dir / "arsenal_norwich_seg001.jsonl"
    p.write_text(
        p.read_text() + _row(21, *_pos(21), ["in_play_ball", "labeller"]) + "\n"
    )
    changed = create_snapshot(spec, out, rf=rf)
    assert changed.hash != snap.hash and len(list(out.iterdir())) == 2
    # but the old snapshot still scores the old labels
    assert len(load_snapshot(snap.path).clips["arsenal_norwich_seg001"].gt) == 10
    assert resolve_snapshot(snap.short, out).hash == snap.hash


def test_snapshot_refuses_tampering(world, tmp_path):
    _, spec, rf = world
    snap = create_snapshot(spec, tmp_path / "snaps", rf=rf)
    f = snap.path / "gt" / "arsenal_norwich_seg001.json"
    f.chmod(0o644)
    d = json.loads(f.read_text())
    d["gt"]["10"][0] += 50
    f.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="corrupt"):
        load_snapshot(snap.path)


def test_score_cli_and_report_records_hash(world, tmp_path, capsys):
    _, spec, rf = world
    snap = create_snapshot(spec, tmp_path / "snaps", rf=rf)
    perfect = tmp_path / "perfect.jsonl"
    rows = [
        {
            "clip": "arsenal_norwich_seg001",
            "frame_index": t,
            "x": _pos(t)[0] * W,
            "y": _pos(t)[1] * H,
        }
        for t in range(10, 20)
    ]
    rows.append(
        {"clip": "arsenal_norwich_seg001", "frame_index": 30, "x": 100, "y": 100}
    )  # on a no_ball frame
    rows.append(
        {"clip": "arsenal_norwich_seg001", "frame_index": 50, "x": 1, "y": 1}
    )  # unscored frame
    perfect.write_text("\n".join(json.dumps(r) for r in rows))
    out_json = tmp_path / "r.json"
    rc = cli.main(
        [
            "score",
            "--snapshot",
            str(snap.path),
            "--pred",
            str(perfect),
            "--out-json",
            str(out_json),
        ]
    )
    assert rc == 0
    md = capsys.readouterr().out
    assert f"snapshot norwich@{snap.short}" in md
    res = json.loads(out_json.read_text())
    assert res["snapshot"]["hash"] == snap.hash
    s = res["roles"]["test"]["pooled"]
    assert s["n_gt_ball"] == 10 and s["tp@10"] == 10
    assert s["fp@10"] == 1  # the point on the no_ball frame
    assert s["recall@10"] == 1.0 and s["precision@10"] == pytest.approx(10 / 11)


def test_predictions_formats_and_missing_clip_warning(world, tmp_path):
    _, spec, rf = world
    snap = create_snapshot(spec, tmp_path / "snaps", rf=rf)
    f = tmp_path / "p.jsonl"
    f.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {
                    "clip": "arsenal_norwich_seg001.mp4",
                    "frame_index": 10,
                    "center": {"x": 0.25, "y": 0.5},
                },
                {
                    "clip": "arsenal_norwich_seg001",
                    "frame_index": 11,
                    "x": None,
                    "y": None,
                },
                {
                    "clip": "arsenal_norwich_seg001",
                    "frame_index": 12,
                    "x": 0.255,
                    "y": 0.5,
                },
                {
                    "clip": "arsenal_norwich_seg001",
                    "frame_index": 12,
                    "x": 0.9,
                    "y": 0.9,
                    "conf": 0.9,
                },
            )
        )
    )
    lp = S.load_predictions(f, units="norm")
    assert lp.n_dropped_nan == 1 and lp.n_duplicate_frames == 1
    assert lp.preds["arsenal_norwich_seg001"][10] == pytest.approx((0.25 * W, 0.5 * H))
    assert lp.preds["arsenal_norwich_seg001"][12] == pytest.approx(
        (0.9 * W, 0.9 * H)
    )  # higher conf wins
    _, pooled, warnings = S.score_snapshot(snap, lp, "dev")
    assert any("arsenal_norwich_seg002" in w for w in warnings)
    assert pooled["n_gt_ball"] == 5 and pooled["tp@10"] == 0


def test_baselines_on_snapshot_match_scorer(world, tmp_path):
    """predict -> jsonl -> score gives the same counts as scoring the method in-process."""
    _, spec, rf = world
    snap = create_snapshot(spec, tmp_path / "snaps", rf=rf)
    evs = E.build_sets(snap, spec)
    preds = cli.run_baseline(evs, "b0")
    path = tmp_path / "b0.jsonl"
    S.write_predictions(path, preds)
    _, pooled, _ = S.score_snapshot(snap, S.load_predictions(path), "test")
    direct, _ = E.eval_method(
        [e for e in evs if e.role == "test"], lambda c: bl.b0_per_frame(c, 0.4)
    )
    assert pooled["tp@10"] == sum(v["tp@10"] for v in direct.values()) == 10


def test_paired_bootstrap_detects_a_clear_difference(world, tmp_path):
    _, spec, rf = world
    snap = create_snapshot(spec, tmp_path / "snaps", rf=rf)
    cid = "arsenal_norwich_seg001"
    good = {cid: {t: (_pos(t)[0] * W, _pos(t)[1] * H) for t in range(10, 20)}}
    bad = {
        cid: {
            t: (_pos(t)[0] * W + (50 if t % 2 else 0), _pos(t)[1] * H)
            for t in range(10, 20)
        }
    }
    mk = lambda p: S.LoadedPreds(p, 0, 0, 0, "", "x")  # noqa: E731
    a, b = S.frame_units(snap, mk(good)), S.frame_units(snap, mk(bad))
    r = S.paired_bootstrap(a, b, n_boot=2000, seed=1)
    assert r["recall_diff"]["point"] == pytest.approx(0.5)
    assert (
        r["recall_diff"]["lo"] > 0 and r["a_hit_b_miss"] == 5 and r["b_hit_a_miss"] == 0
    )
    same = S.paired_bootstrap(a, a, n_boot=500, seed=1)
    assert same["recall_diff"]["lo"] == same["recall_diff"]["hi"] == 0.0
    blk = S.paired_bootstrap(a, b, n_boot=500, block=5, seed=1)
    assert blk["n_blocks"] < r["n_blocks"]


def test_viterbi_tracks_the_line_not_the_distractor(world, tmp_path):
    _, spec, rf = world
    snap = create_snapshot(spec, tmp_path / "snaps", rf=rf)
    evs = [e for e in E.build_sets(snap, spec) if e.role == "test"]
    preds = cli.run_baseline(evs, "viterbi")
    pred = preds["arsenal_norwich_seg001"]
    assert all(
        np.hypot(pred[t][0] - _pos(t)[0] * W, pred[t][1] - _pos(t)[1] * H) < 3
        for t in range(10, 20)
    )
    per, pooled = E.eval_method(evs, bl.viterbi)
    assert pooled["tp@10"] == 10 and pooled["recall@10"] == 1.0


def test_metrics_counts_and_denominators():
    gt = {1: (100.0, 100.0), 2: (200.0, 200.0), 3: (300.0, 300.0)}
    pred = {1: (103.0, 104.0), 2: (230.0, 200.0), 5: (5.0, 5.0), 9: (1.0, 1.0)}
    s = score_clip(pred, gt, {9}, 20).summary()
    assert s["n_gt_ball"] == 3 and s["n_no_ball"] == 1
    assert s["tp@10"] == 1 and s["fp@10"] == 2  # frame 2 wrong + no_ball frame 9
    assert abs(s["recall@10"] - 1 / 3) < 1e-9
    assert abs(s["precision@10"] - 1 / 3) < 1e-9
    assert abs(s["coverage"] - 4 / 20) < 1e-9  # frame 5 (unscored) counts for coverage
    assert s["tp@5"] == 1 and s["tp@20"] == 1


def test_b0_threshold(world, tmp_path):
    _, spec, rf = world
    snap = create_snapshot(spec, tmp_path / "snaps", rf=rf)
    clip = E.build_sets(snap, spec)[0].clip
    assert set(bl.b0_per_frame(clip, 0.5)) == set(range(N))  # distractor (0.35) dropped
    pt = bl.b0_per_frame(clip, 0.1)
    assert pt[10] == pytest.approx(
        (_pos(10)[0] * W, _pos(10)[1] * H)
    )  # top-1 is the line
