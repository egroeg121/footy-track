"""Anchor replay harness (design task 13): synthetic fixtures only, no real data."""

from __future__ import annotations

import json

import pytest

from footy_track.ball_mining import anchor_replay as R  # noqa: N812
from footy_track.ball_mining import anchor_search as A  # noqa: N812

CLIP = "fx_clip"


def easy(n=100, stride=10):
    """A straight, slow ball with one correct candidate per frame (conf 0.9)."""
    gt = {f: (0.3 + 0.001 * f, 0.5) for f in range(0, n, stride)}
    cs = A.CandidateSet("fixture", "fx000000")
    for f in range(n):
        cs.by_frame[(CLIP, f)] = [A.Cand(0.3 + 0.001 * f, 0.5, 0.01, 0.02, 0.9)]
    return gt, cs


def test_perfect_candidates_few_taps():
    gt, cs = easy()
    cfg = R.ReplayConfig(hop_cap=100, bridge_stride=5, window_frames=1000)
    w = R.simulate_clip(CLIP, gt, cs, cfg, synthetic=True)
    row = w.row()
    assert row["unreachable"] == 0 and row["wrong_anchors"] == 0
    assert row["taps_to_90"] is not None and row["taps_to_90"] <= 2  # one confirmed seed covers the straight path
    assert row["taps_to_90"] < row["naive_taps_90"]


def test_forbidden_top_candidate_is_rejected_then_next_found():
    gt, cs = easy(50)
    spare = A.Cand(0.8, 0.9, 0.01, 0.02, 0.99)  # parked spare ball outranks the real one everywhere
    for k in list(cs.by_frame):
        cs.by_frame[k] = A._sorted([*cs.by_frame[k], spare])  # noqa: SLF001
    cfg = R.ReplayConfig(policy="requeue", hop_cap=100, bridge_stride=0, window_frames=1000)
    row = R.simulate_clip(CLIP, gt, cs, cfg, synthetic=True).row()
    assert row["taps_to_90"] is not None and row["wrong_anchors"] == 0
    assert row["seed_taps"] >= 1  # the first seed card is the spare: oracle says not_ball


def test_missing_candidates_fall_back_to_hand_and_count_unreachable():
    gt, cs = easy(50)
    for f in (0, 10, 20, 30, 40):  # no candidate on any labelled frame
        cs.by_frame[(CLIP, f)] = []
    cfg = R.ReplayConfig(bridge_stride=0, window_frames=1000)
    row = R.simulate_clip(CLIP, gt, cs, cfg, synthetic=True).row()
    assert row["unreachable"] == 5 and row["reachable"] == 0
    assert row["hand_taps"] >= 5 and row["taps_to_90"] == 5  # 90% of 5 frames = 5 hand placements


def test_noisy_oracle_counts_wrong_anchors():
    total = 0
    for seed in range(5):  # false candidates exist, so a flipped "not_ball" -> "ball" is a wrong anchor
        gt, cs = R.synthetic_clip("synth", seed=seed)
        cfg = R.ReplayConfig(error_rate=0.5, target=1.0, max_taps=200, seed=seed)
        total += R.simulate_clip("synth", gt, cs, cfg, synthetic=True).row()["wrong_anchors"]
    assert total > 0
    gt, cs = R.synthetic_clip("synth", seed=0)
    assert R.simulate_clip("synth", gt, cs, R.ReplayConfig(max_taps=200), synthetic=True).row()["wrong_anchors"] == 0


def test_synthetic_clip_is_deterministic_and_harness_runs():
    a = R.synthetic_clip("synth", seed=4)
    b = R.synthetic_clip("synth", seed=4)
    assert a[0] == b[0] and a[1].by_frame == b[1].by_frame
    gt, cs = a
    for pol in R.POLICIES:
        for order in R.ORDERS:
            row = R.simulate_clip("synth", gt, cs, R.ReplayConfig(policy=pol, order=order, max_taps=300), synthetic=True).row()
            assert row["total_taps"] <= 300 and row["gt_frames"] == len(gt)


def test_train_only(monkeypatch):
    monkeypatch.setattr(A, "clip_split", lambda c: "eval_heldout_match" if c.startswith("held") else "train")
    gt, cs = easy(30)
    R.simulate_clip(CLIP, gt, cs, R.ReplayConfig(bridge_stride=0))  # train: fine
    with pytest.raises(R.ReplayError, match="train clips only"):
        R.simulate_clip("held_clip", gt, cs, R.ReplayConfig(bridge_stride=0))


def test_load_gt_merges_aliases_and_ignores_machine_rows(tmp_path):
    def row(f, x, tags):
        return json.dumps({"frame_index": f, "bbox": {"x": x, "y": 0.5, "w": 0.01, "h": 0.02}, "tags": tags})

    (tmp_path / "mancity_seg010.jsonl").write_text(row(1, 0.3, ["labeller", "ball"]) + "\n" + row(2, 0.4, ["rtdetr", "ball"]) + "\n")
    (tmp_path / "arsenal_mancity_seg010.jsonl").write_text(row(1, 0.9, ["labeller", "ball"]) + "\n" + row(3, 0.5, ["labeller", "ball"]) + "\n")
    gt, info = R.load_gt([tmp_path])
    (clip, frames), = [(c, sorted(v)) for c, v in gt.items()]
    assert clip == "arsenal_mancity_seg010" and frames == [1, 3]  # machine-tagged row 2 is not human GT
    assert info["alias_conflicts"] == 1


def test_load_machine_candidates_by_alias(tmp_path):
    rows = [{"frame_index": f, "bbox": {"x": 0.1, "y": 0.2, "w": 0.01, "h": 0.02}, "tags": ["ball", "rtdetr"], "confidence": c}
            for f, c in [(0, 0.8), (0, 0.05), (1, 0.4)]]
    (tmp_path / "arsenal_mancity_20250925_seg010.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cs = R.load_machine_candidates("arsenal_mancity_seg010", [tmp_path])
    assert [len(cs.get("arsenal_mancity_seg010", f)) for f in (0, 1)] == [1, 1]  # conf < 0.1 dropped
    assert R.load_machine_candidates("arsenal_mancity_seg999", [tmp_path]) is None


def test_summarize_baseline_verdict():
    rows = [{"taps_to_90": 5, "wrong_anchors": 0}, {"taps_to_90": 9, "wrong_anchors": 1}]
    s = R.summarize(rows)
    assert s["mean_taps_to_90"] == 7 and s["beats_baseline"] is True and s["wrong_anchors"] == 1
    assert R.summarize([{"taps_to_90": 30, "wrong_anchors": 0}])["beats_baseline"] is False
    assert R.summarize([{"taps_to_90": None, "wrong_anchors": 0}])["beats_baseline"] is False
