"""Tests for the ball-tracking measurement harness (synthetic data, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from footy_track.ball_eval.detection_replay import (
    HUMAN,
    PROPAGATED,
    CandidateCache,
    Counts,
    cache_from_detections_jsonl,
    load_gt,
    pool,
    read_cache,
    replay,
    score_clip,
    write_cache,
)
from footy_track.ball_trackers.detection_tracker import (
    DETECTED,
    INTERPOLATED,
    NONE,
    BallCandidate,
    BallPoint,
)
from footy_track.ball_trackers.factory import make_ball_tracker

W, H = 1000, 500


def c(cx: float, cy: float, conf: float = 0.9) -> BallCandidate:
    return BallCandidate((cx - 5) / W, (cy - 5) / H, 10 / W, 10 / H, conf)


def pt(f: int, cx: float, cy: float, source: str = DETECTED) -> BallPoint:
    return BallPoint(f, source, 1, (cx - 5) / W, (cy - 5) / H, 10 / W, 10 / H, 0.9)


def row(
    f: int, tag: str, prov: str | None, cx: float = 0.5, cy: float = 0.5, *extra: str
):
    tags = [tag] + ([prov] if prov else []) + list(extra)
    centre = {"x": cx, "y": cy} if tag not in ("no_ball", "not_broadcast") else None
    return {"frame_index": f, "bbox": None, "center": centre, "tags": tags}


def write_gt(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_gt_provenance_tiers_and_markers(tmp_path):
    write_gt(
        tmp_path / "mancity_seg031.jsonl",
        [
            row(1, "in_play_ball", "labeller"),
            row(2, "in_play_ball", "vittrack", 0.1, 0.1, "human_checked"),
            row(3, "in_play_ball", "yolo"),
            row(4, "in_play_ball", "vittrack"),  # not human checked
            row(5, "no_ball", None),
            row(6, "not_broadcast", None),
            row(7, "player", "labeller"),
            {
                "frame_index": 8,
                "bbox": None,
                "center": None,
                "tags": ["in_play_ball", "labeller"],
            },
        ],
    )
    gt = load_gt(tmp_path)["arsenal_mancity_seg031"]
    assert list(gt.marks[HUMAN]) == [1]
    assert list(gt.marks[PROPAGATED]) == [2]
    assert gt.ignored_machine_marks == 2
    assert gt.no_ball == {5} and gt.not_broadcast == {6}
    ball, no_ball = gt.scored((HUMAN,))
    assert set(ball) == {1} and no_ball == {5}
    ball, _ = gt.scored((HUMAN, PROPAGATED))
    assert set(ball) == {1, 2}


def test_aliases_of_one_clip_are_merged(tmp_path):
    write_gt(tmp_path / "mancity_seg010.jsonl", [row(1, "in_play_ball", "labeller")])
    write_gt(
        tmp_path / "arsenal_mancity_seg010.jsonl", [row(9, "in_play_ball", "labeller")]
    )
    gts = load_gt(tmp_path)
    assert list(gts) == ["arsenal_mancity_seg010"]
    assert set(gts["arsenal_mancity_seg010"].marks[HUMAN]) == {1, 9}


def test_array_centre_format(tmp_path):
    (tmp_path / "x.jsonl").write_text(
        json.dumps(
            {
                "frame_index": 0,
                "center": [0.2, 0.3],
                "tags": ["in_play_ball", "labeller"],
            }
        )
        + "\n"
    )
    assert load_gt(tmp_path)["x"].marks[HUMAN][0] == [(0.2, 0.3)]


def _gt(tmp_path, rows):
    write_gt(tmp_path / "clip.jsonl", rows)
    return load_gt(tmp_path)["clip"]


def test_hit_wrong_miss_false_and_unscored_frames(tmp_path):
    gt = _gt(
        tmp_path,
        [
            row(0, "in_play_ball", "labeller", 0.5, 0.5),  # centre (500, 250)
            row(1, "in_play_ball", "labeller", 0.5, 0.5),
            row(2, "in_play_ball", "labeller", 0.5, 0.5),
            row(3, "in_play_ball", "labeller", 0.5, 0.5),
            row(4, "no_ball", None),
            row(5, "no_ball", None),
            row(6, "not_broadcast", None),
        ],
    )
    points = [
        pt(0, 505, 250),  # 5 px: hit at both radii
        pt(1, 520, 250),  # 20 px: hit at 25 only
        pt(2, 700, 250, INTERPOLATED),  # wrong (interpolated)
        BallPoint(3),  # miss
        pt(4, 100, 100),  # false
        BallPoint(5),  # correct "no ball"
        pt(6, 1, 1),  # not scored
        pt(7, 1, 1),  # no GT row: not scored
    ]
    cache = CandidateCache(
        "clip", (W, H), [(0, [c(505, 250, 0.2)]), (1, []), (2, [c(900, 10)]), (3, [])]
    )
    out = score_clip(points, cache, gt, radii=(10, 25))
    a, b = out[10], out[25]
    assert (a.ball_frames, a.hit, a.wrong, a.miss) == (4, 1, 2, 1)
    assert (b.hit, b.wrong, b.miss) == (2, 1, 1)
    assert (b.wrong_interp, b.wrong_detected, b.hit_detected) == (1, 0, 2)
    assert (b.no_ball_frames, b.false, b.false_detected) == (2, 1, 1)
    # detector upper bound: any candidate within R of the mark
    assert (a.det_any, b.det_any) == (1, 1)
    assert (a.det_strong, b.det_strong) == (
        0,
        0,
    )  # the only near candidate had conf 0.2
    r = b.ratios()
    assert r["precision"] == 2 / (2 + 1 + 1)
    assert r["recall"] == 2 / 4
    assert r["wrong_per_prediction"] == 1 / 3


def test_multiple_marks_in_a_frame_use_the_nearest(tmp_path):
    gt = _gt(
        tmp_path,
        [
            row(0, "in_play_ball", "labeller", 0.1, 0.1),
            row(0, "in_play_ball", "labeller", 0.5, 0.5),
        ],
    )
    cache = CandidateCache("clip", (W, H), [(0, [])])
    out = score_clip([pt(0, 500, 250)], cache, gt, radii=(10,))
    assert out[10].hit == 1 and out[10].wrong == 0


def test_pool_adds_counts():
    a, b = Counts(ball_frames=2, hit_detected=1), Counts(ball_frames=3, hit_interp=2)
    pooled = pool({"x": {25.0: a}, "y": {25.0: b}}, ["x", "y", "missing"])
    assert pooled[25.0].ball_frames == 5 and pooled[25.0].hit == 3


def test_cache_roundtrip_and_replay(tmp_path):
    frames = [(f, [c(100 + 10 * f, 200, 0.9)]) for f in range(5)] + [(5, [])]
    cache = CandidateCache("clip", (W, H), frames, {"weights": "w"})
    write_cache(tmp_path / "clip.jsonl", cache)
    back = read_cache(tmp_path / "clip.jsonl")
    assert (
        back.clip == "clip"
        and back.frame_size == (W, H)
        and back.meta["weights"] == "w"
    )
    assert [f for f, _ in back.frames] == list(range(6))
    assert back.frames[5][1] == []
    assert abs(back.frames[2][1][0].confidence - 0.9) < 1e-6
    points = replay(make_ball_tracker("online", (W, H)), back)
    assert [p.source for p in points[:5]] == [DETECTED] * 5
    assert points[5].source == NONE


def test_cache_from_detections_jsonl_keeps_empty_frames(tmp_path):
    rows = [
        {"frame_index": 0, "bbox": {"x": 0.1, "y": 0.2, "w": 0.01, "h": 0.02},
         "tags": ["in_play_ball", "rtdetr"], "confidence": 0.7},
        {"frame_index": 0, "bbox": {"x": 0.5, "y": 0.5, "w": 0.05, "h": 0.1},
         "tags": ["player", "rtdetr"], "confidence": 0.9},
        {"frame_index": 1, "bbox": {"x": 0.5, "y": 0.5, "w": 0.05, "h": 0.1},
         "tags": ["player", "rtdetr"], "confidence": 0.9},
    ]  # fmt: skip
    p = tmp_path / "detections.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cache = cache_from_detections_jsonl(p, (W, H))
    assert [(f, len(cs)) for f, cs in cache.frames] == [(0, 1), (1, 0)]
