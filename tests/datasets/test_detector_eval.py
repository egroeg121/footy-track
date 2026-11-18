"""Detector eval on the tracking items: matching rules, tiers, splits / cohorts, the GPU pass's
idempotency. Synthetic (no weights, no real video)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from footy_track.ball_eval.detection_replay import read_cache
from footy_track.datasets import detector_eval as de
from footy_track.datasets import detector_run as dr
from footy_track.datasets import eval_adapters as ea
from footy_track.datasets import results as res
from footy_track.datasets import snapshot as snap
from footy_track.feature_store import ingest_all as ia
from footy_track.schema import FrameDetections, ObjectDetection

W, H = 1920, 1080


# ---- helpers ---------------------------------------------------------------------------------
def gtbox(label: str, x: float, y: float, w: float = 0.02, h: float = 0.03, *, tier: int = 1, rejected: bool = False) -> dict:
    return {"label": label, "x": x, "y": y, "w": w, "h": h, "tier": tier, "rejected": rejected, "source": "hand_label"}


def item(clip: str, frame: int, boxes: list[dict], marker: str | None = None, **kw) -> dict:
    return {"item_id": f"{clip}:{frame}", "clip": clip, "frame_index": frame,
            "payload": json.dumps({"boxes": boxes, "checks": [], "marker": marker}), **kw}


def dets(clip: str, rows: list[tuple[int, str, tuple, float]], frames: list[int]) -> de.ClipDetections:
    return de.ClipDetections.from_rows(clip, (W, H), ["in_play_ball", "player", "referee"], rows, frames)


def store(tmp_path: Path, *clips: de.ClipDetections) -> de.DetectionStore:
    for c in clips:
        c.save(tmp_path / f"{c.clip}.npz")
    return de.DetectionStore(tmp_path)


# ---- matching primitives ---------------------------------------------------------------------
def test_iou_and_centre_distance():
    a = np.array([[0.1, 0.1, 0.2, 0.2]])
    assert de.iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    b = np.array([[0.2, 0.1, 0.2, 0.2]])  # shifted by half a width: IoU 1/3
    assert de.iou_matrix(a, b)[0, 0] == pytest.approx(1 / 3)
    assert de.iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)
    c = np.array([[0.1 + 10 / W, 0.1, 0.2, 0.2]])  # 10 px to the right
    assert de.centre_dist_px(a, c, (W, H))[0, 0] == pytest.approx(10.0)


def test_tally_ap_precision_recall_and_ignore_regions():
    t = de.Tally()
    gt = [(0.1, 0.1, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)]
    ignore = [(0.8, 0.8, 0.1, 0.1)]
    d = np.array([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1]])
    t.add_frame(d, np.array([0.9, 0.8, 0.7, 0.6]), gt, ignore, (W, H), radius_px=None)
    # the detection on the ignore region is dropped; the stray one is a false positive
    assert (t.n_gt, len(t.scores), sum(t.hits)) == (2, 3, 2)
    assert t.prf(0.3) == (pytest.approx(2 / 3), 1.0, pytest.approx(0.8))
    assert t.prf(0.7)[0] == 1.0
    assert t.ap() == pytest.approx(1.0, abs=0.01)  # both hits outrank the false positive (101-point rule: 0.995 at best, as Ultralytics)
    assert t.best_f1() == (pytest.approx(1.0), pytest.approx(0.8))


def test_a_duplicate_detection_is_a_false_positive_and_ap_is_ranked():
    t = de.Tally()
    gt = [(0.1, 0.1, 0.1, 0.1)]
    d = np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
    t.add_frame(d, np.array([0.9, 0.5]), gt, [], (W, H), radius_px=None)
    assert t.hits == [True, False]
    t2 = de.Tally()  # a false positive scoring higher than the hit halves the precision at full recall
    t2.add_frame(np.array([[0.1, 0.1, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]]), np.array([0.5, 0.9]), gt, [], (W, H), radius_px=None)
    assert t2.ap() == pytest.approx(0.5, abs=0.02)
    assert de.Tally().ap() is None and de.Tally(n_gt=3).ap() == 0.0


def test_ball_centre_distance_rule_vs_iou_on_a_tiny_object():
    gt = [(0.5, 0.5, 11 / W, 11 / H)]
    d = np.array([[0.5 + 6 / W, 0.5, 11 / W, 11 / H]])  # 6 px off an 11 px box: IoU 0.29, centre 6 px
    iou, c10, c25 = de.Tally(), de.Tally(), de.Tally()
    iou.add_frame(d, np.array([0.9]), gt, [], (W, H), radius_px=None)
    c10.add_frame(d, np.array([0.9]), gt, [], (W, H), radius_px=10)
    c25.add_frame(d, np.array([0.9]), gt, [], (W, H), radius_px=25)
    assert (sum(iou.hits), sum(c10.hits), sum(c25.hits)) == (0, 1, 1)


# ---- score_items: tiers, markers, classes -----------------------------------------------------
def test_score_items_classes_tiers_and_markers(tmp_path):
    ball = (0.5, 0.5, 0.01, 0.012)
    items = [
        # human ball + a confirmed (tier 2) player + an unconfirmed player (ignore region)
        item("c1", 0, [gtbox("in_play_ball", *ball[:2], *ball[2:]), gtbox("player", 0.1, 0.1, 0.05, 0.1, tier=2),
                       gtbox("player", 0.7, 0.1, 0.05, 0.1, tier=3)]),
        item("c1", 1, [gtbox("in_play_ball", 0.3, 0.3, 0.01, 0.012, tier=2)]),            # only a confirmed ball
        item("c1", 2, [], marker="no_ball"),
        item("c1", 3, [], marker="not_broadcast"),
        item("c1", 4, [gtbox("in_play_ball", 0.2, 0.2, tier=3)]),                       # machine-only: no GT at all
        item("c1", 5, [gtbox("in_play_ball", 0.4, 0.4, tier=1, rejected=True)]),        # rejected: no GT at all
        item("c9", 0, [gtbox("player", 0.1, 0.1, 0.05, 0.1)]),                          # clip without detections
    ]
    rows = [
        (0, "in_play_ball", ball, 0.9), (0, "player", (0.1, 0.1, 0.05, 0.1), 0.9),
        (0, "player", (0.7, 0.1, 0.05, 0.1), 0.8),            # on the tier-3 player: ignored
        (0, "player", (0.9, 0.9, 0.05, 0.1), 0.6),            # stray: false positive
        (1, "in_play_ball", (0.3, 0.3, 0.01, 0.012), 0.8),    # hits the tier-2 ball
        (2, "in_play_ball", (0.6, 0.6, 0.01, 0.012), 0.7),    # no_ball frame: false positive
        (3, "in_play_ball", (0.6, 0.6, 0.01, 0.012), 0.7),    # not_broadcast: never scored
    ]
    s = de.score_items(items, store(tmp_path, dets("c1", rows, list(range(6)))))
    m = s.metrics
    assert s.n_items == 3  # frames 0, 1, 2 of c1; not_broadcast / tier-3-only / rejected-only frames carry no GT
    assert m["player_n_gt"] == 1 and m["player_precision"] == pytest.approx(0.5) and m["player_recall"] == 1.0
    # tier-1 ball GT: only frame 0; no_ball frame 2 scores the false positive; the tier-2 ball on frame 1 is not GT
    assert m["ball_n_gt"] == 1 and m["ball_n_no_ball_frames"] == 1
    assert m["ball_recall_10px"] == 1.0 and m["ball_precision_10px"] == pytest.approx(0.5)
    # tier 1+2: frame 1's confirmed ball counts too
    assert m["ball_t12_n_gt"] == 2 and m["ball_t12_recall_10px"] == 1.0
    assert m["ball_t12_precision_10px"] == pytest.approx(2 / 3)
    assert m["referee_n_gt"] == 0 and m["referee_ap50"] is None and m["referee_recall"] is None
    assert m["map50"] == pytest.approx((m["player_ap50"] + m["ball_ap_iou50"]) / 2)
    assert m["clips_without_detections"] == 1.0 and m["coverage"] == pytest.approx(3 / 4)  # c9 has an eligible item but no detections


def test_unprocessed_frames_are_not_scored_but_empty_frames_are_misses(tmp_path):
    items = [item("c1", 0, [gtbox("player", 0.1, 0.1, 0.05, 0.1)]), item("c1", 1, [gtbox("player", 0.1, 0.1, 0.05, 0.1)])]
    s = de.score_items(items, store(tmp_path, dets("c1", [], [0])))  # frame 0 processed (nothing found), frame 1 not
    assert s.n_items == 1 and s.metrics["player_n_gt"] == 1
    assert s.metrics["player_recall"] == 0.0 and s.metrics["player_precision"] is None
    assert s.metrics["coverage"] == 0.5


def test_detection_store_roundtrip(tmp_path):
    c = dets("c1", [(2, "player", (0.1, 0.2, 0.3, 0.4), 0.77)], [0, 1, 2])
    c.save(tmp_path / "c1.npz")
    back = de.DetectionStore(tmp_path).get("c1")
    assert back.size == (W, H) and back.has_frame(0) and not back.has_frame(5)
    labels, boxes, conf = back.frame(2)
    assert labels == ["player"] and conf[0] == pytest.approx(0.77) and boxes[0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert back.frame(0)[0] == [] and de.DetectionStore(tmp_path).get("nope") is None
    assert not list(tmp_path.glob(".*tmp*"))  # atomic write leaves no temp file


# ---- the adapter on a real (synthetic-world) dataset version: splits, cohorts, recording -------
@pytest.fixture
def versions(world):
    ia.run(world.db, data_root=world.root, marks_dirs=[world.marks])
    snap.bump("tracking", world.db, world.ds_root)
    return world


def perfect_store(tmp_path, v: snap.DatasetVersion) -> de.DetectionStore:
    """Detections that reproduce every non-rejected ground-truth box exactly (conf 0.9)."""
    by_clip: dict[str, list] = {}
    frames: dict[str, set[int]] = {}
    for r in v.select("eval_all"):
        frames.setdefault(r["clip"], set()).add(r["frame_index"])
        for b in json.loads(r["payload"])["boxes"]:
            by_clip.setdefault(r["clip"], []).append((r["frame_index"], b["label"], (b["x"], b["y"], b["w"], b["h"]), 0.9))
    return store(tmp_path, *(dets(c, by_clip.get(c, []), sorted(f)) for c, f in frames.items()))


def test_adapter_scores_each_split_and_cohort(versions, tmp_path):
    v = snap.load_version("tracking", 1, versions.ds_root)
    model = perfect_store(tmp_path / "d", v)
    ad = ea.DetectorAdapter()
    held = ad.evaluate(model, v, "eval_heldout_match")
    inm = ad.evaluate(model, v, "eval_in_match")
    both = ad.evaluate(model, v, "eval_all")
    assert held.metrics["ball_recall_10px"] == 1.0 and held.metrics["player_recall"] == 1.0
    assert held.metrics["ball_n_gt"] == 2 and held.metrics["ball_n_no_ball_frames"] == 1  # frames 0, 1; no_ball frame 2
    assert held.metrics["player_n_gt"] == 1
    assert inm.metrics["ball_n_gt"] == 1 and inm.metrics["player_n_gt"] == 0 and inm.metrics["coverage"] == 1.0
    assert both.metrics["ball_n_gt"] == 3 and both.n_items == held.n_items + inm.n_items
    assert ad.evaluate(model, v, "eval_all", cohort=0).n_items == 0  # nothing was first seen at version 0
    with pytest.raises(ValueError):
        ad.evaluate(model, v, "nonsense")
    # a model with nothing stored scores no frames
    empty = ad.evaluate(de.DetectionStore(tmp_path / "empty"), v, "eval_all")
    assert empty.n_items == 0 and empty.metrics["coverage"] == 0.0


def test_evaluate_and_record_with_contamination_flags(versions, tmp_path):
    v = snap.load_version("tracking", 1, versions.ds_root)
    model = perfect_store(tmp_path / "d", v)
    model.name, model.version = "detector", "w1"
    # a Roboflow-style training export that contains an image of the in-match eval clip
    export = tmp_path / "rf"
    (export / "train" / "images").mkdir(parents=True)
    (export / "train" / "images" / "tmatch_seg001_000000_jpg.rf.abababababababababababab.jpg").write_bytes(b"x")
    ref = res.ModelRef("detector", "w1", training=res.TrainingData(roboflow_dir=export))
    rows = ea.evaluate_and_record(ea.DetectorAdapter(), model, v, versions.db, model_ref=ref, cohorts=[1])
    stored = res.list_results(versions.db, dataset="tracking", model="detector")
    assert len(stored) == len(rows) > 0
    by = {(r["split"], r["cohort_version"], r["metric"]): r for r in stored}
    assert by[("eval_all", None, "ball_recall_10px")]["value"] == 1.0
    assert by[("eval_all", 1, "ball_recall_10px")]["n_items"] == by[("eval_all", None, "ball_recall_10px")]["n_items"]
    assert by[("eval_in_match", None, "map50")]["contaminated"] is True
    assert by[("eval_heldout_match", None, "map50")]["contaminated"] is False


# ---- the GPU pass: video sourcing, idempotency, checkpointing ----------------------------------
class FakeDetector:
    """Finds a ball at the frame's brightest-pixel position... or, simpler, a fixed ball + player per frame."""

    classes = {0: "in_play_ball", 1: "player"}

    def __init__(self) -> None:
        self.calls = 0

    def predict_frames(self, frames):
        self.calls += 1
        out = []
        for f in frames:
            h, w = f.shape[:2]
            out.append(FrameDetections(uri=Path("x"), width=w, height=h, detections=[
                ObjectDetection(label="in_play_ball", confidence=0.8, x=0.5, y=0.5, w=0.02, h=0.03),
                ObjectDetection(label="player", confidence=0.6, x=0.1, y=0.1, w=0.05, h=0.1),
            ]))
        return out


def make_video(path: Path, n: int = 7) -> Path:
    cv2 = pytest.importorskip("cv2")
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    if not vw.isOpened():
        pytest.skip("no mp4v encoder in this opencv build")
    for i in range(n):
        vw.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    vw.release()
    return path


def test_run_inference_is_idempotent_and_resumable(tmp_path):
    vids = {"c1": make_video(tmp_path / "c1.mp4"), "c2": make_video(tmp_path / "c2.mp4", 4)}
    out = tmp_path / "out"
    det = FakeDetector()
    st = dr.run_inference(vids, out, detector=det, weights_name="w", batch_size=3, log=lambda m: None)
    assert st["done"] == ["c1", "c2"] and not st["failed"] and st["frames"] == 11
    d1 = de.ClipDetections.load(out / "dets" / "c1.npz")
    assert len(d1.frames) == 7 and d1.size == (64, 48) and d1.meta["settings"]["imgsz"] == dr.DEFAULT_IMGSZ
    labels, _, conf = d1.frame(6)
    assert sorted(labels) == ["in_play_ball", "player"]
    cache = read_cache(out / "cache" / "c1.jsonl")
    assert [f for f, _ in cache.frames] == list(range(7)) and len(cache.frames[0][1]) == 1  # ball candidates only
    assert not [p for p in out.rglob("*") if ".tmp" in p.name]

    calls = det.calls
    st2 = dr.run_inference(vids, out, detector=det, log=lambda m: None)  # nothing left to do
    assert st2["done"] == [] and st2["skipped"] == ["c1", "c2"] and det.calls == calls

    (out / "cache" / "c2.jsonl").unlink()  # a crash between the two renames: that clip, only, is redone
    st3 = dr.run_inference(vids, out, detector=det, batch_size=3, log=lambda m: None)
    assert st3["done"] == ["c2"] and st3["skipped"] == ["c1"]

    st4 = dr.run_inference(vids, out, detector=det, imgsz=640, log=lambda m: None)  # other settings: redo everything
    assert st4["done"] == ["c1", "c2"]


def test_run_inference_survives_a_bad_clip_and_can_stop_early(tmp_path):
    vids = {"a": make_video(tmp_path / "a.mp4"), "b": tmp_path / "missing.mp4", "c": make_video(tmp_path / "c.mp4")}
    st = dr.run_inference(vids, tmp_path / "o", detector=FakeDetector(), log=lambda m: None)
    assert st["done"] == ["a", "c"] and list(st["failed"]) == ["b"]
    st = dr.run_inference(vids, tmp_path / "o2", detector=FakeDetector(), log=lambda m: None, max_seconds=-1)
    assert st["stopped_early"] and st["done"] == []


def test_resolve_videos_prefers_labeller_dirs_and_handles_aliases(tmp_path):
    kf10, plain = tmp_path / "kf10", tmp_path / "clips"
    kf10.mkdir(); plain.mkdir()  # noqa: E702
    (kf10 / "arsenal_mancity_20250925_seg120.mp4").write_bytes(b"k")   # alias stem of the canonical clip
    (plain / "arsenal_mancity_20250925_seg120.mp4").write_bytes(b"p")
    (plain / "other_seg001.mp4").write_bytes(b"p")
    found, missing = dr.resolve_videos(
        {"arsenal_mancity_seg120": {"arsenal_mancity_seg120", "arsenal_mancity_20250925_seg120"},
         "other_seg001": {"other_seg001"}, "gone_seg002": {"gone_seg002"}},
        [kf10, plain])
    assert found["arsenal_mancity_seg120"] == kf10 / "arsenal_mancity_20250925_seg120.mp4"
    assert found["other_seg001"] == plain / "other_seg001.mp4" and missing == ["gone_seg002"]


def test_plan_lists_eval_clips_and_their_videos(versions, tmp_path, monkeypatch):
    v = snap.load_version("tracking", 1, versions.ds_root)
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "hmatch_seg001.mp4").write_bytes(b"x" * 10)
    monkeypatch.setenv("FOOTY_DATA", str(tmp_path))
    p = dr.plan(v)
    assert set(p["clips"]) == {"hmatch_seg001"} and p["clips"]["hmatch_seg001"]["bytes"] == 10
    assert p["missing"] == ["tmatch_seg001"] and p["clips"]["hmatch_seg001"]["labelled_frames"] == 3


def test_cli_score_and_plan(versions, tmp_path, capsys):
    v = snap.load_version("tracking", 1, versions.ds_root)
    perfect_store(tmp_path / "d", v)
    assert dr.main(["--root", str(versions.ds_root), "score", "--version", "1", "--dets", str(tmp_path / "d")]) == 0
    out = capsys.readouterr().out
    assert "eval_all" in out and '"ball_recall_10px": 1.0' in out
