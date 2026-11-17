"""eval_result round trip, contamination flags, and the eval adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from footy_track import splits
from footy_track.ball_eval.detection_replay import CandidateCache, write_cache
from footy_track.ball_trackers.detection_tracker import DETECTED, BallCandidate, BallPoint
from footy_track.datasets import eval_adapters as ea
from footy_track.datasets import results as res
from footy_track.datasets import snapshot as snap
from footy_track.datasets.__main__ import main as cli
from footy_track.feature_store import ingest_all as ia
from tests.datasets.fixtures import box


@pytest.fixture
def versions(world):
    ia.run(world.db, data_root=world.root, marks_dirs=[world.marks])
    for ds in ("tracking", "broadcast", "calibration"):
        snap.bump(ds, world.db, world.ds_root)
    return world


def dv(w, ds, v=1):
    return snap.load_version(ds, v, w.ds_root)


def row(**kw):
    base = dict(task="broadcast", dataset="broadcast", dataset_version=1, split="eval_all",
                model_name="m", model_version="w1", metric="accuracy", value=0.5, n_items=10)
    return res.ResultRow(**(base | kw))


# ---- eval_result -----------------------------------------------------------------------------
def test_results_round_trip_and_latest_view(world):
    res.record_results(world.db, [row(), row(metric="f1_yes", value=0.4, cohort_version=1, contaminated=True,
                                             contamination_note="3 of 9")])
    got = {(r["metric"], r["cohort_version"]): r for r in res.list_results(world.db)}
    assert got[("accuracy", None)]["value"] == 0.5
    assert got[("f1_yes", 1)]["contaminated"] is True and got[("f1_yes", 1)]["contamination_note"] == "3 of 9"
    assert got[("accuracy", None)]["contaminated"] is None  # unchecked is NULL, not False
    assert got[("accuracy", None)]["run_at"] and got[("accuracy", None)]["n_items"] == 10

    # a re-run appends; the view returns the newest, history keeps both
    from datetime import UTC, datetime

    res.record_results(world.db, [row(value=0.7)], now=datetime(2099, 1, 1, tzinfo=UTC))
    latest = [r for r in res.list_results(world.db) if r["metric"] == "accuracy"]
    assert [r["value"] for r in latest] == [0.7]
    hist = [r for r in res.list_results(world.db, latest_only=False) if r["metric"] == "accuracy"]
    assert sorted(r["value"] for r in hist) == [0.5, 0.7]
    # a different dataset version or model is a different key
    res.record_results(world.db, [row(dataset_version=2, value=0.9), row(model_name="other", value=0.1)])
    assert len([r for r in res.list_results(world.db) if r["metric"] == "accuracy"]) == 3
    assert [r["dataset_version"] for r in res.list_results(world.db, model="m", split="eval_all", dataset_version=2)] == [2]


def test_record_validates_split_and_task(world):
    with pytest.raises(ValueError):
        res.record_results(world.db, [row(split="validation")])
    with pytest.raises(ValueError):
        res.record_results(world.db, [row(task="nope")])


def test_results_cli(world, capsys):
    argv = ["--db", str(world.db), "--root", str(world.ds_root)]
    ia.run(world.db, data_root=world.root, marks_dirs=[world.marks])
    snap.bump("broadcast", world.db, world.ds_root)
    assert cli([*argv, "results", "record", "--task", "broadcast", "--dataset", "broadcast", "--dataset-version", "1",
                "--split", "eval_heldout_match", "--model", "clf", "--model-version", "abc", "--metric", "accuracy",
                "--value", "0.91", "--n-items", "4"]) == 0
    capsys.readouterr()
    assert cli([*argv, "results", "list"]) == 0
    out = capsys.readouterr().out
    assert "clf@abc" in out and "0.9100" in out and "(1 rows)" in out


# ---- contamination ---------------------------------------------------------------------------
def make_export(tmp_path: Path, names: dict[str, list[str]]) -> Path:
    d = tmp_path / "rf"
    for subset, files in names.items():
        (d / subset / "images").mkdir(parents=True)
        for f in files:
            (d / subset / "images" / f).write_bytes(b"x")
    return d


H = "." + "ab" * 12


def test_contamination_flag_from_a_roboflow_export(world, versions, tmp_path):
    clean = make_export(tmp_path, {"train": [f"tmatch_seg002_{i:06d}_jpg.rf{H}.jpg" for i in range(3)]})
    flags = res.contamination_flags(res.TrainingData(roboflow_dir=clean), dv(versions, "tracking"))
    assert flags[splits.EVAL_HELDOUT].contaminated is False
    assert flags[splits.EVAL_IN_MATCH].contaminated is False and flags["eval_all"].contaminated is False
    assert flags["train"].contaminated is None

    dirty = make_export(
        tmp_path / "d",
        {"train": [f"tmatch_seg002_000001_jpg.rf{H}.jpg", f"hmatch_seg001_000002_jpg.rf{H}.jpg",
                   f"tmatch_seg001_000003_jpg.rf{H}.jpg"]},
    )
    flags = res.contamination_flags(res.TrainingData(roboflow_dir=dirty), dv(versions, "tracking"))
    assert flags[splits.EVAL_HELDOUT].contaminated and "1 of 3" in flags[splits.EVAL_HELDOUT].note
    assert flags[splits.EVAL_IN_MATCH].contaminated
    assert flags["eval_all"].contaminated and "eval_heldout_match" in flags["eval_all"].note

    only_heldout = make_export(tmp_path / "h", {"train": [f"hmatch_seg001_000002_jpg.rf{H}.jpg"],
                                                "valid": [f"tmatch_seg001_000003_jpg.rf{H}.jpg"]})
    flags = res.contamination_flags(res.TrainingData(roboflow_dir=only_heldout), dv(versions, "tracking"))
    assert flags[splits.EVAL_HELDOUT].contaminated and flags[splits.EVAL_IN_MATCH].contaminated is False


def test_undeclared_training_data_is_unchecked_not_clean(versions):
    flags = res.contamination_flags(None, dv(versions, "tracking"))
    assert {f.contaminated for f in flags.values()} == {None}


def test_contamination_flag_from_a_dataset_version(versions):
    tv = dv(versions, "tracking")
    flags = res.contamination_flags(res.TrainingData(dataset="tracking", version=1, root=versions.ds_root), tv)
    assert flags["eval_all"].contaminated is False  # sticky assignment: train items never share eval clips
    # a register edit that moves an eval clip into the training pool contaminates the old eval set
    rows = tv.rows()
    for r in rows:
        if r["clip"] == "tmatch_seg001":
            r["split"] = "train"
    fake = snap.DatasetVersion("tracking", 2, tv.manifest, tv.table.from_pylist(rows, schema=snap.ITEM_SCHEMA_ARROW), tv.path)
    flags = res.contamination_from_dataset(fake, tv)
    assert flags[splits.EVAL_IN_MATCH].contaminated and "tmatch_seg001" not in flags[splits.EVAL_HELDOUT].note


# ---- adapters --------------------------------------------------------------------------------
def test_broadcast_adapter_scores_stored_predictions(versions):
    out = ea.BroadcastAdapter().evaluate(ea.StoredPredictionModel(), dv(versions, "broadcast"), "eval_heldout_match")
    # frames: 10 yes/Yes edge (tp), 11 no/Yes random (fp), 12 no/No random (tn), 13 yes/No edge (fn);
    # 14 unsure is excluded
    m = out.metrics
    assert out.n_items == 4
    # headline = the random stratum (frames 11 and 12: one right, one wrong)
    assert m["accuracy"] == 0.5 and m["accuracy_random"] == 0.5 and m["n_random"] == 2
    assert m["precision_yes"] == 0.0 and m["recall_yes"] is None and m["f1_yes"] is None
    assert (m["n_yes"], m["n_no"]) == (0, 2)
    # edge stratum (10 right, 13 wrong) and all frames are separate, diagnostic metrics
    assert m["accuracy_edge"] == 0.5 and m["n_edge"] == 2
    assert m["accuracy_all_biased"] == 0.5 and m["f1_yes_all_biased"] == 0.5
    assert ea.BroadcastAdapter().evaluate(ea.StoredPredictionModel(), dv(versions, "broadcast"), "train").n_items == 0


class AlwaysYes:
    name, version = "always_yes", "1"

    def says_yes(self, item):
        return True


def test_broadcast_adapter_accepts_any_model(versions):
    out = ea.BroadcastAdapter().evaluate(AlwaysYes(), dv(versions, "broadcast"), "eval_all")
    # random frames: 11 no, 12 no -> never right; edge frames 10 yes, 13 yes -> both right
    assert out.metrics["accuracy"] == 0.0 and out.metrics["accuracy_edge"] == 1.0
    assert out.metrics["accuracy_all_biased"] == 0.5 and out.metrics["precision_yes"] == 0.0


class _FakeDV:
    """Just enough of DatasetVersion for the adapter: ``select`` returns the item rows."""

    def __init__(self, rows):
        self.rows = rows

    def select(self, split, cohort=None):
        return self.rows


def _bitem(i, verdict, p_yes, reason):
    payload = {"verdict": verdict, "p_yes": p_yes, "pred": "Yes" if p_yes >= 0.5 else "No", "ts_s": i / 10}
    if reason is not None:
        payload["reason"] = reason
    return {"item_id": f"m:{i}", "label_kind": verdict, "payload": json.dumps(payload)}


def test_broadcast_adapter_reports_random_accuracy_as_headline():
    rows = [
        # 6 edge frames at p ~ 0.5: stored call is Yes (0.51) but every one is a human "no" -> edge acc 0
        *[_bitem(i, "no", 0.51, "edge_mid") for i in range(4)],
        *[_bitem(i, "no", 0.49, "edge_boundary") for i in range(4, 6)],
        # 4 random frames: 3 right, 1 wrong
        _bitem(10, "yes", 0.95, "random"), _bitem(11, "no", 0.02, "random"),
        _bitem(12, "yes", 0.90, "random"), _bitem(13, "yes", 0.10, "random"),
        # legacy frames with no reason: never assumed random
        _bitem(20, "yes", 0.9, None), _bitem(21, "no", 0.9, None),
    ]
    out = ea.BroadcastAdapter().evaluate(ea.StoredPredictionModel(), _FakeDV(rows), "eval_all")
    m = out.metrics
    assert m["accuracy"] == m["accuracy_random"] == 0.75 and m["n_random"] == 4  # the headline
    assert m["f1_yes"] == m["f1_yes_random"] == pytest.approx(0.8)  # tp=2 fp=0 fn=1
    # edge: 4 x 0.51 -> Yes (wrong), 2 x 0.49 -> No (right)
    assert m["accuracy_edge"] == pytest.approx(2 / 6) and m["n_edge"] == 6
    assert m["f1_yes_edge"] is None or m["f1_yes_edge"] == 0.0
    # all frames, edge-weighted: right = 2 edge + 3 random + 1 legacy = 6 of 12
    assert m["accuracy_all_biased"] == 0.5 and "f1_yes_all_biased" in m
    assert m["n_unknown"] == 2 and out.n_items == 12  # legacy frames excluded from the headline only


def test_broadcast_stratum_never_assumes_legacy_is_random():
    s = ea.broadcast_stratum
    assert s(_bitem(0, "yes", 0.5, "random")) == "random"
    assert s(_bitem(0, "yes", 0.5, "edge")) == "edge"  # legacy review value
    assert s(_bitem(0, "yes", 0.5, "edge_mid")) == s(_bitem(0, "yes", 0.5, "edge_boundary")) == "edge"
    assert s(_bitem(0, "yes", 0.5, None)) == "unknown"
    assert s(_bitem(0, "yes", 0.5, "band")) == "unknown"


def test_calibration_adapter_reprojection(versions, tmp_path):
    gt_dir = tmp_path / "gt"
    (gt_dir / "calibration_proposals").mkdir(parents=True)
    good = [[100.0, 0, -50], [0, 68, -34], [0, 0, 1]]
    bad = [[100.0, 0, -50 + 3.0], [0, 68, -34 + 2.0], [0, 0, 1]]  # a few metres off
    for clip, frame, H in (("tmatch_seg001", 4, good), ("tmatch_seg002", 6, good), ("tmatch_seg002", 7, bad)):
        p = gt_dir / "calibration_proposals" / f"{clip}.json"
        doc = json.loads(p.read_text()) if p.exists() else {"clip": clip, "runs": {"r1": {"source": "x", "frames": {}}}}
        doc["runs"]["r1"]["frames"][str(frame)] = {"status": "proposed", "score": 0.9, "H": H, "correspondences": []}
        p.write_text(json.dumps(doc))
    model = ea.ProposalRunModel("r1", [gt_dir])
    a = ea.CalibrationAdapter()
    out = a.evaluate(model, dv(versions, "calibration"), "eval_in_match")
    assert out.n_items == 1 and out.metrics["success_at_5px"] == 1.0 and out.metrics["mean_reprojection_px"] < 1e-6
    out = a.evaluate(model, dv(versions, "calibration"), "train")
    assert out.n_items == 2 and out.metrics["success_at_10px"] == 0.5  # the shifted H misses by > 10 px
    assert out.metrics["mean_reprojection_px"] > 10 and out.metrics["coverage"] == 1.0
    # held-out frame has no proposal: nothing scored, coverage 0
    out = a.evaluate(model, dv(versions, "calibration"), "eval_heldout_match")
    assert out.n_items == 0 and out.metrics["coverage"] == 0.0


class Perfect:
    """Tracker that reports the single best (highest-confidence) candidate on every frame."""

    def __init__(self, size):
        self.pts = []

    def update(self, f, cands):
        best = max(cands, key=lambda c: c.confidence, default=None)
        self.pts.append(BallPoint(f, DETECTED, 1, best.x, best.y, best.w, best.h, best.confidence) if best else BallPoint(f))

    def points(self):
        return self.pts


def test_ball_tracker_adapter_replays_cached_detections(versions, tmp_path):
    cache = tmp_path / "cache"
    # hmatch_seg001: GT ball at frame 0 (0.51, 0.515) and frame 1; frame 2 is no_ball
    write_cache(cache / "hmatch_seg001.jsonl", CandidateCache("hmatch_seg001", (1920, 1080), [
        (0, [BallCandidate(0.5, 0.5, 0.02, 0.03, 0.9)]),          # exactly the GT box: hit
        (1, [BallCandidate(0.1, 0.1, 0.02, 0.03, 0.9)]),          # far from GT: wrong
        (2, [BallCandidate(0.3, 0.3, 0.02, 0.03, 0.9)]),          # no_ball frame: false positive
    ]))
    model = ea.BallTrackerReplay("online", cache, detector_version="w1", make_tracker=Perfect)
    out = ea.BallTrackerAdapter().evaluate(model, dv(versions, "tracking"), "eval_heldout_match")
    assert out.n_items == 2  # two ball frames
    assert out.metrics["ball_recall_10px"] == 0.5
    assert out.metrics["ball_precision_10px"] == pytest.approx(1 / 3)  # 1 hit, 1 wrong, 1 false positive
    assert out.metrics["clips_scored"] == 1 and out.metrics["clips_without_cache"] == 0
    out = ea.BallTrackerAdapter().evaluate(model, dv(versions, "tracking"), "eval_in_match")
    assert out.metrics["clips_without_cache"] == 1 and out.n_items == 0


def test_evaluate_and_record_writes_full_and_cohort_rows_with_flags(versions, tmp_path):
    # v2: a new eval frame in the held-out match
    versions.sidecar("hmatch_seg001", [box(0, "in_play_ball"), box(0, "player", x=0.2), box(1, "in_play_ball"),
                                       box(7, "in_play_ball")])
    ia.run(versions.db, data_root=versions.root, marks_dirs=[versions.marks])
    import tests.datasets.fixtures as fx

    fx.write_broadcast(versions, "hmatch_full", {10: ("yes", "Yes", "edge"), 11: ("no", "Yes", "random"),
                                                  12: ("no", "No", "random"), 13: ("yes", "No", "edge"),
                                                  14: ("unsure", "No", "edge"), 15: ("yes", "Yes", "random")})
    ia.run(versions.db, data_root=versions.root, marks_dirs=[versions.marks])
    assert snap.bump("broadcast", versions.db, versions.ds_root).version == 2
    dirty = make_export(tmp_path, {"train": [f"hmatch_seg001_000002_jpg.rf{H}.jpg"]})
    ref = res.ModelRef("clf", "w1", training=res.TrainingData(roboflow_dir=dirty))
    rows = ea.evaluate_and_record(ea.BroadcastAdapter(), ea.StoredPredictionModel(), dv(versions, "broadcast", 2),
                                  versions.db, model_ref=ref, cohorts=[1])
    stored = res.list_results(versions.db, dataset_version=2)
    assert len(stored) == len(rows)
    by = {(r["split"], r["cohort_version"], r["metric"]): r for r in stored}
    full, cohort = by[("eval_all", None, "accuracy")], by[("eval_all", 1, "accuracy")]
    assert full["n_items"] == 5 and cohort["n_items"] == 4  # the cohort is the v1 eval set
    # headline = random frames only: the v1 cohort has 11 (wrong) and 12 (right); v2 adds 15 (right)
    assert cohort["value"] == 0.5 and full["value"] == pytest.approx(2 / 3)
    assert by[("eval_all", None, "n_random")]["value"] == 3 and by[("eval_all", None, "accuracy_all_biased")]["value"] == pytest.approx(0.6)
    assert full["contaminated"] is True and by[("eval_heldout_match", None, "accuracy")]["contaminated"] is True
    assert by[("eval_in_match", None, "accuracy")]["contaminated"] is False
