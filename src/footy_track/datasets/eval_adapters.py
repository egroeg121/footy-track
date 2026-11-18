"""Eval adapters: ``evaluate(model, dataset_version, split, cohort) -> EvalOutput``.

One adapter per task, all with the same thin interface, scoring a model on the *items of a
dataset snapshot* (never on "current labels"):

=============  =====================================  =========================================
adapter        runs on                                 status
=============  =====================================  =========================================
calibration    CPU (proposal files vs GT correspond.)  working: success@5px / @10px, reprojection
broadcast      CPU (stored classifier calls)           working: accuracy / precision / recall / F1
ball tracker   CPU (replays cached detections)         working given a candidate cache
detector       GPU (RT-DETR / YOLO inference)          stub: NotImplementedError (Prefect gpu pool)
=============  =====================================  =========================================

:func:`evaluate_and_record` runs one adapter over every requested split (and fixed cohort),
attaches the contamination flags of the model's training data, and appends the rows to
``eval_result``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from footy_track import splits
from footy_track.datasets.results import (
    RESULT_SPLITS,
    ModelRef,
    ResultRow,
    contamination_flags,
    record_results,
)
from footy_track.datasets.snapshot import DatasetVersion
from footy_track.feature_store.store import FeatureStore


@dataclass
class EvalOutput:
    metrics: dict[str, float | None]
    n_items: int


class EvalAdapter(Protocol):
    task: str
    dataset: str

    def evaluate(
        self, model: Any, dataset_version: DatasetVersion, split: str, cohort: int | None = None
    ) -> EvalOutput: ...


def _items(dv: DatasetVersion, split: str, cohort: int | None) -> list[dict]:
    try:
        split = splits.to_result_split(split)  # train / eval / test accepted too
    except ValueError:
        split = None
    if split not in RESULT_SPLITS:
        raise ValueError(f"split not in {RESULT_SPLITS} (or train / eval / test)")
    return dv.select(split, cohort)


def _payload(row: dict) -> dict:
    import json  # noqa: PLC0415

    return json.loads(row["payload"])


# --------------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------------
class CalibrationModel(Protocol):
    name: str
    version: str

    def proposal(self, item: dict) -> dict | None:
        """The model's proposal for the item's frame (``H``, ``status``, ``correspondences`` ...
        in the calibration proposal format) or ``None`` when it produced nothing."""


@dataclass
class ProposalRunModel:
    """A stored proposals run: ``<gt dir>/calibration_proposals/<clip>.json`` -> ``runs[run_id]``."""

    run_id: str
    gt_dirs: Sequence[Path]
    name: str = "calibration_proposals"

    @property
    def version(self) -> str:
        return self.run_id

    def proposal(self, item: dict) -> dict | None:
        from footy_track.calibration.prelabel.proposals import read_proposals  # noqa: PLC0415

        for d in self.gt_dirs:
            run = read_proposals(Path(d), item["clip"])["runs"].get(self.run_id)
            if run and str(item["frame_index"]) in run["frames"]:
                return run["frames"][str(item["frame_index"])]
        return None


class CalibrationAdapter:
    """Reprojection accuracy of a calibration model against the human correspondences.

    Scored items are the ``homography`` items (a human fitted H from clicked points); the ones
    flagged ``not_broadcast`` / ``too_few_points`` are not scored. Reuses
    :func:`footy_track.calibration.prelabel.evaluate.eval_frame`: mean pixel error of the GT
    points reprojected through the model's H.
    """

    task = "calibration"
    dataset = "calibration"

    def evaluate(
        self, model: CalibrationModel, dataset_version: DatasetVersion, split: str, cohort: int | None = None
    ) -> EvalOutput:
        from footy_track.calibration.prelabel.evaluate import (  # noqa: PLC0415
            SUCCESS_PX,
            SUCCESS_PX_STRICT,
            eval_frame,
        )

        rows = [r for r in _items(dataset_version, split, cohort) if r["label_kind"] == "homography"]
        scored: list[dict] = []
        for r in rows:
            gt = _payload(r)
            prop = model.proposal(r)
            if prop is None:
                continue
            gt_in = {"correspondences": gt["correspondences"], "image_size": gt.get("image_size")}
            scored.append(eval_frame(gt_in, prop))
        n = len(scored)
        px = [s["mean_px"] for s in scored if s["mean_px"] is not None]
        mm = [s["mean_m"] for s in scored if s["mean_m"] is not None]
        agree_n = sum(s["n_agree"] for s in scored)
        agree_d = sum(s["n_same_location"] for s in scored)

        def rate(limit: float) -> float | None:
            return sum(1 for p in px if p <= limit) / n if n else None

        return EvalOutput(
            {
                f"success_at_{SUCCESS_PX_STRICT:g}px": rate(SUCCESS_PX_STRICT),
                f"success_at_{SUCCESS_PX:g}px": rate(SUCCESS_PX),
                "mean_reprojection_px": float(np.mean(px)) if px else None,
                "median_reprojection_px": float(np.median(px)) if px else None,
                "mean_reprojection_m": float(np.mean(mm)) if mm else None,
                "failed_rate": sum(s["status"] == "failed" for s in scored) / n if n else None,
                "keypoint_agreement": agree_n / agree_d if agree_d else None,
                "coverage": n / len(rows) if rows else None,
            },
            n,
        )


# --------------------------------------------------------------------------------------------
# broadcast
# --------------------------------------------------------------------------------------------
class BroadcastModel(Protocol):
    name: str
    version: str

    def says_yes(self, item: dict) -> bool | None:
        """The classifier's call for the item's frame (``None`` = no prediction)."""


@dataclass
class StoredPredictionModel:
    """The classifier's call recorded when the frame was queued for review (``pred`` /
    ``p_yes`` in the item payload): the number the review page shows next to the frame.
    Those frames are the ones nearest the decision boundary plus a random sample, so the
    headline numbers are biased towards hard frames; ``accuracy_random`` is the unbiased subset."""

    name: str = "broadcast_classifier_at_review"
    version: str = "stored"

    def says_yes(self, item: dict) -> bool | None:
        p = _payload(item)
        if p.get("pred") in ("Yes", "No"):
            return p["pred"] == "Yes"
        if p.get("p_yes") is not None:
            return p["p_yes"] >= 0.5
        return None


class BroadcastAdapter:
    """Accuracy / precision / recall / F1 of "is this a main-camera frame" against human
    verdicts. ``unsure`` verdicts and frames with no prediction are excluded."""

    task = "broadcast"
    dataset = "broadcast"

    def evaluate(
        self, model: BroadcastModel, dataset_version: DatasetVersion, split: str, cohort: int | None = None
    ) -> EvalOutput:
        tp = fp = tn = fn = 0
        rnd_ok = rnd_n = 0
        for r in _items(dataset_version, split, cohort):
            if r["label_kind"] not in ("yes", "no"):
                continue
            pred = model.says_yes(r)
            if pred is None:
                continue
            truth = r["label_kind"] == "yes"
            tp += pred and truth
            fp += pred and not truth
            tn += (not pred) and (not truth)
            fn += (not pred) and truth
            if _payload(r).get("reason") == "random":
                rnd_n += 1
                rnd_ok += pred == truth

        def div(a: float, b: float) -> float | None:
            return a / b if b else None

        n = tp + fp + tn + fn
        prec, rec = div(tp, tp + fp), div(tp, tp + fn)
        f1 = div(2 * prec * rec, prec + rec) if prec is not None and rec is not None else None
        return EvalOutput(
            {
                "accuracy": div(tp + tn, n),
                "precision_yes": prec,
                "recall_yes": rec,
                "f1_yes": f1,
                "accuracy_random": div(rnd_ok, rnd_n),
                "n_yes": float(tp + fn),
                "n_no": float(tn + fp),
            },
            n,
        )


# --------------------------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------------------------
class DetectorAdapter:
    """Detector precision / recall / mAP50 on the tracking items' boxes.

    Needs the detector weights run over the frames of every eval item: GPU work. Not wired:
    TODO(prefect): a flow on the ``gpu`` work pool (desktop RTX 2070) that loads the weights,
    runs inference over ``dataset_version.select(split)`` frames, computes P / R / mAP50 against
    the item boxes (tiers 1-2, ``rejected`` boxes excluded) and calls
    ``datasets results record`` (or :func:`footy_track.datasets.results.record_results`).
    """

    task = "tracking"
    dataset = "tracking"

    def evaluate(self, model: Any, dataset_version: DatasetVersion, split: str, cohort: int | None = None) -> EvalOutput:
        raise NotImplementedError(
            "detector eval runs inference over every eval frame (GPU); run it as a Prefect flow on "
            "the gpu pool and record the metrics with `python -m footy_track.datasets results record`"
        )


@dataclass
class BallTrackerReplay:
    """A ball tracker replayed over cached detector candidates (CPU only).

    ``cache_dir`` holds ``<canonical clip>.jsonl`` files written by
    ``python -m footy_track.scripts.ball_tracking_eval cache`` (the GPU step, run once per
    detector weights); ``make_tracker(frame_size)`` builds a fresh tracker per clip.
    """

    kind: str
    cache_dir: Path
    detector_version: str = "unknown"
    params: dict | None = None
    make_tracker: Callable[[tuple[int, int]], Any] | None = None

    @property
    def name(self) -> str:
        return f"ball_tracker_{self.kind}"

    @property
    def version(self) -> str:
        return f"{self.detector_version}"

    def tracker_for(self, frame_size: tuple[int, int]) -> Any:
        if self.make_tracker is not None:
            return self.make_tracker(frame_size)
        from footy_track.ball_trackers.factory import make_ball_tracker  # noqa: PLC0415

        return make_ball_tracker(self.kind, frame_size, **(self.params or {}))


class BallTrackerAdapter:
    """Ball tracker precision / recall at 10 px and 25 px, from the replay of cached detections.

    GT is rebuilt from the dataset items by the rules of ``ball_eval.detection_replay``: a
    ball box is *human* GT when tier 1, *propagated* when tier 2 (machine box a human
    confirmed); rejected boxes and tier 3 are never GT. ``no_ball`` frames score false
    positives; ``not_broadcast`` frames and frames without items are not scored.
    """

    task = "tracking"
    dataset = "tracking"

    def __init__(self, radii: Iterable[float] = (10.0, 25.0), include_propagated: bool = False) -> None:
        self.radii = tuple(radii)
        self.include_propagated = include_propagated

    def _gts(self, items: list[dict]) -> dict[str, Any]:
        from footy_track.ball_eval.detection_replay import (  # noqa: PLC0415
            BALL_TAGS,
            HUMAN,
            PROPAGATED,
            ClipGT,
        )

        gts: dict[str, ClipGT] = {}
        for r in items:
            gt = gts.setdefault(r["clip"], ClipGT(r["clip"]))
            p = _payload(r)
            f = r["frame_index"]
            if p["marker"] == "no_ball":
                gt.no_ball.add(f)
            elif p["marker"] == "not_broadcast":
                gt.not_broadcast.add(f)
            for b in p["boxes"]:
                if b["label"].lower() not in BALL_TAGS or b["rejected"]:
                    continue
                c = (b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)
                if b["tier"] == 1:
                    gt.marks[HUMAN].setdefault(f, []).append(c)
                elif b["tier"] == 2 and b["source"] == "vittrack":
                    gt.marks[PROPAGATED].setdefault(f, []).append(c)
                else:
                    gt.ignored_machine_marks += 1
        return gts

    def evaluate(
        self, model: BallTrackerReplay, dataset_version: DatasetVersion, split: str, cohort: int | None = None
    ) -> EvalOutput:
        from footy_track.ball_eval.detection_replay import (  # noqa: PLC0415
            HUMAN,
            PROPAGATED,
            Counts,
            pool,
            read_cache,
            replay,
            score_clip,
        )

        items = _items(dataset_version, split, cohort)
        gts = self._gts(items)
        tiers = (HUMAN, PROPAGATED) if self.include_propagated else (HUMAN,)
        per_clip: dict[str, dict[float, Counts]] = {}
        missing = 0
        for clip, gt in sorted(gts.items()):
            path = Path(model.cache_dir) / f"{clip}.jsonl"
            if not path.is_file():
                missing += 1
                continue
            cache = read_cache(path)
            points = replay(model.tracker_for(cache.frame_size), cache)
            per_clip[clip] = score_clip(points, cache, gt, tiers=tiers, radii=self.radii)
        pooled = pool(per_clip, per_clip)
        metrics: dict[str, float | None] = {"clips_without_cache": float(missing), "clips_scored": float(len(per_clip))}
        n_frames = 0
        for r, counts in pooled.items():
            ratios = counts.ratios()
            metrics[f"ball_precision_{r:g}px"] = ratios["precision"]
            metrics[f"ball_recall_{r:g}px"] = ratios["recall"]
            metrics[f"ball_wrong_per_prediction_{r:g}px"] = ratios["wrong_per_prediction"]
            n_frames = counts.ball_frames
        return EvalOutput(metrics, n_frames)


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------
def evaluate_and_record(
    adapter: EvalAdapter,
    model: Any,
    dataset_version: DatasetVersion,
    target: FeatureStore | Path | str | None,
    *,
    model_ref: ModelRef | None = None,
    result_splits: Sequence[str] = (splits.EVAL_HELDOUT, splits.EVAL_IN_MATCH, "eval_all"),
    cohorts: Sequence[int] = (),
) -> list[ResultRow]:
    """Score ``model`` on ``dataset_version`` for every split in ``result_splits`` (and, for
    each fixed ``cohort``, on the items first seen at or before it), attach contamination flags
    from ``model_ref.training`` and record everything. Returns the rows written.

    Two rows per metric for an eval split when a cohort is given: the full current split
    (``cohort_version`` NULL) and the fixed cohort, so a progress plot can show both.
    """
    ref = model_ref or ModelRef(getattr(model, "name", "model"), str(getattr(model, "version", "unknown")))
    flags = contamination_flags(ref.training, dataset_version)
    rows: list[ResultRow] = []
    for split in result_splits:
        for cohort in (None, *(cohorts if split != splits.TRAIN else ())):
            out = adapter.evaluate(model, dataset_version, split, cohort)
            flag = flags[split]
            for metric, value in out.metrics.items():
                rows.append(
                    ResultRow(
                        task=adapter.task, dataset=dataset_version.dataset,
                        dataset_version=dataset_version.version, split=split, cohort_version=cohort,
                        model_name=ref.name, model_version=ref.version, metric=metric, value=value,
                        n_items=out.n_items, contaminated=flag.contaminated, contamination_note=flag.note,
                    )
                )
    record_results(target, rows)
    return rows


ADAPTERS: dict[str, Callable[[], EvalAdapter]] = {
    "calibration": CalibrationAdapter,
    "broadcast": BroadcastAdapter,
    "tracking": BallTrackerAdapter,
    "tracking-detector": DetectorAdapter,
}
