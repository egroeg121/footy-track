"""Named evaluation sets: where the GT, the detector candidates and the videos of
one match live.

========  ========================================  ========================================
set       GT (labeller sidecars)                    candidates (+ frame-count manifest)
========  ========================================  ========================================
mancity   ``ball_gt_marks/``                        ``machine_labels/``
norwich   ``ball_gt_marks_norwich/``                ``ball_candidates_norwich/`` (+
          (the labeller's ``FOOTY_GT_MARKS_DIR``    ``machine_labels_norwich/_manifest.json``)
          for the held-out Norwich instance)
========  ========================================  ========================================

Every path can be overridden, so any labeller output directory can be scored.
Ball Check verdicts live under ``<gt_dir>/ball_checks/``.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("FOOTY_DATA", "/mnt/storage/footy_data"))


@dataclasses.dataclass(frozen=True)
class EvalSetSpec:
    name: str
    gt_dir: (
        Path  # labeller sidecars (FOOTY_GT_MARKS_DIR of the labeller that wrote them)
    )
    candidates_dir: Path  # ball candidates (``in_play_ball`` rows with confidence)
    manifest_dir: Path | None = (
        None  # detector run manifest (frame counts) if not in candidates_dir
    )

    @property
    def verdicts_dir(self) -> Path:
        return self.gt_dir / "ball_checks"


def eval_set(
    name: str,
    *,
    root: Path | None = None,
    gt_dir: Path | str | None = None,
    candidates_dir: Path | str | None = None,
    manifest_dir: Path | str | None = None,
) -> EvalSetSpec:
    root = Path(root) if root else data_root()
    if name == "mancity":
        base = EvalSetSpec(
            "mancity",
            root / "ball_gt_marks",
            root / "machine_labels",
        )
    elif name == "norwich":
        base = EvalSetSpec(
            "norwich",
            root / "ball_gt_marks_norwich",
            root / "ball_candidates_norwich",
            root / "machine_labels_norwich",
        )
    else:
        if gt_dir is None or candidates_dir is None:
            raise ValueError(
                f"unknown eval set {name!r}: give --gt-dir and --candidates-dir "
                "(known sets: mancity, norwich)"
            )
        base = EvalSetSpec(name, Path(gt_dir), Path(candidates_dir))
    return dataclasses.replace(
        base,
        **{
            k: Path(v)
            for k, v in (
                ("gt_dir", gt_dir),
                ("candidates_dir", candidates_dir),
                ("manifest_dir", manifest_dir),
            )
            if v is not None
        },
    )
