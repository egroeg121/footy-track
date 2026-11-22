"""Export proposals for the review page from a combined match calibration (design ``calibration_match_flow.md`` 3b).

A sample of frames of one match is drawn (stratified, fixed seed), and for each frame the visible pitch keypoints are
projected through the STORED per-frame ``H`` and ``k1`` into ``calibration_proposals/<clip>.json`` files in exactly the
format ``labeller/calibration_proposals.py`` reads (``calibration/prelabel/proposals.py`` is the file helper), so that
accepting a proposal re-fits the stored ``H`` and ``k1`` from the points and writes ``calibration_gt`` with an
``origin``. Nothing is recomputed from video: the proposal is a view of the stored row.

Conventions (all checked against ``match_calibration`` / ``geometry``):

* the combined ``H`` maps UNDISTORTED normalised image (x/1920, y/1080) to pitch metres, origin centre spot, 105 x 68;
* a proposal ``correspondences[].image`` is an OBSERVED (distorted) normalised point: keypoint -> ``H^-1`` ->
  undistorted normalised -> division-model distortion with the row's ``k1`` (``geometry.distort_norm``, centre
  (0.5, 0.5), aspect 16:9). Pixels are these times (1920, 1080); the lens is defined on normalised coords, so the
  two orders are the same map;
* the page's clip is the segment (``<match>_segNNN``) and its frame index is the segment-local frame;
* ``run_id`` is ``<version>@<sha8>`` of the combined file's ``version_id`` (``ptz_v3+73bab0eb`` -> ``ptz_v3@73bab0eb``;
  ``+`` is avoided because the run id is a URL path component). Accepting writes it as ``origin.run_id``.

Sampling: for every stratum an independent draw; a stratum fixes, per segment, an expected count ``e_j`` (capped) and
picks ``floor(e_j) + Bernoulli(frac)`` frames of the segment's pool with the smallest per-frame uniform keys. The keys
are generated once per (stratum, global frame id) from ``default_rng(sha256(f"{match}:{seed}"))``, so every version of
the combined file is asked about the same frames wherever the pools agree. The marginal inclusion probability of a
frame in a stratum is exactly ``e_j / m_j`` (m_j: pool size in the segment); across strata
``pi = 1 - prod(1 - pi_s)`` (the draws are independent). Both are stored per proposal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration import match_calibration as mc
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.proposals import update_proposals
from footy_track.clip_id import canonical_clip_id

SOURCE = "ptz_v3"
TARGET_FRAMES = 300
#: a proposal needs this many visible keypoints (the accept path refits H from them)
MIN_POINTS = 4
#: frames within this many frames of a segment's start / end are "boundary" frames
BOUNDARY_FRAMES = 25
#: per-segment cap of a budgeted stratum (design: <= 3 per segment per stratum)
SEGMENT_CAP = 3
#: recovered segments get at least this many frames (design: >= 5 per flagged or recovered segment)
RECOVERED_MIN = 5
#: segments of at least this many frames get at least one frame (design)
FLOOR_SEGMENT_FRAMES = 100

#: share of the remaining budget of every budgeted stratum (design 3b; the status strata ``uncalibrated`` 10% and
#: ``not_broadcast`` 5% and the far-touchline tertiles are not implemented: see the module notes in the script)
SHARES = {
    "accepted": 0.50, "calibrated_not_accepted": 0.25,
    "zoom_t1": 0.0167, "zoom_t2": 0.0167, "zoom_t3": 0.0166,
    "pan_t1": 0.0167, "pan_t2": 0.0167, "pan_t3": 0.0166,
    "boundary": 0.02,
}


# --------------------------------------------------------------------------
# reading the combined output (the one place that knows where the rows come from)
# --------------------------------------------------------------------------


@dataclass
class FrameRows:
    """Per-frame rows of one combined match calibration: what the exporter needs, nothing store-specific."""

    match: str
    version: str
    version_id: str
    frame: np.ndarray  # (N,) global frame id
    segment: np.ndarray  # (N,)
    local_frame: np.ndarray  # (N,)
    calibrated: np.ndarray  # (N,) bool
    accepted: np.ndarray  # (N,) bool
    evidence_f1: np.ndarray  # (N,)
    source: np.ndarray  # (N,) object: main | recovered | none
    pan: np.ndarray
    logf: np.ndarray
    H: np.ndarray  # (N, 3, 3) undistorted normalised image -> pitch metres
    k1: np.ndarray
    segment_files: dict[int, str] = field(default_factory=dict)  # segment -> mp4 name
    total_frames: int | None = None
    content_sha256: str = ""

    @property
    def run_id(self) -> str:
        return self.version_id.replace("+", "@", 1)


def load_rows(match: str, *, version: str = mc.DEFAULT_VERSION, root: Path | None = None) -> FrameRows:
    """The combined calibration of ``match`` as :class:`FrameRows`. Swap this for a feature-store reader (P5) later."""
    c = mc.load_match_calibration(match, version=version, root=root)
    mf = c.manifest
    return FrameRows(
        match=match, version=version, version_id=c.version_id, frame=c.frame, segment=c.segment,
        local_frame=c.local_frame, calibrated=c.calibrated, accepted=c.accepted, evidence_f1=c.evidence_f1,
        source=c.source, pan=c.pan, logf=c.logf, H=c.H, k1=c.k1,
        segment_files={int(s["segment"]): s["file"] for s in mf.get("segments", [])},
        total_frames=(mf.get("inputs") or {}).get("video_total_frames"),
        content_sha256=mf.get("content_sha256", ""),
    )


def segment_clip(rows: FrameRows, segment: int) -> str:
    """Canonical clip id of a segment (what the review page files proposals under)."""
    name = rows.segment_files.get(int(segment)) or f"{rows.match}_seg{int(segment):03d}"
    return canonical_clip_id(name)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def _seed_int(match: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{match}:{seed}".encode()).digest()[:8], "big")


@dataclass
class Pick:
    row: int
    strata: list[str]
    pi: float


def _strata(rows: FrameRows) -> list[tuple[str, np.ndarray, str]]:
    """``(name, pool mask, kind)`` in a fixed order; ``kind`` is ``budget`` | ``floor`` | ``recovered``."""
    cal, acc = rows.calibrated, rows.accepted
    out: list[tuple[str, np.ndarray, str]] = [
        ("accepted", acc, "budget"),
        ("calibrated_not_accepted", cal & ~acc, "budget"),
    ]
    for key, vals in (("zoom", rows.logf), ("pan", rows.pan)):
        if cal.any():
            q = np.quantile(vals[cal], [1 / 3, 2 / 3])
            t = np.digitize(vals, q)  # 0, 1, 2
        else:
            t = np.zeros(len(vals), int)
        out += [(f"{key}_t{i + 1}", cal & (t == i), "budget") for i in range(3)]
    seg_len = {s: int((rows.segment == s).sum()) for s in np.unique(rows.segment)}
    n_seg = np.array([seg_len[s] for s in rows.segment])
    edge = (rows.local_frame < BOUNDARY_FRAMES) | (rows.local_frame >= n_seg - BOUNDARY_FRAMES)
    out.append(("boundary", cal & edge, "budget"))
    out.append(("recovered_segment", cal & (rows.source == "recovered"), "recovered"))
    out.append(("segment_floor", cal, "floor"))
    return out


def _expected_counts(kind: str, pool: np.ndarray, seg: np.ndarray, budget: float) -> dict[int, float]:
    """Expected sample size ``e_j`` per segment with a non-empty pool."""
    segs, counts = np.unique(seg[pool], return_counts=True)
    n_pool = int(counts.sum())
    e: dict[int, float] = {}
    for s_, m_ in zip(segs, counts, strict=True):
        s, m = int(s_), int(m_)
        if kind == "recovered":
            e[s] = float(min(RECOVERED_MIN, m))
        elif kind == "floor":
            e[s] = 1.0 if int((seg == s).sum()) >= FLOOR_SEGMENT_FRAMES else 0.0
        else:
            e[s] = min(float(SEGMENT_CAP), m, budget * m / n_pool)
    return e


def sample_frames(rows: FrameRows, seed: int = 0, target: int = TARGET_FRAMES) -> tuple[list[Pick], dict]:
    """Stratified, fixed-seed sample of rows; returns picks (union over strata, with inclusion probabilities) and
    a ``design`` dict (per-stratum pool sizes, budgets and picked counts) for the run params."""
    strata = _strata(rows)
    total = int(rows.total_frames or (rows.frame.max() + 1 if len(rows.frame) else 0))
    rng = np.random.default_rng(_seed_int(rows.match, seed))
    # keys are drawn for every (stratum, global frame) and (stratum, segment) up front, in a fixed order, so a
    # stratum's draw does not depend on what any other stratum's pool looks like
    n_seg_ids = int(rows.segment.max()) + 1 if len(rows.segment) else 0
    U = rng.random((len(strata), max(total, int(rows.frame.max()) + 1 if len(rows.frame) else 1)))
    V = rng.random((len(strata), max(n_seg_ids, 1)))

    seg = rows.segment
    fixed = 0.0
    e_fixed: dict[str, dict[int, float]] = {}
    for name, pool, kind in strata:
        if kind != "budget":
            e_fixed[name] = _expected_counts(kind, pool, seg, 0.0)
            fixed += sum(e_fixed[name].values())
    remaining = max(0.0, target - fixed)

    pi_s = np.zeros((len(strata), len(rows.frame)))
    picked = [np.zeros(len(rows.frame), bool) for _ in strata]
    design: dict[str, dict] = {}
    for si, (name, pool, kind) in enumerate(strata):
        budget = SHARES.get(name, 0.0) * remaining
        e = e_fixed.get(name) or _expected_counts(kind, pool, seg, budget)
        for s, ej in e.items():
            idx = np.nonzero(pool & (seg == s))[0]
            if len(idx) == 0 or ej <= 0:
                continue
            k = int(np.floor(ej)) + int(V[si, s] < (ej - np.floor(ej)))
            pi_s[si, idx] = ej / len(idx)
            if k:
                order = idx[np.argsort(U[si, rows.frame[idx]], kind="stable")]
                picked[si][order[:k]] = True
        design[name] = {"kind": kind, "pool_frames": int(pool.sum()), "budget": round(budget, 2),
                        "expected": round(float(sum(e.values())), 2), "picked": int(picked[si].sum())}
    any_pick = np.any(picked, axis=0)
    pi = 1.0 - np.prod(1.0 - pi_s, axis=0)
    picks = [Pick(int(r), [strata[si][0] for si in range(len(strata)) if picked[si][r]], float(pi[r]))
             for r in np.nonzero(any_pick)[0]]
    design["_total"] = {"target": target, "fixed_strata_expected": round(fixed, 2),
                        "remaining_for_budget_strata": round(remaining, 2), "picked_union": len(picks)}
    return picks, design


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------


def project_keypoints(H: np.ndarray, k1: float, pitch: PitchModel | None = None, *, aspect: float = geo.DEFAULT_ASPECT
                      ) -> list[dict]:
    """Correspondences of the pitch keypoints inside the image: stored ``H`` (undistorted normalised -> metres) ->
    inverse -> undistorted normalised -> division-model distortion with ``k1``. Out-of-image, behind-camera and
    no-distortion-root points are dropped. ``image`` is the observed normalised coordinate the page stores."""
    pitch = pitch or PitchModel()
    kps = pitch.keypoints()
    ids = list(kps)
    xy = np.array([kps[i] for i in ids], float)
    uv, ok = mc.project_pitch_to_image(H, xy, k1, aspect=aspect)
    inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] <= 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1)
    return [
        {"image": [float(uv[i, 0]), float(uv[i, 1])], "pitch_id": ids[i], "pitch_xy": [float(xy[i, 0]), float(xy[i, 1])],
         "confidence": 1.0, "candidate_id": None}
        for i in np.nonzero(inside)[0]
    ]


def build_record(rows: FrameRows, row: int, pick: Pick, pitch: PitchModel | None = None) -> dict | None:
    """The proposal record of one row (page format), or ``None`` when fewer than :data:`MIN_POINTS` keypoints are
    visible."""
    pitch = pitch or PitchModel()
    H, k1 = rows.H[row], float(rows.k1[row])
    corr = project_keypoints(H, k1 if np.isfinite(k1) else 0.0, pitch)
    if len(corr) < MIN_POINTS:
        return None
    f1 = float(rows.evidence_f1[row])
    rec = {
        "frame_index": int(rows.local_frame[row]),
        "correspondences": corr, "H": H.tolist(), "H_convention": "normalised_image_to_pitch_m",
        "pitch_dims": [pitch.length, pitch.width], "image_size": [mc.CANON_W, mc.CANON_H],
        "method": "ptz_v3_projection", "inliers": [True] * len(corr), "mean_err_m": None, "err_px": [None] * len(corr),
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "proposed", "score": f1 if np.isfinite(f1) else None, "failure": None, "view": None,
        "sampling": {
            "inclusion_prob": pick.pi, "strata": pick.strata, "global_frame": int(rows.frame[row]),
            "segment": int(rows.segment[row]), "stored_status": "accepted" if rows.accepted[row] else "calibrated",
            "segment_source": str(rows.source[row]),
        },
    }
    d = geo.make_distortion(k1) if np.isfinite(k1) else None
    if d is not None:  # absent == plain homography, as the page expects
        rec["distortion"] = d
    return rec


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def export_proposals(rows: FrameRows, gt_dir: Path, *, seed: int = 0, target: int = TARGET_FRAMES,
                     dry_run: bool = False) -> dict:
    """Sample, project and write ``<gt_dir>/calibration_proposals/<segment clip>.json`` (one run per version).

    A re-run replaces still-``proposed`` frames of the run and keeps every frame a reviewer already acted on.
    Returns a summary (counts per stratum, skipped frames, written clips)."""
    picks, design = sample_frames(rows, seed, target)
    pitch = PitchModel()
    run_id = rows.run_id
    by_clip: dict[str, dict[str, dict]] = {}
    skipped: list[dict] = []
    for p in picks:
        rec = build_record(rows, p.row, p, pitch)
        if rec is None:
            skipped.append({"global_frame": int(rows.frame[p.row]), "segment": int(rows.segment[p.row]),
                            "inclusion_prob": p.pi, "reason": f"< {MIN_POINTS} visible keypoints"})
            continue
        by_clip.setdefault(segment_clip(rows, int(rows.segment[p.row])), {})[str(rec["frame_index"])] = rec
    params = {
        "match": rows.match, "version": rows.version, "version_id": rows.version_id, "content_sha256": rows.content_sha256,
        "seed": seed, "target": target, "design": design, "shares": SHARES, "segment_cap": SEGMENT_CAP,
        "h_convention": mc.H_CONVENTION, "min_points": MIN_POINTS, "skipped": skipped,
    }
    created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not dry_run:
        for clip, frames in by_clip.items():
            def _store(doc, frames=frames):
                run = doc["runs"].setdefault(run_id, {"source": SOURCE, "created": created, "params": params, "frames": {}})
                run["params"] = params
                for k, rec in frames.items():
                    old = run["frames"].get(k)
                    if old is None or old.get("status") == "proposed":
                        run["frames"][k] = rec

            update_proposals(gt_dir, clip, _store)
    counts = {n: d["picked"] for n, d in design.items() if not n.startswith("_")}
    return {"run_id": run_id, "n_picked": len(picks), "n_written": len(picks) - len(skipped), "skipped_few_points": len(skipped),
            "n_clips": len(by_clip), "per_stratum": counts, "design": design,
            "by_clip": {c: sorted(int(k) for k in f) for c, f in by_clip.items()}}
