"""Level-2 rig consensus (``docs/design/calibration_match_flow.md`` section 2.2-2.4, PR P2). Pure numpy/scipy.

Input: one ``rig_estimate.json`` per level-1 segment solve::

    {"rig": [Cx, Cy, Cz, roll_rad], "cov": 4x4, "k1": float, "converged": bool, "n_anchors": int, ...}

Only ``rig`` is required; ``cov`` missing/non-finite makes the segment *uninformative*; ``converged`` defaults True.
The segment id is the directory name (``<match>_segNNN@...``).

Algorithm
---------
* **Fit** (``robust_consensus``): minimise ``sum_i rho(d_i^2)``, ``d_i^2 = (r_i - r*)^T S_i^-1 (r_i - r*)``, with the
  **Cauchy** loss ``rho(t) = c^2 log(1 + t/c^2)`` (``c^2`` = chi2(df=4) quantile ``cauchy_quantile``, default 0.5:
  it redescends, so a far second camera has ~no pull, unlike Huber). Solved by IRLS (weights ``w_i = 1/(1+d_i^2/c^2)``,
  closed-form weighted-least-squares step with full 4x4 precision matrices), initialised at the per-coordinate
  weighted median (weights 1/S_ii). ``Sigma*`` = ``(sum_i w_i S_i^-1)^-1`` (an approximation: it ignores
  between-segment scatter beyond S_i and is inflated by the down-weighting).
* Only informative, converged, finite segments enter the fit; every segment still gets ``d_i``.
* **Variants** (``find_variants``): outliers of the main consensus are clustered by repeated robust consensus seeded
  at the most precise unassigned outlier; its members are segments with ``d^2`` below the same threshold under the
  cluster centre (covariance-aware). Clusters with >= ``min_variant_size`` members are candidate second cameras
  (``variant:<k>``, k=1.. by weight), each with its own consensus. The main rig is the global robust consensus
  (assumes the main camera holds the majority of segments; ``main_weight_share`` = fraction of fitted segments consistent with it, is reported).
* **Flags** (precedence): ``unconverged`` > ``uninformative`` > ``inlier`` (d^2 <= chi2(df=4) ``outlier_quantile``,
  default 0.999) > ``variant:<k>`` > ``outlier``. ``uninformative`` = largest position-block sd > ``max_pos_sd_m``
  (10 m) or roll sd > ``max_roll_sd_rad`` (0.05 rad ~ 2.9 deg) or no/invalid covariance: never an outlier on deviation
  alone. A floor (``floor_pos_sd_m`` 0.05 m, ``floor_roll_sd_rad`` 3.5e-4) is added to each S_i for conditioning.

Label-free: nothing here reads GT; ``reference_distance`` only *reports* distance to an optional reference rig.
Limitations: parameter-space combination is exact only for near-Gaussian per-segment likelihoods; banana-shaped
(C_y / f ridge) ones are approximated by their Gaussian S_i. If rig* disagrees with the stored cost surfaces use the
pooled refit (b) of the design doc.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import chi2

DIM = 4


@dataclass
class ConsensusParams:
    cauchy_quantile: float = 0.5
    outlier_quantile: float = 0.999
    max_pos_sd_m: float = 10.0
    max_roll_sd_rad: float = 0.05
    floor_pos_sd_m: float = 0.05
    floor_roll_sd_rad: float = 3.5e-4
    min_variant_size: int = 2
    max_iter: int = 200
    tol: float = 1e-10

    @property
    def cauchy_c2(self) -> float:
        return float(chi2.ppf(self.cauchy_quantile, DIM))

    @property
    def outlier_d2(self) -> float:
        return float(chi2.ppf(self.outlier_quantile, DIM))


@dataclass
class SegmentEstimate:
    seg_id: str
    rig: np.ndarray  # (4,)
    cov: np.ndarray | None  # (4,4) or None
    converged: bool = True
    cost: float | None = None
    path: str = ""
    sha256: str = ""
    extra: dict = field(default_factory=dict)


def load_estimate(path: str | Path) -> SegmentEstimate:
    p = Path(path)
    raw = p.read_bytes()
    d = json.loads(raw)
    rig = np.asarray(d["rig"], float).reshape(DIM)
    cov = None
    if d.get("cov") is not None:
        try:
            c = np.asarray(d["cov"], float).reshape(DIM, DIM)
            cov = c if np.all(np.isfinite(c)) else None
        except ValueError:
            cov = None
    keys = (
        "k1",
        "n_anchors",
        "n_evidence_frames",
        "n_motion_pairs",
        "f_range",
        "pan_range_deg",
        "code_sha",
    )
    return SegmentEstimate(
        seg_id=p.parent.name,
        rig=rig,
        cov=cov,
        converged=bool(d.get("converged", True)),
        cost=d.get("cost"),
        path=str(p),
        sha256=hashlib.sha256(raw).hexdigest(),
        extra={k: d[k] for k in keys if k in d},
    )


def load_estimates(root_or_files) -> list[SegmentEstimate]:
    """A directory (searched for ``*/rig_estimate.json``, then recursively) or an explicit list of files."""
    if isinstance(root_or_files, (str, Path)):
        r = Path(root_or_files)
        files = sorted(r.glob("*/rig_estimate.json")) or sorted(
            r.rglob("rig_estimate.json")
        )
    else:
        files = [Path(f) for f in root_or_files]
    return [load_estimate(f) for f in files]


# ---------------------------------------------------------------------------- core maths


def _precisions(est: list[SegmentEstimate], prm: ConsensusParams):
    """Return (W precision (n,4,4), informative bool (n,))."""
    n = len(est)
    W = np.zeros((n, DIM, DIM))
    inf = np.zeros(n, bool)
    floor = np.diag([prm.floor_pos_sd_m**2] * 3 + [prm.floor_roll_sd_rad**2])
    for i, e in enumerate(est):
        if e.cov is None or not np.all(np.isfinite(e.rig)):
            continue
        S = 0.5 * (e.cov + e.cov.T)
        ev = np.linalg.eigvalsh(S[:3, :3])
        if ev[0] < -1e-9 or S[3, 3] < 0:
            continue
        if (
            np.sqrt(max(ev[-1], 0.0)) > prm.max_pos_sd_m
            or np.sqrt(S[3, 3]) > prm.max_roll_sd_rad
        ):
            continue
        inf[i] = True
        W[i] = np.linalg.inv(S + floor)
    return W, inf


def _d2(R, W, r):
    D = R - r
    return np.einsum("ni,nij,nj->n", D, W, D)


def _weighted_median(R, W):
    w = np.maximum(np.einsum("nii->ni", W), 1e-30)
    out = np.empty(DIM)
    for k in range(DIM):
        o = np.argsort(R[:, k])
        cw = np.cumsum(w[o, k])
        out[k] = R[o, k][min(np.searchsorted(cw, 0.5 * cw[-1]), len(o) - 1)]
    return out


def robust_consensus(R, W, prm: ConsensusParams | None = None, init=None):
    """Cauchy IRLS. R (n,4), W (n,4,4) precisions. Returns (r*, Sigma*, robust weights w_i, cost)."""
    prm = prm or ConsensusParams()
    c2 = prm.cauchy_c2
    r = _weighted_median(R, W) if init is None else np.asarray(init, float)
    for _ in range(prm.max_iter):
        w = 1.0 / (1.0 + _d2(R, W, r) / c2)
        A = np.einsum("n,nij->ij", w, W)
        b = np.einsum("n,nij,nj->i", w, W, R)
        rn = np.linalg.solve(A, b)
        done = np.max(np.abs(rn - r) / (1 + np.abs(r))) < prm.tol
        r = rn
        if done:
            break
    d2 = _d2(R, W, r)
    w = 1.0 / (1.0 + d2 / c2)
    Sigma = np.linalg.inv(np.einsum("n,nij->ij", w, W))
    cost = float(np.sum(c2 * np.log1p(d2 / c2)))
    return r, Sigma, w, cost


def find_variants(R, W, cand, prm: ConsensusParams):
    """Cluster candidate (outlier) indices into camera variants. Returns list of dicts (sorted by weight)."""
    left = [int(i) for i in cand]
    out = []
    thr = prm.outlier_d2
    while len(left) >= prm.min_variant_size:
        seed = left[int(np.argmax([np.trace(W[i]) for i in left]))]
        idx = np.array(left)
        r = R[seed].copy()
        members = [seed]
        for _ in range(20):
            members = [int(i) for i in idx[_d2(R[idx], W[idx], r) <= thr]] or [seed]
            if len(members) < 2:
                break
            r2 = robust_consensus(R[members], W[members], prm, init=r)[0]
            moved = not np.allclose(r2, r, atol=1e-9)
            r = r2
            if not moved:
                break
        if len(members) < prm.min_variant_size:
            left.remove(seed)
            continue
        rv, Sv, wv, cv = robust_consensus(R[members], W[members], prm, init=r)
        out.append(
            {
                "members": members,
                "rig": rv,
                "cov": Sv,
                "weight": float(np.trace(np.einsum("n,nij->ij", wv, W[members]))),
                "cost": cv,
            }
        )
        left = [i for i in left if i not in members]
    out.sort(key=lambda v: -v["weight"])
    return out


# ---------------------------------------------------------------------------- result


@dataclass
class ConsensusResult:
    rig: list
    cov: list
    cost: float
    params: dict
    thresholds: dict
    segments: list  # per-segment table (dicts)
    variants: list
    inputs: list  # [{"seg_id","path","sha256"}]
    main_weight_share: float
    n_fit: int
    reference: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ConsensusResult:
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> ConsensusResult:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def flags(self) -> dict[str, str]:
        return {s["seg_id"]: s["flag"] for s in self.segments}

    def summary(self) -> str:
        c = Counter(s["flag"].split(":")[0] for s in self.segments)
        r = self.rig
        sd = np.sqrt(np.diag(np.asarray(self.cov)))
        lines = [
            f"segments {len(self.segments)}  fit {self.n_fit}  "
            + "  ".join(f"{k} {v}" for k, v in sorted(c.items())),
            f"rig*  C=({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}) m  roll={np.degrees(r[3]):.3f} deg   "
            f"sd=({sd[0]:.3f}, {sd[1]:.3f}, {sd[2]:.3f}) m, {np.degrees(sd[3]):.3f} deg",
            f"main weight share {self.main_weight_share:.2f}  cost {self.cost:.2f}",
        ]
        for k, v in enumerate(self.variants, 1):
            vr = v["rig"]
            lines.append(
                f"variant:{k}  C=({vr[0]:.2f}, {vr[1]:.2f}, {vr[2]:.2f})  n={len(v['members'])}  {', '.join(v['members'][:8])}"
            )
        outl = [s["seg_id"] for s in self.segments if s["flag"] == "outlier"]
        if outl:
            lines.append(
                "outliers: " + ", ".join(outl[:12]) + (" ..." if len(outl) > 12 else "")
            )
        if self.reference:
            q = self.reference
            lines.append(
                f"vs reference (eval only): dC=({q['dC'][0]:+.3f}, {q['dC'][1]:+.3f}, {q['dC'][2]:+.3f}) |dC|={q['dC_norm']:.3f} m  droll={q['droll_deg']:+.3f} deg"
            )
        return "\n".join(lines)


def reference_distance(rig, ref) -> dict:
    """Distance of ``rig`` to a reference rig (eval reporting only; never enters the fit)."""
    d = np.asarray(rig, float) - np.asarray(ref, float)
    return {
        "dC": d[:3].tolist(),
        "dC_norm": float(np.linalg.norm(d[:3])),
        "droll_deg": float(np.degrees(d[3])),
    }


def compute_consensus(
    est: list[SegmentEstimate], prm: ConsensusParams | None = None, reference=None
) -> ConsensusResult:
    prm = prm or ConsensusParams()
    if not est:
        raise ValueError("no segment estimates")
    R = np.stack([e.rig for e in est])
    W, inf = _precisions(est, prm)
    conv = np.array([e.converged for e in est])
    finite = np.all(np.isfinite(R), axis=1)
    fit = inf & conv & finite
    if fit.sum() < 1:
        raise ValueError("no informative converged segments to fit")
    r, Sig, w, cost = robust_consensus(R[fit], W[fit], prm)
    d2 = np.full(len(est), np.nan)
    ok = inf & finite
    d2[ok] = _d2(R[ok], W[ok], r)
    thr = prm.outlier_d2
    rw = np.zeros(len(est))
    rw[ok] = 1.0 / (1.0 + d2[ok] / prm.cauchy_c2)
    cand = np.where(fit & (d2 > thr))[0]
    variants = find_variants(R, W, cand, prm) if len(cand) else []
    vlabel = {i: k for k, v in enumerate(variants, 1) for i in v["members"]}
    rows = []
    for i, e in enumerate(est):
        if not e.converged:
            flag = "unconverged"
        elif not inf[i]:
            flag = "uninformative"
        elif d2[i] <= thr:
            flag = "inlier"
        elif i in vlabel:
            flag = f"variant:{vlabel[i]}"
        else:
            flag = "outlier"
        rows.append(
            {
                "seg_id": e.seg_id,
                "d": None if np.isnan(d2[i]) else float(np.sqrt(d2[i])),
                "flag": flag,
                "weight": float(rw[i]),
                "rig": R[i].tolist(),
                "informative": bool(inf[i]),
            }
        )
    share = float(np.mean(d2[fit] <= thr))
    out_variants = [
        {
            "rig": v["rig"].tolist(),
            "cov": v["cov"].tolist(),
            "members": [est[i].seg_id for i in v["members"]],
            "weight": v["weight"],
            "cost": v["cost"],
        }
        for v in variants
    ]
    return ConsensusResult(
        rig=r.tolist(),
        cov=Sig.tolist(),
        cost=cost,
        params=asdict(prm),
        thresholds={
            "cauchy_c2": prm.cauchy_c2,
            "outlier_d2": thr,
            "outlier_d": float(np.sqrt(thr)),
        },
        segments=rows,
        variants=out_variants,
        inputs=[{"seg_id": e.seg_id, "path": e.path, "sha256": e.sha256} for e in est],
        main_weight_share=share,
        n_fit=int(fit.sum()),
        reference=reference_distance(r, reference) if reference is not None else None,
    )


def stop_rule(
    prev: ConsensusResult,
    new: ConsensusResult,
    max_dC_m: float = 0.2,
    max_droll_deg: float = 0.05,
    max_cost_rel: float = 0.005,
) -> dict:
    """Section 2.4 stop rule: rig* change, no flag flips (on common segments), relative cost change."""
    a, b = np.asarray(prev.rig), np.asarray(new.rig)
    dC = float(np.max(np.abs(a[:3] - b[:3])))
    dr = float(abs(np.degrees(a[3] - b[3])))
    fa, fb = prev.flags(), new.flags()
    flips = sorted(s for s in fa.keys() & fb.keys() if fa[s] != fb[s])
    cost_rel = abs(new.cost - prev.cost) / max(abs(prev.cost), 1e-12)
    reasons = []
    if dC >= max_dC_m:
        reasons.append(f"C moved {dC:.3f} m")
    if dr >= max_droll_deg:
        reasons.append(f"roll moved {dr:.3f} deg")
    if flips:
        reasons.append(f"{len(flips)} flag flips")
    if cost_rel >= max_cost_rel:
        reasons.append(f"cost changed {100 * cost_rel:.2f}%")
    return {
        "stop": not reasons,
        "dC_max_m": dC,
        "droll_deg": dr,
        "flips": flips,
        "cost_rel_change": float(cost_rel),
        "reasons": reasons,
    }
