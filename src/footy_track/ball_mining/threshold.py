"""Learn-then-Test auto-accept threshold from audit verdicts (design §3.3).

Question: the lowest score threshold ``t`` such that the nodes with ``p_real >= t`` are at least ``target`` (default
0.98) precise, with probability >= ``1 - delta`` (default 0.9) over the audit sample.

Method (Angelopoulos et al., Learn then Test, with fixed-sequence testing): for a descending grid of thresholds the
null is ``H0(t): precision(s >= t) < target``. The audited verdicts with ``p_real >= t`` give ``k`` correct out of
``n``; the exact binomial (Clopper-Pearson) p-value of H0 is ``P(Bin(n, target) >= k)``. Walk the grid from the
highest ``t`` down, keep rejecting H0 (``p <= delta``), and stop at the first ``t`` that is not rejected. The last
rejected ``t`` is the answer. The sets ``s >= t`` are nested, so the single walk controls the family-wise error at
``delta`` without a multiplicity correction.

The walk starts at the highest grid point that *can* be rejected: with ``n`` audits at or above ``t`` and not a single
error the p-value is ``target ** n``, so a ``t`` with fewer than ``ceil(ln(delta) / ln(target))`` audits (114 at 0.98 / 0.1)
cannot be rejected whatever the labels say, and the walk would stop there before it began. Skipping such points depends
only on the ``p_real`` values of the audits, never on their verdicts, so the order stays label-independent.

Only **A-bucket** verdicts are valid (uniform random draws inside the auto-accepted region): U / M / N cards were
chosen *because* they were unusual, so they are not exchangeable with what auto-accept will see. Every row carries its
bucket tag and :func:`learn_then_test` raises :class:`ExchangeabilityError` on any tag other than ``"A"``; callers
filter explicitly with :func:`audit_rows`.

Correct = ``ball`` / ``corrected``; wrong = ``not_ball`` / ``box_off``; ``unsure`` is excluded.

If the rows carry unequal inclusion probabilities ``pi`` (A is stratified by match and band) the counts are
reduced to a Kish effective sample (rounded against us) before the exact test; with equal ``pi`` it is the plain exact test.

Nothing here needs scipy: the binomial tail is a log-space sum.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

CORRECT = ("ball", "corrected")
WRONG = ("not_ball", "box_off")
#: Interim rule while no threshold is valid: 0.95 on the Claude-verify confidence scale (design §3.3).
FALLBACK_THRESHOLD = 0.95
FALLBACK_SCALE = "claude_verify_conf"
DEFAULT_GRID = tuple(round(0.99 - 0.01 * i, 4) for i in range(50))  # 0.99 ... 0.50


class ExchangeabilityError(ValueError):
    """A verdict from outside the A (audit) bucket reached the threshold test."""


def binom_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)`` (exact)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    logs = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * lp
        + (n - i) * lq
        for i in range(k, n + 1)
    ]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.1) -> float:
    """One-sided ``1 - alpha`` lower confidence bound of a binomial proportion (exact)."""
    if n <= 0 or k <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_sf(k, n, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return lo


@dataclass
class LTTResult:
    status: str  # "valid" | "no_valid_threshold"
    threshold: float | None  # the LTT threshold, None when no valid one
    n: int  # audits with p_real >= threshold (or at the highest grid point when none)
    k: int  # of which correct
    precision: float | None  # k / n (weighted when pi varies)
    p_value: float | None
    target: float
    delta: float
    n_audits: int  # all usable audits (verdict not unsure)
    n_unsure: int
    effective_threshold: (
        float  # what auto-accept should use: the LTT t, or the fallback
    )
    effective_scale: (
        str  # "p_real" for a valid t, "claude_verify_conf" for the fallback
    )
    fallback: bool
    precision_lower_bound: float | None = (
        None  # exact 1 - delta lower bound at the chosen t
    )
    scale: str = "p_real"
    scorer_version: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def is_audit(row: dict) -> bool:
    return row.get("bucket") == "A"


def audit_rows(rows: Iterable[dict]) -> list[dict]:
    """The A-bucket rows (the only ones :func:`learn_then_test` accepts)."""
    return [r for r in rows if is_audit(r)]


def _usable(rows: Sequence[dict]) -> tuple[list[tuple[float, bool, float]], int]:
    out, unsure = [], 0
    for r in rows:
        v = r.get("verdict")
        if v == "unsure":
            unsure += 1
        elif v in CORRECT or v in WRONG:
            if r.get("p_real") is None:
                continue
            out.append(
                (
                    float(r["p_real"]),
                    v in CORRECT,
                    float(r["pi"]) if r.get("pi") else 1.0,
                )
            )
    return out, unsure


def _counts(sel: list[tuple[float, bool, float]]) -> tuple[int, int, float]:
    """``(n, k, precision)``; Kish-effective when the inclusion probabilities differ."""
    n = len(sel)
    if n == 0:
        return 0, 0, 0.0
    w = [1.0 / pi for _, _, pi in sel]
    if max(w) - min(w) <= 1e-9 * max(w):
        k = sum(c for _, c, _ in sel)
        return n, k, k / n
    sw, sw2 = sum(w), sum(x * x for x in w)
    prec = sum(x for x, (_, c, _) in zip(w, sel, strict=True) if c) / sw
    n_eff = int(sw * sw / sw2)
    k_eff = n_eff - math.ceil((1.0 - prec) * n_eff - 1e-9)
    return n_eff, max(k_eff, 0), prec


def learn_then_test(
    rows: Sequence[dict],
    *,
    target: float = 0.98,
    delta: float = 0.1,
    grid: Sequence[float] = DEFAULT_GRID,
    fallback: float = FALLBACK_THRESHOLD,
) -> LTTResult:
    """The lowest valid threshold (see the module docstring). ``rows``: dicts with ``bucket`` (must be ``"A"``),
    ``p_real``, ``verdict`` and optionally ``pi`` and ``scorer_version``."""
    bad = sorted({str(r.get("bucket")) for r in rows if not is_audit(r)})
    if bad:
        raise ExchangeabilityError(
            f"only A-bucket (audit) verdicts are exchangeable with the auto-accepted set; got bucket tag(s) {bad}"
        )
    versions = sorted(
        {str(r.get("scorer_version")) for r in rows if r.get("scorer_version")}
    )
    if len(versions) > 1:
        raise ExchangeabilityError(
            f"audits come from several scorer versions {versions}; p_real scales differ"
        )
    usable, n_unsure = _usable(rows)
    best = None
    first = None
    n_min = math.ceil(math.log(delta) / math.log(target)) if 0 < target < 1 else 1
    for t in sorted(grid, reverse=True):
        sel = [u for u in usable if u[0] >= t]
        n, k, prec = _counts(sel)
        if first is None and n < n_min and t > min(grid):
            continue  # untestable whatever the labels: see the module docstring
        p = binom_sf(k, n, target) if n else 1.0
        if first is None:
            first = (t, n, k, prec, p)
        if p <= delta:
            best = (t, n, k, prec, p)
        else:
            break
    common = {"target": target, "delta": delta, "n_audits": len(usable), "n_unsure": n_unsure,
              "scorer_version": versions[0] if versions else None}  # fmt: skip
    if best is None:
        t, n, k, prec, p = first if first else (None, 0, 0, None, None)
        return LTTResult("no_valid_threshold", None, n, k, prec if n else None, p, effective_threshold=fallback,
                         effective_scale=FALLBACK_SCALE, fallback=True, scale=FALLBACK_SCALE, **common)  # fmt: skip
    t, n, k, prec, p = best
    return LTTResult("valid", t, n, k, prec, p, effective_threshold=t, effective_scale="p_real", fallback=False,
                     precision_lower_bound=clopper_pearson_lower(k, n, delta), **common)  # fmt: skip


def learn_then_test_per_match(rows: Sequence[dict], **kw) -> dict[str, LTTResult]:
    """:func:`learn_then_test` on each match's audits (broadcaster shift is the likely failure, so the
    circuit breaker should trip per match). Rows need a ``match``."""
    bad = sorted({str(r.get("bucket")) for r in rows if not is_audit(r)})
    if bad:
        raise ExchangeabilityError(
            f"only A-bucket (audit) verdicts are exchangeable; got bucket tag(s) {bad}"
        )
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r.get("match")), []).append(r)
    return {m: learn_then_test(rs, **kw) for m, rs in sorted(by.items())}
