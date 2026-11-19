"""``mine_round``: pick one round of Ball Check cards from a train-match pool (design §3.2, §4).

Rules enforced in code, not by convention:

* **Train matches only.** Every (match, clip) goes through ``split_of``; heldout matches, in-match eval clips and
  matches no register knows are *refused* (dropped and counted, or ``strict=True`` raises :class:`LeakageError`).
  Nodes within ``eval_buffer_s`` (10 s) of footage time of an in-match eval clip are refused too. A final assertion
  re-checks every card.
* **Dedupe.** At most one card per clip per ``window_s`` (2 s) window, across all buckets and across rounds: a window
  that already holds a *live* verdict (``already_labelled``; tombstones do not count) or a pending card is not
  mined again.
* **Spacing and per-clip cap** (``BucketConfig.exclude_s`` = 3, ``max_per_clip`` = 3). No card lies within +-``exclude_s``
  s (``frame / fps``, a true distance, not a window) of a verdicted / pending / already-issued card of any round in the
  same clip (U, M, N, R, A; gold is exempt), nor of another card of this round. At most ``max_per_clip`` U / M / N / R cards
  per clip per round. The R / A frame is the set of units left after these rules and after the earlier buckets, so ``pi``
  is exact over that frame; two draws of one bucket that land within ``exclude_s`` of each other are dropped
  (``report["diversity"]["spacing_rejects"]``), which makes ``pi`` nominal for those few.
* **Per-match cap** (default 15 % of the budget). When fewer than ``1 / cap`` matches have anything to mine, the cap
  relaxes to an equal share (reported), otherwise a one-match pool could never fill a round.
* **U and M are diverse**: top-ranked nodes per match are clustered (k-means on the appearance embedding if the pool
  has one, else position, plus footage time) and only the best node of each cluster is a candidate.
* **Quota spill.** What a bucket cannot fill goes to R.
* **R and A are probability samples with logged inclusion probabilities.** The frame is a set of *units*
  (clip, 2 s window) not already holding a card; a unit is drawn uniformly, then a node uniformly inside the unit.
  For a stratum (R: the match; A: the match and p_real band) of ``N`` units from which ``n`` were drawn and a unit
  with ``m`` eligible nodes, ``pi = n / N * 1 / m``. It is exact conditional on the cards the earlier buckets took
  (the buckets are taken U, M, N, A, G, R), which is what ``sampling.frame_units`` records.

Cards are ball_check *inline* items (``labeller/jobs.py``). Only ``clip``, ``frame_index`` and ``bbox`` reach the
phone: every card, gold or not, carries no ``label`` / ``provenance`` / ``confidence`` / ``note``, and everything
else (``sampling``, ``p_real``, ...) is a hidden key that only comes back in the job results.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from footy_track.ball_mining.buckets import DEFAULT_QUOTAS, BucketConfig, bucket_flags
from footy_track.ball_mining.clips import neighbour_clips
from footy_track.ball_mining.schema import validate_pool
from footy_track.clip_id import canonical_clip_id

LOGGER = logging.getLogger(__name__)

TRAIN = "train"
EVAL_HELDOUT = "eval_heldout_match"
EVAL_IN_MATCH = "eval_in_match"
UNREGISTERED = "unregistered"
ORDER = (
    "U",
    "M",
    "N",
    "A",
    "G",
    "R",
)  # selection order; R last (its frame is what is left)

SplitOf = Callable[[str, str], str]


class LeakageError(RuntimeError):
    """A heldout / eval / unregistered row reached a training sampler (or ``strict`` met one)."""


def default_split_of(match: str, clip: str) -> str:
    """The repo's split authority: ``train`` / ``eval_in_match`` / ``eval_heldout_match``, or ``unregistered``
    for a match no register knows (fail closed: a not-yet-assigned match could be a heldout one)."""
    from footy_track import splits  # noqa: PLC0415

    if splits.unregistered_match(match, clip):
        return UNREGISTERED
    return splits.split_of(match, clip)


def default_eval_clips() -> list[tuple[str, str]]:
    """``(match, clip)`` of every in-match eval clip in the clip register (they need not be in the pool: the +-10 s rule
    must see them anyway)."""
    from footy_track import splits  # noqa: PLC0415

    reg = splits.load_clip_register()
    return [
        (e["match"], cid)
        for cid, e in reg["clips"].items()
        if e.get("split") == splits.CLIP_EVAL and e.get("match")
    ]


def default_match_of(clip: str) -> str | None:
    from footy_track import splits  # noqa: PLC0415

    return splits.match_key_of(clip)


@dataclass
class MineResult:
    cards: list[dict]
    report: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(self.cards)

    def __len__(self) -> int:
        return len(self.cards)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def allocate(total: int, weights: Mapping, capacity: Mapping) -> dict:
    """Split an integer ``total`` over keys in proportion to ``weights``, never above ``capacity`` (largest
    remainder; overflow is redistributed over the keys with room left)."""
    out = dict.fromkeys(weights, 0)
    remaining = int(total)
    active = [k for k in weights if capacity.get(k, 0) > 0 and weights[k] > 0]
    while remaining > 0 and active:
        wsum = float(sum(weights[k] for k in active))
        raw = {k: remaining * weights[k] / wsum for k in active}
        base = {k: min(int(math.floor(raw[k])), capacity[k] - out[k]) for k in active}
        given = sum(base.values())
        left = remaining - given
        order = sorted(active, key=lambda k: (-(raw[k] - math.floor(raw[k])), str(k)))
        for k in order:
            if left <= 0:
                break
            if out[k] + base[k] < capacity[k]:
                base[k] += 1
                left -= 1
        for k in active:
            out[k] += base[k]
        got = given + (remaining - given - left)
        remaining -= got
        active = [k for k in active if out[k] < capacity[k]]
        if got == 0:
            break
    return out


def kmeans(
    x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25
) -> np.ndarray:
    """Plain Lloyd k-means with k-means++ seeding; returns the cluster label of every row."""
    n = len(x)
    k = max(1, min(k, n))
    if k == n:
        return np.arange(n)
    x = x.astype(np.float32, copy=False)
    centres = np.empty((k, x.shape[1]), np.float32)
    centres[0] = x[rng.integers(n)]
    d2 = ((x - centres[0]) ** 2).sum(1)
    for i in range(1, k):
        tot = float(d2.sum())
        j = rng.choice(n, p=d2 / tot) if tot > 0 else rng.integers(n)
        centres[i] = x[j]
        d2 = np.minimum(d2, ((x - centres[i]) ** 2).sum(1))
    labels = np.zeros(n, np.int64)
    for _ in range(iters):
        dist = (
            (x**2).sum(1, keepdims=True)
            - 2 * x @ centres.T
            + (centres**2).sum(1)[None, :]
        )
        new = dist.argmin(1)
        if (new == labels).all() and _ > 0:
            break
        labels = new
        for c in range(k):
            sel = labels == c
            if sel.any():
                centres[c] = x[sel].mean(0)
    return labels


# ---------------------------------------------------------------------------
# Gate: split, eval buffer, already labelled
# ---------------------------------------------------------------------------


class _Gate:
    def __init__(
        self,
        *,
        split_of: SplitOf,
        match_of: Callable[[str], str | None],
        cfg: BucketConfig,
        clip_spans,
        strict: bool,
    ):
        self.split_of = split_of
        self.cfg = cfg
        self.strict = strict
        self.spans = {
            canonical_clip_id(k): (float(v[0]), float(v[1]))
            for k, v in (clip_spans or {}).items()
        }
        self.match_of = match_of
        self.wid_of: dict[tuple, int] = {}
        self._split_cache: dict[tuple, str] = {}
        self.eval_spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.eval_clips: dict[str, set[str]] = defaultdict(set)
        self.blocked: set[int] = set()
        self.lab_t: dict[str, list[float]] = defaultdict(list)  # clip-local seconds of every labelled / pending card
        self.report: dict = {"refused": {}, "excluded": Counter()}

    def split(self, match: str, clip: str) -> str:
        key = (match, canonical_clip_id(clip))
        if key not in self._split_cache:
            self._split_cache[key] = self.split_of(match, clip)
        return self._split_cache[key]

    def register_known_clips(self, pairs: Iterable[tuple[str, str]]) -> None:
        """Learn which clips are in-match eval (from the pool and from the spans file) for the +-buffer rule."""
        seen = set()
        for match, clip in pairs:
            cid = canonical_clip_id(clip)
            if (match, cid) in seen:
                continue
            seen.add((match, cid))
            if self.split(match, clip) == EVAL_IN_MATCH:
                self.eval_clips[match].add(cid)
        for cid in self.spans:
            m = self.match_of(cid)
            if m and (m, cid) not in seen:
                seen.add((m, cid))
                if self.split(m, cid) == EVAL_IN_MATCH:
                    self.eval_clips[m].add(cid)
        for m, cids in self.eval_clips.items():
            for cid in cids:
                if cid in self.spans:
                    self.eval_spans[m].append(self.spans[cid])

    def wid(self, cid: str, win: int) -> int:
        return self.wid_of.setdefault((cid, int(win)), len(self.wid_of))

    def block_labelled(
        self, labelled: Iterable[tuple[str, int]], fps_of: Mapping[str, float]
    ) -> None:
        for clip, frame in labelled:
            cid = canonical_clip_id(clip)
            fps = fps_of.get(cid)
            if fps:
                self.blocked.add(self.wid(cid, int(frame // fps // self.cfg.window_s)))
                self.lab_t[cid].append(frame / fps)

    def apply(self, df: pd.DataFrame, what: str) -> pd.DataFrame:
        """Refuse / exclude rows; return the clean frame with ``_cid _win _wid _t`` columns."""
        if df.empty:
            return df.assign(_cid="", _win=0, _wid=0, _t=np.nan)
        df = df.copy()
        df["_cid"] = df["clip"].map(canonical_clip_id)
        pairs = df[["match", "clip"]].drop_duplicates()
        split = {
            (m, canonical_clip_id(c)): self.split(m, c)
            for m, c in zip(pairs["match"], pairs["clip"], strict=True)
        }
        df["_split"] = [
            split[(m, c)] for m, c in zip(df["match"], df["_cid"], strict=True)
        ]
        bad = df[df["_split"] != TRAIN]
        if len(bad):
            counts = bad.groupby("_split").size().to_dict()
            self.report["refused"][what] = {
                "rows": {k: int(v) for k, v in counts.items()},
                "matches": sorted(bad["match"].unique().tolist()),
            }
            if self.strict:
                raise LeakageError(
                    f"{what}: refused non-train rows {counts} in matches {sorted(bad['match'].unique())}"
                )
            LOGGER.warning("%s: refused %s rows %s", what, len(bad), counts)
            df = df[df["_split"] == TRAIN]
        df["_win"] = np.floor(
            df["frame"].to_numpy(dtype=float)
            / df["fps"].to_numpy(dtype=float)
            / self.cfg.window_s
        ).astype(np.int64)
        span = df["_cid"].map(lambda c: self.spans.get(c))
        df["_t"] = [
            s[0] + f / fps if s is not None else np.nan
            for s, f, fps in zip(span, df["frame"], df["fps"], strict=True)
        ]
        keep = np.ones(len(df), bool)
        n_span = n_fallback = 0
        buf = self.cfg.eval_buffer_s
        for m, idx in df.groupby("match").indices.items():
            spans = self.eval_spans.get(m, [])
            t = df["_t"].to_numpy()[idx]
            known = ~np.isnan(t)
            for lo, hi in spans:
                hit = known & (t >= lo - buf) & (t <= hi + buf)
                n_span += int(hit.sum())
                keep[idx[hit]] = False
            if self.eval_clips.get(m):
                near = set()
                for cid in self.eval_clips[m]:
                    near.update(neighbour_clips(cid))
                cids = df["_cid"].to_numpy()[idx]
                hit = ~known & np.array([c in near for c in cids])
                n_fallback += int(hit.sum())
                keep[idx[hit]] = False
        if n_span:
            self.report["excluded"]["eval_buffer_footage_time"] += n_span
        if n_fallback:
            self.report["excluded"]["eval_buffer_neighbour_segment_fallback"] += (
                n_fallback
            )
        df = df[keep]
        df["_wid"] = [
            self.wid(c, w) for c, w in zip(df["_cid"], df["_win"], strict=True)
        ]
        before = len(df)
        df = df[~df["_wid"].isin(self.blocked)]
        if before - len(df):
            self.report["excluded"]["already_labelled_window"] += before - len(df)
        if what == "pool" and self.cfg.exclude_s > 0 and self.lab_t and len(df):
            before = len(df)
            df = df[~self._near_labelled(df)]
            if before - len(df):
                self.report["excluded"]["already_labelled_neighbourhood"] += before - len(df)
        return df

    def _near_labelled(self, df: pd.DataFrame) -> np.ndarray:
        """Rows within ``exclude_s`` (clip-local seconds, ``frame / fps``) of a labelled or pending card of the same clip."""
        out = np.zeros(len(df), bool)
        cids = df["_cid"].to_numpy()
        t = df["frame"].to_numpy(dtype=float) / df["fps"].to_numpy(dtype=float)
        for cid in set(cids.tolist()) & set(self.lab_t):
            pos = np.flatnonzero(cids == cid)
            out[pos] = _within(t[pos], np.sort(np.asarray(self.lab_t[cid])), self.cfg.exclude_s)
        return out


# ---------------------------------------------------------------------------
# The miner
# ---------------------------------------------------------------------------


def _within(t: np.ndarray, ref_sorted: np.ndarray, dist: float) -> np.ndarray:
    """``True`` where ``t`` is within ``dist`` (inclusive) of any value of the sorted ``ref_sorted``."""
    i = np.searchsorted(ref_sorted, t)
    lo = np.abs(t - ref_sorted[np.clip(i - 1, 0, len(ref_sorted) - 1)])
    hi = np.abs(t - ref_sorted[np.clip(i, 0, len(ref_sorted) - 1)])
    return np.minimum(lo, hi) <= dist + 1e-9


def _reason(bucket: str, row, extra: str = "") -> str:
    p = float(row["p_real"])
    base = {
        "U": f"uncertain: p_real {p:.2f}",
        "M": f"missed: gap node, plausibility {p:.2f}",
        "N": f"hard negative: {'static' if row.get('static_flag') == True else 'off-track'} node, score {max(row['source_scores']):.2f}",  # noqa: E712
        "R": f"random draw, p_real {p:.2f}",
        "A": f"auto-accept audit, p_real {p:.2f}",
    }[bucket]
    return f"{base}{('; ' + extra) if extra else ''}"


def _bbox(cx: float, cy: float, w: float, h: float) -> dict:
    w = min(max(float(w), 1e-4), 1.0)
    h = min(max(float(h), 1e-4), 1.0)
    return {
        "x": round(min(max(cx - w / 2, 0.0), 1.0 - w), 6),
        "y": round(min(max(cy - h / 2, 0.0), 1.0 - h), 6),
        "w": round(w, 6),
        "h": round(h, 6),
    }


def mine_round(  # noqa: PLR0912, PLR0913, PLR0915
    pool: pd.DataFrame,
    *,
    budget: int = 300,
    quotas: Mapping[str, float] | None = None,
    per_match_cap: float = 0.15,
    already_labelled: Iterable[tuple[str, int]] = (),
    split_of: SplitOf | None = None,
    seed: int = 0,
    cfg: BucketConfig | None = None,
    gold: pd.DataFrame | None = None,
    clip_spans: Mapping[str, tuple[float, float]] | None = None,
    match_of: Callable[[str], str | None] | None = None,
    round_key: str | None = None,
    strict: bool = False,
    eval_clips: Iterable[tuple[str, str]] | None = None,
) -> MineResult:
    """Pick up to ``budget`` cards (see the module docstring). ``already_labelled`` is ``(clip, frame)`` of every
    live verdict and pending card; ``split_of(match, clip)`` returns ``"train"`` for minable clips. ``eval_clips`` is
    ``(match, clip)`` of in-match eval clips that may be absent from the pool (default: the clip register when
    ``split_of`` is the default one)."""
    cfg = cfg or BucketConfig()
    quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    if (
        set(quotas) - set(ORDER)
        or any(v < 0 for v in quotas.values())
        or not 0.99 <= sum(quotas.values()) <= 1.01
    ):
        raise ValueError(f"quotas must name buckets {ORDER} and sum to 1, got {quotas}")
    for b in ORDER:
        quotas.setdefault(b, 0.0)
    if eval_clips is None and split_of is None:
        eval_clips = default_eval_clips()
    split_of = split_of or default_split_of
    match_of = match_of or default_match_of
    pool = validate_pool(pool)
    rng_root = np.random.SeedSequence(seed)
    rngs = dict(
        zip(
            ORDER,
            (np.random.default_rng(s) for s in rng_root.spawn(len(ORDER))),
            strict=True,
        )
    )
    rng_order = np.random.default_rng(rng_root.spawn(len(ORDER) + 1)[-1])

    gate = _Gate(
        split_of=split_of,
        match_of=match_of,
        cfg=cfg,
        clip_spans=clip_spans,
        strict=strict,
    )
    known = list(zip(pool["match"], pool["clip"], strict=True))
    if gold is not None and len(gold):
        known += list(zip(gold["match"], gold["clip"], strict=True))
    known += list(eval_clips or [])
    gate.register_known_clips(set(known))
    # Pool fps wins: gold rows of a pool clip (a not_ball verdict of an earlier round) carry a default 25 fps, which used to
    # overwrite the clip's real rate (50 fps broadcast) and put its labelled frames in the wrong window.
    fps_of: dict[str, float] = {}
    if gold is not None and len(gold):
        fps_of.update(zip(gold["clip"].map(canonical_clip_id), gold["fps"].astype(float), strict=True))
    fps_of.update(zip(pool["clip"].map(canonical_clip_id), pool["fps"].astype(float), strict=True))
    if gold is not None and len(gold):
        gold = gold.assign(fps=gold["clip"].map(canonical_clip_id).map(fps_of).astype(np.float32))
    gate.block_labelled(already_labelled, fps_of)
    n_in = len(pool)
    df = gate.apply(pool, "pool").reset_index(drop=True)
    gdf = (
        gate.apply(gold, "gold").reset_index(drop=True)
        if gold is not None and len(gold)
        else None
    )
    flags = (
        bucket_flags(df, cfg)
        if len(df)
        else pd.DataFrame(
            columns=["U", "M", "N", "A", "a_band", "u_rank", "m_rank", "n_rank"]
        )
    )

    matches = sorted(df["match"].unique().tolist()) if len(df) else []
    cap_n = max(1, math.ceil(per_match_cap * budget))
    cap_relaxed = False
    if (
        matches and len(matches) * cap_n < budget
    ):  # only matches with pool nodes count; gold cards are a few
        cap_n = math.ceil(budget / len(matches))
        cap_relaxed = True
    used: Counter = Counter()
    taken: set[int] = set()
    cards: list[dict] = []
    seed_meta = {"seed": seed, "round": round_key}

    def room(m: str) -> int:
        return cap_n - used[m]

    capped_buckets = ("U", "M", "N", "R")
    clip_used: Counter = Counter()  # U / M / N / R cards per clip this round
    taken_t: dict[str, list[float]] = defaultdict(list)  # clip-local seconds of every card taken this round
    spacing_rejects: Counter = Counter()

    def clip_ok(row, bucket: str) -> bool:
        """Within-round rules for one more card: <= ``max_per_clip`` (U/M/N/R) and > ``exclude_s`` from every card taken."""
        cid = row["_cid"]
        if bucket in capped_buckets and cfg.max_per_clip > 0 and clip_used[cid] >= cfg.max_per_clip:
            return False
        if cfg.exclude_s > 0 and cid in taken_t:
            t = np.array([float(row["frame"]) / float(row["fps"])])
            if _within(t, np.sort(np.asarray(taken_t[cid])), cfg.exclude_s)[0]:
                return False
        return True

    def free_mask(d: pd.DataFrame, bucket: str) -> np.ndarray:
        """Rows of ``d`` still allowed given the cards taken so far (window, spacing, per-clip cap): the frame of a
        probability bucket is this set, so its pi is exact conditional on the earlier buckets."""
        ok = ~d["_wid"].isin(taken).to_numpy()
        if d.empty or not taken_t:
            return ok
        cids = d["_cid"].to_numpy()
        t = d["frame"].to_numpy(dtype=float) / d["fps"].to_numpy(dtype=float)
        for cid in set(cids.tolist()) & set(taken_t):
            pos = np.flatnonzero(cids == cid)
            if bucket in capped_buckets and cfg.max_per_clip > 0 and clip_used[cid] >= cfg.max_per_clip:
                ok[pos] = False
            elif cfg.exclude_s > 0:
                ok[pos] &= ~_within(t[pos], np.sort(np.asarray(taken_t[cid])), cfg.exclude_s)
        return ok

    def take(
        row, bucket: str, sampling: dict, *, gold_card: dict | None = None
    ) -> None:
        taken.add(int(row["_wid"]))
        used[row["match"]] += 1
        cid = row["_cid"]
        taken_t[cid].append(float(row["frame"]) / float(row["fps"]))
        if bucket in capped_buckets:
            clip_used[cid] += 1
        if gold_card is not None:
            cx, cy, w, h = (
                gold_card["cx"],
                gold_card["cy"],
                gold_card["w"],
                gold_card["h"],
            )
        else:
            cx, cy, w, h = (
                float(row["x"]),
                float(row["y"]),
                float(row["w"]),
                float(row["h"]),
            )
        item = {
            "clip": row["clip"],
            "frame_index": int(row["frame"]),
            "bbox": _bbox(cx, cy, w, h),
            "match": row["match"],
            "p_real": None if gold_card is not None else round(float(row["p_real"]), 5),
            "scorer_version": None
            if gold_card is not None
            else str(row["scorer_version"]),
            "detector_sha": None if gold_card is not None else str(row["detector_sha"]),
            "sources": ["gold"] if gold_card is not None else list(row["sources"]),
            "sampling": {
                "regime": "train",
                "bucket": bucket,
                "split": TRAIN,
                "match": row["match"],
                "window_s": cfg.window_s,
                "scorer_version": None
                if gold_card is not None
                else str(row["scorer_version"]),
                **seed_meta,
                **sampling,
            },  # fmt: skip
        }
        cards.append(item)

    # ---- U and M: diverse, ranked -------------------------------------------------------------------------
    def pick_ranked(bucket: str, target: int, rank_col: str, ascending: bool) -> int:
        if target <= 0 or df.empty:
            return 0
        sel = df[flags[bucket].to_numpy() & free_mask(df, bucket)]
        sel = sel.assign(_rank=flags.loc[sel.index, rank_col])
        per_match: dict[str, list] = {}
        elig = {m: int((sel["match"] == m).sum()) for m in matches}
        k_total = int(math.ceil(target * cfg.cluster_overshoot))
        ks = allocate(
            k_total,
            dict.fromkeys(matches, 1.0),
            {m: min(elig[m], max(room(m), 0)) for m in matches},
        )
        for m in matches:
            d = sel[sel["match"] == m]
            if not len(d) or ks[m] <= 0:
                continue
            d = d.sort_values(
                ["_rank", "frame"], ascending=[ascending, True], kind="stable"
            ).head(max(cfg.cluster_candidates, 20 * ks[m]))
            reps = _cluster_reps(d, ks[m], cfg, rngs[bucket], ascending)
            per_match[m] = reps
        got = 0
        pos = 0
        while got < target and any(pos < len(v) for v in per_match.values()):
            for m in matches:
                lst = per_match.get(m, [])
                if got >= target:
                    break
                if pos >= len(lst) or room(m) <= 0:
                    continue
                row, cluster = lst[pos]
                if int(row["_wid"]) in taken or not clip_ok(row, bucket):
                    continue
                take(row, bucket, {"stratum": f"{m}|cluster{cluster}", "reason": _reason(bucket, row, f"cluster {cluster}"),
                                   "pi": None, "rank": float(row["_rank"])})  # fmt: skip
                got += 1
            pos += 1
        return got

    # ---- N: hard negatives, location capped -----------------------------------------------------------------
    def pick_n(target: int) -> int:
        if target <= 0 or df.empty:
            return 0
        sel = df[flags["N"].to_numpy() & free_mask(df, "N")]
        sel = sel.assign(_rank=flags.loc[sel.index, "n_rank"])
        cell = cfg.n_cell
        sel = sel.assign(
            _loc=list(
                zip(
                    sel["_cid"],
                    (sel["x"] // cell).astype(int),
                    (sel["y"] // cell).astype(int),
                    strict=True,
                )
            )
        )
        lists = {
            m: sel[sel["match"] == m].sort_values(
                ["_rank", "frame"], ascending=[False, True], kind="stable"
            )
            for m in matches
        }
        loc_count: Counter = Counter()
        cursors = dict.fromkeys(matches, 0)
        got = 0
        progressed = True
        while got < target and progressed:
            progressed = False
            for m in matches:
                if got >= target:
                    break
                d = lists[m]
                while cursors[m] < len(d) and room(m) > 0:
                    row = d.iloc[cursors[m]]
                    cursors[m] += 1
                    if (
                        int(row["_wid"]) in taken
                        or loc_count[row["_loc"]] >= cfg.n_per_location
                        or not clip_ok(row, "N")
                    ):
                        continue
                    loc_count[row["_loc"]] += 1
                    take(row, "N", {"stratum": f"{m}|loc{row['_loc'][1]}_{row['_loc'][2]}", "reason": _reason("N", row), "pi": None,
                                    "rank": float(row["_rank"])})  # fmt: skip
                    got += 1
                    progressed = True
                    break
        return got

    # ---- R and A: probability samples with logged pi -------------------------------------------------------
    def draw_units(
        d: pd.DataFrame, n: int, rng: np.random.Generator
    ) -> list[tuple[pd.Series, float, int]]:
        """Uniform draw of ``n`` units from the nodes in ``d``; ``(row, pi, frame_units)`` for each."""
        if n <= 0 or d.empty:
            return []
        groups = d.groupby("_wid").indices
        units = list(groups)
        n_units = len(units)
        chosen = rng.choice(n_units, size=min(n, n_units), replace=False)
        out = []
        for j in chosen:
            idx = groups[units[j]]
            m_unit = len(idx)
            row = d.iloc[int(idx[int(rng.integers(m_unit))])]
            out.append((row, len(chosen) / n_units / m_unit, n_units))
        return out

    inclusion: dict[str, dict] = {"R": {}, "A": {}}

    def pick_a(target: int) -> int:
        if target <= 0 or df.empty:
            return 0
        sel = df[flags["A"].to_numpy() & free_mask(df, "A")]
        sel = sel.assign(_band=flags.loc[sel.index, "a_band"])
        n_units_m = {
            m: int(sel.loc[sel["match"] == m, "_wid"].nunique()) for m in matches
        }
        per_m = allocate(
            target,
            dict.fromkeys(matches, 1.0),
            {m: min(n_units_m[m], max(room(m), 0)) for m in matches},
        )
        got = 0
        for m in matches:
            dm = sel[sel["match"] == m]
            bands = sorted(dm["_band"].unique().tolist())
            if per_m[m] <= 0 or not bands:
                continue
            w = {b: float(dm.loc[dm["_band"] == b, "_wid"].nunique()) for b in bands}
            cap_b = {b: int(w[b]) for b in bands}
            n_b = allocate(per_m[m], w, cap_b)
            if (
                per_m[m] >= len(bands) * cfg.a_min_per_band
            ):  # every non-empty band gets a floor of audits
                short = [b for b in bands if n_b[b] < min(cfg.a_min_per_band, cap_b[b])]
                for b in short:
                    donor = max(bands, key=lambda x: n_b[x])
                    need = min(cfg.a_min_per_band, cap_b[b]) - n_b[b]
                    if n_b[donor] - need >= min(cfg.a_min_per_band, cap_b[donor]):
                        n_b[donor] -= need
                        n_b[b] += need
            for b in bands:
                db = dm[(dm["_band"] == b) & ~dm["_wid"].isin(taken)]
                for row, pi, n_units in draw_units(db, n_b[b], rngs["A"]):
                    if room(m) <= 0 or int(row["_wid"]) in taken:
                        continue
                    if not clip_ok(row, "A"):  # two draws of this bucket within exclude_s of each other
                        spacing_rejects["A"] += 1
                        continue
                    n_drawn = n_b[b]
                    take(row, "A", {"stratum": f"{m}|band{cfg.band_label(int(b))}", "band": cfg.band_label(int(b)), "pi": round(pi, 8),
                                    "frame_units": n_units, "n_drawn": n_drawn, "reason": _reason("A", row)})  # fmt: skip
                    inclusion["A"].setdefault(
                        f"{m}|{cfg.band_label(int(b))}",
                        {
                            "frame_units": n_units,
                            "n_drawn": n_drawn,
                            "pi_node_mean": 0.0,
                            "_pis": [],
                        },
                    )["_pis"].append(pi)
                    got += 1
        return got

    def pick_r(target: int) -> int:
        if target <= 0 or df.empty:
            return 0
        sel = df[free_mask(df, "R")]
        n_units_m = {
            m: int(sel.loc[sel["match"] == m, "_wid"].nunique()) for m in matches
        }
        per_m = allocate(
            target,
            dict.fromkeys(matches, 1.0),
            {m: min(n_units_m[m], max(room(m), 0)) for m in matches},
        )
        got = 0
        for m in matches:
            dm = sel[sel["match"] == m]
            for row, pi, n_units in draw_units(dm, per_m[m], rngs["R"]):
                if room(m) <= 0 or not clip_ok(row, "R"):
                    spacing_rejects["R"] += 1
                    continue
                take(
                    row,
                    "R",
                    {
                        "stratum": m,
                        "pi": round(pi, 8),
                        "frame_units": n_units,
                        "n_drawn": per_m[m],
                        "reason": _reason("R", row),
                    },
                )
                inclusion["R"].setdefault(
                    m, {"frame_units": n_units, "n_drawn": per_m[m], "_pis": []}
                )["_pis"].append(pi)
                got += 1
        return got

    # ---- G: gold -----------------------------------------------------------------------------------------------
    def pick_g(target: int) -> int:
        if target <= 0 or gdf is None or gdf.empty:
            return 0
        order = rngs["G"].permutation(len(gdf))
        W, H = cfg.frame_wh
        got = 0
        for i in order:
            if got >= target:
                break
            row = gdf.iloc[int(i)]
            if int(row["_wid"]) in taken or room(row["match"]) <= 0 or not clip_ok(row, "G"):
                continue
            cx, cy = float(row["x"]), float(row["y"])
            shift = None
            if row["kind"] == "gt_shifted":
                d = float(rngs["G"].uniform(*cfg.g_shift_px))
                th = float(rngs["G"].uniform(0, 2 * math.pi))
                cx, cy = (
                    min(max(cx + d * math.cos(th) / W, 0.0), 1.0),
                    min(max(cy + d * math.sin(th) / H, 0.0), 1.0),
                )
                shift = round(d, 2)
            row = row.copy()
            row["frame"] = int(row["frame"])
            take(row, "G", {"stratum": row["kind"], "pi": None, "reason": f"gold: {row['kind']}",
                            "gold": {"kind": row["kind"], "shift_px": shift,
                                     "expected": ["box_off", "not_ball"] if row["kind"] == "gt_shifted" else ["not_ball"]}},
                 gold_card={"cx": cx, "cy": cy, "w": float(row["w"]), "h": float(row["h"])})  # fmt: skip
            got += 1
        return got

    # ---- run ---------------------------------------------------------------------------------------------------
    counts = _largest_remainder_targets(budget, quotas)
    spill = 0
    got_by: dict[str, int] = {}
    for b in ORDER:
        target = counts[b] + (spill if b == "R" else 0)
        if b == "U":
            got = pick_ranked("U", target, "u_rank", True)
        elif b == "M":
            got = pick_ranked("M", target, "m_rank", False)
        elif b == "N":
            got = pick_n(target)
        elif b == "A":
            got = pick_a(target)
        elif b == "G":
            got = pick_g(target)
        else:
            got = pick_r(target)
        got_by[b] = got
        if b != "R":
            spill += target - got
    for per_stratum in inclusion.values():
        for rec in per_stratum.values():
            pis = rec.pop("_pis")
            rec["pi_node_mean"] = round(float(np.mean(pis)), 8) if pis else None

    # ---- order: shuffled, with U / M cards in priority order inside their own slots -------------------------
    cards = _initial_order(cards, rng_order)
    _assert_train_only(cards, split_of, gate)

    by_match_bucket: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in cards:
        by_match_bucket[c["match"]][c["sampling"]["bucket"]] += 1
    report = {
        "budget": budget, "n_cards": len(cards), "seed": seed, "round": round_key, "pool_rows_in": n_in,
        "pool_rows_minable": len(df), "matches_minable": matches,
        "targets": counts, "got": got_by, "spilled_to_R": spill,
        "per_match_cap": {"fraction": per_match_cap, "cards": cap_n, "relaxed_to_equal_share": cap_relaxed},
        "by_match_bucket": {m: dict(v) for m, v in sorted(by_match_bucket.items())},
        "refused": gate.report["refused"], "excluded": dict(gate.report["excluded"]),
        "diversity": {"exclude_s": cfg.exclude_s, "max_per_clip": cfg.max_per_clip, "spacing_rejects": dict(spacing_rejects),
                      "distinct_clips": len({c["clip"] for c in cards}),
                      "max_cards_per_clip": max(Counter(c["clip"] for c in cards).values(), default=0)},
        "inclusion_probabilities": inclusion,
    }  # fmt: skip
    LOGGER.info(
        "mine_round: %s cards of %s; got %s; spill %s; inclusion %s",
        len(cards),
        budget,
        got_by,
        spill,
        inclusion,
    )
    return MineResult(cards, report)


def _largest_remainder_targets(
    budget: int, quotas: Mapping[str, float]
) -> dict[str, int]:
    return allocate(
        budget,
        {b: quotas[b] for b in ORDER if quotas[b] > 0},
        dict.fromkeys(ORDER, budget),
    ) | {b: 0 for b in ORDER if quotas[b] <= 0}


def _cluster_reps(
    d: pd.DataFrame,
    k: int,
    cfg: BucketConfig,
    rng: np.random.Generator,
    ascending: bool,
) -> list[tuple[pd.Series, int]]:
    """Best-ranked node of each of ``k`` k-means clusters over (embedding | position) and footage time; ordered by rank."""
    n = len(d)
    emb = d.get("embedding")
    if (
        emb is not None
        and emb.map(lambda v: v is not None and not isinstance(v, float)).all()
        and n
    ):
        e = np.stack(emb.map(np.asarray).to_numpy()).astype(np.float32)
        base = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    else:
        base = d[["x", "y"]].to_numpy(dtype=np.float32)
    t = d["_t"].to_numpy(dtype=float)
    if np.isnan(
        t
    ).any():  # unknown footage time: clips are ordered by name, 10 minutes apart
        order = {c: i for i, c in enumerate(sorted(d["_cid"].unique()))}
        t = np.where(
            np.isnan(t),
            d["_cid"].map(order).to_numpy() * 600.0
            + d["frame"].to_numpy() / d["fps"].to_numpy(),
            t,
        )
    span = max(float(t.max() - t.min()), 1.0)
    feats = np.hstack(
        [
            base,
            (cfg.cluster_time_weight * (t - t.min()) / span)[:, None].astype(
                np.float32
            ),
        ]
    )
    labels = kmeans(feats, k, rng)
    rank = d["_rank"].to_numpy()
    best: dict[int, int] = {}
    for i, c in enumerate(
        labels
    ):  # d is sorted best-first, so the first row of a cluster is its representative
        best.setdefault(int(c), i)
    reps = sorted(
        best.items(), key=lambda kv: (rank[kv[1]] if ascending else -rank[kv[1]], kv[1])
    )
    return [(d.iloc[i], c) for c, i in reps]


def _initial_order(cards: list[dict], rng: np.random.Generator) -> list[dict]:
    """A random order (so gold / audit / random cards are not clustered at the end), with the U cards re-dealt by
    ascending ``rank`` into the slots U cards hold and the M cards by descending ``rank`` into the M slots."""
    perm = rng.permutation(len(cards))
    out = [cards[i] for i in perm]
    for b, rev in (("U", False), ("M", True)):
        slots = [i for i, c in enumerate(out) if c["sampling"]["bucket"] == b]
        ranked = sorted(
            (out[i] for i in slots),
            key=lambda c: c["sampling"].get("rank", 0.0),
            reverse=rev,
        )
        for i, c in zip(slots, ranked, strict=True):
            out[i] = c
    return out


def _assert_train_only(cards: list[dict], split_of: SplitOf, gate: _Gate) -> None:
    for c in cards:
        s = split_of(c["match"], c["clip"])
        if s != TRAIN:
            raise LeakageError(f"card {c['clip']}@{c['frame_index']} is {s}, not train")
