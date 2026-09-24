"""Ball candidates with real confidences, and how to sample them.

The GT sidecars cannot drive confidence-aware review: they never persist
confidence (README §2, LAB-104 synthesizes 1.0 for hand marks and 0.5 for
everything else). The detector's own output does — one JSONL per clip under
``server._CANDIDATES_DIR``, ball-class rows only, pulled from
``s3://…/footy_data/machine_labels`` (run ``rtdetr-l_1920_v11``, already
emitted at ``conf_threshold`` 0.1, so borderline detections are present
without re-running anything).

Row format, as produced by the detector::

    {"frame_index": 0, "bbox": {"x":…,"y":…,"w":…,"h":…},
     "tags": ["in_play_ball", "rtdetr"], "confidence": 0.41,
     "model_id": "rtdetr-l_1920_v11", "run_id": "…", "reviewed": false}

A candidate's identity is ``(clip, frame_index, cand_index)`` where
``cand_index`` is the ordinal of the row among that frame's ball rows in file
order — the same shape of identity review uses for sidecar boxes, but a
separate namespace, which is why verdicts carry a ``source``.

Why the sampling is not uniform
-------------------------------
Uniform sampling spends almost all of its effort where the model is already
certain, which teaches you nothing: the informative region is the decision
boundary, where a small threshold change flips the answer. But sampling
*only* near the boundary leaves the rest of the range unmeasured, and you
need that to state precision at any threshold. So the order is a mixture:
most weight in a window around the threshold, the rest spread across the
whole range (see ``stratified_order``).
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np

#: Ball classes as they appear in the detector's tags.
BALL_TAGS = {"ball", "in_play_ball", "out_of_play_ball"}

#: Default decision threshold the sampling concentrates around.
DEFAULT_TAU = 0.25

#: Width of the borderline window (in confidence units, one sigma).
DEFAULT_SIGMA = 0.10

#: Share of served cards drawn uniformly at random from the whole pool,
#: rather than from the borderline window. This is a share of *cards*, not a
#: weight: one card in four is uniform, by construction (see ``select``).
UNIFORM_SHARE = 0.25

#: How sharply the weighted stream favours the boundary. The floor is
#: deliberately tiny (1 - focus = 0.02): coverage of the rest of the range is
#: the uniform stream's job, not the weight's. With a meaningful floor a huge
#: band floods the weighted stream purely by count — measured on a pool with
#: 5,000 far-from-boundary rows against 400 near it, a 0.3 floor let the
#: far band take 23 of 40 supposedly-borderline cards.
DEFAULT_FOCUS = 0.98

#: Height of the borderline peak relative to the uniform floor. With the
#: defaults the peak is ~13x the tail, which puts roughly two thirds of early
#: cards in the borderline window while still visiting every band. A plain
#: focus/(1-focus) mixture was tried first and is far too flat — it gives only
#: a ~3x peak, so most effort still lands where the model is already certain.
PEAK_GAIN = 5.0

#: Confidence resolution of the sampling index. The weight is a function of
#: confidence alone, so candidates sharing a bin share a weight — and within a
#: bin the race order is then just ascending draw, which does not depend on
#: the weight at all and can be precomputed once.
SAMPLE_BINS = 100


def _candidates_dir() -> Path:
    from footy_track.labeller import server  # noqa: PLC0415 — avoid import cycle

    return server._CANDIDATES_DIR


#: Parsed rows per file, keyed by path -> (mtime_ns, size, rows). The real
#: corpus is ~357k rows across 162 files; re-parsing it on every queue fetch
#: (which now happens every ~15 verdicts, so the sampling can adapt) would
#: make the UI unusable. Invalidated by mtime+size, so a re-pull is picked up.
_FILE_CACHE: dict[Path, tuple[int, int, list[dict]]] = {}


#: Two candidates are the same *event* if they are within this many frames
#: and this far apart. Consecutive detections of one ball produce visually
#: identical cards; measured on 40 clips, collapsing them removes 2.7x
#: redundancy (77,560 candidates -> 28,221 events).
GROUP_FRAME_GAP = 3
GROUP_DIST = 0.03

#: Appearance buckets, from a random projection of the crop descriptor. Used
#: to spread a page across the appearance range, not to recognise anything.
LSH_BITS = 8
_LSH_SEED = 20260924


def _lsh_planes(dim: int) -> np.ndarray:
    rng = np.random.default_rng(_LSH_SEED)
    return rng.standard_normal((LSH_BITS, dim)).astype(np.float32)


def _attach_groups(rows: list[dict]) -> None:
    """Tag each row with an event id: one card per event, not per frame."""
    rows.sort(key=lambda r: (r["frame_index"], r["cand_index"]))
    open_events: list[tuple[int, float, float, int]] = []  # frame, cx, cy, gid
    next_gid = 0
    for r in rows:
        b = r["bbox"]
        cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        f = r["frame_index"]
        open_events = [e for e in open_events if f - e[0] <= GROUP_FRAME_GAP]
        hit = next(
            (
                e
                for e in open_events
                if abs(cx - e[1]) < GROUP_DIST and abs(cy - e[2]) < GROUP_DIST
            ),
            None,
        )
        if hit is None:
            gid = next_gid
            next_gid += 1
        else:
            gid = hit[3]
        r["group"] = f"{r['clip']}:{gid}"
        open_events.append((f, cx, cy, gid))


def _attach_appearance(clip_stem: str, rows: list[dict]) -> None:
    """Attach an appearance bucket from the precomputed crop descriptors.

    Absent descriptors are not an error: the pass runs offline and may lag
    the candidates, so rows simply stay unbucketed and sampling falls back to
    event-level dedupe alone.
    """
    path = _candidates_dir().parent / "ball_embeddings" / f"{clip_stem}.npz"
    if not path.exists():
        return
    try:
        with np.load(path) as data:
            frames = data["frame_index"]
            cands = data["cand_index"]
            vecs = data["vec"].astype(np.float32)
    except (OSError, ValueError, KeyError):
        return
    if len(vecs) == 0:
        return
    codes = (vecs @ _lsh_planes(vecs.shape[1]).T) > 0
    packed = codes.dot(1 << np.arange(LSH_BITS))
    lookup = {
        (int(f), int(c)): int(code)
        for f, c, code in zip(frames, cands, packed, strict=False)
    }
    for r in rows:
        code = lookup.get((r["frame_index"], r["cand_index"]))
        if code is not None:
            r["lsh"] = code


def _read_file(path: Path) -> list[dict]:
    try:
        st = path.stat()
    except OSError:
        return []
    cached = _FILE_CACHE.get(path)
    if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    rows = _parse_file(path)
    _attach_groups(rows)
    _attach_appearance(path.stem, rows)
    _FILE_CACHE[path] = (st.st_mtime_ns, st.st_size, rows)
    return rows


def read_candidates(clip: str | None = None) -> list[dict]:
    """Read ball candidates from every clip file (or just one)."""
    cdir = _candidates_dir()
    if not cdir.exists():
        return []
    paths = [cdir / f"{clip}.jsonl"] if clip else sorted(cdir.glob("*.jsonl"))
    out: list[dict] = []
    for path in paths:
        out.extend(_read_file(path))
    return out


def _parse_file(path: Path) -> list[dict]:
    """Parse one clip's candidate rows (ball classes only)."""
    out: list[dict] = []
    if not path.exists():
        return out
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    per_frame: dict[int, int] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        tags = rec.get("tags") or []
        label = next((t for t in tags if t in BALL_TAGS), None)
        bbox = rec.get("bbox")
        if label is None or not bbox:
            continue
        try:
            frame_index = int(rec["frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        idx = per_frame.get(frame_index, 0)
        per_frame[frame_index] = idx + 1
        out.append(
            {
                "clip": path.stem,
                "frame_index": frame_index,
                "cand_index": idx,
                "bbox": {
                    "x": float(bbox["x"]),
                    "y": float(bbox["y"]),
                    "w": float(bbox["w"]),
                    "h": float(bbox["h"]),
                },
                "label": label,
                "confidence": float(rec.get("confidence") or 0.0),
                "provenance": rec.get("model_id") or "rtdetr",
                "source": "cand",
                # Hashing 357k keys per request dominated the queue latency;
                # the draw is a property of the candidate, so cache it here.
                "u": _uniform_from_key(f"{path.stem}:{frame_index}:{idx}"),
            }
        )
    return out


def read_candidate_box(
    clip: str, frame_index: int, cand_index: int
) -> tuple[float, float, float, float] | None:
    """(x, y, w, h) for one candidate, or None."""
    for rec in read_candidates(clip):
        if rec["frame_index"] == frame_index and rec["cand_index"] == cand_index:
            b = rec["bbox"]
            return b["x"], b["y"], b["w"], b["h"]
    return None


def sampling_weight(
    confidence: float,
    tau: float = DEFAULT_TAU,
    sigma: float = DEFAULT_SIGMA,
    focus: float = DEFAULT_FOCUS,
) -> float:
    """Mixture weight: a Gaussian bump on the threshold plus a uniform floor.

    ``focus`` splits the weight between the two and ``PEAK_GAIN`` sets how
    sharply the boundary is favoured. The uniform part is what keeps the high-
    and low-confidence bands populated, so precision can be reported per band
    rather than only at the boundary.
    """
    sigma = max(sigma, 1e-6)
    bump = math.exp(-0.5 * ((confidence - tau) / sigma) ** 2)
    return focus * PEAK_GAIN * bump + (1.0 - focus)


def _uniform_from_key(key: str) -> float:
    """Deterministic uniform in (0, 1) from a stable hash of the key."""
    digest = hashlib.sha1(key.encode()).digest()
    n = int.from_bytes(digest[:8], "big")
    return (n + 0.5) / (1 << 64)


def bin_centre(confidence: float) -> float:
    """Quantise a confidence to its sampling bin centre.

    The weight is deliberately a step function of confidence at 1/100
    resolution — far finer than sigma (0.10), so the shape is unchanged — and
    quantising makes the binned fast path exactly equivalent to racing every
    candidate rather than merely close to it.
    """
    c = min(max(confidence, 0.0), 0.999999)
    return (int(c * SAMPLE_BINS) + 0.5) / SAMPLE_BINS


def _draw(rec: dict) -> float:
    u = rec.get("u")
    if u is None:
        u = _uniform_from_key(
            f"{rec['clip']}:{rec['frame_index']}:{rec.get('cand_index', 0)}"
        )
    return -math.log(u)


def _race(records: list[dict], weight_of, top: int | None = None) -> list[dict]:
    """Weighted sampling without replacement via the exponential race.

    ``top`` returns only the head of the order, which is all a page of cards
    needs: partial selection over ~357k candidates instead of a full sort.
    """
    keyed = ((_draw(r) / max(weight_of(r), 1e-9), i, r) for i, r in enumerate(records))
    chosen = heapq.nsmallest(top, keyed) if top else sorted(keyed)
    return [r for _, _, r in chosen]


def stratified_order(
    records: list[dict],
    tau: float = DEFAULT_TAU,
    sigma: float = DEFAULT_SIGMA,
    focus: float = DEFAULT_FOCUS,
) -> list[dict]:
    """Order candidates by weighted sampling without replacement.

    Uses the exponential-race trick: with key ``-ln(u)/w`` and u uniform, the
    ascending order of keys is exactly a weighted sample without replacement.
    ``u`` comes from a hash of the candidate's identity rather than an RNG, so
    the order is stable across restarts — the queue must not reshuffle under a
    labeller mid-session — while still being random with respect to weight.
    """
    return _race(
        records, lambda r: sampling_weight(r.get("confidence", 0.0), tau, sigma, focus)
    )


def confidence_histogram(records: list[dict], bins: int = 10) -> list[dict]:
    """Counts per 0.1-wide confidence band — the coverage view for the UI."""
    counts = [0] * bins
    for rec in records:
        c = min(max(rec.get("confidence", 0.0), 0.0), 0.999999)
        counts[int(c * bins)] += 1
    return [
        {"lo": round(i / bins, 2), "hi": round((i + 1) / bins, 2), "n": n}
        for i, n in enumerate(counts)
    ]


# ---------------------------------------------------------------------------
# Adaptive sampling: let the verdicts steer the queue
# ---------------------------------------------------------------------------

#: Confidence band width for the running per-band precision estimate.
BAND = 0.1


def band_of(confidence: float) -> int:
    return min(int(max(confidence, 0.0) / BAND), int(1 / BAND) - 1)


def band_stats(verdicts: list[dict], present_verdicts: tuple[str, ...]) -> list[dict]:
    """Per-band decided counts and precision, from verdicts judged so far."""
    n_bands = int(1 / BAND)
    hits = [0] * n_bands
    decided = [0] * n_bands
    for rec in verdicts:
        conf = rec.get("confidence")
        verdict = rec.get("verdict")
        if conf is None or verdict is None:
            continue
        if verdict not in present_verdicts and verdict != "not_ball":
            continue  # unsure: no information about correctness
        b = band_of(float(conf))
        decided[b] += 1
        if verdict in present_verdicts:
            hits[b] += 1
    return [
        {
            "lo": round(i * BAND, 2),
            "hi": round((i + 1) * BAND, 2),
            "decided": decided[i],
            "precision": (hits[i] / decided[i]) if decided[i] else None,
        }
        for i in range(n_bands)
    ]


def estimate_tau(stats: list[dict], fallback: float = DEFAULT_TAU) -> float:
    """Where precision crosses 0.5 — the real decision boundary, measured.

    Interpolates between the band centres either side of the crossing. Falls
    back to the configured threshold until enough bands have been decided,
    because a boundary estimated from two labels is worse than a prior.
    """
    known = [
        (s["lo"] + BAND / 2, s["precision"], s["decided"])
        for s in stats
        if s["decided"] >= 3
    ]
    if len(known) < 2:
        return fallback
    known.sort()
    for (c0, p0, _), (c1, p1, _) in zip(known, known[1:], strict=False):
        if (p0 - 0.5) * (p1 - 0.5) <= 0 and p1 != p0:
            t = (0.5 - p0) / (p1 - p0)
            return max(0.0, min(1.0, c0 + t * (c1 - c0)))
    # No crossing seen yet: aim just below the lowest confidence that still
    # looks reliable, which is where the boundary must lie.
    if all(p > 0.5 for _, p, _ in known):
        return max(0.0, known[0][0] - BAND)
    return min(1.0, known[-1][0] + BAND)


def adaptive_weight(
    confidence: float,
    tau: float,
    stats: list[dict],
    sigma: float = DEFAULT_SIGMA,
    focus: float = DEFAULT_FOCUS,
) -> float:
    """Borderline weight, boosted where the running estimate is still thin.

    Two pressures, deliberately: concentrate near the measured boundary
    (that is where labels change the model), and keep sampling bands whose
    precision is still poorly known (that is where labels change the
    *measurement*). Without the second term the queue would stop visiting the
    tails entirely and precision there would never be established.
    """
    base = sampling_weight(confidence, tau, sigma, focus)
    decided = stats[band_of(confidence)]["decided"] if stats else 0
    uncertainty = 1.0 / math.sqrt(1.0 + decided)
    return base * (0.5 + 0.5 * uncertainty)


def adaptive_order(
    records: list[dict],
    verdicts: list[dict],
    present_verdicts: tuple[str, ...],
    sigma: float = DEFAULT_SIGMA,
    focus: float = DEFAULT_FOCUS,
    tau: float | None = None,
    top: int | None = None,
) -> tuple[list[dict], float, list[dict]]:
    """Order candidates using the boundary measured so far. Returns (queue, tau, stats)."""
    stats = band_stats(verdicts, present_verdicts)
    tau_hat = estimate_tau(stats) if tau is None else tau
    queue = _race(
        records,
        lambda r: adaptive_weight(
            r.get("confidence", 0.0), tau_hat, stats, sigma, focus
        ),
        top=top,
    )
    return queue, tau_hat, stats


# ---------------------------------------------------------------------------
# Binned pool: the same race, but O(bins x page) per request
# ---------------------------------------------------------------------------


class BinnedPool:
    """Candidates bucketed by confidence, each bucket pre-sorted by its draw.

    Racing all ~357k candidates per request cost ~1.5 s, which stalls a refill
    on a phone. Because weight is constant within a bin, selecting the head of
    the global order only ever needs the heads of each bin: the cost drops to
    ``bins x page`` and the result is identical to racing the whole pool.
    """

    def __init__(self, records: list[dict]) -> None:
        bins: list[list[dict]] = [[] for _ in range(SAMPLE_BINS)]
        for rec in records:
            c = min(max(rec.get("confidence", 0.0), 0.0), 0.999999)
            bins[int(c * SAMPLE_BINS)].append(rec)
        for bucket in bins:
            bucket.sort(key=_draw)
        self.bins = bins
        self.total = len(records)
        # What is actually left to look at: one card per event, not per frame.
        self.events = len({r.get("group", id(r)) for r in records})

    def select(
        self,
        weight_of,
        top: int,
        skip=None,
        uniform_share: float = UNIFORM_SHARE,
    ) -> list[dict]:
        """A page of cards: mostly borderline, with a fixed uniform fraction.

        The two streams are drawn separately and interleaved so the uniform
        fraction is exactly ``uniform_share`` of the page. Expressing it as a
        weight instead makes the realised share depend on the shape of the
        pool — which here is wildly lopsided (201k of 357k candidates sit in
        the 0.1-0.2 band), so a "25% uniform" weight would not deliver 25%
        uniform cards.
        """
        if uniform_share <= 0:
            return self._stream(weight_of, top, skip)
        n_uniform = int(round(top * uniform_share))
        n_focus = top - n_uniform
        focused = self._stream(weight_of, n_focus, skip)
        seen = {id(r) for r in focused}
        uniform = [
            r
            for r in self._stream(lambda _c: 1.0, n_uniform + len(focused), skip)
            if id(r) not in seen
        ][:n_uniform]
        out: list[dict] = []
        fi = ui = 0
        # 3 borderline : 1 uniform, so the mix holds for any prefix of the page.
        while fi < len(focused) or ui < len(uniform):
            for _ in range(3):
                if fi < len(focused):
                    out.append(focused[fi])
                    fi += 1
            if ui < len(uniform):
                out.append(uniform[ui])
                ui += 1
        return out

    def _stream(
        self, weight_of, top: int, skip=None, diversify: bool = True
    ) -> list[dict]:
        """Top ``top`` of one weighted order, skipping anything ``skip`` rejects.

        Two diversity rules ride along, because the pool is enormously
        repetitive: at most one card per event (consecutive detections of one
        ball are the same picture), and a cap per appearance bucket so a page
        cannot fill up with twenty crowd crops. The cap is a *soft* one — it
        is dropped once nothing else qualifies, so a page is never short.
        """
        heap: list[tuple[float, int, int]] = []
        weights = []
        for b, bucket in enumerate(self.bins):
            w = max(weight_of((b + 0.5) / SAMPLE_BINS), 1e-9)
            weights.append(w)
            if bucket:
                heapq.heappush(heap, (_draw(bucket[0]) / w, b, 0))
        out: list[dict] = []
        seen_events: set[str] = set()
        bucket_counts: dict[int, int] = {}
        bucket_cap = max(2, int(top / 4)) if diversify else top
        deferred: list[dict] = []
        while heap and len(out) < top:
            _key, b, i = heapq.heappop(heap)
            rec = self.bins[b][i]
            if i + 1 < len(self.bins[b]):
                heapq.heappush(
                    heap, (_draw(self.bins[b][i + 1]) / weights[b], b, i + 1)
                )
            if skip is not None and skip(rec):
                continue
            if diversify:
                event = rec.get("group")
                if event is not None:
                    if event in seen_events:
                        continue
                    seen_events.add(event)
                bucket = rec.get("lsh")
                if bucket is not None and bucket_counts.get(bucket, 0) >= bucket_cap:
                    deferred.append(rec)
                    continue
                if bucket is not None:
                    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            out.append(rec)
        # Never return a short page just to honour a soft cap.
        for rec in deferred:
            if len(out) >= top:
                break
            out.append(rec)
        return out


_POOL_CACHE: dict[tuple, BinnedPool] = {}


def binned_pool(clip: str | None = None) -> BinnedPool:
    """Cached BinnedPool for the candidate corpus (rebuilt when files change)."""
    cdir = _candidates_dir()
    paths = (
        [cdir / f"{clip}.jsonl"]
        if clip
        else sorted(cdir.glob("*.jsonl"))
        if cdir.exists()
        else []
    )
    sig = (
        clip,
        tuple(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in paths
            if p.exists()
        ),
    )
    pool = _POOL_CACHE.get(sig)
    if pool is None:
        pool = BinnedPool(read_candidates(clip))
        _POOL_CACHE.clear()
        _POOL_CACHE[sig] = pool
    return pool
