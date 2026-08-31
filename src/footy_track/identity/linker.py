"""Gap bridging: rejoin tracklets that the tracker split, and ask when unsure.

The tracker fragments because ~4.5% of consecutive-frame associations fail
(measured over 1,604,326 pairs across all 162 clips). Each failure ends a
tracklet and starts a new one for the same player. Bridging those gaps recovers
identity that was never really lost.

**The bridging window is set by measurement, not preference.** Embedding AUC
against guaranteed negatives, by temporal gap:

    gap (frames)   4      12     25     50     100    200
    seconds        0.2    0.5    1.0    2.0    4.0    8.0
    AUC            0.912  0.867  0.761  0.718  0.724  0.634

Appearance decides confidently for about half a second and then decays to a weak
residual. So the default window is 12 frames. Beyond that, appearance alone
cannot carry a merge and the decision belongs to motion, to a human, or to
nothing at all.

Three cheap constraints run BEFORE any embedding, because they are free and
strictly more reliable than a similarity score:

* **Temporal** — one tracklet must end before the other begins. Overlapping
  tracklets co-exist in some frame, so they are provably different players.
* **Spatial** — a player cannot teleport. The gap is bridged only if the
  positions are consistent with plausible movement over its duration.
* **Team** — different kit means different player (~99% separable, though
  measured on one clip; treated here as a strong prior, not a hard rule).

What survives is scored, and only the AMBIGUOUS middle is shown to a human:
confident merges and confident rejections are not worth anyone's attention.
"""

from __future__ import annotations

from dataclasses import dataclass

# Frames. Set from the decay curve above, not from taste.
DEFAULT_MAX_GAP = 12
# Normalised units per frame. A player crossing the full frame width in ~4s is
# ~0.01/frame; this allows well above that so camera pan does not veto real links.
MAX_SPEED_PER_FRAME = 0.035
# Below this an embedding says "different"; above it, "same". Between them the
# model is not entitled to an opinion and a human is asked.
UNSURE_LOW = 0.55
UNSURE_HIGH = 0.80


@dataclass(frozen=True)
class Tracklet:
    """Minimal tracklet summary needed to propose a link."""

    track_id: int
    start_frame: int
    end_frame: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    n_detections: int
    team: int | None = None

    @property
    def span(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class Candidate:
    """A proposed bridge between two tracklets."""

    a: int
    b: int
    gap: int
    distance: float
    similarity: float | None = None

    @property
    def decision(self) -> str:
        """``merge`` / ``reject`` / ``ask`` — ask covers the ambiguous middle."""
        if self.similarity is None:
            return "ask"
        if self.similarity >= UNSURE_HIGH:
            return "merge"
        if self.similarity < UNSURE_LOW:
            return "reject"
        return "ask"

    @property
    def uncertainty(self) -> float:
        """Distance from the decision boundary — 0 is maximally uncertain.

        Used to order the human queue. Reviewing the most uncertain candidates
        first extracts the most information per decision, and is also where the
        model's errors live.
        """
        if self.similarity is None:
            return 0.0
        mid = (UNSURE_LOW + UNSURE_HIGH) / 2
        return abs(self.similarity - mid)


def candidate_links(
    tracklets: list[Tracklet],
    *,
    max_gap: int = DEFAULT_MAX_GAP,
    max_speed: float = MAX_SPEED_PER_FRAME,
    respect_team: bool = True,
) -> list[Candidate]:
    """Pairs that could plausibly be the same player, cheapest filters first.

    Returns candidates ordered by gap then distance, without similarity: scoring
    needs video decoding and an embedding model, so it is a separate step and
    this stays pure and testable.
    """
    out: list[Candidate] = []
    ordered = sorted(tracklets, key=lambda t: t.start_frame)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b.start_frame > a.end_frame + max_gap:
                break  # ordered by start: nothing later can be closer
            gap = b.start_frame - a.end_frame
            if gap <= 0:
                # Overlapping in time: they co-exist, so they are provably
                # different players. Not a candidate at any similarity.
                continue
            if respect_team and a.team is not None and b.team is not None:
                if a.team != b.team:
                    continue
            dx = b.start_xy[0] - a.end_xy[0]
            dy = b.start_xy[1] - a.end_xy[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > max_speed * max(gap, 1):
                continue  # would require teleporting
            out.append(Candidate(a=a.track_id, b=b.track_id, gap=gap, distance=dist))
    return sorted(out, key=lambda c: (c.gap, c.distance))


def triage(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    """Split scored candidates into merge / reject / ask.

    The ``ask`` list is ordered most-uncertain-first: that is where the model is
    least useful and a human decision buys the most.
    """
    buckets: dict[str, list[Candidate]] = {"merge": [], "reject": [], "ask": []}
    for c in candidates:
        buckets[c.decision].append(c)
    buckets["ask"].sort(key=lambda c: c.uncertainty)
    buckets["merge"].sort(key=lambda c: -(c.similarity or 0))
    buckets["reject"].sort(key=lambda c: (c.similarity or 0))
    return buckets


def tracklets_from_rows(rows: list[dict], *, min_frames: int = 25) -> list[Tracklet]:
    """Build tracklet summaries from detection rows carrying ``track_id``."""
    by_track: dict[int, list[dict]] = {}
    for r in rows:
        tid = r.get("track_id")
        if tid is None or not isinstance(r.get("bbox"), dict):
            continue
        by_track.setdefault(int(tid), []).append(r)

    out: list[Tracklet] = []
    for tid, rs in by_track.items():
        if len(rs) < min_frames:
            continue
        rs.sort(key=lambda r: int(r["frame_index"]))
        first, last = rs[0], rs[-1]

        def centre(r: dict) -> tuple[float, float]:
            b = r["bbox"]
            return (b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)

        out.append(
            Tracklet(
                track_id=tid,
                start_frame=int(first["frame_index"]),
                end_frame=int(last["frame_index"]),
                start_xy=centre(first),
                end_xy=centre(last),
                n_detections=len(rs),
            )
        )
    return out
