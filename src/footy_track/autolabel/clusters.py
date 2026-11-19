"""Pure-geometry clustering of person boxes (docs/design/full_frame_labelling.md section 4c, step 1).

Two boxes are *linked* when they overlap (IoU > ``IOU_LINK``) or their centres are close
(centre distance < ``CENTRE_LINK_FRAC`` * the mean of the two box widths). A cluster is a
connected component of that link graph. Isolated boxes are clusters of size 1.

No dependencies beyond the standard library, so the Player Check task can reuse it. Boxes are
``(x, y, w, h)`` with ``(x, y)`` the top-left corner, in any consistent unit (use pixels, or
normalised coordinates scaled to a common aspect ratio, so IoU and distances are meaningful).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

Box = tuple[float, float, float, float]

#: Two boxes are linked when their IoU is strictly greater than this.
IOU_LINK = 0.2
#: ... or when the distance between centres is strictly less than this fraction of the mean box width.
CENTRE_LINK_FRAC = 0.6
#: A cluster of at least this many boxes counts as "crowded" in the frame score.
CROWD_MIN_SIZE = 3


@dataclass(frozen=True)
class Cluster:
    """One connected component: indices into the input box list (ascending), plus its overlap."""

    indices: tuple[int, ...]
    mean_iou: float  # mean pairwise IoU inside the cluster (0.0 for a singleton)

    @property
    def size(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class CrowdScore:
    """Per-frame crowdedness summary."""

    score: float  # ranking key, see :func:`crowd_score`
    largest_cluster: int
    n_in_crowds: int  # boxes in clusters of size >= CROWD_MIN_SIZE
    mean_iou: float  # mean pairwise IoU over all pairs inside clusters of size >= 2
    n_boxes: int


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def linked(a: Box, b: Box) -> bool:
    """True when two boxes belong to the same cluster (IoU or centre-distance rule)."""
    if iou(a, b) > IOU_LINK:
        return True
    dx = (a[0] + a[2] / 2) - (b[0] + b[2] / 2)
    dy = (a[1] + a[3] / 2) - (b[1] + b[3] / 2)
    mean_w = (a[2] + b[2]) / 2
    return (dx * dx + dy * dy) ** 0.5 < CENTRE_LINK_FRAC * mean_w


def cluster_boxes(boxes: Sequence[Box]) -> list[Cluster]:
    """Connected components of the link graph, largest first (ties: lowest first index)."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in combinations(range(n), 2):
        if linked(boxes[i], boxes[j]):
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        ious = [iou(boxes[i], boxes[j]) for i, j in combinations(idx, 2)]
        out.append(Cluster(tuple(idx), sum(ious) / len(ious) if ious else 0.0))
    out.sort(key=lambda c: (-c.size, c.indices[0]))
    return out


def crowd_score(boxes: Sequence[Box]) -> CrowdScore:
    """Crowdedness of one frame: ``largest_cluster + 0.25 * n_in_crowds + mean_iou``.

    The largest cluster dominates (integer part); boxes in size>=3 clusters break ties between
    frames with equally big worst clusters; mean in-cluster IoU (0..1) breaks the rest.
    """
    clusters = cluster_boxes(boxes)
    largest = clusters[0].size if clusters else 0
    in_crowds = sum(c.size for c in clusters if c.size >= CROWD_MIN_SIZE)
    multi = [c for c in clusters if c.size >= 2]
    total = sum(c.mean_iou * (c.size * (c.size - 1) / 2) for c in multi)
    pairs = sum(c.size * (c.size - 1) / 2 for c in multi)
    mean_iou = total / pairs if pairs else 0.0
    return CrowdScore(
        largest + 0.25 * in_crowds + mean_iou, largest, in_crowds, mean_iou, len(boxes)
    )


# --------------------------------------------------------------------------------------------- #
# Player Check (design section 4c steps 1-2): clusters over time, count timeline, flicker, triage.
# --------------------------------------------------------------------------------------------- #

#: A frame cluster joins an existing track when the overlap coefficient of their union boxes
#: (intersection / smaller area, robust to a cluster growing by one person) is at least this.
TRACK_LINK_OVERLAP = 0.5
#: ... and the track was last seen at most this many frames earlier.
TRACK_MAX_GAP = 5
#: A run of equal counts at most this long, between two runs of the same count, is a flicker.
FLICKER_MAX_RUN = 2

#: Triage policies. Per design 6b nothing is auto-accepted for players on detector confidence, so
#: the default audits everything; ``permissive`` is opt-in per match once blind audits justify it.
POLICY_AUDIT_ALL = "audit_all"
POLICY_PERMISSIVE = "permissive"
Policy = Literal["audit_all", "permissive"]


@dataclass(frozen=True)
class ClusterObs:
    """One cluster seen on one frame: box indices (into that frame's box list) and union box."""

    frame: int
    indices: tuple[int, ...]
    bbox: Box
    mean_iou: float

    @property
    def count(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class ClusterTrack:
    """A cluster followed through a window (``c0``, ``c1``, ... in order of first appearance)."""

    track_id: str
    obs: tuple[ClusterObs, ...]  # ascending frame

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(o.frame for o in self.obs)


@dataclass(frozen=True)
class CountRun:
    """Consecutive observations of a cluster with the same person count."""

    start: int
    end: int
    count: int


def union_box(boxes: Sequence[Box]) -> Box:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _overlap_coeff(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    smaller = min(a[2] * a[3], b[2] * b[3])
    return ix * iy / smaller if smaller > 0 else 0.0


def track_clusters(
    frames: Mapping[int, Sequence[Box]],
    *,
    link_overlap: float = TRACK_LINK_OVERLAP,
    max_gap: int = TRACK_MAX_GAP,
) -> list[ClusterTrack]:
    """Cluster every frame, then link clusters across frames by union-box overlap.

    Matching is greedy one-to-one, best overlap first, against tracks last seen within
    ``max_gap`` frames. Unmatched clusters start new tracks. Deterministic.
    """
    tracks: list[list[ClusterObs]] = []
    for fi in sorted(frames):
        boxes = frames[fi]
        obs = [
            ClusterObs(
                fi, c.indices, union_box([boxes[i] for i in c.indices]), c.mean_iou
            )
            for c in cluster_boxes(boxes)
        ]
        pairs = []
        for oi, o in enumerate(obs):
            for ti, t in enumerate(tracks):
                last = t[-1]
                if 0 < fi - last.frame <= max_gap:
                    ov = _overlap_coeff(o.bbox, last.bbox)
                    if ov >= link_overlap:
                        pairs.append((-ov, ti, oi))
        pairs.sort()
        used_t: set[int] = set()
        used_o: set[int] = set()
        for _, ti, oi in pairs:
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            tracks[ti].append(obs[oi])
        for oi, o in enumerate(obs):
            if oi not in used_o:
                tracks.append([o])
    return [ClusterTrack(f"c{i}", tuple(t)) for i, t in enumerate(tracks)]


def count_timeline(track: ClusterTrack) -> list[tuple[int, int]]:
    """``[(frame, n_people), ...]`` for one cluster track."""
    return [(o.frame, o.count) for o in track.obs]


def count_runs(timeline: Sequence[tuple[int, int]]) -> list[CountRun]:
    runs: list[CountRun] = []
    for fi, n in timeline:
        if runs and runs[-1].count == n:
            runs[-1] = CountRun(runs[-1].start, fi, n)
        else:
            runs.append(CountRun(fi, fi, n))
    return runs


def change_points(timeline: Sequence[tuple[int, int]]) -> list[int]:
    """Frames where the count differs from the previous observation."""
    return [b[0] for a, b in zip(timeline, timeline[1:], strict=False) if a[1] != b[1]]


def flicker_frames(
    timeline: Sequence[tuple[int, int]], max_run: int = FLICKER_MAX_RUN
) -> list[int]:
    """Frames of a short count run squeezed between two runs of the *same* count (3,3,2,3).

    A count that changes and stays changed (3,3,4,4) is someone arriving, not a flicker.
    Observations are counted (not frame indices), so sparse windows behave the same.
    """
    frames_by_run: list[list[int]] = []
    counts: list[int] = []
    for fi, n in timeline:
        if counts and counts[-1] == n:
            frames_by_run[-1].append(fi)
        else:
            counts.append(n)
            frames_by_run.append([fi])
    out: list[int] = []
    for i in range(1, len(counts) - 1):
        if counts[i - 1] == counts[i + 1] and len(frames_by_run[i]) <= max_run:
            out.extend(frames_by_run[i])
    return out


def track_severity(track: ClusterTrack) -> float:
    """Queue ranking key (higher first): flicker dominates, then size, then overlap."""
    tl = count_timeline(track)
    biggest = max((o.count for o in track.obs), default=0)
    iou_ = max((o.mean_iou for o in track.obs), default=0.0)
    return 10.0 * len(flicker_frames(tl)) + biggest + iou_


@dataclass(frozen=True)
class TrackTriage:
    track_id: str
    decision: Literal["auto_ok", "review"]
    reason: str


def triage_track(
    track: ClusterTrack,
    *,
    policy: str = POLICY_AUDIT_ALL,
    confidences: Mapping[tuple[int, int], float] | None = None,
    min_len: int = 3,
    min_conf: float = 0.5,
) -> TrackTriage:
    """``auto_ok`` only for an isolated, stable, confident track AND only under the permissive policy.

    ``confidences`` maps ``(frame, box_index)`` to detector confidence (missing = 0).
    """
    if policy != POLICY_PERMISSIVE:
        return TrackTriage(track.track_id, "review", "audit_all")
    if any(o.count > 1 for o in track.obs):
        return TrackTriage(track.track_id, "review", "cluster")
    if len(track.obs) < min_len:
        return TrackTriage(track.track_id, "review", "unstable_short_track")
    conf = confidences or {}
    if any(conf.get((o.frame, o.indices[0]), 0.0) < min_conf for o in track.obs):
        return TrackTriage(track.track_id, "review", "low_confidence")
    return TrackTriage(track.track_id, "auto_ok", "isolated_stable")
