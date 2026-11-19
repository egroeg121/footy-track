"""Pure-geometry clustering of person boxes (docs/design/full_frame_labelling.md section 4c, step 1).

Two boxes are *linked* when they overlap (IoU > ``IOU_LINK``) or their centres are close
(centre distance < ``CENTRE_LINK_FRAC`` * the mean of the two box widths). A cluster is a
connected component of that link graph. Isolated boxes are clusters of size 1.

No dependencies beyond the standard library, so the Player Check task can reuse it. Boxes are
``(x, y, w, h)`` with ``(x, y)`` the top-left corner, in any consistent unit (use pixels, or
normalised coordinates scaled to a common aspect ratio, so IoU and distances are meaningful).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

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
