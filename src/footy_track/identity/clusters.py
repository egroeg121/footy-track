"""Build identity clusters from pairwise verdicts, and detect contradictions.

Pairwise labelling is cheap for humans but not self-consistent: SAME is
transitive, so a single wrong SAME permanently fuses two players, and the error
is invisible afterwards — the merged cluster looks like any other. DIFFERENT is
not transitive, but it *constrains*: if A~B and B~C then A and C are the same,
so an explicit "A is different from C" contradicts the merges.

This module therefore does two things that a plain union-find does not:

* it keeps DIFFERENT verdicts as constraints and reports every pair that the
  SAME-merges contradict, rather than discarding them;
* it never silently resolves a contradiction. Which verdict was wrong is a
  question for the human, not for a tie-break rule.

UNKNOWN verdicts are ignored for clustering by design — they are ignore-regions,
not weak evidence (see ``labels.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from footy_track.identity.labels import PairLabel, TrackletRef, Verdict

TrackletKey = tuple[str, int, str, str | None]


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[TrackletKey, TrackletKey] = {}

    def add(self, x: TrackletKey) -> None:
        self._parent.setdefault(x, x)

    def find(self, x: TrackletKey) -> TrackletKey:
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: TrackletKey, b: TrackletKey) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


@dataclass
class ClusterResult:
    """Clusters plus everything that did not fit.

    ``contradictions`` lists DIFFERENT-labelled pairs that ended up in the same
    cluster. A non-empty list means at least one human verdict is wrong and the
    affected pairs must be re-shown. Do NOT treat clusters as trustworthy while
    contradictions remain.
    """

    clusters: dict[TrackletKey, int]
    contradictions: list[tuple[TrackletRef, TrackletRef]]
    n_clusters: int
    n_unknown: int

    def cluster_of(self, t: TrackletRef) -> int | None:
        return self.clusters.get(t.key())

    def members(self, cluster_id: int) -> list[TrackletKey]:
        return sorted(k for k, c in self.clusters.items() if c == cluster_id)

    @property
    def is_consistent(self) -> bool:
        return not self.contradictions


def build_clusters(labels: list[PairLabel]) -> ClusterResult:
    """Merge SAME verdicts, then check the DIFFERENT verdicts still hold.

    Args:
        labels: pairwise human verdicts, in any order. Order does not affect the
            resulting partition.

    Returns:
        A :class:`ClusterResult`. Cluster ids are small integers assigned in a
        deterministic order so results are reproducible across runs.
    """
    uf = _UnionFind()
    same: list[PairLabel] = []
    different: list[PairLabel] = []
    n_unknown = 0

    for lbl in labels:
        a, b = lbl.ordered()
        uf.add(a.key())
        uf.add(b.key())
        if lbl.verdict is Verdict.SAME:
            same.append(lbl)
        elif lbl.verdict is Verdict.DIFFERENT:
            different.append(lbl)
        else:
            n_unknown += 1

    for lbl in same:
        a, b = lbl.ordered()
        uf.union(a.key(), b.key())

    # Assign compact, deterministic cluster ids.
    root_to_id: dict[TrackletKey, int] = {}
    clusters: dict[TrackletKey, int] = {}
    for key in sorted(uf._parent):
        root = uf.find(key)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        clusters[key] = root_to_id[root]

    contradictions = [
        lbl.ordered()
        for lbl in different
        if clusters[lbl.ordered()[0].key()] == clusters[lbl.ordered()[1].key()]
    ]

    return ClusterResult(
        clusters=clusters,
        contradictions=contradictions,
        n_clusters=len(root_to_id),
        n_unknown=n_unknown,
    )


def merge_savings(n_tracklets: int, n_labels: int) -> float:
    """Fraction of the exhaustive pairwise space avoided.

    Exhaustive pairwise labelling is O(n^2) — 4,900 tracklets is ~12M pairs,
    which is why propose-and-verify with uncertainty ordering is the only
    workable strategy. Useful for reporting how much a labelling session
    actually cost against the naive baseline.
    """
    total = n_tracklets * (n_tracklets - 1) / 2
    if total <= 0:
        return 0.0
    return max(0.0, 1.0 - n_labels / total)
