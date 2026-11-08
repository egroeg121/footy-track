"""Canonical clip identity (T-03).

One physical clip carries several names across the data plane. Verified on
``/mnt/storage/footy_data`` (2026-11-08) via the ``ballcheck_clips/`` and
``eval_data/clips/`` symlinks, which point every alias at the same video file
(so frame indices agree across aliases):

=================================================  ===============================
Name seen on disk                                   Canonical clip id
=================================================  ===============================
``arsenal_mancity_20250925_seg010`` (clips/,        ``arsenal_mancity_seg010``
machine_labels/, ball_candidates/, GT written by
ball_check)
``arsenal_mancity_seg010`` (GT, eval)               ``arsenal_mancity_seg010``
``mancity_seg010`` (GT, eval)                       ``arsenal_mancity_seg010``
``arsenal_mancity_20250925_part050``                ``arsenal_mancity_part050``
(``split_video_10s/``), ``mancity_part050``
``Arsenal - Aston Villa_seg020``, ``astonvilla_``   ``arsenal_astonvilla_seg020``
``seg020``
``Bournemouth vs Arsenal 1_seg015``,                ``arsenal_bournemouth_1st_seg015``
``bournemouth_1st_seg015``,
``bournemouth_broadcast_seg015``,
``bournemouth_seg015``
``arsenal_mancity_example_video``,                  ``arsenal_example``
``arsenal_example``
=================================================  ===============================

Canonical form: ``<match>_<kind><NNN>`` where ``match`` is ``arsenal_<opponent>``
(plus ``_1st`` for the Bournemouth first half), ``kind`` is ``seg`` (broadcast
segments) or ``part`` (the 10 s ``split_video_10s`` cut, a *different* video
from ``seg`` with the same number, so they are never merged), and ``NNN`` is
the zero-padded segment number. The date infix is dropped. Names matching no
rule are returned unchanged (minus directory and extension), so unknown clips
keep their own identity instead of being guessed into another.

``bournemouth_segNNN`` and ``bournemouth_broadcast_segNNN`` say nothing about the
half; they map to the first half because every such symlink on disk points at
``arsenal_bournmouth_1st_half/.../Bournemouth vs Arsenal 1_segNNN.mp4``.

Where it is used: the training export (eval exclusion and by-clip split),
``ball_check`` ``save_box`` (writes into the existing GT sidecar of the clip)
and ``ball_eval`` (``EvalClip.clip_id``, optional alias video lookup).

Feature-store ``game_id`` is deliberately NOT canonicalised at ingest: it stays
the sidecar stem, which is the unit ``ingest_gt`` (re)imports and the key of the
``gt_import:<game_id>`` run group. Readers canonicalise instead. Migration, if
ever wanted: run ``python -m footy_track.clip_id --db <store>`` (read-only) to
list the ``game_id`` values that would collapse, then re-ingest those sidecars
under their canonical stem into a fresh store; never rewrite ``game_id`` in
place, since the detection PK ``(game_id, source, run_id, frame_index,
detection_id)`` of two aliases can collide.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import PurePath

_VIDEO_OR_DATA_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".jsonl", ".json"}

#: (pattern, canonical template). Applied in order with ``re.fullmatch``;
#: ``{kind}`` and ``{n}`` come from the named groups.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?:arsenal_)?mancity_(?:\d{8}_)?(?P<kind>seg|part)(?P<n>\d+)"),
        "arsenal_mancity_{kind}{n}",
    ),
    (
        re.compile(
            r"(?:arsenal_|Arsenal - )?(?:astonvilla|Aston Villa)_(?P<kind>seg)(?P<n>\d+)"
        ),
        "arsenal_astonvilla_{kind}{n}",
    ),
    (
        re.compile(
            r"(?:arsenal_bournemouth_1st|bournemouth(?:_1st|_broadcast)?"
            r"|Bournemouth vs Arsenal 1)_(?P<kind>seg)(?P<n>\d+)"
        ),
        "arsenal_bournemouth_1st_{kind}{n}",
    ),
    (re.compile(r"arsenal_mancity_example_video|arsenal_example"), "arsenal_example"),
)


def _stem(name: str) -> str:
    p = PurePath(name)
    return p.stem if p.suffix.lower() in _VIDEO_OR_DATA_SUFFIXES else p.name


def canonical_clip_id(name: str) -> str:
    """Canonical id for a clip name, path, or stem (extension stripped)."""
    stem = _stem(name)
    for pattern, template in _RULES:
        m = pattern.fullmatch(stem)
        if m:
            groups = m.groupdict()
            n = f"{int(groups['n']):03d}" if groups.get("n") else ""
            return template.format(kind=groups.get("kind") or "", n=n)
    return stem


def same_clip(a: str, b: str) -> bool:
    return canonical_clip_id(a) == canonical_clip_id(b)


def aliases_of(clip: str, names: Iterable[str]) -> list[str]:
    """Stems from *names* that denote the same physical clip as *clip*, sorted."""
    cid = canonical_clip_id(clip)
    return sorted({_stem(n) for n in names if canonical_clip_id(n) == cid})


def group_by_clip(names: Iterable[str]) -> dict[str, list[str]]:
    """``{canonical_id: sorted stems}`` for every name."""
    groups: dict[str, set[str]] = defaultdict(set)
    for n in names:
        groups[canonical_clip_id(n)].add(_stem(n))
    return {cid: sorted(stems) for cid, stems in sorted(groups.items())}


def collisions(names: Iterable[str]) -> dict[str, list[str]]:
    """Canonical ids that more than one distinct stem in *names* collapses to."""
    return {cid: s for cid, s in group_by_clip(names).items() if len(s) > 1}


def store_game_id_collisions(db_path: str) -> dict[str, list[str]]:
    """Report which existing feature-store ``game_id`` values would collapse
    under :func:`canonical_clip_id`. Opens the DuckDB file read-only."""
    import duckdb  # noqa: PLC0415

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT game_id FROM game UNION SELECT DISTINCT game_id FROM detection"
        ).fetchall()
    finally:
        con.close()
    return collisions(r[0] for r in rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Read-only report of clip names that share a canonical id."
    )
    parser.add_argument("--db", help="feature store DuckDB file (opened read-only)")
    parser.add_argument("dirs", nargs="*", help="directories whose file stems to group")
    args = parser.parse_args(argv)

    from pathlib import Path  # noqa: PLC0415

    if args.db:
        found = store_game_id_collisions(args.db)
        print(f"feature store {args.db}: {len(found)} canonical id(s) with aliases")
        for cid, stems in found.items():
            print(f"  {cid}: {stems}")
    for d in args.dirs:
        names = [p.name for p in Path(d).iterdir() if not p.is_dir()]
        groups = group_by_clip(names)
        print(f"{d}: {len(names)} files -> {len(groups)} canonical ids")
        for cid, stems in collisions(names).items():
            print(f"  {cid}: {stems}")


if __name__ == "__main__":
    main()
