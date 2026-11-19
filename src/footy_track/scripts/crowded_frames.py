"""Rank the most crowded / overlapping-player frames so they can be hand-labelled first.

    python -m footy_track.scripts.crowded_frames --db /mnt/storage/footy_data/feature_store/feature_store.duckdb \\
        [--clips a,b | --match M] [--split train|eval_heldout_match|all] [--min-gap 25] [--top 200] \\
        [--out crowded.json | crowded.csv] [--dataset calib [--labeller-url http://localhost:8011]]

Reads canonical person detections (player, player_sub, referee, coach, person) from the
``detections_enriched`` view (read-only), scores each frame with :mod:`footy_track.autolabel.clusters`,
drops frames that are already ``frame_complete`` or already have a hand-labelled person box, keeps
at most one pick per ``--min-gap`` frames per clip, and prints a ranked plain-text table.

Deep links: the labeller opens ``/labeller?dataset=<d>&clip=<stem>&frame=<n>`` (index.html
``openFromLink``), so each row carries one when ``--dataset`` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode

import duckdb

from footy_track import splits
from footy_track.autolabel.clusters import crowd_score
from footy_track.feature_store.store import FeatureStore

PERSON_LABELS = ("player", "player_sub", "referee", "coach", "person")
DEFAULT_DB = "/mnt/storage/footy_data/feature_store/feature_store.duckdb"
FIELDS = (
    "clip",
    "frame_index",
    "crowd_score",
    "largest_cluster",
    "n_persons",
    "split",
    "url",
)

_SQL = f"""
    SELECT d.game_id, d.frame_index, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, d.source,
           f.width, f.height
    FROM detections_enriched d JOIN frame f USING (game_id, frame_index)
    WHERE d.canonical AND d.label IN ({", ".join("?" for _ in PERSON_LABELS)})
      AND NOT EXISTS (SELECT 1 FROM frame_complete c
                      WHERE c.game_id = d.game_id AND c.frame_index = d.frame_index)
"""


def open_readonly(path: str) -> FeatureStore:
    return FeatureStore(duckdb.connect(path, read_only=True))


def rank_frames(
    store: FeatureStore,
    *,
    clips: set[str] | None = None,
    split: str = "train",
    min_gap: int = 25,
    top: int | None = None,
    split_fn: Callable[[str], str] = splits.split_of_name,
) -> list[dict]:
    """Ranked picks, most crowded first. Frames with a hand-labelled person box are skipped."""
    by_frame: dict[tuple[str, int], list] = defaultdict(list)
    hand: set[tuple[str, int]] = set()
    df = store.query(_SQL, list(PERSON_LABELS))
    for gid, fi, x, y, w, h, src, fw, fh in df.itertuples(index=False):
        if clips is not None and gid not in clips:
            continue
        key = (gid, int(fi))
        if src == "hand_label":
            hand.add(key)
        by_frame[key].append(
            (x * fw, y * fh, w * fw, h * fh)
        )  # pixels: IoU needs the real aspect
    split_cache: dict[str, str] = {}
    scored = []
    for key, boxes in by_frame.items():
        if key in hand:
            continue
        if key[0] not in split_cache:
            split_cache[key[0]] = split_fn(key[0])
        sp = split_cache[key[0]]
        if split not in ("all", sp):
            continue
        s = crowd_score(boxes)
        scored.append((s.score, key, s, sp))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked: list[dict] = []
    taken: dict[str, list[int]] = defaultdict(list)
    for score, (gid, fi), s, sp in scored:
        if any(abs(fi - o) < min_gap for o in taken[gid]):
            continue
        taken[gid].append(fi)
        picked.append({
            "clip": gid, "frame_index": fi, "crowd_score": round(score, 4),
            "largest_cluster": s.largest_cluster, "n_persons": s.n_boxes, "split": sp,
        })  # fmt: skip
        if top and len(picked) >= top:
            break
    return picked


def deep_link(base: str, dataset: str, clip: str, frame: int) -> str:
    return f"{base.rstrip('/')}/labeller?" + urlencode(
        {"dataset": dataset, "clip": clip, "frame": frame}
    )


def format_table(rows: list[dict]) -> str:
    cols = [c for c in FIELDS if any(c in r for r in rows)] or list(FIELDS[:-1])
    cells = [[str(r.get(c, "")) for c in cols] for r in rows]
    widths = [
        max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)
    ]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True))]
    lines += [
        "  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)) for row in cells
    ]
    return "\n".join(lines)


def write_out(rows: list[dict], out: Path) -> None:
    if out.suffix == ".csv":
        cols = [c for c in FIELDS if c != "url" or (rows and "url" in rows[0])]
        with out.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            wr.writerows(rows)
    else:
        out.write_text(json.dumps(rows, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--clips", help="comma-separated clip ids (game_id)")
    ap.add_argument("--match", help="only clips whose id starts with this match key")
    ap.add_argument(
        "--split", choices=("train", "eval_heldout_match", "all"), default="train"
    )
    ap.add_argument("--min-gap", type=int, default=25)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--out", type=Path, help=".json or .csv")
    ap.add_argument("--dataset", help="labeller dataset name; adds a deep-link column")
    ap.add_argument("--labeller-url", default="http://localhost:8011")
    a = ap.parse_args(argv)

    if a.split in ("eval_heldout_match", "all"):
        print(
            "REMINDER: heldout-match labels are EVAL-ONLY; never export them into training data.",
            file=sys.stderr,
        )
    clips = {c for c in a.clips.split(",") if c} if a.clips else None
    with open_readonly(a.db) as store:
        rows = rank_frames(
            store, clips=clips, split=a.split, min_gap=a.min_gap, top=None
        )
    if a.match:
        rows = [r for r in rows if r["clip"].startswith(a.match)]
    rows = rows[: a.top] if a.top else rows
    if a.dataset:
        for r in rows:
            r["url"] = deep_link(a.labeller_url, a.dataset, r["clip"], r["frame_index"])
    if a.out:
        write_out(rows, a.out)
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
