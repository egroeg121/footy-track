"""Measure ball trackers against human ball marks.

    # 1. detect once per clip (GPU), cache every ball candidate
    python -m footy_track.scripts.ball_tracking_eval cache \
        --videos clips_dir --gt gt_dir --out cache_dir --weights best.pt

    # 2+3. replay trackers on the cache (CPU) and score against GT
    python -m footy_track.scripts.ball_tracking_eval score \
        --cache cache_dir --gt gt_dir --split split.json \
        --trackers online,bidirectional,viterbi

``--split`` is a JSON object ``{"<canonical clip id>": "seen" | "held_out"}``; the
headline numbers are the held-out clips. ``--param tracker.key=value`` overrides a
tracker parameter. See docs/ball-tracking-eval.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from footy_track.ball_eval.detection_replay import (
    HUMAN,
    PROPAGATED,
    TABLE_HEADER,
    build_cache,
    cache_from_detections_jsonl,
    evaluate,
    format_counts,
    load_gt,
    pool,
    read_cache,
    replay,
    write_cache,
)
from footy_track.ball_trackers.factory import TRACKER_KINDS, make_ball_tracker
from footy_track.clip_id import canonical_clip_id

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def _find_videos(video_dirs: list[Path], wanted: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for d in video_dirs:
        for p in sorted(d.iterdir()):
            cid = canonical_clip_id(p.name)
            if p.suffix.lower() in VIDEO_SUFFIXES and cid in wanted:
                found.setdefault(cid, p)
    return found


def cmd_cache(args: argparse.Namespace) -> int:
    from footy_track.detectors.ultralytics import (  # noqa: PLC0415
        UltralyticsObjectDetector,
    )
    from footy_track.inference.pipeline import select_device  # noqa: PLC0415

    gts = load_gt(args.gt)
    wanted = (
        set(args.clip)
        if args.clip
        else {c for c, g in gts.items() if g.marks[HUMAN] or g.no_ball}
    )
    videos = _find_videos(args.videos, wanted)
    for missing in sorted(wanted - set(videos)):
        print(f"no video for {missing}")
    detector = UltralyticsObjectDetector(
        model_uri=str(args.weights),
        min_confidence=args.detect_conf,
        imgsz=args.imgsz,
        use_model_names=True,
        device=args.device or select_device(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for clip, video in videos.items():
        target = args.out / f"{clip}.jsonl"
        if target.exists() and not args.force:
            print(f"{clip}: cached")
            continue
        cache = build_cache(video, detector, clip, batch_size=args.batch_size)
        cache.meta.update(
            weights=args.weights.parent.name,
            imgsz=args.imgsz,
            detect_conf=args.detect_conf,
        )
        write_cache(target, cache)
        print(f"{clip}: {len(cache.frames)} frames", flush=True)
    return 0


def _overrides(items: list[str]) -> dict[str, dict[str, float | str]]:
    out: dict[str, dict[str, float | str]] = {}
    for item in items:
        key, value = item.split("=", 1)
        tracker, name = key.split(".", 1)
        try:
            parsed: float | str = float(value)
        except ValueError:
            parsed = value
        out.setdefault(tracker, {})[name] = parsed
    return out


def cmd_score(args: argparse.Namespace) -> int:
    gts = load_gt(args.gt)
    caches = {p.stem: read_cache(p) for p in sorted(args.cache.glob("*.jsonl"))}
    split = json.loads(args.split.read_text()) if args.split else {}
    overrides = _overrides(args.param)
    radii = tuple(float(r) for r in args.radii.split(","))
    groups = {
        "held-out": [c for c in caches if split.get(c) == "held_out"],
        "seen": [c for c in caches if split.get(c) == "seen"],
        "all": list(caches),
    }
    tiers_sets = {"human": (HUMAN,), "human+propagated": (HUMAN, PROPAGATED)}
    lines: list[str] = []
    for name, tiers in tiers_sets.items():
        lines += ["", f"## GT tier: {name}", ""]
        for kind in args.trackers.split(","):
            params = overrides.get(kind, {})
            per_clip = evaluate(
                caches,
                gts,
                lambda size, kind=kind, params=params: make_ball_tracker(
                    kind, size, **params
                ),
                tiers=tiers,
                radii=radii,
            )
            lines += [f"### {kind}", "", *TABLE_HEADER]
            for group, clips in groups.items():
                if group == "seen" and not split:
                    continue
                lines += format_counts(group, pool(per_clip, clips))
            if args.per_clip:
                lines += ["", *TABLE_HEADER]
                for clip, counts in per_clip.items():
                    lines += format_counts(f"{clip} [{split.get(clip, '?')}]", counts)
            lines.append("")
    print("\n".join(lines))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """No-GT proxy: how many frames each tracker changes vs online, and how many
    of its positions sit in the stands (normalised y under ``--crowd-y``)."""
    kinds = args.trackers.split(",")
    print(
        "| clip | frames | "
        + " | ".join(
            f"{k}: ball frames / stands"
            + ("" if k == "online" else " / differs from online")
            for k in kinds
        )
        + " |"
    )
    print("|---|---|" + "---|" * len(kinds))
    totals = {k: [0, 0, 0] for k in kinds}
    n_total = 0
    for d in sorted(
        p for p in args.runs.iterdir() if (p / "detections.jsonl").exists()
    ):
        size = (1920, 1080)
        summary = d / "summary.json"
        if summary.exists():
            cfg = json.loads(summary.read_text())
            size = tuple(cfg.get("frame_size", size))
        cache = cache_from_detections_jsonl(d / "detections.jsonl", size, d.name)
        results = {k: replay(make_ball_tracker(k, size), cache) for k in kinds}
        base = results[kinds[0]]
        cells = []
        for k in kinds:
            pts = results[k]
            have = [p for p in pts if p.x is not None]
            stands = sum(1 for p in have if p.y + p.h / 2 < args.crowd_y)
            diff = sum(
                1 for a, b in zip(base, pts, strict=True) if _differs(a, b, size)
            )
            cells.append(
                f"{len(have)} / {stands}" + ("" if k == kinds[0] else f" / {diff}")
            )
            totals[k][0] += len(have)
            totals[k][1] += stands
            totals[k][2] += diff
        n_total += len(base)
        print(f"| {d.name} | {len(base)} | " + " | ".join(cells) + " |")
    print(
        f"| total | {n_total} | "
        + " | ".join(
            f"{t[0]} / {t[1]}" + ("" if k == kinds[0] else f" / {t[2]}")
            for k, t in totals.items()
        )
        + " |"
    )
    return 0


def _differs(a, b, size, tol_px: float = 5.0) -> bool:
    if (a.x is None) != (b.x is None):
        return True
    if a.x is None:
        return False
    return math.hypot((a.x - b.x) * size[0], (a.y - b.y) * size[1]) > tol_px


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="run the detector once per GT clip")
    c.add_argument("--videos", type=Path, nargs="+", required=True)
    c.add_argument("--gt", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--weights", type=Path, required=True)
    c.add_argument(
        "--clip", nargs="*", help="canonical clip ids (default: all with GT)"
    )
    c.add_argument("--device", default=None)
    c.add_argument("--imgsz", type=int, default=1280)
    c.add_argument("--detect-conf", type=float, default=0.1)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cmd_cache)

    s = sub.add_parser("score", help="replay trackers on the cache and score them")
    s.add_argument("--cache", type=Path, required=True)
    s.add_argument("--gt", type=Path, required=True)
    s.add_argument("--split", type=Path, default=None)
    s.add_argument("--trackers", default=",".join(TRACKER_KINDS))
    s.add_argument("--radii", default="10,25")
    s.add_argument("--param", action="append", default=[], metavar="TRACKER.KEY=VALUE")
    s.add_argument("--per-clip", action="store_true")
    s.set_defaults(fn=cmd_score)

    r = sub.add_parser("review", help="proxy comparison on runs without GT")
    r.add_argument(
        "--runs", type=Path, required=True, help="dir of <clip>/detections.jsonl"
    )
    r.add_argument("--trackers", default=",".join(TRACKER_KINDS))
    r.add_argument("--crowd-y", type=float, default=0.12)
    r.set_defaults(fn=cmd_review)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
