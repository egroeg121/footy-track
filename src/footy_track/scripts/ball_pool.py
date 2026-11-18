"""Per-match stages of the training miner's candidate pool (``docs/design/ball_sampling.md`` §3.1, §6 Loop 1, train branch).

    python -m footy_track.scripts.ball_pool candidates --match M --weights W [--device cpu] [--clips a,b] [--limit N]
    python -m footy_track.scripts.ball_pool motion     --match M [--detector-sha8 S] [--max-minutes 40]
    python -m footy_track.scripts.ball_pool pool       --match M [--detector-sha8 S] [--sam3-dir D]
    python -m footy_track.scripts.ball_pool score      --match M [--scorer latest] [--detector-sha8 S]
    python -m footy_track.scripts.ball_pool status     [--match M | --all] [--scorer latest]
    python -m footy_track.scripts.ball_pool probe-frames --match M [--every 100] [--max-gap 300]   (optional SAM 3 probes)
    python -m footy_track.scripts.ball_pool train-scorer [--scorer-root /mnt/storage/models/ball_scorer] [--dry-run]

Layout (``--pool-root``, default ``--root`` = ``$FOOTY_DATA`` or ``/mnt/storage/footy_data``)::

    ball_pool/<match>/candidates.<detector_sha8>/   RT-DETR @1280 every 5th frame of the main-camera segments (GPU)
    ball_pool/<match>/motion.<detector_sha8>/       camera-compensated motion blobs (CPU)
    ball_pool/<match>/nodes.<detector_sha8>.pq merged nodes + tracker + features, unscored   (+ .json)
    ball_pool/<match>/pool.<scorer_version>.parquet the scored pool, ``ball_mining.schema`` -- what the miner reads

Every stage is idempotent (a finished stage has its ``meta.json`` / output written last), refuses heldout matches (fail
closed through ``footy_track.splits``) and prints ONE json object on stdout (progress goes to stderr).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ROOT = "/mnt/storage/footy_data"


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _root(a: argparse.Namespace) -> Path:
    return Path(a.root or os.environ.get("FOOTY_DATA") or DEFAULT_ROOT)


def _pool_root(a: argparse.Namespace) -> Path:
    return Path(a.pool_root) if a.pool_root else _root(a)


def _code_sha() -> str | None:
    sha = os.environ.get("FOOTY_CODE_SHA")
    if sha:
        return sha
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=False,
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def check_train_match(match: str) -> None:
    """Fail closed: only a registered TRAIN match may be processed (heldout footage never feeds training)."""
    from footy_track import splits  # noqa: PLC0415

    if splits.unregistered_match(match, None):
        raise SystemExit(
            f"{match}: not in the match split register; refusing (fail closed)"
        )
    kind = str(splits.split_of(match, None))
    if kind != "train":
        raise SystemExit(
            f"{match}: split is {kind}, not train; heldout matches never enter the training pool"
        )


def _detector_sha8(a: argparse.Namespace, match_dir: Path) -> str:
    from footy_track.ball_mining import candidates as C  # noqa: N812, PLC0415

    if getattr(a, "detector_sha8", None):
        return a.detector_sha8
    if getattr(a, "weights", None):
        return C.sha8(C.sha256_file(a.weights))
    found = sorted(
        p.name.split(".", 1)[1] for p in match_dir.glob("candidates.*") if p.is_dir()
    )
    if len(found) == 1:
        return found[0]
    raise SystemExit(
        f"cannot tell which detector: pass --detector-sha8 / --weights (candidates present: {found})"
    )


def _emit(doc: dict) -> int:
    print(json.dumps(doc, sort_keys=True, default=str), flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- stages
def cmd_candidates(a: argparse.Namespace) -> int:
    from footy_track.ball_mining import candidates as C  # noqa: N812, PLC0415

    check_train_match(a.match)
    weights = Path(a.weights) if a.weights else None
    if weights is None:
        from footy_track.detectors.ultralytics import current_best_detector_path  # noqa: I001, PLC0415

        weights = current_best_detector_path()
    sha8 = C.sha8(C.sha256_file(weights))
    only = [c for c in a.clips.split(",") if c] if a.clips else None
    all_segs = C.segments_of(_root(a), a.match)
    segs = C.segments_of(_root(a), a.match, limit=a.limit, only=only)
    if not segs:
        raise SystemExit(
            f"{a.match}: no segments{' matching --clips' if only else ''} under {_root(a)}"
        )
    out = Path(a.out) if a.out else C.candidates_dir(_pool_root(a), a.match, sha8)
    if C.stage_done(out) and not a.force:
        return _emit(
            {
                "stage": "candidates",
                "match": a.match,
                "status": "exists",
                "detector_sha8": sha8,
                "dir": str(out),
            }
        )
    detector = C.make_detector(weights, imgsz=a.imgsz, conf=C.CONF, device=a.device)
    expected = (
        [s.clip for s in all_segs] if not (a.limit or only) else [s.clip for s in segs]
    )
    st = C.run_match(segs, out, detector=detector, weights=weights, stride=a.stride, imgsz=a.imgsz, batch=a.batch, expected=expected,
                     max_seconds=a.max_minutes * 60 if a.max_minutes else None, code_sha=_code_sha(), log=_err)  # fmt: skip
    return _emit(
        {
            "stage": "candidates",
            "match": a.match,
            "status": "complete" if st["complete"] else "partial",
            "detector_sha8": sha8,
            "dir": str(out),
            **st,
        }
    )


def cmd_motion(a: argparse.Namespace) -> int:
    from footy_track.ball_mining import candidates as C  # noqa: N812, PLC0415
    from footy_track.ball_mining import motion as M  # noqa: N812, PLC0415

    check_train_match(a.match)
    pdir = C.pool_dir(_pool_root(a), a.match)
    sha8 = _detector_sha8(a, pdir)
    cdir = pdir / f"candidates.{sha8}"
    out = Path(a.out) if a.out else M.motion_dir(pdir, sha8)
    params = M.MParams()
    prev = json.loads((out / C.META).read_text()) if (out / C.META).is_file() else {}
    if prev.get("complete") and prev.get("params_key") == params.key() and not a.force:
        return _emit(
            {
                "stage": "motion",
                "match": a.match,
                "status": "exists",
                "detector_sha8": sha8,
                "dir": str(out),
            }
        )
    segs = C.segments_of(_root(a), a.match)
    st = M.run_match(
        segs,
        cdir,
        out,
        params=params,
        max_seconds=a.max_minutes * 60 if a.max_minutes else None,
        code_sha=_code_sha(),
        log=_err,
    )
    per = st["seconds"] / max(st["frames"], 1)
    return _emit(
        {
            "stage": "motion",
            "match": a.match,
            "status": "complete" if st["complete"] else "partial",
            "detector_sha8": sha8,
            "dir": str(out),
            "s_per_frame": round(per, 4),
            **st,
        }
    )


def cmd_pool(a: argparse.Namespace) -> int:
    from footy_track.ball_mining import candidates as C  # noqa: N812, PLC0415
    from footy_track.ball_mining import pool as P  # noqa: N812, PLC0415

    check_train_match(a.match)
    pdir = C.pool_dir(_pool_root(a), a.match)
    sha8 = _detector_sha8(a, pdir)
    out = P.nodes_path(pdir, sha8)
    if out.is_file() and not a.force:
        meta = (
            json.loads(out.with_suffix(".json").read_text())
            if out.with_suffix(".json").is_file()
            else {}
        )
        if meta.get("motion") or not C.stage_done(pdir / f"motion.{sha8}"):
            return _emit(
                {
                    "stage": "pool",
                    "match": a.match,
                    "status": "exists",
                    "detector_sha8": sha8,
                    "path": str(out),
                }
            )
    meta = P.build_match_nodes(
        pdir, sha8, sam_dir=a.sam3_dir, log=_err, code_sha=_code_sha()
    )
    return _emit(
        {"stage": "pool", "match": a.match, "status": "built", "path": str(out), **meta}
    )


def cmd_score(a: argparse.Namespace) -> int:
    import pandas as pd  # noqa: PLC0415

    from footy_track.ball_mining import candidates as C  # noqa: N812, PLC0415
    from footy_track.ball_mining import pool as P  # noqa: N812, PLC0415
    from footy_track.ball_mining import schema as S  # noqa: N812, PLC0415
    from footy_track.ball_mining import scorer as SC  # noqa: N812, PLC0415

    check_train_match(a.match)
    pdir = C.pool_dir(_pool_root(a), a.match)
    sha8 = _detector_sha8(a, pdir)
    sc = SC.load(a.scorer, a.scorer_root)
    out = Path(a.out) if a.out else pdir / f"pool.{sc.version}.parquet"
    meta_p = out.with_suffix(".json")
    prev = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    # the file name carries only the scorer version: a pool built from another detector checkpoint is stale and is rewritten
    if out.is_file() and prev.get("detector_sha8") == sha8 and not a.force:
        return _emit(
            {
                "stage": "score",
                "match": a.match,
                "status": "exists",
                "scorer_version": sc.version,
                "path": str(out),
            }
        )
    nodes_p = P.nodes_path(pdir, sha8)
    if not nodes_p.is_file():
        raise SystemExit(f"{nodes_p} missing: run the pool stage first")
    nodes = pd.read_parquet(nodes_p)
    scored = sc.apply(nodes)
    keep = [c for c in S.COLUMNS if c in scored.columns]
    pool = scored[keep]
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    S.write_pool(pool, tmp)
    os.replace(tmp, out)
    C.atomic_write_json(meta_p, {"match": a.match, "detector_sha8": sha8, "scorer_version": sc.version, "n_nodes": int(len(pool)), "mode": sc.manifest.get("mode"), "code_sha": _code_sha(), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})  # fmt: skip
    gap = scored["gap_frame"].eq(True)
    return _emit({
        "stage": "score", "match": a.match, "status": "scored", "scorer_version": sc.version, "path": str(out), "n_nodes": int(len(pool)),
        "n_gap_nodes": int(gap.sum()), "p_real_mean": float(scored["p_real"].mean()) if len(scored) else None,
        "n_p_ge_0.5": int((scored["p_real"] >= 0.5).sum()), "n_p_ge_0.95": int((scored["p_real"] >= 0.95).sum()),
    })  # fmt: skip


def match_status(
    root: Path, pool_root: Path, match: str, scorer_version: str | None
) -> dict:
    from footy_track.ball_mining import candidates as C  # noqa: N812, PLC0415

    pdir = C.pool_dir(pool_root, match)
    segs = []
    with contextlib.suppress(OSError, ValueError):
        segs = C.segments_of(root, match)

    def stage(prefix: str) -> dict:
        res = {}
        for p in sorted(pdir.glob(f"{prefix}.*")):
            if p.is_dir():
                meta = {}
                with contextlib.suppress(OSError, ValueError):
                    meta = json.loads((p / C.META).read_text())
                res[p.name.split(".", 1)[1]] = {
                    "complete": bool(meta.get("complete")),
                    "clips": len(meta.get("clips") or {}),
                    "expected": len(meta.get("expected") or []),
                }
        return res

    return {
        "match": match, "segments": len(segs), "candidates": stage("candidates"), "motion": stage("motion"),
        "nodes": sorted(p.name for p in pdir.glob("nodes.*.pq")),
        "pools": sorted(p.name for p in pdir.glob("pool.*.parquet")),
        "current_scorer": scorer_version, "current_pool": bool(scorer_version) and (pdir / f"pool.{scorer_version}.parquet").is_file(),
    }  # fmt: skip


def cmd_status(a: argparse.Namespace) -> int:
    from footy_track.ball_mining import scorer as SC  # noqa: N812, PLC0415

    ver = SC.latest_version(a.scorer_root) if a.scorer == "latest" else a.scorer
    if a.all:
        from footy_track import splits  # noqa: PLC0415

        matches = sorted(
            p.parent.parent.name
            for p in _root(a).glob("*/split_video_broadcast_frames/segments.json")
        )
        matches = [
            m
            for m in matches
            if not splits.unregistered_match(m, None)
            and str(splits.split_of(m, None)) == "train"
        ]
    else:
        matches = [a.match]
    return _emit(
        {
            "stage": "status",
            "scorer": ver,
            "matches": [match_status(_root(a), _pool_root(a), m, ver) for m in matches],
        }
    )


def cmd_probe_frames(a: argparse.Namespace) -> int:
    """Frames for the optional SAM 3 text probes: every ``--every`` frames of each clip (multiples of the stride) plus the
    tracker's gap frames (capped by ``--max-gap``, longest-and-most-plausible first)."""
    import pandas as pd  # noqa: PLC0415

    from footy_track.ball_mining import candidates as C  # noqa: N812, PLC0415
    from footy_track.ball_mining import pool as P  # noqa: N812, PLC0415

    check_train_match(a.match)
    pdir = C.pool_dir(_pool_root(a), a.match)
    sha8 = _detector_sha8(a, pdir)
    cdir = pdir / f"candidates.{sha8}"
    if not C.stage_done(cdir):
        raise SystemExit(f"candidates in {cdir} are not complete")
    meta = json.loads((cdir / C.META).read_text())
    stride = int(meta["stride"])
    every = max(a.every - a.every % stride, stride)
    frames: dict[str, list[int]] = {}
    for clip in meta["expected"]:
        z = C.load_clip(cdir / f"{clip}.npz")
        frames[clip] = sorted(int(f) for f in z.frames if int(f) % every == 0)
    nodes_p = P.nodes_path(pdir, sha8)
    n_gap = 0
    if nodes_p.is_file():
        df = pd.read_parquet(
            nodes_p, columns=["clip", "frame", "gap_frame", "source_scores"]
        )
        g = df[df["gap_frame"].eq(True)].copy()
        g["p"] = g["source_scores"].map(lambda v: float(v[0]))
        g = g.sort_values("p", ascending=False).head(a.max_gap)
        for clip, f in zip(g["clip"], g["frame"], strict=True):
            frames.setdefault(clip, []).append(int(f))
        n_gap = len(g)
    frames = {c: sorted(set(v)) for c, v in frames.items() if v}
    out = Path(a.out) if a.out else pdir / f"sam3_probe_frames.{sha8}.json"
    out.write_text(
        json.dumps(
            {
                "detector_sha8": sha8,
                "every": every,
                "n_gap_frames": n_gap,
                "frames": frames,
            },
            indent=1,
        )
    )
    return _emit(
        {
            "stage": "probe-frames",
            "match": a.match,
            "detector_sha8": sha8,
            "path": str(out),
            "n_frames": sum(map(len, frames.values())),
            "n_clips": len(frames),
            "n_gap_frames": n_gap,
        }
    )


def cmd_train_scorer(a: argparse.Namespace) -> int:
    from footy_track.ball_mining import scorer as SC  # noqa: N812, PLC0415
    from footy_track.ball_mining import scorer_data as SD  # noqa: N812, PLC0415

    t0 = time.time()
    table = SD.build_table(log=_err)
    _err(
        f"table: {len(table)} nodes, {int((table['label'] == 1).sum())} positive, in {time.time() - t0:.0f} s"
    )
    if a.save_table:
        table.to_parquet(a.save_table)
    scorer, manifest = SC.train(
        table,
        code_sha=_code_sha(),
        source={
            "experiment": "exp/ball-candidate-pool@07930b1",
            "dir": str(SD.EXP),
            "stride": SD.STRIDE,
        },
    )
    where = None
    if not a.dry_run:
        where = str(SC.save(scorer, a.scorer_root))
    pooled = manifest["lomo"]["pooled"]
    return _emit(
        {
            "stage": "train-scorer",
            "scorer_version": scorer.version,
            "dir": where,
            "training": manifest["training"],
            "lomo_pooled": pooled,
        }
    )


# ---------------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m footy_track.scripts.ball_pool",
        description=__doc__.split("\n")[0],
    )
    ap.add_argument(
        "--root",
        default=None,
        help="footy_data root holding <match>/split_video_broadcast_frames (default $FOOTY_DATA)",
    )
    ap.add_argument(
        "--pool-root",
        default=None,
        help="where ball_pool/<match>/ is read / written (default --root)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def match(p: argparse.ArgumentParser) -> None:
        p.add_argument("--match", required=True)
        p.add_argument("--detector-sha8", default=None)
        p.add_argument("--force", action="store_true")
        p.add_argument("--out", default=None)

    p = sub.add_parser(
        "candidates", help="RT-DETR at stride 5 over the main-camera segments (GPU)"
    )
    match(p)
    p.add_argument(
        "--weights",
        default=None,
        help="detector checkpoint (default: the production one)",
    )
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--clips", default="", help="comma-separated clip stems (dry runs)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-minutes", type=float, default=0)
    p.set_defaults(fn=cmd_candidates)

    p = sub.add_parser("motion", help="camera-compensated motion blobs (CPU)")
    match(p)
    p.add_argument("--max-minutes", type=float, default=0)
    p.set_defaults(fn=cmd_motion)

    p = sub.add_parser("pool", help="merge sources into nodes, run the tracker (CPU)")
    match(p)
    p.add_argument(
        "--sam3-dir", default=None, help="SAM 3 probe jsonl directory (optional source)"
    )
    p.set_defaults(fn=cmd_pool)

    p = sub.add_parser(
        "score", help="apply the scorer: nodes -> pool.<scorer>.parquet (CPU)"
    )
    match(p)
    p.add_argument("--scorer", default="latest")
    p.add_argument("--scorer-root", default=None)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser(
        "status", help="stage state of one match / every segmented train match"
    )
    p.add_argument("--match", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--scorer", default="latest")
    p.add_argument("--scorer-root", default=None)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser(
        "probe-frames", help="frame list for the optional SAM 3 text probes"
    )
    match(p)
    p.add_argument("--every", type=int, default=100)
    p.add_argument("--max-gap", type=int, default=300)
    p.set_defaults(fn=cmd_probe_frames)

    p = sub.add_parser(
        "train-scorer",
        help="build the labelled table from the experiment's data, fit, save a versioned scorer",
    )
    p.add_argument("--scorer-root", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--save-table", default=None)
    p.set_defaults(fn=cmd_train_scorer)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "status" and not a.all and not a.match:
        raise SystemExit("status needs --match or --all")
    if getattr(a, "scorer_root", None) is None and hasattr(a, "scorer_root"):
        a.scorer_root = (
            os.environ.get("FOOTY_BALL_SCORER_ROOT")
            or "/mnt/storage/models/ball_scorer"
        )
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
