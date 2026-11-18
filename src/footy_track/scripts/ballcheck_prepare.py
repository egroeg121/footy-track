"""Render a labelling job's Ball Check cards into the disk cache, in parallel.

The worker behind the ``ballcheck-prepare`` Prefect flow (``~/code/jobs/footy_ballcheck.py``), also
usable by hand. It takes the plan the labeller serves at ``GET /api/jobs/<job>/prepare_plan`` (every
card still to judge, its geometry, clip file and the exact cache keys) and renders each missing
still and clip with the *same functions the server uses* (``ball_check.render_crop_bytes`` /
``_render_clip``), so a prepared file is what the server would have made.

    python -m footy_track.scripts.ballcheck_prepare render --plan plan.json \\
        --clips-dir /mnt/storage/footy_data/arsenal_nottinghamforest/split_video_broadcast_frames_kf10 \\
        --out /mnt/storage/footy_data/ballcheck_renders --job j-05c22fd49a69 --workers 3 --nice 10

    python -m footy_track.scripts.ballcheck_prepare ingest --tar staged.tar --root /mnt/storage/footy_data/ballcheck_renders

``render`` writes ``<out>/blobs/..`` atomically and registers the job's keys in ``<out>/refs/``.
Run it with ``--out`` = the live cache on the mini, or into a staging directory on another machine
and ship it back as a tar for ``ingest`` (which merges it into the live cache the same atomic way).
One JSON object per line on stdout reports progress (``{"event": "progress", ...}``). Exit codes:
0 all rendered, 1 some cards failed, 3 stopped at the free-disk floor.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from footy_track.labeller import ball_check_cache as bcc

EXIT_OK, EXIT_FAILED, EXIT_FLOOR = 0, 1, 3


def _init_worker(nice: int) -> None:
    """One decode thread per worker: parallelism comes from the pool, not from OpenCV."""
    import cv2  # noqa: PLC0415

    cv2.setNumThreads(1)
    if nice:
        os.nice(nice)


def render_card(card: dict, clip_path: str, out: str) -> dict:
    """Render the missing variants of one card into the cache at ``out``. Runs in a worker."""
    from footy_track.labeller import ball_check  # noqa: PLC0415

    rc = bcc.RenderCache(Path(out))
    video = Path(clip_path)
    bbox = tuple(card["bbox"])
    ob = tuple(card["other_bbox"]) if card.get("other_bbox") else None
    f = card["frame_index"]
    made = 0
    t0 = time.monotonic()
    try:
        for v in card["variants"]:
            if rc.has(v["key"], v["ext"]):
                continue
            if v["kind"] == "crop":
                data = ball_check.render_crop_bytes(
                    video, f, f, bbox, ob, v["pad"], v.get("reticle", False)
                )
                meta: dict = {}
            else:
                res = ball_check._render_clip(
                    video,
                    card["source"],
                    card["clip"],
                    f,
                    card["box_index"],
                    bbox,
                    v["pad"],
                    ob,
                )
                if res is None:
                    data, meta = None, {}
                else:
                    data, media_type, n_frames, dur = res
                    if (media_type == "video/mp4") != (v["ext"] == "mp4"):
                        raise RuntimeError(
                            f"encoder mismatch: plan wants {v['ext']}, this host made {media_type}"
                        )
                    meta = {"type": media_type, "n": n_frames, "ms": dur}
            if data is None:
                return {
                    "position": card["position"],
                    "ok": False,
                    "error": f"{v['kind']} render returned nothing",
                }
            rc.put(v["key"], v["ext"], data, meta or None)
            made += 1
    except Exception as e:  # noqa: BLE001 - one bad card must not stop the batch
        return {
            "position": card["position"],
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }
    return {
        "position": card["position"],
        "ok": True,
        "made": made,
        "seconds": round(time.monotonic() - t0, 2),
    }


def _emit(**kw) -> None:
    print(json.dumps(kw), flush=True)


def run_render(  # noqa: PLR0912, PLR0915
    plan: dict,
    clips_dir: Path,
    out: Path,
    *,
    job: str | None,
    workers: int,
    nice: int,
    floor: int | None,
) -> int:
    rc = bcc.RenderCache(out)
    cards = sorted(plan["cards"], key=lambda c: c["position"])
    todo, bad = [], []
    sizes: dict[str, int | None] = {}
    for c in cards:
        name = c["video"]["name"]
        if name not in sizes:
            p = clips_dir / name
            sizes[name] = p.stat().st_size if p.is_file() else None
        if sizes[name] != c["video"]["size"]:
            bad.append({"position": c["position"], "ok": False,
                        "error": f"clip {name}: {'missing' if sizes[name] is None else 'size differs from the server copy'}"})  # fmt: skip
        elif any(not rc.has(v["key"], v["ext"]) for v in c["variants"]):
            todo.append(c)
    total = len(cards)
    already = total - len(todo) - len(bad)
    done = already
    failed = list(bad)
    _emit(event="start", cards=total, to_render=len(todo), already_cached=already, bad=len(bad),
          workers=workers, fingerprint=plan.get("fingerprint"))  # fmt: skip
    t0 = time.monotonic()
    stopped_by_floor = False
    if todo:
        ctx = multiprocessing.get_context("spawn")
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(nice,),
        ) as ex:
            it = iter(todo)
            running: dict[cf.Future, dict] = {}

            def submit_next() -> bool:
                nonlocal stopped_by_floor
                try:
                    bcc.check_floor(out, 0, floor)
                except bcc.DiskFloorError as e:
                    stopped_by_floor = True
                    _emit(event="error", error=str(e))
                    return False
                c = next(it, None)
                if c is None:
                    return False
                running[
                    ex.submit(
                        render_card, c, str(clips_dir / c["video"]["name"]), str(out)
                    )
                ] = c
                return True

            for _ in range(workers * 2):
                if not submit_next():
                    break
            while running:
                finished, _pending = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                for fut in finished:
                    running.pop(fut)
                    r = fut.result()
                    if r["ok"]:
                        done += 1
                    else:
                        failed.append(r)
                    _emit(event="progress", cards_done=done, cards_total=total, failed=len(failed),
                          elapsed=round(time.monotonic() - t0, 1), last=r)  # fmt: skip
                    if not stopped_by_floor:
                        submit_next()
    ok_positions = {c["position"] for c in cards} - {f["position"] for f in failed}
    if job:
        keys = [
            (v["key"], v["ext"])
            for c in cards
            if c["position"] in ok_positions
            for v in c["variants"]
        ]
        rc.register(job, keys)
    elapsed = time.monotonic() - t0
    _emit(event="done", cards=total, ok=len(ok_positions), failed=len(failed), rendered=len(todo),
          elapsed=round(elapsed, 1), cards_per_s=round(len(todo) / elapsed, 3) if elapsed and todo else None,
          errors=failed[:10])  # fmt: skip
    if stopped_by_floor:
        return EXIT_FLOOR
    return EXIT_FAILED if failed else EXIT_OK


def run_ingest(tar_path: Path, root: Path) -> int:
    """Merge a staged render tree (blobs + refs) into the live cache. Atomic per file; never
    overwrites a blob that is already there (same key = same bytes)."""
    rc = bcc.RenderCache(root)
    rc.tmp.mkdir(parents=True, exist_ok=True)
    n_blobs = n_refs = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            parts = Path(m.name).parts
            if (
                not m.isfile()
                or ".." in parts
                or not parts
                or parts[0] not in ("blobs", "refs")
            ):
                continue
            data = tf.extractfile(m).read()  # type: ignore[union-attr]
            if parts[0] == "refs":
                doc = json.loads(data)
                rc.register(doc["job"], [tuple(k) for k in doc.get("keys", [])])
                n_refs += 1
                continue
            dst = root.joinpath(*parts)
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix=dst.name + ".", suffix=".part", dir=rc.tmp
            )
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.chmod(tmp, 0o664)
            os.replace(tmp, dst)
            n_blobs += 1
    _emit(event="ingested", blobs=n_blobs, refs=n_refs)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render a plan's missing stills and clips")
    r.add_argument(
        "--plan",
        required=True,
        help="plan JSON (from GET /api/jobs/<job>/prepare_plan)",
    )
    r.add_argument("--clips-dir", required=True, type=Path)
    r.add_argument(
        "--out",
        required=True,
        type=Path,
        help="cache root to write (live cache, or a staging dir)",
    )
    r.add_argument("--job", help="register the rendered keys under this job id")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--nice", type=int, default=0)
    r.add_argument(
        "--min-free-gb",
        type=float,
        help=f"stop below this much free disk (default ${bcc.ENV_MIN_FREE_GB} or 20)",
    )
    i = sub.add_parser("ingest", help="merge a staged render tar into the live cache")
    i.add_argument("--tar", required=True, type=Path)
    i.add_argument("--root", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "ingest":
        return run_ingest(a.tar, a.root)
    plan = json.loads(Path(a.plan).read_text())
    floor = None if a.min_free_gb is None else int(a.min_free_gb * bcc.GB)
    return run_render(
        plan,
        a.clips_dir,
        a.out,
        job=a.job,
        workers=max(1, a.workers),
        nice=a.nice,
        floor=floor,
    )


if __name__ == "__main__":
    sys.exit(main())
