"""GPU half of the tracking detector eval: run the detector over the eval clips, store the output.

One pass over *every* frame of each eval clip yields both products the nightly needs:

* ``dets/<clip>.npz``     every detection of every class (conf >= 0.1), scored on CPU by
                          :class:`footy_track.datasets.eval_adapters.DetectorAdapter`;
* ``cache/<clip>.jsonl``  the ball candidates (the ``ball_tracking_eval cache`` format) that
                          :class:`~footy_track.datasets.eval_adapters.BallTrackerAdapter` replays
                          trackers over on CPU.

Frame sourcing. Items are labelled frames of a clip (``game_id`` = the video stem, ``frame_index`` =
index into that video), so the frames are decoded from the clip video, sequentially from frame 0,
exactly as ``ball_tracking_eval cache`` does (sequential decode is frame-exact; seeking is not). The
video is the one the labeller showed: the labeller-registry clip dirs (the ``*_kf10`` re-encodes)
first, then ``<FOOTY_DATA>/clips``. Aliases of one clip (``arsenal_mancity_20250925_seg120`` /
``arsenal_mancity_seg120``) are the same video file (see ``footy_track.clip_id``), so videos are keyed
by canonical clip id and any alias's file serves. Detections are keyed the same way.

Idempotent / checkpointed per clip: a clip is done when both of its files exist and were made with
the same settings; both are written to a temp name and renamed (dets first, cache last), so a crash
mid-clip leaves nothing that looks complete and a rerun redoes only that clip.

    python -m footy_track.datasets.detector_run plan --dataset tracking --version N
    python -m footy_track.datasets.detector_run run  --dataset tracking --version N \\
        --weights best.pt --videos DIR --out DIR/<sha12>
    python -m footy_track.datasets.detector_run score --dataset tracking --version N --dets DIR/<sha12>/dets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from footy_track.datasets.detector_eval import CONF_FLOOR, ClipDetections
from footy_track.datasets.snapshot import DatasetVersion, load_version

DEFAULT_IMGSZ = 1280  # the pipeline's serving size (the weights were trained at 1920)
DEFAULT_BATCH = 8
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi")
EVAL_SPLITS = ("eval_heldout_match", "eval_in_match")


# --------------------------------------------------------------------------------------------
# plan: which clips / which videos
# --------------------------------------------------------------------------------------------
def data_root() -> Path:
    return Path(os.environ.get("FOOTY_DATA", "/mnt/storage/footy_data"))


def video_dirs() -> list[Path]:
    """Directories to look for clip videos, in priority order: every labeller-registry ``clips_dir`` /
    ``extra_clips_dirs`` (the ``*_kf10`` clips the labels were drawn on), then ``<FOOTY_DATA>/clips``."""
    reg = Path(os.environ.get("FOOTY_LABELLER_DATASETS") or data_root() / "labeller_datasets.json")
    dirs: list[Path] = []
    try:
        for d in json.loads(reg.read_text()).get("datasets", {}).values():
            for p in (d.get("clips_dir"), *(d.get("extra_clips_dirs") or [])):
                if p and Path(p) not in dirs:
                    dirs.append(Path(p))
    except (OSError, ValueError, AttributeError):
        pass
    plain = data_root() / "clips"
    if plain not in dirs:
        dirs.append(plain)
    return dirs


def eval_clip_games(dv: DatasetVersion, split: str = "eval_all") -> dict[str, set[str]]:
    """``{canonical clip: {game_id (video stem) of its items}}`` for the eval items of ``dv``."""
    out: dict[str, set[str]] = defaultdict(set)
    for r in dv.select(split):
        out[r["clip"]].add(r["game_id"])
    return dict(out)


def resolve_videos(clip_games: dict[str, set[str]], dirs: list[Path]) -> tuple[dict[str, Path], list[str]]:
    """``({clip: video path}, [clips without a video])``. A clip's own alias stems are tried in every
    dir in priority order, then any file in a dir whose canonical id matches."""
    found: dict[str, Path] = {}
    for clip, games in sorted(clip_games.items()):
        for d in dirs:
            for stem in (*sorted(games), clip):
                for ext in VIDEO_SUFFIXES:
                    p = d / f"{stem}{ext}"
                    if p.is_file():
                        found.setdefault(clip, p)
                        break
                if clip in found:
                    break
            if clip in found:
                break
    return found, sorted(set(clip_games) - set(found))


def plan(dv: DatasetVersion) -> dict[str, Any]:
    """What a GPU run needs: eval clips, their videos (on THIS host) and sizes, the frames labelled per clip."""
    games = eval_clip_games(dv)
    videos, missing = resolve_videos(games, video_dirs())
    frames: dict[str, list[int]] = defaultdict(list)
    for r in dv.select("eval_all"):
        frames[r["clip"]].append(int(r["frame_index"]))
    return {
        "dataset": dv.dataset, "version": dv.version, "snapshot_dir": str(dv.path),
        "clips": {c: {"video": str(p), "bytes": p.resolve().stat().st_size, "games": sorted(games[c]),
                      "labelled_frames": len(frames[c])} for c, p in videos.items()},
        "missing": missing,
    }


# --------------------------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------------------------
def settings(imgsz: int, conf: float) -> dict:
    return {"imgsz": imgsz, "conf": conf, "conf_floor": CONF_FLOOR}


def clip_done(out: Path, clip: str, want: dict) -> bool:
    dets, cache = out / "dets" / f"{clip}.npz", out / "cache" / f"{clip}.jsonl"
    if not (dets.is_file() and cache.is_file()):
        return False
    try:
        return ClipDetections.load(dets).meta.get("settings") == want
    except Exception:  # noqa: BLE001 - a truncated / foreign file counts as not done
        return False


def detect_clip(video: Path, detector: Any, clip: str, *, batch_size: int = DEFAULT_BATCH,
                on_batch: Callable[[int], None] | None = None) -> tuple[ClipDetections, Any]:
    """Run ``detector`` (``predict_frames(list[bgr]) -> list[FrameDetections]``) over every frame of
    ``video``. Returns the stored detections and the ball-candidate cache of the same pass."""
    import cv2  # noqa: PLC0415

    from footy_track.ball_eval.detection_replay import CandidateCache  # noqa: PLC0415
    from footy_track.inference.pipeline import (  # noqa: PLC0415
        ball_candidates,
        iter_frames,
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rows: list[tuple[int, str, tuple[float, float, float, float], float]] = []
    cache_frames: list = []
    done: list[int] = []
    batch: list[tuple[int, Any]] = []

    def flush() -> None:
        results = detector.predict_frames([f for _, f in batch])
        for (index, _), res in zip(batch, results, strict=True):
            done.append(index)
            cache_frames.append((index, ball_candidates(res)))
            rows.extend((index, d.label, (d.x, d.y, d.w, d.h), d.confidence) for d in res.detections)
        if on_batch:
            on_batch(len(batch))
        batch.clear()

    try:
        for item in iter_frames(cap, 0, None, 1):
            batch.append(item)
            if len(batch) >= batch_size:
                flush()
        if batch:
            flush()
    finally:
        cap.release()
    names = [str(v) for _, v in sorted(getattr(detector, "classes", {}).items())]
    dets = ClipDetections.from_rows(clip, size, names, rows, done, {"fps": fps, "n_frames": len(done)})
    return dets, CandidateCache(clip, size, cache_frames, {"fps": fps})


def run_inference(
    videos: dict[str, Path], out: Path, *, detector: Any, weights_name: str = "", imgsz: int = DEFAULT_IMGSZ,
    conf: float = CONF_FLOOR, batch_size: int = DEFAULT_BATCH, log: Callable[[str], None] = print,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Process every clip in ``videos`` not yet done under ``out``; ``max_seconds`` stops starting new
    clips after that long (a rerun continues). Returns per-clip status."""
    from footy_track.ball_eval.detection_replay import write_cache  # noqa: PLC0415

    want = settings(imgsz, conf)
    status: dict[str, Any] = {"done": [], "skipped": [], "failed": {}, "frames": 0, "seconds": 0.0, "stopped_early": False}
    t_start = time.time()
    for n, (clip, video) in enumerate(sorted(videos.items()), 1):
        if clip_done(out, clip, want):
            status["skipped"].append(clip)
            continue
        if max_seconds is not None and time.time() - t_start > max_seconds:
            status["stopped_early"] = True
            break
        t0 = time.time()
        try:
            dets, cache = detect_clip(video, detector, clip, batch_size=batch_size)
        except Exception as e:  # noqa: BLE001 - one bad clip must not sink the run; it is retried next time
            status["failed"][clip] = f"{type(e).__name__}: {e}"
            log(f"[{n}/{len(videos)}] {clip}: FAILED {status['failed'][clip]}")
            continue
        dets.meta["settings"] = want
        dets.meta["weights"] = weights_name
        cache.meta.update(weights=weights_name, imgsz=imgsz, detect_conf=conf)
        cache_path = out / "cache" / f"{clip}.jsonl"
        tmp_cache = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
        write_cache(tmp_cache, cache)
        dets.save(out / "dets" / f"{clip}.npz")  # dets first ...
        os.replace(tmp_cache, cache_path)         # ... the cache last: both present = clip done
        dt = time.time() - t0
        nf = len(dets.frames)
        status["done"].append(clip)
        status["frames"] += nf
        status["seconds"] += dt
        log(f"[{n}/{len(videos)}] {clip}: {nf} frames, {nf / dt:.1f} fps")
    status["wall_seconds"] = round(time.time() - t_start, 1)
    return status


def make_detector(weights: Path, imgsz: int, conf: float, device: str | None = None) -> Any:
    from footy_track.detectors.ultralytics import UltralyticsObjectDetector  # noqa: PLC0415
    from footy_track.inference.pipeline import select_device  # noqa: PLC0415

    return UltralyticsObjectDetector(
        model_uri=str(weights), min_confidence=conf, imgsz=imgsz, use_model_names=True,
        device=device or select_device(),
    )


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _cmd_plan(a: argparse.Namespace) -> int:
    dv = load_version(a.dataset, a.version, a.root)
    p = plan(dv)
    print(json.dumps(p, sort_keys=True) if a.json else
          f"{len(p['clips'])} clips with a video ({sum(c['bytes'] for c in p['clips'].values()) / 1e9:.2f} GB), "
          f"{len(p['missing'])} without: {p['missing'][:10]}")
    return 1 if p["missing"] and a.strict else 0


def _cmd_run(a: argparse.Namespace) -> int:
    dv = load_version(a.dataset, a.version, a.root)
    clips = eval_clip_games(dv)
    videos: dict[str, Path] = {}
    for c in clips:
        for ext in VIDEO_SUFFIXES:
            p = a.videos / f"{c}{ext}"
            if p.is_file():
                videos[c] = p
                break
    missing = sorted(set(clips) - set(videos))
    detector = make_detector(a.weights, a.imgsz, CONF_FLOOR, a.device)
    st = run_inference(videos, a.out, detector=detector, weights_name=a.weights.parent.name, imgsz=a.imgsz,
                       batch_size=a.batch_size, log=lambda m: print(m, flush=True), max_seconds=a.max_seconds)
    st["missing_videos"] = missing
    print("::result::" + json.dumps(st, sort_keys=True))
    return 1 if st["failed"] else 0


def _cmd_score(a: argparse.Namespace) -> int:
    from footy_track.datasets.detector_eval import (  # noqa: PLC0415
        DetectionStore,
        score_items,
    )

    dv = load_version(a.dataset, a.version, a.root)
    store = DetectionStore(a.dets)
    for split in (*EVAL_SPLITS, "eval_all"):
        s = score_items(dv.select(split), store)
        m = {k: (None if v is None else round(v, 4)) for k, v in s.metrics.items()}
        print(f"{split} (n={s.n_items}): {json.dumps(m, sort_keys=True)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m footy_track.datasets.detector_run", description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=None, help="datasets root (default $FOOTY_DATASETS_ROOT / $FOOTY_DATA)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="eval clips + the video file of each on this host")
    p.add_argument("--dataset", default="tracking")
    p.add_argument("--version", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 1 when an eval clip has no video")
    p.set_defaults(fn=_cmd_plan)
    p = sub.add_parser("run", help="detector over every frame of every eval clip (GPU); resumable")
    p.add_argument("--dataset", default="tracking")
    p.add_argument("--version", type=int, default=None)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--videos", type=Path, required=True, help="dir holding <canonical clip>.mp4")
    p.add_argument("--out", type=Path, required=True, help="work dir; dets/ and cache/ are written under it")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--device", default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.set_defaults(fn=_cmd_run)
    p = sub.add_parser("score", help="score stored detections on the eval splits (CPU; prints, records nothing)")
    p.add_argument("--dataset", default="tracking")
    p.add_argument("--version", type=int, default=None)
    p.add_argument("--dets", type=Path, required=True)
    p.set_defaults(fn=_cmd_score)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
