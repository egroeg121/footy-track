"""The labeller's own ball tracker as a fourth tracker: the REAL labeller code path (footy_track.labeller.video_utils on origin/main) run headless (CPU tracker + GPU detector).

Code used unchanged (imported from a snapshot of origin/main, ``--main-src``): ``yolo_seed_objects`` (what ``POST /autodetect`` calls: ``get_current_best_detector`` = RT-DETR-L v11
@1280, conf 0.35, NMS iou 0.5), ``BackgroundLabeller`` (what the Run button drives: worker thread, ``_track_anomaly_reason`` jump / size anomaly stop, per-class confidence handback),
``VitTrackVideoLabeller`` / ``VitTrackSOT`` (OpenCV-zoo ViT SOT, ONNX).  Only the ball is seeded (every object has its own independent tracker and the anomaly / handback checks are per
label, so the ball track is identical to the one in a run that also carries ~18 players; it is ~18x cheaper).  The detector is wrapped in a memo (same weights, same call) so a clip's
re-detections do not reload the checkpoint each time.

  arm 1  "labeller-realistic": autodetect on frame 0 -> take the best ball box -> Run.  When the run hands back (confidence below the class threshold or an anomaly), the headless
         "human" does what the labeller offers: autodetect on the stop frame, take its best ball box (never a hand-drawn one), Run again from there; if the detector has no ball
         there, step forward and retry (<= ``--max-scan`` frames) and give up on the clip tail after that.  Class thresholds = the page defaults {in_play_ball: 0.5, other: 0.5}
         (index.html DEFAULT_THRESHOLDS), a run therefore pauses as soon as a ball box scores < 0.5 (``--ball-thr`` to change).
  arm 2  "comparable": per seed of the shared seed file (RT-DETR >= 0.8 anchors + SAM 3 text seeds), forward and backward (backward = the same labeller code on a reversed copy of the
         clip, the labeller has no backward Run), never past the neighbouring seed + overlap, up to reach frames; hand-backs are NOT applied (the track runs on), but every frame records
         the VitTrack score and whether the labeller would have stopped there (score < ball threshold, or ``_track_anomaly_reason``), computed with the labeller's own functions.

    python vittrack_track.py --main-src DIR --seeds seeds.jsonl --videos DIR --out DIR --config '{"arm": 2}' --clips a,b
Outputs per clip: ``<clip>.labels.jsonl`` (rows: frame, x, y, w, h centre / size px, score, kind, run, ...) and ``<clip>.raw.json.gz``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def setup(main_src: str):
    sys.path.insert(0, main_src)
    import footy_track.detectors.ultralytics as U
    memo = {}
    orig = U.get_current_best_detector

    def cached(min_confidence=0.3, iou_threshold=0.90, verbose=False):
        k = (min_confidence, iou_threshold)
        if k not in memo:
            memo[k] = orig(min_confidence=min_confidence, iou_threshold=iou_threshold, verbose=verbose)
        return memo[k]

    U.get_current_best_detector = cached
    import footy_track.labeller.video_utils as VU
    return VU, U


def detect_balls(VU, video: Path, frame: int, w: int, h: int, conf: float = 0.35, iou: float = 0.5) -> list:
    """What /autodetect adds to a frame: the detector's boxes after NMS, ball-class ones, best first (``_nms_filter`` keeps the confidence order)."""
    objs = VU.yolo_seed_objects(video, "", conf, w, h, iou, frame)
    return [o for o in objs if o.label in VU._BALL_LABELS or "ball" in o.label]


def frames_of(fds, start: int, VU):
    out = {}
    for fd in fds:
        if fd is None:
            continue
        out[VU._frame_index_from_uri(fd, start)] = fd
    return out


def row(fd, VU, w, h, kind, **kw):
    d = fd.detections[0] if fd.detections else None
    if d is None:
        return None
    return {"x": round((d.x + d.w / 2) * w, 2), "y": round((d.y + d.h / 2) * h, 2), "w": round(d.w * w, 1), "h": round(d.h * h, 1), "score": round(d.confidence, 4), "kind": kind, **kw}


def arm1(VU, video: Path, w: int, h: int, n: int, cfg: dict) -> tuple[list, dict]:
    thr = {"in_play_ball": cfg["ball_thr"], "ball": cfg["ball_thr"], "other": cfg["other_thr"]}
    rows, runs, f = {}, [], 0
    t0 = time.time()
    scan_total = 0
    while f < n:
        # --- (re-)detect: the autodetect button on frame f, then a few forward steps if the detector sees no ball
        obj = None
        s = f
        while s < min(n, f + cfg["max_scan"] + 1):
            balls = detect_balls(VU, video, s, w, h)
            scan_total += 1
            if balls:
                obj, f = balls[0], s
                break
            s += 1
        if obj is None:
            runs.append({"start": f, "end": None, "stop": "no_ball_found", "scan_to": s})
            break
        bg = VU.BackgroundLabeller()
        bg.submit(video, [obj], None, 0.25, f, 512, dict(thr), None)
        while bg.running or (bg._thread is not None and bg._thread.is_alive()):
            time.sleep(0.05)
        fds = frames_of(bg.frames, f, VU)
        last = max(fds) if fds else f
        for i, fd in fds.items():
            r = row(fd, VU, w, h, "seed" if i == f else "track", run=len(runs))
            if r:
                rows[i] = r
        stop = bg.anomaly_reason
        runs.append({"start": f, "end": last, "seed": [round(v, 1) for v in obj.bbox_xyxy_abs], "stop": stop, "anomaly_frame": bg.anomaly_frame, "error": str(bg.error) if bg.error else None})
        if bg.error:
            break
        if stop is None:        # ran to the end of the clip
            break
        f = bg.anomaly_frame if bg.anomaly_frame is not None else last
        if f is None or f >= n - 1:
            break
        # the stop frame itself is re-detected; its own tracked (dubious) box is kept unless the detector replaces it
        rows.pop(f, None)
    return [{"frame": i, **r} for i, r in sorted(rows.items())], {"runs": runs, "sec": round(time.time() - t0, 1), "detect_calls": scan_total}


def reverse_video(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", "reverse", "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p",
                    "-an", "-g", "1", str(tmp)], check=True)
    tmp.replace(dst)


def arm2(VU, video: Path, rev: Path, w: int, h: int, n: int, seeds: list[dict], cfg: dict) -> tuple[list, dict]:
    sd = sorted(seeds, key=lambda s: s["frame"])
    tracks = []
    t0 = time.time()
    for i, s in enumerate(sd):
        f0 = s["frame"]
        prev_t = sd[i - 1]["frame"] if i > 0 else None
        next_t = sd[i + 1]["frame"] if i + 1 < len(sd) else None
        box = tuple(float(v) for v in s["box"])
        obj = VU.LabelledObject(label="in_play_ball", bbox_xyxy_abs=box)
        for d in ("fwd", "bwd"):
            lim = (min(n - 1, f0 + cfg["reach"], (next_t + cfg["overlap"]) if next_t is not None else 10**9) if d == "fwd" else
                   max(0, f0 - cfg["reach"], (prev_t - cfg["overlap"]) if prev_t is not None else -10**9))
            if (d == "fwd" and lim <= f0) or (d == "bwd" and lim >= f0):
                continue
            vid, start = (video, f0) if d == "fwd" else (rev, n - 1 - f0)
            lab = VU.VitTrackVideoLabeller(vid, [obj])
            frames, prev, stops = {}, None, []
            span = abs(lim - f0)
            for fd in lab.iter_frames_from(start_frame=start):
                k = VU._frame_index_from_uri(fd, start)
                g = k if d == "fwd" else n - 1 - k
                off = abs(g - f0)
                if off > span:
                    break
                r = row(fd, VU, w, h, "seed" if off == 0 else "track")
                if r is None:
                    continue
                if off > 0 and prev is not None:
                    reason = VU._track_anomaly_reason(prev, fd)
                    low = fd.detections[0].confidence < cfg["ball_thr"]
                    r["would_stop"] = "anomaly" if reason else ("low_conf" if low else None)
                frames[g] = r
                prev = fd
            tracks.append({"seed": s["id"], "dir": d, "t": f0, "source": s.get("source", ""), "frames": frames})
    return tracks, {"sec": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-src", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--reach", type=int, default=250)
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--clips", default="")
    ap.add_argument("--engine", default="")
    ap.add_argument("--score-gate", type=float, default=1.0)
    ap.add_argument("--max-seeds", type=int, default=0)
    a = ap.parse_args()
    cfg = {"arm": 2, "ball_thr": 0.5, "other_thr": 0.5, "max_scan": 25, "reach": a.reach, "overlap": a.overlap, **json.loads(a.config)}
    VU, _U = setup(a.main_src)
    from common import video_info
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds_by_clip: dict = {}
    for line in Path(a.seeds).read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            seeds_by_clip.setdefault(s["clip"], []).append(s)
    only = set(a.clips.split(",")) if a.clips else None
    for clip in sorted(seeds_by_clip if cfg["arm"] == 2 else (only or seeds_by_clip)):
        if only and clip not in only:
            continue
        final = out / f"{clip}.labels.jsonl"
        if final.exists():
            print(f"{clip}: done", flush=True)
            continue
        video = Path(a.videos) / f"{clip}.mp4"
        if not video.exists():
            print(f"{clip}: video missing", flush=True)
            continue
        vi = video_info(str(video))
        w, h = vi["w"], vi["h"]
        n = vi["n"] or int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
                                          capture_output=True, text=True).stdout.strip().split(",")[0])
        t0 = time.time()
        if cfg["arm"] == 1:
            rows, info = arm1(VU, video, w, h, n, cfg)
            raw = {"arm": 1, "cfg": cfg, "n": n, **info}
        else:
            rev = out / ".rev" / f"{clip}.mp4"
            rev.parent.mkdir(exist_ok=True)
            reverse_video(video, rev)
            tracks, info = arm2(VU, video, rev, w, h, n, seeds_by_clip[clip][:a.max_seeds or None], cfg)
            rows = [{"frame": g, "seed": t["seed"], "dir": t["dir"], "dist": abs(g - t["t"]), "src": t["source"], **r} for t in tracks for g, r in sorted(t["frames"].items())]
            raw = {"arm": 2, "cfg": cfg, "n": n, "tracks": [{k: v for k, v in t.items() if k != "frames"} | {"n": len(t["frames"])} for t in tracks], **info}
            rev.unlink(missing_ok=True)
        with gzip.open(out / f"{clip}.raw.json.gz", "wt") as fh:
            json.dump(raw, fh)
        tmp = final.with_name(final.name + ".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps({"clip": clip, **r}) + "\n")
        os.replace(tmp, final)
        print(f"{clip}: arm {cfg['arm']} {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
