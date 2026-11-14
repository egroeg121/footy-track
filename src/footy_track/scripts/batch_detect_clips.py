#!/usr/bin/env python3
"""Run the RT-DETR detector over a folder of clips and write ``machine_labels`` JSONL.

Reproduces the format of ``footy_data/machine_labels`` (run ``rtdetr-l_1920_v11``;
the producer script of that run was never committed): one ``<clip>.jsonl`` per clip,
one row per box, rows in descending confidence, ``bbox`` = normalised top-left x/y +
w/h, ``center`` normalised, ``tags = [class_name, "rtdetr"]``, plus ``_manifest.json``
(run_id, model_id, model_sha256_16, checkpoint, imgsz, conf_threshold, provenance_tier,
class_names, per-clip frames/detections/seconds, totals).

Held-out matches go to their OWN ``--out`` directory and manifest, never into
``machine_labels/``, so nothing that globs that folder (candidate builders, Ball Check,
training export) can pick them up. ``--heldout`` stamps the manifest
``"heldout": true, "do_not_train": true``.

Idempotent and crash-safe: a clip whose ``<out>/<clip>.jsonl`` exists is skipped (rows
are written to ``.tmp`` and renamed); the manifest is rewritten after every clip and
merges the ``clips`` of an earlier partial run with the same run_id file.

Usage:
  uv run python -m footy_track.scripts.batch_detect_clips CLIPS_DIR --out DIR \
      --weights best.pt [--imgsz 1280] [--conf 0.1] [--glob 'arsenal_norwich_seg*.mp4'] \
      [--ball-out DIR2] [--heldout] [--device 0] [--batch 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

BALL_TAGS = {"ball", "in_play_ball", "out_of_play_ball"}
MODEL_ID = "rtdetr-l_1920_v11"


def sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def detect_clip(model, clip: Path, *, imgsz: int, conf: float, device, half: bool,
                batch: int, run_id: str, model_id: str) -> tuple[list[str], int, int]:
    """Return (jsonl lines, frames, detections) for one clip."""
    names = model.names
    lines: list[str] = []
    frames = 0
    stream = model.predict(source=str(clip), stream=True, imgsz=imgsz, conf=conf,
                           device=device, half=half, batch=batch, verbose=False)
    for idx, res in enumerate(stream):
        frames = idx + 1
        b = res.boxes
        if b is None or len(b) == 0:
            continue
        cls = b.cls.int().tolist()
        cf = b.conf.tolist()
        xywhn = b.xywhn.tolist()  # centre x/y, w/h, normalised
        for c, s, (cx, cy, w, h) in zip(cls, cf, xywhn, strict=True):
            lines.append(json.dumps({
                "frame_index": idx,
                "bbox": {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h},
                "center": {"x": cx, "y": cy},
                "tags": [names[c], "rtdetr"],
                "confidence": round(float(s), 4),
                "model_id": model_id,
                "run_id": run_id,
                "reviewed": False,
            }))
    return lines, frames, len(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clips_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: cuda if available)")
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--ball-out", type=Path, default=None,
                    help="also write ball-class-only rows here (ball_candidates format)")
    ap.add_argument("--heldout", action="store_true",
                    help="stamp the manifest heldout/do_not_train")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    args = ap.parse_args(argv)

    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    half = (not args.no_half) and device != "cpu"
    clips = sorted(args.clips_dir.glob(args.glob))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print(f"no clips match {args.glob} in {args.clips_dir}")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ball_out:
        args.ball_out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out / "_manifest.json"
    prev = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    model = YOLO(str(args.weights))
    if prev:
        run_id, started, done = prev["run_id"], prev["started"], dict(prev["clips"])
    else:
        started = datetime.now().isoformat(timespec="seconds")
        run_id = f"{args.model_id}__{datetime.now():%Y%m%d-%H%M%S}"
        done = {}
    manifest = {
        "run_id": run_id, "model_id": args.model_id,
        "model_sha256_16": sha256_16(args.weights), "checkpoint": str(args.weights),
        "imgsz": args.imgsz, "conf_threshold": args.conf, "provenance_tier": 3,
        "reviewed": False,
        "class_names": {str(k): v for k, v in model.names.items()},
        "started": started,
        "device": str(device), "half": half,
        **({"heldout": True, "do_not_train": True} if args.heldout else {}),
    }
    if torch.cuda.is_available():
        manifest["gpu"] = torch.cuda.get_device_name(0)

    def write_manifest(finished: bool) -> None:
        m = {**manifest, "clips": dict(sorted(done.items()))}
        if finished:
            m["finished"] = datetime.now().isoformat(timespec="seconds")
        m["total_frames"] = sum(c["frames"] for c in done.values())
        m["total_detections"] = sum(c["detections"] for c in done.values())
        _atomic_write(manifest_path, json.dumps(m, indent=2) + "\n")

    for i, clip in enumerate(clips):
        out = args.out / f"{clip.stem}.jsonl"
        if out.exists() and clip.stem in done:
            print(f"[{i+1}/{len(clips)}] {clip.stem}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        lines, frames, n = detect_clip(model, clip, imgsz=args.imgsz, conf=args.conf,
                                       device=device, half=half, batch=args.batch,
                                       run_id=run_id, model_id=args.model_id)
        secs = round(time.time() - t0, 1)
        _atomic_write(out, "".join(line + "\n" for line in lines))
        if args.ball_out:
            ball = [ln for ln in lines if set(json.loads(ln)["tags"]) & BALL_TAGS]
            _atomic_write(args.ball_out / f"{clip.stem}.jsonl", "".join(x + "\n" for x in ball))
        done[clip.stem] = {"frames": frames, "detections": n, "seconds": secs}
        write_manifest(False)
        print(f"[{i+1}/{len(clips)}] {clip.stem}: {frames} frames, {n} boxes, {secs}s "
              f"({frames / max(secs, 1e-9):.1f} fps)", flush=True)
    write_manifest(True)
    print(f"done: {len(done)} clips -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
