"""Cheap appearance descriptors for ball candidates, for diversity sampling.

The point is not recognition, it is *spread*: a reviewer should not be shown
the same ball twenty times, and the cards they do see should cover the range
of situations (grass, crowd, boot, line marking) rather than clustering
wherever the detector is busiest.

A learned embedding is the wrong tool here. The Mac mini has no GPU, there
are 356,812 candidates, and the distinctions that matter at 11 px are coarse
— brightness, contrast against the background, colour. A 72-dim hand-crafted
descriptor (8x8 grey patch + 8-bin hue histogram) captures those, costs
nothing but the video decode, and needs no weights.

Decode dominates, so frames are read sequentially per clip and only
retrieved when they carry a candidate (``grab`` skips the rest).

    uv run python -m footy_track.scripts.embed_ball_candidates \
        --candidates-dir /mnt/storage/footy_data/ball_candidates \
        --clips-dir /mnt/storage/footy_data/ballcheck_clips \
        --out-dir /mnt/storage/footy_data/ball_embeddings
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

#: Context window as a fraction of frame width — matches the review crop, so
#: the descriptor describes what the reviewer is actually shown.
WINDOW_FRAC = 0.125
PATCH = 8
HUE_BINS = 8
DIM = PATCH * PATCH + HUE_BINS

BALL_TAGS = {"ball", "in_play_ball", "out_of_play_ball"}


def _candidates_by_frame(path: Path) -> dict[int, list[tuple[int, dict]]]:
    out: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    per_frame: dict[int, int] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        tags = rec.get("tags") or []
        if not any(t in BALL_TAGS for t in tags) or not rec.get("bbox"):
            continue
        frame_index = int(rec["frame_index"])
        idx = per_frame.get(frame_index, 0)
        per_frame[frame_index] = idx + 1
        out[frame_index].append((idx, rec["bbox"]))
    return out


def describe(frame: np.ndarray, bbox: dict) -> np.ndarray:
    """72-dim descriptor of the crop window around one box."""
    h_px, w_px = frame.shape[:2]
    cx = (bbox["x"] + bbox["w"] / 2) * w_px
    cy = (bbox["y"] + bbox["h"] / 2) * h_px
    side = max(w_px * WINDOW_FRAC, max(bbox["w"] * w_px, bbox["h"] * h_px) * 1.6)
    half = side / 2
    x1, y1 = max(0, int(cx - half)), max(0, int(cy - half))
    x2, y2 = min(w_px, int(cx + half)), min(h_px, int(cy + half))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(DIM, dtype=np.float32)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    patch = cv2.resize(grey, (PATCH, PATCH), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    # Centre and scale: absolute brightness says more about the broadcast than
    # about the object, so the descriptor keys on structure and contrast.
    patch -= patch.mean()
    norm = float(np.linalg.norm(patch))
    if norm > 1e-6:
        patch /= norm
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = cv2.calcHist([hsv], [0], None, [HUE_BINS], [0, 180]).ravel()
    total = float(hue.sum())
    if total > 0:
        hue /= total
    return np.concatenate([patch.ravel(), hue.astype(np.float32)])


def embed_clip(cand_path: Path, video: Path, out_path: Path) -> int:
    by_frame = _candidates_by_frame(cand_path)
    if not by_frame:
        return 0
    last = max(by_frame)
    cap = cv2.VideoCapture(str(video))
    frames_idx: list[int] = []
    cand_idx: list[int] = []
    vecs: list[np.ndarray] = []
    try:
        pos = 0
        while pos <= last:
            if pos not in by_frame:
                if not cap.grab():  # skip without decoding — decode dominates
                    break
                pos += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            for idx, bbox in by_frame[pos]:
                frames_idx.append(pos)
                cand_idx.append(idx)
                vecs.append(describe(frame, bbox))
            pos += 1
    finally:
        cap.release()
    if not vecs:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frame_index=np.asarray(frames_idx, dtype=np.int32),
        cand_index=np.asarray(cand_idx, dtype=np.int16),
        vec=np.asarray(vecs, dtype=np.float16),
    )
    return len(vecs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="embed_ball_candidates")
    ap.add_argument("--candidates-dir", type=Path, required=True)
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="only N clips (timing)")
    args = ap.parse_args(argv)

    paths = sorted(args.candidates_dir.glob("*.jsonl"))
    if args.limit:
        paths = paths[: args.limit]
    total = 0
    started = time.time()
    for i, cand_path in enumerate(paths, 1):
        out_path = args.out_dir / f"{cand_path.stem}.npz"
        if out_path.exists():
            continue
        video = None
        for ext in (".mp4", ".mov", ".avi", ".mkv"):
            candidate = args.clips_dir / f"{cand_path.stem}{ext}"
            if candidate.exists():
                video = candidate
                break
        if video is None:
            print(f"[{i}/{len(paths)}] {cand_path.stem}: no video, skipped", flush=True)
            continue
        n = embed_clip(cand_path, video, out_path)
        total += n
        rate = total / max(time.time() - started, 1e-6)
        print(
            f"[{i}/{len(paths)}] {cand_path.stem}: {n} vecs "
            f"({total} total, {rate:.0f}/s)",
            flush=True,
        )
    print(f"DONE {total} descriptors in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
