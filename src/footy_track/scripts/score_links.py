"""Score candidate tracklet links with an appearance embedding.

    python -m footy_track.scripts.score_links --clip <stem> [--out proposals/]

Reads tracker output, proposes gap bridges (``identity.linker``), scores each
with cosine similarity between crops either side of the gap, and writes
proposals for review.

**Crops are taken from the frames adjacent to the gap.** Identity signal decays
fast — measured AUC 0.912 at a 4-frame separation, 0.867 at 12, 0.761 by 25 —
so a link is judged on the frames closest to the break, where the evidence is
strongest. Sampling the whole tracklet would average away the signal it depends
on.

Several crops per side are averaged rather than using one: a single crop can be
motion-blurred, occluded or half a player, and the linker's whole value is that
it does not need a human to notice that.

The embedding is ImageNet ResNet50, deliberately unremarkable. It is the model
measured in the decay curve above, so the thresholds match what was observed.
A ReID-trained backbone should be better at the long gaps this tool refuses to
bridge — swapping it in is a one-line change and would justify widening the
window, but only after re-measuring the curve.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from footy_track.identity.linker import (
    DEFAULT_MAX_GAP,
    Candidate,
    candidate_links,
    tracklets_from_rows,
    triage,
)

CROPS_PER_SIDE = 3


def load_rows(path: Path, label: str = "player") -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("tags") or [""])[0] == label and isinstance(r.get("bbox"), dict):
            rows.append(r)
    return rows


def _embedder(device: str):
    import torchvision

    net = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    )
    net.fc = nn.Identity()
    net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(crops: list[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack([c[:, :, ::-1] for c in crops]))
        x = x.permute(0, 3, 1, 2).float() / 255
        x = ((x - mean) / std).to(device)
        out = [net(x[i : i + 32]).cpu() for i in range(0, len(x), 32)]
        return torch.nn.functional.normalize(torch.cat(out), dim=1)

    return embed


def score_clip(
    clip: str,
    tracklets_dir: Path,
    clips_dir: Path,
    *,
    max_gap: int = DEFAULT_MAX_GAP,
    min_frames: int = 25,
    device: str | None = None,
) -> list[Candidate]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(tracklets_dir / f"{clip}.jsonl")
    if not rows:
        return []
    tracklets = tracklets_from_rows(rows, min_frames=min_frames)
    cands = candidate_links(tracklets, max_gap=max_gap)
    if not cands:
        return []

    by_track: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_track[int(r["track_id"])].append(r)
    for rs in by_track.values():
        rs.sort(key=lambda r: int(r["frame_index"]))

    # Frames adjacent to each gap: the end of A, the start of B.
    need: dict[int, list[tuple[int, dict]]] = collections.defaultdict(list)
    sides: dict[tuple[int, int], dict[str, list[int]]] = {}
    for c in cands:
        a_rows = by_track[c.a][-CROPS_PER_SIDE:]
        b_rows = by_track[c.b][:CROPS_PER_SIDE]
        sides[(c.a, c.b)] = {"a": [], "b": []}
        for side, rs in (("a", a_rows), ("b", b_rows)):
            for r in rs:
                need[int(r["frame_index"])].append((int(r["track_id"]), r["bbox"]))
                sides[(c.a, c.b)][side].append(int(r["frame_index"]))

    crops: list[np.ndarray] = []
    index: dict[tuple[int, int], int] = {}
    cap = cv2.VideoCapture(str(clips_dir / f"{clip}.mp4"))
    idx, target = 0, max(need) if need else -1
    while idx <= target:
        ok, img = cap.read()
        if not ok:
            break
        if idx in need:
            H, W = img.shape[:2]
            for tid, b in need[idx]:
                x1, y1 = max(0, int(b["x"] * W)), max(0, int(b["y"] * H))
                x2 = min(W, int((b["x"] + b["w"]) * W))
                y2 = min(H, int((b["y"] + b["h"]) * H))
                if x2 > x1 + 4 and y2 > y1 + 8:
                    index[(tid, idx)] = len(crops)
                    crops.append(cv2.resize(img[y1:y2, x1:x2], (128, 256)))
        idx += 1
    cap.release()
    if not crops:
        return cands

    E = _embedder(device)(crops)

    scored: list[Candidate] = []
    for c in cands:
        ia = [index[(c.a, f)] for f in sides[(c.a, c.b)]["a"] if (c.a, f) in index]
        ib = [index[(c.b, f)] for f in sides[(c.a, c.b)]["b"] if (c.b, f) in index]
        if not ia or not ib:
            scored.append(c)  # unscored -> triage sends it to "ask", never "merge"
            continue
        va = torch.nn.functional.normalize(E[ia].mean(0, keepdim=True), dim=1)
        vb = torch.nn.functional.normalize(E[ib].mean(0, keepdim=True), dim=1)
        scored.append(
            Candidate(a=c.a, b=c.b, gap=c.gap, distance=c.distance,
                      similarity=round(float(va @ vb.T), 4))
        )
    return scored


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tracklets", type=Path,
                    default=Path.home() / "footy/footy_data/tracklets")
    ap.add_argument("--clips", type=Path,
                    default=Path.home() / "footy/footy_data/clips")
    ap.add_argument("--out", type=Path,
                    default=Path.home() / "footy/footy_data/link_proposals")
    ap.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    args = ap.parse_args(argv)

    scored = score_clip(args.clip, args.tracklets, args.clips, max_gap=args.max_gap)
    buckets = triage(scored)
    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / f"{args.clip}.jsonl"
    tmp = dst.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for c in scored:
            fh.write(json.dumps({
                "a": c.a, "b": c.b, "gap": c.gap,
                "distance": round(c.distance, 4),
                "similarity": c.similarity,
                "decision": c.decision,
                "uncertainty": round(c.uncertainty, 4),
            }) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(dst)

    print(f"{args.clip}: {len(scored)} candidates")
    for k in ("merge", "ask", "reject"):
        print(f"  {k:>6}: {len(buckets[k])}")
    print(f"  -> {dst}")
    print("  'ask' is ordered most-uncertain-first; only it needs a human")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
