"""Build the judged-card table (George's Ball Check verdicts) and the dev/test split.

Read-only on every marks dir. Writes only to /mnt/storage/exp/opus_ballcheck/.
Card key: (set, clip, frame_index, box_index); the latest verdict wins, a
``verdict: null`` tombstone retracts (same rule as labeller.ball_check._read_verdicts).
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import random
from pathlib import Path

ROOT = Path("/mnt/storage/footy_data")
OUT = Path("/mnt/storage/exp/opus_ballcheck")
SETS = {
    # name: (marks_dir, list file or None, source filter, video dirs)
    "sidecar": ("ball_gt_marks", None, "sidecar",
                ["eval_clips_kf10", "ballcheck_clips_kf10"]),
    "v1": ("ball_gt_marks_nottinghamforest",
           "nottinghamforest_label_queue/ball_check_list_v1.json", "list",
           ["arsenal_nottinghamforest/split_video_broadcast_frames_kf10"]),
    "v2": ("ball_gt_marks_nottinghamforest_disagree",
           "nottinghamforest_label_queue/ball_check_list_v2_disagree.json", "list",
           ["arsenal_nottinghamforest/split_video_broadcast_frames_kf10"]),
}
# "corrected" = George dragged the box onto the ball: ball present, box wrong.
GEORGE_MAP = {"ball": "ball", "not_ball": "not_ball", "box_off": "box_off",
              "corrected": "box_off", "unsure": "unsure"}


def read_verdicts(marks: Path, source: str) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for p in sorted(glob.glob(str(marks / "ball_checks" / "*.jsonl"))):
        for raw in open(p):
            raw = raw.strip()
            if not raw:
                continue
            r = json.loads(raw)
            src = r.get("source") or "sidecar"
            if src != source:
                continue
            k = (r["clip"], int(r["frame_index"]), int(r["box_index"]))
            if r.get("verdict") is None:
                out.pop(k, None)
            else:
                out[k] = r
    return out


def find_video(dirs: list[str], clip: str) -> str | None:
    for d in dirs:
        for ext in (".mp4", ".mov", ".avi", ".mkv"):
            p = ROOT / d / f"{clip}{ext}"
            if p.exists():
                return str(p)
    return None


def main() -> None:
    cards = []
    for name, (marks, lst, src, vdirs) in SETS.items():
        verd = read_verdicts(ROOT / marks, src)
        items = None
        if lst:
            d = json.load(open(ROOT / lst))
            items = d["items"] if isinstance(d, dict) else d
        for (clip, fi, bi), r in sorted(verd.items()):
            other = None
            stratum = name
            if items is not None:
                it = items[bi] if 0 <= bi < len(items) else None
                if it is None or it["clip"] != clip or int(it["frame_index"]) != fi:
                    print("LIST MISMATCH", name, clip, fi, bi)
                    continue
                other = it.get("other_bbox")
                stratum = f"{name}:{it.get('set') or it.get('stratum')}"
            video = find_video(vdirs, clip)
            if video is None:
                print("NO VIDEO", name, clip)
                continue
            cid = hashlib.sha1(f"{name}:{clip}:{fi}:{bi}".encode()).hexdigest()[:10]
            cards.append({
                "id": cid, "set": name, "stratum": stratum, "clip": clip,
                "frame_index": fi, "box_index": bi, "bbox": r["bbox"],
                "other_bbox": other, "video": video,
                "george_raw": r["verdict"], "george": GEORGE_MAP[r["verdict"]],
                "conf": r.get("confidence"),
            })
    # Split: cards in the frozen first snapshot (cards_v0_split.json) keep their assignment;
    # cards George judged later (the v2 list is still being labelled) are assigned by id hash,
    # 30% dev / 70% test, so a rebuild never moves an already-rendered or already-scored card.
    frozen = {}
    p0 = OUT / "cards_v0_split.json"
    if p0.exists():
        frozen = {c["id"]: c["split"] for c in json.load(open(p0))}
    for c in cards:
        c["split"] = frozen.get(c["id"]) or (
            "dev" if int(hashlib.sha1(("split" + c["id"]).encode()).hexdigest(), 16) % 100 < 30 else "test")
    # dev-iter: frozen subset used for prompt iteration (iter_ids.json, written once: every dev
    # non-ball + about half of the dev balls, as of the first snapshot)
    pit = OUT / "iter_ids.json"
    iter_ids = set(json.load(open(pit))) if pit.exists() else set()
    for c in cards:
        c["iter"] = c["id"] in iter_ids
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(cards, open(OUT / "cards.json", "w"), indent=1)
    cnt = collections.Counter((c["set"], c["split"], c["george"]) for c in cards)
    for k in sorted(cnt):
        print(k, cnt[k])
    print("total", len(cards), collections.Counter(c["split"] for c in cards))
    print("two-box", sum(1 for c in cards if c["other_bbox"]))
    print(collections.Counter(c["stratum"] for c in cards))


if __name__ == "__main__":
    main()
