"""Round 3 step 4 (mini, one process): S1 fusion + selection + round-2 filter for one target match.

Adapts experiments/pseudo_labels_v1/pseudo_fuse.py (Man City) to a labels directory, 25-fps-subsampled frame indices, a list of
excluded frame windows and the round-2 filter:
  * S1 fills: all kept;  easy B1 nodes (conf >= 0.8): random, capped;
  * low-conf B1 nodes (conf < 0.6): kept only if conf >= 0.45 AND a conf >= 0.8 B1 node lies within +-25 (subsampled) frames;
  * per-clip stride >= 5 subsampled frames; frames within 2 of a cut / clip end are not eligible;
  * excluded windows (Nottingham Forest label-queue windows +-25 frames; held-out / v11 val+test clips are simply absent from the pool);
  * total capped at --cap.
Selected rows carry t (subsampled index) and t_orig = t * stride (frame of the source video that is trained on).

    python r3_fuse.py --match nottinghamforest --prep <dir> --tracks <dir> --labels-dir <dir> --stride-orig 1 --out <dir> [--exclude-windows w.json]
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SAM2 = Path("/home/george/footy-track/.claude/worktrees/agent-a61a56024c56ea1ff/experiments/sam2_bidir_ball")
sys.path.insert(0, str(SAM2))
import machine_score as MS  # noqa: E402

V11_CLASS = {"ball": 0, "coach": 1, "in_play_ball": 2, "person": 3, "player": 4, "player_sub": 5, "referee": 6}
BALL_IDS = {0, 2}
MIN_PLAYER_CONF = 0.5
LOWCONF_MIN, WINDOW, HIGH = 0.45, 25, 0.8
V11_META = Path("/mnt/storage/footy_data/multiframe_v1/k2/meta.jsonl")


def machine_boxes(path: Path) -> dict[int, list]:
    out: dict[int, list] = defaultdict(list)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            c = r.get("confidence", 1.0)
            if c < MIN_PLAYER_CONF:
                continue
            cls = V11_CLASS.get(r["tags"][0])
            if cls is None or cls in BALL_IDS:
                continue
            b = r["bbox"]
            out[r["frame_index"]].append((cls, b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, b["w"], b["h"], c))
    return out


def near(sorted_frames: list[int], t: int, k: int) -> bool:
    i = bisect.bisect_left(sorted_frames, t - k)
    return i < len(sorted_frames) and sorted_frames[i] <= t + k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--stride-orig", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude-windows", type=Path, default=None, help='json {clip: [[lo, hi], ...]} in subsampled frame indices (already padded)')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--easy-max", type=int, default=800)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--no-lowconf", action="store_true", help="drop all low-confidence B1 detections (used when the off-Man-City audit says they are wrong)")
    a = ap.parse_args()
    pool = json.loads((a.prep / "pool_b1.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    excl = json.loads(a.exclude_windows.read_text()) if a.exclude_windows else {}
    v11_train: dict[str, list[int]] = defaultdict(list)
    for line in V11_META.read_text().splitlines():
        r = json.loads(line)
        if r["split"] == "train":
            v11_train[r["clip_id"]].append(r["frame"] // a.stride_orig)
    for v in v11_train.values():
        v.sort()
    rng = random.Random(a.seed)
    cand = {"S1_fill": [], "B1_det_lowconf": [], "easy": []}
    stats = Counter()
    rows_all = []
    for cid, c in sorted(pool.items()):
        tracks, _ = MS.load_tracks(a.tracks, cid, "small", "crop512")
        nodes = {n["t"]: n for n in c["nodes"]}
        b1 = {int(t): v for t, v in c["b1"].items()}
        A = c["anchors"]
        n_frames, cuts = c["n_frames"], c["cuts"]
        high = sorted(t for t, n in nodes.items() if n["conf"] >= HIGH)
        gap = [f for f in range(n_frames) if f not in b1]
        dec = MS.sam2_decisions({"cuts": cuts}, tracks, A, gap)

        def d_anchor(t, A=A):
            if not A:
                return None
            j = bisect.bisect_left(A, t)
            return min(abs(A[k] - t) for k in (j - 1, j) if 0 <= k < len(A))

        for f in range(n_frames):
            row = None
            if f in nodes:
                n = nodes[f]
                row = {"source": "B1_det", "x": n["xy"][0], "y": n["xy"][1], "w": n["wh"][0], "h": n["wh"][1], "det_conf": n["conf"], "solver_p": n["p"],
                       "agree_px": None, "dist_anchor": d_anchor(f), "anchor_conf": n["conf"] if f in A else None, "is_anchor": f in A}
            elif f in b1:
                continue
            elif f in dec and not dec[f]["crosses_cut"]:
                r = dec[f]
                used = [nodes[x] for x in (r["ap"], r["an"]) if x is not None and x in nodes]
                row = {"source": "S1_fill", "x": round(r["pos"][0], 2), "y": round(r["pos"][1], 2), "w": round(sum(n["wh"][0] for n in used) / len(used), 1),
                       "h": round(sum(n["wh"][1] for n in used) / len(used), 1), "det_conf": None, "solver_p": None,
                       "agree_px": None if r["agree_px"] is None else round(r["agree_px"], 2), "rule": r["rule"], "dist_anchor": r["dist"],
                       "anchor_conf": nodes[r["anchor"]]["conf"] if r["anchor"] in nodes else None, "is_anchor": False}
            elif f in dec:
                stats["S1_fill_rejected_crosses_cut"] += 1
                continue
            else:
                continue
            row.update(clip=cid, t=f, t_orig=f * a.stride_orig)
            rows_all.append(row)
            stats[f"rows_{row['source']}"] += 1
            if not (2 <= f <= n_frames - 3 and not any(abs(f - cu) <= 2 for cu in cuts)):
                stats["excluded_edge_or_near_cut"] += 1
                continue
            if any(lo <= f <= hi for lo, hi in excl.get(cid, [])):
                stats["excluded_label_queue_window"] += 1
                continue
            row["near_v11_train_5"] = near(v11_train.get(cid, []), f, 5)
            if row["source"] == "S1_fill":
                cand["S1_fill"].append(row)
            elif row["det_conf"] < 0.6:
                if a.no_lowconf:
                    stats["lowconf_dropped_no_lowconf_flag"] += 1
                elif row["det_conf"] < LOWCONF_MIN:
                    stats["lowconf_dropped_<0.45"] += 1
                elif not near(high, f, WINDOW):
                    stats["lowconf_dropped_no_high_node_within_25"] += 1
                else:
                    cand["B1_det_lowconf"].append(row)
            elif row["det_conf"] >= 0.8:
                cand["easy"].append(row)
    with gzip.open(a.out / "provenance_all.jsonl.gz", "wt") as f:
        for r in rows_all:
            f.write(json.dumps(r) + "\n")
    taken: dict[str, list[int]] = defaultdict(list)

    def ok(r):
        return all(abs(r["t"] - u) >= a.stride for u in taken[r["clip"]])

    selected = []
    for cat, limit in (("S1_fill", a.cap), ("B1_det_lowconf", a.cap), ("easy", a.easy_max)):
        pool_c = cand[cat][:]
        rng.shuffle(pool_c)
        n_c = 0
        for r in pool_c:
            if n_c >= limit or len(selected) >= a.cap:
                break
            if ok(r):
                taken[r["clip"]].append(r["t"])
                r["category"] = {"S1_fill": "hard_S1_fill", "B1_det_lowconf": "hard_B1_det_lowconf", "easy": "easy"}[cat]
                selected.append(r)
                n_c += 1
        stats[f"candidates_{cat}"] = len(cand[cat])
        stats[f"selected_{cat}"] = n_c
    by_clip = defaultdict(list)
    for r in selected:
        by_clip[r["clip"]].append(r)
    cls_count, n_other, ball_only = Counter(), [], 0
    for cid, rs in by_clip.items():
        mb = machine_boxes(a.labels_dir / f"{pool[cid]['stem']}.jsonl")
        for r in rs:
            boxes = mb.get(r["t"], [])
            r["other_boxes"] = [[cls, round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6), round(cf, 4)] for cls, cx, cy, bw, bh, cf in boxes]
            ball_only += not boxes
            n_other.append(len(boxes))
            for cls, *_ in boxes:
                cls_count[cls] += 1
    stats = dict(stats)
    stats.update({"selected_total": len(selected), "ball_only_images": ball_only, "other_class_boxes_by_v11_class": dict(sorted(cls_count.items())),
                  "mean_other_boxes_per_image": round(sum(n_other) / max(len(n_other), 1), 2), "clips_with_selection": len(by_clip),
                  "selected_near_v11_train_5": sum(r["near_v11_train_5"] for r in selected), "pool_clips": len(pool),
                  "pool_frames": sum(c["n_frames"] for c in pool.values()), "match": a.match, "stride_orig": a.stride_orig})
    selected.sort(key=lambda r: (r["clip"], r["t"]))
    with (a.out / "selection.jsonl").open("w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")
    (a.out / "selection_stats.json").write_text(json.dumps(stats, indent=1, default=int))
    print(json.dumps({k: v for k, v in stats.items() if k != "other_class_boxes_by_v11_class"}, indent=1, default=int))


if __name__ == "__main__":
    main()
