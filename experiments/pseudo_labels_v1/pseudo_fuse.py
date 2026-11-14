"""S1 fusion over the pool and frame selection for the pseudo-label set (CPU, mini; one process).

Inputs: pool_b1.json / exclusions.json (pseudo_prep.py), SAM2 tracks (machine_tracks.py output dir).
Outputs (in --out):
  provenance_all.jsonl.gz   one row per frame that has a track point (source B1_det / B1_interp / S1_fill)
  selection.jsonl           the selected frames with ball label + machine player/other boxes (conf >= 0.5)
  selection_stats.json      counts by source/category, stride, near-GT counts, ball-only image count

Per-frame provenance fields: det conf, solver p, source, forward/backward agreement in px (S1 fills:
|fwd - bwd| when both passes agree, null for a lone pass), distance to the nearest anchor, anchor conf.

usage: pseudo_fuse.py --prep <pp dir> --tracks <dir> --out <dir> [--seed 0 --stride 5 --easy 800]
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "sam2_bidir_ball"))
import machine_score as MS  # noqa: E402

THR, SPACING = 0.8, 10
V11_CLASS = {"ball": 0, "coach": 1, "in_play_ball": 2, "person": 3, "player": 4, "player_sub": 5, "referee": 6}
BALL_IDS = {0, 2}
BALL_CLASS = 2
W_PX, H_PX = 1920.0, 1080.0
MIN_PLAYER_CONF = 0.5
V11_META = Path("/mnt/storage/footy_data/multiframe_v1/k2/meta.jsonl")


def machine_boxes(stem: str, min_conf: float = MIN_PLAYER_CONF) -> dict[int, list]:
    """frame -> [(cls, cx, cy, w, h, conf)] of non-ball classes at conf >= min_conf (normalised)."""
    out: dict[int, list] = defaultdict(list)
    p = Path("/mnt/storage/footy_data/machine_labels") / f"{stem}.jsonl"
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            c = r.get("confidence", 1.0)
            if c < min_conf:
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
    ap.add_argument("--prep", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--easy", type=int, default=800, help="max easy frames (random sample of conf >= 0.8 nodes)")
    ap.add_argument("--hard-lowconf-max", type=float, default=0.6)
    a = ap.parse_args()
    prep = Path(a.prep)
    pool = json.loads((prep / "pool_b1.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v11 = [json.loads(x) for x in V11_META.read_text().splitlines() if x.strip()]
    v11_train: dict[str, list[int]] = defaultdict(list)
    for r in v11:
        if r["split"] == "train":
            v11_train[r["clip_id"]].append(r["frame"])
    for v in v11_train.values():
        v.sort()
    rng = random.Random(a.seed)
    rows_all, cand = [], {"S1_fill": [], "B1_det_lowconf": [], "easy": []}
    stats = Counter()
    for cid, c in sorted(pool.items()):
        tracks, _ = MS.load_tracks(Path(a.tracks), cid, "small", "crop512")
        nodes = {n["t"]: n for n in c["nodes"]}
        b1 = {int(t): v for t, v in c["b1"].items()}
        A = c["anchors"]
        n_frames, cuts = c["n_frames"], c["cuts"]
        gt = sorted(int(t) for t in c["gt"])
        gap = [f for f in range(n_frames) if f not in b1]
        dec = MS.sam2_decisions({"cuts": cuts}, tracks, A, gap)

        def d_anchor(t):
            if not A:
                return None
            j = bisect.bisect_left(A, t)
            ds = [abs(A[k] - t) for k in (j - 1, j) if 0 <= k < len(A)]
            return min(ds)

        for f in range(n_frames):
            row = None
            if f in nodes:
                n = nodes[f]
                row = {"source": "B1_det", "x": n["xy"][0], "y": n["xy"][1], "w": n["wh"][0], "h": n["wh"][1], "det_conf": n["conf"],
                       "solver_p": n["p"], "agree_px": None, "dist_anchor": d_anchor(f), "anchor_conf": n["conf"] if f in A else None, "is_anchor": f in A}
            elif f in b1:
                row = {"source": "B1_interp", "x": b1[f][0], "y": b1[f][1], "w": None, "h": None, "det_conf": None, "solver_p": None,
                       "agree_px": None, "dist_anchor": d_anchor(f), "anchor_conf": None, "is_anchor": False}
            elif f in dec and not dec[f]["crosses_cut"]:
                r = dec[f]
                anchors_used = [nodes[x] for x in (r["ap"], r["an"]) if x is not None and x in nodes]
                w = sum(n["wh"][0] for n in anchors_used) / len(anchors_used)
                h = sum(n["wh"][1] for n in anchors_used) / len(anchors_used)
                row = {"source": "S1_fill", "x": round(r["pos"][0], 2), "y": round(r["pos"][1], 2), "w": round(w, 1), "h": round(h, 1),
                       "det_conf": None, "solver_p": None, "agree_px": None if r["agree_px"] is None else round(r["agree_px"], 2),
                       "rule": r["rule"], "dist_anchor": r["dist"], "anchor_conf": nodes[r["anchor"]]["conf"] if r["anchor"] in nodes else None,
                       "is_anchor": False}
            elif f in dec:
                stats["S1_fill_rejected_crosses_cut"] += 1
            if row is None:
                continue
            row.update(clip=cid, t=f)
            rows_all.append(row)
            stats[f"rows_{row['source']}"] += 1
            ok_edge = 2 <= f <= n_frames - 3 and not any(abs(f - cu) <= 2 for cu in cuts)
            if not ok_edge:
                stats["excluded_edge_or_near_cut"] += 1
                continue
            row["near_human_gt_5"] = near(gt, f, 5)
            row["near_v11_train_5"] = near(v11_train.get(cid, []), f, 5)
            if row["source"] == "S1_fill":
                cand["S1_fill"].append(row)
            elif row["source"] == "B1_det" and row["det_conf"] < a.hard_lowconf_max:
                cand["B1_det_lowconf"].append(row)
            elif row["source"] == "B1_det" and row["det_conf"] >= 0.8:
                cand["easy"].append(row)
    with gzip.open(out / "provenance_all.jsonl.gz", "wt") as f:
        for r in rows_all:
            f.write(json.dumps(r) + "\n")
    stats["rows_total"] = len(rows_all)
    # ---- selection with a per-clip stride ------------------------------------------------
    taken: dict[str, list[int]] = defaultdict(list)

    def ok(r):
        return all(abs(r["t"] - u) >= a.stride for u in taken[r["clip"]])

    selected = []
    for cat in ("S1_fill", "B1_det_lowconf"):
        pool_c = cand[cat][:]
        rng.shuffle(pool_c)
        n_c = 0
        for r in pool_c:
            if ok(r):
                taken[r["clip"]].append(r["t"])
                r["category"] = "hard_" + cat
                selected.append(r)
                n_c += 1
        stats[f"candidates_{cat}"] = len(cand[cat])
        stats[f"selected_{cat}"] = n_c
    easy_pool = cand["easy"][:]
    rng.shuffle(easy_pool)
    n_e = 0
    for r in easy_pool:
        if n_e >= a.easy:
            break
        if ok(r):
            taken[r["clip"]].append(r["t"])
            r["category"] = "easy"
            selected.append(r)
            n_e += 1
    stats["candidates_easy"] = len(cand["easy"])
    stats["selected_easy"] = n_e
    # ---- machine player / other-class boxes ----------------------------------------------
    by_clip = defaultdict(list)
    for r in selected:
        by_clip[r["clip"]].append(r)
    ball_only = 0
    cls_count = Counter()
    n_other = []
    for cid, rs in by_clip.items():
        mb = machine_boxes(pool[cid]["stem"])
        for r in rs:
            boxes = mb.get(r["t"], [])
            r["other_boxes"] = [[cls, round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6), round(cf, 4)] for cls, cx, cy, bw, bh, cf in boxes]
            if not boxes:
                ball_only += 1
            for cls, *_ in boxes:
                cls_count[cls] += 1
            n_other.append(len(boxes))
    stats["selected_total"] = len(selected)
    stats["images_without_any_other_class_box_conf>=0.5 (ball-only)"] = ball_only
    stats["other_class_boxes_by_v11_class"] = dict(sorted(cls_count.items()))
    stats["mean_other_boxes_per_image"] = round(sum(n_other) / max(len(n_other), 1), 2)
    stats["selected_near_human_gt_5"] = sum(1 for r in selected if r["near_human_gt_5"])
    stats["selected_near_v11_train_5"] = sum(1 for r in selected if r["near_v11_train_5"])
    stats["selected_by_clip"] = dict(Counter(r["clip"] for r in selected))
    selected.sort(key=lambda r: (r["clip"], r["t"]))
    stems = {cid: {"stem": pool[cid]["stem"], "video": f"/mnt/storage/footy_data/clips/{pool[cid]['stem']}.mp4"} for cid in by_clip}
    (out / "stems.json").write_text(json.dumps(stems))
    with (out / "selection.jsonl").open("w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")
    (out / "selection_stats.json").write_text(json.dumps(stats, indent=1, default=int))
    print(json.dumps({k: v for k, v in stats.items() if k != "selected_by_clip"}, indent=1, default=int))


if __name__ == "__main__":
    main()
