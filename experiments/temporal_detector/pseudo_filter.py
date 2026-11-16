"""Round-2 filter of the pseudo-label set (`multiframe_pseudo_v1`, stdlib only).

Keep all easy images and all S1 gap fills; keep a low-confidence B1 detection (conf < 0.6) only if conf >= 0.45 AND a B1 node
with conf >= 0.8 of the same track segment lies within +-25 frames. The dataset meta has no segment id, so segments are
rebuilt from the per-frame provenance table (`provenance_all.jsonl.gz`, every B1 row of the clip): rows separated by gaps of
at most 25 frames (the solver's `max_gap`) form one segment; a node is "in the same segment" when it is within 25 frames
(then every row between them has a gap <= 25, so the proxy reduces to: a conf >= 0.8 B1 detection within +-25 frames).
Also drops any frame within +-5 frames of a v11 val/test/heldout image or of a human GT frame of arsenal_example / the
trajectory test clips (zero by construction: those clips are excluded from the pool; checked here explicitly).

    python pseudo_filter.py --pseudo /mnt/storage/footy_data/multiframe_pseudo_v1 --prov provenance_all.jsonl.gz \
        --k2-meta /mnt/storage/footy_data/multiframe_v1/k2/meta.jsonl --out kept.json
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

TEST_CLIPS = {"arsenal_example"} | {f"arsenal_mancity_seg{n}" for n in ("010", "011", "031", "094", "101", "115", "134", "135", "140")}
NEAR = 5
WINDOW = 25
MIN_LOW = 0.45
HIGH = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo", type=Path, required=True)
    ap.add_argument("--prov", type=Path, required=True)
    ap.add_argument("--k2-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    metas = [json.loads(line) for line in open(a.pseudo / "meta.jsonl")]
    # B1 detections with conf >= 0.8, per clip
    high = collections.defaultdict(list)
    with gzip.open(a.prov, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] == "B1_det" and (r["det_conf"] or 0) >= HIGH:
                high[r["clip"]].append(r["t"])
    for v in high.values():
        v.sort()
    # eval frames (v11 valid/test/heldout images) per clip
    eval_frames = collections.defaultdict(set)
    for line in open(a.k2_meta):
        m = json.loads(line)
        if m["split"] in ("valid", "test", "heldout") and m.get("frame") is not None:
            eval_frames[m["clip_id"]].add(m["frame"])

    def near_high(clip, t):
        return any(abs(t - u) <= WINDOW for u in high.get(clip, ()))

    keep, dropped = [], collections.Counter()
    stats = collections.Counter()
    for m in metas:
        p = m["pseudo"]
        cat, clip, t = p["category"], m["clip_id"], m["frame"]
        stats[("total", cat)] += 1
        # eval-set exclusions (explicit check)
        if clip in TEST_CLIPS or clip in eval_frames and any(abs(t - e) <= NEAR for e in eval_frames[clip]):
            dropped["eval_overlap"] += 1
            continue
        if cat == "hard_B1_det_lowconf":
            if p["det_conf"] < MIN_LOW:
                dropped["lowconf<0.45"] += 1
                continue
            if not near_high(clip, t):
                dropped["no_conf>=0.8_node_within_25"] += 1
                continue
        keep.append(m["id"])
        stats[("kept", cat)] += 1
    out = {"kept": keep, "n_total": len(metas), "n_kept": len(keep), "dropped": dict(dropped),
           "by_category": {f"{k}/{c}": v for (k, c), v in sorted(stats.items())},
           "rule": {"low_conf_min": MIN_LOW, "anchor_conf": HIGH, "window_frames": WINDOW}}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "kept"}, indent=1))


if __name__ == "__main__":
    main()
