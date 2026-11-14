"""Coverage report: how many v11 images / ball instances map to source video with
+-k neighbour frames, per split and per source family.

    uv run python -m footy_track.scripts.multiframe_mapping_report [--out report.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from footy_track.multiframe.v11 import SPLITS, load_v11

KS = (1, 2, 4)


def verify(items, n: int, seed: int = 0) -> dict:
    """Mean abs grey diff between the v11 JPEG and the decoded frame t (and t+-1).

    A correct mapping gives ~0.3-0.5 (JPEG noise); t+-1 is ~10+ on moving footage.
    """
    import random

    import cv2
    import numpy as np

    from footy_track.multiframe.frames import ClipReader
    from footy_track.multiframe.v11 import V11_DIR

    mapped = [i for i in items if i.ref.has_neighbours(1)]
    random.Random(seed).shuffle(mapped)
    out, by_fam = [], {}
    for it in mapped[:n]:
        rd = ClipReader(it.ref.video, cache=4)
        ref = cv2.imread(str(V11_DIR / it.split / "images" / it.name), cv2.IMREAD_GRAYSCALE)
        errs = {}
        for d in (-1, 0, 1):
            fr = rd.get(it.ref.frame + d)
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            errs[d] = float(np.abs(g.astype(np.float32) - ref.astype(np.float32)).mean())
        rd.close()
        best = min(errs, key=errs.get)
        out.append((it.prefix, errs[0], best))
    fam: dict = {}
    for pre, e0, best in out:
        f = fam.setdefault(pre, {"n": 0, "exact": 0, "max_mad_at_t": 0.0})
        f["n"] += 1
        f["exact"] += int(best == 0 and e0 < 3)
        f["max_mad_at_t"] = max(f["max_mad_at_t"], e0)
    return {"n": len(out), "exact": sum(f["exact"] for f in fam.values()), "by_prefix": fam}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", type=int, default=0,
                    help="pixel-check N random mapped images against the decoded frame")
    args = ap.parse_args(argv)
    items = load_v11()
    rep: dict = {"splits": {}, "unmapped_reasons": {}, "clips_by_split": {}}
    for split in SPLITS:
        its = [i for i in items if i.split == split]
        s = {"images": len(its), "balls": sum(i.n_ball for i in its),
             "images_with_ball": sum(1 for i in its if i.n_ball),
             "mapped_images": sum(i.ref.mapped for i in its),
             "mapped_balls": sum(i.n_ball for i in its if i.ref.mapped)}
        for k in KS:
            ok = [i for i in its if i.ref.has_neighbours(k)]
            s[f"k{k}_images"] = len(ok)
            s[f"k{k}_balls"] = sum(i.n_ball for i in ok)
        rep["splits"][split] = s
        rep["unmapped_reasons"][split] = dict(Counter(i.ref.reason for i in its if not i.ref.mapped))
        rep["clips_by_split"][split] = sorted({i.ref.clip_id for i in its if i.ref.mapped})
    fam = defaultdict(lambda: Counter())
    for i in items:
        f = fam[(i.split, i.prefix)]
        f["images"] += 1
        f["mapped"] += i.ref.mapped
        f["k2"] += i.ref.has_neighbours(2)
        f["balls"] += i.n_ball
    rep["by_prefix"] = {f"{s}/{p}": dict(c) for (s, p), c in sorted(fam.items())}
    seen = Counter((i.prefix, i.idx) for i in items)
    rep["duplicate_source_frames_across_splits"] = sum(1 for v in seen.values() if v > 1)
    tr = set(rep["clips_by_split"]["train"])
    rep["canonical_clips_shared_train_valid"] = sorted(tr & set(rep["clips_by_split"]["valid"]))
    rep["canonical_clips_shared_train_test"] = sorted(tr & set(rep["clips_by_split"]["test"]))
    if args.verify:
        rep["pixel_verification"] = verify(items, args.verify)
    print(json.dumps({k: v for k, v in rep.items() if k != "clips_by_split"}, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
