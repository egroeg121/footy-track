"""Union test: do 5ch-YOLO candidates ADD to the production RT-DETR-L candidates (rather than replace them)? (CPU, numpy only)

Per frame: RT-DETR candidates (conf >= 0.1, harness calibration knots -> p) and 5ch YOLO candidates (conf >= 0.1, isotonic map
fitted on v11 *valid* minus arsenal_example -> p) are merged (cross-source duplicates within 6 px collapse to one candidate at the
higher-p position with p = max, optionally + a logit bonus when both sources agree). The merged p is written back as a
"pseudo-confidence" through the inverse of the harness knots, so the harness' B0 (tau 0.4) and B1 (camera-on parameters) run with
UNCHANGED parameters and RT-DETR alone is reproduced exactly (cal^-1(cal(conf)) = conf).

    uv run --no-project --python /usr/bin/python3 --with numpy python union_eval.py --harness-src <wt>/src \
        --clips-out <dir> --runs <dir> --pipeline-doc docs/experiments/temporal_pipeline_eval
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_eval import (B0_TAU, CAMERA_ON, HIT_PX, N_BOOT, cands_from_npz, fit_isotonic, frame_records,  # noqa: E402
                           load_harness, with_cands)

MERGE_PX = 6.0
BONUS_LOGIT = 0.5
SECTION = "## Union test: do the 5ch candidates add to RT-DETR-L?"


_CACHE: dict = {}


def cands_cached(z, key):
    k = (id(z), key)
    if k not in _CACHE:
        _CACHE[k] = cands_from_npz(z, key)
    return _CACHE[k]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def val_isotonic(preds_json: Path, k2_meta: Path):
    clip = {}
    for line in open(k2_meta):
        r = json.loads(line)
        if r["split"] == "valid":
            clip[r["id"]] = r["clip_id"]
    d = json.loads(preds_json.read_text())
    conf, corr, n_img = [], [], 0
    for im in d["images"]:
        if im["split"] != "val" or not im["gt"] or clip.get(im["id"]) == "arsenal_example":
            continue
        n_img += 1
        g = np.array([[b[0], b[1]] for b in im["gt"]])
        for x, y, _, _, c in im["dets"]:
            if c >= 0.1:
                conf.append(c)
                corr.append(float(np.min(np.hypot(g[:, 0] - x, g[:, 1] - y)) <= HIT_PX))
    conf, corr = np.array(conf), np.array(corr)
    f, xs, ps = fit_isotonic(conf, corr)
    return f, {"n_images": n_img, "n_candidates": int(len(conf)), "n_correct": int(corr.sum()), "knots_conf": xs.tolist(), "knots_p": ps.tolist()}


def merge_frame(xy_a, p_a, wh_a, xy_b, p_b, wh_b, bonus: float):
    """a = RT-DETR, b = YOLO. Greedy nearest matching within MERGE_PX; returns (xy, p, wh, n_agree)."""
    out_xy, out_p, out_wh = [], [], []
    ub = np.zeros(len(p_b), bool)
    ua = np.zeros(len(p_a), bool)
    agree = 0
    if len(p_a) and len(p_b):
        d = np.hypot(xy_a[:, None, 0] - xy_b[None, :, 0], xy_a[:, None, 1] - xy_b[None, :, 1])
        pairs = sorted((d[i, j], i, j) for i in range(len(p_a)) for j in range(len(p_b)) if d[i, j] <= MERGE_PX)
        for _, i, j in pairs:
            if ua[i] or ub[j]:
                continue
            ua[i] = ub[j] = True
            hi_a = p_a[i] >= p_b[j]
            p = max(p_a[i], p_b[j])
            if bonus:
                p = float(sigmoid(logit(p) + bonus))
            out_xy.append(xy_a[i] if hi_a else xy_b[j])
            out_wh.append(wh_a[i] if hi_a else wh_b[j])
            out_p.append(p)
            agree += 1
    for i in np.where(~ua)[0]:
        out_xy.append(xy_a[i]), out_p.append(p_a[i]), out_wh.append(wh_a[i])
    for j in np.where(~ub)[0]:
        out_xy.append(xy_b[j]), out_p.append(p_b[j]), out_wh.append(wh_b[j])
    return (np.array(out_xy).reshape(-1, 2), np.array(out_p), np.array(out_wh).reshape(-1, 2), agree)


def boot(arr, kinds, clips, which, idx_fn):
    gt = kinds == "gt"
    x, k = arr[idx_fn], gt[idx_fn]
    if which == "recall10":
        return x[k, 0].mean() if k.any() else np.nan
    if which == "ceiling10":
        return x[k, 2].mean() if k.any() else np.nan
    tp, fp = x[:, 0].sum(), x[:, 1].sum()
    return tp / (tp + fp) if tp + fp else np.nan


def paired(arr_a, arr_b, kinds, clips, rng):
    n = len(kinds)
    res = {}
    uniq = np.unique(clips)
    members = [np.where(clips == c)[0] for c in uniq]
    for which in ("recall10", "precision10", "ceiling10"):
        d0 = boot(arr_b, kinds, clips, which, np.arange(n)) - boot(arr_a, kinds, clips, which, np.arange(n))
        ds = []
        for _ in range(N_BOOT):
            i = rng.integers(0, n, n)
            ds.append(boot(arr_b, kinds, clips, which, i) - boot(arr_a, kinds, clips, which, i))
        ds = np.array(ds)
        ds = ds[~np.isnan(ds)]
        r = {"diff": float(d0), "ci_frames": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))] if len(ds) else [None, None]}
        if len(uniq) > 1:
            dc = []
            for _ in range(N_BOOT // 5):
                pick = rng.integers(0, len(uniq), len(uniq))
                i = np.concatenate([members[p] for p in pick])
                dc.append(boot(arr_b, kinds, clips, which, i) - boot(arr_a, kinds, clips, which, i))
            dc = np.array(dc)
            dc = dc[~np.isnan(dc)]
            r["ci_clips"] = [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))] if len(dc) else [None, None]
        res[which] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-src", type=Path, required=True)
    ap.add_argument("--clips-out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--pipeline-doc", type=Path, required=True, help="prefix of temporal_pipeline_eval (.md/.json are extended)")
    ap.add_argument("--k2-meta", type=Path, default=Path("/mnt/storage/footy_data/multiframe_v1/k2/meta.jsonl"))
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/storage/footy_data"))
    a = ap.parse_args()
    bl, hd, he, hs = load_harness(a.harness_src)
    knots_c, knots_p = np.array(hs.CALIB_KNOTS_CONF), np.array(hs.CALIB_KNOTS_P)
    cal = hs.calibrated_p

    def inv_cal(p):
        return np.interp(np.clip(p, knots_p[0], knots_p[-1]), knots_p, knots_c)

    evs = he.build_sets(a.data_root)
    test = [ev for ev in evs if ev.role == "test"]
    zs = {ev.clip.clip_id: np.load(a.clips_out / f"{ev.clip.clip_id}.npz") for ev in test}
    zx = np.load(a.clips_out / "arsenal_example.npz")
    n_ex = int(zx["n_frames"])
    pp, parr = zx["players_ptr"], zx["players_xywh"].astype(float)
    players = [parr[pp[t]:pp[t + 1]].reshape(-1, 4) for t in range(n_ex)]
    shift, cut = hd.estimate_camera_shift(players)
    gt, nb, aliases, conflicts = hd.load_gt(a.data_root, "arsenal_example")
    z0 = cands_from_npz(zx, "ref")
    ex_clip = hd.ClipData(clip_id="arsenal_example", n_frames=n_ex, ptr=z0[0], xy=z0[1], conf=z0[2], wh=z0[3],
                          gt={t: g for t, g in gt.items() if t < n_ex}, no_ball=set(), cam_shift=shift, cam_cut=cut, players=players,
                          gt_aliases=aliases, gt_conflicts=conflicts)
    groups = {"test (9 clips, harness exclusions)": [(ev.clip, ev.clean_scored, zs[ev.clip.clip_id]) for ev in test],
              "arsenal_example, all labelled frames (NOT harness-clean)": [(ex_clip, set(ex_clip.gt), zx)]}

    iso, iso_info = {}, {}
    for key in ("B_s0", "B_s1"):
        iso[key], iso_info[key] = val_isotonic(a.runs / f"preds_{key}.json", a.k2_meta)

    params = hs.SolverParams(**CAMERA_ON)

    def build(clip, z, key, mode):
        """mode: 'rt' | 'yolo' | 'union' | 'union+bonus'; returns ClipData with pseudo-confidences."""
        rt = clip if clip.clip_id != "arsenal_example" else clip  # test clips: harness machine labels; arsenal: rerun (already in clip)
        n = clip.n_frames
        if mode == "rt":
            return clip, {"agree": 0, "yolo_only": 0, "rt_only": 0}
        py_ptr, py_xy, py_conf, py_wh = cands_cached(z, key)
        py = iso[key](py_conf)
        rows_xy, rows_c, rows_wh, ptr = [], [], [], [0]
        stats = {"agree": 0, "yolo_only": 0, "rt_only": 0}
        bonus = BONUS_LOGIT if mode == "union+bonus" else 0.0
        for t in range(n):
            xa, ca = rt.cands(t)
            s, e = rt.ptr[t], rt.ptr[t + 1]
            wha = rt.wh[s:e]
            pa = cal(ca) if len(ca) else np.zeros(0)
            xb, pb, whb = py_xy[py_ptr[t]:py_ptr[t + 1]], py[py_ptr[t]:py_ptr[t + 1]], py_wh[py_ptr[t]:py_ptr[t + 1]]
            if mode == "yolo":
                xy, p, wh, ag = xb, pb, whb, 0
            else:
                xy, p, wh, ag = merge_frame(xa, pa, wha, xb, pb, whb, bonus)
            stats["agree"] += ag
            stats["rt_only"] += len(pa) - ag
            stats["yolo_only"] += len(pb) - ag
            o = np.argsort(-p) if len(p) else np.zeros(0, int)
            rows_xy.append(xy[o]), rows_c.append(inv_cal(p[o])), rows_wh.append(wh[o])
            ptr.append(ptr[-1] + len(p))
        c = dataclasses.replace(clip, ptr=np.array(ptr, np.int64), xy=np.concatenate(rows_xy) if ptr[-1] else np.zeros((0, 2)),
                                conf=np.concatenate(rows_c) if ptr[-1] else np.zeros(0), wh=np.concatenate(rows_wh) if ptr[-1] else np.zeros((0, 2)))
        return c, stats

    def union_ceiling_hit(clip, z, key, t, gxy):
        """any raw candidate (conf >= 0.1) of either source within 10 px."""
        xa, ca = clip.cands(t)
        ok = len(xa) and np.any((np.hypot(xa[:, 0] - gxy[0], xa[:, 1] - gxy[1]) <= HIT_PX) & (ca >= 0.1))
        return bool(ok)

    configs = {"RT-DETR alone": ("rt", None)}
    for key in ("B_s0", "B_s1"):
        configs[f"5ch {key} alone (val-iso p)"] = ("yolo", key)
        configs[f"union RT + {key}"] = ("union", key)
        configs[f"union RT + {key}, agreement bonus"] = ("union+bonus", key)

    out_tables, recs = {}, {}
    for gname, parts in groups.items():
        out_tables[gname] = {}
        for cname, (mode, key) in configs.items():
            for method in ("B0", "B1"):
                rows, counts_nb, n_emit, n_fr = [], [0, 0], 0, 0
                for clip, scored, z in parts:
                    c, st = build(clip, z, key, mode)
                    pred = bl.b0_per_frame(c, B0_TAU) if method == "B0" else bl.b1_solver(c, params)[0]
                    rr = frame_records(c, scored, pred)
                    # ceiling over the raw union of both sources (conf >= 0.1)
                    for i, r in enumerate(rr):
                        if r[1] == "gt":
                            gxy = clip.gt[r[0]].xy
                            ok = union_ceiling_hit(clip, z, key, r[0], gxy)
                            if mode != "rt" and key is not None:
                                py_ptr, py_xy, py_conf, _ = cands_cached(z, key)
                                s, e = py_ptr[r[0]], py_ptr[r[0] + 1]
                                ok = ok or bool(e > s and np.any((np.hypot(py_xy[s:e, 0] - gxy[0], py_xy[s:e, 1] - gxy[1]) <= HIT_PX) & (py_conf[s:e] >= 0.1)))
                                if mode == "yolo":
                                    ok = bool(e > s and np.any((np.hypot(py_xy[s:e, 0] - gxy[0], py_xy[s:e, 1] - gxy[1]) <= HIT_PX) & (py_conf[s:e] >= 0.1)))
                            rr[i] = (r[0], r[1], r[2], r[3], int(ok))
                    rows += [(clip.clip_id, *r) for r in rr]
                    n_emit += len(pred)
                    n_fr += clip.n_frames
                    for t in clip.no_ball:
                        if t in scored:
                            counts_nb[0] += 1
                            counts_nb[1] += int(t in pred)
                recs[(gname, cname, method)] = {"kinds": np.array([r[2] for r in rows]), "clips": np.array([r[0] for r in rows]),
                                                "arr": np.array([[r[3], r[4], r[5]] for r in rows], float)}
                arr, kinds = recs[(gname, cname, method)]["arr"], recs[(gname, cname, method)]["kinds"]
                gtm = kinds == "gt"
                tp, fp = arr[:, 0].sum(), arr[:, 1].sum()
                out_tables[gname][f"{cname}|{method}"] = {
                    "n_gt": int(gtm.sum()), "recall10": float(arr[gtm, 0].mean()), "precision10": float(tp / (tp + fp)) if tp + fp else None,
                    "ceiling10": float(arr[gtm, 2].mean()), "tp": int(tp), "fp": int(fp), "no_ball_frames": counts_nb[0],
                    "no_ball_emitted": counts_nb[1], "emitted_frames": n_emit, "all_frames": n_fr}
    # agreement stats (union merge counts), summed over the group's clips
    merge_stats = {}
    for gname, parts in groups.items():
        for key in ("B_s0", "B_s1"):
            tot = {"agree": 0, "yolo_only": 0, "rt_only": 0}
            for clip, scored, z in parts:
                _, st = build(clip, z, key, "union")
                for k in tot:
                    tot[k] += st[k]
            merge_stats[f"{gname}|{key}"] = tot

    rng = np.random.default_rng(0)
    pair = {}
    for gname in groups:
        pair[gname] = {}
        for cname in configs:
            if cname == "RT-DETR alone":
                continue
            for method in ("B1",):
                ra, rb = recs[(gname, "RT-DETR alone", method)], recs[(gname, cname, method)]
                pair[gname][f"{cname}|{method}"] = paired(ra["arr"], rb["arr"], ra["kinds"], ra["clips"], rng)
    res = {"merge_px": MERGE_PX, "bonus_logit": BONUS_LOGIT, "val_isotonic": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("knots")} | {"n_knots": len(v["knots_p"]), "knots_p_range": [min(v["knots_p"]), max(v["knots_p"])]}
                                                                              for k, v in iso_info.items()},
           "tables": out_tables, "paired_vs_rtdetr": pair, "merge_candidate_counts": merge_stats}
    jp = Path(str(a.pipeline_doc) + ".json")
    doc = json.loads(jp.read_text())
    doc["union_test"] = res
    jp.write_text(json.dumps(doc, indent=1, default=lambda o: None if o != o else float(o)))
    mp = Path(str(a.pipeline_doc) + ".md")
    md = mp.read_text()
    if SECTION in md:
        md = md[:md.index(SECTION)]
    mp.write_text(md.rstrip() + "\n\n" + render(res, groups) + "\n")
    print("union section written")


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def render(res, groups) -> str:
    L = [SECTION + "\n"]
    L.append("Per frame the candidates of RT-DETR-L (harness knots -> p) and of a 5ch YOLO arm (isotonic map fitted on v11 *valid* without "
             f"arsenal_example -> p) are merged: cross-source duplicates within {res['merge_px']:.0f} px collapse to one candidate (higher-p position, "
             f"p = max), optionally with a +{res['bonus_logit']} logit bonus when both agree. The merged p is fed to the harness' B0 (tau 0.4) / B1 (camera-on "
             "parameters) unchanged, through the inverse of the harness knots (RT-DETR alone is reproduced exactly). Ceiling of a union row = "
             "any candidate of either source (conf >= 0.1) within 10 px. 5ch YOLO = arm B of v1, seeds 0 and 1 reported separately.\n")
    L.append("Val isotonic maps (n small: " + "; ".join(f"{k}: {v['n_candidates']} candidates ({v['n_correct']} correct) from {v['n_images']} images, {v['n_knots']} knots, p in "
                                                         f"[{v['knots_p_range'][0]:.2f}, {v['knots_p_range'][1]:.2f}]" for k, v in res["val_isotonic"].items()) + ").\n")
    for gname in groups:
        tb = res["tables"][gname]
        L.append(f"### {gname}\n")
        L.append("| candidate set | B0 recall / prec | B1 recall / prec | ceiling | B1 emitted on no_ball frames |")
        L.append("|---|---|---|---|---|")
        for cname in {k.split("|")[0] for k in tb}:
            pass
        seen = []
        for k in tb:
            cname = k.split("|")[0]
            if cname in seen:
                continue
            seen.append(cname)
            b0, b1 = tb[f"{cname}|B0"], tb[f"{cname}|B1"]
            L.append(f"| {cname} | {fmt(b0['recall10'])} / {fmt(b0['precision10'])} | {fmt(b1['recall10'])} / {fmt(b1['precision10'])} | {fmt(b1['ceiling10'])} | "
                     f"{b1['no_ball_emitted']} / {b1['no_ball_frames']} (B0: {b0['no_ball_emitted']}) |")
        L.append("")
        L.append("Paired bootstrap vs RT-DETR alone, B1 (95% CI over frames; over clips in brackets, test only): ")
        L.append("")
        L.append("| candidate set | recall@10 diff | precision diff | ceiling diff |")
        L.append("|---|---|---|---|")
        for cname in seen[1:]:
            p = res["paired_vs_rtdetr"][gname][f"{cname}|B1"]
            cells = []
            for w in ("recall10", "precision10", "ceiling10"):
                d = p[w]
                c = f"{d['diff']:+.3f} [{fmt(d['ci_frames'][0])}, {fmt(d['ci_frames'][1])}]"
                if "ci_clips" in d:
                    c += f" ({fmt(d['ci_clips'][0])}, {fmt(d['ci_clips'][1])})"
                cells.append(c)
            L.append(f"| {cname} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("Reading: the 5ch candidates add to, rather than replace, the production detector, but modestly. The ceiling rises only a little "
             "(test +0.012 / +0.006 for seeds 0 / 1, arsenal_example +0.006 / +0.011) because most 5ch candidates duplicate RT-DETR's (about 6.5k "
             "agreeing candidates against about 1k YOLO-only on the test clips). B1 recall rises +0.029 / +0.041 on the test clips (frame CI excludes 0; the "
             "clip-level CI touches 0 because 7 of the 9 clips have a single GT frame and seg010 carries 65% of the frames) and +0.062 / +0.023 on "
             "arsenal_example; precision is -0.034 (seed 0) / +0.001 (seed 1) on test, so seed 0 also adds some wrong candidates. The agreement "
             "bonus changes almost nothing. On the 2 harness no_ball frames of the test clips no configuration emits a point (n = 2: uninformative). "
             "The val isotonic maps rest on 43-44 candidates (4-5 knots): too small to trust, so the union numbers are indicative; they do show "
             "that the gain over RT-DETR alone does not depend on a well-calibrated YOLO map.\n")
    L.append("Merged candidate counts (all frames): " + "; ".join(f"{k}: agree {v['agree']}, RT-DETR only {v['rt_only']}, YOLO only {v['yolo_only']}"
                                                                for k, v in res["merge_candidate_counts"].items()) + ".")
    return "\n".join(L)


if __name__ == "__main__":
    main()
