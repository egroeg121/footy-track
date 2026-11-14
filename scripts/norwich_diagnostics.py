#!/usr/bin/env python
# ruff: noqa: PLR0912, PLR0915, PLR0917, C408, T201
"""Unsupervised B0/B1 diagnostics and a dense-labelling queue for the held-out
``arsenal_norwich`` match (actually Arsenal v Nottingham Forest; never train on it).

    PYTHONPATH=src python scripts/norwich_diagnostics.py \
        --video-feats <dir from norwich_video_cuts.py> --out-sheets <dir>

Stages (all in one run): load RT-DETR rows -> camera shift + cuts -> B0 / B1 ->
per-frame diagnostics -> per-clip and per-250-frame-tile summaries -> sliding
window search for the label queue -> contact sheets -> json / csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from footy_track.ball_offline import baselines as bl  # noqa: E402
from footy_track.ball_offline.data import (  # noqa: E402
    ClipData,
    _load_machine,
    estimate_camera_shift,
)
from footy_track.ball_offline.solver import SolverParams, calibrated_p  # noqa: E402

ROOT = Path("/mnt/storage/footy_data")
VIDEO_DIR = ROOT / "arsenal_norwich/split_video_broadcast_frames"
LABEL_DIR = ROOT / "machine_labels_norwich"
CAND_DIR = ROOT / "ball_candidates_norwich"
WIN = 250
STRIDE = 25
B0_TAU = 0.4
DIS_PX = 10.0
GAP_MIN = 10

# B1 = "camera_on" config selected on Man City dev clips (offline_ball_baseline.md s6)
B1_PARAMS = SolverParams(
    min_conf=0.3, bias=1.0, emit_scale=1.0, kappa=1.5, lam_pos=0.5, lam_vel=0.0,
    vmax=40.0, r0=12.0, dmax=2.0, mu=0.02, static_px=3.0, max_gap=25, camera=True,
    fill_gap=10, fill_min_p=0.8, fill_mode="kalman",
)  # fmt: skip


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_clip(stem: str, n_frames: int, vfeat: dict | None, hist_thr: float, mad_thr: float):
    ptr, xy, conf, wh, players = _load_machine(LABEL_DIR / f"{stem}.jsonl", n_frames)
    shift, cam_cut = estimate_camera_shift(players)
    vcut = np.zeros(n_frames, bool)
    if vfeat is not None:
        n = min(n_frames, len(vfeat["hist_d"]))
        vcut[:n] = (vfeat["hist_d"][:n] > hist_thr) & (vfeat["mad"][:n] > mad_thr)
        # frames 1-4 of every clip show an encoder/dissolve-in transient (clips are already
        # cut at broadcast boundaries), not a camera cut
        vcut[:5] = False
    # B1 sees both cut flags (a link must not cross a cut); shift is zeroed on video cuts
    cut = cam_cut | vcut
    shift[vcut] = 0.0
    clip = ClipData(
        clip_id=stem, n_frames=n_frames, ptr=ptr, xy=xy, conf=conf, wh=wh, gt={},
        no_ball=set(), cam_shift=shift, cam_cut=cut, players=players,
    )  # fmt: skip
    return clip, cam_cut, vcut


# --------------------------------------------------------------------------
# per-frame diagnostics
# --------------------------------------------------------------------------


def frame_arrays(clip: ClipData) -> dict[str, np.ndarray]:
    n = clip.n_frames
    pred1, nodes = bl.b1_solver(clip, B1_PARAMS)
    pred0 = bl.b0_per_frame(clip, B0_TAU)
    present = np.zeros(n, bool)
    b1xy = np.full((n, 2), np.nan)
    for t, p in pred1.items():
        present[t] = True
        b1xy[t] = p
    node = np.zeros(n, bool)
    node_p = np.full(n, np.nan)
    node_conf = np.full(n, np.nan)
    for nd in nodes:
        node[nd.t] = True
        node_conf[nd.t] = nd.conf
        node_p[nd.t] = float(calibrated_p(np.array([nd.conf]))[0])
    maxconf = np.zeros(n)
    top1xy = np.full((n, 2), np.nan)
    n03 = np.zeros(n, int)
    multi = np.zeros(n, bool)
    for t in range(n):
        xy, conf = clip.cands(t)
        if not len(conf):
            continue
        k = int(np.argmax(conf))
        maxconf[t] = conf[k]
        top1xy[t] = xy[k]
        hi = conf >= 0.3
        n03[t] = int(hi.sum())
        if hi.sum() >= 2:
            pts = xy[hi]
            d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
            multi[t] = bool((d > 30).any())
    b0 = np.zeros(n, bool)
    for t in pred0:
        b0[t] = True
    d01 = np.hypot(*(top1xy - b1xy).T)
    # B0 confident top-1 present: B1 either moved it (> DIS_PX) or dropped it
    moved = b0 & present & (d01 > DIS_PX)
    dropped = b0 & ~present
    nplayers = np.array([len(p) for p in clip.players])
    return dict(
        present=present, node=node, node_p=node_p, node_conf=node_conf, b0=b0,
        moved=moved, dropped=dropped, maxconf=maxconf, n03=n03, multi=multi,
        nplayers=nplayers, b1xy=b1xy, top1xy=top1xy, d01=d01,
    )  # fmt: skip


def gap_stats(present: np.ndarray) -> dict:
    """Runs of absence inside [0, len): internal gaps are bounded by present frames."""
    n = len(present)
    idx = np.flatnonzero(present)
    if not len(idx):
        return dict(n_segments=0, n_gaps=0, n_gaps_ge10=0, max_gap=0, gap_frames_ge10=0,
                    lead_absent=n, trail_absent=0)  # fmt: skip
    gaps = np.diff(idx) - 1
    gaps = gaps[gaps > 0]
    segs = 1 + int((np.diff(idx) > 1).sum())
    return dict(
        n_segments=segs,
        n_gaps=int(len(gaps)),
        n_gaps_ge10=int((gaps >= GAP_MIN).sum()),
        max_gap=int(gaps.max()) if len(gaps) else 0,
        gap_frames_ge10=int(gaps[gaps >= GAP_MIN].sum()),
        lead_absent=int(idx[0]),
        trail_absent=int(n - 1 - idx[-1]),
    )


def window_stats(A: dict, s: int, e: int) -> dict:
    sl = slice(s, e)
    n = e - s
    present = A["present"][sl]
    node_p = A["node_p"][sl]
    mc = A["maxconf"][sl]
    out = dict(
        n=n,
        b1_present=float(present.mean()),
        b1_nodes=float(A["node"][sl].mean()),
        b0_present=float(A["b0"][sl].mean()),
        mean_p=float(np.nanmean(node_p)) if np.isfinite(node_p).any() else None,
        disagree_moved=int(A["moved"][sl].sum()),
        disagree_dropped=int(A["dropped"][sl].sum()),
        disagree_frac=float((A["moved"][sl] | A["dropped"][sl]).mean()),
        max_conf=float(mc.max()),
        mean_frame_maxconf=float(mc.mean()),
        frac_conf_ge03=float((mc >= 0.3).mean()),
        frac_conf_ge01=float((mc >= 0.1).mean()),
        frac_conf_ge07=float((mc >= 0.7).mean()),
        multi_frac=float(A["multi"][sl].mean()),
        median_players=float(np.median(A["nplayers"][sl])),
    )
    out.update(gap_stats(present))
    return out


def rnd(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


# --------------------------------------------------------------------------
# queue selection
# --------------------------------------------------------------------------

STRATA = ("easy", "gappy", "contested", "low_evidence")


def classify(w: dict) -> dict[str, float]:
    """Stratum -> score for every stratum a window qualifies for."""
    out: dict[str, float] = {}
    mp = w["mean_p"] or 0.0
    if w["b1_present"] >= 0.9 and mp >= 0.9:
        out["easy"] = w["b1_present"] + mp + 0.5 * w["frac_conf_ge07"]
    if w["gap_frames_ge10"] >= GAP_MIN and w["b1_present"] >= 0.4 and w["median_players"] >= 6:
        out["gappy"] = min(w["gap_frames_ge10"], 100) / 100 + w["b1_present"] + 0.3 * w["n_gaps_ge10"]
    # thresholds below are set relative to the Norwich tile distribution (95th pct of
    # disagree_frac is ~0.10; this broadcast has big, confidently detected balls)
    if (w["disagree_frac"] >= 0.06 or w["multi_frac"] >= 0.25) and w["b1_present"] >= 0.3:
        out["contested"] = w["disagree_frac"] + 0.5 * w["multi_frac"]
    if w["frac_conf_ge07"] <= 0.35 and w["frac_conf_ge01"] >= 0.5 and w["median_players"] >= 8:
        out["low_evidence"] = (1 - w["frac_conf_ge07"]) + 0.5 * (1 - w["frac_conf_ge03"])
    return out


def reason(stratum: str, w: dict) -> str:
    pc = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    if stratum == "easy":
        return (
            f"B1 covers {pc(w['b1_present'])} of frames, mean calibrated p {w['mean_p']:.2f}, "
            f"{pc(w['frac_conf_ge07'])} frames have a candidate >=0.7, "
            f"B0/B1 disagree on {pc(w['disagree_frac'])}"
        )
    if stratum == "gappy":
        return (
            f"B1 present {pc(w['b1_present'])} with {w['n_gaps_ge10']} gap(s) >=10 frames "
            f"(longest {w['max_gap']}, {w['gap_frames_ge10']} frames) - occlusion, aerial ball or out of shot"
        )
    if stratum == "contested":
        return (
            f"B0 top-1 and B1 disagree on {pc(w['disagree_frac'])} of frames "
            f"({w['disagree_moved']} moved >10px, {w['disagree_dropped']} confident top-1 dropped); "
            f"{pc(w['multi_frac'])} of frames have >=2 distant candidates >=0.3"
        )
    return (
        f"only {pc(w['frac_conf_ge07'])} of frames have a candidate >=0.7 ({pc(w['frac_conf_ge03'])} >=0.3, max {w['max_conf']:.2f}); "
        f"B1 present {pc(w['b1_present'])}; "
        f"{w['median_players']:.0f} players in view, ball probably tiny or occluded"
    )


def select_queue(cands: list[dict], clip_times: dict[str, int], per_stratum: int = 6, max_per_clip: int = 2):
    """cands: windows with .clip/.start/.strata. Spread each stratum over the match
    timeline (``per_stratum`` equal time bins), best score per bin, no overlap, and
    at most ``max_per_clip`` windows per clip overall."""
    lo = min(clip_times.values())
    hi = max(clip_times[c["clip"]] + c["start"] for c in cands)
    chosen: list[dict] = []
    used: dict[str, list[tuple[int, int]]] = {}
    per_clip: dict[str, int] = {}

    def ok(c) -> bool:
        if per_clip.get(c["clip"], 0) >= max_per_clip:
            return False
        s, e = c["start"], c["start"] + WIN
        return all(e <= a or s >= b for a, b in used.get(c["clip"], []))

    def take(c, stratum):
        chosen.append({**c, "stratum": stratum, "score": c["strata"][stratum]})
        used.setdefault(c["clip"], []).append((c["start"], c["start"] + WIN))
        per_clip[c["clip"]] = per_clip.get(c["clip"], 0) + 1

    pools = {s: [c for c in cands if s in c["strata"]] for s in STRATA}
    order = sorted(STRATA, key=lambda s: len(pools[s]))  # scarcest first
    for s in order:
        pool = sorted(pools[s], key=lambda c: -c["strata"][s])
        edges = np.linspace(lo, hi + 1, per_stratum + 1)
        got: list[dict] = []
        for b in range(per_stratum):
            for c in pool:
                g = clip_times[c["clip"]] + c["start"]
                if edges[b] <= g < edges[b + 1] and ok(c):
                    take(c, s)
                    got.append(c)
                    break
        # fill empty bins: farthest-in-time from what we have, among the better half
        while len(got) < per_stratum:
            rest = [c for c in pool[: max(50, len(pool) // 2)] if ok(c)]
            if not rest:
                rest = [c for c in pool if ok(c)]
            if not rest:
                break
            gt = [clip_times[c["clip"]] + c["start"] for c in got] or [lo]
            best = max(rest, key=lambda c: (min(abs(clip_times[c["clip"]] + c["start"] - x) for x in gt), c["strata"][s]))
            take(best, s)
            got.append(best)
    chosen.sort(key=lambda c: (STRATA.index(c["stratum"]), clip_times[c["clip"]] + c["start"]))
    return chosen


# --------------------------------------------------------------------------
# contact sheets
# --------------------------------------------------------------------------


def contact_sheet(item: dict, A: dict, dst: Path, n_thumb: int = 6) -> dict:
    stem, s = item["clip"], item["start"]
    frames = [s + round(i * (WIN - 1) / (n_thumb - 1)) for i in range(n_thumb)]
    want = set(frames)
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{stem}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    got: dict[int, np.ndarray] = {}
    for t in range(s, s + WIN):
        ok, f = cap.read()
        if not ok:
            break
        if t in want:
            got[t] = f
    cap.release()
    TW, TH = 960, 540
    sc = TW / 1920
    sheet = np.zeros((2 * TH, 3 * TW, 3), np.uint8)
    black = 0
    for i, t in enumerate(frames):
        f = got.get(t)
        r, c = divmod(i, 3)
        if f is None or f.mean() < 8:
            black += 1
            if f is None:
                continue
        th = cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA)
        b1 = A["b1xy"][t]
        t1 = A["top1xy"][t]
        label = f"{stem[-6:]} f{t}"
        if np.isfinite(b1[0]):
            col = (0, 255, 0) if A["node"][t] else (0, 255, 255)
            cx, cy = int(b1[0] * sc), int(b1[1] * sc)
            cv2.circle(th, (cx, cy), 14, col, 2)
            kind = "det" if A["node"][t] else "interp"
            p = A["node_p"][t]
            label += f"  B1 {kind}" + (f" p={p:.2f}" if np.isfinite(p) else "")
            ix, iy = int(b1[0]), int(b1[1])
        else:
            label += "  B1: none"
            ix, iy = (int(t1[0]), int(t1[1])) if np.isfinite(t1[0]) else (960, 540)
        if np.isfinite(t1[0]) and A["b0"][t] and (not np.isfinite(b1[0]) or A["d01"][t] > DIS_PX):
            cv2.circle(th, (int(t1[0] * sc), int(t1[1] * sc)), 14, (0, 0, 255), 2)
            label += f"  B0 top1 c={A['maxconf'][t]:.2f} (red)"
        # 3x zoom inset around the marked point
        h = 40
        x0, y0 = min(max(ix - h, 0), 1920 - 2 * h), min(max(iy - h, 0), 1080 - 2 * h)
        crop = cv2.resize(f[y0 : y0 + 2 * h, x0 : x0 + 2 * h], (240, 240), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop, (0, 0), (239, 239), (255, 255, 255), 1)
        th[0:240, TW - 240 : TW] = crop
        cv2.rectangle(th, (0, 0), (TW - 245, 26), (0, 0, 0), -1)
        cv2.putText(th, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        sheet[r * TH : (r + 1) * TH, c * TW : (c + 1) * TW] = th
        cv2.rectangle(sheet, (c * TW, r * TH), ((c + 1) * TW - 1, (r + 1) * TH - 1), (60, 60, 60), 1)
    cv2.imwrite(str(dst), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return {"frames": frames, "black_thumbs": black, "decoded": len(got)}


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-feats", type=Path, required=True)
    ap.add_argument("--out-sheets", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=Path("docs/experiments/norwich_unsupervised_diagnostics.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("docs/experiments/norwich_label_queue.csv"))
    ap.add_argument("--hist-thr", type=float, default=0.1)
    ap.add_argument("--mad-thr", type=float, default=2.5)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--cache", type=Path, default=None, help="pickle cache of per-frame arrays")
    args = ap.parse_args()
    cv2.setNumThreads(2)
    import pickle

    manifest = json.loads((LABEL_DIR / "_manifest.json").read_text())
    seg = json.loads((VIDEO_DIR / "segments.json").read_text())
    seg_by = {s["file"][:-4]: s for s in seg["segments"]}
    clips_meta = manifest["clips"]
    stems = sorted(clips_meta)
    assert len(stems) == 76, len(stems)

    if args.cache and args.cache.exists():
        store = pickle.loads(args.cache.read_bytes())
        log("loaded cache")
    else:
        store = {}
        t0 = time.time()
        for stem in stems:
            n = clips_meta[stem]["frames"]
            vf = None
            vp = args.video_feats / f"{stem}.npz"
            if vp.exists():
                z = np.load(vp)
                vf = {k: z[k] for k in z.files}
            clip, cam_cut, vcut = load_clip(stem, n, vf, args.hist_thr, args.mad_thr)
            A = frame_arrays(clip)
            # camera-estimation health
            valid = np.array([len(p) >= 4 for p in clip.players])
            sh = clip.cam_shift
            A["cam_valid"] = valid
            A["cam_mag"] = np.hypot(sh[:, 0], sh[:, 1])
            A["cam_cut"] = cam_cut
            A["vcut"] = vcut
            A["cut"] = clip.cam_cut
            A["vmean"] = vf["mean"] if vf else np.full(n, np.nan)
            A["n_video_frames"] = len(vf["mean"]) if vf else None
            A["n_cand"] = int(len(clip.conf))
            store[stem] = A
            log(f"{stem}: n={n} cand={len(clip.conf)} b1={A['present'].mean():.2f} {time.time() - t0:.0f}s")
        if args.cache:
            args.cache.write_bytes(pickle.dumps(store))

    # ---------------- per clip / per tile -------------------------------
    per_clip, tiles = [], []
    for stem in stems:
        A = store[stem]
        n = clips_meta[stem]["frames"]
        w = window_stats(A, 0, n)
        w.update(
            clip=stem, n_frames=n,
            source_start_frame=seg_by[stem]["start_frame"], source_start_s=seg_by[stem]["start_s"],
            n_cam_cuts=int(A["cam_cut"].sum()), n_video_cuts=int(A["vcut"].sum()),
            n_cuts_any=int(A["cut"].sum()),
            cam_est_valid_frac=float(A["cam_valid"][1:].mean()) if n > 1 else None,
            cam_shift_median_px=float(np.median(A["cam_mag"])),
            n_candidates=A["n_cand"], video_frames=A["n_video_frames"],
            min_video_mean=float(np.nanmin(A["vmean"])) if np.isfinite(A["vmean"]).any() else None,
        )  # fmt: skip
        per_clip.append(rnd(w))
        for s in range(0, n, WIN):
            e = min(n, s + WIN)
            if e - s < 50:
                continue
            t = window_stats(A, s, e)
            t.update(clip=stem, start_frame=s, end_frame=e - 1, partial=(e - s) < WIN,
                     cut_inside=bool(A["cut"][s + 1 : e].any()))  # fmt: skip
            tiles.append(rnd(t))

    # ---------------- overall -------------------------------------------
    allA = {k: np.concatenate([store[s][k] for s in stems]) for k in ("present", "node", "b0", "moved", "dropped", "maxconf", "node_conf", "node_p", "multi")}
    ncl = int(len(allA["present"]))
    nc = allA["node_conf"][np.isfinite(allA["node_conf"])]
    npv = allA["node_p"][np.isfinite(allA["node_p"])]
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    overall = dict(
        n_clips=len(stems), n_frames=ncl,
        b1_present=float(allA["present"].mean()), b1_det_nodes=float(allA["node"].mean()),
        b1_interpolated=float((allA["present"] & ~allA["node"]).mean()),
        b0_present_tau04=float(allA["b0"].mean()),
        disagree_frac=float((allA["moved"] | allA["dropped"]).mean()),
        disagree_moved_frac=float(allA["moved"].mean()), disagree_dropped_frac=float(allA["dropped"].mean()),
        disagree_given_b0=float((allA["moved"] | allA["dropped"]).sum() / max(1, allA["b0"].sum())),
        node_conf_pctl={q: round(pct(nc, q), 3) for q in (5, 10, 25, 50, 75, 90)},
        node_calibrated_p_mean=float(npv.mean()),
        node_conf_bands={
            "<0.42": float((nc < 0.42).mean()), "0.42-0.60": float(((nc >= 0.42) & (nc < 0.6)).mean()),
            "0.60-0.70": float(((nc >= 0.6) & (nc < 0.7)).mean()), ">=0.70": float((nc >= 0.7).mean()),
        },
        frames_with_cand_ge01=float((allA["maxconf"] >= 0.1).mean()),
        frames_with_cand_ge03=float((allA["maxconf"] >= 0.3).mean()),
        frames_with_cand_ge07=float((allA["maxconf"] >= 0.7).mean()),
        n_nodes=int(len(nc)),
    )  # fmt: skip
    cam = dict(
        cam_cuts=int(sum(store[s]["cam_cut"].sum() for s in stems)),
        video_cuts=int(sum(store[s]["vcut"].sum() for s in stems)),
        both=int(sum((store[s]["cam_cut"] & store[s]["vcut"]).sum() for s in stems)),
        cam_est_valid_frac=float(np.concatenate([store[s]["cam_valid"][1:] for s in stems]).mean()),
        black_frames=int(sum((store[s]["vmean"] < 8).sum() for s in stems)),
    )

    # ---------------- queue ---------------------------------------------
    clip_times = {s: seg_by[s]["start_frame"] for s in stems}
    cands = []
    for stem in stems:
        A = store[stem]
        n = clips_meta[stem]["frames"]
        for s in range(10, n - WIN + 1, STRIDE):  # skip the 10-frame dissolve-in at clip start
            if A["cut"][s + 1 : s + WIN].any():
                continue
            vm = A["vmean"][s : s + WIN]
            if np.isfinite(vm).any() and vm.min() < 20:
                continue
            w = window_stats(A, s, s + WIN)
            if w["median_players"] < 5:
                continue
            st = classify(w)
            if st:
                cands.append(dict(clip=stem, start=s, w=w, strata=st))
    log(f"{len(cands)} qualifying windows: " + ", ".join(f"{s}={sum(s in c['strata'] for c in cands)}" for s in STRATA))
    queue = select_queue(cands, clip_times)
    args.out_sheets.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, q in enumerate(queue, 1):
        w = q["w"]
        q["reason"] = reason(q["stratum"], w)
        sheet = args.out_sheets / f"q{i:02d}_{q['stratum']}_{q['clip']}_f{q['start']:04d}-{q['start'] + WIN - 1:04d}.png"
        info = {}
        if not args.no_sheets:
            info = contact_sheet(q, store[q["clip"]], sheet)
        rows.append(
            dict(
                id=i, clip=q["clip"], start_frame=q["start"], end_frame=q["start"] + WIN - 1,
                stratum=q["stratum"], reason=q["reason"], sheet=str(sheet),
                source_time_min=round((seg_by[q["clip"]]["start_frame"] + q["start"]) / 25 / 60, 1),
                other_strata=sorted(set(q["strata"]) - {q["stratum"]}),
                stats=rnd(w), sheet_info=info,
            )
        )  # fmt: skip
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            dict(
                note="arsenal_norwich is actually Arsenal v Nottingham Forest; held out, never train on it.",
                b1_params=B1_PARAMS.__dict__, b0_tau=B0_TAU, disagree_px=DIS_PX, window=WIN,
                cut_thresholds=dict(hist_d=args.hist_thr, mad=args.mad_thr),
                overall=rnd(overall), camera_and_cuts=cam, per_clip=per_clip, tiles_250=tiles,
                queue=rows,
            ),
            indent=1,
        )
    )  # fmt: skip
    with args.out_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip", "start_frame", "end_frame", "stratum", "reason"])
        for r in rows:
            wr.writerow([r["clip"], r["start_frame"], r["end_frame"], r["stratum"], r["reason"]])
    log(f"wrote {args.out_json} and {args.out_csv}; {len(rows)} windows")


if __name__ == "__main__":
    main()
