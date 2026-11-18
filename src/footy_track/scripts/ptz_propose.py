"""V2 of the PTZ rig proposal: fit a ground's rig, validate it, solve frames and publish a review run.

No LLM calls; single process (run it with ``OMP_NUM_THREADS=1``)::

    python -m footy_track.scripts.ptz_propose rig     --ground vitality --rig-out PATH [--boot 40] [--out DIR]
    python -m footy_track.scripts.ptz_propose loso    --ground vitality [--rig-out PATH] --out DIR
    python -m footy_track.scripts.ptz_propose propose --ground vitality --rig PATH --run-id ptz_bmth1
                  [--clips a,b] [--frames 10,20] [--from-run bmth_pre1] [--extras 3] [--render-dir DIR] [--dry-run]

``rig``     joint fit of the ground's rig (C, roll, k1) from its calibration GT, robust (inconsistent points / frames
            are reported and dropped), shot-bootstrap spread, pitch-size check; writes the rig JSON.
``loso``    leave-one-shot-out: the rig is refit without the clip of each GT frame, then that frame is solved from 2
            / 3 of its GT points and from the video's white lines only; error in metres against the GT homography
            (a metric of the same kind as ``ptz_validate``; GT is itself a fit, see reports/ptz_v2.md).
``publish``  adds a dry run's ``records.json`` to the store (no re-solve).
``propose`` solves frames (default targets: the failed / rejected records of ``--from-run`` plus ``--extras``
            evenly spaced frames of every clip of the ground that has neither GT nor a not-broadcast mark), renders
            overlays and, unless ``--dry-run``, adds the records as run ``--run-id`` to the proposals store the review
            page reads (``<FOOTY_GT_MARKS_DIR>/calibration_proposals``); other runs are never touched.
"""

# ruff: noqa: PLR0912, PLR0915, PLC0415, E702
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration import ptz
from footy_track.calibration import ptz_propose as pp
from footy_track.calibration.pitch_model import PitchModel
from footy_track.clip_id import canonical_clip_id

GT_ROOT = Path(os.environ.get("FOOTY_GT_MARKS_DIR", "/mnt/storage/footy_data/ball_gt_marks"))
GROUND_MATCHES = {"vitality": ("bmth",), "emirates": ("mancity", "villa")}


def _gt(ground: str):
    from footy_track.scripts import ptz_validate as pv

    return pv.load_gt(GROUND_MATCHES[ground])


def _rigframes(gt):
    return [ptz.RigFrame(f.img, f.pitch, f.key, f.clip) for f in gt]


# --------------------------------------------------------------------------
# rig
# --------------------------------------------------------------------------


def cmd_rig(args):
    gt = _gt(args.ground)
    C0 = pp.GROUNDS[args.ground]["C0"]
    t0 = time.time()
    rf = pp.fit_rig_robust(_rigframes(gt), C0=C0, point_px=args.point_px, frame_px=args.frame_px)
    rig = rf.rig
    print(f"{args.ground}: {len(gt)} GT frames, used {len(rf.used)}, {time.time() - t0:.0f}s")
    print("C", np.round(rig.C, 3), "roll_deg", round(float(np.degrees(rig.roll)), 3), "k1", round(rig.k1, 4),
          "cost", round(rig.info["cost"], 1))
    per = []
    for f in gt:
        c = rf.cams.get(f.key)
        if c is None:
            # an excluded frame: how does it fit the rig with its camera pose free (C fixed)?
            s = ptz.solve_points(f.img, f.pitch, rig.with_sigma((0, 0, 0)), n_starts=10)
            per.append({"key": f.key, "n": len(f.ids), "excluded": True,
                        "rms_px_fixedC": s.info["rms_px"] if s else None})
            continue
        per.append({"key": f.key, "n": len(f.ids), "rms_px": c.info["rms_px"], "f_px": c.f,
                    "pan_deg": float(np.degrees(c.pan)), "tilt_deg": float(np.degrees(c.tilt))})
    for p in per:
        print("  ", p)
    print("excluded:", rf.excluded)
    print("dropped points:", rf.dropped_points)
    out = {"ground": args.ground, "C": [float(c) for c in rig.C], "roll_deg": float(np.degrees(rig.roll)),
           "k1": float(rig.k1), "cost": rig.info["cost"], "per_frame": per, "excluded": rf.excluded,
           "dropped_points": rf.dropped_points}
    if args.boot:
        out["bootstrap"] = pp.bootstrap_rig(_rigframes(gt), args.boot, C0=C0, point_px=args.point_px,
                                            frame_px=args.frame_px)
        print("bootstrap", out["bootstrap"])
        rig.info["bootstrap"] = out["bootstrap"]
    if args.dims:
        ids = {f.key: list(f.ids) for f in gt}
        keep = {fr.key for fr in rf.used}
        fi = [(fr, ids[fr.key]) for fr in _rigframes(gt) if fr.key in keep]
        out["pitch_dims_cost"] = {}
        for dims in ((105.0, 68.0), (100.0, 64.0), (104.0, 68.0), (105.0, 66.0), (102.0, 66.0)):
            # points of kept frames only (dropped points stay: the point-level filter is re-applied by the fit)
            cost = pp.rig_cost_for_pitch(fi, dims, C0=C0, point_px=args.point_px, frame_px=args.frame_px)
            out["pitch_dims_cost"][f"{dims[0]:g}x{dims[1]:g}"] = cost
            print("pitch", dims, "cost", round(cost, 1), flush=True)
    if args.k1_lines:
        ks = []
        for f in gt:
            c = rf.cams.get(f.key)
            if c is None:
                continue
            ev = ptz.LineEvidence.from_masks(*pv_masks(f, Path(args.mask_cache or args.out) / "masks"))
            c2 = ptz._refine_lines(ptz.Camera(c.pan, c.tilt, c.logf, rig.roll, rig.C.copy(), rig.k1), ev,
                                   rig.with_sigma((0, 0, 0)), PitchModel(*rig.pitch), fit_k1=True, k1_prior=0.05)
            ks.append({"key": f.key, "k1": float(c2.k1)})
            print("  k1 (lines)", f.key, round(c2.k1, 4), flush=True)
        out["k1_lines"] = ks
        print("k1 from lines: median", np.median([k["k1"] for k in ks]))
    rig.info.update(ground=args.ground, n_frames=len(rf.used), excluded=rf.excluded,
                    dropped_points=rf.dropped_points, fit_date=time.strftime("%Y-%m-%d"))
    if args.rig_out:
        Path(args.rig_out).parent.mkdir(parents=True, exist_ok=True)
        rig.to_json(args.rig_out)
        print("wrote", args.rig_out)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f"rig_{args.ground}.json").write_text(json.dumps(out, indent=1, default=float))


# --------------------------------------------------------------------------
# leave-one-shot-out
# --------------------------------------------------------------------------


def _draws(f, k: int, n_draws: int, rng):
    import itertools

    combos = [c for c in itertools.combinations(range(len(f.ids)), k)
              if all(np.linalg.norm(f.pitch[a] - f.pitch[b]) > 1.0 for a, b in itertools.combinations(c, 2))]
    if len(combos) > n_draws:
        combos = [combos[i] for i in rng.choice(len(combos), n_draws, replace=False)]
    return combos


def cmd_loso(args):
    from footy_track.scripts import ptz_validate as pv

    gt = _gt(args.ground)
    C0 = pp.GROUNDS[args.ground]["C0"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    recs = []
    for f in gt:
        train = [fr for fr in _rigframes(gt) if fr.shot != f.clip]
        rf = pp.fit_rig_robust(train, C0=C0)
        rig = rf.rig
        truth = f.truth()
        base = {"key": f.key, "n_gt": len(f.ids), "C": [float(c) for c in rig.C], "k1": float(rig.k1),
                "train_excluded": [e["key"] for e in rf.excluded]}
        for k in (2, 3):
            errs = []
            for c in _draws(f, k, 8, rng):
                cam = ptz.solve_points(f.img[list(c)], f.pitch[list(c)], rig)
                if cam is not None:
                    errs.append(pv.error_record(pv.cam_map(cam), truth)["med"])
            if errs:
                recs.append({**base, "method": f"pts{k}", "med": float(np.median(errs)), "all": errs})
        ev = pp.ptz.LineEvidence.from_masks(*pv_masks(f, out_dir / "masks", adaptive=not args.fixed_mask))
        t = time.time()
        cam, hyps, info = ptz.search_lines(ev, rig, margin=pp.MARGIN, min_f1=pp.MIN_SUPPORT)
        r = {**base, "method": "lines", "secs": time.time() - t, "support": info.get("f1"), "support_raw": info.get("support"),
             "margin": info.get("margin"), "reason": info.get("reason"), "abstain": cam is None}
        if hyps:
            r["best_med"] = pv.error_record(pv.cam_map(hyps[0].cam), truth)["med"]
        if cam is not None:
            r["med"] = r["best_med"]
        recs.append(r)
        print(f.key, {x["method"]: round(x["med"], 2) if x.get("med") is not None else x.get("reason") for x in recs
                      if x["key"] == f.key}, flush=True)
        (out_dir / "loso.json").write_text(json.dumps(recs, indent=1, default=float))
    summarise_loso(recs)


def pv_masks(f, cache: Path, adaptive: bool = True):
    """(lines, grass) masks of a GT frame, cached as npz."""
    from footy_track.calibration.video import read_frame
    from footy_track.scripts import ptz_validate as pv

    cp = cache / f"{f.clip}_{f.frame}.npz"
    if cp.exists():
        d = np.load(cp)
        return d["lines"], d["grass"]
    v = pv.find_clip_video(f.clip)
    lines, grass = pp.frame_masks(read_frame(v, f.frame), adaptive)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, lines=lines, grass=grass)
    return lines, grass


def summarise_loso(recs):
    for m in ("pts2", "pts3", "lines"):
        rs = [r for r in recs if r["method"] == m]
        ans = [r for r in rs if r.get("med") is not None]
        e = np.array([r["med"] for r in ans])
        if len(e):
            print(f"{m}: n={len(rs)} answered={len(ans)} median={np.median(e):.2f} p90={np.percentile(e, 90):.2f} "
                  f"max={e.max():.2f}")


# --------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------


def _targets(args, gt_dir: Path, clips_dirs: list[Path]):
    """-> [(clip, frame, llm correspondences, source tag)]."""
    from footy_track.calibration.frame_type import not_broadcast_frames
    from footy_track.calibration.prelabel.pipeline import (
        find_video,
        gt_frames,
        video_frame_count,
    )
    from footy_track.calibration.prelabel.proposals import (
        list_clips_with_proposals,
        read_proposals,
    )

    prefixes = pp.GROUNDS[args.ground]["prefixes"]
    clips = args.clips.split(",") if args.clips else [c for c in list_clips_with_proposals(gt_dir)
                                                      if c.startswith(prefixes)]
    only = {int(x) for x in args.frames.split(",")} if args.frames else None
    out = []
    for clip in clips:
        canon = canonical_clip_id(clip)
        nb = not_broadcast_frames(gt_dir, canon)
        gt = gt_frames(gt_dir, canon)
        runs = read_proposals(gt_dir, canon)["runs"]
        chosen: set[int] = set()
        if args.from_run and args.from_run in runs:
            for key, r in runs[args.from_run]["frames"].items():
                k = int(key)
                if r["status"] in ("failed", "rejected") and k not in nb and k not in gt:
                    # real labels only: earlier fits pad failed / proposed records with synthetic points
                    labels = [c for c in (r.get("correspondences") or []) if not c.get("synthetic")]
                    out.append((canon, k, labels, f"{args.from_run}:{r['status']}"))
                    chosen.add(k)
        if getattr(args, "frame_set_run", None) and args.frame_set_run in runs:
            # re-solve exactly the frames of an earlier run; labels: the original pre-label run's if it has the frame
            # (all of its real labels), else that earlier run's own real (consensus) labels
            label_frames = runs.get(args.label_run, {}).get("frames", {}) if getattr(args, "label_run", None) else {}
            for key, r in runs[args.frame_set_run]["frames"].items():
                k = int(key)
                if k in nb or k in gt:
                    continue
                src = label_frames.get(key) or r
                labels = [c for c in (src.get("correspondences") or []) if not c.get("synthetic")]
                out.append((canon, k, labels, f"{args.frame_set_run}:{r['status']}"))
                chosen.add(k)
        if args.extras and not gt and not nb:
            video = next((v for d in clips_dirs if d.is_dir() and (v := find_video(d, canon)) is not None), None)
            if video is not None:
                avoid = chosen | {int(k) for r in runs.values() for k in r["frames"]}
                for k in pp.evenly_spaced(video_frame_count(video), args.extras, avoid):
                    out.append((canon, k, [], "extra"))
    if only is not None:
        out = [t for t in out if t[1] in only]
    return out


def _render(bgr, cam, pitch, label: str):
    import cv2

    from footy_track.calibration.prelabel.pipeline import overlay_render

    dist = geo.make_distortion(cam.k1, aspect=ptz.ASPECT) if cam is not None else None
    img = overlay_render(bgr, cam.homography(), pitch, dist) if cam is not None else bgr.copy()
    if cam is None:
        from footy_track.calibration.prelabel.candidates import resize_to_work

        img = resize_to_work(bgr)[0].copy()
    cv2.rectangle(img, (0, img.shape[0] - 34), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, label[:140], (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def cmd_propose(args):
    import cv2

    from footy_track.calibration.prelabel.pipeline import find_video
    from footy_track.calibration.video import read_frame
    from footy_track.scripts import ptz_validate as pv

    gt_dir = Path(args.gt_dir or GT_ROOT)
    rig = ptz.Rig.from_json(args.rig)
    pitch = PitchModel(*rig.pitch)
    clips_dirs = [Path(p) for p in (args.clips_dirs.split(",") if args.clips_dirs else
                                    [str(d) for d in pv.CLIP_DIRS])]
    targets = _targets(args, gt_dir, clips_dirs)
    print(len(targets), "targets", flush=True)
    render_dir = Path(args.render_dir) if args.render_dir else None
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else None
    per_clip: dict[str, dict[int, dict]] = {}
    summary = []
    rec_path = render_dir / "records.json" if render_dir else None
    if rec_path is not None and rec_path.exists():  # resume: frames solved by an interrupted run are kept
        per_clip = {c: {int(k): v for k, v in fr.items()} for c, fr in json.loads(rec_path.read_text()).items()}
        sp = render_dir / "summary.json"
        summary = json.loads(sp.read_text()) if sp.exists() else []
    for clip, idx, llm, tag in targets:
        if idx in per_clip.get(clip, {}):
            continue
        video = next((v for d in clips_dirs if d.is_dir() and (v := find_video(d, clip)) is not None), None)
        bgr = read_frame(video, idx) if video is not None else None
        if bgr is None:
            print(f"[{clip}:{idx}] cannot read frame; skipped", flush=True)
            continue
        h, w = bgr.shape[:2]
        cp = cache / f"{clip}_{idx}.npz" if cache else None
        cached = None
        if cp is not None and cp.exists():
            try:
                d = np.load(cp)
                cached = (d["lines"], d["grass"])
            except Exception:  # noqa: BLE001 - a truncated cache file (disk full): recompute
                cached = None
        if cached is not None:
            lines, grass = cached
        else:
            lines, grass = pp.frame_masks(bgr)
            if cp is not None:
                cache.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cp, lines=lines, grass=grass)
        llm_xy = np.array([c["pitch_xy"] for c in llm], float).reshape(-1, 2)
        llm_uv = np.array([c["image"] for c in llm], float).reshape(-1, 2)
        out = pp.solve_frame(lines, grass, rig, llm_uv, llm_xy, p_main=pp.main_camera_p(bgr), pitch=pitch,
                             use_llm_points=not args.no_llm_points)
        extra = {"source_frame": tag}
        if out.cam is not None:
            rec = pp.proposal_record(idx, (w, h), pitch, out, llm, extra)
        else:
            rec = pp.failed_record(idx, (w, h), pitch, out, llm, extra)
        per_clip.setdefault(clip, {})[idx] = rec
        label = f"{clip[-14:]}:{idx} [{tag}] " + (f"{out.mode} support {out.support:.2f}" if out.cam is not None else
                                                  f"ABSTAIN {out.reason}")
        print(label, f"({out.secs:.0f}s)", flush=True)
        summary.append({"clip": clip, "frame": idx, "tag": tag, "status": rec["status"], "mode": out.mode,
                        "support": out.support, "margin": out.margin, "reason": out.reason,
                        "n_inliers": out.n_inliers, "n_labels": out.n_labels, "p_main": out.p_main,
                        "params": pp.cam_params(out.cam) if out.cam is not None else None})
        if render_dir:
            cv2.imwrite(str(render_dir / f"{clip}_{idx}.jpg"), _render(bgr, out.cam, pitch, label))
        if render_dir:  # the solved records, so a reviewed dry run can be published without solving again
            (render_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
            rec_path.write_text(json.dumps({c: {str(k): v for k, v in fr.items()} for c, fr in per_clip.items()},
                                           default=float))
    if args.dry_run:
        print("dry run: nothing written to the proposals store")
        return
    _publish(args, rig, gt_dir, per_clip)


def _publish(args, rig, gt_dir: Path, per_clip) -> None:
    params = {"rig": str(args.rig), "rig_C": [float(c) for c in rig.C], "rig_k1": float(rig.k1),
              "rig_roll_deg": float(np.degrees(rig.roll)), "pitch": list(rig.pitch), "min_support": pp.MIN_SUPPORT,
              "margin": pp.MARGIN, "from_run": args.from_run, "extras": args.extras, "ground": args.ground}
    pp.write_run(gt_dir, args.run_id, "ptz-rig", params, per_clip)
    print("wrote run", args.run_id, "to", gt_dir / "calibration_proposals", {c: len(v) for c, v in per_clip.items()})


def cmd_publish(args):
    """Add the records of a reviewed dry run (``<render-dir>/records.json``) to the proposals store as ``--run-id``."""
    gt_dir = Path(args.gt_dir or GT_ROOT)
    rig = ptz.Rig.from_json(args.rig)
    raw = json.loads((Path(args.render_dir) / "records.json").read_text())
    per_clip = {c: {int(k): v for k, v in fr.items()} for c, fr in raw.items()}
    for spec in args.veto or []:  # "<clip>:<frame>:<reason>": a proposal rejected on visual inspection -> failed record
        clip, frame, reason = spec.split(":", 2)
        old = per_clip[clip][int(frame)]
        real = [c for c in old["correspondences"] if not c.get("synthetic")]
        out = pp.Outcome(None, "abstain", f"vetoed on visual check ({reason})", (old.get("ptz") or {}).get("support"),
                         n_labels=len(real))
        per_clip[clip][int(frame)] = pp.failed_record(int(frame), tuple(old["image_size"]), PitchModel(*rig.pitch),
                                                      out, real, {"vetoed_params": old.get("ptz")})
        print("vetoed", clip, frame, reason)
    _publish(args, rig, gt_dir, per_clip)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rig", "loso", "propose", "publish"):
        s = sub.add_parser(name)
        s.add_argument("--ground", choices=sorted(pp.GROUNDS), default="vitality")
    r = sub.choices["rig"]
    r.add_argument("--rig-out")
    r.add_argument("--out")
    r.add_argument("--boot", type=int, default=0)
    r.add_argument("--dims", action="store_true", help="cost of the rig fit under other pitch sizes")
    r.add_argument("--point-px", type=float, default=20.0)
    r.add_argument("--frame-px", type=float, default=20.0)
    r.add_argument("--k1-lines", action="store_true", help="per-frame k1 from the white-line mask (C fixed)")
    r.add_argument("--mask-cache")
    lo = sub.choices["loso"]
    lo.add_argument("--out", required=True)
    lo.add_argument("--fixed-mask", action="store_true", help="the V1 line mask (absolute saturation limit)")
    pr = sub.choices["propose"]
    pr.add_argument("--rig", "--rig-in", dest="rig", required=True)
    pr.add_argument("--run-id", required=True)
    pr.add_argument("--clips")
    pr.add_argument("--frames")
    pr.add_argument("--from-run", help="take this run's failed / rejected records (and their LLM points) as targets")
    pr.add_argument("--frame-set-run", help="solve exactly the frames of this earlier run (any status), e.g. ptz_bmth1")
    pr.add_argument("--label-run", help="with --frame-set-run: take the LLM labels from this run's record of the frame "
                                        "when it has one (e.g. bmth_pre1)")
    pr.add_argument("--extras", type=int, default=0, help="evenly spaced frames per clip without GT / not-broadcast")
    pr.add_argument("--gt-dir")
    pr.add_argument("--clips-dirs")
    pr.add_argument("--render-dir")
    pr.add_argument("--cache", help="directory to cache the white-line / grass masks")
    pr.add_argument("--no-llm-points", action="store_true",
                    help="ignore the LLM-labelled points: lines-only proposals (default: points allowed, gated on "
                         "the same line support as lines-only)")
    pr.add_argument("--dry-run", action="store_true")
    pb = sub.choices["publish"]
    for a in ("--rig", "--run-id", "--render-dir"):
        pb.add_argument(a, required=True)
    pb.add_argument("--gt-dir")
    pb.add_argument("--veto", action="append", help="<clip>:<frame>:<reason> proposal to publish as failed")
    pb.add_argument("--from-run")
    pb.add_argument("--extras", type=int, default=0)
    args = ap.parse_args(argv)
    {"rig": cmd_rig, "loso": cmd_loso, "propose": cmd_propose, "publish": cmd_publish}[args.cmd](args)


if __name__ == "__main__":
    main()
