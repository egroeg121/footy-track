"""The Claude + detector auto-labelling loop, labelling mode (one round, step by step).

    # 0. pick clips (hard-fails on any evaluation clip) -> R/clips.json
    python -m footy_track.scripts.autolabel_loop pool --run R --clips-dir V \
        --split split.json --gt-dir G --eval-dirs E --random 3
    # (GPU) detector candidate cache for R/clips.json:
    #   python -m footy_track.scripts.ball_tracking_eval cache --videos V --gt G \
    #       --out CACHE --weights best.pt --clip <ids>
    # 1. tracker, camera motion, static filter, sampled frames -> questions + images
    python -m footy_track.scripts.autolabel_loop prepare --run R --cache CACHE \
        --budget 300
    # 2. ask Claude (resumable, per-run budget, exit 3 on auth or usage limit)
    python -m footy_track.scripts.claude_verify_v2 ask --run R --model opus --budget 300
    # 3. decision rule, objects, blind review list, circuit breaker
    python -m footy_track.scripts.autolabel_loop decide --run R --answers R/answers_*.json
    #    FOOTY_BALL_CHECK_LIST=R/review_list.json -> /ball_check?source=list
    # 4. after the review: apply the verdicts, write labels, update the round report
    python -m footy_track.scripts.autolabel_loop ingest --run R --verdicts-dir G2/ball_checks \
        --labels-dir L

Round extras: ``prepare`` also draws the round's random no-candidate frames
(``random_frames.json``, ``--random-frames``, ``add-random`` to add them to a prepared run);
``decide`` samples every gate (``--audit-rate/--audit-min``, settled rates after 3 passing
rounds, history in ``<rounds dir>/gate_state.json``) and ``ingest`` reports each gate's audit
precision with a Wilson interval and the miss rate of the random frames. Sparse
eval labelling of NEW eval clips: ``eval-register`` (registry, so the loop and the
training export refuse the clip), ``eval-list`` (one tap-to-place card per second of
video), ``eval-finalise`` (write the no-ball frames to the GT marks).

Other commands: ``static-eval`` (measure the static filter on a calibration run) and
``legacy-import`` (a person's earlier re-check verdicts on the held-out disagreement
cards, kept as evaluation corrections only). See ``docs/design/claude_autolabel_loop.md``,
section 12.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from footy_track import eval_registry
from footy_track.autolabel import loop
from footy_track.autolabel.loop import StaticParams, Thresholds
from footy_track.clip_id import canonical_clip_id
from footy_track.labeller.constants import card_box_index

REDETECT_CONF = 0.05
REDETECT_IMGSZ = 960


def _atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _eval_ids(args) -> set[str]:
    return loop.eval_clip_ids(
        args.split,
        gt_dir=args.gt_dir,
        eval_dirs=args.eval_dirs or [],
        cache_dir=args.eval_cache_dir,
        registry=args.eval_registry,
    )


def _thr(args) -> Thresholds:
    return Thresholds(
        auto_conf=args.auto_conf,
        drop_conf=args.drop_conf,
        audit_rate=args.audit_rate,
        audit_min=args.audit_min,
        settled_rate=args.settled_rate,
        settled_min=args.settled_min,
        settled_rounds=args.settled_rounds,
        drop_precision=args.drop_precision,
        random_frames=args.random_frames,
        control_rate=args.control_rate,
        breaker_precision=args.breaker_precision,
    )


def _round_thr(rep: dict, args) -> Thresholds:
    """The thresholds a round was decided with (its report is authoritative: the cards
    depend on them). A report from before the gate audits has no ``scheme``: scheme 1."""
    stored = dict(rep.get("thresholds") or {})
    if not stored:
        return _thr(args)
    fields = set(Thresholds.__dataclass_fields__)
    kw = {k: v for k, v in stored.items() if k in fields}
    kw["scheme"] = stored.get("scheme", 1)
    return Thresholds(**kw)


def _state_path(rounds_dir: Path | None, run: Path) -> Path:
    return (rounds_dir or run.parent) / "gate_state.json"


def _read_random_frames(run: Path) -> list[dict]:
    p = run / "random_frames.json"
    return json.loads(p.read_text())["frames"] if p.exists() else []


# ----------------------------------------------------------------------------- pool


def cmd_pool(args) -> int:
    eval_ids = _eval_ids(args)
    existing = args.run / "clips.json"
    if existing.exists() and not args.force:  # idempotent: keep the round's pool
        clips = json.loads(existing.read_text())["clips"]
        loop.assert_not_eval(clips, eval_ids)
        print(f"pool already chosen: {clips}")
        return 0
    exclude = set()
    if args.rounds_dir and args.rounds_dir.is_dir():
        for p in args.rounds_dir.glob("*/clips.json"):
            if p.parent != args.run:
                exclude |= set(json.loads(p.read_text())["clips"])
    clips = loop.pick_clips(
        args.clips_dir,
        eval_ids,
        names=args.clips or [],
        glob=args.glob,
        n_random=args.random,
        seed=args.seed,
        exclude=exclude,
    )
    if not clips:
        print("no clips selected", file=sys.stderr)
        return 2
    videos = loop.list_videos(args.clips_dir)
    _atomic(
        args.run / "clips.json",
        {
            "clips": clips,
            "videos": {c: str(videos[c]) for c in clips},
            "seed": args.seed,
            "n_eval_ids": len(eval_ids),
            "created_at": time.time(),
        },
    )
    print(f"{len(clips)} clips: {clips}")
    return 0


def cmd_check(args) -> int:
    loop.assert_not_eval(args.clips, _eval_ids(args))
    print("ok: no evaluation clip")
    return 0


# ------------------------------------------------------------------------- prepare


def apply_budget(
    items: list[dict], budget: int, tracker_share: float = 0.6
) -> list[dict]:
    """Keep at most ``budget`` asked items: up to ``tracker_share`` of the budget for
    tracker candidates (the ones that can be auto-accepted), the rest for the others;
    leftover room goes to whichever group has more. Order inside a group is a hash of
    the id, so every clip and frame is spread evenly. Filtered items are not asked."""
    asked = [i for i in items if i["status"] == "asked"]
    if not budget or len(asked) <= budget:
        return items
    key = lambda i: hashlib.sha1(i["id"].encode()).hexdigest()  # noqa: E731
    trk = sorted((i for i in asked if i["source"] == "tracker"), key=key)
    oth = sorted((i for i in asked if i["source"] != "tracker"), key=key)
    n_t = min(len(trk), max(round(budget * tracker_share), budget - len(oth)))
    keep = {i["id"] for i in trk[:n_t]} | {i["id"] for i in oth[: budget - n_t]}
    out = []
    for i in items:
        if i["status"] == "asked" and i["id"] not in keep:
            out.append({**i, "status": "over_budget"})
        else:
            out.append(i)
    return out


GAP_NEAR_PX = 25.0
GAP_BOX_PX = 12.0


def gap_items(
    clip: str,
    frames: list[int],
    cands: dict[int, list[tuple[float, float]]],
    path: dict[int, tuple[float, float]],
    det: dict,
    near_px: float = GAP_NEAR_PX,
) -> list[dict]:
    """Missed-ball candidates: sampled frames where the tracker's path has a position
    (interpolated, no detection chosen) but no cached detector candidate lies within
    ``near_px`` of it. Not asked to Claude (nothing to ask about): they become review
    cards that show the predicted position, so a person can drag the ball onto it or
    say no ball is visible. Detector recall is the ceiling of the whole loop."""
    out = []
    for f in frames:
        if f not in path or f in det:
            continue
        x, y = path[f]
        if any(math.hypot(x - u, y - v) <= near_px for u, v in cands.get(f, [])):
            continue
        out.append(
            {
                "id": f"{clip}_{f:06d}_g",
                "clip": clip,
                "frame": f,
                "cx": x,
                "cy": y,
                "w": GAP_BOX_PX,
                "h": GAP_BOX_PX,
                "det_conf": 0.0,
                "source": "gap",
                "track_id": None,
                "status": "gap",
            }
        )
    return out


def _tracker_paths(cache):
    from footy_track.ball_eval.detection_replay import replay  # noqa: PLC0415
    from footy_track.ball_trackers.detection_tracker import (  # noqa: PLC0415
        DETECTED,
        INTERPOLATED,
    )
    from footy_track.ball_trackers.factory import make_ball_tracker  # noqa: PLC0415

    size = cache.frame_size
    det, path = {}, {}
    for p in replay(make_ball_tracker("bidirectional", size), cache):
        if p.x is None or p.source not in (DETECTED, INTERPOLATED):
            continue
        c = ((p.x + p.w / 2) * size[0], (p.y + p.h / 2) * size[1])
        path[p.frame_index] = c
        if p.source == DETECTED:
            det[p.frame_index] = (c, p.track_id)
    return path, det


def cmd_add_gaps(args) -> int:
    """(Re)compute the missed-ball items of a prepared run (idempotent)."""
    from footy_track.ball_eval.detection_replay import read_cache  # noqa: PLC0415

    run = args.run
    spec = json.loads((run / "clips.json").read_text())
    loop.assert_not_eval(spec["clips"], _eval_ids(args))
    items = [
        i for i in json.loads((run / "items.json").read_text()) if i["status"] != "gap"
    ]
    n = 0
    for clip in spec["clips"]:
        cache = read_cache(args.cache / f"{clip}.jsonl")
        path, det = _tracker_paths(cache)
        w, h = cache.frame_size
        cands = {
            f: [((c.x + c.w / 2) * w, (c.y + c.h / 2) * h) for c in cs]
            for f, cs in cache.frames
        }
        frames = sorted(f for f in cands if f % args.stride == 0)
        g = gap_items(clip, frames, cands, path, det)
        items += g
        n += len(g)
    _atomic(run / "items.json", items)
    print(f"{n} missed-ball (gap) frames")
    return 0


def _cache_cands(cache) -> dict[int, list[list[float]]]:
    """Every cached candidate by frame: ``[cx, cy, conf]`` in native px."""
    w, h = cache.frame_size
    return {
        f: [[(c.x + c.w / 2) * w, (c.y + c.h / 2) * h, c.confidence] for c in cs]
        for f, cs in cache.frames
    }


def write_random_frames(
    run: Path, cache_dir: Path, stride: int, n: int, seed: int
) -> int:
    """Draw the round's random no-candidate frames into ``random_frames.json`` (kept as
    it is when it exists: the cards and their ids must not change under a reviewer).
    Each record keeps every cached candidate of the frame, so the report can tell a ball
    the loop never selected from one the detector never saw."""
    from footy_track.ball_eval.detection_replay import read_cache  # noqa: PLC0415

    out = run / "random_frames.json"
    if out.exists():
        return len(json.loads(out.read_text())["frames"])
    spec = json.loads((run / "clips.json").read_text())
    cands = {}
    for clip in spec["clips"]:
        cands[clip] = _cache_cands(read_cache(cache_dir / f"{clip}.jsonl"))
    frames = {c: [f for f in cs if f % stride == 0] for c, cs in cands.items()}
    picked = loop.pick_random_frames(frames, n, round_id=run.name, seed=seed)
    rows = [
        {
            "id": f"{c}_{f:06d}_r",
            "clip": c,
            "frame": f,
            "cands": [
                [round(x, 1), round(y, 1), round(q, 3)] for x, y, q in cands[c][f]
            ],
        }
        for c, f in picked
    ]
    _atomic(out, {"frames": rows, "stride": stride, "seed": seed})
    return len(rows)


def cmd_add_random(args) -> int:
    """(Re)draw the random no-candidate frames of a prepared run (kept if present)."""
    loop.assert_not_eval(
        json.loads((args.run / "clips.json").read_text())["clips"], _eval_ids(args)
    )
    n = write_random_frames(
        args.run, args.cache, args.stride, args.random_frames, args.random_seed
    )
    print(f"{n} random no-candidate frames")
    return 0


def cmd_prepare(args) -> int:  # noqa: PLR0912, PLR0915
    import cv2  # noqa: PLC0415

    from footy_track.autolabel.camera import estimate_camera  # noqa: PLC0415
    from footy_track.autolabel.render_v2 import render_question  # noqa: PLC0415
    from footy_track.autolabel.verify_pilot import Cand  # noqa: PLC0415
    from footy_track.autolabel.verify_v2 import (  # noqa: PLC0415
        question_id,
        select_candidates,
    )
    from footy_track.ball_eval.detection_replay import (  # noqa: PLC0415
        read_cache,
        replay,
    )
    from footy_track.ball_trackers.detection_tracker import (  # noqa: PLC0415
        DETECTED,
        INTERPOLATED,
    )
    from footy_track.ball_trackers.factory import make_ball_tracker  # noqa: PLC0415

    run = args.run
    done = run / "prepare.done.json"
    spec = json.loads((run / "clips.json").read_text())
    clips = spec["clips"]
    loop.assert_not_eval(clips, _eval_ids(args))  # again: never trust the caller
    sp = StaticParams(
        args.static_radius, args.static_frames, args.static_window, args.static_cam
    )
    phash = hashlib.sha256(
        json.dumps(
            [
                clips,
                args.stride,
                args.budget,
                args.top_others,
                args.no_static,
                asdict(sp),
            ],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:16]
    if done.exists() and not args.force:
        prev = json.loads(done.read_text())
        if prev.get("params") != phash:
            print(
                f"{run} was prepared with other parameters; use --force",
                file=sys.stderr,
            )
            return 2
        print(f"prepare already done: {prev['n_questions']} questions; skipping")
        write_random_frames(
            run, args.cache, args.stride, args.random_frames, args.random_seed
        )
        return 0
    done.unlink(missing_ok=True)

    items: list[dict] = []
    sizes: dict[str, list[int]] = {}
    per_clip = {}
    for clip in clips:
        cache = read_cache(args.cache / f"{clip}.jsonl")
        size = cache.frame_size
        sizes[clip] = list(size)
        video = Path(spec["videos"][clip])
        cached = dict(cache.frames)
        pts = replay(make_ball_tracker("bidirectional", size), cache)
        det: dict[int, tuple[tuple[float, float], int | None]] = {}
        path: dict[int, tuple[float, float]] = {}
        for p in pts:
            if p.x is None or p.source not in (DETECTED, INTERPOLATED):
                continue
            c = ((p.x + p.w / 2) * size[0], (p.y + p.h / 2) * size[1])
            path[p.frame_index] = c
            if p.source == DETECTED:
                det[p.frame_index] = (c, p.track_id)

        def px(cs, size=size):
            return [
                Cand(
                    (c.x + c.w / 2) * size[0],
                    (c.y + c.h / 2) * size[1],
                    c.w * size[0],
                    c.h * size[1],
                    c.confidence,
                    "orig",
                )
                for c in cs
            ]

        flagged: set[tuple[int, int]] = set()
        cam_file = run / "camera" / f"{clip}.json"
        if not args.no_static:
            if cam_file.exists():
                cam = {
                    int(k): tuple(v)
                    for k, v in json.loads(cam_file.read_text()).items()
                }
            else:
                cam = estimate_camera(video)
                _atomic(cam_file, {str(k): list(v) for k, v in cam.items()})
            all_c = {f: [(c.cx, c.cy) for c in px(cs)] for f, cs in cached.items()}
            flagged = loop.static_candidates(all_c, cam, sp)
        frames = sorted(f for f in cached if f % args.stride == 0)
        wanted: dict[int, list[dict]] = {}
        n_cached = 0
        for f in frames:
            full = px(cached[f])
            n_cached += len(full)
            ok = [c for i, c in enumerate(full) if (f, i) not in flagged]
            stat_pos = {(c.cx, c.cy) for i, c in enumerate(full) if (f, i) in flagged}
            tp = det.get(f, (None, None))
            sel_full = select_candidates(
                full, tp[0], (), top_others=args.top_others, max_redetect=0
            )
            sel = select_candidates(
                ok, tp[0], (), top_others=args.top_others, max_redetect=0
            )
            static_sel = [c for c in sel_full if (c.cx, c.cy) in stat_pos]
            rows = []
            for status, cs in (("asked", sel), ("filtered", static_sel)):
                for k, c in enumerate(cs):
                    rows.append(
                        {
                            "id": question_id(
                                clip, f, k if status == "asked" else 90 + k
                            ),
                            "clip": clip,
                            "frame": f,
                            "cx": c.cx,
                            "cy": c.cy,
                            "w": c.w,
                            "h": c.h,
                            "det_conf": c.conf,
                            "source": c.source,
                            "track_id": tp[1] if c.source == "tracker" else None,
                            "status": status,
                        }
                    )
            wanted[f] = rows
            items.extend(rows)
        items.extend(
            gap_items(
                clip,
                frames,
                {f: [(c.cx, c.cy) for c in px(cached[f])] for f in frames},
                path,
                det,
            )
        )
        per_clip[clip] = {"frames": len(frames), "cached_candidates": n_cached}
        print(
            f"{clip}: {len(frames)} frames, {sum(len(v) for v in wanted.values())} candidates",
            flush=True,
        )

    items = apply_budget(items, args.budget)
    by_clip: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        if it["status"] == "asked":
            by_clip[it["clip"]][it["frame"]].append(it)
    questions = []
    for clip, fmap in sorted(by_clip.items()):
        cap = cv2.VideoCapture(spec["videos"][clip])
        idx, grabbed = 0, {}
        while len(grabbed) < len(fmap) and idx <= max(fmap):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in fmap:
                grabbed[idx] = frame
            idx += 1
        cap.release()
        for f, rows in sorted(fmap.items()):
            if f not in grabbed:
                print(f"{clip}: cannot read frame {f}", file=sys.stderr)
                for r in rows:
                    r["status"] = "unreadable"
                continue
            for r in rows:
                from footy_track.autolabel.verify_v2 import Candidate  # noqa: PLC0415

                c = Candidate(
                    r["cx"], r["cy"], r["w"], r["h"], r["det_conf"], r["source"]
                )
                img = run / "images" / f"{r['id']}.png"
                if not img.exists():
                    render_question(grabbed[f], c, img)
                questions.append(
                    {
                        "id": r["id"],
                        "fkey": f"{clip}|{f}",
                        "clip": clip,
                        "frame": f,
                        "split": "new",
                        "stratum": "new",
                        "k": 0,
                        "image": img.name,
                        "patch": [round(c.cx - 64), round(c.cy - 64)],
                        "cand": {
                            "cx": c.cx,
                            "cy": c.cy,
                            "w": c.w,
                            "h": c.h,
                            "conf": c.conf,
                            "source": c.source,
                        },
                    }
                )
    questions.sort(key=lambda q: hashlib.sha256(q["id"].encode()).hexdigest())
    _atomic(run / "questions.json", questions)
    _atomic(run / "items.json", items)
    _atomic(run / "sizes.json", sizes)
    stat = defaultdict(int)
    for it in items:
        stat[it["status"]] += 1
    src = defaultdict(int)
    for q in questions:
        src[q["cand"]["source"]] += 1
    _atomic(
        done,
        {
            "params": phash,
            "n_questions": len(questions),
            "by_source": dict(src),
            "by_status": dict(stat),
            "per_clip": per_clip,
            "n_frames": sum(v["frames"] for v in per_clip.values()),
        },
    )
    print(f"{len(questions)} questions {dict(src)}; items by status {dict(stat)}")
    n_rand = write_random_frames(
        run, args.cache, args.stride, args.random_frames, args.random_seed
    )
    print(f"{n_rand} random no-candidate frames")
    return 0


# --------------------------------------------------------------------------- decide


def _load_answers(paths) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        out |= json.loads(Path(p).read_text())
    return out


def _previous_reports(rounds_dir: Path | None, run: Path) -> list[dict]:
    reps = []
    if rounds_dir and rounds_dir.is_dir():
        for p in rounds_dir.glob("*/round_report.json"):
            if p.parent.resolve() != run.resolve():
                r = json.loads(p.read_text())
                if r.get("created_at"):
                    reps.append(r)
    return sorted(reps, key=lambda r: r["created_at"])


def _stems(spec: dict) -> dict[str, str]:
    return {c: Path(v).stem for c, v in spec["videos"].items()}


def cmd_decide(args) -> int:
    run = args.run
    thr = _thr(args)
    items = json.loads((run / "items.json").read_text())
    answers = _load_answers(args.answers)
    spec = json.loads((run / "clips.json").read_text())
    sizes = json.loads((run / "sizes.json").read_text())
    for it in items:
        a = answers.get(it["id"])
        it["verdict"] = None if a is None else a["verdict"]
        it["claude_conf"] = None if a is None else a["confidence"]
    rounds_dir = args.rounds_dir or run.parent
    open_, why = loop.breaker_state(_previous_reports(rounds_dir, run), thr)
    if args.force_breaker:
        open_, why = True, "forced by --force-breaker"
    state = loop.load_gate_state(_state_path(args.rounds_dir, run))
    gate_plan = loop.gate_params(thr, state, exclude_round=run.name)
    random_frames = _read_random_frames(run)
    built = loop.build_round(
        items,
        round_id=run.name,
        sizes=sizes,
        stems=_stems(spec),
        thr=thr,
        breaker_open=open_,
        gate_plan=gate_plan,
        random_frames=random_frames,
    )
    _atomic(run / "review_list.json", {"version": 2, "items": built["cards"]})
    _atomic(run / "cards_meta.json", built["meta"])
    _atomic(run / "objects.json", built["objects"])
    _atomic(run / "decisions.json", built["decisions"])
    old = {}
    rp = run / "round_report.json"
    if rp.exists():
        old = json.loads(rp.read_text())
    report = {
        "round": run.name,
        "created_at": old.get("created_at") or time.time(),
        "thresholds": loop.asdict_thr(thr),
        "breaker": {"open": open_, "why": why},
        "gates_plan": gate_plan,
        "random_frames": len(random_frames),
        "clips": len(spec["clips"]),
        "frames": json.loads((run / "prepare.done.json").read_text())["n_frames"],
        "counts": built["counts"],
    }
    _atomic(rp, report)
    _atomic_text(run / "round_report.md", render_report(report))
    print(json.dumps(report["counts"], indent=1))
    print("breaker:", report["breaker"])
    print(f"{len(built['cards'])} cards -> {run / 'review_list.json'}")
    return 0


def _pct(p) -> str:
    return "-" if p is None else f"{100 * p:.1f}%"


def _ci(g: dict, key: str = "ci95") -> str:
    ci = g.get(key)
    return "-" if not ci else f"{100 * ci[0]:.0f}-{100 * ci[1]:.0f}%"


def render_report(rep: dict) -> str:
    c = rep["counts"]
    lines = [
        f"# Round {rep['round']}",
        "",
        f"- clips {rep['clips']}, frames sampled {rep['frames']}",
        f"- candidates {c['candidates']}, dropped by the static filter {c['filtered_static']}, "
        f"left unasked by the budget {c.get('over_budget', 0)}",
        f"- Claude questions answered {c['answered']} (asked {c['asked']})",
        f"- auto-accepted candidates {c['auto_candidates']}, to the queue {c['review_candidates']}, "
        f"dropped {c['dropped_candidates']}",
        f"- objects {c['objects']} {c['objects_by_class']}",
        f"- review cards {c['cards']} by role {c['cards_by_role']}",
        f"- circuit breaker: {'OPEN' if rep['breaker']['open'] else 'closed'} ({rep['breaker']['why']})",
    ]
    by_gate = c.get("cards_by_gate") or {}
    if by_gate:
        lines += ["", "## Gate audits (blind random samples)", ""]
        lines += [
            "| gate | objects | cards | rate | floor | settled |",
            "|---|---|---|---|---|---|",
        ]
        for g, v in by_gate.items():
            lines.append(
                f"| {g} | {v['objects']} | {v['cards']} | {v.get('rate')} | {v.get('min')} "
                f"| {v.get('settled')} |"
            )
        if c.get("random_cards"):
            lines.append(f"\nRandom no-candidate frames: {c['random_cards']} cards.")
    a = rep.get("audit")
    if a:
        lines += [
            "",
            "## After review",
            f"- audit of auto-accepted: {a['ball']} ball, {a['not_ball']} not ball, "
            f"{a['unsure']} unsure; precision by objects {a['precision_objects']}, "
            f"by frames {a['precision_frames']} (this one drives the circuit breaker)",
            f"- controls (dropped candidates a person checked): {rep.get('controls')}",
            f"- labels written: {rep.get('ingest', {}).get('counts')}",
        ]
    gates = rep.get("gates")
    if gates:
        lines += [
            "",
            "### Precision per gate (report only, except auto-accept)",
            "",
            "| gate | audited | ball | not ball | unsure | precision | Wilson 95% | "
            "ball rate | threshold | passed |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for g, v in gates.items():
            lines.append(
                f"| {g} | {v['audited']}/{v['objects']} | {v['ball']} | {v['not_ball']} | "
                f"{v['unsure']} | {_pct(v['precision'])} ({v['correct']}/{v['decided']}) | "
                f"{_ci(v)} | {_pct(v['ball_rate'])} | {v['threshold']} | {v['passed']} |"
            )
        lines.append(
            "\nPrecision: for auto-accept the share that is the ball, for the other "
            "gates the share that is not (ball rate = what the gate wrongly lost)."
        )
    r = rep.get("random")
    if r:
        lines += [
            "",
            "### Random no-candidate frames (does the system miss balls?)",
            "",
            f"- {r['frames']} frames: ball visible {r['ball_visible']}, no ball {r['no_ball']}, "
            f"unsure {r['unsure']}, not answered {r['pending']}",
            f"- of the visible balls: found by a candidate the loop asked about "
            f"{r['found_selected']}, cached but never selected {r['found_not_selected']}, "
            f"no candidate at all {r['missed']}",
            f"- miss rate (no candidate) {_pct(r['miss_rate'])} (Wilson 95% {_ci(r, 'miss_ci95')}); "
            f"never examined (no candidate or not selected) {_pct(r['unexamined_rate'])} "
            f"({_ci(r, 'unexamined_ci95')})",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- ingest


def read_card_verdicts(verdicts_dir: Path, cards: list[dict]) -> dict[str, dict]:
    """Latest live verdict per card id, from ``<dir>/<clip stem>.jsonl``."""
    from footy_track.feature_store.ball_checks import (  # noqa: PLC0415
        read_latest_verdicts,
    )

    out: dict[str, dict] = {}
    cache: dict[str, dict] = {}
    for c in cards:
        if c["clip"] not in cache:
            p = verdicts_dir / f"{c['clip']}.jsonl"
            cache[c["clip"]] = read_latest_verdicts(p) if p.exists() else {}
        rec = cache[c["clip"]].get(("list", c["frame_index"], card_box_index(c["id"])))
        if rec is not None:
            out[c["id"]] = rec
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_text(path, "".join(json.dumps(r) + "\n" for r in rows))


def rebuild_labels(
    rounds_dir: Path, labels_dir: Path, backup_dir: Path
) -> dict[str, int]:
    """Rebuild ``labels_dir/<clip>.jsonl`` and ``labels_dir/ball_checks/<clip>.jsonl``
    from every round's own files (merged, deduped, strongest tier wins). A file that
    would change is first copied to ``backup_dir``. Returns lines per clip."""
    lines: dict[str, list[dict]] = defaultdict(list)
    bcs: dict[str, list[dict]] = defaultdict(list)
    for rd in sorted(rounds_dir.glob("*/labels")):
        for p in sorted(rd.glob("*.jsonl")):
            lines[p.stem] += [
                json.loads(x) for x in p.read_text().splitlines() if x.strip()
            ]
    for rd in sorted(rounds_dir.glob("*/ball_checks_propagated")):
        for p in sorted(rd.glob("*.jsonl")):
            bcs[p.stem] += [
                json.loads(x) for x in p.read_text().splitlines() if x.strip()
            ]
    ts = time.strftime("%Y%m%dT%H%M%S")
    n = {}
    for clip, ls in lines.items():
        merged = loop.merge_label_lines(ls)
        n[clip] = len(merged)
        _replace_with_backup(labels_dir / f"{clip}.jsonl", merged, backup_dir, ts)
    for clip, rs in bcs.items():
        seen, uniq = set(), []
        for r in rs:
            k = (r["frame_index"], r["box_index"], r["verdict"])
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        _replace_with_backup(
            labels_dir / "ball_checks" / f"{clip}.jsonl", uniq, backup_dir, ts
        )
    return n


def _replace_with_backup(
    path: Path, rows: list[dict], backup_dir: Path, ts: str
) -> None:
    new = "".join(json.dumps(r) + "\n" for r in rows)
    if path.exists():
        if path.read_text() == new:
            return
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / f"{path.parent.name}.{path.stem}.{ts}.jsonl")
    _atomic_text(path, new)


def cmd_ingest(args) -> int:
    run = args.run
    rep = json.loads((run / "round_report.json").read_text())
    thr = _round_thr(rep, args)
    items = json.loads((run / "items.json").read_text())
    answers = _load_answers(args.answers) if args.answers else {}
    for it in items:
        a = answers.get(it["id"])
        it["verdict"] = it.get("verdict") if a is None else a["verdict"]
        it["claude_conf"] = it.get("claude_conf") if a is None else a["confidence"]
    # the decision is recomputed from the same inputs: build_round is deterministic
    spec = json.loads((run / "clips.json").read_text())
    sizes = json.loads((run / "sizes.json").read_text())
    random_frames = _read_random_frames(run)
    built = loop.build_round(
        items,
        round_id=run.name,
        sizes=sizes,
        stems=_stems(spec),
        thr=thr,
        breaker_open=rep["breaker"]["open"],
        gate_plan=rep.get("gates_plan"),
        random_frames=random_frames,
    )
    stored = json.loads((run / "cards_meta.json").read_text())
    if stored != built["meta"]:
        print(
            "cards_meta.json differs from the recomputed round: pass the same "
            "--answers and thresholds as `decide`",
            file=sys.stderr,
        )
        return 2
    verdicts = read_card_verdicts(args.verdicts_dir, built["cards"])
    plan = loop.plan_ingest(
        items,
        built,
        verdicts,
        sizes=sizes,
        eval_ids=_eval_ids(args),
        thr=thr,
        random_frames=random_frames,
    )
    for clip in spec["clips"]:
        _write_jsonl(run / "labels" / f"{clip}.jsonl", plan.labels.get(clip, []))
    bc: dict[str, list[dict]] = defaultdict(list)
    for i, neg in enumerate(plan.negatives):
        bc[neg["clip"]].append(
            {
                "clip": neg["clip"],
                "frame_index": neg["frame_index"],
                "box_index": neg["frame_index"] * 100 + i % 100,
                "source": "cand",
                "verdict": "not_ball",
                "bbox": neg["bbox"],
                "label": loop.BALL_LABEL,
                "provenance": "autolabel_loop",
                "confidence": None,
            }
        )
    for clip, ls in plan.labels.items():
        for i, ln in enumerate(ls):
            if loop.tier(ln["tags"]) < 3:
                bc[clip].append(
                    {
                        "clip": clip,
                        "frame_index": ln["frame_index"],
                        "box_index": ln["frame_index"] * 100 + 50 + i % 50,
                        "source": "cand",
                        "verdict": "ball",
                        "bbox": ln["bbox"],
                        "label": loop.BALL_LABEL,
                        "provenance": "autolabel_loop",
                        "confidence": None,
                    }
                )
    for clip in spec["clips"]:
        _write_jsonl(run / "ball_checks_propagated" / f"{clip}.jsonl", bc.get(clip, []))
    _write_jsonl(run / "hard_negatives.jsonl", plan.negatives)
    n = rebuild_labels(
        args.rounds_dir or run.parent, args.labels_dir, args.labels_dir / ".backups"
    )
    rep["audit"] = plan.audit
    rep["controls"] = plan.controls
    rep["gates"] = plan.gates
    rep["random"] = plan.random
    if thr.scheme >= 2:  # noqa: PLR2004 - the gate history belongs to the new sampling
        loop.update_gate_state(
            _state_path(args.rounds_dir, run), run.name, plan.gates, rep["created_at"]
        )
    rep["ingest"] = {
        "counts": plan.counts,
        "inconsistent_objects": plan.inconsistent,
        "object_verdicts": len(plan.object_verdicts),
        "cards_with_verdict": len(verdicts),
        "cards": len(built["cards"]),
        "labels_dir_lines": n,
        "at": time.time(),
    }
    _atomic(run / "round_report.json", rep)
    _atomic_text(
        run / "round_report.md", render_report({**rep, "counts": built["counts"]})
    )
    _atomic(
        run / "ingest_plan.json",
        {
            "counts": plan.counts,
            "audit": plan.audit,
            "controls": plan.controls,
            "gates": plan.gates,
            "random": plan.random,
        },
    )
    nxt, why = loop.breaker_state(
        [*_previous_reports(args.rounds_dir or run.parent, run), rep], thr
    )
    print(
        json.dumps(
            {
                "audit": plan.audit,
                "controls": plan.controls,
                "gates": plan.gates,
                "random": plan.random,
                "counts": plan.counts,
            },
            indent=1,
        )
    )
    print(f"circuit breaker for the next round: {'OPEN' if nxt else 'closed'} ({why})")
    return 0


# ------------------------------------------------------- sparse eval labelling (12.9)


def _registry_path(args) -> Path:
    path = args.registry or eval_registry.default_path()
    if path is None:
        raise SystemExit("no registry: pass --registry or set FOOTY_EVAL_REGISTRY")
    return path


def cmd_eval_register(args) -> int:
    """Register NEW clips as evaluation clips. From this moment the loop (pool, check,
    prepare, ingest) and the training export (``--eval-registry``) refuse them. A clip an
    earlier round used is refused here: the loop may have labelled it already."""
    videos = loop.list_videos(args.clips_dir)
    ids = [canonical_clip_id(c) for c in args.clips]
    if args.glob:
        ids += [
            canonical_clip_id(p.name)
            for p in sorted(args.clips_dir.glob(args.glob))
            if p.suffix.lower() in loop.VIDEO_SUFFIXES
        ]
    missing = [c for c in ids if c not in videos]
    if missing:
        print(f"no video for {missing} in {args.clips_dir}", file=sys.stderr)
        return 2
    used: set[str] = set()
    if args.rounds_dir and args.rounds_dir.is_dir():
        for q in args.rounds_dir.glob("*/clips.json"):
            used |= {canonical_clip_id(c) for c in json.loads(q.read_text())["clips"]}
    bad = sorted(set(ids) & used)
    if bad:
        print(
            f"refusing to register {bad}: used by an earlier loop round (labelled)",
            file=sys.stderr,
        )
        return 5
    added = eval_registry.register_clips(
        _registry_path(args), ids, mode="sparse", note=args.note
    )
    print(f"registered {added} (already registered: {sorted(set(ids) - set(added))})")
    return 0


def _video_info(path: Path) -> tuple[float, int, tuple[int, int]]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()
    return (fps if fps > 1 else 25.0), total, size


def eval_cards(clip: str, stem: str, fps: float, total: int, size, stride_s: float):
    """One tap-to-place card per ``stride_s`` seconds of a registered clip."""
    step = max(1, round(fps * stride_s))
    return [
        {
            "id": loop._h(f"evalcard|{clip}|{f}")[:16],
            "clip": stem,
            "run_clip": clip,
            "frame_index": f,
            "frame_size": list(size),
            "bbox": None,
            "place": True,
            "label": loop.BALL_LABEL,
            "provenance": "list",
            "confidence": None,
        }
        for f in range(0, total, step)
    ]


def cmd_eval_list(args) -> int:
    """The card list for sparse eval labelling: about one frame per second of every
    registered clip (or ``--clips``), each a tap-to-place card, no box. Serve it with
    ``FOOTY_BALL_CHECK_LIST`` and the REAL GT-marks folder (the ``corrected`` boxes are
    written there as ``labeller`` marks); then run ``eval-finalise``."""
    reg = eval_registry.read_registry(_registry_path(args))
    want = {canonical_clip_id(c) for c in args.clips} or set(reg)
    unreg = sorted(want - set(reg))
    if unreg:
        print(f"not registered as eval clips: {unreg}", file=sys.stderr)
        return 5
    videos = loop.list_videos(args.clips_dir)
    cards = []
    for clip in sorted(want):
        if clip not in videos:
            print(f"no video for {clip} in {args.clips_dir}", file=sys.stderr)
            return 2
        fps, total, size = _video_info(videos[clip])
        cards += eval_cards(clip, videos[clip].stem, fps, total, size, args.stride_s)
    _atomic(args.out, {"version": 2, "items": cards})
    print(f"{len(cards)} cards for {len(want)} clips -> {args.out}")
    return 0


def _sidecar_stem(gt_dir: Path, clip: str) -> str:
    from footy_track.labeller.ball_check import _gt_sidecar_stem  # noqa: PLC0415

    return _gt_sidecar_stem(gt_dir, clip)


def cmd_eval_finalise(args) -> int:
    """Write what a sparse eval pass decided but the card page did not: a ``not_ball``
    (no ball visible) verdict becomes a ``no_ball`` marker line in the clip's GT sidecar
    (a tapped ball was already appended there by Ball Check). Idempotent."""
    cards = json.loads(args.list.read_text())["items"]
    verdicts = read_card_verdicts(args.gt_dir / "ball_checks", cards)
    n = {"ball": 0, "no_ball": 0, "unsure": 0, "pending": 0, "markers_written": 0}
    new: dict[str, list[dict]] = defaultdict(list)
    for c in cards:
        rec = verdicts.get(c["id"])
        v = None if rec is None else rec["verdict"]
        if v in loop.PRESENT:
            n["ball"] += 1
        elif v == "not_ball":
            n["no_ball"] += 1
            new[c["clip"]].append(
                {
                    "frame_index": c["frame_index"],
                    "bbox": None,
                    "center": None,
                    "tags": ["no_ball"],
                }
            )
        elif v == "unsure":
            n["unsure"] += 1
        else:
            n["pending"] += 1
    for stem, rows in new.items():
        path = args.gt_dir / f"{_sidecar_stem(args.gt_dir, stem)}.jsonl"
        have = set()
        text = path.read_text() if path.exists() else ""
        for line in text.splitlines():
            if line.strip():
                d = json.loads(line)
                if "no_ball" in (d.get("tags") or []):
                    have.add(d["frame_index"])
        add = [r for r in rows if r["frame_index"] not in have]
        if add:
            args.gt_dir.mkdir(parents=True, exist_ok=True)
            prefix = "" if (not text or text.endswith("\n")) else "\n"
            with path.open("a") as fh:
                fh.write(prefix + "".join(json.dumps(r) + "\n" for r in add))
            n["markers_written"] += len(add)
    print(json.dumps(n))
    return 0


# ---------------------------------------------------------------------- legacy import


def cmd_legacy_import(args) -> int:
    """A person's earlier verdicts on the held-out disagreement cards, as evaluation
    corrections only. Nothing is written to a label directory: the clips are evaluation
    clips, so the verdicts correct the eval truth and are kept as hard negatives for
    inspection, never as training data."""
    from footy_track.autolabel.verify_v2 import apply_decisions  # noqa: PLC0415
    from footy_track.scripts.claude_verify_v2 import load_decisions  # noqa: PLC0415

    run = args.run
    items = json.loads((run / "review_list.json").read_text())["items"]
    questions = json.loads((run / "questions.json").read_text())
    truth = json.loads((run / "truth.json").read_text())
    decisions = load_decisions(items, args.verdicts_dir)
    new, stats = apply_decisions(truth, questions, items, decisions)
    corrections, negatives = [], []
    for fkey, t in sorted(new.items()):
        old = truth[fkey]
        added = loop.dedupe_points(t["gt"])
        if (
            added == [list(map(float, g)) for g in old["gt"]]
            and t["kind"] == old["kind"]
        ):
            continue
        corrections.append(
            {
                "fkey": fkey,
                "kind_before": old["kind"],
                "kind_after": t["kind"],
                "marks_before": old["gt"],
                "marks_after": added,
            }
        )
    for it in items:
        rec = decisions.get(it["id"])
        if rec and rec["verdict"] == "not_ball":
            negatives.append(
                {
                    "clip": it["run_clip"],
                    "frame_index": it["frame_index"],
                    "bbox": it["bbox"],
                    "role": it["role"],
                }
            )
    out = {
        "run": run.name,
        "eval_only": True,
        "stats": stats,
        "frames_corrected": len(corrections),
        "hard_negatives": len(negatives),
        "corrections": corrections,
        "negatives": negatives,
    }
    _atomic(args.out, out)
    print(
        json.dumps(
            {k: v for k, v in out.items() if k not in ("corrections", "negatives")}
        )
    )
    return 0


# ------------------------------------------------------------------------ static eval


def cmd_static_eval(args) -> int:
    """How the static filter does on a calibration run (``claude_verify_v2 prepare``):
    of the candidates Claude called ``ball`` (via the answers) how many that are not near
    a human mark (false ``ball``) it removes, and how many near ones (true balls)."""
    from footy_track.autolabel.camera import estimate_camera  # noqa: PLC0415
    from footy_track.autolabel.verify_v2 import score_candidates  # noqa: PLC0415
    from footy_track.ball_eval.detection_replay import read_cache  # noqa: PLC0415

    run = args.run
    questions = json.loads((run / "questions.json").read_text())
    truth = (
        json.loads(args.truth.read_text())
        if args.truth
        else json.loads((run / "truth.json").read_text())
    )
    answers = _load_answers(args.answers)
    rows = {r["id"]: r for r in score_candidates(questions, truth, answers, 25.0)}
    spec_videos = {}
    for d in args.videos:
        spec_videos |= loop.list_videos(d)
    cams, caches = {}, {}
    for clip in sorted({q["clip"] for q in questions}):
        f = run / "camera" / f"{clip}.json"
        if f.exists():
            cams[clip] = {
                int(k): tuple(v) for k, v in json.loads(f.read_text()).items()
            }
        else:
            cams[clip] = estimate_camera(spec_videos[clip])
            _atomic(f, {str(k): list(v) for k, v in cams[clip].items()})
        c = read_cache(args.cache / f"{clip}.jsonl")
        caches[clip] = {
            fr: [
                ((x.x + x.w / 2) * c.frame_size[0], (x.y + x.h / 2) * c.frame_size[1])
                for x in cs
            ]
            for fr, cs in c.frames
        }
    results = []
    for rad, k, win, cam_px in [
        (6, 8, 75, 30),
        (4, 8, 75, 30),
        (6, 5, 50, 20),
        (6, 12, 75, 30),
        (8, 8, 75, 30),
        (6, 8, 75, 60),
        (10, 6, 75, 30),
    ]:
        sp = StaticParams(rad, k, win, cam_px)
        flagged = {c: loop.static_candidates(caches[c], cams[c], sp) for c in caches}
        res = {
            "params": {
                "radius_px": rad,
                "min_frames": k,
                "window": win,
                "min_cam_px": cam_px,
            },
            "false_ball": 0,
            "false_ball_removed": 0,
            "false_ball_tracker": 0,
            "false_ball_tracker_removed": 0,
            "true_ball": 0,
            "true_ball_removed": 0,
            "true_ball_removed_ids": [],
            "all_cands": 0,
            "all_removed": 0,
        }
        for q in questions:
            r = rows.get(q["id"])
            if r is None:
                continue
            fl = loop_flag(q, caches[q["clip"]], flagged[q["clip"]])
            res["all_cands"] += 1
            res["all_removed"] += fl
            if r["verdict"] != "ball":
                continue
            if r["near"]:
                res["true_ball"] += 1
                res["true_ball_removed"] += fl
                if fl:
                    res["true_ball_removed_ids"].append(q["id"])
            else:
                res["false_ball"] += 1
                res["false_ball_removed"] += fl
                if q["cand"]["source"] == "tracker":
                    res["false_ball_tracker"] += 1
                    res["false_ball_tracker_removed"] += fl
        results.append(res)
        print(
            json.dumps(
                {k2: v for k2, v in res.items() if k2 != "true_ball_removed_ids"}
            )
        )
    _atomic(args.out, results)
    return 0


def loop_flag(q: dict, cache_frames: dict, flagged: set) -> bool:
    """Is the question's candidate one the filter flags (matched by position)?"""
    f = q["frame"]
    for i, (x, y) in enumerate(cache_frames.get(f, [])):
        if abs(x - q["cand"]["cx"]) < 1.0 and abs(y - q["cand"]["cy"]) < 1.0:
            return (f, i) in flagged
    return False


# ------------------------------------------------------------------------------ main


def _common(p: argparse.ArgumentParser, run: bool = True) -> None:
    if run:
        p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--split", type=Path, required=True, help="eval split JSON (held-out and seen)"
    )
    p.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="GT marks dir: its clips are eval clips",
    )
    p.add_argument("--eval-dirs", type=Path, nargs="*", default=[])
    p.add_argument("--eval-cache-dir", type=Path, default=None)
    p.add_argument(
        "--eval-registry",
        type=Path,
        default=None,
        help="registry of new eval clips (default $FOOTY_EVAL_REGISTRY): refused too",
    )


def _thr_args(p: argparse.ArgumentParser) -> None:
    d = Thresholds()
    p.add_argument("--auto-conf", type=float, default=d.auto_conf)
    p.add_argument("--drop-conf", type=float, default=d.drop_conf)
    p.add_argument(
        "--audit-rate", type=float, default=d.audit_rate, help="new gate sample share"
    )
    p.add_argument("--audit-min", type=int, default=d.audit_min, help="new gate floor")
    p.add_argument("--settled-rate", type=float, default=d.settled_rate)
    p.add_argument("--settled-min", type=int, default=d.settled_min)
    p.add_argument(
        "--settled-rounds",
        type=int,
        default=d.settled_rounds,
        help="passing rounds in a row before a gate is sampled at the settled rate",
    )
    p.add_argument("--drop-precision", type=float, default=d.drop_precision)
    p.add_argument("--random-frames", type=int, default=d.random_frames)
    p.add_argument(
        "--control-rate",
        type=float,
        default=d.control_rate,
        help="only for rounds decided before the gate audits (scheme 1)",
    )
    p.add_argument("--breaker-precision", type=float, default=d.breaker_precision)


def main(  # noqa: PLR0915
    argv: list[str] | None = None,
) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pool")
    _common(p)
    p.add_argument("--clips-dir", type=Path, required=True)
    p.add_argument("--clips", nargs="*", default=[])
    p.add_argument("--glob", default=None)
    p.add_argument("--random", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--rounds-dir",
        type=Path,
        default=None,
        help="earlier rounds: their clips are skipped",
    )
    p.add_argument("--force", action="store_true", help="choose again")
    p.set_defaults(fn=cmd_pool)

    p = sub.add_parser("check")
    _common(p, run=False)
    p.add_argument("--clips", nargs="+", required=True)
    p.set_defaults(fn=cmd_check)

    d = StaticParams()
    p = sub.add_parser("prepare")
    _common(p)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--top-others", type=int, default=2)
    p.add_argument("--no-static", action="store_true")
    p.add_argument("--static-radius", type=float, default=d.radius_px)
    p.add_argument("--static-frames", type=int, default=d.min_frames)
    p.add_argument("--static-window", type=int, default=d.window)
    p.add_argument("--static-cam", type=float, default=d.min_cam_px)
    p.add_argument("--random-frames", type=int, default=Thresholds().random_frames)
    p.add_argument("--random-seed", type=int, default=7)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("add-random")
    _common(p)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--random-frames", type=int, default=Thresholds().random_frames)
    p.add_argument("--random-seed", type=int, default=7)
    p.set_defaults(fn=cmd_add_random)

    p = sub.add_parser("add-gaps")
    _common(p)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--stride", type=int, default=5)
    p.set_defaults(fn=cmd_add_gaps)

    p = sub.add_parser("decide")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--answers", type=Path, nargs="+", required=True)
    p.add_argument("--rounds-dir", type=Path, default=None)
    p.add_argument("--force-breaker", action="store_true")
    _thr_args(p)
    p.set_defaults(fn=cmd_decide)

    p = sub.add_parser("ingest")
    _common(p)
    p.add_argument("--answers", type=Path, nargs="+", default=[])
    p.add_argument("--verdicts-dir", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--rounds-dir", type=Path, default=None)
    _thr_args(p)
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("eval-register")
    p.add_argument("--registry", type=Path, default=None)
    p.add_argument("--clips-dir", type=Path, required=True)
    p.add_argument("--clips", nargs="*", default=[])
    p.add_argument("--glob", default=None)
    p.add_argument("--rounds-dir", type=Path, default=None)
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_eval_register)

    p = sub.add_parser("eval-list")
    p.add_argument("--registry", type=Path, default=None)
    p.add_argument("--clips-dir", type=Path, required=True)
    p.add_argument("--clips", nargs="*", default=[])
    p.add_argument("--stride-s", type=float, default=1.0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_eval_list)

    p = sub.add_parser("eval-finalise")
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.set_defaults(fn=cmd_eval_finalise)

    p = sub.add_parser("legacy-import")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--verdicts-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_legacy_import)

    p = sub.add_parser("static-eval")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--answers", type=Path, nargs="+", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--videos", type=Path, nargs="+", required=True)
    p.add_argument("--truth", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_static_eval)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except loop.EvalClipError as e:
        print(f"EVAL CLIP REFUSED: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
