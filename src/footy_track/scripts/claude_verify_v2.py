"""Claude verify, method v2: one question per detector candidate.

    # 1. GPU machine: sample frames, tracker, candidates, crop-and-re-detect, images
    python -m footy_track.scripts.claude_verify_v2 prepare \
        --cache C --gt G --videos V --split split.json --weights best.pt --run R

    # 2. any machine with `claude` installed (standard library only), resumable
    python -m footy_track.scripts.claude_verify_v2 ask --run R --model opus \
        --prompt-version verify-v2
    # second look: re-ask the not_ball / unsure answers of a first model
    python -m footy_track.scripts.claude_verify_v2 ask --run R --model opus \
        --prompt-version verify-v2 --only-from R/answers_verify-v2_sonnet.json

    # 3. score against the human marks (+ contact sheet; needs OpenCV)
    python -m footy_track.scripts.claude_verify_v2 score --run R \
        --answers R/answers_verify-v2_opus.json --label opus

    # 4. list the frames where Claude and the human marks disagree (writes
    #    disagreements.csv and review_list.json), then a person re-checks them in
    #    Ball Check (see docs/design/claude_autolabel_loop.md, section 8):
    #    FOOTY_BALL_CHECK_LIST=R/review_list.json  ->  /ball_check?source=list
    python -m footy_track.scripts.claude_verify_v2 disagreements --run R \
        --answers R/answers_verify-v2.2_opus.json --videos V
    # 5. re-score against the marks as corrected by those verdicts
    python -m footy_track.scripts.claude_verify_v2 score --run R \
        --answers R/answers_verify-v2.2_opus.json --label opus \
        --decisions <gt dir>/ball_checks

``questions.json`` is what Claude is shown (an image per candidate); ``truth.json`` holds
the human marks and is read only by ``score``. The marks never choose a crop or a
candidate. Exit codes of ``ask``: 0 done, 2 failed, 3 stopped on an auth or usage-limit
error (never retried, no other backend is tried).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from footy_track.autolabel import claude_cli
from footy_track.autolabel.prompts_v2 import DEFAULT_V2, PROMPTS_V2
from footy_track.autolabel.verify_pilot import (
    UPSCALE,
    Cand,
    Sample,
    sample_frames,
)
from footy_track.autolabel.verify_v2 import (
    MAX_REDETECT,
    NEAR_PX,
    RADII_PX,
    REDETECT_CROP_PX,
    TOP_OTHERS,
    apply_decisions,
    ball_precision_by_det_conf,
    build_review_items,
    find_disagreements,
    frame_key,
    normalise_answer,
    question_id,
    redetect_origin,
    score_candidates,
    score_frames,
    select_candidates,
    summarise,
    tracker_anchor,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
REDETECT_CONF = 0.05
REDETECT_IMGSZ = 960
BATCH = 8
EXIT_LIMIT = 3


# --------------------------------------------------------------------------- prepare


def _px(cands, size):
    w, h = size
    return [
        Cand(
            (c.x + c.w / 2) * w,
            (c.y + c.h / 2) * h,
            c.w * w,
            c.h * h,
            c.confidence,
            "orig",
        )
        for c in cands
    ]


def _params_hash(args: argparse.Namespace) -> str:
    keys = {
        "seed": args.seed,
        "only": args.only,
        "clips": sorted(args.clips or []),
        "seen_with_cand": args.seen_with_cand,
        "seen_without_cand": args.seen_without_cand,
        "seen_no_ball": args.seen_no_ball,
        "max_frames": args.max_frames,
        "top_others": args.top_others,
        "max_redetect": args.max_redetect,
        "weights": args.weights.parent.name,
        "split": json.loads(args.split.read_text()),
    }
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:16]


def cmd_prepare(args: argparse.Namespace) -> int:  # noqa: PLR0912, PLR0915
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    done = run / "prepare.done.json"
    phash = _params_hash(args)
    if done.exists() and not args.force:
        prev = json.loads(done.read_text())
        if prev.get("params") != phash:
            print(
                f"{run} was prepared with different parameters ({prev.get('params')}"
                f" vs {phash}); use a new run name or --force",
                file=sys.stderr,
            )
            return 2
        print(f"prepare already done: {prev['n_questions']} questions; skipping")
        return 0
    done.unlink(missing_ok=True)

    import cv2  # noqa: PLC0415

    from footy_track.autolabel.render_v2 import render_question  # noqa: PLC0415
    from footy_track.ball_eval.detection_replay import (  # noqa: PLC0415
        HUMAN,
        load_gt,
        read_cache,
        replay,
    )
    from footy_track.ball_trackers.detection_tracker import (  # noqa: PLC0415
        DETECTED,
        INTERPOLATED,
    )
    from footy_track.ball_trackers.factory import make_ball_tracker  # noqa: PLC0415
    from footy_track.clip_id import canonical_clip_id  # noqa: PLC0415
    from footy_track.detectors.ultralytics import (  # noqa: PLC0415
        UltralyticsObjectDetector,
    )
    from footy_track.inference.pipeline import (  # noqa: PLC0415
        ball_candidates,
        select_device,
    )

    split = json.loads(args.split.read_text())
    if args.only != "both":
        split = {c: (s if s == args.only else "") for c, s in split.items()}
    if args.clips:
        split = {c: (s if c in args.clips else "") for c, s in split.items()}
    gts = load_gt(args.gt)
    caches = {
        p.stem: read_cache(p)
        for p in sorted(args.cache.glob("*.jsonl"))
        if split.get(p.stem)
    }
    clips = {}
    for clip, cache in caches.items():
        if clip not in gts:
            continue
        w, h = cache.frame_size
        ball, no_ball = gts[clip].scored((HUMAN,))
        clips[clip] = {
            "ball": {f: [(x * w, y * h) for x, y in cs] for f, cs in ball.items()},
            "no_ball": no_ball,
            "cands": {
                f: [(c.cx, c.cy) for c in _px(cs, cache.frame_size)]
                for f, cs in cache.frames
            },
        }
    samples = sample_frames(
        clips,
        split,
        seed=args.seed,
        seen_with_cand=args.seen_with_cand,
        seen_without_cand=args.seen_without_cand,
        seen_no_ball=args.seen_no_ball,
    )
    if args.max_frames:
        samples = samples[: args.max_frames]
    (run / "samples.json").write_text(
        json.dumps([asdict(s) for s in samples], indent=1)
    )
    print(f"{len(samples)} sampled frames from {len(clips)} clips", flush=True)

    videos = {}
    for d in args.videos:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in VIDEO_SUFFIXES:
                videos.setdefault(canonical_clip_id(p.name), p)

    tracks, detections = {}, {}
    for clip in {s.clip for s in samples}:
        w, h = caches[clip].frame_size
        pts = replay(make_ball_tracker("bidirectional", (w, h)), caches[clip])
        tracks[clip], detections[clip] = {}, {}
        for p in pts:
            if p.x is None or p.source not in (DETECTED, INTERPOLATED):
                continue
            c = ((p.x + p.w / 2) * w, (p.y + p.h / 2) * h)
            tracks[clip][p.frame_index] = c
            if p.source == DETECTED:
                detections[clip][p.frame_index] = c

    detector = UltralyticsObjectDetector(
        model_uri=str(args.weights),
        min_confidence=REDETECT_CONF,
        imgsz=REDETECT_IMGSZ,
        use_model_names=True,
        device=select_device(),
    )
    order = {(s.clip, s.frame): i for i, s in enumerate(samples)}
    by_clip: dict[str, list[Sample]] = {}
    for s in samples:
        by_clip.setdefault(s.clip, []).append(s)
    images = run / "images"
    frames_out, questions, truth = [], [], {}
    for clip, ss in sorted(by_clip.items()):
        wanted = {s.frame: s for s in ss}
        size = caches[clip].frame_size
        cached = dict(caches[clip].frames)
        grabbed = {}
        cap = cv2.VideoCapture(str(videos[clip]))
        idx = 0
        while len(grabbed) < len(wanted) and idx <= max(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                grabbed[idx] = frame
            idx += 1
        cap.release()
        missing = sorted(set(wanted) - set(grabbed))
        if missing:
            print(f"{clip}: could not read frames {missing[:5]}...", file=sys.stderr)
        # crop-and-re-detect around the tracker path, one batch per clip
        todo, crops = [], []
        for f in sorted(grabbed):
            anchor = tracker_anchor(f, tracks[clip])
            if anchor is None:
                continue
            x0, y0 = redetect_origin(anchor, size)
            crop = grabbed[f][y0 : y0 + REDETECT_CROP_PX, x0 : x0 + REDETECT_CROP_PX]
            crops.append(
                cv2.resize(
                    crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC
                )
            )
            todo.append((f, x0, y0))
        redet: dict[int, list[Cand]] = {}
        for i in range(0, len(crops), 8):
            res = detector.predict_frames(crops[i : i + 8])
            for (f, x0, y0), r in zip(todo[i : i + 8], res, strict=True):
                redet[f] = [
                    Cand(
                        x0 + (c.x + c.w / 2) * REDETECT_CROP_PX,
                        y0 + (c.y + c.h / 2) * REDETECT_CROP_PX,
                        c.w * REDETECT_CROP_PX,
                        c.h * REDETECT_CROP_PX,
                        c.confidence,
                        "redetect",
                    )
                    for c in ball_candidates(r)
                ]
        for f in sorted(grabbed):
            s = wanted[f]
            fkey = frame_key(clip, f)
            cands = select_candidates(
                _px(cached.get(f, []), size),
                detections[clip].get(f),
                redet.get(f, []),
                top_others=args.top_others,
                max_redetect=args.max_redetect,
            )
            g = clips[clip]["ball"].get(f, [])
            truth[fkey] = {"kind": s.kind, "gt": [list(p) for p in g]}
            frames_out.append(
                {
                    "fkey": fkey,
                    "clip": clip,
                    "frame": f,
                    "split": s.split,
                    "kind": s.kind,
                    "stratum": s.stratum,
                    "n_cached": len(cached.get(f, [])),
                    "n_redetect_raw": len(redet.get(f, [])),
                    "n_questions": len(cands),
                    "order": order[(clip, f)],
                }
            )
            for k, c in enumerate(cands):
                qid = question_id(clip, f, k)
                img = images / f"{qid}.png"
                origin = [round(c.cx - 64), round(c.cy - 64)]
                if not img.exists():
                    render_question(grabbed[f], c, img)
                questions.append(
                    {
                        "id": qid,
                        "fkey": fkey,
                        "clip": clip,
                        "frame": f,
                        "split": s.split,
                        "stratum": s.stratum,
                        "k": k,
                        "image": img.name,
                        "patch": origin,
                        "cand": asdict(c),
                    }
                )
        print(
            f"{clip}: {len(grabbed)} frames, {len(questions)} questions so far",
            flush=True,
        )
    # fixed shuffled order of questions (frames spread out, deterministic)
    questions.sort(key=lambda q: hashlib.sha256(q["id"].encode()).hexdigest())
    _atomic(run / "questions.json", questions)
    _atomic(run / "frames.json", sorted(frames_out, key=lambda f: f["order"]))
    _atomic(run / "truth.json", truth)
    src = {}
    for q in questions:
        src[q["cand"]["source"]] = src.get(q["cand"]["source"], 0) + 1
    _atomic(
        done,
        {
            "params": phash,
            "n_frames": len(frames_out),
            "n_questions": len(questions),
            "by_source": src,
        },
    )
    print(f"{len(frames_out)} frames, {len(questions)} questions, by source {src}")
    return 0


def _atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ------------------------------------------------------------------------------- ask


def _tag(version: str, model: str) -> str:
    return f"{version}_{model}"


def cmd_ask(args: argparse.Namespace) -> int:  # noqa: PLR0915
    if args.backend == "api":
        print(
            "backend=api is not wired: only the Claude Code CLI backend exists",
            file=sys.stderr,
        )
        return 2
    run = args.run.resolve()
    questions = json.loads((run / "questions.json").read_text())
    if args.only_from:
        first = json.loads(Path(args.only_from).read_text())
        keep = set(args.only_verdicts.split(","))
        questions = [
            q for q in questions if first.get(q["id"], {}).get("verdict") in keep
        ]
    if args.limit:
        questions = questions[: args.limit]
    tag = _tag(args.prompt_version, args.model)
    out_path = run / f"answers_{tag}.json"
    log_path = run / f"calls_{tag}.jsonl"
    status_path = run / f"ask_status_{tag}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    batches = [
        questions[i : i + args.batch] for i in range(0, len(questions), args.batch)
    ]
    todo = [b for b in batches if not all(q["id"] in done for q in b)]
    if args.budget:  # a per-run cap on NEW questions (answers already saved are free)
        left, capped = args.budget, []
        for b in todo:
            fresh = [q for q in b if q["id"] not in done]
            if left <= 0:
                break
            capped.append(fresh[:left])
            left -= len(capped[-1])
        todo = capped
        print(
            f"budget {args.budget}: asking at most {sum(map(len, todo))} new questions"
        )
    print(
        f"{len(questions)} questions, {len(todo)} of {len(batches)} batches to run "
        f"with {args.model} ({args.prompt_version})",
        flush=True,
    )
    failure: list[claude_cli.ClaudeCallError | Exception] = []
    lock = threading.Lock()
    img_dir = run / "images"

    def work(batch):
        if failure:
            return
        items = [
            {"id": q["id"], "image": str((img_dir / q["image"]).resolve())}
            for q in batch
        ]
        try:
            attempts = claude_cli.ask_batch(
                items, args.model, img_dir, args.prompt_version
            )
        except (claude_cli.ClaudeCallError, OSError) as e:
            failure.append(e)
            return
        with lock:
            for a in attempts:
                rec = {k: v for k, v in asdict(a).items() if k != "answers"}
                rec |= {"prompt_version": args.prompt_version, "n": len(items)}
                with log_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            final = attempts[-1]
            for i, q in enumerate(batch):
                raw = final.answers[i] if final.answers is not None else None
                done[q["id"]] = normalise_answer(raw) | {
                    "raw": raw,
                    "model": args.model,
                    "prompt_version": args.prompt_version,
                }
            _atomic(out_path, done)
            print(
                f"batch {batch[0]['id']}.. {final.seconds:.0f}s ({len(done)} done)",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        list(ex.map(work, todo))
    if failure:
        e = failure[0]
        kind = getattr(e, "kind", "other")
        status = {
            "status": "stopped",
            "kind": kind,
            "message": str(e)[:1500],
            "answered": len(done),
        }
        _atomic(status_path, status)
        print(f"STOPPED[{kind}]: {e}", file=sys.stderr)
        return EXIT_LIMIT if kind in ("auth", "usage_limit") else 2
    _atomic(status_path, {"status": "ok", "answered": len(done)})
    return 0


# ------------------------------------------------------------------------------ score


def _fmt(v):
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


HEAD_KEYS = [
    "ball_frames_answerable",
    "ball_frames_no_cand_within_R",
    "frame_agreement",
    "frame_agreement_no_ball",
    "frame_agreement_all",
    "near_cands",
    "far_cands",
    "ball_precision",
    "ball_recall",
    "false_not_ball_rate",
    "false_ball_rate",
    "false_ball_on_no_ball_cands",
    "false_ball_on_far_cands_ball_frames",
    "unsure_rate",
    "invalid",
]


def load_answers(paths: list[Path]) -> dict[str, dict]:
    """Merge answer files; a later file overrides an earlier one (second look)."""
    out: dict[str, dict] = {}
    for p in paths:
        out |= json.loads(Path(p).read_text())
    return out


def load_decisions(items: list[dict], ball_checks_dir: Path) -> dict[str, dict]:
    """Latest live Ball Check verdict per review card id (source ``list``)."""
    from footy_track.feature_store.ball_checks import (  # noqa: PLC0415
        read_latest_verdicts,
    )

    out: dict[str, dict] = {}
    by_clip: dict[str, dict] = {}
    for i, it in enumerate(items):
        path = ball_checks_dir / f"{it['clip']}.jsonl"
        if it["clip"] not in by_clip:
            by_clip[it["clip"]] = read_latest_verdicts(path) if path.exists() else {}
        rec = by_clip[it["clip"]].get(("list", it["frame_index"], i))
        if rec is not None:
            out[it["id"]] = rec
    return out


def cmd_disagreements(args: argparse.Namespace) -> int:
    import cv2  # noqa: PLC0415

    from footy_track.clip_id import canonical_clip_id  # noqa: PLC0415

    run = args.run
    questions = json.loads((run / "questions.json").read_text())
    truth = json.loads((run / "truth.json").read_text())
    answers = load_answers(args.answers)
    questions = [q for q in questions if args.split in ("all", q["split"])]
    dis = find_disagreements(questions, truth, answers, args.radius)
    videos = {}
    for d in args.videos:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in VIDEO_SUFFIXES:
                videos.setdefault(canonical_clip_id(p.name), p)
    sizes, stems = {}, {}
    for clip in sorted({d["clip"] for d in dis}):
        if clip not in videos:
            print(f"no video for {clip}", file=sys.stderr)
            return 2
        cap = cv2.VideoCapture(str(videos[clip]))
        sizes[clip] = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        stems[clip] = videos[clip].stem
    items = build_review_items(dis, sizes, stems)
    _atomic(run / "review_list.json", {"version": 1, "items": items})
    _atomic(run / "disagreements.json", dis)
    cols = [
        "type", "clip", "frame", "split", "source", "verdict", "claude_conf",
        "det_conf", "cand_cx", "cand_cy", "cand_w", "cand_h", "gt",
        "dist_to_nearest_gt", "qid", "reason",
    ]  # fmt: skip
    with (run / "disagreements.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in dis:
            w.writerow(d | {"gt": json.dumps(d["gt"])})
    by_type: dict[str, int] = {}
    for d in dis:
        k = d["type"] + (" (unsure)" if d["verdict"] == "unsure" else "")
        by_type[k] = by_type.get(k, 0) + 1
    print(f"{len(dis)} disagreements on {len({d['fkey'] for d in dis})} frames")
    print("by type:", by_type)
    print(f"{len(items)} review cards -> {run / 'review_list.json'}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    run = args.run
    questions = json.loads((run / "questions.json").read_text())
    frames = json.loads((run / "frames.json").read_text())
    truth = json.loads((run / "truth.json").read_text())
    answers = load_answers(args.answers)
    if args.decisions:
        items = json.loads((run / "review_list.json").read_text())["items"]
        decisions = load_decisions(items, args.decisions)
        truth, changes = apply_decisions(truth, questions, items, decisions)
        args.label = f"{args.label}_rechecked"
        _atomic(run / f"truth_{args.label}.json", truth)
        print("recheck decisions applied:", json.dumps(changes))
    summary: dict = {}
    for radius in args.radii:
        rows = score_candidates(questions, truth, answers, radius)
        frs = score_frames(frames, rows, questions)
        groups = {"all": None, "held_out": "held_out", "seen": "seen"}
        for g, sp in groups.items():
            fg = [f for f in frames if sp is None or f["split"] == sp]
            rg = [r for r in rows if sp is None or r["split"] == sp]
            frg = [f for f in frs if sp is None or f["split"] == sp]
            summary[f"R{radius:g}|{g}"] = summarise(fg, rg, frg)
        if radius == args.radii[0]:
            summary["detector_conf_bands"] = ball_precision_by_det_conf(rows)
            summary["rows_R"] = radius
    calls = {}
    for lp in args.calls:
        if lp.exists():
            cs = [json.loads(x) for x in lp.read_text().splitlines() if x.strip()]
            for c in cs:
                m = calls.setdefault(
                    c["model"],
                    {
                        "calls": 0,
                        "seconds": 0.0,
                        "cost_usd": 0.0,
                        "questions": 0,
                        "retries": 0,
                    },
                )
                m["calls"] += 1
                m["seconds"] += c["seconds"]
                m["cost_usd"] += c.get("cost_usd") or 0.0
                m["questions"] += c["n"]
                m["retries"] += c["attempt"] > 1
    for m in calls.values():
        m["seconds_per_question"] = m["seconds"] / max(m["questions"], 1)
        m["cost_usd_per_question"] = m["cost_usd"] / max(m["questions"], 1)
    summary["calls"] = calls
    summary["label"] = args.label
    summary["n_answers"] = len(answers)
    summary["n_questions"] = len(questions)
    _atomic(run / f"summary_{args.label}.json", summary)
    for k, s in summary.items():
        if k.startswith("R"):
            print(f"{k}: " + ", ".join(f"{n}={_fmt(s[n])}" for n in HEAD_KEYS))
            if k.endswith("|all") or k.endswith("|held_out"):
                print("   by source:", s["by_source"])
                print(
                    "   Claude-conf band [said ball, of which near]:",
                    s["ball_verdict_by_conf_band"],
                )
    print("detector conf bands [n, near]:", summary["detector_conf_bands"])
    print("calls:", json.dumps(calls))
    _contact_sheet(run, questions, truth, answers, args.label, args.radii[0])
    return 0


def _contact_sheet(run, questions, truth, answers, label, radius):
    from footy_track.autolabel.render_v2 import contact_sheet  # noqa: PLC0415

    rows = {r["id"]: r for r in score_candidates(questions, truth, answers, radius)}
    qs = {q["id"]: q for q in questions}
    fn = [r for r in rows.values() if r["near"] and r["verdict"] != "ball"]
    fp = [r for r in rows.values() if not r["near"] and r["verdict"] == "ball"]
    picked = fn[:8] + fp[:8]
    items = []
    for r in picked:
        q = qs[r["id"]]
        gt = truth[q["fkey"]]["gt"]
        pos = (
            min(
                gt,
                key=lambda g: (g[0] - q["cand"]["cx"]) ** 2
                + (g[1] - q["cand"]["cy"]) ** 2,
            )
            if gt
            else None
        )
        cap = f"{q['split'][:4]} {r['source']} {r['verdict']} {'FN' if r['near'] else 'FP'}"
        items.append(
            (
                run / "images" / q["image"],
                cap,
                tuple(pos) if pos else None,
                tuple(q["patch"]),
            )
        )
    out = run / f"contact_sheet_{label}.jpg"
    contact_sheet(items, out)
    print(f"contact sheet: {out} ({len(items)} tiles)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--videos", type=Path, nargs="+", required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--only", choices=("held_out", "seen", "both"), default="both")
    p.add_argument("--clips", nargs="*", default=[])
    p.add_argument("--seen-with-cand", type=int, default=50)
    p.add_argument("--seen-without-cand", type=int, default=50)
    p.add_argument("--seen-no-ball", type=int, default=20)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--top-others", type=int, default=TOP_OTHERS)
    p.add_argument("--max-redetect", type=int, default=MAX_REDETECT)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare)
    a = sub.add_parser("ask")
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--model", default="opus")
    a.add_argument("--prompt-version", default=DEFAULT_V2, choices=sorted(PROMPTS_V2))
    a.add_argument("--limit", type=int, default=0)
    a.add_argument(
        "--budget",
        type=int,
        default=0,
        help="ask at most this many NEW questions in this run (0 = no cap)",
    )
    a.add_argument("--batch", type=int, default=BATCH)
    a.add_argument("--parallel", type=int, default=3)
    a.add_argument("--backend", choices=("cli", "api"), default="cli")
    a.add_argument("--only-from", help="answers file; re-ask only its --only-verdicts")
    a.add_argument("--only-verdicts", default="not_ball,unsure")
    a.set_defaults(fn=cmd_ask)
    s = sub.add_parser("score")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--answers", type=Path, nargs="+", required=True)
    s.add_argument("--calls", type=Path, nargs="*", default=[])
    s.add_argument("--label", default="run")
    s.add_argument("--radii", type=float, nargs="+", default=list(RADII_PX))
    s.add_argument(
        "--decisions",
        type=Path,
        help="ball_checks dir holding a person's verdicts on review_list.json cards; "
        "re-scores against the marks as corrected by them (label gets _rechecked)",
    )
    s.set_defaults(fn=cmd_score)
    d = sub.add_parser("disagreements")
    d.add_argument("--run", type=Path, required=True)
    d.add_argument("--answers", type=Path, nargs="+", required=True)
    d.add_argument("--videos", type=Path, nargs="+", required=True)
    d.add_argument("--radius", type=float, default=NEAR_PX)
    d.add_argument("--split", choices=("held_out", "seen", "all"), default="all")
    d.set_defaults(fn=cmd_disagreements)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
