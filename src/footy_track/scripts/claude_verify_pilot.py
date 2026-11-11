"""Calibration pilot for the Claude verify step (docs/design/claude_autolabel_loop.md).

    # 1. GPU machine: sample frames, crop, re-detect, draw question images
    python -m footy_track.scripts.claude_verify_pilot prepare \
        --cache C --gt G --videos V --split split.json --weights best.pt --out P

    # 2. any machine with `claude` installed (no GPU, standard library only)
    python -m footy_track.scripts.claude_verify_pilot ask --pilot P --model sonnet
    python -m footy_track.scripts.claude_verify_pilot ask --pilot P --model opus --limit 60

    # 3. score against the human marks (+ contact sheet; needs OpenCV)
    python -m footy_track.scripts.claude_verify_pilot score --pilot P \
        --results P/results_sonnet.json P/results_opus.json

``questions.json`` holds what Claude is shown; ``truth.json`` holds the human marks and is
used only by ``score``. The GT never decides where a crop or a candidate goes.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from footy_track.autolabel import claude_cli
from footy_track.autolabel.verify_pilot import (
    CROP_PX,
    RADII_PX,
    REDETECT_CONF,
    UPSCALE,
    VIEW_PX,
    Cand,
    Sample,
    crop_origin,
    expected_position,
    in_crop,
    merge_candidates,
    normalise_answer,
    sample_frames,
    score_question,
    summarise,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
BATCH = 6


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


def cmd_prepare(args: argparse.Namespace) -> int:  # noqa: PLR0912, PLR0915
    import cv2  # noqa: PLC0415

    from footy_track.autolabel.render import render_question  # noqa: PLC0415
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
    gts = load_gt(args.gt)
    caches = {p.stem: read_cache(p) for p in sorted(args.cache.glob("*.jsonl"))}
    clips = {}
    for clip, cache in caches.items():
        if clip not in gts or clip not in split:
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
    samples = sample_frames(clips, split, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "samples.json").write_text(
        json.dumps([asdict(s) for s in samples], indent=1)
    )
    print(f"{len(samples)} samples")

    videos = {}
    for d in args.videos:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in VIDEO_SUFFIXES:
                videos.setdefault(canonical_clip_id(p.name), p)

    # expected positions from the bidirectional tracker (cache only, no GT)
    tracks = {}
    for clip in {s.clip for s in samples}:
        pts = replay(
            make_ball_tracker("bidirectional", caches[clip].frame_size), caches[clip]
        )
        w, h = caches[clip].frame_size
        tracks[clip] = {
            p.frame_index: ((p.x + p.w / 2) * w, (p.y + p.h / 2) * h)
            for p in pts
            if p.source in (DETECTED, INTERPOLATED) and p.x is not None
        }

    detector = UltralyticsObjectDetector(
        model_uri=str(args.weights),
        min_confidence=REDETECT_CONF,
        imgsz=args.imgsz,
        use_model_names=True,
        device=select_device(),
    )
    questions, truth = [], {}
    by_clip: dict[str, list[Sample]] = {}
    for s in samples:
        by_clip.setdefault(s.clip, []).append(s)
    order = {(s.clip, s.frame): i for i, s in enumerate(samples)}
    frames_dir = args.out / "images"
    pending = []  # (sample, frame_bgr, origin, size, orig_in_crop, expected_src)
    for clip, ss in sorted(by_clip.items()):
        wanted = {s.frame: s for s in ss}
        cap = cv2.VideoCapture(str(videos[clip]))
        size = caches[clip].frame_size
        cands_by_frame = dict(caches[clip].frames)
        idx = 0
        while wanted and idx <= max(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                s = wanted.pop(idx)
                ex, ey, src = expected_position(idx, tracks[clip], size)
                origin = crop_origin(ex, ey, size)
                orig = [
                    c
                    for c in _px(cands_by_frame[idx], size)
                    if in_crop((c.cx, c.cy), *origin)
                ]
                pending.append((s, frame, origin, size, orig, src))
            idx += 1
        cap.release()
        print(f"{clip}: read frames", flush=True)

    for i in range(0, len(pending), 8):
        batch = pending[i : i + 8]
        ups = [
            cv2.resize(
                f[o[1] : o[1] + CROP_PX, o[0] : o[0] + CROP_PX],
                None,
                fx=UPSCALE,
                fy=UPSCALE,
                interpolation=cv2.INTER_CUBIC,
            )
            for _, f, o, *_ in batch
        ]
        results = detector.predict_frames(ups)
        for (s, frame, origin, _size, orig, src), res in zip(
            batch, results, strict=True
        ):
            re_c = [
                Cand(
                    origin[0] + (c.x + c.w / 2) * VIEW_PX / UPSCALE,
                    origin[1] + (c.y + c.h / 2) * VIEW_PX / UPSCALE,
                    c.w * VIEW_PX / UPSCALE,
                    c.h * VIEW_PX / UPSCALE,
                    c.confidence,
                    "redetect",
                )
                for c in ball_candidates(res)
            ]
            final = merge_candidates(orig, re_c)
            qid = f"q{order[(s.clip, s.frame)]:03d}"
            img = frames_dir / f"{qid}.png"
            render_question(frame, origin, final, img)
            questions.append(
                {
                    "id": qid,
                    "clip": s.clip,
                    "frame": s.frame,
                    "split": s.split,
                    "stratum": s.stratum,
                    "crop": list(origin),
                    "expected_source": src,
                    "image": str(img),
                    "n_orig_in_crop": len(orig),
                    "n_redetect_raw": len(re_c),
                    "candidates": [asdict(c) for c in final],
                }
            )
            g = clips[s.clip]["ball"].get(s.frame, [])
            truth[qid] = {"kind": s.kind, "gt": [list(p) for p in g]}
        print(f"rendered {min(i + 8, len(pending))}/{len(pending)}", flush=True)
    questions.sort(key=lambda q: q["id"])
    (args.out / "questions.json").write_text(json.dumps(questions, indent=1))
    (args.out / "truth.json").write_text(json.dumps(truth))
    return 0


# ------------------------------------------------------------------------------- ask


def cmd_ask(args: argparse.Namespace) -> int:
    pilot = args.pilot.resolve()
    questions = json.loads((pilot / "questions.json").read_text())
    if args.limit:
        questions = questions[: args.limit]
    out_path = pilot / f"results_{args.model}.json"
    log_path = pilot / f"calls_{args.model}.jsonl"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    batches = [questions[i : i + BATCH] for i in range(0, len(questions), BATCH)]
    todo = [b for b in batches if not all(q["id"] in done for q in b)]
    print(f"{len(todo)} of {len(batches)} batches to run with {args.model}", flush=True)
    failure: list[str] = []

    def work(batch):
        if failure:
            return
        items = [
            {
                "id": q["id"],
                "image": str(_image(pilot, q)),
                "n_candidates": len(q["candidates"]),
            }
            for q in batch
        ]
        try:
            attempts = claude_cli.ask_batch(items, args.model, pilot)
        except (claude_cli.ClaudeCallError, OSError) as e:
            failure.append(str(e))
            return
        for a in attempts:
            rec = {k: v for k, v in asdict(a).items() if k != "answers"}
            rec |= {"prompt_version": claude_cli.PROMPT_VERSION, "n": len(items)}
            with log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        final = attempts[-1]
        for i, q in enumerate(batch):
            raw = final.answers[i] if final.answers is not None else None
            n = len(q["candidates"])
            done[q["id"]] = normalise_answer(raw, n) | {"raw": raw, "model": args.model}
        out_path.write_text(json.dumps(done, indent=1))
        print(
            f"batch {batch[0]['id']}..{batch[-1]['id']} {final.seconds:.0f}s",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        list(ex.map(work, todo))
    out_path.write_text(json.dumps(done, indent=1))
    if failure:
        print("STOPPED: " + failure[0], file=sys.stderr)
        return 2
    return 0


# ------------------------------------------------------------------------------ score


def _rows(questions, truth, results, radius):
    rows = []
    for q in questions:
        a = results.get(q["id"])
        if a is None:
            continue
        r = score_question(q, truth[q["id"]], a, radius)
        rows.append(
            r
            | {
                "id": q["id"],
                "split": q["split"],
                "kind": truth[q["id"]]["kind"],
                "verdict": a["verdict"],
                "confidence": a["confidence"],
                "stratum": q["stratum"],
            }
        )
    return rows


def _image(pilot: Path, q: dict) -> Path:
    """The question image inside this pilot folder (it may have been copied elsewhere)."""
    return (pilot / "images" / Path(q["image"]).name).resolve()


def _fmt(v):
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


SUMMARY_KEYS = [
    "n",
    "ball_frames",
    "no_ball_frames",
    "agreement",
    "agreement_cand_available",
    "agreement_no_cand",
    "ball_cand_within",
    "ball_no_cand",
    "false_none_rate",
    "false_none_cand_available",
    "false_ball_rate",
    "false_ball_on_no_ball",
    "wrong_pick_rate_on_ball",
    "unsure_rate",
    "invalid",
    "gt_outside_crop",
    "agreement_adjusted",
    "agreement_gt_in_crop",
    "false_none_gt_in_crop",
    "cell_answers",
    "cell_correct",
    "redetect_only",
    "redetect_only_picked",
]


def cmd_score(args: argparse.Namespace) -> int:
    pilot = args.pilot
    questions = json.loads((pilot / "questions.json").read_text())
    truth = json.loads((pilot / "truth.json").read_text())
    summary: dict = {}
    all_results = {}
    for rp in args.results:
        results = json.loads(rp.read_text())
        all_results[next(iter(results.values()))["model"]] = results
    shared = set.intersection(*(set(r) for r in all_results.values()))
    for model, results in all_results.items():
        for r in RADII_PX:
            rows = _rows(questions, truth, results, r)
            groups = {
                "all": rows,
                "held_out": [x for x in rows if x["split"] == "held_out"],
                "seen": [x for x in rows if x["split"] == "seen"],
            }
            if len(all_results) > 1:
                groups["shared_subset"] = [x for x in rows if x["id"] in shared]
            for g, rs in groups.items():
                summary[f"{model}|R{r:g}|{g}"] = summarise(rs)
            if r == RADII_PX[0]:
                bands = {}
                for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 1.01)):
                    b = [
                        x
                        for x in rows
                        if x["confidence"] is not None and lo <= x["confidence"] < hi
                    ]
                    bands[f"{lo}-{min(hi, 1)}"] = [
                        len(b),
                        sum(x["outcome"] == "correct" for x in b),
                    ]
                summary[f"{model}|confidence_bands_R{r:g}"] = bands
    for k, s in summary.items():
        if "confidence_bands" in k:
            print(f"{k} [n, correct]: {s}")
        else:
            print(f"{k}: " + ", ".join(f"{n}={_fmt(s[n])}" for n in SUMMARY_KEYS))
    (pilot / "summary.json").write_text(json.dumps(summary, indent=1))

    for model in all_results:
        lp = pilot / f"calls_{model}.jsonl"
        if lp.exists():
            calls = [json.loads(x) for x in lp.read_text().splitlines()]
            secs = [c["seconds"] for c in calls]
            costs = [c["cost_usd"] for c in calls if c.get("cost_usd") is not None]
            print(
                f"calls {model}: {len(calls)} calls, mean {sum(secs) / len(secs):.1f}s, "
                f"cost total {sum(costs):.3f} mean {sum(costs) / max(len(costs), 1):.3f} USD"
            )
    _contact_sheet(pilot, questions, truth, args.sheet_model)
    return 0


def _contact_sheet(pilot, questions, truth, model):
    from footy_track.autolabel.render import contact_sheet  # noqa: PLC0415

    results = json.loads((pilot / f"results_{model}.json").read_text())
    rows = _rows(questions, truth, results, RADII_PX[0])
    qs = {q["id"]: q for q in questions}
    wrong = [
        r for r in rows if r["outcome"] in ("wrong_pick", "false_none", "false_ball")
    ]
    wins = [r for r in rows if r["redetect_only"] and r["outcome"] == "correct"]
    picked = wrong[:8] + wins[:4]
    picked += (wrong[8:] + wins[4:])[: 12 - len(picked)]
    items, origins = [], []
    for r in picked:
        q = qs[r["id"]]
        g = truth[r["id"]]["gt"]
        cap = f"{r['id']} {q['split'][:4]} {r['kind']} {r['verdict']} -> {r['outcome']}"
        items.append((_image(pilot, q), cap, tuple(g[0]) if g else None))
        origins.append(tuple(q["crop"]))
    out = pilot / "contact_sheet.jpg"
    contact_sheet(items, out, origin_of=origins)
    print(f"contact sheet: {out}: " + ", ".join(r["id"] for r in picked))


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
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--imgsz", type=int, default=VIEW_PX)
    p.set_defaults(fn=cmd_prepare)
    a = sub.add_parser("ask")
    a.add_argument("--pilot", type=Path, required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--parallel", type=int, default=3)
    a.set_defaults(fn=cmd_ask)
    s = sub.add_parser("score")
    s.add_argument("--pilot", type=Path, required=True)
    s.add_argument("--results", type=Path, nargs="+", required=True)
    s.add_argument("--sheet-model", default="sonnet")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
