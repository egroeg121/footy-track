"""Ball evaluation panel: build the sample, submit the blind labelling job, score a model on it.

    python -m footy_track.scripts.ball_eval_panel build  --match M [--n-windows 24] [--dry-run] [--force]
    python -m footy_track.scripts.ball_eval_panel build  --all-heldout           # every segmented heldout match
    python -m footy_track.scripts.ball_eval_panel job    --match M [--dry-run]   # blind ball_place job, key ball-eval-panel/<ver>/<M>
    python -m footy_track.scripts.ball_eval_panel suggest --match M [--device cpu] [--weights W]  # detector suggestions for the page
    python -m footy_track.scripts.ball_eval_panel render [--match M | --all-heldout] [--force]  # keyframe-dense clips (kf10)
    python -m footy_track.scripts.ball_eval_panel score  --pred preds.jsonl --model NAME [--blind-only] [--out report.json]
    python -m footy_track.scripts.ball_eval_panel status [--labeller-url URL]

Design and estimator: ``docs/design/ball_sampling.md`` section 2, ``footy_track/ball_eval/panel.py`` (sampler),
``panel_score.py`` (Horvitz-Thompson + stratified cluster bootstrap). Data root: ``$FOOTY_DATA`` (default
``/mnt/storage/footy_data``); ``--root`` reads from / writes to another one (scratch runs). ``--out-dir`` keeps
reading the real segments but writes the panel files elsewhere.

Every command prints JSON on stdout (the last line of ``build`` / ``job`` is the machine-readable summary the
Prefect flows parse).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from footy_track.ball_eval import panel as P
from footy_track.ball_eval import panel_score as S


def _matches(args: argparse.Namespace) -> list[str]:
    if getattr(args, "all_heldout", False):
        return P.segmented_heldout_matches(args.root)
    if not args.match:
        raise SystemExit("give --match M (repeatable) or --all-heldout")
    return list(args.match)


def _summary(doc: dict, status: str, path: Path | None) -> dict:
    return {
        "match": doc["match"],
        "panel_version": doc["panel_version"],
        "status": status,
        "path": str(path) if path else None,
        "N_windows": doc["N_windows"],
        "n_windows": doc["n_windows"],
        "n_frames": doc["n_frames"],
        "n_qa_repeat": doc["n_qa_repeat"],
        "n_qa_gold": doc["n_qa_gold"],
        "n_cards": doc["n_frames"] + len(doc["qa"]),
        "pi_window": round(doc["pi_window"], 6),
        "fps": doc["fps"],
        "frame_gap": doc["design"]["frame_gap"],
        "frame_size": [doc["width"], doc["height"]],
        "seed": doc["seed"],
        "code_sha": doc["code_sha"],
    }


def cmd_build(args: argparse.Namespace) -> int:
    out_root = Path(args.out_dir) if args.out_dir else args.root
    gold = [Path(g) for g in args.gold_dir] if args.gold_dir else None
    rc = 0
    for match in _matches(args):
        try:
            doc = P.build_panel(
                match,
                panel_version=args.panel_version,
                n_windows=args.n_windows,
                root=args.root,
                gold_dirs=gold,
                probe=None if args.no_probe else P.cv2_probe,
            )
            if args.dry_run:
                print(json.dumps(_summary(doc, "dry-run", None), sort_keys=True))
                continue
            path, status = P.write_panel(doc, root=out_root, force=args.force)
            print(json.dumps(_summary(doc, status, path), sort_keys=True))
        except (P.PanelError, OSError, ValueError) as e:
            print(
                json.dumps({"match": match, "status": "error", "error": str(e)}),
                flush=True,
            )
            rc = 1
    return rc


def cmd_suggest(args: argparse.Namespace) -> int:
    """Precompute the top-1 RT-DETR ball box of every panel frame (``<match>.suggest.<sha8>.json``).

    One match at a time, threads capped (``$BALL_POOL_THREADS`` / ``$OMP_NUM_THREADS``, default 4): merlin is shared.
    Heldout data, eval-only inference (see ``ball_eval/panel_suggest.py``). Never touches the panel files or marks."""
    from footy_track.ball_eval import panel_suggest as PS  # noqa: N812, PLC0415

    out_root = Path(args.out_dir) if args.out_dir else args.root
    rc = 0
    for match in _matches(args):
        try:
            doc = P.read_panel(P.panel_path(match, args.panel_version, out_root))
            r = PS.suggest_match(
                match,
                panel_version=args.panel_version,
                root=out_root,
                weights=Path(args.weights) if args.weights else None,
                device=args.device,
                imgsz=args.imgsz,
                batch=args.batch,
                tag=getattr(args, "tag", "") or "",
                labelled_keys=lambda d=doc: PS.labelled_panel_keys(d, args.root),
                log=lambda m: print(m, file=sys.stderr, flush=True),
            )
            print(json.dumps({k: v for k, v in r.items() if k not in ("blind_windows", "pre_labelled")} | {"n_pre_labelled": len(r.get("pre_labelled", []))}, sort_keys=True), flush=True)
        except (P.PanelError, OSError, ValueError) as e:
            print(json.dumps({"match": match, "status": "error", "error": str(e)}), flush=True)
            rc = 1
    return rc


def _render_best_effort(doc: dict, root: Path | None, **kw) -> dict:
    """``P.render_clips`` that can never fail the caller: on any error the clips stay symlinks (slow seeks, correct)."""
    try:
        r = P.render_clips(doc, root, **kw)
    except Exception as e:  # noqa: BLE001 - rendering is an optimisation, never fail the job for it
        logging.getLogger(__name__).warning(
            "render_clips failed for %s, keeping symlinks: %r", doc.get("match"), e
        )
        return {"error": repr(e)[:300]}
    return {k: (v if k == "bytes" else len(v)) for k, v in r.items()} | {
        "failed_clips": r["failed"]
    }


def cmd_render(args: argparse.Namespace) -> int:
    """Replace the symlinks in the panel's clips dir by keyframe-dense re-encodes (idempotent; a failed clip keeps its symlink)."""
    rc = 0
    root = Path(args.out_dir) if args.out_dir else args.root
    # default: every panel file of the version (the panels, not the heldout matches that might get one)
    matches = (
        list(args.match)
        if args.match
        else list(P.list_panels(args.panel_version, root))
    )
    for match in matches:
        try:
            path = P.panel_path(match, args.panel_version, root)
            if not path.is_file():
                raise P.PanelError(f"no panel for {match} ({path}); run `build` first")
            doc = P.read_panel(path)
            P.require_heldout(match)
            P.link_clips(
                doc, args.root
            )  # every name resolvable first: a failed render then still leaves a working symlink
            t = time.time()
            r = _render_best_effort(
                doc,
                args.root,
                force=args.force,
                gop=args.gop,
                crf=args.crf,
                threads=args.threads,
            )
            print(
                json.dumps(
                    {
                        "match": match,
                        "status": "error" if "error" in r else "ok",
                        **r,
                        "seconds": round(time.time() - t, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            rc = rc or int("error" in r or bool(r.get("failed")))
        except (P.PanelError, OSError, ValueError) as e:
            print(
                json.dumps({"match": match, "status": "error", "error": str(e)}),
                flush=True,
            )
            rc = 1
    return rc


def cmd_job(args: argparse.Namespace) -> int:
    from footy_track.labeller import jobs_client  # noqa: PLC0415

    rc = 0
    for match in _matches(args):
        try:
            path = P.panel_path(
                match,
                args.panel_version,
                Path(args.out_dir) if args.out_dir else args.root,
            )
            if not path.is_file():
                raise P.PanelError(f"no panel for {match} ({path}); run `build` first")
            doc = P.read_panel(path)
            P.require_heldout(
                match
            )  # the gate again: the job path must never carry a train match
            body = P.job_body(doc)
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "match": match,
                            "status": "dry-run",
                            "key": body["key"],
                            "dataset": body["dataset"],
                            "n_items": len(body["items"]),
                            "n_cards": doc["n_frames"] + len(doc["qa"]),
                        }
                    )
                )
                continue
            made = P.link_clips(doc, args.root)
            rendered = None if args.no_render else _render_best_effort(doc, args.root)
            registered = P.register_dataset(args.panel_version, args.root)
            job = jobs_client.submit(
                body["title"],
                body["kind"],
                body["items"],
                dataset=body["dataset"],
                why=body["why"],
                device=body["device"],
                priority=body["priority"],
                key=body["key"],
                url=args.labeller_url,
            )
            print(
                json.dumps(
                    {
                        "match": match,
                        "status": "created" if job.get("created") else "exists",
                        "job_id": job.get("id"),
                        "job_url": job.get("url"),
                        "key": body["key"],
                        "dataset": body["dataset"],
                        "n_cards": doc["n_frames"] + len(doc["qa"]),
                        "clip_links_made": made,
                        "render": rendered,
                        "dataset_registered": registered,
                    },
                    sort_keys=True,
                )
            )
        except (P.PanelError, OSError, ValueError, jobs_client.JobsError) as e:
            print(
                json.dumps({"match": match, "status": "error", "error": str(e)}),
                flush=True,
            )
            rc = 1
    return rc


def _load_panels(args: argparse.Namespace) -> list[dict]:
    paths = P.list_panels(args.panel_version, args.root)
    names = args.match or list(paths)
    missing = [m for m in names if m not in paths]
    if missing:
        raise SystemExit(f"no {args.panel_version} panel for: {', '.join(missing)}")
    return [P.read_panel(paths[m]) for m in names]


def _human(args: argparse.Namespace, panels: list[dict]) -> dict:
    clips = {w["clip"] for p in panels for w in p["windows"]} | {
        q["clip"] for p in panels for q in p["qa"]
    }
    clips |= {q["of"]["clip"] for p in panels for q in p["qa"]}
    human = S.load_human_sidecars(
        Path(args.human_marks)
        if args.human_marks
        else P.marks_dir(args.panel_version, args.root),
        clips,
    )
    for res in args.human_results or []:
        human.update(S.load_human_results(Path(res)))
    return human


def _predictions(args: argparse.Namespace, panels: list[dict]) -> dict:
    if args.pred_format == "jsonl":
        if (
            len({(p["width"], p["height"]) for p in panels}) > 1
            and args.pred_units == "norm"
        ):
            raise SystemExit(
                "normalised predictions need one frame size; give px predictions"
            )
        size = (panels[0]["width"], panels[0]["height"]) if panels else None
        return S.load_predictions(
            Path(args.pred), units=args.pred_units, size=size, min_score=args.min_score
        )
    out: dict = {}
    if args.pred_format == "suggest":
        d = Path(args.pred)
        for p in panels:
            f = S.suggest_file_for(d, p["match"], args.suggest_ident)
            if f is None:
                print(f"WARNING: no suggest file for {p['match']} in {d}", file=sys.stderr)
                continue
            out.update(S.predictions_from_suggest_file(f, p, min_score=args.min_score))
        return out
    for p in panels:
        size = (p["width"], p["height"])
        for clip in sorted({w["clip"] for w in p["windows"]}):
            f = S.find_clip_file(Path(args.pred), clip)
            if f is None:
                continue
            fn = (
                S.predictions_from_machine_labels
                if args.pred_format == "machine_labels"
                else S.predictions_from_cache
            )
            out.update(fn(f, clip, size, min_score=args.min_score))
    return out


def cmd_score(args: argparse.Namespace) -> int:
    panels = _load_panels(args)
    report = S.score(
        panels,
        _human(args, panels),
        _predictions(args, panels),
        radii=tuple(args.radius),
        draws=args.draws,
        seed=args.seed,
        include_partial=args.include_partial,
        model=args.model,
        blind_only=args.blind_only,
        root=args.root,
        require_predictions=True,
    )
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(_pretty(report) if args.pretty else text)
    for w in report["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    if report["matches_without_predictions"] and not args.allow_missing_predictions:
        print(
            "ERROR: no predictions for "
            + ", ".join(report["matches_without_predictions"])
            + " (pass --allow-missing-predictions to score the rest anyway)",
            file=sys.stderr,
        )
        return 2
    return 0


def _pretty(rep: dict) -> str:
    lines = [
        f"panel {rep['panel_version']}  model {rep['model']}  radii {rep['radii_px']} px  draws {rep['draws']} seed {rep['seed']}"
    ]

    def fmt(c: dict) -> str:
        return "  n/a" if c["est"] is None else f"{c['est']:.3f} [{c['lo']}, {c['hi']}]"

    for scope, block in [
        ("match-averaged", rep["pooled"].get("match_averaged", {})),
        ("frame-weighted", rep["pooled"].get("frame_weighted", {})),
        *[(m, d.get("metrics", {})) for m, d in rep["matches"].items() if d.get("metrics")],
    ]:
        for r, cells in block.items():
            lines.append(
                f"{scope:<42} {r:<5} recall {fmt(cells['recall'])}  precision {fmt(cells['precision'])}  false_ball {fmt(cells['false_ball'])}"
            )
    lines += [f"WARNING: {w}" for w in rep.get("warnings", [])]
    lines.append("QA " + json.dumps(rep["qa"], sort_keys=True))
    sg = rep.get("suggestions") or {}
    for scope, blk in [("pooled", sg.get("pooled")), *sorted((sg.get("matches") or {}).items())]:
        if not blk:
            continue
        cmp_ = blk["comparison"]
        lines.append(
            f"suggestions {scope}: accepted {blk['accepted']}/{blk['labelled_balls']} ball labels "
            f"(shown windows {blk['shown_accepted']}/{blk['shown_balls']}); windows blind {cmp_['n_windows_blind']} "
            f"shown {cmp_['n_windows_shown']}"
        )
        for r, cells in cmp_["diff_blind_minus_shown"].items():
            lines.append(
                f"   blind-shown {r:<5} recall {fmt(cells['recall'])}  false_ball {fmt(cells['false_ball'])}"
            )
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    from footy_track.labeller import jobs_client  # noqa: PLC0415

    rows = []
    for match, path in P.list_panels(args.panel_version, args.root).items():
        doc = P.read_panel(path)
        clips = {w["clip"] for w in doc["windows"]} | {q["clip"] for q in doc["qa"]}
        human = S.load_human_sidecars(P.marks_dir(args.panel_version, args.root), clips)
        frames = [(w["clip"], f) for w in doc["windows"] for f in w["frames"]]
        qa = [(q["clip"], q["frame"]) for q in doc["qa"]]
        row = {
            "match": match,
            "n_windows": doc["n_windows"],
            "N_windows": doc["N_windows"],
            "n_frames": len(frames),
            "n_qa": len(qa),
            "labelled_frames": sum(k in human for k in frames),
            "labelled_qa": sum(k in human for k in qa),
            "created_at": doc["created_at"],
            "code_sha": doc["code_sha"],
        }
        if args.labeller_url:
            try:
                key = P.job_key(match, args.panel_version)
                import hashlib  # noqa: PLC0415

                jid = "j-" + hashlib.sha1(key.encode()).hexdigest()[:12]  # noqa: S324 (jobs._sha_id)
                job = jobs_client.get(jid, url=args.labeller_url)
                row["job"] = {
                    "id": jid,
                    "status": job.get("status"),
                    "progress": job.get("progress"),
                }
            except jobs_client.JobsError as e:
                row["job"] = {"error": str(e)}
            except OSError as e:
                row["job"] = {"unreachable": str(e)}
        rows.append(row)
    print(
        json.dumps(
            {
                "panel_version": args.panel_version,
                "panels": rows,
                "segmented_heldout_without_panel": [
                    m
                    for m in P.segmented_heldout_matches(args.root)
                    if m not in {r["match"] for r in rows}
                ],
            },
            indent=1,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--panel-version", default=P.PANEL_VERSION)
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="data root (default $FOOTY_DATA or /mnt/storage/footy_data)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--match", action="append", help="heldout match key (repeatable)"
        )
        sp.add_argument(
            "--all-heldout", action="store_true", help="every segmented heldout match"
        )
        sp.add_argument(
            "--out-dir",
            help="write/read panel files under this root instead of --root (scratch)",
        )

    b = sub.add_parser("build", help="sample the panel of a heldout match")
    common(b)
    b.add_argument("--n-windows", type=int, default=P.N_WINDOWS_DEFAULT)
    b.add_argument(
        "--dry-run", action="store_true", help="compute and print, write nothing"
    )
    b.add_argument(
        "--force",
        action="store_true",
        help="replace an existing, different panel file (old one kept)",
    )
    b.add_argument(
        "--gold-dir",
        action="append",
        help="marks dir to take tier-1 gold from (default: labeller registry)",
    )
    b.add_argument(
        "--no-probe",
        action="store_true",
        help="do not open the clips (tests; the panel then has no frame size)",
    )
    b.set_defaults(fn=cmd_build)

    j = sub.add_parser("job", help="submit the blind ball_place job for a built panel")
    common(j)
    j.add_argument("--labeller-url", default=None, help="default $FOOTY_LABELLER_URL")
    j.add_argument("--dry-run", action="store_true")
    j.add_argument(
        "--no-render",
        action="store_true",
        help="leave the clips as symlinks (skip the keyframe-dense render)",
    )
    j.set_defaults(fn=cmd_job)

    g = sub.add_parser(
        "suggest",
        help="precompute detector (top-1 RT-DETR ball box) suggestions for the panel frames",
    )
    common(g)
    g.add_argument("--weights", default=None, help="detector weights (default: the current best)")
    g.add_argument("--device", default="cpu")
    g.add_argument("--imgsz", type=int, default=1280)
    g.add_argument("--batch", type=int, default=4)
    g.add_argument("--tag", default="", help="write <match>.suggest.<sha8><tag>.json beside the existing file (top-k recompute; the old file and its blind-equivalent set are kept)")
    g.set_defaults(fn=cmd_suggest)

    r = sub.add_parser(
        "render",
        help="re-encode the panel's clips with a keyframe every 10 frames (same frames, same names)",
    )
    r.add_argument(
        "--match",
        action="append",
        help="heldout match key (repeatable; default: every panel of the version)",
    )
    r.add_argument(
        "--out-dir",
        help="read the panel files under this root instead of --root (scratch)",
    )
    r.add_argument(
        "--force", action="store_true", help="re-render clips that are already rendered"
    )
    r.add_argument("--gop", type=int, default=P.KF_GOP)
    r.add_argument("--crf", type=int, default=P.KF_CRF)
    r.add_argument("--threads", type=int, default=P.KF_THREADS)
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser(
        "score", help="HT recall/precision/false-ball with bootstrap CIs"
    )
    s.add_argument(
        "--match",
        action="append",
        help="only these matches (default: every panel of the version)",
    )
    s.add_argument(
        "--pred",
        required=True,
        help="predictions jsonl, or a directory with --pred-format machine_labels|cache|suggest "
        "(suggest: the panel dir holding <match>.suggest.<sha8>.json)",
    )
    s.add_argument(
        "--pred-format",
        choices=["jsonl", "machine_labels", "cache", "suggest"],
        default="jsonl",
    )
    s.add_argument(
        "--suggest-ident",
        default=None,
        help="--pred-format suggest: exact file ident, e.g. b4ea4b22 or b4ea4b22k5 (default: the plain sha8 file)",
    )
    s.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="exit 0 even when a match has no predictions (it is still reported and left out of the pooled numbers)",
    )
    s.add_argument("--pred-units", choices=["px", "norm"], default="px")
    s.add_argument("--min-score", type=float, default=0.0)
    s.add_argument(
        "--human-marks", help="sidecar dir (default: the panel dataset's marks dir)"
    )
    s.add_argument(
        "--human-results",
        action="append",
        help="labelling-job results json (repeatable)",
    )
    s.add_argument(
        "--radius",
        type=float,
        action="append",
        default=None,
        help="px, repeatable (default 10 25)",
    )
    s.add_argument("--draws", type=int, default=S.DEFAULT_DRAWS)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--include-partial",
        action="store_true",
        help="score windows with unlabelled frames too",
    )
    s.add_argument("--model", default=None)
    s.add_argument(
        "--blind-only",
        action="store_true",
        help="score only blind windows (random 25% control + labels placed before suggestions existed): the "
        "suggesting detector is otherwise the model most favoured by its own suggestions",
    )
    s.add_argument("--out", help="write the JSON report here")
    s.add_argument(
        "--pretty", action="store_true", help="human-readable summary instead of JSON"
    )
    s.set_defaults(fn=cmd_score)

    t = sub.add_parser(
        "status",
        help="panels, labelling progress and segmented heldout matches without a panel",
    )
    t.add_argument("--labeller-url", default=None)
    t.set_defaults(fn=cmd_status)

    args = ap.parse_args(argv)
    if args.cmd == "score" and not args.radius:
        args.radius = list(S.DEFAULT_RADII)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
