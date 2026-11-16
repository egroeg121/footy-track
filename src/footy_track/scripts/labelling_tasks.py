"""Edit the shared labelling task board (the "To do now" list on the labeller home page).

    python -m footy_track.scripts.labelling_tasks list
    python -m footy_track.scripts.labelling_tasks add "Title" --url URL --why "one line" \
        --device phone --priority 2 [--count-url URL --count-label "left"] [--owner NAME] \
        [--done-when-gt CLIP:FRAME | --done-when-frames CLIP:FRAME,CLIP:FRAME,... | --done-when-count-zero]
    python -m footy_track.scripts.labelling_tasks done <id>

A task with a done-when completes itself (greyed into the "Done" section of the board);
``done <id>`` still removes one by hand. See labeller/task_done.py and README section 17.

The file is ``$FOOTY_TASKS_FILE`` (default /mnt/storage/footy_data/labelling_tasks.json).
Writes are atomic; ids are stable. The servers re-read the file per request.
"""

from __future__ import annotations

import argparse
import sys

from footy_track.labeller import tasks as T


def _load_or_die() -> list[dict]:
    tasks, err = T.load_tasks()
    if err:
        sys.exit(f"refusing to edit a malformed tasks file: {err}")
    return tasks


def _clip_frame(spec: str) -> tuple[str, int]:
    clip, sep, frame = spec.strip().rpartition(":")
    if not sep or not clip or not frame.isdigit():
        sys.exit(f"expected CLIP:FRAME, got {spec!r}")
    return clip, int(frame)


def cmd_list(args: argparse.Namespace) -> int:
    tasks, err = T.load_tasks()
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"# {T.tasks_path()}")
    for t in tasks:
        print(f"{t.get('id', '?'):<24} p{t.get('priority', 2)} {t.get('device', 'either'):<6} {t['title']}  {t.get('url', '')}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    tasks = _load_or_die()
    task = {
        "id": T.new_task_id(tasks),
        "title": args.title,
        "why": args.why or "",
        "url": args.url or "",
        "device": args.device,
        "priority": args.priority,
        "added": T.today(),
    }
    if args.count_url:
        task["count_url"] = args.count_url
        task["count_label"] = args.count_label or "left"
    if args.done_when_gt:
        clip, frame = _clip_frame(args.done_when_gt)
        task["done_when"] = {"gt": {"clip": clip, "frame": frame, "since": "added"}}
    elif args.done_when_frames:
        task["done_when"] = {"frames": [list(_clip_frame(x)) for x in args.done_when_frames.split(",") if x.strip()]}
    elif args.done_when_count_zero:
        if not args.count_url:
            sys.exit("--done-when-count-zero needs --count-url")
        task["done_when"] = {"count_zero": True}
    if args.owner:
        task["owner"] = args.owner
    tasks.append(task)
    T.write_tasks_atomic(T.tasks_path(), tasks)
    print(task["id"])
    return 0


def cmd_done(args: argparse.Namespace) -> int:
    tasks = _load_or_die()
    keep = [t for t in tasks if t.get("id") != args.id]
    if len(keep) == len(tasks):
        print(f"no task with id {args.id}", file=sys.stderr)
        return 1
    T.write_tasks_atomic(T.tasks_path(), keep)
    print(f"removed {args.id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    a = sub.add_parser("add")
    a.add_argument("title")
    a.add_argument("--url")
    a.add_argument("--why")
    a.add_argument("--device", choices=T.DEVICES, default="either")
    a.add_argument("--priority", type=int, default=2, help="1 = first (default 2)")
    a.add_argument("--count-url", help="GET returning JSON with total/count/remaining")
    a.add_argument("--count-label")
    a.add_argument("--owner")
    w = a.add_mutually_exclusive_group()
    w.add_argument("--done-when-gt", metavar="CLIP:FRAME", help="done when calibration GT exists for that frame")
    w.add_argument("--done-when-frames", metavar="CLIP:FRAME,...",
                   help="done when every frame has GT or a not_broadcast / too_few_points mark")
    w.add_argument("--done-when-count-zero", action="store_true", help="done when --count-url returns 0")
    a.set_defaults(fn=cmd_add)
    d = sub.add_parser("done")
    d.add_argument("id")
    d.set_defaults(fn=cmd_done)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
