"""Edit the shared labelling task board (the "To do now" list on the labeller home page).

    python -m footy_track.scripts.labelling_tasks list
    python -m footy_track.scripts.labelling_tasks add "Title" --url URL --why "one line" \
        --device phone --priority 2 [--count-url URL --count-label "left"] [--owner NAME]
    python -m footy_track.scripts.labelling_tasks done <id>

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
    a.set_defaults(fn=cmd_add)
    d = sub.add_parser("done")
    d.add_argument("id")
    d.set_defaults(fn=cmd_done)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
