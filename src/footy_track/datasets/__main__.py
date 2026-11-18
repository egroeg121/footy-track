"""``python -m footy_track.datasets {build,bump,list,show,diff,results}``.

    build <dataset>            assemble the item table from the store; print counts + hash and
                               whether it differs from the latest snapshot (writes nothing)
    bump  <dataset|all>        write the next immutable snapshot iff the content hash changed
    list  [dataset]            versions on disk
    show  <dataset> [vN]       a manifest (default: latest)
    diff  <dataset> A B        item-level diff between two versions
    results list               eval_result rows (newest per key)
    results record ...         append one metric row (or --json file of rows)

Store: ``--db`` > ``$FOOTY_FEATURE_STORE`` > /mnt/storage/footy_data/feature_store/feature_store.duckdb.
Snapshots: ``--root`` > ``$FOOTY_DATASETS_ROOT`` > ``$FOOTY_DATA`` > /mnt/storage/footy_data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

from footy_track.datasets import results as res
from footy_track.datasets import snapshot as snap
from footy_track.datasets.items import DATASETS
from footy_track.feature_store.lock import default_db_path, store_lock


def _fmt_counts(c: dict) -> list[str]:
    out = [f"  items: {c['n_items']}"]
    for split, n in c["by_split"].items():
        kinds = ", ".join(f"{k}={v}" for k, v in c["by_split_kind"].get(split, {}).items())
        out.append(f"  {split:20s} {n:6d}  clips={c['clips_by_split'].get(split, 0):3d}  [{kinds}]")
    return out


def _cmd_build(a: argparse.Namespace) -> int:
    db = a.db or default_db_path()
    if not Path(db).exists():
        print(f"no store at {db}; run `python -m footy_track.feature_store.ingest_all` first", file=sys.stderr)
        return 2
    with store_lock(db, exclusive=False, timeout=30):
        conn = duckdb.connect(str(db), read_only=True)
        try:
            built = snap.build(a.dataset, conn)
        finally:
            conn.close()
    latest = snap.latest_version(a.dataset, a.root)
    parent = snap.load_version(a.dataset, latest, a.root) if latest else None
    print(f"{a.dataset}: content hash {built.content_hash}")
    for line in _fmt_counts(snap.count_summary(built.rows({}, (latest or 0) + 1))):
        print(line)
    if parent is None:
        print("no snapshot yet: `bump` would write v1")
    elif parent.content_hash == built.content_hash:
        print(f"identical to v{latest}: `bump` would write nothing")
    else:
        rows = built.rows({r["item_id"]: r["first_version"] for r in parent.rows()}, latest + 1)
        d = snap.diff_items({r["item_id"]: r for r in parent.rows()}, {r["item_id"]: r for r in rows})
        print(f"differs from v{latest}: +{d['added']} -{d['removed']} ~{d['changed']}; `bump` would write v{latest + 1}")
    return 0


def _cmd_bump(a: argparse.Namespace) -> int:
    rc = 0
    for ds in DATASETS if a.dataset == "all" else [a.dataset]:
        try:
            r = snap.bump(ds, a.db, a.root, note=a.note, allow_preview=a.allow_preview, allow_empty=a.allow_empty)
        except Exception as e:  # noqa: BLE001 - one dataset failing must not stop `bump all`
            print(f"{ds}: {e}", file=sys.stderr)
            rc = 1
            continue
        if r.changed:
            d = r.manifest["diff_vs_parent"]
            print(f"{ds}: wrote v{r.version} ({r.manifest['counts']['n_items']} items; +{d['added']} -{d['removed']} ~{d['changed']}) {r.path}")
        else:
            print(f"{ds}: unchanged at v{r.version} ({r.content_hash[:12]}); nothing written")
    return rc


def _cmd_list(a: argparse.Namespace) -> int:
    for ds in [a.dataset] if a.dataset else DATASETS:
        versions = snap.list_versions(ds, a.root)
        print(f"{ds}: {', '.join(f'v{v}' for v in versions) or '(no snapshots)'}")
        for v in versions:
            m = json.loads((snap.version_dir(ds, v, a.root) / "manifest.json").read_text())
            print(f"  v{v}  {m['created_at']}  {m['content_hash'][:12]}  items={m['counts']['n_items']}  "
                  f"{m['counts']['by_split']}  {m.get('note') or ''}")
    return 0


def _cmd_show(a: argparse.Namespace) -> int:
    dv = snap.load_version(a.dataset, int(a.version.lstrip("v")) if a.version else None, a.root)
    print(json.dumps(dv.manifest, indent=1, sort_keys=True))
    return 0


def _cmd_diff(a: argparse.Namespace) -> int:
    d = snap.diff_versions(a.dataset, int(a.a.lstrip("v")), int(a.b.lstrip("v")), a.root)
    for k in ("added_ids", "removed_ids", "changed_ids"):
        d[k] = d[k][: a.limit] + (["..."] if len(d[k]) > a.limit else [])
    print(json.dumps(d, indent=1, sort_keys=True))
    return 0


def _cmd_results(a: argparse.Namespace) -> int:
    if a.action == "list":
        rows = res.list_results(
            a.db, dataset=a.dataset, task=a.task, model=a.model, dataset_version=a.dataset_version,
            split=a.split, latest_only=not a.history,
        )
        for r in rows:
            c = "" if r["contaminated"] is None else (" CONTAMINATED" if r["contaminated"] else " clean")
            cohort = "-" if r["cohort_version"] is None else f"c{r['cohort_version']}"
            v = "None" if r["value"] is None else f"{r['value']:.4f}"
            print(f"{r['dataset']} v{r['dataset_version']} {r['split']:19s} {cohort:4s} "
                  f"{r['model_name']}@{r['model_version'][:12]:12s} {r['metric']:28s} {v:>8s} n={r['n_items']}{c}")
        print(f"({len(rows)} rows)")
        return 0
    training = None
    if a.train_roboflow_dir or a.train_dataset:
        training = res.TrainingData(
            roboflow_dir=a.train_roboflow_dir,
            dataset=a.train_dataset, version=a.train_version, root=a.root,
        )
    if a.json:
        raw = json.loads(Path(a.json).read_text())
        rows = [res.ResultRow(**r) for r in raw]
    else:
        if not (a.dataset and a.dataset_version is not None and a.model and a.metric and a.value is not None and a.task and a.split):
            print("record needs --task --dataset --dataset-version --split --model --metric --value (or --json)", file=sys.stderr)
            return 2
        dv = snap.load_version(a.dataset, a.dataset_version, a.root)
        flags = res.contamination_flags(training, dv)
        flag = flags[a.split]
        rows = [res.ResultRow(
            task=a.task, dataset=a.dataset, dataset_version=a.dataset_version, split=a.split,
            cohort_version=a.cohort_version, model_name=a.model, model_version=a.model_version or "unknown",
            metric=a.metric, value=a.value, n_items=a.n_items,
            contaminated=flag.contaminated, contamination_note=flag.note,
        )]
    n = res.record_results(a.db, rows)
    print(f"recorded {n} row(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m footy_track.datasets", description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="assemble + summarise (no write)")
    p.add_argument("dataset", choices=DATASETS)
    p.set_defaults(fn=_cmd_build)

    p = sub.add_parser("bump", help="write the next snapshot iff the content changed")
    p.add_argument("dataset", choices=(*DATASETS, "all"))
    p.add_argument("--note", default=None)
    p.add_argument("--allow-preview", action="store_true", help="bump even though a split register file is missing")
    p.add_argument("--allow-empty", action="store_true")
    p.set_defaults(fn=_cmd_bump)

    p = sub.add_parser("list")
    p.add_argument("dataset", nargs="?", choices=DATASETS)
    p.set_defaults(fn=_cmd_list)

    p = sub.add_parser("show")
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("version", nargs="?")
    p.set_defaults(fn=_cmd_show)

    p = sub.add_parser("diff")
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=_cmd_diff)

    p = sub.add_parser("results", help="eval_result: list / record")
    p.add_argument("action", choices=("list", "record"))
    p.add_argument("--task", choices=res.TASKS)
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--dataset-version", type=int)
    p.add_argument("--split", choices=res.RESULT_SPLITS)
    p.add_argument("--cohort-version", type=int, default=None)
    p.add_argument("--model")
    p.add_argument("--model-version")
    p.add_argument("--metric")
    p.add_argument("--value", type=float)
    p.add_argument("--n-items", type=int)
    p.add_argument("--json", type=Path, help="record: a JSON list of ResultRow fields instead of the flags")
    p.add_argument("--train-roboflow-dir", type=Path, help="record: the model's Roboflow/YOLO training export (sets contaminated)")
    p.add_argument("--train-dataset", choices=DATASETS, help="record: the model was trained on this dataset's train split ...")
    p.add_argument("--train-version", type=int, help="... at this version")
    p.add_argument("--history", action="store_true", help="list: every run, not just the newest per key")
    p.set_defaults(fn=_cmd_results)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
