# ruff: noqa: PLR0911, PLR0912, PLR0915, PLR0917, PLW2901, SIM105
"""Rename a match everywhere in the footy_data tree (and optionally S3).

Written for "Norwich" -> "Nottingham Forest" (the held-out match was misnamed:
the scoreboard says NFO), but generic in the old and new names::

    # 1. read-only plan with counts (the default; writes nothing)
    python -m footy_track.scripts.migrate_rename_match
    # 2. do it (refuses if the live labeller wrote to an affected path in the last 10 min)
    python -m footy_track.scripts.migrate_rename_match --apply
    # 3. undo, from the manifest written by step 2
    python -m footy_track.scripts.migrate_rename_match --rollback /mnt/storage/footy_data/_backups/rename_<ts>.manifest.json

What it does, with the defaults (``--old-match arsenal_norwich --new-match
arsenal_nottinghamforest --old-short norwich --new-short nottinghamforest
--old-title Norwich --new-title "Nottingham Forest"``):

* **Renames** every directory, file and symlink under ``--root`` whose *name*
  contains the old match/short name (``arsenal_norwich/``, ``ball_gt_marks_norwich/``,
  ``arsenal_norwich_seg003.jsonl``, ...) with ``os.rename`` (atomic within one
  filesystem), deepest entries first.
* **Rewrites** ids inside ``.json/.jsonl/.csv/.txt/.md/.yaml`` files (sidecar ``clip``
  fields, ``windows.csv``, ``ball_check_list_v1.json``, manifests, the labeller registry
  ``labeller_datasets.json``, ``labelling_jobs/*.json``, ``labelling_tasks.json``) with a
  temp file + ``os.replace``, and checks that JSON / JSONL still parses before replacing.
* **Backs up** every text file it rewrites or renames to
  ``<root>/_backups/rename_<utc timestamp>.tar.gz`` *before changing anything*, and writes
  ``rename_<ts>.manifest.json`` next to it (renames, rewrites with before/after sha256).
* **Is idempotent**: it re-plans from the current disk state, so re-running after a
  complete (or crashed) run only does what is left; an empty plan is the "all done" check.
* **Refuses ``--apply``** if any file or directory it would touch was modified in the last
  ``--live-window-min`` (default 10) minutes (the live labeller is writing there),
  unless ``--force``.
* ``--rollback MANIFEST`` reverses the renames and restores the rewritten files from the
  tar. A file changed after the migration (sha256 != the manifest's ``sha_after``) is
  skipped with a warning (restoring would drop labels written since) unless ``--force``.
* ``--include-s3`` adds a plan (dry-run) / execution (``--apply``) for the S3 copy under
  ``--s3-prefix``. S3 has no rename: renamed objects are copied under the new key
  (text label files are uploaded from the migrated local copy, everything else is a
  server-side ``aws s3 cp``), and the old keys are deleted only with ``--s3-delete-old``.

Not renamed on purpose: ``classifier_dataset/`` (Roboflow exports; those image names are
fixed by Roboflow, see ``footy_track.clip_id`` for the alias that keeps them resolving),
``_backups/``. Add more with ``--exclude`` (paths relative to ``--root``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ROOT = "/mnt/storage/footy_data"
DEFAULT_S3_PREFIX = "s3://georgebarnett-general-purpose/footy_data/"
DEFAULT_EXCLUDES = ("classifier_dataset", "_backups")
TEXT_SUFFIXES = frozenset(
    {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml"}
)
MAX_TEXT_BYTES = 256 * 1024 * 1024
MANIFEST_VERSION = 1

#: Applied (to file *contents* only) before the generic rules. The registry label used to
#: say "ids say 'norwich'"; after the rename that sentence would be nonsense.
DEFAULT_LITERALS: tuple[tuple[str, str], ...] = (
    (
        "Arsenal v Nottingham Forest (held-out; ids say 'norwich')",
        "Arsenal v Nottingham Forest (held-out)",
    ),
)

EXIT_OK, EXIT_ERROR, EXIT_CONFLICT, EXIT_LIVE = 0, 1, 2, 3


class MigrationError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rules:
    old_match: str = "arsenal_norwich"
    new_match: str = "arsenal_nottinghamforest"
    old_short: str = "norwich"
    new_short: str = "nottinghamforest"
    old_title: str = "Norwich"
    new_title: str = "Nottingham Forest"
    literals: tuple[tuple[str, str], ...] = DEFAULT_LITERALS

    def _pairs(self) -> tuple[tuple[str, str], ...]:
        return (
            (self.old_match, self.new_match),
            (self.old_short, self.new_short),
            (self.old_title, self.new_title),
        )

    def validate(self) -> None:
        olds = [o for o, _ in self._pairs()]
        if not all(olds):
            raise MigrationError("old names must be non-empty")
        for o, n in self._pairs():
            if o == n:
                raise MigrationError(f"old and new are identical: {o!r}")
            for old in olds:
                if old in n:
                    raise MigrationError(
                        f"new name {n!r} contains the old name {old!r}; the rename would not be idempotent"
                    )
        for _lo, ln in self.literals:
            if any(old in ln for old in olds):
                raise MigrationError(
                    f"literal replacement {ln!r} still contains an old name"
                )

    @property
    def token_re(self) -> re.Pattern[str]:
        toks = sorted({o for o, _ in self._pairs()}, key=len, reverse=True)
        return re.compile("|".join(re.escape(t) for t in toks))

    @property
    def residual_re(self) -> re.Pattern[str]:
        """Any case variant of the short name (NORWICH, Norwich...) after rewriting."""
        return re.compile(re.escape(self.old_short), re.IGNORECASE)

    def name(self, s: str) -> str:
        """Rename a path component / id."""
        for o, n in self._pairs():
            s = s.replace(o, n)
        return s

    def text(self, s: str) -> str:
        """Rewrite file contents."""
        for o, n in self.literals:
            s = s.replace(o, n)
        return self.name(s)

    def has_token(self, s: str) -> bool:
        return self.token_re.search(s) is not None


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


@dataclass
class Rename:
    src: Path
    dst: Path
    kind: str  # "dir" | "file" | "link"


@dataclass
class Rewrite:
    path: Path  # path at plan time (before any rename)
    hits: int
    size: int


@dataclass
class Relink:
    path: Path
    old_target: str
    new_target: str


@dataclass
class Plan:
    root: Path
    rules: Rules
    renames: list[Rename] = field(default_factory=list)
    rewrites: list[Rewrite] = field(default_factory=list)
    relinks: list[Relink] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    excluded: Counter = field(
        default_factory=Counter
    )  # excluded dir -> entries with the old name
    backup_files: list[Path] = field(default_factory=list)
    scanned_text_files: int = 0
    scanned_entries: int = 0

    @property
    def n_dirs(self) -> int:
        return sum(r.kind == "dir" for r in self.renames)

    @property
    def n_files(self) -> int:
        return sum(r.kind != "dir" for r in self.renames)

    @property
    def backup_bytes(self) -> int:
        total = 0
        for p in self.backup_files:
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    @property
    def empty(self) -> bool:
        return not (self.renames or self.rewrites or self.relinks)


def _is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def build_plan(
    root: Path,
    rules: Rules,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
    max_text_bytes: int = MAX_TEXT_BYTES,
) -> Plan:
    """Walk *root* (read-only) and work out every rename / rewrite / relink."""
    root = Path(root)
    plan = Plan(root=root, rules=rules)
    excl = {e.strip("/") for e in excludes}
    dst_seen: dict[Path, Path] = {}
    backup: set[Path] = set()

    def count_named(d: Path) -> int:
        n = 0
        for _dp, dns, fns in os.walk(d):
            n += sum(rules.has_token(x) for x in dns) + sum(
                rules.has_token(x) for x in fns
            )
        return n

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        d = Path(dirpath)
        keep = []
        for dn in dirnames:
            rel = _rel(d / dn, root)
            if rel in excl:
                n = count_named(d / dn) + (1 if rules.has_token(dn) else 0)
                if n:
                    plan.excluded[rel] += n
            else:
                keep.append(dn)
        dirnames[:] = keep

        entries = [(dn, "dir") for dn in dirnames] + [(fn, "file") for fn in filenames]
        for name, kind in entries:
            plan.scanned_entries += 1
            p = d / name
            if p.is_symlink():
                kind = "link"
                target = os.readlink(p)
                if rules.has_token(target):
                    plan.relinks.append(Relink(p, target, rules.name(target)))
            if rules.has_token(name):
                dst = d / rules.name(name)
                if os.path.lexists(dst):
                    plan.conflicts.append(f"destination already exists: {p} -> {dst}")
                elif dst in dst_seen:
                    plan.conflicts.append(
                        f"two sources map to {dst}: {dst_seen[dst]} and {p}"
                    )
                else:
                    dst_seen[dst] = p
                    plan.renames.append(Rename(p, dst, kind))
            if kind == "file" and _is_text(p):
                plan.scanned_text_files += 1
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if size > max_text_bytes:
                    plan.warnings.append(
                        f"text file over size cap, not scanned: {p} ({size} bytes)"
                    )
                    continue
                try:
                    text = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    plan.warnings.append(f"not valid UTF-8, not scanned: {p}")
                    continue
                hits = len(rules.token_re.findall(text)) + sum(
                    text.count(lo) for lo, _ in rules.literals
                )
                if hits:
                    plan.rewrites.append(Rewrite(p, hits, size))
                    backup.add(p)
                    if rules.residual_re.search(rules.text(text)):
                        plan.warnings.append(
                            f"other-case variant of {rules.old_short!r} survives in {p}"
                        )

    # text files that are renamed, or live under a renamed directory, are backed up too
    for r in plan.renames:
        if r.kind == "file" and _is_text(r.src):
            backup.add(r.src)
        elif r.kind == "dir":
            for dp, dns, fns in os.walk(r.src):
                dns[:] = [x for x in dns if _rel(Path(dp) / x, root) not in excl]
                for fn in fns:
                    fp = Path(dp) / fn
                    if _is_text(fp) and not fp.is_symlink():
                        try:
                            if fp.stat().st_size <= max_text_bytes:
                                backup.add(fp)
                        except OSError:
                            pass
    plan.backup_files = sorted(backup)
    plan.renames.sort(key=lambda r: (-len(r.src.parts), str(r.src)))
    return plan


# --------------------------------------------------------------------------
# live-write guard
# --------------------------------------------------------------------------


def touched_paths(plan: Plan) -> list[Path]:
    """Every path the migration would change, plus everything inside renamed directories."""
    out: dict[Path, None] = {}
    for rw in plan.rewrites:
        out[rw.path] = None
    for rn in plan.renames:
        out[rn.src] = None
        if rn.kind == "dir":
            for dp, dns, fns in os.walk(rn.src):
                for x in dns + fns:
                    out[Path(dp) / x] = None
    return list(out)


def newest_write(
    paths: Sequence[Path], ignore: Callable[[Path], bool] | None = None
) -> tuple[float, Path | None]:
    best, best_p = 0.0, None
    for p in paths:
        if ignore and ignore(p):
            continue
        try:
            m = p.lstat().st_mtime
        except OSError:
            continue
        if m > best:
            best, best_p = m, p
    return best, best_p


def live_guard(
    paths: Sequence[Path], window_min: float, now: float, ignore=None
) -> tuple[bool, str]:
    """(ok, message). ok is False when something was written within *window_min* minutes."""
    mtime, p = newest_write(paths, ignore)
    if p is None:
        return True, "no files touched"
    age_min = (now - mtime) / 60
    return age_min >= window_min, f"newest write {age_min:.1f} min ago: {p}"


# --------------------------------------------------------------------------
# helpers: sizes, hashing, atomic write
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def estimate_compressed(files: Sequence[Path]) -> int:
    """Bytes the files compress to (zlib level 6, as tar.gz does) without writing anything."""
    total = 0
    for p in files:
        co = zlib.compressobj(6)
        try:
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    total += len(co.compress(chunk))
        except OSError:
            continue
        total += len(co.flush())
    return total


def _validate_content(path: Path, text: str) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        json.loads(text)
    elif suffix == ".jsonl":
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"line {i}: {e}") from e


def atomic_write_bytes(path: Path, data: bytes, like: Path | None = None) -> None:
    tmp = path.with_name(f".{path.name}.migrate-tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if like is not None:
            shutil.copymode(like, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def _utc_stamp(now: float) -> str:
    return datetime.fromtimestamp(now, UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json_atomic(path: Path, obj) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=1) + "\n").encode())


# --------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _aws_bin() -> str:
    return shutil.which("aws") or str(Path.home() / ".local/bin/aws")


def default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    argv = [_aws_bin() if argv and argv[0] == "aws" else argv[0], *argv[1:]]
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _split_s3(url: str) -> tuple[str, str]:
    m = re.fullmatch(r"s3://([^/]+)/?(.*)", url)
    if not m:
        raise MigrationError(f"not an s3 url: {url}")
    base = m.group(2)
    if base and not base.endswith("/"):
        base += "/"
    return m.group(1), base


def s3_list(url: str, runner: Runner) -> list[tuple[str, int]]:
    """``[(key relative to the prefix, size)]`` for every object under *url*."""
    bucket, base = _split_s3(url)
    res = runner(["aws", "s3", "ls", f"s3://{bucket}/{base}", "--recursive"])
    if res.returncode != 0:
        raise MigrationError(f"aws s3 ls failed: {res.stderr.strip()}")
    out = []
    for line in res.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4 or not parts[2].isdigit():
            continue
        key = parts[3]
        if key.startswith(base):
            out.append((key[len(base) :], int(parts[2])))
    return out


@dataclass
class S3Action:
    op: str  # "copy" (server side) | "upload" (from the local migrated file)
    src_rel: str
    dst_rel: str
    size: int


@dataclass
class S3Plan:
    url: str
    actions: list[S3Action] = field(default_factory=list)
    rewritten_in_place: list[str] = field(
        default_factory=list
    )  # old ids inside, name unchanged
    total_keys: int = 0

    @property
    def bytes_copied(self) -> int:
        return sum(a.size for a in self.actions if a.op == "copy")

    @property
    def bytes_uploaded(self) -> int:
        return sum(a.size for a in self.actions if a.op == "upload")


def build_s3_plan(
    url: str,
    root: Path,
    rules: Rules,
    plan: Plan,
    excludes: Sequence[str],
    runner: Runner,
) -> S3Plan:
    keys = s3_list(url, runner)
    sp = S3Plan(url=url, total_keys=len(keys))
    excl = {e.strip("/") for e in excludes}
    rewrite_rels = {_rel(r.path, root) for r in plan.rewrites}
    for rel, size in keys:
        if rel.split("/", 1)[0] in excl:
            continue
        new_rel = rules.name(rel)
        if new_rel != rel:
            is_text = Path(rel).suffix.lower() in TEXT_SUFFIXES
            sp.actions.append(
                S3Action("upload" if is_text else "copy", rel, new_rel, size)
            )
        elif rel in rewrite_rels:
            sp.rewritten_in_place.append(rel)
    return sp


def apply_s3(
    sp: S3Plan, root: Path, runner: Runner, delete_old: bool, workers: int = 4
) -> list[str]:
    """Execute *sp*. Returns a list of error strings (empty = success)."""
    bucket, base = _split_s3(sp.url)
    errors: list[str] = []

    def one(a: S3Action) -> str | None:
        dst = f"s3://{bucket}/{base}{a.dst_rel}"
        local = root / a.dst_rel
        if a.op == "upload" and local.exists():
            r = runner(["aws", "s3", "cp", str(local), dst, "--only-show-errors"])
        else:  # binary, or a label file that is not on this disk: server-side copy
            r = runner(
                [
                    "aws",
                    "s3",
                    "cp",
                    f"s3://{bucket}/{base}{a.src_rel}",
                    dst,
                    "--only-show-errors",
                ]
            )
        return (
            None
            if r.returncode == 0
            else f"{a.op} {a.src_rel} -> {a.dst_rel}: {r.stderr.strip()}"
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        errors.extend(e for e in ex.map(one, sp.actions) if e)
    if errors:
        return errors

    # verify every destination exists before anything is deleted
    have = dict(s3_list(sp.url, runner))
    for a in sp.actions:
        if a.dst_rel not in have:
            errors.append(f"missing after copy: {a.dst_rel}")
        elif a.op == "copy" and have[a.dst_rel] != a.size:
            errors.append(
                f"size mismatch after copy: {a.dst_rel} ({have[a.dst_rel]} != {a.size})"
            )
    if errors or not delete_old:
        return errors

    def rm(a: S3Action) -> str | None:
        r = runner(
            [
                "aws",
                "s3",
                "rm",
                f"s3://{bucket}/{base}{a.src_rel}",
                "--only-show-errors",
            ]
        )
        return None if r.returncode == 0 else f"rm {a.src_rel}: {r.stderr.strip()}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        errors.extend(e for e in ex.map(rm, sp.actions) if e)
    return errors


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _shape(rel: str) -> str:
    """Collapse digits so 600 frame jpgs print as one line."""
    return re.sub(r"\d+", "N", rel)


def print_plan(
    plan: Plan,
    out=None,
    verbose: bool = False,
    compressed: int | None = None,
    free: int | None = None,
    guard: tuple[bool, str] | None = None,
    s3: S3Plan | None = None,
) -> None:
    root, r = plan.root, plan.rules

    def w(s: str = "") -> None:
        print(s, file=out or sys.stdout)

    w(f"root: {root}")
    w(
        f"rename  {r.old_match} -> {r.new_match}; {r.old_short} -> {r.new_short}; {r.old_title} -> {r.new_title!r}"
    )
    w()
    w(f"scanned {plan.scanned_entries} entries, {plan.scanned_text_files} text files")
    w(f"directories to rename: {plan.n_dirs}")
    w(f"files/symlinks to rename: {plan.n_files}")
    w(
        f"files to rewrite (ids inside): {len(plan.rewrites)}  ({sum(x.hits for x in plan.rewrites)} occurrences)"
    )
    w(f"symlinks to re-point: {len(plan.relinks)}")
    w()
    w("renames by pattern (count  kind  pattern):")
    shapes = Counter(
        f"{x.kind:<4} {_shape(_rel(x.src, root))}  ->  {_shape(_rel(x.dst, root))}"
        for x in plan.renames
    )
    for s, n in sorted(shapes.items(), key=lambda kv: (kv[0].split()[1], -kv[1])):
        w(f"  {n:>5}  {s}")
    w()
    w("rewrites (files x avg occurrences, pattern):")
    groups: dict[str, list[Rewrite]] = {}
    for x in plan.rewrites:
        groups.setdefault(_shape(_rel(x.path, root)), []).append(x)
    for s, xs in sorted(groups.items()):
        w(f"  {len(xs):>5} file(s) x ~{sum(i.hits for i in xs) // len(xs)}  {s}")
    if verbose:
        w()
        w("every operation:")
        for x in plan.renames:
            w(f"  RENAME {_rel(x.src, root)} -> {_rel(x.dst, root)}")
        for x in plan.rewrites:
            w(f"  REWRITE {_rel(x.path, root)} ({x.hits})")
        for x in plan.relinks:
            w(f"  RELINK {_rel(x.path, root)}: {x.old_target} -> {x.new_target}")
    w()
    w(
        f"backup: {len(plan.backup_files)} text files, {_fmt_bytes(plan.backup_bytes)} uncompressed"
        + (f", ~{_fmt_bytes(compressed)} as tar.gz" if compressed is not None else "")
    )
    if free is not None:
        w(f"free space on {root}: {_fmt_bytes(free)}")
    if plan.excluded:
        w("not renamed (excluded; names fixed elsewhere):")
        for k, n in plan.excluded.items():
            w(f"  {k}/: {n} entries carry the old name")
    if guard is not None:
        ok, msg = guard
        w(f"live-write guard: {'ok' if ok else 'WOULD REFUSE --apply'} ({msg})")
    for c in plan.conflicts:
        w(f"CONFLICT: {c}")
    for x in plan.warnings:
        w(f"WARNING: {x}")
    if s3 is not None:
        w()
        w(f"S3 {s3.url}: {s3.total_keys} keys listed")
        w(
            f"  {sum(a.op == 'copy' for a in s3.actions)} object(s) to copy server-side ({_fmt_bytes(s3.bytes_copied)}, no transfer)"
        )
        w(
            f"  {sum(a.op == 'upload' for a in s3.actions)} label file(s) to upload from local ({_fmt_bytes(s3.bytes_uploaded)})"
        )
        w(f"  old keys deleted only with --s3-delete-old ({len(s3.actions)} keys)")
        for rel in s3.rewritten_in_place:
            w(
                f"  WARNING: S3 key {rel} holds old ids but keeps its name; re-upload it after --apply"
            )
        if verbose:
            for a in s3.actions:
                w(f"  S3 {a.op.upper()} {a.src_rel} -> {a.dst_rel}")
    w()
    if plan.empty:
        w("PLAN EMPTY: nothing to do (already migrated).")


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------


def make_backup(plan: Plan, dest: Path) -> int:
    """Write the tar.gz atomically; returns the member count."""
    tmp = dest.with_name(dest.name + ".partial")
    n = 0
    with tarfile.open(tmp, "w:gz") as tf:
        for p in plan.backup_files:
            if p.exists():
                tf.add(p, arcname=_rel(p, plan.root), recursive=False)
                n += 1
    with tarfile.open(tmp, "r:gz") as tf:  # verify it reads back
        if len(tf.getnames()) != n:
            raise MigrationError("backup verification failed (member count)")
    os.replace(tmp, dest)
    return n


def apply_plan(plan: Plan, backup_dir: Path, now: float, say=print) -> Path:
    """Backup, rewrite, relink, rename. Returns the manifest path."""
    rules, root = plan.rules, plan.root
    stamp = _utc_stamp(now)
    backup_dir.mkdir(parents=True, exist_ok=True)
    tar_path = backup_dir / f"rename_{stamp}.tar.gz"
    manifest_path = backup_dir / f"rename_{stamp}.manifest.json"
    if tar_path.exists() or manifest_path.exists():
        raise MigrationError(
            f"backup for {stamp} already exists; wait a second and retry"
        )

    n = make_backup(plan, tar_path)
    say(f"backup: {n} files -> {tar_path} ({_fmt_bytes(tar_path.stat().st_size)})")
    manifest: dict = {
        "version": MANIFEST_VERSION,
        "created": stamp,
        "status": "backup_done",
        "root": str(root),
        "rules": {
            "old_match": rules.old_match,
            "new_match": rules.new_match,
            "old_short": rules.old_short,
            "new_short": rules.new_short,
            "old_title": rules.old_title,
            "new_title": rules.new_title,
            "literals": [list(x) for x in rules.literals],
        },
        "backup": tar_path.name,
        "rewrites": [],
        "relinks": [],
        "renames": [],
    }
    _write_json_atomic(manifest_path, manifest)

    # 1. rewrite contents at the original paths
    for rw in plan.rewrites:
        p = rw.path
        st = p.stat()
        raw = p.read_bytes()
        text = raw.decode("utf-8")
        new_text = rules.text(text)
        if new_text == text:
            continue
        try:
            _validate_content(p, new_text)
        except ValueError as e:
            raise MigrationError(f"rewritten {p} would not parse: {e}") from e
        st2 = p.stat()
        if (st2.st_mtime_ns, st2.st_size) != (st.st_mtime_ns, st.st_size):
            raise MigrationError(
                f"{p} changed while the migration was running; aborting (re-run to continue)"
            )
        new_raw = new_text.encode("utf-8")
        atomic_write_bytes(p, new_raw, like=p)
        manifest["rewrites"].append(
            {
                "path": _rel(p, root),
                "sha_before": hashlib.sha256(raw).hexdigest(),
                "sha_after": hashlib.sha256(new_raw).hexdigest(),
            }
        )
        _write_json_atomic(manifest_path, manifest)
    say(f"rewrote {len(manifest['rewrites'])} files")

    # 2. symlink targets
    for rl in plan.relinks:
        tmp = rl.path.with_name(f".{rl.path.name}.migrate-tmp")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(rl.new_target, tmp)
        os.replace(tmp, rl.path)
        manifest["relinks"].append(
            {
                "path": _rel(rl.path, root),
                "old_target": rl.old_target,
                "new_target": rl.new_target,
            }
        )
        _write_json_atomic(manifest_path, manifest)

    # 3. renames, deepest first (plan.renames is already sorted that way)
    for rn in plan.renames:
        if not os.path.lexists(rn.src):
            continue
        if os.path.lexists(rn.dst):
            raise MigrationError(f"destination appeared during the run: {rn.dst}")
        os.rename(rn.src, rn.dst)
        manifest["renames"].append(
            {"src": _rel(rn.src, root), "dst": _rel(rn.dst, root), "kind": rn.kind}
        )
        if len(manifest["renames"]) % 200 == 0:
            _write_json_atomic(manifest_path, manifest)
    manifest["status"] = "complete"
    _write_json_atomic(manifest_path, manifest)
    say(f"renamed {len(manifest['renames'])} entries; manifest {manifest_path}")
    return manifest_path


def rollback(
    manifest_path: Path, now: float, window_min: float, force: bool, say=print
) -> int:
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest["root"])
    tar_path = Path(manifest_path).with_name(manifest["backup"])
    if not tar_path.exists():
        say(f"backup tar missing: {tar_path}")
        return EXIT_ERROR
    renames = manifest["renames"]
    rewrites = manifest["rewrites"]

    def final_location(rel: str, start: int = 0) -> Path:
        """Where *rel* (a path as it was when rename #start ran) lives after the later renames.
        Renames are recorded deepest first, so a child's record names its old parent."""
        for r in renames[start:]:
            if rel == r["src"] or rel.startswith(r["src"] + "/"):
                rel = r["dst"] + rel[len(r["src"]) :]
        return root / rel

    current_location = final_location

    rewritten_now = {current_location(r["path"]): r["sha_after"] for r in rewrites}

    def ours(p: Path) -> bool:
        want = rewritten_now.get(p)
        try:
            return want is not None and sha256_file(p) == want
        except OSError:
            return False

    paths: list[Path] = list(rewritten_now)
    for i, r in enumerate(
        renames
    ):  # files only: directory mtimes were bumped by the migration's own renames
        dst = final_location(r["dst"], i + 1)
        if dst.is_dir() and not dst.is_symlink():
            paths.extend(Path(dp) / f for dp, _dn, fns in os.walk(dst) for f in fns)
        elif dst.exists() or dst.is_symlink():
            paths.append(dst)
    ok, msg = live_guard(paths, window_min, now, ignore=ours)
    if not ok and not force:
        say(
            f"REFUSING rollback: {msg}\n(something is writing there; wait, or pass --force)"
        )
        return EXIT_LIVE

    # files first (at their current location), then undo the renames
    skipped = 0
    to_restore: list[tuple[dict, Path]] = []
    for r in rewrites:
        p = current_location(r["path"])
        if not p.exists():
            say(f"WARNING: {p} missing, cannot restore")
            skipped += 1
        elif sha256_file(p) != r["sha_after"] and not force:
            say(
                f"WARNING: {p} changed since the migration; left as is (use --force to overwrite)"
            )
            skipped += 1
        else:
            to_restore.append((r, p))
    restored = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for r, p in to_restore:
            f = tf.extractfile(r["path"])
            data = f.read() if f is not None else None
            if data is None or hashlib.sha256(data).hexdigest() != r["sha_before"]:
                say(
                    f"WARNING: backup of {r['path']} missing or does not match the manifest hash; skipped"
                )
                skipped += 1
                continue
            atomic_write_bytes(p, data, like=p)
            restored += 1

    for r in reversed(renames):
        src, dst = root / r["src"], root / r["dst"]
        if os.path.lexists(dst) and not os.path.lexists(src):
            os.rename(dst, src)
        elif os.path.lexists(src) and os.path.lexists(dst):
            say(f"CONFLICT: both {src} and {dst} exist; leaving them")
            return EXIT_CONFLICT
    for rl in manifest.get("relinks", []):
        p = root / rl["path"]
        p = current_location(rl["path"]) if not os.path.lexists(p) else p
        tmp = p.with_name(f".{p.name}.migrate-tmp")
        os.symlink(rl["old_target"], tmp)
        os.replace(tmp, p)
    say(
        f"reversed {len(renames)} renames; restored {restored} files from {tar_path.name}; skipped {skipped}"
    )
    return EXIT_OK if not skipped else EXIT_ERROR


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Rename a match across footy_data (dry-run by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the footy_data directory")
    ap.add_argument(
        "--old-match", default="arsenal_norwich", help="old match / clip prefix"
    )
    ap.add_argument("--new-match", default="arsenal_nottinghamforest")
    ap.add_argument(
        "--old-short",
        default="norwich",
        help="old short name (dataset key, dir suffixes)",
    )
    ap.add_argument("--new-short", default="nottinghamforest")
    ap.add_argument(
        "--old-title",
        default="Norwich",
        help="old display name (titles, prose in JSON)",
    )
    ap.add_argument("--new-title", default="Nottingham Forest")
    ap.add_argument(
        "--literal",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="extra exact-text replacement for file contents (applied first); repeatable",
    )
    ap.add_argument(
        "--no-default-literals",
        action="store_true",
        help="drop the built-in registry label fix",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="REL",
        help=f"path relative to --root to leave alone; repeatable (default: {', '.join(DEFAULT_EXCLUDES)})",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="print the plan only (the default)"
    )
    mode.add_argument(
        "--apply", action="store_true", help="back up, rewrite and rename"
    )
    mode.add_argument(
        "--rollback",
        metavar="MANIFEST",
        help="undo a previous --apply from its manifest",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="skip the live-write guard / overwrite changed files on rollback",
    )
    ap.add_argument("--live-window-min", type=float, default=10.0)
    ap.add_argument("--backup-dir", default=None, help="default: <root>/_backups")
    ap.add_argument("--verbose", action="store_true", help="list every operation")
    ap.add_argument(
        "--no-estimate",
        action="store_true",
        help="skip the compressed-size estimate in the plan",
    )
    ap.add_argument(
        "--include-s3", action="store_true", help="also plan / do the S3 copy"
    )
    ap.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    ap.add_argument(
        "--s3-delete-old",
        action="store_true",
        help="after a verified copy, delete the old S3 keys",
    )
    return ap.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner | None = None,
    now: float | None = None,
) -> int:
    a = _parse(argv)
    now = time.time() if now is None else now
    runner = runner or default_runner
    root = Path(a.root)
    backup_dir = Path(a.backup_dir) if a.backup_dir else root / "_backups"

    if a.rollback:
        return rollback(Path(a.rollback), now, a.live_window_min, a.force)

    literals = [] if a.no_default_literals else list(DEFAULT_LITERALS)
    for lit in a.literal:
        if "=" not in lit:
            print(f"--literal needs OLD=NEW, got {lit!r}", file=sys.stderr)
            return EXIT_ERROR
        o, n = lit.split("=", 1)
        literals.append((o, n))
    rules = Rules(
        a.old_match,
        a.new_match,
        a.old_short,
        a.new_short,
        a.old_title,
        a.new_title,
        tuple(literals),
    )
    try:
        rules.validate()
    except MigrationError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    if not root.is_dir():
        print(f"error: --root {root} is not a directory", file=sys.stderr)
        return EXIT_ERROR

    excludes = list(a.exclude) if a.exclude is not None else list(DEFAULT_EXCLUDES)
    try:
        excludes.append(_rel(backup_dir, root))
    except ValueError:
        pass  # backup dir outside the root

    plan = build_plan(root, rules, excludes)
    guard = live_guard(touched_paths(plan), a.live_window_min, now)
    free = shutil.disk_usage(root).free
    compressed = (
        None if (a.no_estimate or a.apply) else estimate_compressed(plan.backup_files)
    )
    s3plan = None
    if a.include_s3:
        try:
            s3plan = build_s3_plan(a.s3_prefix, root, rules, plan, excludes, runner)
        except MigrationError as e:
            print(f"error: {e}", file=sys.stderr)
            return EXIT_ERROR

    if not a.apply:
        print_plan(
            plan,
            verbose=a.verbose,
            compressed=compressed,
            free=free,
            guard=guard,
            s3=s3plan,
        )
        print("DRY RUN: nothing was changed. Re-run with --apply to do it.")
        return EXIT_CONFLICT if plan.conflicts else EXIT_OK

    # ---- apply ----
    print_plan(plan, verbose=False, free=free, guard=guard, s3=s3plan)
    if plan.conflicts:
        print(
            "refusing: resolve the conflicts above first (nothing was changed)",
            file=sys.stderr,
        )
        return EXIT_CONFLICT
    if not guard[0] and not a.force:
        print(
            f"REFUSING --apply: {guard[1]}\nthe live labeller wrote there within {a.live_window_min:g} min; "
            "wait for an idle moment or pass --force",
            file=sys.stderr,
        )
        return EXIT_LIVE
    need = int(plan.backup_bytes * 1.2) + (1 << 30)
    if free < need:
        print(
            f"refusing: need ~{_fmt_bytes(need)} free for the backup, have {_fmt_bytes(free)}",
            file=sys.stderr,
        )
        return EXIT_ERROR
    try:
        if plan.empty:
            print("local tree: nothing to do")
        else:
            apply_plan(plan, backup_dir, now)
            after = build_plan(root, rules, excludes)
            left = len(after.renames) + len(after.rewrites)
            print(
                f"verify: re-planned after apply: {left} operation(s) left"
                + (" (OK)" if not left else " (!!)")
            )
            if left:
                return EXIT_ERROR
        if s3plan is not None:
            errs = apply_s3(s3plan, root, runner, delete_old=a.s3_delete_old)
            if errs:
                print("S3 errors:\n  " + "\n  ".join(errs[:20]), file=sys.stderr)
                return EXIT_ERROR
            print(
                f"S3: {len(s3plan.actions)} object(s) copied/uploaded"
                + (
                    "; old keys deleted"
                    if a.s3_delete_old
                    else "; old keys kept (use --s3-delete-old)"
                )
            )
    except MigrationError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
