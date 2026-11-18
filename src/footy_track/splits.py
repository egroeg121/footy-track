"""The one train/eval partition authority for every dataset, export and eval script.

Two tiers of evaluation data, both *sticky* (an assignment never changes once made) and both
*growing* (new matches / clips add eval items, nothing ever leaves the eval set automatically):

``eval_heldout_match``
    Whole matches that are evaluation-only. Recorded in the per-match register
    (:mod:`footy_track.match_splits`, ``match_splits.json``: ~25 % of matches, new ones assigned by
    ``sha1(match key) % 4 == 0``, manual edits win, a missing file fails closed).

``eval_in_match``
    Inside a *train* match, clips whose ``sha1(canonical clip id) % 5 == 0`` (~20 %, one clip in
    five) are evaluation-only. Assigned at *clip* granularity (never frame, so near-duplicate
    neighbouring frames cannot straddle the line) and recorded once in the clip register
    ``clip_splits.json`` (sibling of ``match_splits.json``; override with env
    ``FOOTY_CLIP_SPLITS``). The register stores the rule it was assigned under, so changing the
    code constant never reshuffles old clips.

Everything else is ``train``.

Why a sibling file and not a ``clips`` section of ``match_splits.json``: that file is rewritten by
the jobs repo (``footy_splits.py``) and validated entry-by-entry by :mod:`match_splits`; a foreign
top-level key would break both. A sibling with its own lock keeps the two writers independent.

Clips are keyed by :func:`footy_track.clip_id.canonical_clip_id`. Things that are not clips but
need an eval/train decision (frames of a broadcast-classifier review) are mapped to a clip-like
id (:func:`broadcast_window_id`: ``<match key>_w<frame // 6000>``).

Fail closed: a missing/invalid clip register raises :class:`footy_track.match_splits.RegisterError`
(``FOOTY_MATCH_SPLITS_ALLOW_SEED=1``, tests/CI only, substitutes an empty register that falls back
to the hash rule). A missing *match* register already raises.

CLI::

    python -m footy_track.splits show [NAME ...]          # registers, or how NAMEs resolve
    python -m footy_track.splits assign [--apply]         # dry-run by default
    python -m footy_track.splits stats                    # counts per split per dataset
    python -m footy_track.splits contamination DIR [DIR ...]   # eval items inside training sets
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from footy_track import match_splits as ms
from footy_track.clip_id import canonical_clip_id
from footy_track.match_splits import RegisterError

Split = Literal["train", "eval_heldout_match", "eval_in_match"]
TRAIN: Split = "train"
EVAL_HELDOUT: Split = "eval_heldout_match"
EVAL_IN_MATCH: Split = "eval_in_match"
SPLITS: tuple[Split, ...] = (TRAIN, EVAL_IN_MATCH, EVAL_HELDOUT)

# --- The three canonical partitions (what every consumer, export and report should speak) -------
#: ``train`` = ``split_of`` value ``train``; ``eval`` (validation: model selection, early stopping)
#: = ``eval_in_match``; ``test`` (never used for tuning, final reporting only) =
#: ``eval_heldout_match``. The ``split_of`` vocabulary stays valid everywhere (stored rows,
#: ``eval_result.split``, old CLIs); :func:`to_partition` maps any old spelling to the new one.
EVAL = "eval"
TEST = "test"
PARTITIONS: tuple[str, ...] = (TRAIN, EVAL, TEST)
#: Not a partition: a dataset image whose match cannot be found. Reported, never put in ``test``.
UNMAPPED = "unmapped"
PARTITION_OF_SPLIT: dict[str, str] = {TRAIN: TRAIN, EVAL_IN_MATCH: EVAL, EVAL_HELDOUT: TEST}
SPLIT_OF_PARTITION: dict[str, Split] = {v: k for k, v in PARTITION_OF_SPLIT.items()}  # type: ignore[misc]
#: Every spelling found in the repo today -> its canonical partition.
PARTITION_ALIASES: dict[str, str] = {
    **PARTITION_OF_SPLIT,
    "valid": EVAL, "val": EVAL, "validation": EVAL, "eval": EVAL,
    "heldout": TEST, "held_out": TEST, "held-out": TEST, "test": TEST,
    "seen": TRAIN,
}

#: One clip in ``IN_MATCH_MODULUS`` is in-match eval (``sha1(clip id) % 5 == 0``, ~20 %).
IN_MATCH_MODULUS = 5
IN_MATCH_REMAINDER = 0
CLIP_TRAIN, CLIP_EVAL = "train", "eval"

#: Frames per broadcast-review "clip" window (2 min at 50 fps, 4 min at 25 fps).
BROADCAST_WINDOW_FRAMES = 6000

#: Roboflow-era image-name prefixes that are not in the register's alias lists.
LEGACY_NAME_ALIASES: dict[str, str] = {
    "Arsenal-Aston-Villa": "arsenal_astonvilla",
    "Bournemouth-vs-Arsenal-1": "arsenal_bournmouth_1st_half",
    "Bournemouth-vs-Arsenal-2": "arsenal_bournmouth_2nd_half",
}
#: Single-video clips that are not named ``..._segNNN`` / ``..._partNNN``.
STANDALONE_CLIPS = frozenset({"arsenal_example"})

DEFAULT_CLIP_REGISTER = "clip_splits.json"
_CLIP_RE = re.compile(r".*_(?:seg|part)\d+$")
_WINDOW_RE = re.compile(r"^(?P<match>.+)_w(?P<n>\d+)$")


# --------------------------------------------------------------------------------------------
# Paths, locking, atomic writes
# --------------------------------------------------------------------------------------------
def data_root() -> Path:
    return Path(os.environ.get("FOOTY_DATA", "/mnt/storage/footy_data"))


def clip_register_path() -> Path:
    env = os.environ.get("FOOTY_CLIP_SPLITS")
    if env:
        return Path(env)
    return ms.register_path().with_name(DEFAULT_CLIP_REGISTER)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Exclusive inter-process lock on ``<path>.lock`` (same scheme as the jobs-side writer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _atomic_write_json(path: Path, doc: dict) -> None:
    """Write ``doc`` to ``path`` via a temp file in the same directory + ``os.replace``."""
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o664
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------------------------
# Clip register
# --------------------------------------------------------------------------------------------
def _empty_register() -> dict:
    return {
        "version": 1,
        "rule": {
            "hash": "sha1(canonical_clip_id)",
            "modulus": IN_MATCH_MODULUS,
            "eval_remainder": IN_MATCH_REMAINDER,
        },
        "clips": {},
    }


_clip_cache: dict = {"key": None, "data": None}


def _validate_clip_register(doc: object) -> dict:
    if not isinstance(doc, dict) or not isinstance(doc.get("clips"), dict):
        raise ValueError(
            "expected {rule: {...}, clips: {<clip id>: {split: train|eval}}}"
        )
    rule = doc.get("rule") or {}
    if not isinstance(rule.get("modulus"), int) or rule["modulus"] < 1:
        raise ValueError("rule.modulus must be a positive integer")
    if not isinstance(rule.get("eval_remainder", 0), int):
        raise ValueError("rule.eval_remainder must be an integer")
    for cid, entry in doc["clips"].items():
        if not isinstance(entry, dict) or entry.get("split") not in (
            CLIP_TRAIN,
            CLIP_EVAL,
        ):
            raise ValueError(f"clip {cid!r}: split must be 'train' or 'eval'")
    return doc


def _preview() -> bool:
    """``--preview`` / env ``FOOTY_CLIP_SPLITS_PREVIEW=1``: read-only look at what the hash rule
    would give while no clip register exists yet (never use for exports)."""
    return os.environ.get("FOOTY_CLIP_SPLITS_PREVIEW", "") not in ("", "0")


def load_clip_register(
    path: Path | None = None, allow_seed: bool | None = None
) -> dict:
    """The clip register document. Raises ``RegisterError`` when invalid (always) or missing
    (unless ``allow_seed`` / env ``FOOTY_MATCH_SPLITS_ALLOW_SEED=1``: tests/CI only)."""
    p = path or clip_register_path()
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        if allow_seed if allow_seed is not None else (ms._allow_seed() or _preview()):
            return _empty_register()
        raise RegisterError(
            f"clip split register not found: {p} (set FOOTY_CLIP_SPLITS, create it with "
            "`python -m footy_track.splits assign --apply`, or FOOTY_MATCH_SPLITS_ALLOW_SEED=1 "
            "for tests)"
        ) from None
    except OSError as e:
        raise RegisterError(f"cannot read clip split register {p}: {e}") from e
    if _clip_cache["key"] == key:
        return _clip_cache["data"]
    try:
        data = _validate_clip_register(json.loads(p.read_text()))
    except (OSError, ValueError) as e:
        raise RegisterError(f"invalid clip split register {p}: {e}") from e
    _clip_cache.update(key=key, data=data)
    return data


def clear_cache() -> None:
    _clip_cache.update(key=None, data=None)
    ms.clear_cache()


def hash_clip_split(
    clip_id: str, modulus: int = IN_MATCH_MODULUS, remainder: int = 0
) -> str:
    """``"eval"`` for ~1 clip in ``modulus`` by a stable hash of the canonical clip id."""
    h = int(hashlib.sha1(canonical_clip_id(clip_id).encode()).hexdigest(), 16)  # noqa: S324
    return CLIP_EVAL if h % modulus == remainder else CLIP_TRAIN


def clip_tier(clip_id: str, register: dict | None = None) -> str:
    """``"eval"`` / ``"train"`` for a clip *within a train match*: the register entry when there
    is one (sticky), else the hash rule recorded in the register."""
    reg = register if register is not None else load_clip_register()
    cid = canonical_clip_id(clip_id)
    entry = reg["clips"].get(cid)
    if entry:
        return entry["split"]
    rule = reg["rule"]
    return hash_clip_split(cid, rule["modulus"], rule.get("eval_remainder", 0))


def broadcast_window_id(match_key: str, frame: int) -> str:
    """The clip-like unit for broadcast-review frames: a window of consecutive frames."""
    return f"{match_key}_w{int(frame) // BROADCAST_WINDOW_FRAMES:04d}"


# --------------------------------------------------------------------------------------------
# Name resolution: names -> match key, clip id
# --------------------------------------------------------------------------------------------
_fullmatch_cache: dict = {"path": None, "mtime": None, "map": {}}


def _post_key(post_url: str) -> str:
    return re.sub(r"-full-match", "", post_url.rstrip("/").rsplit("/", 1)[-1])


def fullmatch_keys() -> dict[str, str]:
    """``{full-match file stem: match key}`` from ``full_matches/manifest.jsonl`` (the broadcast
    review calls a match by its video file stem, e.g. ``nfo.v.ars.full.epl.18.10.2026...``)."""
    p = Path(
        os.environ.get(
            "FOOTY_FULL_MATCHES_MANIFEST", data_root() / "full_matches/manifest.jsonl"
        )
    )
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        return {}
    if _fullmatch_cache["path"] == str(p) and _fullmatch_cache["mtime"] == mtime:
        return _fullmatch_cache["map"]
    out: dict[str, str] = {}
    for line in p.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("mp4") and rec.get("post_url"):
            out[Path(rec["mp4"]).stem] = _post_key(rec["post_url"])
    _fullmatch_cache.update(path=str(p), mtime=mtime, map=out)
    return out


def _name_index() -> list[tuple[str, str]]:
    """``[(name, match key)]`` for every register key/alias plus the legacy Roboflow prefixes."""
    reg = ms.load_register()
    out = [(n, k) for k, e in reg.items() for n in ms._names_of(k, e)]
    out += [(n, k) for n, k in LEGACY_NAME_ALIASES.items() if k in reg]
    return out


def _matching_keys(cid: str) -> list[tuple[int, str]]:
    return [
        (len(n), k) for n, k in _name_index() if cid == n or cid.startswith(n + "_")
    ]


def match_key_of(name: str) -> str | None:
    """The register key of the match a clip / frame / file name belongs to (longest alias
    prefix wins), or ``None`` for a name no registered match claims."""
    stem = (
        Path(name).stem
        if Path(name).suffix.lower() in {".mp4", ".mkv", ".jpg", ".png"}
        else name
    )
    key = fullmatch_keys().get(stem) or fullmatch_keys().get(name)
    if key:
        return key
    cid = canonical_clip_id(name)
    window = _WINDOW_RE.match(cid)
    hits = _matching_keys(cid) or (_matching_keys(window["match"]) if window else [])
    return max(hits)[1] if hits else None


def _match_is_heldout(match_key: str | None, cid: str | None) -> bool:
    if match_key and ms.split_for_match(match_key) == ms.HELDOUT:
        return True
    if cid:
        probe = cid
        window = _WINDOW_RE.match(cid)
        if window:
            probe = window["match"]
        return ms.is_heldout_clip(probe)
    return False


def split_of(match_key: str | None, clip_id: str | None = None) -> Split:
    """The partition of one item: ``train`` / ``eval_heldout_match`` / ``eval_in_match``.

    * a heldout match (by ``match_key`` or by the clip id's match prefix) -> ``eval_heldout_match``;
    * otherwise a clip registered (or hashed) as eval -> ``eval_in_match``;
    * otherwise ``train``. With no ``clip_id`` this is the *match-level* answer: ``train`` means
      the match is in the training pool, not that all of its clips are.

    A match absent from the register is treated as a train-pool match (the clip rule still
    applies); :func:`unregistered_match` flags it. Raises ``RegisterError`` when a register
    needed for the answer is missing/invalid.
    """
    cid = canonical_clip_id(clip_id) if clip_id else None
    if _match_is_heldout(match_key, cid):
        return EVAL_HELDOUT
    if cid is None:
        return TRAIN
    return EVAL_IN_MATCH if clip_tier(cid) == CLIP_EVAL else TRAIN


def unregistered_match(match_key: str | None, clip_id: str | None = None) -> bool:
    """True when neither ``match_key`` nor the clip id belongs to a registered match."""
    if match_key and ms.split_for_match(match_key) != "unassigned":
        return False
    return not (clip_id and match_key_of(clip_id))


def split_of_name(name: str) -> Split:
    """:func:`split_of` for a bare clip / game id / file name (match derived from the name)."""
    return split_of(match_key_of(name), name)


def split_label(name: str) -> str:
    """:func:`split_of_name` for *reports*: a missing/invalid register gives ``"unknown"``
    instead of raising (exports must still fail closed; a metrics table should not crash)."""
    try:
        return split_of_name(name)
    except RegisterError:
        return "unknown"


def split_counts(names: Iterable[str]) -> dict[str, int]:
    """``{split: n}`` over clip / game ids (for ingest reports; unknown when no register)."""
    return dict(sorted(Counter(split_label(n) for n in names).items()))


def is_eval(split: str) -> bool:
    return split != TRAIN


EXPORT_DIR = {TRAIN: "train", EVAL_IN_MATCH: "val", EVAL_HELDOUT: "test"}


def export_dir(
    split: str, *, include_test: bool = True, heldout_as_test: bool | None = None
) -> str | None:
    """Where an export puts an item of ``split`` (either vocabulary): ``train`` / ``val`` (the
    ``eval`` partition) / ``test`` (the ``test`` partition). ``include_test=False`` leaves test
    items out (``None``). ``heldout_as_test`` is the deprecated spelling of ``include_test``
    (``--heldout-as-test`` used to be needed to *get* a test dir; it is now the default)."""
    if heldout_as_test is not None:
        include_test = heldout_as_test
    part = to_partition(split)
    if part == TEST and not include_test:
        return None
    return EXPORT_DIR[SPLIT_OF_PARTITION[part]]


def to_partition(value: str) -> str:
    """Canonical partition (``train`` / ``eval`` / ``test``) for any split spelling in use."""
    try:
        return PARTITION_ALIASES[value]
    except KeyError:
        raise ValueError(
            f"unknown split {value!r}; expected one of {sorted(PARTITION_ALIASES)}"
        ) from None


def to_split(value: str) -> Split:
    """The ``split_of`` vocabulary value (stored / ``eval_result.split``) for a partition spelling."""
    return SPLIT_OF_PARTITION[to_partition(value)]


EVAL_ALL = "eval_all"  # result/select pseudo-split: both eval tiers (eval + test), never train


def to_result_split(value: str) -> str:
    """Stored/queried ``eval_result.split`` / ``DatasetVersion.select`` value for any spelling:
    ``eval`` -> ``eval_in_match``, ``test`` -> ``eval_heldout_match``, ``eval_all`` kept."""
    if value in (EVAL_ALL, "eval+test", "eval_and_test"):
        return EVAL_ALL
    return to_split(value)


def partition_of(match_key: str | None, clip_id: str | None = None) -> str:
    """:func:`split_of` in the canonical vocabulary: ``train`` / ``eval`` / ``test``."""
    return PARTITION_OF_SPLIT[split_of(match_key, clip_id)]


def partition_of_name(name: str) -> str:
    return PARTITION_OF_SPLIT[split_of_name(name)]


def partition_label(name: str) -> str:
    """:func:`partition_of_name` for reports: ``"unknown"`` when the register is unavailable."""
    label = split_label(name)
    return PARTITION_OF_SPLIT.get(label, label)


def partition_counts(names: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(partition_label(n) for n in names).items()))


def match_partition(match_key: str) -> str:
    """Match-level partition: ``test`` (heldout match), ``train`` (train pool: some of its clips
    are ``eval``) or ``unassigned`` (not in the register). For dashboards."""
    state = ms.split_for_match(match_key)
    return {ms.HELDOUT: TEST, ms.TRAIN: TRAIN}.get(state, "unassigned")


def image_partition(filename: str) -> tuple[str, str | None]:
    """``(partition, flag)`` for a dataset image file name (Roboflow or ours).

    * match not found -> ``("unmapped", "unmapped")``: unknown provenance, it is never ``test``
      and must not be reported as a test image; training on it is allowed but flagged;
    * match known, clip not (frames cut from a full match video) -> ``test`` if the match is
      heldout, else ``("train", "clip_unknown")`` (it may be in-match ``eval``; cannot be told);
    * otherwise the authority's answer, flag ``None``.
    """
    ref = parse_image_name(filename)
    if ref.match_key is None and ref.clip_id is None:
        return UNMAPPED, UNMAPPED
    if ref.clip_id is None:
        if split_of(ref.match_key) == EVAL_HELDOUT:
            return TEST, None
        return TRAIN, "clip_unknown"
    return partition_of(ref.match_key, ref.clip_id), None


def split_map(clip_ids: Iterable[str]) -> dict[str, Split]:
    """``{canonical clip id: split}`` for an eval script."""
    return {canonical_clip_id(c): split_of_name(c) for c in clip_ids}


def legacy_split_map(clip_ids: Iterable[str]) -> dict[str, str]:
    """The old hand-made ``split.json`` vocabulary: ``held_out`` = any eval tier, else ``seen``."""
    return {
        c: ("seen" if s == TRAIN else "held_out")
        for c, s in split_map(clip_ids).items()
    }


def load_split_arg(path: Path | None, clip_ids: Iterable[str]) -> dict[str, str]:
    """``{clip: seen|held_out}`` for scripts that take ``--split``: the legacy JSON file when one
    is given (explicit override), else the authority's map over ``clip_ids``."""
    if path:
        return json.loads(Path(path).read_text())
    return legacy_split_map(clip_ids)


def annotate_split(df, col: str = "game_id", out: str = "split"):  # noqa: ANN001
    """Add a ``split`` column to a DataFrame keyed by clip / game id (the feature store keeps no
    stored split column: it would go stale as the eval set grows)."""
    cache: dict[str, str] = {}
    df = df.copy()
    df[out] = [cache.setdefault(g, split_of_name(g)) for g in df[col]]
    return df


def eval_set(clip_ids: Iterable[str]) -> dict[str, Split]:
    """The eval subset of ``clip_ids``: ``{canonical id: split}``, tiers other than ``train``."""
    return {c: s for c, s in split_map(clip_ids).items() if is_eval(s)}


# --------------------------------------------------------------------------------------------
# Assigning new clips
# --------------------------------------------------------------------------------------------
def plan_assign(
    clip_ids: Iterable[str], register: dict | None = None
) -> dict[str, dict]:
    """Register entries that *would* be added for ``clip_ids``: clips of train-pool matches that
    have no entry yet. Heldout-match clips are decided by their match and never registered."""
    reg = register if register is not None else load_clip_register()
    rule = reg["rule"]
    now = datetime.now(UTC).isoformat(timespec="seconds")
    plan: dict[str, dict] = {}
    for raw in clip_ids:
        cid = canonical_clip_id(raw)
        if cid in reg["clips"] or cid in plan:
            continue
        key = match_key_of(cid)
        if _match_is_heldout(key, cid):
            continue
        plan[cid] = {
            "split": hash_clip_split(
                cid, rule["modulus"], rule.get("eval_remainder", 0)
            ),
            "match": key,
            "assigned": now,
            "reason": f"auto: sha1(clip id) % {rule['modulus']} == {rule.get('eval_remainder', 0)} -> eval",
        }
    return plan


def register_clips(
    clip_ids: Iterable[str], path: Path | None = None, *, dry_run: bool = True
) -> dict[str, dict]:
    """Add entries for clips not yet in the clip register and return the new entries.

    Dry-run by default. With ``dry_run=False`` the file is created if missing; the write is
    atomic (lock + temp file + rename) and re-reads the file under the lock, so concurrent
    writers never lose each other's entries. Existing entries (including manual edits) are never
    touched, which is what makes the assignment sticky and the eval set monotone.
    """
    p = path or clip_register_path()
    ids = list(clip_ids)
    if dry_run:
        try:
            reg = load_clip_register(p)
        except RegisterError:
            reg = _empty_register()
        return plan_assign(ids, reg)
    with _locked(p):
        created = False
        try:
            reg = _validate_clip_register(json.loads(p.read_text()))
        except FileNotFoundError:
            reg, created = _empty_register(), True
        except ValueError as e:
            raise RegisterError(f"invalid clip split register {p}: {e}") from e
        plan = plan_assign(ids, reg)
        if plan or created:
            reg["clips"].update(plan)
            _atomic_write_json(p, reg)
    clear_cache()
    return plan


# --------------------------------------------------------------------------------------------
# Discovering the clip universe (for assign / stats)
# --------------------------------------------------------------------------------------------
def _stems(d: Path, pattern: str) -> set[str]:
    return {canonical_clip_id(p.name) for p in d.glob(pattern)} if d.is_dir() else set()


def discover_clips(root: Path | None = None) -> dict[str, set[str]]:
    """``{dataset: {canonical clip id}}`` found under the footy data directory.

    ``tracking``      broadcast segments per match + the flat clip dirs + GT / machine labels
    ``ball_gt``       clips with a ball GT sidecar
    ``calibration``   clips with calibration GT / proposals / frame-type labels
    ``broadcast``     broadcast-review windows (``<match>_w<NNNN>``) of reviewed frames
    """
    root = root or data_root()
    out: dict[str, set[str]] = defaultdict(set)
    for match_dir in (
        (p for p in sorted(root.iterdir()) if p.is_dir()) if root.is_dir() else []
    ):
        for seg in match_dir.glob("split_video_broadcast_frames*"):
            out["tracking"] |= _stems(seg, "*.mp4")
    for flat in (
        "clips",
        "eval_clips_kf10",
        "ballcheck_clips_kf10",
        "ballcheck_clips",
        "eval_data/clips",
    ):
        out["tracking"] |= _stems(root / flat, "*.mp4")
    gt: set[str] = set()
    for d in sorted(
        root.glob("ball_gt_marks*")
    ):  # per-dataset marks dirs; backups share the stems
        gt |= _stems(d, "*.jsonl")
    out["ball_gt"] |= gt
    out["tracking"] |= gt
    for d in sorted(root.glob("machine_labels*")):
        out["tracking"] |= _stems(d, "*.jsonl")
    marks = root / "ball_gt_marks"
    for sub in ("calibration_gt", "calibration_proposals", "frame_type"):
        out["calibration"] |= _stems(marks / sub, "*.json")
    out["tracking"] |= out["calibration"]
    review = root / "broadcast_review"
    keys = fullmatch_keys()
    for match_dir in review.iterdir() if review.is_dir() else []:
        key = keys.get(match_dir.name, match_dir.name)
        for jpg in match_dir.glob("f*.jpg"):
            if jpg.stem[1:].isdigit():
                out["broadcast"].add(broadcast_window_id(key, int(jpg.stem[1:])))
    return dict(out)


def all_clips(found: dict[str, set[str]]) -> set[str]:
    return set().union(*found.values()) if found else set()


# --------------------------------------------------------------------------------------------
# Contamination check
# --------------------------------------------------------------------------------------------
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-fA-F]{20,}$")
_EXT_SUFFIX = re.compile(r"_(?:jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)
_FRAME_SUFFIX = re.compile(r"^(?P<stem>.+?)_f?(?P<frame>\d+)$")
_SPLIT_DIRS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


@dataclass(frozen=True)
class ImageRef:
    """What an image file name says about where the picture came from."""

    stem: (
        str  # clip / video the frame was cut from (Roboflow + extension noise removed)
    )
    frame: int | None
    match_key: str | None
    clip_id: str | None  # set only when the stem is a real clip (seg/part/standalone)

    @property
    def clip_known(self) -> bool:
        return self.clip_id is not None


def parse_image_name(filename: str) -> ImageRef:
    """Map an image file name back to its match / clip.

    Handles our export names (``<clip>_<frame:06d>.jpg``, ``<match>_f<frame>.jpg``) and Roboflow
    exports, which rewrite ``<clip>_<frame>.jpg`` to ``<clip>_<frame>_jpg.rf.<hash>.jpg``: the
    original name is everything before ``.rf.``, with ``_jpg`` (the renamed dot) removed.
    A frame cut from a *full* match video (``arsenal_mancity_20250925_000472``) names the match
    but no clip, so ``clip_id`` is ``None`` for it.
    """
    name = Path(filename).name
    base = re.sub(r"\.(?:jpe?g|png|bmp|webp)$", "", name, flags=re.IGNORECASE)
    base = _RF_SUFFIX.sub("", base)
    base = _EXT_SUFFIX.sub("", base)
    frame = None
    m = _FRAME_SUFFIX.match(base)
    if m:
        base, frame = m["stem"], int(m["frame"])
    cid = canonical_clip_id(base)
    key = match_key_of(cid)
    is_clip = (
        bool(_CLIP_RE.match(cid))
        or cid in STANDALONE_CLIPS
        or bool(_WINDOW_RE.match(cid))
    )
    if frame is not None and not is_clip and re.search(r"\.com$", base):
        # broadcast review: <video stem>_f<frame> -> window clip id
        key = fullmatch_keys().get(base)
        if key:
            return ImageRef(base, frame, key, broadcast_window_id(key, frame))
    return ImageRef(base, frame, key, cid if is_clip else None)


def _image_split_dir(rel: Path) -> str | None:
    for part in rel.parts[:-1]:
        if part in _SPLIT_DIRS:
            return _SPLIT_DIRS[part]
    return None


def contamination(
    dataset_dir: Path, *, train_splits: Iterable[str] = ("train",), max_listed: int = 25
) -> dict:
    """Eval items found inside the training part of a dataset (read-only).

    Every image under ``dataset_dir`` is mapped to a match / clip by its file name and given a
    split. ``train_splits`` are the dataset's own subsets treated as training data (Roboflow
    ``train`` by default; ``valid``/``test`` are shown in ``by_subset`` but are not training).
    Images that only name a match (frames of a full match video) are certain to be
    ``eval_heldout_match`` when the match is held out, but cannot be placed in ``eval_in_match``
    (``train_match_clip_unknown``): that count is the uncertainty in the result.
    """
    train_splits = {_SPLIT_DIRS.get(s, s) for s in train_splits}  # valid -> val
    by_subset: dict[str, Counter] = defaultdict(Counter)
    listed: dict[str, list[str]] = defaultdict(list)
    clips_hit: dict[str, Counter] = defaultdict(Counter)
    unregistered: Counter = Counter()
    n_images = 0
    for path in sorted(
        p for p in dataset_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXT
    ):
        n_images += 1
        subset = _image_split_dir(path.relative_to(dataset_dir)) or "(none)"
        ref = parse_image_name(path.name)
        if ref.match_key is None and ref.clip_id is None:
            cat = "unmapped"
        elif ref.clip_id is None:
            cat = (
                EVAL_HELDOUT
                if split_of(ref.match_key) == EVAL_HELDOUT
                else "train_match_clip_unknown"
            )
        else:
            cat = split_of(ref.match_key, ref.clip_id)
            if ref.match_key is None:
                unregistered[ref.clip_id] += 1
        by_subset[subset][cat] += 1
        if subset in train_splits and cat in (EVAL_HELDOUT, EVAL_IN_MATCH):
            clips_hit[cat][ref.clip_id or ref.match_key or ref.stem] += 1
            if len(listed[cat]) < max_listed:
                listed[cat].append(path.name)
    train = Counter()
    for s in train_splits:
        train += by_subset.get(s, Counter())
    n_train = sum(train.values())
    return {
        "dataset": str(dataset_dir),
        "train_subsets": sorted(train_splits),
        "n_images": n_images,
        "by_subset": {k: dict(v) for k, v in sorted(by_subset.items())},
        "train_images": n_train,
        "train_contaminated": {
            EVAL_HELDOUT: train[EVAL_HELDOUT],
            EVAL_IN_MATCH: train[EVAL_IN_MATCH],
        },
        "train_contaminated_fraction": (train[EVAL_HELDOUT] + train[EVAL_IN_MATCH])
        / n_train
        if n_train
        else 0.0,
        "train_clip_unknown": train["train_match_clip_unknown"],
        "train_unmapped": train["unmapped"],
        "contaminating_clips": {k: dict(v.most_common()) for k, v in clips_hit.items()},
        "examples": dict(listed),
        "clips_with_unregistered_match": dict(unregistered),
    }


def format_contamination(rep: dict) -> str:
    n = rep["train_images"]
    c = rep["train_contaminated"]
    lines = [
        f"{rep['dataset']}  ({rep['n_images']} images; training subsets: {', '.join(rep['train_subsets'])})",
        f"  train images: {n}",
        f"    eval_heldout_match: {c[EVAL_HELDOUT]}   eval_in_match: {c[EVAL_IN_MATCH]}   "
        f"=> {rep['train_contaminated_fraction']:.1%} contaminated",
        f"    match known but clip unknown (in-match eval cannot be checked): {rep['train_clip_unknown']}"
        f" ({rep['train_clip_unknown'] / n:.1%})"
        if n
        else "",
        f"    unmapped (no match, no clip): {rep['train_unmapped']}",
        "  by subset:",
    ]
    for subset, counts in rep["by_subset"].items():
        lines.append(f"    {subset}: {sum(counts.values())} images  {counts}")
    for tier, clips in rep["contaminating_clips"].items():
        lines.append(f"  {tier} in training, by clip/match: {clips}")
    if rep["clips_with_unregistered_match"]:
        lines.append(
            f"  clips whose match is not in the register: {rep['clips_with_unregistered_match']}"
        )
    return "\n".join(x for x in lines if x)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _stats(found: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for ds, clips in sorted({**found, "ALL (unique clips)": all_clips(found)}.items()):
        c = Counter(split_of_name(x) for x in clips)
        out[ds] = {s: c.get(s, 0) for s in SPLITS} | {"total": len(clips)}
    return out


def _cmd_show(args: argparse.Namespace) -> int:
    reg = ms.load_register()
    held = [k for k, e in reg.items() if e["split"] == ms.HELDOUT]
    print(
        f"match register {ms.register_path()}: {len(reg)} matches, {len(held)} heldout"
    )
    print(f"  heldout: {', '.join(sorted(held))}")
    try:
        creg = load_clip_register()
    except RegisterError as e:
        print(f"clip register: {e}")
        creg = None
    if creg:
        c = Counter(e["split"] for e in creg["clips"].values())
        print(
            f"clip register {clip_register_path()}: {len(creg['clips'])} clips "
            f"({c.get(CLIP_EVAL, 0)} eval, {c.get(CLIP_TRAIN, 0)} train), rule {creg['rule']}"
        )
    for name in args.names:
        key = match_key_of(name)
        cid = canonical_clip_id(name)
        print(
            f"{name}: match={key} clip={cid} split={split_of(key, cid)}"
            + ("  (match not in register)" if unregistered_match(key, cid) else "")
        )
    return 0


def _cmd_assign(args: argparse.Namespace) -> int:
    root = Path(args.data_root) if args.data_root else data_root()
    found = discover_clips(root)
    clips = all_clips(found)
    path = Path(args.register) if args.register else clip_register_path()
    dry = not args.apply
    plan = register_clips(clips, path, dry_run=dry)
    n_eval = sum(e["split"] == CLIP_EVAL for e in plan.values())
    print(
        f"{'DRY RUN: would add' if dry else 'added'} {len(plan)} clip(s) to {path} "
        f"({n_eval} eval, {len(plan) - n_eval} train); {len(clips)} clips discovered under {root}"
    )
    for ds, ids in sorted(found.items()):
        new = Counter(plan[c]["split"] for c in ids if c in plan)
        total = Counter(
            split_of_name(c) if not dry else _hypothetical(c, plan) for c in ids
        )
        print(
            f"  {ds}: {len(ids)} clips -> "
            + ", ".join(f"{s}={total.get(s, 0)}" for s in SPLITS)
            + f"  (new entries: {sum(new.values())})"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(plan, indent=1, sort_keys=True))
        print(f"plan written to {args.json}")
    return 0


def _hypothetical(cid: str, plan: dict[str, dict]) -> str:
    """Split of ``cid`` as if the dry-run plan were already applied."""
    key = match_key_of(cid)
    if _match_is_heldout(key, cid):
        return EVAL_HELDOUT
    entry = plan.get(cid)
    if entry:
        return EVAL_IN_MATCH if entry["split"] == CLIP_EVAL else TRAIN
    try:
        return split_of(key, cid)
    except RegisterError:
        return TRAIN


def _cmd_stats(args: argparse.Namespace) -> int:
    found = discover_clips(Path(args.data_root) if args.data_root else None)
    st = _stats(found)
    print(f"{'dataset':24s} " + " ".join(f"{s:>20s}" for s in (*SPLITS, "total")))
    for ds, row in st.items():
        print(f"{ds:24s} " + " ".join(f"{row[s]:20d}" for s in (*SPLITS, "total")))
    return 0


def _cmd_contamination(args: argparse.Namespace) -> int:
    reps = [
        contamination(Path(d), train_splits=args.train_subsets.split(","))
        for d in args.dirs
    ]
    if args.json:
        print(json.dumps(reps, indent=1))
    else:
        print("\n\n".join(format_contamination(r) for r in reps))
    return (
        1
        if args.strict and any(sum(r["train_contaminated"].values()) for r in reps)
        else 0
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m footy_track.splits",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="registers; how each NAME resolves")
    s.add_argument("names", nargs="*")
    s.set_defaults(fn=_cmd_show)
    a = sub.add_parser(
        "assign",
        help="add unregistered clips to the clip register (dry-run by default)",
    )
    a.add_argument("--dry-run", action="store_true", help="(default) only report")
    a.add_argument("--apply", action="store_true", help="really write the register")
    a.add_argument("--data-root")
    a.add_argument(
        "--register", help="clip register path (default: sibling of match_splits.json)"
    )
    a.add_argument("--json", help="write the plan here")
    a.set_defaults(fn=_cmd_assign)
    t = sub.add_parser("stats", help="clips per split per dataset")
    t.add_argument("--data-root")
    t.set_defaults(fn=_cmd_stats)
    c = sub.add_parser(
        "contamination", help="eval items present in training datasets (read-only)"
    )
    c.add_argument("dirs", nargs="+")
    c.add_argument(
        "--train-subsets",
        default="train",
        help="comma list of dataset subsets that are training data (train, val|valid, test)",
    )
    c.add_argument("--json", action="store_true")
    c.add_argument(
        "--strict", action="store_true", help="exit 1 when anything is contaminated"
    )
    c.set_defaults(fn=_cmd_contamination)
    ap.add_argument(
        "--preview",
        action="store_true",
        help="treat a missing clip register as empty (hash rule only); for read-only previews",
    )
    args = ap.parse_args(argv)
    if args.preview:
        os.environ["FOOTY_CLIP_SPLITS_PREVIEW"] = "1"
    if args.cmd == "assign" and args.apply and args.dry_run:
        ap.error("--apply and --dry-run are mutually exclusive")
    try:
        return args.fn(args)
    except RegisterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
