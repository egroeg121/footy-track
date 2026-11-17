"""Labelling jobs: other branches / worktrees / Prefect flows ask main for labels.

A *job* is a named batch of things to label in one dataset. Anyone can submit
one (``POST /api/jobs``), the open jobs show up on the home page task board with
a deep link, and the requester later fetches the labels by job id or title
(``GET /api/jobs/{id-or-title}/results``). Progress is **computed from the real
label store** (sidecars, verdict logs, calibration GT), never from a counter, so
it cannot drift from what was actually saved.

Storage: one JSON file per job under ``$FOOTY_LABELLING_JOBS_DIR`` (default
``/mnt/storage/footy_data/labelling_jobs``, created on first submit), written
atomically. The label files themselves are only ever read here.

Item shapes per kind (``items`` is always a list of objects):

==============  ==============================================================
kind            item
==============  ==============================================================
ball_place      ``{clip, frame_start, frame_end, step?}`` (a window, both ends
                inclusive; ``start_frame``/``end_frame`` are accepted aliases)
                or ``{clip, frame}``. A frame is labelled when the clip's GT
                sidecar has a human ball box (tag ``labeller`` or
                ``human_checked``) or a ``no_ball``/``not_broadcast`` marker.
calibration     same shapes. Labelled when the frame has a ``calibration_gt``
                entry or is flagged not-broadcast.
frame_type      same shapes. Labelled when the frame has a ``frame_type`` entry.
ball_check      pointer card ``{clip, frame_index, box_index, source?}`` (``source``
                is ``sidecar`` (default), ``cand`` or ``list``) or an **inline
                card** ``{clip, frame_index, bbox{x,y,w,h normalised}, other_bbox?,
                label?, ...}`` that carries its own geometry (see below). Labelled
                when the verdict log has a live verdict for it.
identity        ``{clip}``. Labelled when an identity pair/review record
                mentions the clip.
other           anything; no computed progress (close it by hand).
==============  ==============================================================

Extra keys on an item (``stratum``, ``reason``, ...) are kept and echoed back.

Inline Ball Check cards
-----------------------
An item with a ``bbox`` is served by the job itself, in item order, with no review-list file and no
dataset-registry edit (``/ball_check?dataset=D&source=list&job=<id>``). The server assigns each card
a ``box_index`` of ``INLINE_BOX_BASE + hash(job id, item position)`` (>= 2**30, list files can never
reach it) and the verdict is written with ``source: "list"`` exactly like a list-mode verdict, into
``<marks_dir>/ball_checks/<clip>.jsonl``. Because the index is derived from the job id, two jobs on
one dataset can never collide on ``(clip, frame, box_index)`` even for the same frame, and a re-run
under a new job id is a fresh set of verdicts. What the page *shows or draws* is only ``label``,
``provenance``/``role``, ``confidence``, ``note`` and ``other_bbox``: put nothing there for a blind
batch. Every other key is a hidden key: kept in the job, returned in ``results`` (``labels[].meta`` and
``items[].item``) and never sent to the phone.

Bounds
------
Jobs are capped (``MAX_CARDS_DEFAULT`` ball_check cards, ``max_cards`` raises it to
``MAX_CARDS_HARD``; frame kinds have their own larger caps), and the cards waiting to be labelled
across all *prepared* jobs are capped too (``MAX_QUEUED_CARDS``). A job never expires: it stays on
the board until done or cancelled. Only its pre-rendered stills/clips are temporary, see
``ball_check_cache.gc``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlencode

from footy_track.clip_id import canonical_clip_id
from footy_track.labeller import datasets
from footy_track.labeller.constants import BALL_LABELS, NO_BALL_TAG, NOT_BROADCAST_TAG

DEFAULT_JOBS_DIR = "/mnt/storage/footy_data/labelling_jobs"
KINDS = ("ball_place", "ball_check", "calibration", "frame_type", "identity", "other")
STATUSES = ("preparing", "open", "in_progress", "done", "cancelled")
LIVE = ("open", "in_progress")
#: ``preparing`` = renders are being made ahead of time; the phone and the board do not offer it yet.
ACTIVE = ("preparing", *LIVE)
DEVICES = ("phone", "laptop", "either")
RESULTS_SCHEMA = "labelling-job-results/v1"


def _env_int(var: str, default: int) -> int:
    try:
        return int(os.environ.get(var, default))
    except ValueError:
        return default


#: Ball Check cards per job: default cap, and the ceiling an explicit ``max_cards`` cannot pass.
MAX_CARDS_DEFAULT = _env_int("FOOTY_JOBS_MAX_CARDS", 500)
MAX_CARDS_HARD = max(MAX_CARDS_DEFAULT, _env_int("FOOTY_JOBS_MAX_CARDS_HARD", 2000))
#: Frames per job for the frame kinds (no renders involved, so a larger bound).
MAX_FRAMES_DEFAULT = _env_int("FOOTY_JOBS_MAX_FRAMES", 5000)
MAX_FRAMES_HARD = max(MAX_FRAMES_DEFAULT, _env_int("FOOTY_JOBS_MAX_FRAMES_HARD", 20000))
#: Unlabelled cards allowed across all jobs that are being / have been prepared.
MAX_QUEUED_CARDS = _env_int("FOOTY_JOBS_MAX_QUEUED_CARDS", 1500)
#: Inline cards get ``box_index`` in [INLINE_BOX_BASE, INLINE_BOX_BASE + 2**29).
INLINE_BOX_BASE = 1 << 30
#: A queued run nobody picked up / a running one that stopped reporting is declared failed.
PREPARE_STALE_QUEUED_S = 6 * 3600
PREPARE_STALE_RUNNING_S = 20 * 60
#: Rough disk need of one prepared card (still + clip), for the free-space floor check.
EST_BYTES_PER_CARD = 600_000
INLINE_SOURCE = "list"
#: What the Ball Check page shows or draws for a card; everything else stays server-side.
PAGE_KEYS = ("label", "provenance", "role", "confidence", "note", "other_bbox")
_CARD_CORE = ("clip", "frame_index", "box_index", "source", "bbox", *PAGE_KEYS)
HUMAN_TAGS = {"labeller", "human_checked"}
_FRAME_KINDS = ("ball_place", "calibration", "frame_type")
_REQUESTER_KEYS = ("branch", "commit", "host", "session")

_LOCK = threading.RLock()


class JobError(Exception):
    """A client error: ``status`` is the HTTP code, ``detail`` the message."""

    def __init__(self, status: int, detail: str, **extra) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.extra = extra


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def jobs_dir() -> Path:
    return Path(os.environ.get("FOOTY_LABELLING_JOBS_DIR") or DEFAULT_JOBS_DIR).expanduser()


def _job_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.json"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_atomic(path: Path, doc: dict, *, exclusive: bool = False) -> bool:
    """Write ``doc`` via tmp + rename. ``exclusive``: fail (False) if ``path`` exists.

    Exclusive uses ``os.link`` (atomic, never replaces), so two processes
    submitting the same idempotency key cannot both create the job.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o664)
        if exclusive:
            try:
                os.link(tmp, path)
            except FileExistsError:
                return False
            os.unlink(tmp)
            return True
        os.replace(tmp, path)
        return True
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_job(job_id: str) -> dict | None:
    try:
        doc = json.loads(_job_path(job_id).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("id") == job_id else None


def all_jobs() -> list[dict]:
    d = jobs_dir()
    if not d.is_dir():
        return []
    out = []
    for p in d.glob("*.json"):
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("id") and doc.get("title"):
            out.append(doc)
    out.sort(key=lambda j: (j.get("created") or "", j["id"]))
    return out


# ---------------------------------------------------------------------------
# Validation / normalisation
# ---------------------------------------------------------------------------


def _int(v, what: str) -> int:
    if isinstance(v, bool):
        raise JobError(422, f"{what} must be an integer")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise JobError(422, f"{what} must be an integer") from None
    if f != int(f) or f < 0:
        raise JobError(422, f"{what} must be a non-negative integer")
    return int(f)


def _clip(item: dict, i: int) -> str:
    clip = item.get("clip")
    if not isinstance(clip, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", clip):
        raise JobError(422, f"items[{i}].clip must be a clip name like 'arsenal_nottinghamforest_seg003'")
    return clip


def _norm_box(v: object, what: str) -> dict:
    """A normalised ``{x, y, w, h}`` box (all in 0..1, positive size)."""
    if not isinstance(v, dict):
        raise JobError(
            422,
            f"{what} must be an object {{x, y, w, h}} in normalised 0..1 coordinates",
        )
    try:
        b = {k: float(v[k]) for k in "xywh"}
    except (KeyError, TypeError, ValueError):
        raise JobError(422, f"{what} needs numeric x, y, w, h") from None
    if not all(0.0 <= b[k] <= 1.0 for k in "xywh") or b["w"] <= 0 or b["h"] <= 0:
        raise JobError(422, f"{what} must be normalised (0..1) with positive w and h")
    return b


def _wants_inline(item: dict) -> bool:
    """A ball_check item with a ``bbox`` and no ``box_index`` is an inline card."""
    return "bbox" in item and "box_index" not in item


def is_inline(item: dict) -> bool:
    """An inline card that has been stored (its ``box_index`` was assigned)."""
    return (
        "bbox" in item
        and item.get("source") == INLINE_SOURCE
        and isinstance(item.get("box_index"), int)
        and item["box_index"] >= INLINE_BOX_BASE
    )


def _unit_cap(kind: str, raw_max: object) -> tuple[int | None, int | None]:
    """``(effective cap, hard ceiling)`` for a kind, honouring a requested ``max_cards``."""
    if kind == "ball_check":
        default, hard = MAX_CARDS_DEFAULT, MAX_CARDS_HARD
    elif kind in _FRAME_KINDS:
        default, hard = MAX_FRAMES_DEFAULT, MAX_FRAMES_HARD
    else:
        return None, None
    if raw_max is None:
        return default, hard
    if isinstance(raw_max, bool) or not isinstance(raw_max, int) or raw_max < 1:
        raise JobError(422, "max_cards must be a positive integer")
    if raw_max > hard:
        raise JobError(
            422,
            f"max_cards {raw_max} is above the hard ceiling of {hard} for {kind} jobs",
        )
    return raw_max, hard


def _normalise_items(kind: str, raw: object, max_cards: object = None) -> list[dict]:  # noqa: PLR0912
    if raw is None:
        raw = []
    if not isinstance(raw, list) or not all(isinstance(x, dict) for x in raw):
        raise JobError(422, "items must be a list of objects")
    if kind == "other":
        return [dict(x) for x in raw]
    if not raw:
        raise JobError(422, f"kind {kind!r} needs at least one item")
    out: list[dict] = []
    for i, src in enumerate(raw):
        it = dict(src)
        it["clip"] = _clip(it, i)
        if kind in _FRAME_KINDS:
            if "frame" in it:
                it["frame"] = _int(it["frame"], f"items[{i}].frame")
            else:
                for new, old in (("frame_start", "start_frame"), ("frame_end", "end_frame")):
                    if new not in it and old in it:
                        it[new] = it.pop(old)
                if "frame_start" not in it or "frame_end" not in it:
                    raise JobError(
                        422, f"items[{i}] needs {{clip, frame}} or {{clip, frame_start, frame_end, step?}}"
                    )
                it["frame_start"] = _int(it["frame_start"], f"items[{i}].frame_start")
                it["frame_end"] = _int(it["frame_end"], f"items[{i}].frame_end")
                it["step"] = _int(it.get("step") or 1, f"items[{i}].step")
                if it["step"] < 1 or it["frame_end"] < it["frame_start"]:
                    raise JobError(422, f"items[{i}]: need frame_end >= frame_start and step >= 1")
        elif kind == "ball_check":
            if _wants_inline(it):  # inline card: carries its own geometry
                it["frame_index"] = _int(
                    it.get("frame_index"), f"items[{i}].frame_index"
                )
                it["bbox"] = _norm_box(it["bbox"], f"items[{i}].bbox")
                if it.get("other_bbox") is not None:
                    it["other_bbox"] = _norm_box(
                        it["other_bbox"], f"items[{i}].other_bbox"
                    )
                else:
                    it.pop("other_bbox", None)
                if it.get("source", INLINE_SOURCE) != INLINE_SOURCE:
                    raise JobError(
                        422, f"items[{i}]: an inline card (with bbox) has source 'list'"
                    )
                it["source"] = (
                    INLINE_SOURCE  # box_index is assigned by the server from the job id
                )
            else:
                for k in ("frame_index", "box_index"):
                    it[k] = _int(it.get(k), f"items[{i}].{k}")
                it["source"] = str(it.get("source") or "sidecar")
                if it["source"] not in ("sidecar", "cand", "list"):
                    raise JobError(
                        422, f"items[{i}].source must be sidecar, cand or list"
                    )
        out.append(it)
    cap, hard = _unit_cap(kind, max_cards)
    n_units = sum(len(_item_units(kind, it)) for it in out)
    if cap is not None and n_units > cap:
        what = "cards" if kind == "ball_check" else "frames"
        if cap < hard:
            hint = f"; pass max_cards (CLI: --allow-large) to raise the cap, up to a hard ceiling of {hard}"
        else:
            hint = f"; {hard} is the hard ceiling and cannot be raised, split the batch"
        raise JobError(
            422,
            f"job has {n_units} {what}, above the cap of {cap}{hint}",
            cap=cap,
            hard_cap=hard,
        )
    return out


def _assign_inline_boxes(job_id: str, dataset: str, items: list[dict]) -> None:
    """Give every inline card its ``box_index`` (see the module docstring); unique per
    ``(clip, frame)`` within the job and against every other job on the dataset."""
    taken: set[tuple] = set(inline_index(dataset))
    for i, it in enumerate(items):
        if not _wants_inline(it):
            continue
        h = int(hashlib.sha1(f"{job_id}:{i}".encode()).hexdigest()[:8], 16) & (
            (1 << 29) - 1
        )  # noqa: S324
        while (it["clip"], it["frame_index"], INLINE_BOX_BASE + h) in taken:
            h = (h + 1) & ((1 << 29) - 1)
        it["box_index"] = INLINE_BOX_BASE + h
        taken.add((it["clip"], it["frame_index"], it["box_index"]))


def _item_units(kind: str, item: dict) -> list:
    """The things to label in one item: frame indices, card keys or the clip."""
    if kind in _FRAME_KINDS:
        if "frame" in item:
            return [item["frame"]]
        return list(range(item["frame_start"], item["frame_end"] + 1, item["step"]))
    if kind == "ball_check":
        return [(item["source"], item["frame_index"], item.get("box_index"))]
    if kind == "identity":
        return [item["clip"]]
    return []


def _clean_requester(raw: object) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise JobError(422, "requester must be an object {branch, commit, host, session}")
    return {k: str(raw[k])[:200] for k in _REQUESTER_KEYS if raw.get(k) not in (None, "")}


def _sha_id(key: str) -> str:
    return "j-" + hashlib.sha1(key.encode()).hexdigest()[:12]  # noqa: S324 (not security)


def _new_id() -> str:
    return f"j-{datetime.now(UTC):%Y%m%d}-{secrets.token_hex(3)}"


# ---------------------------------------------------------------------------
# Submit / lookup / state changes
# ---------------------------------------------------------------------------


def submit(body: dict, default_dataset: str) -> tuple[dict, bool]:  # noqa: PLR0912, PLR0915
    """Validate and store a job. Returns ``(job, created)``; a repeated ``key`` returns the
    existing job (``created`` False) instead of a duplicate."""
    if not isinstance(body, dict):
        raise JobError(422, "body must be a JSON object")
    title = body.get("title")
    if not isinstance(title, str) or not title.strip():
        raise JobError(422, "title is required")
    title = title.strip()
    if len(title) > 200:
        raise JobError(422, "title must be at most 200 characters")
    kind = body.get("kind") or "other"
    if kind not in KINDS:
        raise JobError(422, f"kind must be one of {list(KINDS)}")
    dataset = body.get("dataset") or default_dataset
    try:
        dataset = datasets.get(dataset).name
    except datasets.UnknownDatasetError:
        raise JobError(422, f"unknown dataset {dataset!r}", datasets=datasets.names()) from None
    device = body.get("device") or "either"
    if device not in DEVICES:
        raise JobError(422, f"device must be one of {list(DEVICES)}")
    priority = body.get("priority", 2)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise JobError(422, "priority must be an integer (1 = first)")
    key = body.get("key")
    if key is not None and (not isinstance(key, str) or not key.strip() or len(key) > 300):
        raise JobError(422, "key must be a non-empty string")
    url = body.get("url")
    if url is not None and (not isinstance(url, str) or not url.startswith("/")):
        raise JobError(422, "url must be a path on this server, like /ball_check?dataset=nottinghamforest")
    items = _normalise_items(kind, body.get("items"), body.get("max_cards"))
    want_prepare = body.get("prepare")
    where = None
    if want_prepare:
        if kind != "ball_check":
            raise JobError(422, "prepare is only for ball_check jobs")
        where = _where(
            want_prepare.get("where") if isinstance(want_prepare, dict) else None
        )
    job = {
        "id": _sha_id(key) if key else _new_id(),
        "title": title,
        "kind": kind,
        "dataset": dataset,
        "items": items,
        "why": str(body.get("why") or "")[:1000],
        "device": device,
        "priority": priority,
        "requester": _clean_requester(body.get("requester")),
        "key": key,
        "status": "open",
        "created": _now(),
        "updated": _now(),
    }
    if url:
        job["url"] = url
    with _LOCK:
        if key:
            existing = read_job(job["id"])
            if existing is not None:
                return existing, False
        if want_prepare:  # refuse before anything is created
            _check_prepare_allowed(len(items), exclude=None)
            job["status"] = "preparing"
            job["prepare"] = _new_prepare(where, hold=True, total=len(items))
        if key:
            _assign_inline_boxes(job["id"], dataset, items)
            if _write_atomic(_job_path(job["id"]), job, exclusive=True):
                return job, True
            return read_job(job["id"]) or job, False  # another process created it first
        for _ in range(5):
            job["id"] = _new_id()
            _assign_inline_boxes(job["id"], dataset, items)
            if _write_atomic(_job_path(job["id"]), job, exclusive=True):
                return job, True
    raise JobError(500, "could not allocate a job id")


def resolve(ref: str) -> dict:
    """A job by id, else by exact title (must be unambiguous). Raises JobError 404/409."""
    ref = ref.strip()
    if re.fullmatch(r"j-[0-9a-z-]+", ref):
        job = read_job(ref)
        if job is not None:
            return job
    matches = [j for j in all_jobs() if j["title"] == ref]
    if len(matches) > 1:
        live = [j for j in matches if j["status"] != "cancelled"]
        matches = live or matches
    if not matches:
        raise JobError(404, f"no job with id or title {ref!r}")
    if len(matches) > 1:
        raise JobError(
            409,
            f"title {ref!r} matches {len(matches)} jobs; use an id",
            ids=[j["id"] for j in matches],
        )
    return matches[0]


def _set_status(job_id: str, new: str, *, only_from: tuple[str, ...] | None = None, by: str = "api") -> dict:
    with _LOCK:
        job = read_job(job_id)
        if job is None:
            raise JobError(404, f"no job {job_id!r}")
        if only_from is not None and job["status"] not in only_from:
            return job
        if job["status"] != new:
            job["status"] = new
            job["updated"] = _now()
            if new in ("done", "cancelled"):
                job["closed"] = _now()
                job["closed_by"] = by
            else:
                job.pop("closed", None)
                job.pop("closed_by", None)
            _write_atomic(_job_path(job_id), job)
        return job


def close(job_id: str, new: str) -> dict:
    if new not in ("done", "cancelled", "open"):
        raise JobError(422, "bad status")
    return _set_status(job_id, new)


# ---------------------------------------------------------------------------
# Label store readers (read-only; cached by file signature)
# ---------------------------------------------------------------------------

_FILE_CACHE: dict[str, tuple[tuple, object]] = {}


def _cached(path: Path, parse):
    try:
        st = path.stat()
    except OSError:
        return None, None
    sig = (st.st_mtime_ns, st.st_size)
    hit = _FILE_CACHE.get(str(path))
    if hit is not None and hit[0] == sig:
        return hit[1], st.st_mtime
    try:
        val = parse(path)
    except (OSError, ValueError):
        return None, None
    _FILE_CACHE[str(path)] = (sig, val)
    return val, st.st_mtime


def _gt_stem(marks_dir: Path, clip: str) -> str:
    """Existing GT sidecar stem for ``clip``: exact, else an alias with the same canonical id."""
    if (marks_dir / f"{clip}.jsonl").exists():
        return clip
    cid = canonical_clip_id(clip)
    aliases = sorted(p.stem for p in marks_dir.glob("*.jsonl") if canonical_clip_id(p.stem) == cid)
    if aliases:
        return cid if cid in aliases else aliases[0]
    return clip


def _parse_sidecar(path: Path) -> dict[int, list[dict]]:
    """frame -> label rows that count as a human label (see module docstring)."""
    frames: dict[int, list[dict]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            idx = int(d["frame_index"])
        except (ValueError, KeyError, TypeError):
            continue
        tags = [t for t in (d.get("tags") or []) if isinstance(t, str)]
        row = {"frame_index": idx, "tags": tags}
        if NOT_BROADCAST_TAG in tags or NO_BALL_TAG in tags:
            row.update(status=NOT_BROADCAST_TAG if NOT_BROADCAST_TAG in tags else NO_BALL_TAG)
        else:
            label = next((t for t in tags if t in BALL_LABELS), None)
            if label is None or not HUMAN_TAGS & set(tags):
                continue  # not a ball, or machine-only (tier 3): not a label
            tier = 1 if "labeller" in tags else 2
            row.update(
                status="ball",
                label=label,
                bbox=d.get("bbox"),
                center=d.get("center"),
                provenance="labeller" if tier == 1 else "machine+human_checked",
                tier=tier,
            )
        frames.setdefault(idx, []).append(row)
    return frames


def _parse_json(path: Path) -> dict:
    doc = json.loads(path.read_text())
    return doc if isinstance(doc, dict) else {}


def _frame_unit_labels(kind: str, ds: datasets.Dataset, clip: str) -> tuple[dict[int, list[dict]], dict]:
    """For a frame kind: ``({frame: [label rows]}, clip_info)`` from the real store."""
    if kind == "ball_place":
        stem = _gt_stem(ds.marks_dir, clip)
        frames, mtime = _cached(ds.marks_dir / f"{stem}.jsonl", _parse_sidecar)
        info = {"clip": clip, "store": f"{ds.marks_dir.name}/{stem}.jsonl", "store_mtime": _iso(mtime)}
        return frames or {}, info
    cid = canonical_clip_id(clip)
    out: dict[int, list[dict]] = {}
    store = {}
    if kind == "calibration":
        gt, mtime = _cached(ds.marks_dir / "calibration_gt" / f"{cid}.json", _parse_json)
        store["store"] = f"calibration_gt/{cid}.json"
        store["store_mtime"] = _iso(mtime)
        for k, v in ((gt or {}).get("frames") or {}).items():
            if isinstance(v, dict) and str(k).isdigit():
                out[int(k)] = [{"frame_index": int(k), "status": "calibrated", **_pick(v, _CALIB_FIELDS)}]
    ft, mtime = _cached(ds.marks_dir / "frame_type" / f"{cid}.json", _parse_json)
    for k, v in ((ft or {}).get("frames") or {}).items():
        if isinstance(v, dict) and str(k).isdigit():
            row = {"frame_index": int(k), "status": v.get("type") or "flagged", **_pick(v, ("ts", "run_id", "note"))}
            if kind == "frame_type":
                out[int(k)] = [row]
            else:  # calibration: a not-broadcast frame needs no calibration
                out.setdefault(int(k), [row])
    if kind == "frame_type":
        store["store"] = f"frame_type/{cid}.json"
        store["store_mtime"] = _iso(mtime)
    return out, {"clip": clip, **store}


_CALIB_FIELDS = (
    "frame_index", "correspondences", "H", "H_convention", "pitch_dims", "image_size",
    "method", "inliers", "mean_err_m", "ts", "origin",
)  # fmt: skip


def _pick(d: dict, keys) -> dict:
    return {k: d[k] for k in keys if k in d}


def _iso(mtime: float | None) -> str | None:
    return datetime.fromtimestamp(mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if mtime else None


def _verdicts(ds: datasets.Dataset, clip: str) -> dict[tuple, dict]:
    """Live verdict per (source, frame, box) for one clip (last line wins, null = undo)."""
    path = ds.marks_dir / "ball_checks" / f"{clip}.jsonl"

    def parse(p: Path) -> dict:
        out: dict[tuple, dict] = {}
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec.get("source") or "sidecar", int(rec["frame_index"]), int(rec["box_index"]))
            except (ValueError, KeyError, TypeError):
                continue
            if rec.get("verdict") is None:
                out.pop(key, None)
            else:
                out[key] = rec
        return out

    val, _ = _cached(path, parse)
    return val or {}


def _identity_records(ds: datasets.Dataset) -> list[dict]:
    assert ds.identity_labels_dir is not None
    out: list[dict] = []
    for name in ("identity_pairs.jsonl", "identity_reviews.jsonl"):
        def parse(p: Path) -> list:
            rows = []
            for raw in p.read_text().splitlines():
                with contextlib.suppress(ValueError):
                    rows.append(json.loads(raw))
            return rows

        val, _ = _cached(ds.identity_labels_dir / name, parse)
        out.extend(val or [])
    return out


def _identity_clips(rec: dict) -> set[str]:
    refs = [rec.get("a"), rec.get("b"), rec.get("tracklet")]
    return {r["clip"] for r in refs if isinstance(r, dict) and r.get("clip")}


# ---------------------------------------------------------------------------
# Progress + results (one pass computes both)
# ---------------------------------------------------------------------------


def compute(job: dict) -> dict:  # noqa: PLR0912
    """Walk the job's items against the label store.

    Returns ``{"supported", "done", "total", "labels", "items", "next", "stores"}``.
    ``labels`` are the canonical rows (results); ``items`` has per-item counts.
    """
    kind = job["kind"]
    if kind == "other":
        return {"supported": False, "done": None, "total": None, "labels": [], "items": [], "next": None, "stores": []}
    try:
        ds = datasets.get(job["dataset"])
    except datasets.UnknownDatasetError:
        return {
            "supported": False, "error": f"dataset {job['dataset']!r} is not in the registry",
            "done": None, "total": None, "labels": [], "items": [], "next": None, "stores": [],
        }  # fmt: skip
    labels: list[dict] = []
    per_item: list[dict] = []
    stores: dict[str, dict] = {}
    nxt: dict | None = None
    done = total = 0
    ident = _identity_records(ds) if kind == "identity" else []
    for i, it in enumerate(job["items"]):
        units = _item_units(kind, it)
        n_done = 0
        first_missing = None
        if kind in _FRAME_KINDS:
            frames, info = _frame_unit_labels(kind, ds, it["clip"])
            stores[it["clip"]] = info
            for f in units:
                rows = frames.get(f)
                if rows:
                    n_done += 1
                    labels.extend({"clip": it["clip"], **r} for r in rows)
                elif first_missing is None:
                    first_missing = {"clip": it["clip"], "frame": f}
        elif kind == "ball_check":
            verd = _verdicts(ds, it["clip"])
            for unit in units:
                rec = verd.get(unit)
                if rec is not None:
                    n_done += 1
                    row = _verdict_row(rec)
                    if is_inline(it):
                        row["meta"] = hidden_keys(
                            it
                        )  # hidden keys travel with the result
                    labels.append(row)
                elif first_missing is None:
                    first_missing = {
                        "clip": it["clip"], "source": unit[0], "frame": unit[1], "box_index": unit[2]
                    }  # fmt: skip
        elif kind == "identity":
            recs = [r for r in ident if it["clip"] in _identity_clips(r)]
            if recs:
                n_done = 1
                labels.extend({"clip": it["clip"], **r} for r in recs)
            else:
                first_missing = {"clip": it["clip"]}
        per_item.append({"index": i, "item": it, "total": len(units), "done": n_done})
        done += n_done
        total += len(units)
        if nxt is None and first_missing is not None:
            nxt = first_missing
    return {
        "supported": True, "done": done, "total": total, "labels": labels,
        "items": per_item, "next": nxt, "stores": list(stores.values()),
    }  # fmt: skip


_VERDICT_FIELDS = (
    "clip", "frame_index", "box_index", "source", "verdict", "bbox", "corrected_bbox",
    "label", "provenance", "confidence", "ts",
)  # fmt: skip


def _verdict_row(rec: dict) -> dict:
    row = {k: rec.get(k) for k in _VERDICT_FIELDS}
    row["source"] = row["source"] or "sidecar"
    return row


def progress_of(c: dict) -> dict:
    if not c["supported"]:
        out = {"supported": False, "done": None, "total": None, "remaining": None, "fraction": None, "next": None}
        if c.get("error"):
            out["error"] = c["error"]
        return out
    total, done = c["total"], c["done"]
    return {
        "supported": True,
        "done": done,
        "total": total,
        "remaining": total - done,
        "fraction": round(done / total, 4) if total else 0.0,
        "next": c["next"],
    }


def frame_list(job: dict) -> dict:
    """The job's frames in order (items in job order, each ``frame_start..frame_end`` by ``step``)
    with a ``done`` flag decided by the same rule as :func:`compute` (``_frame_unit_labels``).

    Feeds the labeller's job mode (``/labeller?job=<id>``). Only frame kinds have one.
    """
    kind = job["kind"]
    if kind not in _FRAME_KINDS:
        raise JobError(422, f"job kind {kind!r} has no frame list")
    try:
        ds = datasets.get(job["dataset"])
    except datasets.UnknownDatasetError:
        msg = f"dataset {job['dataset']!r} is not in the registry"
        raise JobError(422, msg) from None
    by_clip: dict[str, dict[int, list[dict]]] = {}
    frames: list[dict] = []
    for i, it in enumerate(job["items"]):
        clip = it["clip"]
        if clip not in by_clip:
            by_clip[clip] = _frame_unit_labels(kind, ds, clip)[0]
        labelled = by_clip[clip]
        meta = {k: it[k] for k in ("stratum", "reason") if k in it}
        frames.extend(
            {"clip": clip, "frame": f, "item": i, "done": bool(labelled.get(f)), **meta}
            for f in _item_units(kind, it)
        )
    done = sum(1 for f in frames if f["done"])
    return {"id": job["id"], "title": job["title"], "kind": kind, "dataset": job["dataset"],
            "total": len(frames), "done": done, "frames": frames}  # fmt: skip


def refresh_status(job: dict, c: dict) -> dict:
    """Persist the cheap, safe transitions: open -> in_progress at the first label, and
    -> done at 100% (only ever from open/in_progress; never reopens a cancelled/done job)."""
    if not c["supported"] or job["status"] not in LIVE or not c["total"]:
        return job
    if c["done"] >= c["total"]:
        return _set_status(job["id"], "done", only_from=LIVE, by="auto")
    if c["done"] > 0 and job["status"] == "open":
        return _set_status(job["id"], "in_progress", only_from=("open",), by="auto")
    return job


# ---------------------------------------------------------------------------
# Public views
# ---------------------------------------------------------------------------


def deep_link(job: dict, c: dict | None = None) -> str:
    """Path (with ``dataset=``) of the page where this job gets labelled, at the next unlabelled
    thing when known. A submitter-supplied ``url`` wins."""
    if job.get("url"):
        return job["url"]
    q: dict = {"dataset": job["dataset"]}
    nxt = (c or {}).get("next") or {}
    items = job.get("items") or []
    first = items[0] if items else {}
    kind = job["kind"]
    if kind in ("ball_place", "calibration", "frame_type"):
        q["clip"] = nxt.get("clip") or first.get("clip")
        frame = nxt.get("frame", first.get("frame", first.get("frame_start")))
        if frame is not None:
            q["frame"] = frame
        page = "/labeller" if kind == "ball_place" else "/calibration"
        if kind == "ball_place":
            # job mode: the page steps through this job's own frames
            q["job"] = job["id"]
    elif kind == "ball_check":
        q["source"] = nxt.get("source") or first.get("source", "sidecar")
        clips = {it["clip"] for it in items}
        if len(clips) == 1:
            q["clip"] = next(iter(clips))
        q["job"] = job[
            "id"
        ]  # lets the page show prep progress and re-prepare evicted renders
        if any(is_inline(it) for it in items):
            q["source"] = "list"
        page = "/ball_check"
    elif kind == "identity":
        page = "/identity"
    else:
        page = "/main"
    return f"{page}?{urlencode({k: v for k, v in q.items() if v is not None}, quote_via=quote)}"


def public(job: dict, c: dict | None = None, *, items: bool = True) -> dict:
    out = {k: v for k, v in job.items() if k != "items"}
    out["n_items"] = len(job.get("items") or [])
    if items:
        out["items"] = job.get("items") or []
    out["url"] = deep_link(job, c)
    if c is not None:
        out["progress"] = progress_of(c)
    return out


def job_with_progress(job: dict) -> tuple[dict, dict]:
    """``(job, compute())`` with auto status transitions applied (runs under the job's dataset)."""
    c = compute(job)
    return refresh_status(job, c), c


def results_doc(job: dict, c: dict) -> dict:
    return {
        "schema": RESULTS_SCHEMA,
        "generated_at": _now(),
        "job": public(job, c, items=False),
        "progress": progress_of(c),
        "items": c["items"],
        "stores": c["stores"],
        "labels": c["labels"],
    }


def filter_jobs(
    *,
    status: str | None = None,
    title: str | None = None,
    kind: str | None = None,
    dataset: str | None = None,
    branch: str | None = None,
) -> list[dict]:
    out = []
    for j in all_jobs():
        if status and j["status"] != status:
            continue
        if kind and j["kind"] != kind:
            continue
        if dataset and j["dataset"] != dataset:
            continue
        if branch and (j.get("requester") or {}).get("branch") != branch:
            continue
        if title and title != j["title"] and title.lower() not in j["title"].lower():
            continue
        out.append(j)
    return out


# ---------------------------------------------------------------------------
# Task board
# ---------------------------------------------------------------------------


def board_tasks() -> list[dict]:
    """Open jobs as task-board entries. Never raises: the board must not break on a bad job."""
    tasks = []
    try:
        jobs = [j for j in all_jobs() if j["status"] in ACTIVE]
    except OSError:
        return []
    for stored in jobs:
        try:
            if stored["status"] == "preparing":
                tasks.append(_preparing_task(stored))
                continue
            j, c = job_with_progress(stored)
            if j["status"] not in LIVE:
                continue  # just auto-completed
            prog = progress_of(c)
            req = j.get("requester") or {}
            t = {
                "id": j["id"],
                "title": j["title"],
                "why": j.get("why") or f"{j['kind']} job in {j['dataset']}"
                + (f" from {req['branch']}" if req.get("branch") else ""),
                "url": deep_link(j, c),
                "device": j.get("device", "either"),
                "priority": j.get("priority", 2),
                "added": (j.get("created") or "")[:10],
                "owner": req.get("branch") or "",
                "job": True,
                "kind": j["kind"],
                "dataset": j["dataset"],
            }
            if prog["supported"]:
                t["count_url"] = f"/api/jobs/{j['id']}"
                t["count_path"] = "progress.remaining"
                t["count_label"] = "left"
                t["remaining"] = prog["remaining"]
            tasks.append(t)
        except Exception:  # noqa: BLE001, S112 - one bad job must not hide the others
            continue
    return tasks


# ---------------------------------------------------------------------------
# Ball Check: inline cards
# ---------------------------------------------------------------------------

_INLINE_CACHE: dict = {"sig": None, "by_dataset": {}}


def hidden_keys(item: dict) -> dict:
    """The submitter's keys that the page never shows (they stay in the job and the results)."""
    return {k: v for k, v in item.items() if k not in _CARD_CORE}


def page_card(item: dict) -> dict:
    """What the phone gets for an inline card: geometry plus the few keys it prints or draws."""
    return {k: item[k] for k in _CARD_CORE if k in item}


def _inline_by_dataset() -> dict[str, dict[tuple, tuple[str, dict]]]:
    d = jobs_dir()
    try:
        sig = (str(d), d.stat().st_mtime_ns)
    except OSError:
        return {}
    if _INLINE_CACHE["sig"] == sig:
        return _INLINE_CACHE["by_dataset"]
    out: dict[str, dict[tuple, tuple[str, dict]]] = {}
    for j in all_jobs():
        if j.get("kind") != "ball_check" or j.get("status") == "cancelled":
            continue
        for it in j.get("items") or []:
            if is_inline(it):
                out.setdefault(j["dataset"], {})[
                    (it["clip"], it["frame_index"], it["box_index"])
                ] = (j["id"], it)
    _INLINE_CACHE["sig"] = sig
    _INLINE_CACHE["by_dataset"] = out
    return out


def inline_index(dataset: str) -> dict[tuple, tuple[str, dict]]:
    """``{(clip, frame_index, box_index): (job id, card)}`` of the inline cards of one dataset."""
    return _inline_by_dataset().get(dataset, {})


def inline_card(
    dataset: str, clip: str, frame_index: int, box_index: int
) -> tuple[str, dict] | None:
    return inline_index(dataset).get((clip, frame_index, box_index))


def todo_items(job: dict) -> list[tuple[int, dict]]:
    """``(position, item)`` of the ball_check cards with no live verdict yet, in job order."""
    ds = datasets.get(job["dataset"])
    out = []
    for i, it in enumerate(job["items"]):
        if _verdicts(ds, it["clip"]).get(_item_units("ball_check", it)[0]) is None:
            out.append((i, it))
    return out


# ---------------------------------------------------------------------------
# Ball Check: preparation (renders made ahead of time by the ballcheck-prepare flow)
# ---------------------------------------------------------------------------

#: Called as ``dispatch(job, where) -> run id`` to queue the Prefect run; tests replace it.
DISPATCH = None


def _where(where: object) -> str:
    where = where or os.environ.get("FOOTY_BALLCHECK_PREPARE_WHERE") or "mini"
    if where not in ("mini", "desktop"):
        raise JobError(422, "prepare.where must be 'mini' or 'desktop'")
    return where


def _new_prepare(where: str, *, hold: bool, total: int) -> dict:
    now = time.time()
    return {
        "state": "queued", "where": where, "total": total, "prepared": 0, "failed": 0, "run_id": None,
        "hold": hold, "requested": now, "heartbeat": now, "finished": None, "error": None,
    }  # fmt: skip


def prepare_in_flight(job: dict, now: float | None = None) -> bool:
    """A prepare run is queued or running and has not gone stale."""
    p = job.get("prepare") or {}
    now = time.time() if now is None else now
    if p.get("state") == "queued":
        return now - p.get("requested", now) < PREPARE_STALE_QUEUED_S
    if p.get("state") == "running":
        return now - p.get("heartbeat", now) < PREPARE_STALE_RUNNING_S
    return False


def queued_cards(exclude: str | None = None) -> int:
    """Cards waiting to be labelled across the jobs that are being or have been prepared."""
    n = 0
    for j in all_jobs():
        if (
            j["id"] == exclude
            or j.get("kind") != "ball_check"
            or j["status"] not in ACTIVE
        ):
            continue
        if (j.get("prepare") or {}).get("state") in ("queued", "running", "ready"):
            try:
                n += len(todo_items(j))
            except datasets.UnknownDatasetError:
                continue
    return n


def _check_prepare_allowed(n_cards: int, exclude: str | None) -> None:
    from footy_track.labeller import ball_check_cache as bcc  # noqa: PLC0415

    queued = queued_cards(exclude)
    if queued + n_cards > MAX_QUEUED_CARDS:
        raise JobError(
            429,
            f"{n_cards} cards to prepare on top of {queued} already waiting would pass the cap of "
            f"{MAX_QUEUED_CARDS} queued cards; label or cancel earlier jobs first",
            queued=queued, cap=MAX_QUEUED_CARDS,
        )  # fmt: skip
    base = bcc.root()
    if base is not None:
        try:
            bcc.check_floor(base, n_cards * EST_BYTES_PER_CARD)
        except bcc.DiskFloorError as e:
            raise JobError(507, f"not enough disk to prepare: {e}") from None


def prepare_request(
    job_id: str, *, hold: bool = False, where: str | None = None
) -> dict:
    """Mark a ball_check job ``queued`` for preparation (caps, duplicates and disk checked).

    ``hold`` also flips an ``open`` job to ``preparing`` so nobody is offered it half-prepared;
    an in-progress job (someone is labelling it) is never held. Returns the stored job.
    """
    where = _where(where)
    with _LOCK:
        job = read_job(job_id)
        if job is None:
            raise JobError(404, f"no job {job_id!r}")
        if job["kind"] != "ball_check":
            raise JobError(422, "only ball_check jobs can be prepared")
        if job["status"] not in ACTIVE:
            raise JobError(409, f"job is {job['status']}; only open jobs are prepared")
        if prepare_in_flight(job):
            raise JobError(409, "a prepare run is already queued or running for this job",
                           run_id=(job.get("prepare") or {}).get("run_id"))  # fmt: skip
        todo = todo_items(job)
        if not todo:
            raise JobError(
                422, "every card of this job is already judged; nothing to prepare"
            )
        _check_prepare_allowed(len(todo), exclude=job_id)
        hold = hold and job["status"] == "open"
        job["prepare"] = _new_prepare(where, hold=hold, total=len(todo))
        if hold:
            job["status"] = "preparing"
        job["updated"] = _now()
        _write_atomic(_job_path(job_id), job)
        return job


def dispatch_prepare(job: dict) -> dict:
    """Queue the Prefect run for a job whose prepare state is ``queued``; on failure the job is
    usable again (renders on demand) with ``prepare.state == "failed"`` and the reason."""
    p = job.get("prepare") or {}
    if p.get("state") != "queued":
        return job
    try:
        run = (
            DISPATCH(job, p["where"])
            if DISPATCH
            else _prefect_dispatch(job, p["where"])
        )
    except Exception as e:  # noqa: BLE001 - any failure leaves the job usable
        return prepare_event(
            job["id"], "abort", error=f"could not queue the Prefect run: {e}"
        )
    return prepare_event(job["id"], "queued", run_id=run)


def _prefect_dispatch(job: dict, where: str) -> str:
    from footy_track.labeller import prefect_client  # noqa: PLC0415

    params: dict = {"job": job["id"], "where": where}
    # A server that is not the live one (tests, a throwaway port) tells the flow how to reach it and
    # which footy-track ``src`` to render with; unset, the flow's defaults (live :8011, deployed main) apply.
    if os.environ.get("FOOTY_LABELLER_URL"):
        params["labeller_url"] = os.environ["FOOTY_LABELLER_URL"]
    if os.environ.get("FOOTY_BALLCHECK_PREPARE_CODE"):
        params["code"] = os.environ["FOOTY_BALLCHECK_PREPARE_CODE"]
    return prefect_client.create_flow_run(params, name=f"bc-prepare-{job['id']}")


def start_prepare(job_id: str, *, hold: bool = False, where: str | None = None) -> dict:
    return dispatch_prepare(prepare_request(job_id, hold=hold, where=where))


def prepare_event(job_id: str, action: str, **kw) -> dict:  # noqa: PLR0912
    """State changes reported by the flow: ``queued`` (run id), ``run``, ``progress``, ``finish``,
    ``abort``. ``finish``/``abort`` release a ``preparing`` job to ``open``."""
    with _LOCK:
        job = read_job(job_id)
        if job is None:
            raise JobError(404, f"no job {job_id!r}")
        p = dict(job.get("prepare") or {})
        now = time.time()
        if (
            not p and action == "run"
        ):  # a run started by hand (``$PF run ballcheck-prepare/...``)
            p = _new_prepare(_where(kw.get("where")), hold=False, total=0)
        if not p:
            raise JobError(409, "job has no prepare request")
        if action == "queued":
            p["run_id"] = kw.get("run_id")
        elif action == "run":
            p.update(
                state="running",
                run_id=kw.get("run_id") or p.get("run_id"),
                started=now,
                heartbeat=now,
            )
        elif action == "progress":
            if p.get("state") not in ("queued", "running"):
                return job
            p.update(state="running", heartbeat=now)
            for k in ("prepared", "total", "failed"):
                if isinstance(kw.get(k), int):
                    p[k] = kw[k]
        elif action in ("finish", "abort"):
            ok = action == "finish" and kw.get("ok", True)
            p.update(
                state="ready" if ok else "failed",
                finished=now,
                heartbeat=now,
                error=None if ok else kw.get("error") or "failed",
            )
            for k in ("prepared", "total", "failed"):
                if isinstance(kw.get(k), int):
                    p[k] = kw[k]
            if job["status"] == "preparing":
                job["status"] = "open"
        else:
            raise JobError(422, f"unknown prepare action {action!r}")
        job["prepare"] = p
        job["updated"] = _now()
        _write_atomic(_job_path(job_id), job)
        return job


def maybe_reprepare(job_id: str) -> bool:
    """Labelling resumed on a job whose renders were evicted: queue a fresh prepare run for its
    remaining unlabelled cards (judged ones are skipped). Quiet and rate-limited; the phone is served
    by on-demand rendering meanwhile. Returns True when a run was queued."""
    job = read_job(job_id)
    if job is None or job.get("kind") != "ball_check" or job["status"] not in LIVE:
        return False
    p = job.get("prepare") or {}
    if p.get("state") != "evicted" or time.time() < p.get("next_auto", 0):
        return False
    try:
        job = prepare_request(job_id, hold=False, where=p.get("where"))
    except JobError:
        _note_auto_failure(job_id)
        return False
    job = dispatch_prepare(job)
    if (job.get("prepare") or {}).get("state") == "failed":
        _note_auto_failure(job_id, error=job["prepare"].get("error"))
        return False
    return True


def _note_auto_failure(job_id: str, error: str | None = None) -> None:
    """Back to ``evicted`` with a 10 minute pause so a broken queue is not hammered."""
    with _LOCK:
        job = read_job(job_id)
        if job is None:
            return
        p = dict(job.get("prepare") or {})
        p.update(state="evicted", next_auto=time.time() + 600)
        if error:
            p["error"] = error
        job["prepare"] = p
        _write_atomic(_job_path(job_id), job)


def _preparing_task(j: dict) -> dict:
    p = j.get("prepare") or {}
    req = j.get("requester") or {}
    k, n = p.get("prepared", 0), p.get("total", len(j.get("items") or []))
    return {
        "id": j["id"],
        "title": j["title"],
        "why": f"Preparing renders {k}/{n}; available when it finishes",
        "url": "",
        "device": j.get("device", "either"),
        "priority": j.get("priority", 2),
        "added": (j.get("created") or "")[:10],
        "owner": req.get("branch") or "",
        "job": True,
        "kind": j["kind"],
        "dataset": j["dataset"],
        "preparing": {"prepared": k, "total": n},
    }


# ---------------------------------------------------------------------------
# Housekeeping: stale runs, idle / finished renders, cache size
# ---------------------------------------------------------------------------


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return (
            datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
        )
    except ValueError:
        return None


def last_activity(job: dict) -> float:
    """Latest sign of life on a job: a verdict on one of its cards, its prepare run, or its creation.
    (Plain ``updated`` is not used: a status flip is not labelling.)"""
    t = max(_ts(job.get("created")) or 0.0, 0.0)
    p = job.get("prepare") or {}
    t = max(t, p.get("finished") or 0.0, p.get("heartbeat") or 0.0)
    if job["kind"] == "ball_check":
        try:
            ds = datasets.get(job["dataset"])
        except datasets.UnknownDatasetError:
            return t
        seen: set[str] = set()
        for it in job["items"]:
            if it["clip"] in seen:
                continue
            seen.add(it["clip"])
            for rec in _verdicts(ds, it["clip"]).values():
                t = max(t, float(rec.get("ts") or 0.0))
    return t


def gc(
    *, dry_run: bool = False, now: float | None = None, cap: int | None = None
) -> dict:  # noqa: PLR0912
    """Housekeeping (``labelling_jobs gc``, and hourly inside the server).

    * a prepare run that stopped reporting is declared failed (a held job is released to ``open``);
    * renders of done/cancelled jobs go 24 h after the job closed, those of a live job after 3 days
      with no labelling activity on it (the job is then marked ``prepare.state = "evicted"`` and is
      re-prepared automatically when labelling resumes);
    * the render cache is held under its size cap by LRU eviction (finished/idle jobs first).

    Jobs and verdicts are never touched apart from the prepare bookkeeping above.
    """
    from footy_track.labeller import ball_check_cache as bcc  # noqa: PLC0415

    now = time.time() if now is None else now
    report: dict = {
        "dry_run": dry_run,
        "stale_prepares": [],
        "evicted_jobs": [],
        "cache": None,
    }
    for j in all_jobs():
        p = j.get("prepare") or {}
        if p.get("state") in ("queued", "running") and not prepare_in_flight(j, now):
            report["stale_prepares"].append(j["id"])
            if not dry_run:
                prepare_event(
                    j["id"],
                    "abort",
                    error="prepare run went stale (no progress report)",
                )
    rc = bcc.cache()
    if rc is None:
        return report
    infos: dict[str, dict] = {}
    by_id = {j["id"]: j for j in all_jobs()}
    for job_id in rc.all_refs():
        j = by_id.get(job_id)
        if j is not None:
            infos[job_id] = {
                "status": j["status"],
                "closed_ts": _ts(j.get("closed")),
                "last_activity": last_activity(j),
            }
    cache_report = bcc.gc(rc, infos, now, cap=cap, dry_run=dry_run)
    report["cache"] = cache_report
    for job_id, why in cache_report["dropped_refs"].items():
        j = by_id.get(job_id)
        if (
            why == "idle"
            and j is not None
            and (j.get("prepare") or {}).get("state") in ("ready", "failed", "running")
        ):
            report["evicted_jobs"].append(job_id)
            if not dry_run:
                with _LOCK:
                    cur = read_job(job_id)
                    if cur is not None and cur["status"] in LIVE:
                        cur["prepare"] = {
                            **(cur.get("prepare") or {}),
                            "state": "evicted",
                            "evicted": now,
                        }
                        _write_atomic(_job_path(job_id), cur)
    return report
