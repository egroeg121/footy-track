"""Ball evaluation panel: a fixed, versioned, probability-sampled set of frames from HELDOUT matches.

Design: ``docs/design/ball_sampling.md`` section 2 (v1 = match is the only stratum, windows drawn uniformly).

Population. The segments of a heldout match (``<FD>/<match>/split_video_broadcast_frames*/segments.json``,
the main-camera cuts) are tiled, per segment, into disjoint *windows*. A window is ``WINDOW_FRAMES`` (5)
frames ``gap`` source frames apart, where ``gap = 5 * max(1, round(0.2 s * fps / 5))`` (so the frames lie on
the stride-5 grid and are ~0.2 s apart: 5 frames at 25 fps, 10 at 50 fps, 10 at 60 fps; the window spans
~0.7-1 s). Window ``k`` of a segment starts at local frame ``k * 5 * gap`` and must end inside the clip, so
windows never cross a segment boundary. The population is the set of these windows, fixed by the
segmentation run (not by any model). Frames are local to the segment clip ``<match>_segNNN`` (frame 0 =
``start_frame`` of ``segments.json``).

Sampling. ``n`` of the ``N`` windows are drawn uniformly without replacement, seeded by
``sha1(panel_version + match)``. Inclusion probabilities are exact: ``pi_window = n / N`` and every frame of
a chosen window has ``pi_frame = pi_window`` (a frame belongs to exactly one window). The joint inclusion
probability of two distinct windows is ``n (n - 1) / (N (N - 1))``.

QA cards (excluded from every estimate, ``panel_score``): ``QA_REPEAT_FRAC`` of the panel frames are repeats
of panel frames and ``QA_GOLD_FRAC`` are existing tier-1 (``labeller``) GT frames of the match. QA cards live
on an alias clip ``<clip>__b`` (a symlink to the same video) so they get their own sidecar: a repeat is an
independent second placement, never an overwrite of the first. They go last in the job, shuffled.

Storage. ``<FD>/ball_eval_panel/<panel_version>/<match>.json``, written atomically, append-only: building
again reproduces the same file (a no-op); a build that would change an existing file is refused unless
``force`` (the old file is kept as ``<match>.json.prev-<utc stamp>``).

Coordinates. Sidecar marks are normalised (0..1, top-left origin). The panel records the clip frame size
(``width`` x ``height``, probed from the clip) and the estimator scores in **pixels of that clip frame**.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import random
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from footy_track import match_splits, splits
from footy_track.clip_id import canonical_clip_id
from footy_track.labeller.constants import BALL_LABELS

PANEL_VERSION = "ball-eval-v1"
SCHEMA = "ball-eval-panel/v1"
STRIDE = 5  # population grid: every 5th source frame
WINDOW_FRAMES = 5
WINDOW_GAP_S = 0.2  # target spacing of the frames in a window
N_WINDOWS_DEFAULT = 24
QA_REPEAT_FRAC = 0.05
QA_GOLD_FRAC = 0.03
QA_SUFFIX = (
    "__b"  # alias clip for QA cards (neutral on purpose: the labeller sees clip names)
)
PANEL_DIRNAME = "ball_eval_panel"
SEGMENTS_GLOB = "split_video_broadcast_frames*/segments.json"
JOB_KEY_FMT = "ball-eval-panel/{panel_version}/{match}"


class PanelError(RuntimeError):
    """The panel cannot be built / used (message says why)."""


# --------------------------------------------------------------------------------------------
# Paths and identifiers
# --------------------------------------------------------------------------------------------
def panel_root(root: Path | None = None) -> Path:
    return Path(root) if root else splits.data_root()


def panel_dir(version: str = PANEL_VERSION, root: Path | None = None) -> Path:
    return panel_root(root) / PANEL_DIRNAME / version


def panel_path(
    match: str, version: str = PANEL_VERSION, root: Path | None = None
) -> Path:
    return panel_dir(version, root) / f"{match}.json"


def dataset_name(version: str = PANEL_VERSION) -> str:
    """Labeller dataset of a panel version (``ball-eval-v1`` -> ``ball_eval_v1``)."""
    return version.replace("-", "_")


def marks_dir(version: str = PANEL_VERSION, root: Path | None = None) -> Path:
    """Where the panel's blind placements land (its own sidecars: nothing prefilled, nothing shared with training GT)."""
    return panel_root(root) / f"ball_gt_marks_{dataset_name(version)}"


def clips_dir(version: str = PANEL_VERSION, root: Path | None = None) -> Path:
    return panel_dir(version, root) / "clips"


def job_key(match: str, version: str = PANEL_VERSION) -> str:
    return JOB_KEY_FMT.format(panel_version=version, match=match)


def seed_for(panel_version: str, match: str) -> int:
    """``sha1(panel_version + match)`` as an int (first 60 bits, so it is a plain Python/JSON integer)."""
    return int(hashlib.sha1((panel_version + match).encode()).hexdigest()[:15], 16)  # noqa: S324 (not security)


def qa_clip(clip: str) -> str:
    return clip + QA_SUFFIX


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def code_sha() -> str:
    """Short git sha of the checkout this module runs from (``FOOTY_CODE_SHA`` overrides), else ``unknown``."""
    if os.environ.get("FOOTY_CODE_SHA"):
        return os.environ["FOOTY_CODE_SHA"]
    try:
        r = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return r.stdout.strip() or "unknown" if r.returncode == 0 else "unknown"


# --------------------------------------------------------------------------------------------
# Heldout gate and segments
# --------------------------------------------------------------------------------------------
def require_heldout(match: str) -> None:
    """Refuse anything that is not a registered heldout match (fail closed: an unregistered match is a
    train-pool match, a missing register raises ``RegisterError``)."""
    split = splits.split_of(match)
    if split != splits.EVAL_HELDOUT:
        raise PanelError(
            f"{match!r} is not a heldout match (split_of -> {split!r}); the evaluation panel only samples "
            "heldout matches"
        )


def segments_path(match: str, root: Path | None = None) -> Path | None:
    """``<FD>/<match>/split_video_broadcast_frames*/segments.json`` (the plain directory wins over ``_kf10``)."""
    found = sorted((panel_root(root) / match).glob(SEGMENTS_GLOB))
    if not found:
        return None
    plain = [p for p in found if p.parent.name == "split_video_broadcast_frames"]
    return (plain or found)[0]


def load_segments(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or not doc.get("segments") or not doc.get("fps"):
        raise PanelError(f"{path}: not a segments.json (needs fps and segments)")
    return doc


def segmented_heldout_matches(root: Path | None = None) -> list[str]:
    """Registered heldout matches with a non-empty ``segments.json`` (sorted)."""
    out = []
    for key, entry in sorted(match_splits.load_register().items()):
        if entry.get("split") != match_splits.HELDOUT:
            continue
        p = segments_path(key, root)
        if p is None:
            continue
        try:
            load_segments(p)
        except (OSError, ValueError, PanelError):
            continue
        out.append(key)
    return out


# --------------------------------------------------------------------------------------------
# Population: disjoint windows inside segments
# --------------------------------------------------------------------------------------------
def window_gap(fps: float) -> int:
    """Source frames between the frames of a window: a multiple of ``STRIDE`` giving ~``WINDOW_GAP_S``."""
    return STRIDE * max(1, round(WINDOW_GAP_S * fps / STRIDE))


@dataclass(frozen=True)
class Window:
    window_id: str
    clip: str
    segment: int
    index: int  # global order in the population
    frames: tuple[int, ...]  # local to the clip

    @property
    def step(self) -> int:
        return self.frames[1] - self.frames[0]


def enumerate_windows(
    match: str, doc: dict, frame_counts: dict[str, int] | None = None
) -> tuple[list[Window], dict]:
    """Every window of the population, in segment then time order, plus ``info`` (gap, per-segment counts).

    ``frame_counts`` (clip stem -> real frame count of the playable clip) caps a segment's ``frame_count``
    when the clip is shorter than ``segments.json`` says. The last frame of a clip is never used (1-frame
    safety margin: container frame counts are off by one for some 60 fps clips).
    """
    gap = window_gap(float(doc["fps"]))
    tile = WINDOW_FRAMES * gap
    windows: list[Window] = []
    segs = []
    for seg in sorted(doc["segments"], key=lambda s: int(s["index"])):
        clip = Path(seg["file"]).stem
        n = int(seg["frame_count"])
        if frame_counts and clip in frame_counts:
            n = min(n, int(frame_counts[clip]))
        usable = n - 1  # local frames 0..usable-1
        k = 0
        count = 0
        while k * tile + (WINDOW_FRAMES - 1) * gap <= usable - 1:
            t0 = k * tile
            frames = tuple(t0 + j * gap for j in range(WINDOW_FRAMES))
            windows.append(
                Window(
                    f"{match}:w{len(windows):05d}",
                    clip,
                    int(seg["index"]),
                    len(windows),
                    frames,
                )
            )
            k += 1
            count += 1
        segs.append(
            {
                "clip": clip,
                "index": int(seg["index"]),
                "frame_count": n,
                "n_windows": count,
            }
        )
    return windows, {"gap": gap, "tile_frames": tile, "segments": segs}


def draw_windows(windows: list[Window], n: int, seed: int) -> list[Window]:
    """``n`` windows uniformly without replacement (all of them when the population is smaller), in population order."""
    n = min(n, len(windows))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(windows)), n))
    return [windows[i] for i in chosen]


# --------------------------------------------------------------------------------------------
# Tier-1 gold from existing GT
# --------------------------------------------------------------------------------------------
def gold_candidates(match_clips: set[str], marks_dirs: list[Path]) -> list[dict]:
    """Tier-1 ball positions of this match's clips: ``[{clip, frame, x, y, source}]`` (normalised centres).

    A row counts when its tags contain ``labeller`` and a ball class and it has a centre. Sidecars are matched
    to the match's clips by canonical clip id; the first marks dir that has a frame wins.
    """
    by_canon = {canonical_clip_id(c): c for c in match_clips}
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for d in marks_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jsonl")):
            clip = by_canon.get(canonical_clip_id(p.stem))
            if clip is None:
                continue
            for raw in p.read_text().splitlines():
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                    tags = row.get("tags") or []
                    c = row.get("center") or {}
                    f = int(row["frame_index"])
                    x, y = float(c["x"]), float(c["y"])
                except (ValueError, KeyError, TypeError):
                    continue
                if (
                    "labeller" not in tags
                    or not BALL_LABELS & set(tags)
                    or (clip, f) in seen
                ):
                    continue
                seen.add((clip, f))
                out.append(
                    {
                        "clip": clip,
                        "frame": f,
                        "x": x,
                        "y": y,
                        "source": f"{d.name}/{p.name}",
                    }
                )
    return sorted(out, key=lambda g: (g["clip"], g["frame"]))


def registry_marks_dirs(
    root: Path | None = None, exclude: Path | None = None
) -> list[Path]:
    """Marks dirs of the labeller registry (``labeller_datasets.json``), minus ``exclude`` (the panel's own)."""
    reg = Path(
        os.environ.get("FOOTY_LABELLER_DATASETS")
        or panel_root(root) / "labeller_datasets.json"
    )
    try:
        specs = json.loads(reg.read_text()).get("datasets", {}).values()
    except (OSError, ValueError, AttributeError):
        return []
    out = []
    for s in specs:
        m = s.get("marks_dir")
        if m and Path(m) != exclude and Path(m) not in out:
            out.append(Path(m))
    return out


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------
def cv2_probe(path: Path) -> dict | None:
    """``{frames, width, height}`` of a video from its header, ``None`` when it cannot be opened."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        return {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def _round_count(frac: float, n: int) -> int:
    return max(1, round(frac * n))


def build_panel(
    match: str,
    *,
    panel_version: str = PANEL_VERSION,
    n_windows: int = N_WINDOWS_DEFAULT,
    root: Path | None = None,
    gold_dirs: list[Path] | None = None,
    probe: Callable[[Path], dict | None] | None = cv2_probe,
    sha: str | None = None,
    created_at: str | None = None,
) -> dict:
    """The panel document of one heldout match (pure: writes nothing; see :func:`write_panel`)."""
    require_heldout(match)
    sp = segments_path(match, root)
    if sp is None:
        raise PanelError(
            f"{match}: no segments.json under {panel_root(root) / match}/{SEGMENTS_GLOB}"
        )
    doc = load_segments(sp)
    clip_dir = sp.parent

    frame_counts: dict[str, int] | None = None
    size: dict | None = None
    if probe is not None:
        frame_counts = {}
        for seg in doc["segments"]:
            clip = Path(seg["file"]).stem
            info = probe(clip_dir / seg["file"])
            if info is None:
                frame_counts[clip] = 0  # unplayable clip: drops out of the population
                continue
            frame_counts[clip] = info["frames"]
            size = size or {"width": info["width"], "height": info["height"]}
        if size is None:
            raise PanelError(
                f"{match}: none of the segment clips can be opened under {clip_dir}"
            )

    windows, info = enumerate_windows(match, doc, frame_counts)
    if not windows:
        raise PanelError(
            f"{match}: no complete {WINDOW_FRAMES}-frame window fits in any segment"
        )
    seed = seed_for(panel_version, match)
    chosen = draw_windows(windows, n_windows, seed)
    n_pop, n = len(windows), len(chosen)
    pi = n / n_pop

    win_docs = [
        {
            "window_id": w.window_id,
            "clip": w.clip,
            "segment": w.segment,
            "frame_start": w.frames[0],
            "frame_end": w.frames[-1],
            "step": w.step,
            "frames": list(w.frames),
            "pi_window": pi,
            "pi_frame": pi,
        }
        for w in chosen
    ]
    panel_frames = [(w.clip, f, w.window_id) for w in chosen for f in w.frames]

    # QA: repeats of panel frames + gold from existing tier-1 GT (alias clips, own sidecars)
    n_rep = min(len(panel_frames), _round_count(QA_REPEAT_FRAC, len(panel_frames)))
    rep_rng = random.Random(seed + 1)
    reps = sorted(rep_rng.sample(range(len(panel_frames)), n_rep))
    qa: list[dict] = [
        {
            "qa": "repeat",
            "clip": qa_clip(panel_frames[i][0]),
            "frame": panel_frames[i][1],
            "of": {
                "clip": panel_frames[i][0],
                "frame": panel_frames[i][1],
                "window_id": panel_frames[i][2],
            },
        }
        for i in reps
    ]
    used = {(q["clip"], q["frame"]) for q in qa}
    gdirs = (
        gold_dirs
        if gold_dirs is not None
        else registry_marks_dirs(root, exclude=marks_dir(panel_version, root))
    )
    seg_clips = {
        s["clip"]
        for s in info["segments"]
        if frame_counts is None or frame_counts.get(s["clip"], 1) > 0
    }
    frame_limit = {s["clip"]: s["frame_count"] - 1 for s in info["segments"]}
    cands = [
        g
        for g in gold_candidates(seg_clips, gdirs)
        if g["frame"] < frame_limit.get(g["clip"], 0)
        and (qa_clip(g["clip"]), g["frame"]) not in used
    ]
    n_gold = (
        min(len(cands), _round_count(QA_GOLD_FRAC, len(panel_frames))) if cands else 0
    )
    gold_rng = random.Random(seed + 2)
    for i in sorted(gold_rng.sample(range(len(cands)), n_gold)):
        g = cands[i]
        qa.append(
            {
                "qa": "gold",
                "clip": qa_clip(g["clip"]),
                "frame": g["frame"],
                "of": {"clip": g["clip"], "frame": g["frame"]},
                "gold": {
                    "x": g["x"],
                    "y": g["y"],
                    "unit": "normalised",
                    "source": g["source"],
                },
            }
        )
    random.Random(seed + 3).shuffle(qa)  # shuffled to the end of the job

    return {
        "schema": SCHEMA,
        "panel_version": panel_version,
        "match": match,
        "split": "heldout",
        "stratum": match,
        "design": {
            "unit": f"window of {WINDOW_FRAMES} frames {info['gap']} source frames apart (stride {STRIDE} grid)",
            "sampling": "uniform without replacement over disjoint windows inside segments",
            "frame_gap": info["gap"],
            "tile_frames": info["tile_frames"],
            "joint_pi_window": (n * (n - 1)) / (n_pop * (n_pop - 1))
            if n_pop > 1
            else 1.0,
        },
        "seed": seed,
        "seed_basis": "sha1(panel_version + match)",
        "code_sha": sha or code_sha(),
        "created_at": created_at or now_iso(),
        "segments_json": str(sp),
        "fps": float(doc["fps"]),
        "width": (size or {}).get("width"),
        "height": (size or {}).get("height"),
        "coordinates": "marks are normalised 0..1 (top-left origin); scored in pixels of the clip frame width x height",
        "N_windows": n_pop,
        "N_frames": n_pop * WINDOW_FRAMES,
        "n_windows": n,
        "n_frames": len(panel_frames),
        "pi_window": pi,
        "pi_frame": pi,
        "segments": info["segments"],
        "windows": win_docs,
        "qa": qa,
        "n_qa_repeat": n_rep,
        "n_qa_gold": n_gold,
    }


# --------------------------------------------------------------------------------------------
# Storage (append-only per match)
# --------------------------------------------------------------------------------------------
_VOLATILE = ("created_at", "code_sha")


def core(doc: dict) -> dict:
    """The deterministic part of a panel document (what 'same panel' means; stamps excluded)."""
    return {k: v for k, v in doc.items() if k not in _VOLATILE}


def read_panel(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise PanelError(f"{path}: not a {SCHEMA} document")
    return doc


def _atomic_write(
    path: Path, doc: dict, *, exclusive: bool, sort_keys: bool = True
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=sort_keys)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o664)
        if exclusive:
            try:
                os.link(
                    tmp, path
                )  # never replaces: two concurrent builds cannot both create it
            except FileExistsError:
                os.unlink(tmp)
                return False
            os.unlink(tmp)
            return True
        os.replace(tmp, path)
        return True
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_panel(
    doc: dict, *, root: Path | None = None, force: bool = False
) -> tuple[Path, str]:
    """Store a panel. Returns ``(path, status)``, status ``created`` | ``unchanged`` | ``replaced``.

    An existing file is never changed silently: an identical build is ``unchanged`` (idempotent), a different one
    raises :class:`PanelError` unless ``force`` (then the old file is kept as ``<match>.json.prev-<stamp>``).
    """
    path = panel_path(doc["match"], doc["panel_version"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if _atomic_write(path, doc, exclusive=True):
                return path, "created"
            old = read_panel(path)
            if core(old) == core(doc):
                return path, "unchanged"
            if not force:
                diff = sorted(k for k in core(doc) if core(doc)[k] != core(old).get(k))
                raise PanelError(
                    f"{path} exists and the new build differs ({', '.join(diff)}); {doc['panel_version']} is "
                    "append-only per match. Pass --force to replace it (the old file is kept)."
                )
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            os.replace(path, path.with_name(f"{path.name}.prev-{stamp}"))
            _atomic_write(path, doc, exclusive=True)
            return path, "replaced"
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def list_panels(
    version: str = PANEL_VERSION, root: Path | None = None
) -> dict[str, Path]:
    d = panel_dir(version, root)
    return {p.stem: p for p in sorted(d.glob("*.json"))} if d.is_dir() else {}


# --------------------------------------------------------------------------------------------
# Labelling job
# --------------------------------------------------------------------------------------------
def _sampling(doc: dict, **extra) -> dict:
    return {
        "regime": "eval",
        "panel": doc["panel_version"],
        "match": doc["match"],
        "stratum": doc["stratum"],
        "seed": doc["seed"],
        "code_sha": doc["code_sha"],
        **extra,
    }


def job_items(doc: dict) -> list[dict]:
    """Blind ``ball_place`` items: panel windows in time order, then the QA cards (shuffled).

    Items carry ``sampling`` (see ``docs/design/ball_sampling.md`` 2.3) and nothing else a labeller could read
    as a hint: no model box, no candidate, no score, and gold positions stay in the panel file.
    """
    items = [
        {
            "clip": w["clip"],
            "frame_start": w["frame_start"],
            "frame_end": w["frame_end"],
            "step": w["step"],
            "stratum": doc["match"],
            "sampling": _sampling(
                doc,
                window_id=w["window_id"],
                pi_window=w["pi_window"],
                pi_frame=w["pi_frame"],
                qa=None,
            ),
        }
        for w in doc["windows"]
    ]
    for q in doc["qa"]:
        items.append(
            {
                "clip": q["clip"],
                "frame": q["frame"],
                "stratum": doc["match"],
                "sampling": _sampling(
                    doc,
                    window_id=q["of"].get("window_id"),
                    pi_window=None,
                    pi_frame=None,
                    qa=q["qa"],
                    of=q["of"],
                ),
            }
        )
    return items


def job_body(doc: dict) -> dict:
    n = doc["n_frames"] + len(doc["qa"])
    return {
        "title": f"Ball eval panel {doc['panel_version']}: {doc['match']}",
        "kind": "ball_place",
        "dataset": dataset_name(doc["panel_version"]),
        "items": job_items(doc),
        "why": (
            f"Blind ball placement on {n} frames ({doc['n_windows']} random 1 s windows, {len(doc['qa'])} QA "
            "repeats): tap the ball centre, N = no ball, B = not main camera. No model boxes are shown on purpose."
        ),
        "device": "laptop",
        "priority": 2,
        "key": job_key(doc["match"], doc["panel_version"]),
    }


def _link(src: Path, dst: Path) -> bool:
    if dst.is_symlink() and os.readlink(dst) == str(src):
        return False
    if dst.exists() or dst.is_symlink():
        raise PanelError(f"{dst} exists and is not a link to {src}")
    dst.symlink_to(src)
    return True


def link_clips(doc: dict, root: Path | None = None) -> int:
    """Make the panel's clips (and their QA aliases) resolvable in the dataset's clips dir (symlinks)."""
    src_dir = Path(doc["segments_json"]).parent
    dst_dir = clips_dir(doc["panel_version"], root)
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    clips = {w["clip"] for w in doc["windows"]}
    for clip in sorted(clips):
        n += _link(src_dir / f"{clip}.mp4", dst_dir / f"{clip}.mp4")
    for q in doc["qa"]:
        base = q["of"]["clip"]
        n += _link(src_dir / f"{base}.mp4", dst_dir / f"{q['clip']}.mp4")
    return n


def register_dataset(version: str = PANEL_VERSION, root: Path | None = None) -> bool:
    """Add the panel's labeller dataset to ``labeller_datasets.json`` if missing (the site re-reads it on mtime).

    ``ingest: false`` keeps the panel's sidecars out of the feature store (``ingest_all``): they are evaluation
    labels with QA alias clips, not training/tracking GT. Returns True when the registry changed.
    """
    reg = Path(
        os.environ.get("FOOTY_LABELLER_DATASETS")
        or panel_root(root) / "labeller_datasets.json"
    )
    name = dataset_name(version)
    with open(reg.with_name(reg.name + ".lock"), "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            doc = (
                json.loads(reg.read_text())
                if reg.exists()
                else {"default": "calib", "datasets": {}}
            )
            if name in doc.setdefault("datasets", {}):
                return False
            doc["datasets"][name] = {
                "label": f"Ball eval panel {version} (blind placement, heldout matches)",
                "clips_dir": str(clips_dir(version, root)),
                "marks_dir": str(marks_dir(version, root)),
                "ingest": False,
            }
            marks_dir(version, root).mkdir(parents=True, exist_ok=True)
            _atomic_write(reg, doc, exclusive=False, sort_keys=False)
            return True
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
