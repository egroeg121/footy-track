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
import logging
import os
import random
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
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

# Keyframe-dense render of the panel's clips (``render_clips``). Same encoder settings as
# ``footy_track.scripts.reencode_keyframes`` (kf10: GOP 10, crf 16, ultrafast, yuv420p, faststart, no audio).
RENDER_SCHEMA = "ball-eval-clips-render/v1"
RENDER_MANIFEST = ".render_manifest.json"  # inside the clips dir; not a video suffix, so the labeller never lists it
RENDER_TMP_DIRNAME = (
    ".render_tmp"  # sibling of clips/ (same filesystem: the swap is an atomic rename)
)
KF_GOP = 10
KF_CRF = 16
KF_PRESET = "ultrafast"
KF_THREADS = 4
RENDER_TIMEOUT_S = 1800

log = logging.getLogger(__name__)


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
    if dst.is_file() and not dst.is_symlink():
        return False  # a keyframe-dense render of the clip (``render_clips``): keep it
    if (
        dst.is_symlink()
        and "/" not in os.readlink(dst)
        and (dst.parent / os.readlink(dst)).is_file()
    ):
        return False  # alias of a rendered sibling
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


# --------------------------------------------------------------------------------------------
# Keyframe-dense render of the clips
# --------------------------------------------------------------------------------------------
# The segment mp4s the panel samples from have a ~250-frame GOP, so every seek in the labeller decodes up to 250
# frames. ``render_clips`` swaps the clips dir's symlinks for re-encodes with a keyframe every ``KF_GOP`` frames.
# Frames map 1:1 (``-fps_mode passthrough``, no cutting, no audio) and each output is verified to have the same
# frame count / size / fps as its source before it replaces anything, so frame indices in the panel file, job
# items and sidecar marks stay valid. A failed render keeps the symlink (slow but correct); it never raises.
def probe_stream(path: Path) -> dict:
    """``{frames, width, height, fps}`` of the first video stream, frames COUNTED (decoded, ``-count_frames``)."""
    out = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True, timeout=RENDER_TIMEOUT_S,
    ).stdout  # fmt: skip
    st = json.loads(out)["streams"][0]
    rate = (
        Fraction(st["avg_frame_rate"])
        if st.get("avg_frame_rate") not in (None, "0/0")
        else Fraction(0)
    )
    return {
        "frames": int(st["nb_read_frames"]),
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps": f"{rate.numerator}/{rate.denominator}",
    }


def _ffmpeg_kf_cmd(
    src: Path, tmp: Path, gop: int, crf: int, preset: str, threads: int
) -> list[str]:
    return [
        "nice", "-n", "10", "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src),
        "-map", "0:v:0", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-pix_fmt", "yuv420p", "-an", "-threads", str(threads),
        "-movflags", "+faststart", str(tmp),
    ]  # fmt: skip


def render_clip(
    src: Path,
    dst: Path,
    tmp_dir: Path,
    *,
    gop: int = KF_GOP,
    crf: int = KF_CRF,
    preset: str = KF_PRESET,
    threads: int = KF_THREADS,
    src_info: dict | None = None,
) -> dict:
    """Re-encode ``src`` with a keyframe every ``gop`` frames into ``dst`` (atomic rename over a symlink or file).

    The output is written to ``tmp_dir`` and verified against the source (frame count, size, fps) first. Returns the
    manifest entry (``frames``, ``width``, ``height``, ``fps``, ``out_size``). Raises ``PanelError`` on any failure,
    leaving ``dst`` untouched.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{dst.name}.{os.getpid()}.tmp.mp4"
    try:
        try:
            info = src_info or probe_stream(src)
            res = subprocess.run(  # noqa: S603
                _ffmpeg_kf_cmd(src, tmp, gop, crf, preset, threads),
                capture_output=True, text=True, check=False, timeout=RENDER_TIMEOUT_S,
            )  # fmt: skip
            if res.returncode != 0:
                raise PanelError(f"ffmpeg failed: {res.stderr.strip()[-300:]}")
            out = probe_stream(tmp)
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
            KeyError,
            IndexError,
        ) as e:
            raise PanelError(f"{src.name}: {e!r}"[:400]) from e
        if out != info:
            raise PanelError(
                f"{src.name}: re-encode differs from the source (source {info}, output {out}); kept source"
            )
        os.chmod(tmp, 0o664)
        out["out_size"] = tmp.stat().st_size
        os.replace(
            tmp, dst
        )  # atomic: a reader holds either the old file or the new one
        return out
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _read_render_manifest(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text())
        if isinstance(doc, dict) and doc.get("schema") == RENDER_SCHEMA:
            return doc
    except (OSError, ValueError):
        pass
    return {"schema": RENDER_SCHEMA, "clips": {}}


def _render_groups(doc: dict) -> dict[str, dict]:
    """``{source clip: {"files": [clip file names wanted], ...}}``: the base clip when a window uses it, then its QA aliases."""
    window_clips = {w["clip"] for w in doc["windows"]}
    groups: dict[str, list[str]] = {}
    for base in sorted(window_clips | {q["of"]["clip"] for q in doc["qa"]}):
        names = [base] if base in window_clips else []
        names += sorted({q["clip"] for q in doc["qa"] if q["of"]["clip"] == base})
        groups[base] = names
    return groups


def _symlink_atomic(target_name: str, dst: Path) -> None:
    """Make ``dst`` a symlink to the sibling ``target_name`` (relative), replacing whatever is there atomically."""
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.lnk")
    with contextlib.suppress(OSError):
        tmp.unlink()
    tmp.symlink_to(target_name)
    try:
        os.replace(tmp, dst)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _link_aliases(cdir: Path, primary: str, others: list[str]) -> list[str]:
    """Point each alias at the rendered ``primary`` (relative symlink, atomic). Returns the aliases changed."""
    done = []
    for o in others:
        odst = cdir / f"{o}.mp4"
        if odst.is_symlink() and os.readlink(odst) == f"{primary}.mp4":
            continue
        try:
            _symlink_atomic(f"{primary}.mp4", odst)
            done.append(o)
        except OSError as e:
            log.warning("render_clips: alias %s not relinked (%s)", o, e)
    return done


def render_clips(
    doc: dict,
    root: Path | None = None,
    *,
    gop: int = KF_GOP,
    crf: int = KF_CRF,
    preset: str = KF_PRESET,
    threads: int = KF_THREADS,
    force: bool = False,
    render: Callable[..., dict] | None = None,
) -> dict:
    """Replace the symlinks of the panel's clips in its clips dir by keyframe-dense re-encodes (same names).

    For each source clip of the panel (windows and QA ``of`` clips) one file is encoded (``<clip>.mp4`` when a window
    uses it, else its first ``__b`` alias) and every other name of that clip (the ``__b`` aliases) becomes a relative
    symlink to it. Idempotent: a real file that is already rendered (manifest entry matching the source's size and
    mtime, or else equal frame count / size / fps to the source) is skipped. A clip that fails to render keeps its
    symlink and is logged as a warning: this function never raises for a bad clip. One render at a time per panel
    version (flock). Writes ``<clips>/.render_manifest.json`` (settings, source size, frames).

    Returns ``{"rendered", "skipped", "failed", "aliased", "bytes"}`` (lists of clip names, bytes written).
    """
    render = render or render_clip
    src_dir = Path(doc["segments_json"]).parent
    cdir = clips_dir(doc["panel_version"], root)
    cdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "rendered": [],
        "skipped": [],
        "failed": [],
        "aliased": [],
        "bytes": 0,
    }
    settings = {"codec": "libx264", "gop": gop, "keyint_min": gop, "crf": crf, "preset": preset, "pix_fmt": "yuv420p",
                "sc_threshold": 0, "audio": False, "faststart": True, "fps_mode": "passthrough"}  # fmt: skip
    lock_path = panel_dir(doc["panel_version"], root) / ".render.lock"
    with open(lock_path, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            mpath = cdir / RENDER_MANIFEST
            manifest = _read_render_manifest(mpath)
            manifest["settings"] = settings
            for base, names in _render_groups(doc).items():
                if not names:
                    continue
                primary, others = names[0], names[1:]
                src, dst = src_dir / f"{base}.mp4", cdir / f"{primary}.mp4"
                try:
                    st = src.stat()
                    entry = manifest["clips"].get(primary + ".mp4") or {}
                    done = (
                        not force
                        and dst.is_file()
                        and not dst.is_symlink()
                        and entry.get("source_size") == st.st_size
                        and entry.get("source_mtime_ns") == st.st_mtime_ns
                        and entry.get("out_size") == dst.stat().st_size
                    )
                    info = None
                    if (
                        not done
                        and not force
                        and dst.is_file()
                        and not dst.is_symlink()
                    ):
                        info = probe_stream(src)
                        done = probe_stream(dst) == info
                        if done:
                            entry = {**info, "out_size": dst.stat().st_size}
                    if done:
                        summary["skipped"].append(primary)
                    else:
                        t = time.time()
                        entry = render(src, dst, cdir.parent / RENDER_TMP_DIRNAME, gop=gop, crf=crf, preset=preset,
                                       threads=threads, src_info=info)  # fmt: skip
                        entry["seconds"] = round(time.time() - t, 1)
                        summary["rendered"].append(primary)
                        summary["bytes"] += entry["out_size"]
                    manifest["clips"][primary + ".mp4"] = {
                        **entry,
                        "source": str(src),
                        "source_size": st.st_size,
                        "source_mtime_ns": st.st_mtime_ns,
                        "gop": gop,
                        "rendered_at": entry.get("rendered_at") or now_iso(),
                        "aliases": [f"{o}.mp4" for o in others],
                    }
                except (
                    PanelError,
                    OSError,
                    subprocess.SubprocessError,
                    ValueError,
                    KeyError,
                ) as e:
                    log.warning(
                        "render_clips: %s not re-encoded, keeping the symlink (%s)",
                        primary,
                        e,
                    )
                    summary["failed"].append(primary)
                    continue  # aliases keep pointing at the source too
                summary["aliased"] += _link_aliases(cdir, primary, others)
            _atomic_write(mpath, manifest, exclusive=False)
            with contextlib.suppress(OSError):
                (cdir.parent / RENDER_TMP_DIRNAME).rmdir()  # only succeeds when empty
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return summary
