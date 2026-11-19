"""Select windows of contiguous frames to auto-label (docs/design/full_frame_labelling.md section 4, stage 1).

A *window* is ``N`` (default 250, 5 s at 50 fps) consecutive frames of ONE ``_segNNN`` clip.
``_segNNN`` clips are cut from different parts of a match and are not contiguous, so a window
never spans two of them.

Rules (all enforced here, not left to callers):

* **Train split only.** A clip is eligible only when ``footy_track.splits.split_of`` says
  ``train``: heldout matches and in-match eval clips are excluded (rule zero of
  ``docs/design/ball_sampling.md``: a heldout match never feeds training, pseudo-labels
  included). A clip whose match is not in the register is excluded too. A missing/invalid
  split register raises ``RegisterError`` (fail closed, never "assume train").
* **Skip finished work.** Windows containing a frame that is already ``frame_complete`` in the
  store are dropped (``max_complete_frac`` = 0.0 by default).
* **Balanced across matches.** Windows are picked round-robin over matches (sorted by match
  key), so one match with many clips cannot dominate.
* **Ranked within a match** by (fewest existing reviewed labels in the window, lowest mean
  tracker agreement when supplied, clip id, start frame). Everything is sorted, nothing random:
  the same inputs give the same ``windows.json`` byte for byte.

Windows are aligned to multiples of ``N`` inside the clip (non-overlapping); a trailing partial
window shorter than ``N`` is dropped.

``windows.json`` schema (``version`` 1)::

    {"version": 1,
     "window_frames": 250,
     "windows": [
        {"window_id": "<game_id>_f000500",   # unique, stable
         "game_id": "<clip stem>",           # feature-store game_id (the clip stem)
         "clip_id": "<canonical clip id>",
         "match_key": "<register key>",
         "video": "<path or null>",
         "start_frame": 500,                 # inclusive
         "end_frame": 750,                   # exclusive
         "labelled_frames": 0,               # frames in the window with a reviewed detection
         "agreement": null,                  # mean tracker agreement 0..1 if known
         "rank": 0}                          # 0-based selection order
     ]}

CLI::

    python -m footy_track.autolabel.windows --clips-dir DIR --out windows.json [--count 20]
        [--window-frames 250] [--db STORE.duckdb] [--agreement agreement.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from footy_track import splits
from footy_track.clip_id import canonical_clip_id

WINDOWS_VERSION = 1
DEFAULT_WINDOW_FRAMES = 250
_SEG_RE = re.compile(r".*_seg\d+$")
_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")


@dataclass(frozen=True)
class ClipInfo:
    """A clip the selector may draw from. ``game_id`` is the clip stem (the store's game id)."""

    game_id: str
    n_frames: int
    video: str | None = None


@dataclass(frozen=True)
class Window:
    window_id: str
    game_id: str
    clip_id: str
    match_key: str
    video: str | None
    start_frame: int
    end_frame: int
    labelled_frames: int = 0
    agreement: float | None = None
    rank: int = 0


def window_id_of(game_id: str, start_frame: int) -> str:
    return f"{game_id}_f{start_frame:06d}"


def is_segment_clip(clip: str) -> bool:
    """True for a ``..._segNNN`` clip (the only unit windows may be cut from)."""
    return bool(_SEG_RE.match(canonical_clip_id(clip)))


def eligible_match_key(game_id: str) -> str | None:
    """The match key when ``game_id`` is a train-split ``_segNNN`` clip, else ``None``.

    Raises ``RegisterError`` when a split register is missing/invalid (fail closed).
    """
    if not is_segment_clip(game_id):
        return None
    key = splits.match_key_of(game_id)
    if key is None:  # unregistered match: cannot prove it is train
        return None
    if splits.split_of(key, canonical_clip_id(game_id)) != splits.TRAIN:
        return None
    return key


def discover_clips(
    clips_dir: Path, frame_counter: Callable[[Path], int] | None = None
) -> list[ClipInfo]:
    """Video files in ``clips_dir`` (sorted). ``frame_counter`` defaults to OpenCV's count."""
    counter = frame_counter or _cv2_frame_count
    out = []
    for p in sorted(Path(clips_dir).iterdir()):
        if p.suffix.lower() in _VIDEO_SUFFIXES and is_segment_clip(p.stem):
            out.append(ClipInfo(p.stem, counter(p), str(p)))
    return out


def _cv2_frame_count(path: Path) -> int:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def complete_frames_from_store(store, game_id: str) -> set[int]:
    """Frames of ``game_id`` in the ``frame_complete`` view."""
    df = store.query(
        "SELECT frame_index FROM frame_complete WHERE game_id = ?", [game_id]
    )
    return {int(i) for i in df["frame_index"]}


def reviewed_frames_from_store(store, game_id: str) -> set[int]:
    """Frames of ``game_id`` that already carry a reviewed detection."""
    df = store.query(
        "SELECT DISTINCT frame_index FROM detection WHERE game_id = ? AND reviewed",
        [game_id],
    )
    return {int(i) for i in df["frame_index"]}


def select_windows(
    clips: Iterable[ClipInfo],
    *,
    count: int,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    complete_frames: Callable[[str], set[int]] | None = None,
    labelled_frames: Callable[[str], set[int]] | None = None,
    agreement: Mapping[str, Mapping[int, float]] | None = None,
    max_complete_frac: float = 0.0,
) -> list[Window]:
    """Pick up to ``count`` windows, balanced across matches. See the module docstring.

    ``complete_frames(game_id)`` / ``labelled_frames(game_id)`` return frame-index sets (empty
    when no store is available). ``agreement[game_id][frame]`` is tracker agreement in 0..1
    (lower = worse = more worth labelling).
    """
    if window_frames < 1:
        raise ValueError("window_frames must be >= 1")
    complete_frames = complete_frames or (lambda _g: set())
    labelled_frames = labelled_frames or (lambda _g: set())
    agreement = agreement or {}

    per_match: dict[str, list[tuple[tuple, Window]]] = defaultdict(list)
    for clip in sorted(clips, key=lambda c: c.game_id):
        key = eligible_match_key(clip.game_id)
        if key is None:
            continue
        done = complete_frames(clip.game_id)
        lab = labelled_frames(clip.game_id)
        ag = agreement.get(clip.game_id, {})
        for start in range(0, clip.n_frames - window_frames + 1, window_frames):
            end = start + window_frames
            span = range(start, end)
            if sum(1 for f in span if f in done) > max_complete_frac * window_frames:
                continue
            n_lab = sum(1 for f in span if f in lab)
            vals = [ag[f] for f in span if f in ag]
            mean_ag = sum(vals) / len(vals) if vals else None
            w = Window(
                window_id=window_id_of(clip.game_id, start),
                game_id=clip.game_id,
                clip_id=canonical_clip_id(clip.game_id),
                match_key=key,
                video=clip.video,
                start_frame=start,
                end_frame=end,
                labelled_frames=n_lab,
                agreement=mean_ag,
            )
            # unknown agreement sorts after known-low, before known-high
            ag_rank = 0.5 if mean_ag is None else mean_ag
            per_match[key].append(((n_lab, ag_rank, w.game_id, start), w))

    queues = {
        k: [w for _, w in sorted(v, key=lambda t: t[0])] for k, v in per_match.items()
    }
    picked: list[Window] = []
    while len(picked) < count and any(queues.values()):
        for key in sorted(queues):
            if queues[key] and len(picked) < count:
                picked.append(queues[key].pop(0))
    return [Window(**{**asdict(w), "rank": i}) for i, w in enumerate(picked)]


def windows_document(windows: list[Window], window_frames: int) -> dict:
    return {
        "version": WINDOWS_VERSION,
        "window_frames": window_frames,
        "windows": [asdict(w) for w in windows],
    }


def write_windows(path: Path, windows: list[Window], window_frames: int) -> None:
    path.write_text(
        json.dumps(windows_document(windows, window_frames), indent=1, sort_keys=True)
        + "\n"
    )


def read_windows(path: Path) -> list[Window]:
    doc = json.loads(Path(path).read_text())
    if doc.get("version") != WINDOWS_VERSION:
        raise ValueError(f"unsupported windows.json version {doc.get('version')!r}")
    return [Window(**w) for w in doc["windows"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    ap.add_argument(
        "--db", type=Path, help="feature store (skip complete frames, rank by labels)"
    )
    ap.add_argument(
        "--agreement", type=Path, help='JSON {"<game_id>": {"<frame>": 0..1}}'
    )
    args = ap.parse_args(argv)

    agreement = None
    if args.agreement:
        raw = json.loads(args.agreement.read_text())
        agreement = {
            g: {int(f): float(v) for f, v in d.items()} for g, d in raw.items()
        }

    clips = discover_clips(args.clips_dir)
    if args.db:
        from footy_track.feature_store.lock import store_lock  # noqa: PLC0415
        from footy_track.feature_store.store import FeatureStore  # noqa: PLC0415

        with (
            store_lock(args.db, exclusive=False),
            FeatureStore.open(args.db, create=False) as st,
        ):
            windows = select_windows(
                clips,
                count=args.count,
                window_frames=args.window_frames,
                complete_frames=lambda g: complete_frames_from_store(st, g),
                labelled_frames=lambda g: reviewed_frames_from_store(st, g),
                agreement=agreement,
            )
    else:
        windows = select_windows(
            clips,
            count=args.count,
            window_frames=args.window_frames,
            agreement=agreement,
        )
    write_windows(args.out, windows, args.window_frames)
    print(f"{len(windows)} windows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
