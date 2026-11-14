"""Ball ground truth, dense runs and anchor selection for the SAM2 bidirectional experiment.

Pure stdlib (runs on the CPU-only mini and on the desktop).

GT rows (``ball_gt_marks/<clip>.jsonl``): ``frame_index``, normalised ``bbox``/``center``,
``tags = [class, provenance(, human_checked)]``.

* ball      class ``in_play_ball`` / ``out_of_play_ball`` and human provenance: tag
            ``labeller`` or ``human_checked``. Machine-only provenance (``yolo``, bare
            ``vittrack``) is dropped.
* no_ball   explicit ``no_ball`` rows (the labeller asserts there is no ball in the frame)
* not_broadcast  explicit ``not_broadcast`` rows (cut-away / graphic; informational only)

A *run* is a maximal block of consecutive frames that all carry a human ball mark.
``no_ball`` / ``not_broadcast`` frames directly adjacent to a run are attached to it as
``aux`` frames (scored for false positives, never used as anchors).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

W, H = 1920, 1080   # default frame size (arsenal_example); other clips pass their own
BALL_CLASSES = ("in_play_ball", "out_of_play_ball")


@dataclass
class Run:
    clip: str
    start: int
    end: int                      # inclusive; every frame in [start, end] has a ball mark
    boxes: dict[int, tuple[float, float, float, float]]   # frame -> (x0, y0, x1, y1) px in the video's native size
    aux: dict[int, str] = field(default_factory=dict)      # adjacent no_ball / not_broadcast frames

    @property
    def n_ball(self) -> int:
        return self.end - self.start + 1

    @property
    def window(self) -> tuple[int, int]:
        fs = [self.start, self.end, *self.aux]
        return min(fs), max(fs)

    def centre(self, f: int) -> tuple[float, float]:
        x0, y0, x1, y1 = self.boxes[f]
        return (x0 + x1) / 2, (y0 + y1) / 2


def _is_human(tags: list[str]) -> bool:
    return "labeller" in tags or "human_checked" in tags


def load_gt(path: str | Path, w: int = W, h: int = H) -> tuple[dict, dict]:
    """Return (ball_boxes: frame -> box px, other: frame -> 'no_ball' | 'not_broadcast')."""
    ball: dict[int, tuple[float, tuple]] = {}
    other: dict[int, str] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tags, f = r["tags"], r["frame_index"]
        if tags[0] in ("no_ball", "not_broadcast"):
            other[f] = tags[0]
        elif tags[0] in BALL_CLASSES and _is_human(tags) and r.get("bbox"):
            b = r["bbox"]
            box = (b["x"] * w, b["y"] * h, (b["x"] + b["w"]) * w, (b["y"] + b["h"]) * h)
            # prefer in_play over out_of_play, then the first mark
            rank = 0 if tags[0] == "in_play_ball" else 1
            if f not in ball or rank < ball[f][0]:
                ball[f] = (rank, box)
    return {f: v[1] for f, v in ball.items()}, other


def find_runs(clip: str, path: str | Path, min_len: int = 9, w: int = W, h: int = H) -> list[Run]:
    ball, other = load_gt(path, w, h)
    frames = sorted(ball)
    runs: list[Run] = []
    i = 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and frames[j + 1] == frames[j] + 1:
            j += 1
        s, e = frames[i], frames[j]
        if e - s + 1 >= min_len:
            r = Run(clip, s, e, {f: ball[f] for f in range(s, e + 1)})
            f = s - 1
            while f in other:
                r.aux[f] = other[f]
                f -= 1
            f = e + 1
            while f in other:
                r.aux[f] = other[f]
                f += 1
            runs.append(r)
        i = j + 1
    return runs


def anchors(run: Run, n: int) -> list[int]:
    """Anchor every ``n`` labelled frames from the run start, plus the last frame."""
    a = set(range(run.start, run.end + 1, n))
    a.add(run.end)
    return sorted(a)


def reach_for(n: int) -> int:
    """How far from an anchor tracks are propagated (frames, each direction)."""
    return max(n, 25)
