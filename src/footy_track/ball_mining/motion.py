"""Stage 2: the camera-compensated motion-blob source (a RECALL source, never a verdict on its own).

Ported from ``experiments/ball_candidate_pool/motion.py`` (branch ``exp/ball-candidate-pool``, commit 07930b1). For a target
frame ``t`` and half-gap ``k``: LK + RANSAC homographies (``footy_track.calibration.temporal`` through
``multiframe.align``; person boxes and the broadcast overlay masked) align ``t-k`` and ``t+k`` to ``t``;
``D = min(|I_t - W(t-k)|, |I_t - W(t+k)|)`` on Gaussian-blurred 1280x720 grey frames keeps only the pixels that differ from
BOTH neighbours (a moving object); threshold, 2x2 open, connected components with ball-like size / shape. Score = contrast x
fill x a size prior around 9 px x an isolation term. No colour or learned appearance. If one side fails alignment (cut,
grass only) the other side's single difference is used.

Chosen on train clips only (README of the experiment): ``k=3``, threshold 14, person boxes masked, top 20 blobs per frame.
Measured there: 0.09-0.15 s per target frame on 4 threads, 310 MB RSS; recall@10 px 0.641 alone, 0.916 together with
RT-DETR; but 0.2 % of single-source motion nodes are the real ball, so it only proposes candidates.

Streaming and stride: the target frames are the ones ``candidates`` sampled (every ``stride``-th); only ``t-k, t, t+k`` of each
are converted, the rest are ``grab()``-ed. Person boxes for ``t+-k`` are those of the nearest *sampled* frame (the detector
did not run on ``t+-k``): the experiment used the exact frame, a mild difference (boxes are padded 10 %).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from footy_track.ball_mining.candidates import (
    META,
    ClipCandidates,
    Segment,
    atomic_write_json,
    stage_done,
)

SCHEMA = "ball-motion/v1"
WW, HH = 1280, 720
#: columns of a blob row
COLS = (
    "x",
    "y",
    "score",
    "contrast",
    "area",
    "w",
    "h",
    "fill",
    "aspect",
    "dens",
    "in_person",
)  # px@1280


@dataclass(frozen=True)
class MParams:
    k: int = 3
    thr: float = 14.0  # grey levels on the blurred double difference
    blur: float = 1.0  # Gaussian sigma before differencing
    min_area: int = 14
    max_area: int = 260
    max_side: int = 34  # bbox max side (motion blur elongates a ~11 px ball)
    max_aspect: float = 3.2
    min_fill: float = 0.35  # area / bbox area
    mask_persons: bool = (
        True  # mask person boxes in the blob stage (the LK fit always masks them)
    )
    person_pad: float = 0.10
    topn: int = 20
    border: int = 6

    def replace(self, **kw) -> MParams:
        return dataclasses.replace(self, **kw)

    def key(self) -> str:
        import hashlib  # noqa: PLC0415

        return hashlib.sha1(
            json.dumps(dataclasses.asdict(self), sort_keys=True).encode()
        ).hexdigest()[:8]


_OVERLAY: np.ndarray | None = None


def overlay_valid() -> np.ndarray:
    """bool (720, 1280), True = usable (outside the scorebug / logo boxes)."""
    global _OVERLAY  # noqa: PLW0603
    if _OVERLAY is None:
        from footy_track.calibration import temporal as T  # noqa: PLC0415, N812

        _OVERLAY = T.overlay_mask((HH, WW))
    return _OVERLAY.copy()


def person_mask(persons: np.ndarray, pad: float) -> np.ndarray:
    """bool (H, W), True = masked. ``persons``: (n, 4) ``cx, cy, w, h`` px@1280."""
    m = np.zeros((HH, WW), np.uint8)
    for cx, cy, w, h in persons:
        x0, x1 = int(cx - w / 2 * (1 + pad)), int(np.ceil(cx + w / 2 * (1 + pad)))
        y0, y1 = int(cy - h / 2 * (1 + pad)), int(np.ceil(cy + h / 2 * (1 + pad)))
        m[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = 1
    return m.astype(bool)


def pair_h(
    a_gray: np.ndarray,
    b_gray: np.ndarray,
    persons_a: np.ndarray | None,
    persons_b: np.ndarray | None,
) -> np.ndarray | None:
    """Homography ``a -> b`` (px@1280) by LK + RANSAC with person boxes masked, or None when the fit is not ok."""
    from footy_track.multiframe.align import estimate_pair_lk  # noqa: PLC0415

    def norm(p):
        if p is None or len(p) == 0:
            return None
        return [(cx / WW, cy / HH, w / WW, h / HH) for cx, cy, w, h in p]

    pm = estimate_pair_lk(
        a_gray, b_gray, boxes_a=norm(persons_a), boxes_b=norm(persons_b)
    )
    return pm.H_px if pm.ok else None


def _warp(b: np.ndarray, h: np.ndarray) -> np.ndarray:
    import cv2  # noqa: PLC0415

    return cv2.warpPerspective(
        b,
        h,
        (WW, HH),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def diff_map(
    g_t: np.ndarray, g_n: np.ndarray, h: np.ndarray, blur: float
) -> tuple[np.ndarray, np.ndarray]:
    """``|I_t - warp(I_n)|`` on blurred images and an eroded 'inside the warped frame' validity mask."""
    import cv2  # noqa: PLC0415

    gt = (
        cv2.GaussianBlur(g_t, (0, 0), blur).astype(np.float32)
        if blur > 0
        else g_t.astype(np.float32)
    )
    gn = g_n if blur <= 0 else cv2.GaussianBlur(g_n, (0, 0), blur)
    w = _warp(gn, h).astype(np.float32)
    inside = _warp(np.full_like(g_n, 255), h) > 250
    inside = cv2.erode(inside.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return np.abs(gt - w), inside


def blobs_from_diffs(
    diffs: list[tuple[np.ndarray, np.ndarray]],
    p: MParams,
    persons: np.ndarray | None = None,
) -> np.ndarray:
    """Blob rows (:data:`COLS`) from one or two ``(D, inside)`` difference maps, best ``p.topn`` by score."""
    import cv2  # noqa: PLC0415

    d = diffs[0][0] if len(diffs) == 1 else np.minimum(diffs[0][0], diffs[1][0])
    ok = diffs[0][1] if len(diffs) == 1 else (diffs[0][1] & diffs[1][1])
    ok = ok & overlay_valid()
    b = p.border
    ok[:b] = False
    ok[-b:] = False
    ok[:, :b] = False
    ok[:, -b:] = False
    pm = (
        person_mask(persons, p.person_pad)
        if (persons is not None and len(persons))
        else None
    )
    if p.mask_persons and pm is not None:
        ok = ok & ~pm
    d = d * ok
    binm = (d > p.thr).astype(np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(binm, connectivity=8)
    # activity density: fraction of pixels above threshold in a 61x61 window (textured zones: crowd, hoardings, lines)
    dens = cv2.boxFilter(binm.astype(np.float32), -1, (61, 61), normalize=True)
    rows = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < p.min_area or a > p.max_area or max(w, h) > p.max_side:
            continue
        asp = max(w, h) / max(1, min(w, h))
        fill = a / float(w * h)
        if asp > p.max_aspect or fill < p.min_fill:
            continue
        m = lab[y : y + h, x : x + w] == i
        contrast = float(d[y : y + h, x : x + w][m].mean())
        cx, cy = cen[i]
        dd = float(dens[min(HH - 1, int(cy)), min(WW - 1, int(cx))])
        inp = (
            float(pm[min(HH - 1, int(cy)), min(WW - 1, int(cx))])
            if pm is not None
            else 0.0
        )
        size_prior = np.exp(-0.5 * ((np.sqrt(a) - 9.0) / 5.0) ** 2)
        score = (contrast / 255.0) * fill * size_prior * (1.0 - min(1.0, dd * 6.0))
        rows.append((cx, cy, score, contrast, a, w, h, fill, asp, dd, inp))
    if not rows:
        return np.zeros((0, len(COLS)), np.float32)
    r = np.array(rows, np.float32)
    return r[np.argsort(-r[:, 2])][: p.topn]


# ---------------------------------------------------------------------------------------------- per clip
@dataclass
class ClipMotion:
    frames: np.ndarray  # int32 [n] target frames
    ptr: np.ndarray  # int32 [n+1]
    rows: np.ndarray  # float32 [m, 11]  (COLS)
    status: np.ndarray  # int8 [n]: how many neighbour sides aligned (0, 1 or 2)
    params: dict

    def at(self, frame: int) -> np.ndarray | None:
        """Blob rows of ``frame``, or None when the frame was not processed (a frame processed with no blobs gives 0 rows)."""
        i = int(np.searchsorted(self.frames, frame))
        if i >= len(self.frames) or self.frames[i] != frame:
            return None
        return self.rows[self.ptr[i] : self.ptr[i + 1]]


def save_motion(m: ClipMotion, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez(
        tmp,
        frames=m.frames,
        ptr=m.ptr,
        rows=m.rows,
        status=m.status,
        params=json.dumps(m.params),
    )
    os.replace(tmp, path)


def load_motion(path: str | Path) -> ClipMotion:
    z = np.load(path)
    return ClipMotion(
        z["frames"], z["ptr"], z["rows"], z["status"], json.loads(str(z["params"]))
    )


class _Window:
    """Sequential decoder that converts only the wanted frames to 1280x720 grey and keeps the last few."""

    def __init__(self, path: str | Path, wanted: set[int], keep: int) -> None:
        import cv2  # noqa: PLC0415

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        self.wanted = wanted
        self.keep = keep
        self.pos = 0  # next frame index to grab
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.exhausted = False

    def advance_to(self, i: int) -> None:
        """Decode up to and including frame ``i``."""
        while self.pos <= i and not self.exhausted:
            if not self.cap.grab():
                self.exhausted = True
                return
            if self.pos in self.wanted:
                ok, img = self.cap.retrieve()
                if ok:
                    if img.shape[1] != WW or img.shape[0] != HH:
                        img = self.cv2.resize(
                            img, (WW, HH), interpolation=self.cv2.INTER_AREA
                        )
                    self.cache[self.pos] = self.cv2.cvtColor(
                        img, self.cv2.COLOR_BGR2GRAY
                    )
                    while len(self.cache) > self.keep:
                        self.cache.popitem(last=False)
            self.pos += 1

    def gray(self, i: int) -> np.ndarray | None:
        if i < 0:
            return None
        self.advance_to(i)
        return self.cache.get(i)

    def close(self) -> None:
        self.cap.release()


def motion_clip(
    video: str | Path, cand: ClipCandidates, p: MParams | None = None
) -> ClipMotion:
    """Motion blobs at every sampled frame of ``cand`` (the detector's frames), streaming ``video`` once."""
    from footy_track.ball_mining.candidates import limit_threads  # noqa: PLC0415

    limit_threads()
    p = p or MParams()
    targets = [int(f) for f in cand.frames]
    wanted = {g for t in targets for g in (t - p.k, t, t + p.k) if g >= 0}
    win = _Window(video, wanted, keep=2 * p.k + 4)
    frames, rows, status, ptr = [], [], [], [0]
    try:
        for t in targets:
            win.gray(
                t + p.k
            )  # decode ahead first, then read t (and t - k) from the window
            g = win.gray(t)
            if g is None:
                continue
            persons_t = cand.persons_px(cand.nearest_sampled(t) or t, WW, HH)
            sides = []
            for tn in (t - p.k, t + p.k):
                gn = win.gray(tn)
                if gn is None:
                    continue
                persons_n = cand.persons_px(cand.nearest_sampled(tn) or tn, WW, HH)
                h = pair_h(g, gn, persons_t, persons_n)
                if h is not None:
                    sides.append(diff_map(g, gn, h, p.blur))
            r = (
                blobs_from_diffs(sides, p, persons_t)
                if sides
                else np.zeros((0, len(COLS)), np.float32)
            )
            frames.append(t)
            status.append(len(sides))
            rows.append(r)
            ptr.append(ptr[-1] + len(r))
    finally:
        win.close()
    flat = (
        np.concatenate(rows).astype(np.float32)
        if rows and ptr[-1]
        else np.zeros((0, len(COLS)), np.float32)
    )
    return ClipMotion(
        np.asarray(frames, np.int32),
        np.asarray(ptr, np.int32),
        flat,
        np.asarray(status, np.int8),
        dataclasses.asdict(p),
    )


# ---------------------------------------------------------------------------------------------- per match
def motion_dir(pool_match_dir: str | Path, detector_sha8: str) -> Path:
    """Motion uses the detector's person boxes, so it is keyed by the detector sha like the candidates it read."""
    return Path(pool_match_dir) / f"motion.{detector_sha8}"


def run_match(
    segments: list[Segment],
    cand_dir: str | Path,
    out: str | Path,
    *,
    params: MParams | None = None,
    max_seconds: float | None = None,
    code_sha: str | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Motion for every segment whose candidates exist in ``cand_dir``; resumable per clip; ``meta.json`` written last.

    ``complete`` requires every candidate clip to have a motion file made with ``params`` (a different parameter set restarts)."""
    from footy_track.ball_mining.candidates import load_clip  # noqa: PLC0415

    params = params or MParams()
    cand_dir, out = Path(cand_dir), Path(out)
    if not stage_done(cand_dir):
        raise RuntimeError(
            f"candidates in {cand_dir} are not complete (no finished meta.json)"
        )
    cmeta = json.loads((cand_dir / META).read_text())
    out.mkdir(parents=True, exist_ok=True)
    prev: dict = {}
    with contextlib.suppress(OSError, ValueError):
        prev = json.loads((out / META).read_text())
    clips: dict[str, dict] = (
        dict(prev.get("clips") or {}) if prev.get("params_key") == params.key() else {}
    )
    want = [c for c in cmeta["expected"] if (cand_dir / f"{c}.npz").is_file()]
    by_clip = {s.clip: s for s in segments}
    status = {
        "done": [],
        "skipped": [],
        "failed": {},
        "frames": 0,
        "seconds": 0.0,
        "stopped_early": False,
    }
    t_start = time.time()
    for n, clip in enumerate(want, 1):
        dst = out / f"{clip}.npz"
        if dst.is_file() and clip in clips:
            status["skipped"].append(clip)
            continue
        if clip not in by_clip:
            continue
        if max_seconds is not None and time.time() - t_start > max_seconds:
            status["stopped_early"] = True
            break
        t0 = time.time()
        try:
            m = motion_clip(
                by_clip[clip].path, load_clip(cand_dir / f"{clip}.npz"), params
            )
        except Exception as e:  # noqa: BLE001 - retried on the next run
            status["failed"][clip] = f"{type(e).__name__}: {e}"
            log(f"[{n}/{len(want)}] {clip}: FAILED {status['failed'][clip]}")
            continue
        save_motion(m, dst)
        dt = time.time() - t0
        clips[clip] = {
            "n_frames": int(len(m.frames)),
            "n_blobs": int(len(m.rows)),
            "seconds": round(dt, 2),
            "aligned_both": int((m.status == 2).sum()),
        }
        status["done"].append(clip)
        status["frames"] += int(len(m.frames))
        status["seconds"] += dt
        log(
            f"[{n}/{len(want)}] {clip}: {len(m.frames)} frames, {dt / max(len(m.frames), 1):.3f} s/frame"
        )
    complete = all((out / f"{c}.npz").is_file() and c in clips for c in want) and bool(
        want
    )
    atomic_write_json(
        out / META,
        {
            "schema": SCHEMA, "complete": complete, "params": dataclasses.asdict(params), "params_key": params.key(),
            "detector_sha8": cmeta["detector"]["sha8"], "stride": cmeta["stride"], "expected": want, "clips": clips,
            "code_sha": code_sha, "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )  # fmt: skip
    status["complete"] = complete
    status["wall_seconds"] = round(time.time() - t_start, 1)
    return status
