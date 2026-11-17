"""Review sheet of PTZ V3 segment results: three frames per segment with the solved pitch markings projected on them.

    python -m footy_track.scripts.ptz_temporal_sheet --work-root OUT --video-dir VIDEOS --match arsenal_mancity \
        --segments 5 57 84 --rig-json emirates.json --out sheet.jpg [--before-root OLD_RESULTS]

Run where the segment caches (``gray960.npy``, kept with ``keep_cache``) and mp4s are. A frame is drawn green when the solved
camera is *accepted* there (``line_f1_full.json`` ``accepted_ranges``), red when it is solved but not accepted, and shown
without lines when the frame has no solution. The three frames are the 20% / 50% / 80% points of the accepted frames
(of the solved frames when nothing is accepted). The row label gives accepted / calibrated fractions (``--before-root``:
the earlier run's as well) and the recovery summary of ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import ptz
from footy_track.calibration import ptz_temporal as pt
from footy_track.scripts import ptz_temporal as pte

TILE_W = 560
LABEL_W = 300


def pick_frames(accepted: np.ndarray, valid: np.ndarray, n: int = 3) -> list[int]:
    pool = np.flatnonzero(accepted) if accepted.any() else np.flatnonzero(valid)
    if not len(pool):
        return []
    return [int(pool[int(round(q * (len(pool) - 1)))]) for q in np.linspace(0.2, 0.8, n)]


def _mask(ranges, n) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in ranges:
        m[a:b + 1] = True
    return m


def _text(img, lines, x=8, y=24, scale=0.55):
    for i, (s, col) in enumerate(lines):
        cv2.putText(img, s, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def segment_row(work: Path, video: Path, rig: ptz.Rig, seg: int, before: Path | None) -> np.ndarray:
    meta = json.loads((work / "meta.json").read_text())
    f1 = json.loads((work / "line_f1_full.json").read_text())
    states = np.load(work / "states_full.npy")
    # the rig actually used (a recovered segment may sit on a side camera position): stored in the npz
    shared = pt.Shared.from_vec(np.load(work / "calib_full.npz")["rig"])
    n = len(states)
    acc = _mask(f1["accepted_ranges"], n)
    valid = np.isfinite(states).all(axis=1)
    cache = pt.ClipCache(video, work).build(set(), log=lambda *a: None)
    h = int(round(TILE_W * 9 / 16))
    tiles = []
    for t in pick_frames(acc, valid):
        cam = pt.cam_from_state(states[t], shared) if valid[t] else None
        col = (0, 220, 0) if acc[t] else (0, 0, 255)
        img = pte.render(cache, t, {f"t{t} {'accepted' if acc[t] else 'NOT accepted'}" if cam is not None else f"t{t} no solution": (cam, col)},
                         width=TILE_W)
        tiles.append(img)
    while len(tiles) < 3:
        tiles.append(np.zeros((h, TILE_W, 3), np.uint8))
    label = np.zeros((h, LABEL_W, 3), np.uint8)
    rec = meta.get("recovery") or {}
    lines = [(f"seg{seg:03d}  n={n}", (255, 255, 255)),
             (f"accepted {100 * meta['accepted_frac']:.0f}%  calibrated {100 * meta['calibrated_frac']:.0f}%", (0, 255, 0) if meta["accepted_frac"] >= 0.5 else (0, 160, 255))]
    if before is not None and (before / "meta.json").exists():
        b = json.loads((before / "meta.json").read_text())
        lines.append((f"before: accepted {100 * b['accepted_frac']:.0f}%", (200, 200, 200)))
    if rec:
        lines.append((f"recovery used={rec.get('used')} stages={[s.get('stage') for s in rec.get('stages', [])]}", (255, 255, 0)))
        lines.append((f"anchors confirmed: {rec['stages'][-1].get('confirmed', []) if rec.get('stages') else []}"[:44], (255, 255, 0)))
    _text(label, lines)
    return np.hstack([label, *tiles[:3]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--match", default="arsenal_mancity")
    ap.add_argument("--segments", type=int, nargs="+", required=True)
    ap.add_argument("--rig-json", type=Path, required=True)
    ap.add_argument("--before-root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rig = ptz.Rig.from_json(a.rig_json)
    rows = []
    for s in a.segments:
        stem = f"{a.match}_seg{s:03d}@{a.rig_json.stem}"
        video = next(a.video_dir.glob(f"*seg{s:03d}.mp4"))
        pte.WORK_SUFFIX = ""
        rows.append(segment_row(a.work_root / stem, video, rig, s, a.before_root / stem if a.before_root else None))
    cv2.imwrite(str(a.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])


if __name__ == "__main__":
    main()
