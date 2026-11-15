"""Render a calibrated clip: broadcast frame with the re-projected pitch lines | 2D minimap with the players.

    PYTHONPATH=src python -m footy_track.scripts.projection_demo --clip arsenal_mancity_seg010 \\
        --calibration OUT/arsenal_mancity_seg010.k6_every100.calibration.parquet --out OUT/projection \\
        [--start 0 --stop 595 --step 1] [--stills 100,250,400,500] [--width 960]

Writes ``<clip>_projection.mp4`` (H.264, ~2 Mbps) and ``<clip>_still_<frame>.png``. Dots are coloured by kit colour
(2-means over the torso colour of the players of the clip), referees yellow, other staff grey, the ball white; frames
without an ``ok`` calibration show no lines and no dots.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration import temporal as tp
from footy_track.calibration.clip_calibration import ClipCalibration
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.pipeline import find_video
from footy_track.clip_id import canonical_clip_id
from footy_track.projection.players import BALL_LABELS, load_detections, project_players
from footy_track.scripts.calibrate_clip import CLIPS_DIR, find_boxes_path

PITCH = PitchModel()
GRASS_BGR = (52, 110, 52)
LINE_BGR = (235, 235, 235)


def torso_feature(frame: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray | None:
    """Mean Lab colour of the non-grass torso pixels of a normalised box, or None."""
    H, W = frame.shape[:2]
    x0, x1 = int((x + 0.25 * w) * W), int((x + 0.75 * w) * W)
    y0, y1 = int((y + 0.15 * h) * H), int((y + 0.5 * h) * H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = frame[max(y0, 0):y1, max(x0, 0):x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    keep = ~((hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 60))
    if keep.sum() < 4:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[keep].astype(np.float32)
    # chroma (a, b) carries the kit colour; lightness mostly says sun vs shadow, so it is down-weighted
    return np.array([0.25 * lab[:, 0].mean(), lab[:, 1].mean() - 128.0, lab[:, 2].mean() - 128.0], np.float32)


def kit_colours(video: Path, dets, step: int = 15, width: int = 960) -> tuple[np.ndarray, np.ndarray]:
    """2-means centres (weighted L, a, b) and their BGR display colours from the players of every ``step``-th frame."""
    feats = []
    for idx, small, _ in tp.iter_video_small(video, width):
        if idx % step:
            continue
        for r in dets[(dets["frame"] == idx) & (dets["label"] == "player")].itertuples():
            f = torso_feature(small, r.x, r.y, r.w, r.h)
            if f is not None:
                feats.append(f)
    X = np.array(feats, np.float32)
    _, _, centres = cv2.kmeans(X, 2, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 5, cv2.KMEANS_PP_CENTERS)
    lab = np.stack([np.full(2, 150.0), centres[:, 1] * 1.6 + 128, centres[:, 2] * 1.6 + 128], 1)  # bright, saturated display colour
    bgr = cv2.cvtColor(np.clip(lab, 0, 255).reshape(1, 2, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).reshape(2, 3)
    return centres, bgr


class Renderer:
    def __init__(self, cal: ClipCalibration, dets, centres, kit_bgr, width: int = 960):
        self.cal, self.dets, self.centres, self.kit = cal, dets, centres, kit_bgr
        self.pos = project_players(dets, cal)
        self.W = width
        self.lw = int(round(width * 0.6 / 2)) * 2
        self.rw = width - self.lw
        self.h = int(round(self.lw * 9 / 16 / 2)) * 2
        self.s = (self.rw - 12) / (PITCH.length + 8)
        self.mini_bg = self._minimap_bg()

    def _m2px(self, x, y):
        s = self.s
        return int(round((x + PITCH.hl + 4) * s + 6)), int(round((PITCH.hw + 4 - y) * s + (self.h - (PITCH.width + 8) * s) / 2))

    def _minimap_bg(self):
        img = np.full((self.h, self.rw, 3), GRASS_BGR, np.uint8)
        for poly in PITCH.polylines().values():
            pts = np.array([self._m2px(x, y) for x, y in poly], np.int32)
            cv2.polylines(img, [pts], False, LINE_BGR, 1, cv2.LINE_AA)
        return img

    def team_colour(self, frame_small, r):
        if r.label in BALL_LABELS:
            return (255, 255, 255)
        if r.label == "referee":
            return (0, 235, 255)
        if r.label != "player":
            return (150, 150, 150)
        f = torso_feature(frame_small, r.x, r.y, r.w, r.h)
        if f is None:
            return (150, 150, 150)
        k = int(np.argmin(np.linalg.norm(self.centres - f, axis=1)))
        return tuple(int(c) for c in self.kit[k])

    def render(self, frame_small: np.ndarray, idx: int) -> np.ndarray:
        row = self.cal.row(idx)
        left = cv2.resize(frame_small, (self.lw, self.h), interpolation=cv2.INTER_AREA)
        mini = self.mini_bg.copy()
        ok = row is not None and bool(row["ok"])
        sc = max(1, self.W // 640)
        if ok:
            H, dist = self.cal.H(idx), self.cal.distortion(idx)
            for runs in geo.project_pitch_model(H, PITCH, 0.5, dist).values():
                for run in runs:
                    pts = (run * [self.lw, self.h]).astype(np.int32)
                    if np.abs(pts).max() < 20000:
                        cv2.polylines(left, [pts], False, (0, 220, 255), 1, cv2.LINE_AA)
            dd = self.dets[self.dets["frame"] == idx].reset_index(drop=True)
            pp = self.pos[self.pos["frame"] == idx].reset_index(drop=True)
            for r, p in zip(dd.itertuples(), pp.itertuples(), strict=True):
                if not p.in_bounds:
                    continue
                col = self.team_colour(frame_small, r)
                ball = r.label in BALL_LABELS
                u, v = self._m2px(p.x_m, p.y_m)
                rad = (3 if ball else 4) * sc
                cv2.circle(mini, (u, v), rad, col, -1, cv2.LINE_AA)
                cv2.circle(mini, (u, v), rad, (0, 0, 0), 1, cv2.LINE_AA)
                fx, fy = int((r.x + r.w / 2) * self.lw), int((r.y + (r.h / 2 if ball else r.h)) * self.h)
                cv2.circle(left, (fx, fy), 2 * sc, col, -1, cv2.LINE_AA)
                cv2.circle(left, (fx, fy), 2 * sc, (0, 0, 0), 1, cv2.LINE_AA)
        txt = "no calibration" if row is None or row["source"] == "none" else (
            f"{row['source']}  gap {int(row['gap'])}  conf {row['confidence']:.2f}" + ("" if ok else "  NOT OK"))
        cv2.putText(left, f"f{idx}  {txt}", (6, 16 * sc), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, (0, 0, 0), 3 * sc // 2 + 1, cv2.LINE_AA)
        cv2.putText(left, f"f{idx}  {txt}", (6, 16 * sc), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, (255, 255, 255) if ok else (80, 80, 255), 1, cv2.LINE_AA)
        if not ok:
            cv2.putText(mini, "not calibrated", (self.rw // 2 - 60 * sc, self.h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (200, 200, 200), 1, cv2.LINE_AA)
        return np.hstack([left, mini])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--stills", default="", help="comma-separated frames")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--still-width", type=int, default=1440)
    ap.add_argument("--bitrate", default="2M")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cid = canonical_clip_id(a.clip)
    video = find_video(CLIPS_DIR, a.clip)
    cal = ClipCalibration.load(a.calibration)
    dets = load_detections(find_boxes_path(a.clip))
    centres, kit = kit_colours(video, dets)
    still_frames = {int(f) for f in a.stills.split(",") if f}
    vid = Renderer(cal, dets, centres, kit, a.width)
    big = Renderer(cal, dets, centres, kit, a.still_width) if still_frames else None
    ff = None
    if a.step:
        mp4 = a.out / f"{cid}_projection.mp4"
        ff = subprocess.Popen(
            [shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{vid.W}x{vid.h}", "-r", str(a.fps / a.step),
             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-threads", "2", "-b:v", a.bitrate, "-maxrate", "2.5M", "-bufsize", "4M",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    stop = a.stop if a.stop is not None else cal.start + len(cal.table)
    n = 0
    for idx, small, _ in tp.iter_video_small(video, max(a.width, a.still_width) if still_frames else a.width, a.start, stop):
        if ff is not None and (idx - a.start) % a.step == 0:
            ff.stdin.write(vid.render(cv2.resize(small, (a.width, round(small.shape[0] * a.width / small.shape[1])), interpolation=cv2.INTER_AREA)
                                      if small.shape[1] != a.width else small, idx).tobytes())
            n += 1
        if idx in still_frames:
            cv2.imwrite(str(a.out / f"{cid}_still_{idx}.png"), big.render(small, idx))
    if ff is not None:
        ff.stdin.close()
        ff.wait()
        print(f"wrote {n} frames to {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
