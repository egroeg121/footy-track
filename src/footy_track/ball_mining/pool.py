"""Stage 3: merge the sources into pool nodes (8 px clusters), add the tracker's ``gap_frame`` nodes and plausibility.

Sources of a node: ``rtdetr`` (every RT-DETR ball box, conf >= 0.1), ``motion`` (the camera-compensated blob source, recall
only), optionally ``sam3_text`` (sparse SAM 3 probes) and ``tracker_gap`` (a sample the bidirectional tracker interpolated
with no detection; never merged into another node). All points of a frame within :data:`CLUSTER_PX` of each other (union-find)
form one node, as in the experiment (``build_pool.py``). The node sits on its best source's box (RT-DETR > SAM 3 > motion),
so a card shows a real detector box when there is one.

The same functions build the scorer's training table from the experiment's labelled frames (``scorer_data``), so the
features cannot drift between training and production. Every feature is computable from what production has at a sampled
frame ``t``: the RT-DETR boxes and person boxes at ``t``, the RT-DETR boxes at ``t -+ stride`` (the neighbouring SAMPLES, not
frames ``t -+ 1``: the detector runs every ``stride``-th frame), the motion blobs at ``t`` and the tracker's answer at ``t``.
No YOLO / SAM feature and no coverage flag: coverage identified the match in the experiment, so a model that sees it does
not generalise.

Pixels in this module are px@1280 (a 1280x720 frame); the pool stores normalised centre coordinates.

TODO(sam2-gap-fill): SAM2 gap-fill is a later source (``sources`` "sam2", ``sam2_agree``): propagate the tracker's confident anchors across the
``tracker_gap`` frames and record fwd/bwd agreement. The experiment found it covers only frames near confident anchors (recall@10 0.159 on clean
frames, 1 unique hit) and it needs its own GPU stage, so it is not built here; the pool schema already has the column.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from footy_track.ball_mining import motion as motion_mod
from footy_track.ball_mining import schema
from footy_track.ball_mining import track as track_mod
from footy_track.ball_mining.candidates import (
    META,
    ClipCandidates,
    atomic_write_json,
    load_clip,
    stage_done,
)

WW, HH = 1280.0, 720.0
CLUSTER_PX = 8.0
VMAX_PX = 65.0 * 1280.0 / 1920.0  # 65 px/frame at 1080p, as px/frame@1280 (43.3)
HC_CONF = 0.5  # "high confidence" neighbour boxes
PLAUS_CONF = 0.3  # a neighbour box that makes a node plausible
DEFAULT_BALL_PX = 11.0  # ball size at 1280 when only a motion blob proposed the node
SOURCES = ("rtdetr", "sam3_text", "motion")  # precedence for the node's box
GAP = "tracker_gap"
#: columns of the unscored node table that are features (the scorer picks its subset; all production-computable)
FEATURE_COLS = (
    "rt_score", "rt_rank", "rt_n", "xn", "yn", "d_foot", "in_person", "d_head",
    "motion_score", "motion_rank", "motion_n", "mot_contrast", "mot_area", "mot_fill", "mot_asp", "mot_dens", "mot_inperson",
    "d_prev", "d_next", "d_prev_hc", "d_next_hc", "nb_plaus", "nb_plaus_hc", "d_min_nb", "static_nb", "static_hc", "n_src",
    "trk_state", "trk_member", "trk_dist", "trk_agree", "trk_support", "trk_len",
)  # fmt: skip


@dataclass
class TrackSample:
    """The tracker's answer at one sample."""

    state: int  # 0 none, 1 interpolated, 2 detected
    cx: float = np.nan
    cy: float = np.nan
    w: float = np.nan
    h: float = np.nan
    agree: bool = False
    support: float = 0.0
    run_len: int = 0
    track_id: int = -1
    gap_plaus: float = np.nan


@dataclass
class FrameInputs:
    rt: np.ndarray  # (n, 5) cx, cy, w, h, conf
    persons: np.ndarray  # (m, 4) cx, cy, w, h
    motion: np.ndarray | None = None  # (k, 11) motion.COLS rows, None = not computed
    sam: np.ndarray | None = None  # (j, 3) cx, cy, score
    prev: np.ndarray | None = (
        None  # (p, 3) cx, cy, conf of the RT-DETR boxes at t - stride; None = that sample does not exist
    )
    next: np.ndarray | None = None
    trk: TrackSample | None = None


def _cluster(xy: np.ndarray, radius: float = CLUSTER_PX) -> np.ndarray:
    """Union-find labels of points within ``radius`` of each other (single linkage)."""
    n = len(xy)
    par = list(range(n))

    def find(i: int) -> int:
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    if n > 1:
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] <= radius:
                    par[find(i)] = find(j)
    roots: dict[int, int] = {}
    return np.array(
        [roots.setdefault(find(i), len(roots)) for i in range(n)], dtype=int
    )


def _min_dist(
    x: float, y: float, pts: np.ndarray | None, conf_min: float = 0.0
) -> float:
    """Distance to the nearest point of ``pts`` (cx, cy, conf rows); NaN = that sample does not exist, 999 = no point."""
    if pts is None:
        return np.nan
    p = pts[pts[:, 2] >= conf_min] if len(pts) else pts
    return float(np.hypot(p[:, 0] - x, p[:, 1] - y).min()) if len(p) else 999.0


def _nanmin2(a: float, b: float) -> float:
    v = [t for t in (a, b) if not np.isnan(t)]
    return min(v) if v else np.nan


def frame_nodes(fi: FrameInputs, stride: int) -> list[dict]:  # noqa: PLR0912, PLR0915
    """Nodes of one sampled frame: dicts with ``x, y, w, h`` (px@1280), ``sources``, ``source_scores``, ``is_gap``,
    ``plausible``, ``track_id`` and every :data:`FEATURE_COLS` entry."""
    pts: list[dict] = []
    for cx, cy, w, h, c in fi.rt:
        pts.append(
            {
                "src": "rtdetr",
                "x": float(cx),
                "y": float(cy),
                "w": float(w),
                "h": float(h),
                "score": float(c),
            }
        )
    if fi.sam is not None:
        for cx, cy, s in fi.sam:
            pts.append(
                {
                    "src": "sam3_text",
                    "x": float(cx),
                    "y": float(cy),
                    "w": DEFAULT_BALL_PX,
                    "h": DEFAULT_BALL_PX,
                    "score": float(s),
                }
            )
    mot = (
        fi.motion
        if fi.motion is not None
        else np.zeros((0, len(motion_mod.COLS)), np.float32)
    )
    for r in mot:
        pts.append(
            {
                "src": "motion",
                "x": float(r[0]),
                "y": float(r[1]),
                "w": float(r[5]),
                "h": float(r[6]),
                "score": float(r[2]),
                "row": r,
            }
        )
    trk = fi.trk
    nodes: list[dict] = []
    if pts:
        lab = _cluster(np.array([[p["x"], p["y"]] for p in pts]))
        rt_scores = [p["score"] for p in pts if p["src"] == "rtdetr"]
        mot_scores = [p["score"] for p in pts if p["src"] == "motion"]
        persons = fi.persons
        for k in range(int(lab.max()) + 1):
            mem = [pts[i] for i in np.where(lab == k)[0]]
            by: dict[str, list[dict]] = {}
            for m in mem:
                by.setdefault(m["src"], []).append(m)
            best_src = next(s for s in SOURCES if s in by)
            best = max(by[best_src], key=lambda m: m["score"])
            x = float(np.mean([m["x"] for m in mem]))
            y = float(np.mean([m["y"] for m in mem]))
            n: dict = {
                "x": x, "y": y, "w": best["w"], "h": best["h"], "is_gap": False,
                "sources": [s for s in SOURCES if s in by],
                "source_scores": [float(np.clip(max(m["score"] for m in by[s]), 0.0, 1.0)) for s in SOURCES if s in by],
            }  # fmt: skip
            rt = by.get("rtdetr", [])
            n["rt_n"] = float(len(rt))
            n["rt_score"] = max((m["score"] for m in rt), default=np.nan)
            n["rt_rank"] = (
                float(sum(s > n["rt_score"] for s in rt_scores)) if rt else np.nan
            )
            mo = by.get("motion", [])
            n["motion_n"] = float(len(mo))
            n["motion_score"] = max((m["score"] for m in mo), default=np.nan)
            n["motion_rank"] = (
                float(sum(s > n["motion_score"] for s in mot_scores)) if mo else np.nan
            )
            if mo:
                top = max(mo, key=lambda m: m["score"])["row"]
                for name, col in (
                    ("contrast", 3),
                    ("area", 4),
                    ("fill", 7),
                    ("asp", 8),
                    ("dens", 9),
                    ("inperson", 10),
                ):
                    n[f"mot_{name}"] = float(top[col])
            else:
                for name in ("contrast", "area", "fill", "asp", "dens", "inperson"):
                    n[f"mot_{name}"] = np.nan
            n["n_src"] = float(sum(s in by for s in ("rtdetr", "motion")))
            n["xn"], n["yn"] = x / WW, y / HH
            if len(persons):
                n["d_foot"] = float(
                    np.hypot(
                        persons[:, 0] - x, persons[:, 1] + persons[:, 3] / 2 - y
                    ).min()
                )
                n["d_head"] = float(
                    np.hypot(
                        persons[:, 0] - x, persons[:, 1] - persons[:, 3] / 2 - y
                    ).min()
                )
                n["in_person"] = float(
                    (
                        (np.abs(persons[:, 0] - x) <= persons[:, 2] / 2)
                        & (np.abs(persons[:, 1] - y) <= persons[:, 3] / 2)
                    ).any()
                )
            else:
                n["d_foot"] = n["d_head"] = n["in_person"] = np.nan
            n["d_prev"] = _min_dist(x, y, fi.prev)
            n["d_next"] = _min_dist(x, y, fi.next)
            n["d_prev_hc"] = _min_dist(x, y, fi.prev, HC_CONF)
            n["d_next_hc"] = _min_dist(x, y, fi.next, HC_CONF)
            gate = VMAX_PX * stride
            n["nb_plaus"] = float(
                sum((not np.isnan(n[c])) and n[c] <= gate for c in ("d_prev", "d_next"))
            )
            n["nb_plaus_hc"] = float(
                sum(
                    (not np.isnan(n[c])) and n[c] <= gate
                    for c in ("d_prev_hc", "d_next_hc")
                )
            )
            n["d_min_nb"] = _nanmin2(n["d_prev"], n["d_next"])
            n["static_nb"] = (
                float(n["d_min_nb"] <= 2.0) if not np.isnan(n["d_min_nb"]) else np.nan
            )
            hc = _nanmin2(n["d_prev_hc"], n["d_next_hc"])
            n["static_hc"] = float(hc <= 2.0) if not np.isnan(hc) else np.nan
            _trk_features(n, trk)
            n["plausible"] = _plausible(n, fi, gate, stride)
            n["track_id"] = (
                trk.track_id if (trk is not None and n["trk_member"] == 1.0) else -1
            )
            nodes.append(n)
    if trk is not None and trk.state == track_mod.INTERP and not np.isnan(trk.cx):
        g = dict.fromkeys(FEATURE_COLS, np.nan)
        g.update(
            x=trk.cx, y=trk.cy, w=trk.w, h=trk.h, is_gap=True, sources=[GAP],
            source_scores=[float(min(max(trk.gap_plaus if not np.isnan(trk.gap_plaus) else 0.0, 0.0), 1.0))],
            plausible=True, track_id=trk.track_id, trk_state=1.0,
        )  # fmt: skip
        g["xn"], g["yn"] = trk.cx / WW, trk.cy / HH
        nodes.append(g)
    return nodes


def _trk_features(n: dict, trk: TrackSample | None) -> None:
    if trk is None:
        for c in (
            "trk_state",
            "trk_member",
            "trk_dist",
            "trk_agree",
            "trk_support",
            "trk_len",
        ):
            n[c] = np.nan
        return
    n["trk_state"] = float(trk.state)
    if trk.state == track_mod.NONE_ or np.isnan(trk.cx):
        n["trk_dist"] = np.nan
        member = False
    else:
        n["trk_dist"] = float(np.hypot(n["x"] - trk.cx, n["y"] - trk.cy))
        member = trk.state == track_mod.DETECT and n["trk_dist"] <= CLUSTER_PX
    n["trk_member"] = float(member)
    n["trk_agree"] = float(member and trk.agree)
    n["trk_support"] = float(trk.support) if member else 0.0
    n["trk_len"] = float(trk.run_len) if member else 0.0


def _plausible(n: dict, fi: FrameInputs, gate: float, stride: int) -> bool | None:  # noqa: ARG001
    """On the tracker's path, or an RT-DETR box (conf >= 0.3) within ``VMAX * stride`` at a neighbouring sample. False when
    neighbouring samples exist and neither qualifies (off any plausible track); None when there is no neighbouring sample."""
    if n.get("trk_member") == 1.0:
        return True
    if fi.prev is None and fi.next is None:
        return None
    for pts in (fi.prev, fi.next):
        if (
            pts is not None
            and len(pts)
            and _min_dist(n["x"], n["y"], pts, PLAUS_CONF) <= gate
        ):
            return True
    return False


# ---------------------------------------------------------------------------------------------- clip level
def _rt_px(cand: ClipCandidates, frame: int) -> np.ndarray:
    b = cand.balls(frame)
    if not len(b):
        return np.zeros((0, 5), np.float32)
    return np.stack(
        [
            (b[:, 0] + b[:, 2] / 2) * WW,
            (b[:, 1] + b[:, 3] / 2) * HH,
            b[:, 2] * WW,
            b[:, 3] * HH,
            b[:, 4],
        ],
        1,
    ).astype(np.float32)


def track_samples(
    tr: track_mod.TrackResult | None, size: tuple[int, int] = (1280, 720)
) -> dict[int, TrackSample]:
    """``{frame: TrackSample}`` with centres in px@1280."""
    if tr is None:
        return {}
    out = {}
    for i, f in enumerate(tr.frames):
        s = int(tr.state[i])
        if s == track_mod.NONE_:
            out[int(f)] = TrackSample(0)
            continue
        x, y, w, h = (float(v) for v in tr.box[i])
        out[int(f)] = TrackSample(
            s, (x + w / 2) * WW, (y + h / 2) * HH, w * WW, h * HH, bool(tr.agree[i]), float(tr.support[i]), int(tr.run_len[i]),
            int(tr.track_id[i]), float(tr.gap_plaus[i]),
        )  # fmt: skip
    return out


def clip_nodes(
    match: str,
    clip: str,
    cand: ClipCandidates,
    motion: motion_mod.ClipMotion | None,
    tr: track_mod.TrackResult | None,
    *,
    detector_sha: str,
    sam: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    """All nodes of one clip (the unscored node table: pool columns without ``p_real``, plus the feature columns)."""
    stride = cand.stride
    samples = track_samples(tr)
    rows: list[dict] = []
    for t in (int(f) for f in cand.frames):
        prev = _pts3(cand, t - stride) if cand.has(t - stride) else None
        nxt = _pts3(cand, t + stride) if cand.has(t + stride) else None
        fi = FrameInputs(
            rt=_rt_px(cand, t),
            persons=cand.persons_px(t, WW, HH),
            motion=None if motion is None else motion.at(t),
            sam=None if sam is None else sam.get(t),
            prev=prev, next=nxt, trk=samples.get(t),
        )  # fmt: skip
        for n in frame_nodes(fi, stride):
            n.update(
                match=match, clip=clip, frame=t, fps=cand.fps, detector_sha=detector_sha
            )
            rows.append(n)
    return nodes_frame(rows)


def _pts3(cand: ClipCandidates, frame: int) -> np.ndarray:
    r = _rt_px(cand, frame)
    return r[:, [0, 1, 4]] if len(r) else np.zeros((0, 3), np.float32)


def nodes_frame(rows: list[dict]) -> pd.DataFrame:
    """Node dicts -> DataFrame in pool units: normalised centre ``x, y, w, h``, schema columns, feature columns."""
    if not rows:
        return pd.DataFrame(columns=[*schema.COLUMNS, *FEATURE_COLS, "is_gap_node"])
    df = pd.DataFrame(rows)
    df["w"] = (df["w"] / WW).clip(1e-4, 1.0)
    df["h"] = (df["h"] / HH).clip(1e-4, 1.0)
    df["x"] = (df["x"] / WW).clip(0.0, 1.0)
    df["y"] = (df["y"] / HH).clip(0.0, 1.0)
    df["n_sources"] = df["sources"].map(len).astype("int32")
    df["gap_frame"] = df["is_gap"].astype(object)
    df["track_id"] = pd.array(
        df["track_id"].where(df["track_id"] >= 0, other=pd.NA), dtype="Int64"
    )
    df["plausible"] = (
        df["plausible"].astype(object).where(df["plausible"].notna(), None)
    )
    df["static_flag"] = (
        schema.derive_static_flags(df).astype(object) if len(df) else None
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- match level
def nodes_path(match_dir: str | Path, detector_sha8: str) -> Path:
    """The unscored node table. Parquet content, but NOT a ``*.parquet`` name: the miner reads every ``*.parquet`` below ``ball_pool/``
    (``schema.read_pool``), and this table has no ``p_real`` and extra feature columns."""
    return Path(match_dir) / f"nodes.{detector_sha8}.pq"


def build_match_nodes(
    match_dir: str | Path,
    detector_sha8: str,
    *,
    sam_dir: str | Path | None = None,
    log=print,
    code_sha: str | None = None,
) -> dict:
    """Read ``candidates.<sha8>`` and ``motion.<sha8>`` of one match, run the tracker per clip, write ``nodes.<sha8>.pq``
    (+ ``.json``). The candidates must be complete; motion may be absent (then the motion features are NaN and the run says so)."""
    d = Path(match_dir)
    cdir = d / f"candidates.{detector_sha8}"
    if not stage_done(cdir):
        raise RuntimeError(f"candidates in {cdir} are not complete")
    cmeta = json.loads((cdir / META).read_text())
    match = d.name
    mdir = motion_mod.motion_dir(d, detector_sha8)
    have_motion = stage_done(mdir)
    sam_probes = _load_sam_probes(sam_dir) if sam_dir else {}
    frames: list[pd.DataFrame] = []
    t0 = time.time()
    stats = {"clips": 0, "nodes": 0, "gap_nodes": 0, "clips_without_motion": 0}
    for clip in cmeta["expected"]:
        cpath = cdir / f"{clip}.npz"
        if not cpath.is_file():
            continue
        cand = load_clip(cpath)
        mot = (
            motion_mod.load_motion(mdir / f"{clip}.npz")
            if have_motion and (mdir / f"{clip}.npz").is_file()
            else None
        )
        if mot is None:
            stats["clips_without_motion"] += 1
        tr = track_mod.run_tracker(cand)
        df = clip_nodes(
            match,
            clip,
            cand,
            mot,
            tr,
            detector_sha=detector_sha8,
            sam=sam_probes.get(clip),
        )
        frames.append(df)
        stats["clips"] += 1
        stats["nodes"] += len(df)
        stats["gap_nodes"] += int(df["gap_frame"].eq(True).sum()) if len(df) else 0
    allnodes = (
        pd.concat([f for f in frames if len(f)], ignore_index=True)
        if any(len(f) for f in frames)
        else nodes_frame([])
    )
    out = nodes_path(d, detector_sha8)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    allnodes.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    meta = {
        "schema": "ball-nodes/v1", "match": match, "detector_sha8": detector_sha8, "stride": cmeta["stride"], "motion": have_motion,
        "sam3_probes": bool(sam_probes), "features": list(FEATURE_COLS), "code_sha": code_sha, "seconds": round(time.time() - t0, 1), **stats,
    }  # fmt: skip
    atomic_write_json(out.with_suffix(".json"), meta)
    log(
        f"{match}: {stats['nodes']} nodes ({stats['gap_nodes']} gap) over {stats['clips']} clips in {meta['seconds']} s"
    )
    return meta


def _load_sam_probes(sam_dir: str | Path) -> dict[str, dict[int, np.ndarray]]:
    """``{clip: {frame: (n, 3) cx, cy, score}}`` (px@1280) from ``sam3_probes`` jsonl: rows ``{clip, frame, w, h, dets: [[x0,y0,x1,y1,score]]}``."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for f in sorted(Path(sam_dir).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            w, h = float(r.get("w") or 1920), float(r.get("h") or 1080)
            pts = [
                ((d[0] + d[2]) / 2 * WW / w, (d[1] + d[3]) / 2 * HH / h, float(d[4]))
                for d in r.get("dets") or []
            ]
            out.setdefault(r["clip"], {})[int(r["frame"])] = np.asarray(
                pts, np.float32
            ).reshape(-1, 3)
    return out
