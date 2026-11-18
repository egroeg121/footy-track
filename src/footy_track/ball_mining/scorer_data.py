"""The scorer's training table, built from the experiment's labelled frames with the PRODUCTION feature code.

Source: ``experiments/ball_candidate_pool`` (branch ``exp/ball-candidate-pool``, commit 07930b1; outputs in
``/mnt/storage/exp/ball_pool``): human GT ball marks, Ball Check verdicts, per-clip RT-DETR output (``machine_labels``, the
``exp/sam3`` RT-DETR dirs, a CPU run for clips that had none), and the motion source computed on every labelled frame +-1.
Labels and the contamination rule are ported from the experiment's ``labels.py`` unchanged (see below); the nodes are NOT taken
from its ``nodes.pkl`` (those carry YOLO / SAM / coverage columns and +-1 frame neighbours): every labelled frame is turned into
nodes with :func:`footy_track.ball_mining.pool.frame_nodes`, the same function the production stage uses, with neighbours at
the stride-5 grid and the production bidirectional tracker run over the RT-DETR samples of that grid.

Label of a node (px@1280, experiment rule): positive when within 10 px of a human GT ball (tier 1 = tag ``labeller``; tier 2 =
``vittrack`` + ``human_checked``) or a Ball Check ball / corrected position; negative when within 10 px of a Ball Check
``not_ball`` box, or the frame has a confirmed ball more than 25 px away, or the frame is ``no_ball``; otherwise unlabelled
(-1). ``contam``: the frame is within 5 frames of an image the detector trained or validated on (RT-DETR memorised those; their
confidences are not honest), so the scorer trains and is judged on CLEAN frames only.

Needs the experiment's artifacts on disk; it is a one-off builder, not a production stage.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from footy_track.ball_mining import candidates as cand_mod
from footy_track.ball_mining import motion as motion_mod
from footy_track.ball_mining import pool as pool_mod
from footy_track.ball_mining import track as track_mod
from footy_track.clip_id import canonical_clip_id

W, H = 1280.0, 720.0
DATA = Path("/mnt/storage/footy_data")
EXP = Path("/mnt/storage/exp/ball_pool")
GT_DIRS = ["ball_gt_marks", "ball_gt_marks_nottinghamforest"]
VERDICT_DIRS = [
    "ball_gt_marks",
    "ball_gt_marks_nottinghamforest",
    "ball_gt_marks_nottinghamforest_disagree",
    "ball_gt_marks_nottinghamforest_vittrack",
]
POS_PX = 10.0
NEG_ELSEWHERE_PX = 25.0
PERSON_TAGS = {"player", "referee", "coach", "person", "player_sub"}
BALL_TAGS = {"in_play_ball", "ball"}
STRIDE = 5
FRAME_SIZE = (1920, 1080)


def match_of(cid: str) -> str:
    """Canonical clip id -> match. ``arsenal_example`` is the same Arsenal v Man City broadcast (checked visually in the
    experiment), so it folds into ``arsenal_mancity`` for every match-level split."""
    if cid == "arsenal_example":
        return "arsenal_mancity"
    for m in (
        "arsenal_mancity",
        "arsenal_astonvilla",
        "arsenal_bournemouth_1st",
        "arsenal_nottinghamforest",
    ):
        if cid.startswith(m + "_"):
            return m
    return cid


def _tier(tags) -> int:
    t = set(tags)
    if "labeller" in t:
        return 1
    if "vittrack" in t and "human_checked" in t:
        return 2
    return 0


def load_gt(data: Path = DATA, dirs=GT_DIRS):
    """``(gt[clip][frame] = [(x, y, tier)], noball[clip], notbroadcast[clip])`` in px@1280."""
    gt: dict = defaultdict(lambda: defaultdict(dict))
    noball, notbc = defaultdict(set), defaultdict(set)
    for d in dirs:
        for f in sorted(glob.glob(str(data / d / "*.jsonl"))):
            cid = canonical_clip_id(f)
            for line in Path(f).read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                tags = r.get("tags") or []
                fi = r.get("frame_index")
                if "no_ball" in tags:
                    noball[cid].add(fi)
                    continue
                if "not_broadcast" in tags:
                    notbc[cid].add(fi)
                    continue
                if "in_play_ball" not in tags:
                    continue  # out_of_play_ball is not a candidate truth
                tier = _tier(tags)
                if tier == 0 or not r.get("center"):
                    continue
                x, y = r["center"]["x"] * W, r["center"]["y"] * H
                key = (round(x, 1), round(y, 1))
                old = gt[cid][fi].get(key)
                gt[cid][fi][key] = min(tier, old) if old else tier
    out = {
        c: {fi: [(k[0], k[1], t) for k, t in v.items()] for fi, v in fr.items()}
        for c, fr in gt.items()
    }
    return out, dict(noball), dict(notbc)


def load_verdicts(data: Path = DATA, dirs=VERDICT_DIRS) -> dict:
    """Ball Check verdicts ``v[clip][(frame, box_index, source)] = {verdict, xy, cxy}``; last line wins, ``null`` is undo."""
    best: dict = {}
    for d in dirs:
        for f in sorted(glob.glob(str(data / d / "ball_checks" / "*.jsonl"))):
            for line in Path(f).read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                cid = canonical_clip_id(r.get("clip") or f)
                key = (cid, r["frame_index"], r.get("box_index"), r.get("source"))
                ts = r.get("ts", 0)
                if key not in best or ts >= best[key][0]:
                    best[key] = (ts, r)
    v: dict = defaultdict(dict)
    for (cid, fi, bi, src), (_, r) in best.items():
        if r.get("verdict") is None:
            continue
        bb = r["bbox"]
        c = ((bb["x"] + bb["w"] / 2) * W, (bb["y"] + bb["h"] / 2) * H)
        cc = None
        if r.get("corrected_bbox"):
            cb = r["corrected_bbox"]
            cc = ((cb["x"] + cb["w"] / 2) * W, (cb["y"] + cb["h"] / 2) * H)
        v[cid][(fi, bi, src)] = {"verdict": r["verdict"], "xy": c, "cxy": cc}
    return dict(v)


def contaminated_frames_by_clip(
    data: Path = DATA, near: int = 5
) -> dict[str, set[int]]:
    """``{clip: frames}`` within ``near`` frames of an image of the detector's Roboflow v11 / v3 train or valid sets."""
    pat = re.compile(r"^(?P<stem>.*)_(?P<frame>\d{6})_(?:png|jpg|tif|tiff)\.rf\.")
    out: dict[str, set[int]] = defaultdict(set)
    for ver in ("11", "3"):
        for split in ("train", "valid"):
            d = (
                data
                / "detection_dataset"
                / f"roboflow_dataset_{ver}"
                / split
                / "images"
            )
            if d.is_dir():
                for f in os.listdir(d):
                    m = pat.match(f)
                    if m:
                        fr = int(m["frame"])
                        out[canonical_clip_id(m["stem"])].update(
                            range(fr - near, fr + near + 1)
                        )
    return dict(out)


# ---------------------------------------------------------------------------------------------- detector output
def read_machine_labels(
    path: str | Path,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], set[int]]:
    """``(balls[frame] (n,5) cx cy w h conf, persons[frame] (m,4) cx cy w h, processed frames)`` in px@1280."""
    balls: dict[int, list] = defaultdict(list)
    persons: dict[int, list] = defaultdict(list)
    seen: set[int] = set()
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            tg, b, c, fi = set(r["tags"]), r["bbox"], r["center"], r["frame_index"]
            conf = r.get("confidence", 1.0)
            if tg & BALL_TAGS:
                balls[fi].append((c["x"] * W, c["y"] * H, b["w"] * W, b["h"] * H, conf))
                seen.add(fi)
            elif tg & PERSON_TAGS and conf >= cand_mod.PERSON_CONF:
                persons[fi].append((c["x"] * W, c["y"] * H, b["w"] * W, b["h"] * H))
                seen.add(fi)
    return (
        {k: np.asarray(v, np.float32) for k, v in balls.items()},
        {k: np.asarray(v, np.float32) for k, v in persons.items()},
        seen,
    )


def _clip_candidates(
    balls, persons, frames: list[int], stride: int
) -> cand_mod.ClipCandidates:
    """A ``ClipCandidates`` (normalised top-left xywh) over ``frames`` from px@1280 centre arrays, for the tracker."""
    bl, pl = [], []
    for f in frames:
        b = balls.get(f)
        bl.append(
            np.zeros((0, 5), np.float32)
            if b is None
            else np.stack(
                [
                    (b[:, 0] - b[:, 2] / 2) / W,
                    (b[:, 1] - b[:, 3] / 2) / H,
                    b[:, 2] / W,
                    b[:, 3] / H,
                    b[:, 4],
                ],
                1,
            ).astype(np.float32)
        )
        pl.append(np.zeros((0, 5), np.float32))
    return cand_mod.build_clip(
        frames,
        bl,
        pl,
        size=FRAME_SIZE,
        fps=25.0,
        n_src_frames=max(frames) + 1 if frames else 0,
        stride=stride,
    )


def _label_nodes(
    nodes: list[dict],
    pos: list,
    posv: list,
    negv: list,
    has_gt: bool,
    tier_of: dict,
    is_noball: bool,
) -> None:
    allpos = pos + posv
    for n in nodes:
        x, y = n["x"], n["y"]
        dpos = min((np.hypot(px - x, py - y) for px, py in allpos), default=np.inf)
        dneg = min((np.hypot(px - x, py - y) for px, py in negv), default=np.inf)
        tier = 0
        if dpos <= POS_PX:
            lab = 1
            near = [
                t
                for (kx, ky), t in tier_of.items()
                if np.hypot(kx - x, ky - y) <= POS_PX
            ]
            tier = min(near) if near else 3  # 3 = Ball Check verdict
        elif dneg <= POS_PX or has_gt and dpos > NEG_ELSEWHERE_PX or is_noball:
            lab = 0
        else:
            lab = -1
        n["label"], n["pos_tier"] = lab, tier
        n["neg_kind"] = (
            "bc_not_ball"
            if (lab == 0 and dneg <= POS_PX and dpos > POS_PX)
            else "elsewhere"
            if (lab == 0 and has_gt)
            else "no_ball"
            if lab == 0
            else ""
        )


def build_table(  # noqa: PLR0912, PLR0915
    *,
    exp: Path = EXP,
    data: Path = DATA,
    stride: int = STRIDE,
    only_clips: set[str] | None = None,
    log=print,
) -> pd.DataFrame:
    """The labelled nodes of every clip in the experiment's inventory. Columns: ``match, clip, frame, x, y`` (px@1280), the
    :data:`pool.FEATURE_COLS`, ``label, pos_tier, neg_kind, contam, clip_split``."""
    from footy_track import splits  # noqa: PLC0415

    inv = json.loads((exp / "inventory.json").read_text())
    gt, noball, notbc = load_gt(data)
    verdicts = load_verdicts(data)
    rf = contaminated_frames_by_clip(data)
    eval_frames: dict[str, set[int]] = defaultdict(set)
    for c, d in gt.items():
        eval_frames[c] |= set(d)
    for c, s in noball.items():
        eval_frames[c] |= set(s)
    for c, d in verdicts.items():
        eval_frames[c] |= {k[0] for k in d}
    rows: list[dict] = []
    for clip in sorted(eval_frames):
        if clip not in inv or (only_clips and clip not in only_clips):
            continue
        e = inv[clip]
        path = (
            e.get("rtdetr_machine_labels")
            or e.get("rtdetr_sam3_dir")
            or (exp / "rtdetr_cpu" / f"{clip}.jsonl")
        )
        path = Path(path)
        if not path.is_file():
            log(f"{clip}: no RT-DETR output, skipped")
            continue
        balls, persons, processed = read_machine_labels(path)
        mpath = exp / "motion" / f"{clip}.npz"
        mot = motion_mod.load_motion(mpath) if mpath.is_file() else None
        match = match_of(clip)
        phases: dict[int, dict[int, pool_mod.TrackSample]] = {}
        for f in sorted(eval_frames[clip]):
            if f in notbc.get(clip, ()):
                continue
            ph = f % stride
            if (
                ph not in phases
            ):  # one tracker run per grid phase of this clip, over the processed samples of that phase
                frames = sorted(g for g in processed if g % stride == ph)
                phases[ph] = pool_mod.track_samples(
                    track_mod.run_tracker(
                        _clip_candidates(balls, persons, frames, stride), stride=stride
                    )
                )
            rt = balls.get(f, np.zeros((0, 5), np.float32))
            fi = pool_mod.FrameInputs(
                rt=rt,
                persons=persons.get(f, np.zeros((0, 4), np.float32)),
                motion=None if mot is None else mot.at(f),
                prev=_pts3(balls, processed, f - stride),
                next=_pts3(balls, processed, f + stride),
                trk=phases[ph].get(f),
            )  # fmt: skip
            nodes = [n for n in pool_mod.frame_nodes(fi, stride) if not n["is_gap"]]
            if not nodes:
                continue
            pos = [(x, y) for x, y, _ in gt.get(clip, {}).get(f, [])]
            tier_of = {
                (round(x, 1), round(y, 1)): t for x, y, t in gt.get(clip, {}).get(f, [])
            }
            posv, negv = [], []
            for (ff, _, _), r in verdicts.get(clip, {}).items():
                if ff != f:
                    continue
                if r["verdict"] == "ball":
                    posv.append(r["xy"])
                elif r["verdict"] == "corrected" and r["cxy"]:
                    posv.append(r["cxy"])
                elif r["verdict"] == "not_ball":
                    negv.append(r["xy"])
            _label_nodes(
                nodes,
                pos,
                posv,
                negv,
                bool(pos or posv),
                tier_of,
                f in noball.get(clip, ()),
            )
            contam = f in rf.get(clip, ())
            csplit = str(splits.split_of(splits.match_key_of(clip) or match, clip))
            for n in nodes:
                rows.append({**{k: v for k, v in n.items() if k not in ("is_gap", "sources", "source_scores", "plausible", "track_id")},
                             "match": match, "clip": clip, "frame": f, "contam": contam, "clip_split": csplit})  # fmt: skip
        log(f"{clip}: {len(rows)} nodes so far")
    return pd.DataFrame(rows)


def _pts3(
    balls: dict[int, np.ndarray], processed: set[int], frame: int
) -> np.ndarray | None:
    """RT-DETR boxes (cx, cy, conf) of a neighbouring sample; None when that frame was not processed."""
    if frame not in processed:
        return None
    b = balls.get(frame)
    return np.zeros((0, 3), np.float32) if b is None else b[:, [0, 1, 4]]
