"""Fit and evaluate the per-shot camera-type classifier (:mod:`footy_track.calibration.shots`) on the human labels.

    OMP_NUM_THREADS=2 nice PYTHONPATH=src python -m footy_track.scripts.shots_eval --cache DIR [--fit] [--sheets CLIP ...]

Label sources (all read-only):

* **Roboflow "broadcast frame" dataset** (``classifier_dataset/roboflow_dataset_*``, ``Yes`` / ``No`` folders): single frames
  a human marked as the main broadcast view or not, with real negatives (close-ups, crowd, graphics, high / reverse cameras).
  The only source with negatives to speak of; the model is fitted on it alone.
* **Calibration labels** under the calibration GT root, an independent test set (never used to fit): frames of
  ``calibration_gt/<clip>.json`` and proposals with status ``accepted`` are main (human); ``frame_type/<clip>.json`` and
  ``not_broadcast`` proposals are not main (human; none exist yet); Claude's ``view.main_camera_wide`` on a proposal is a weak
  label in either direction (a human label wins over it; disagreements are listed as ``conflicts``).

Frame features are cached under ``--cache`` (``<clip>.feat.json``, ``roboflow.feat.json``) together with the clip's camera motion
(``<clip>.motion.npz``, shot cuts); a cache entry missing a current feature is recomputed. Cross-validation is leave-one-match-out
(the match is the clip id's ``arsenal_<opponent>`` prefix / the Roboflow file-name prefix).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import shots as sh
from footy_track.calibration import temporal as tp
from footy_track.calibration.clip_calibration import ClipCalibration
from footy_track.calibration.frame_type import not_broadcast_frames
from footy_track.calibration.prelabel.pipeline import find_video
from footy_track.calibration.prelabel.proposals import (
    list_clips_with_proposals,
    read_proposals,
)
from footy_track.clip_id import canonical_clip_id
from footy_track.scripts.calibrate_clip import (
    BOXES_DIR,
    CLIPS_DIR,
    GT_DIR,
    find_boxes_path,
)

ROBOFLOW_DIR = Path("/mnt/storage/footy_data/classifier_dataset")
ROBOFLOW_MATCH = (("Arsenal-Aston-Villa", "arsenal_astonvilla"), ("arsenal_mancity", "arsenal_mancity"), ("arsenal_norwich", "arsenal_norwich"),
                  ("Bournemouth", "arsenal_bournemouth_1st"))

# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def match_of(clip: str) -> str:
    """``arsenal_mancity_seg010`` -> ``arsenal_mancity``; ``arsenal_bournemouth_1st_seg005`` -> ``arsenal_bournemouth_1st``."""
    return canonical_clip_id(clip).rsplit("_", 1)[0]


def load_labels(gt_dir: Path = GT_DIR) -> dict[str, dict[int, dict]]:
    """``{clip: {frame: {"main": bool, "tier": "human" | "claude", "src": str, "conflict"?: str}}}`` for every clip with a label."""
    gt_dir = Path(gt_dir)
    out: dict[str, dict[int, dict]] = defaultdict(dict)

    def put(clip, f, main, tier, src):
        cur = out[clip].get(f)
        if cur is not None and cur["tier"] == "human" and tier == "claude":
            if cur["main"] != main:
                cur["conflict"] = src
            return
        if cur is not None and cur["tier"] == "claude" and tier == "human" and cur["main"] != main:
            out[clip][f] = {"main": main, "tier": tier, "src": src, "conflict": cur["src"]}
            return
        out[clip][f] = {"main": main, "tier": tier, "src": src, **({"conflict": cur["conflict"]} if cur and "conflict" in cur else {})}

    clips = set(list_clips_with_proposals(gt_dir))
    gdir = gt_dir / "calibration_gt"
    clips |= {p.stem for p in gdir.glob("*.json")} if gdir.exists() else set()
    fdir = gt_dir / "frame_type"
    clips |= {p.stem for p in fdir.glob("*.json")} if fdir.exists() else set()
    for clip in sorted(clips):
        runs = read_proposals(gt_dir, clip).get("runs", {}).values()
        for run in runs:  # weakest first so a human label overrides in `put`
            for k, fr in run.get("frames", {}).items():
                v = fr.get("view") or {}
                if "main_camera_wide" in v:
                    put(clip, int(k), bool(v["main_camera_wide"]), "claude", "claude_view")
        for run in runs:
            for k, fr in run.get("frames", {}).items():
                if fr.get("status") == "accepted":
                    put(clip, int(k), True, "human", "accepted")
                elif fr.get("status") == "not_broadcast":
                    put(clip, int(k), False, "human", "proposal_not_broadcast")
        gp = gdir / f"{clip}.json"
        if gp.exists():
            for k in json.loads(gp.read_text()).get("frames", {}):
                put(clip, int(k), True, "human", "gt")
        for f in not_broadcast_frames(gt_dir, clip):
            put(clip, int(f), False, "human", "frame_type")
    return dict(out)


def roboflow_items(root: Path = ROBOFLOW_DIR) -> list[dict]:
    """Labelled frames of every ``roboflow_dataset_*`` (train and test), de-duplicated by source frame name (a frame labelled
    both ways in different versions is dropped). ``{"key", "path", "y", "match"}``."""
    seen: dict[str, dict] = {}
    bad: set[str] = set()
    for p in sorted(Path(root).glob("roboflow_dataset_*/*/*/*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or p.parent.name not in {"Yes", "No"}:
            continue
        key = p.name.split("_jpg")[0].split("_png")[0]
        y = int(p.parent.name == "Yes")
        match = next((m for pre, m in ROBOFLOW_MATCH if key.startswith(pre)), "other")
        if key in seen and seen[key]["y"] != y:
            bad.add(key)
        seen.setdefault(key, {"key": key, "path": p, "y": y, "match": match})
    return [v for k, v in sorted(seen.items()) if k not in bad]


# --------------------------------------------------------------------------
# caches
# --------------------------------------------------------------------------


def _complete(d: dict) -> bool:
    return all(k in d for k in sh.FRAME_FEATURES)


def clip_data(clip: str, labelled: list[int], cache: Path, sample_every: int, extra_motion: list[Path] = ()):
    """``(video, boxes, motion, feats, timing)`` of a clip; motion and frame features are computed once and cached."""
    cid = canonical_clip_id(clip)
    video = find_video(CLIPS_DIR, clip)
    bp = find_boxes_path(clip, BOXES_DIR)
    boxes = tp.load_person_boxes(bp) if bp else None
    mp = cache / f"{cid}.motion.npz"
    t0 = time.perf_counter()
    if mp.exists():
        motion = tp.ClipMotion.load(mp)
    else:
        motion = next((tp.ClipMotion.load(p) for p in extra_motion if p.exists()), None)
        if motion is None:
            motion = tp.clip_motion(video, boxes=boxes, max_gap=1, scale=320, mask_mode="all")
        motion.save(mp)
    t_motion = time.perf_counter() - t0
    fp = cache / f"{cid}.feat.json"
    feats = {int(k): v for k, v in json.loads(fp.read_text()).items()} if fp.exists() else {}
    feats = {k: v for k, v in feats.items() if _complete(v)}
    want = sorted((set(sh.sample_frames(motion, sample_every)) | set(labelled)) - set(feats))
    t0 = time.perf_counter()
    if want:
        feats.update(sh.extract_frame_features(video, want, boxes))
        fp.write_text(json.dumps(feats))
    t_feat = time.perf_counter() - t0
    return video, boxes, motion, feats, {"motion_s": t_motion, "feat_s": t_feat, "n_feat": len(want), "frames": motion.n}


def roboflow_features(items: list[dict], cache: Path) -> dict[str, dict]:
    fp = cache / "roboflow.feat.json"
    feats = json.loads(fp.read_text()) if fp.exists() else {}
    feats = {k: v for k, v in feats.items() if _complete(v)}
    todo = [it for it in items if it["key"] not in feats]
    for it in todo:
        img = cv2.imread(str(it["path"]))
        small = cv2.resize(img, (sh.DEFAULT_WORK_WIDTH, int(img.shape[0] * sh.DEFAULT_WORK_WIDTH / img.shape[1])), interpolation=cv2.INTER_AREA)
        feats[it["key"]] = sh.frame_features(small, None)
    if todo:
        fp.write_text(json.dumps(feats))
    return feats


# --------------------------------------------------------------------------
# logistic regression (numpy)
# --------------------------------------------------------------------------


def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1.0, iters: int = 50):
    """Weighted L2-regularised logistic regression by Newton steps; returns (coef, bias)."""
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(d + 1)
    reg = np.diag([l2] * d + [0.0])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-A @ beta))
        g = A.T @ (w * (p - y)) + reg @ beta
        Hs = A.T @ (A * (w * p * (1 - p))[:, None]) + reg + 1e-6 * np.eye(d + 1)
        step = np.linalg.solve(Hs, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta[:d], float(beta[d])


def fit_model(rows: list[dict], names: tuple[str, ...], l2: float = 1.0) -> sh.ShotModel:
    """Fit a :class:`ShotModel` on rows ``{"f": frame features, "y": 0/1}`` with the two classes balanced."""
    X = np.array([[r["f"][n] for n in names] for r in rows], float)
    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(X, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    Z = np.where(np.isfinite(X), np.clip((X - mean) / std, -sh.Z_CLIP, sh.Z_CLIP), 0.0)
    y = np.array([r["y"] for r in rows], float)
    w = np.where(y == 1, 0.5 / max((y == 1).sum(), 1), 0.5 / max((y == 0).sum(), 1)) * len(y)
    coef, bias = fit_logistic(Z, y, w, l2)
    return sh.ShotModel(tuple(names), mean.tolist(), std.tolist(), coef.tolist(), bias)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def confusion(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp_, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    fn, tn = int((~pred & truth).sum()), int((~pred & ~truth).sum())
    return {"tp": tp_, "fp": fp, "fn": fn, "tn": tn, "precision": tp_ / (tp_ + fp) if tp_ + fp else float("nan"),
            "recall": tp_ / (tp_ + fn) if tp_ + fn else float("nan"), "specificity": tn / (tn + fp) if tn + fp else float("nan")}


def lomo_models(rows: list[dict], names: tuple[str, ...], l2: float) -> dict[str, sh.ShotModel]:
    """``{match: model fitted on the rows of every other match}``."""
    return {m: fit_model([r for r in rows if r["match"] != m], names, l2) for m in sorted({r["match"] for r in rows})}


def _by_match(rows: list[dict], pred: np.ndarray, y: np.ndarray) -> dict:
    out = {"all": confusion(pred, y)}
    for m in sorted({r["match"] for r in rows}):
        sel = np.array([r["match"] == m for r in rows])
        out[m] = confusion(pred[sel], y[sel])
    return out


def roboflow_report(rows: list[dict], names: tuple[str, ...], l2: float):
    """Leave-one-match-out frame-level confusion of the logistic model and the threshold rule on the Roboflow frames;
    also returns the held-out scores."""
    y = np.array([r["y"] for r in rows], bool)
    models = lomo_models(rows, names, l2)
    p = np.array([models[r["match"]].p_main(r["f"]) for r in rows])
    rule = np.array([sh.rule_p_main(r["f"]) for r in rows])
    return {"logistic_lomo": _by_match(rows, p >= 0.5, y), "rule": _by_match(rows, rule >= 0.5, y)}, p


def calibration_rows(clip: str, labels: dict[int, dict], motion: tp.ClipMotion, feats: dict[int, dict]) -> list[dict]:
    """One row per labelled frame: its own features, its shot's sample features and its label."""
    shot = motion.shot
    rows = []
    for f, lab in sorted(labels.items()):
        if f not in feats or not motion.start <= f < motion.start + motion.n:
            continue
        sid = int(shot[f - motion.start])
        samples = [feats[g] for g in sorted(feats) if motion.start <= g < motion.start + motion.n and int(shot[g - motion.start]) == sid]
        rows.append({"clip": clip, "match": match_of(clip), "frame": f, "shot_id": sid, "f": feats[f], "samples": samples, "y": int(lab["main"]),
                     "tier": lab["tier"], "src": lab["src"], "conflict": lab.get("conflict")})
    return rows


def _shot_score(r: dict, fn) -> float:
    return float(np.median([fn(s) for s in r["samples"]]))


def calibration_report(rows: list[dict], models: dict[str, sh.ShotModel]) -> dict:
    """Frame-level and shot-level (median over the shot's samples) confusion on the calibration labels, by label tier and match."""
    y = np.array([r["y"] for r in rows], bool)
    tiers = {t: np.array([r["tier"] == t for r in rows]) for t in ("human", "claude")}
    out = {}
    for level in ("frame", "shot"):
        for name in ("logistic", "rule"):
            def fn(r, name=name):
                return models[r["match"]].p_main if name == "logistic" else sh.rule_p_main

            score = [fn(r)(r["f"]) if level == "frame" else _shot_score(r, fn(r)) for r in rows]
            pred = np.array(score) >= 0.5
            out[f"{level}_{name}"] = {**_by_match(rows, pred, y), **{f"tier_{t}": confusion(pred[m], y[m]) for t, m in tiers.items()}}
    return out


def check_calibrated(parquet: Path, cache: Path, sample_every: int, model: sh.ShotModel | None) -> list[dict]:
    """Per shot of a ``calibrate_clip`` run: the classified type and how much of the shot the run calibrated (``ok`` frames)."""
    cal = ClipCalibration.load(parquet)
    cid = parquet.name.split(".")[0]
    video, _, motion, feats, _ = clip_data(cid, [], cache, sample_every)
    shots = sh.classify_shots(video, motion, None, sample_every=sample_every, model=model, features=feats, calibration=cal)
    t = cal.table.set_index("frame")
    out = []
    for s in shots:
        seg = t.loc[s["start"]:s["end"]]
        out.append({"shot_id": s["shot_id"], "frames": [s["start"], s["end"]], "type": s["type"], "p_main": round(s["p_main"], 3),
                    "calibrated_ok_frac": round(float(seg["ok"].mean()), 3), "keyframes": [int(f) for f in seg.index[seg["source"] == "keyframe"]],
                    "keyframe_candidate": s["keyframe_candidate"], "keyframe_line_cost": s["keyframe_line_cost"]})
    return out


# --------------------------------------------------------------------------
# contact sheets
# --------------------------------------------------------------------------


def _grid(tiles: list[np.ndarray], cols: int, tile_w: int) -> np.ndarray:
    th = max(t.shape[0] for t in tiles)
    rows_ = []
    for i in range(0, len(tiles), cols):
        r = [np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0))) for t in tiles[i:i + cols]]
        r += [np.zeros((th, tile_w, 3), np.uint8)] * (cols - len(r))
        rows_.append(np.hstack(r))
    return np.vstack(rows_)


def _annotate(img: np.ndarray, txt: str, col: tuple[int, int, int], tile_w: int) -> np.ndarray:
    img = cv2.resize(img, (tile_w, int(tile_w * img.shape[0] / img.shape[1])))
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, 4)
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


TYPE_COLOUR = {"main": (60, 200, 60), "closeup": (60, 60, 230), "other": (0, 160, 255)}


def contact_sheet(video: Path, shots: list[dict], out: Path, cols: int = 4, tile_w: int = 360, labels: dict[int, dict] | None = None) -> None:
    """One tile per shot (its keyframe candidate, else its middle), annotated with frames, type and p_main; dots mark labelled
    frames in the shot (green main / red not main, dim = Claude's weak label)."""
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for s in shots:
        f = s["keyframe_candidate"] if s["keyframe_candidate"] is not None else (s["start"] + s["end"]) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            img = np.zeros((int(tile_w * 9 / 16), tile_w, 3), np.uint8)
        img = _annotate(img, f"#{s['shot_id']} {s['start']}-{s['end']} {s['type']} p={s['p_main']:.2f} f{f}", TYPE_COLOUR[s["type"]], tile_w)
        for j, v in enumerate([v for k, v in (labels or {}).items() if s["start"] <= k <= s["end"]][:6]):
            c = (0, 255, 0) if v["main"] else (0, 0, 255)
            c = tuple(int(x * 0.5) for x in c) if v["tier"] == "claude" else c
            cv2.circle(img, (img.shape[1] - 10 - 12 * j, img.shape[0] - 10), 5, c, -1)
        tiles.append(img)
    cap.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), _grid(tiles, cols, tile_w))


def frame_sheet(items: list[dict], preds: list[tuple[float, str]], out: Path, cols: int = 6, tile_w: int = 320) -> None:
    """Tiles of single frames (paths in ``items``) annotated with the human label and the predicted p_main / type."""
    tiles = []
    for it, (p, typ) in zip(items, preds, strict=True):
        img = cv2.imread(str(it["path"]))
        tiles.append(_annotate(img, f"{'Yes' if it['y'] else 'No'} p={p:.2f} {typ}", TYPE_COLOUR[typ], tile_w))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), _grid(tiles, cols, tile_w))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=GT_DIR)
    ap.add_argument("--roboflow", type=Path, default=ROBOFLOW_DIR)
    ap.add_argument("--clips", nargs="*", help="canonical clip ids for the calibration-label test (default: every labelled clip)")
    ap.add_argument("--sample-every", type=int, default=30, help="frame feature sampling of the calibration clips")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--features", default=",".join(sh.IMAGE_FEATURES), help="comma list of frame features of the logistic model")
    ap.add_argument("--fit", action="store_true", help="fit on all Roboflow frames and write calibration/shot_model.json")
    ap.add_argument("--sheets", nargs="*", default=[], help="calibration clips to write per-shot contact sheets for (into --cache/sheets)")
    ap.add_argument("--roboflow-sheet", action="store_true", help="write sheets of Roboflow frames the model gets wrong / right")
    ap.add_argument("--extra-motion", type=Path, nargs="*", default=[], help="existing *.motion.npz to reuse (matched by file name)")
    ap.add_argument("--calibrated", type=Path, nargs="*", default=[], help="<clip>.calibration.parquet of calibrate_clip runs to check")
    ap.add_argument("--calibrated-sample-every", type=int, default=10)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    a.cache.mkdir(parents=True, exist_ok=True)
    names = tuple(a.features.split(","))
    res: dict = {"features": names, "l2": a.l2}

    # 1. Roboflow: leave-one-match-out, then the model on everything
    items = roboflow_items(a.roboflow)
    t0 = time.perf_counter()
    rf = roboflow_features(items, a.cache)
    res["roboflow_feature_s"] = time.perf_counter() - t0
    rrows = [{"f": rf[it["key"]], "y": it["y"], "match": it["match"], "key": it["key"]} for it in items]
    res["roboflow_counts_yes_no"] = {m: [sum(1 for r in rrows if r["match"] == m and r["y"]), sum(1 for r in rrows if r["match"] == m and not r["y"])]
                                     for m in sorted({r["match"] for r in rrows})}
    res["roboflow"], p_rf = roboflow_report(rrows, names, a.l2)
    y_rf = np.array([r["y"] for r in rrows], bool)
    res["roboflow_thresholds"] = {str(t): confusion(p_rf >= t, y_rf) for t in (0.3, 0.5, 0.7, 0.8)}
    res["roboflow_ablation"] = {label: roboflow_report(rrows, sub, a.l2)[0]["logistic_lomo"]["all"] for label, sub in (
        ("grass only", ("grass_frac", "grass_low")), ("grass+lines", ("grass_frac", "grass_low", "line_frac", "line_len", "n_dirs")),
        ("grass+blobs", ("grass_frac", "grass_low", "blob_h", "blob_frac", "n_blob", "blob_med_h")),
        ("grass+lines+blobs", ("grass_frac", "grass_low", "line_frac", "line_len", "n_dirs", "blob_h", "blob_frac", "n_blob", "blob_med_h")))}
    full = fit_model(rrows, names, a.l2)
    full.meta = {"trained_on": "roboflow broadcast-frame Yes/No", "n_frames": len(rrows), "l2": a.l2}
    res["model"] = full.to_json()
    if a.fit:
        full.save()

    # 2. calibration labels (independent test)
    labels = load_labels(a.gt_dir)
    clips = a.clips or sorted(labels)
    extra = {p.name: p for p in a.extra_motion}
    crow: list[dict] = []
    clip_cache = {}
    res["timing"] = {}
    for clip in clips:
        lab = labels.get(clip, {})
        em = [extra[f"{canonical_clip_id(clip)}.motion.npz"]] if f"{canonical_clip_id(clip)}.motion.npz" in extra else []
        video, _, motion, feats, tm = clip_data(clip, sorted(lab), a.cache, a.sample_every, em)
        res["timing"][clip] = tm
        clip_cache[clip] = (video, motion, feats)
        crow += calibration_rows(clip, lab, motion, feats)
        print(f"{clip}: {motion.n} frames, {int(motion.cut.sum())} cuts, {len(lab)} labels, {json.dumps({k: round(v, 1) for k, v in tm.items()})}", flush=True)
    models = lomo_models(rrows, names, a.l2)
    models = {m: models.get(m, full) for m in {r["match"] for r in crow}}  # a match with no Roboflow frames: the full model
    res["calibration_labels"] = calibration_report(crow, models)
    counts = defaultdict(lambda: defaultdict(int))
    for r in crow:
        counts[r["match"]][("main" if r["y"] else "not_main") + "_" + r["tier"]] += 1
    res["calibration_label_counts"] = {m: dict(v) for m, v in counts.items()}
    res["calibration_conflicts"] = [(r["clip"], r["frame"], r["src"], r["conflict"]) for r in crow if r["conflict"]]
    res["calibration_shot_level_errors"] = [
        (r["clip"], r["frame"], r["y"], round(_shot_score(r, models[r["match"]].p_main), 2), r["tier"], r["src"])
        for r in crow if (_shot_score(r, models[r["match"]].p_main) >= 0.5) != bool(r["y"])]

    # 3. checks / sheets
    res["calibrated_check"] = {pq.name: check_calibrated(pq, a.cache, a.calibrated_sample_every, full) for pq in a.calibrated}
    for clip in a.sheets:
        video, motion, feats = clip_cache[clip]
        shots = sh.classify_shots(video, motion, None, sample_every=a.sample_every, model=full, features=feats)
        contact_sheet(video, shots, a.cache / "sheets" / f"{canonical_clip_id(clip)}.png", labels=labels.get(clip))
    if a.roboflow_sheet:
        rng = np.random.default_rng(0)
        for name, want_right in (("wrong", False), ("right", True)):
            idx = [i for i, (r, p) in enumerate(zip(rrows, p_rf, strict=True)) if ((p >= 0.5) == bool(r["y"])) == want_right]
            idx = sorted(rng.choice(idx, size=min(36, len(idx)), replace=False)) if idx else []
            if idx:
                preds = [(float(p_rf[i]), sh.decide(rrows[i]["f"], float(p_rf[i]))[0]) for i in idx]
                frame_sheet([items[i] for i in idx], preds, a.cache / "sheets" / f"roboflow_{name}.png")
    text = json.dumps(res, indent=1, default=float)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    raise SystemExit(main())
