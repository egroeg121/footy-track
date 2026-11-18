"""The pool scorer: P(node is the real ball) from features available in production. Train / evaluate / save / load / apply.

Model: ``HistGradientBoostingClassifier`` (depth 3, 250 iterations, lr 0.05, l2 2, min leaf 25, seed 0), the stacker of the
experiment (``experiments/ball_candidate_pool``, README "(c) the scoring model"). It is retrained on **only the features the
production pipeline has**: RT-DETR (score, rank within the frame), person proximity, neighbouring-sample continuity,
the motion source and the tracker's answer (:data:`FEATURES`). No YOLO, SAM or coverage flag: in the experiment coverage
identified the match, so a model that sees it cannot generalise to a new one (``core_rt`` / ``rt_plus_motion`` were the safe
variants; this is ``rt_plus_motion`` plus tracker features, at the stride-5 grid).

Shadow / ranking mode. ``p_real`` orders cards (U / M / N / R buckets and the diversity ranking). It is NOT an auto-accept gate:
the experiment's threshold fitted on two matches gave 0.889 precision on the third, and no scorer reliably hit 0.98 on an
unseen match. Auto-accept is decided by Learn-then-Test on the A-bucket audits (``ball_mining.threshold``), which records
``scorer_version`` so audits from different scorer versions are never mixed.

Versioned artifact ``<root>/<scorer_version>/``::

    model.joblib   {"hgb", "solver_lr", "features"}  (scikit-learn pickle; the manifest records the version)
    features.json  the ordered feature list
    manifest.json  training data (matches, clips, counts, digest), parameters, code sha, mode, leakage policy, LOMO metrics
    lomo.json      the leave-one-match-out metrics alone

``<root>/LATEST`` names the newest version. Training matches are filtered through ``footy_track.splits`` (fail closed): a
heldout match never trains the scorer (design §4, rule zero), but it can be a leave-one-match-out TEST fold.

``solver_post``: the bidirectional tracker has no posterior, so a small logistic regression on the tracker's evidence (member,
fwd/bwd agreement, track support and length) is fitted on the same training rows and stored with the model; it is the
calibrated ``P(real | tracker evidence only)`` the pool's ``solver_post`` column promises.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = "/mnt/storage/models/ball_scorer"
MODE = "shadow"  # ranking only; never the auto-accept gate
FEATURES: tuple[str, ...] = (
    # RT-DETR
    "rt_score", "rt_rank", "rt_n",
    # position / people
    "xn", "yn", "d_foot", "in_person", "d_head",
    # continuity with the neighbouring samples (RT-DETR boxes at t -+ stride)
    "d_prev_hc", "d_next_hc", "nb_plaus_hc", "static_hc", "d_prev", "d_next", "d_min_nb", "nb_plaus", "static_nb",
    # motion source
    "motion_score", "motion_rank", "motion_n", "mot_contrast", "mot_area", "mot_fill", "mot_asp", "mot_dens", "mot_inperson",
    "n_src",
    # bidirectional tracker
    "trk_state", "trk_member", "trk_dist", "trk_agree", "trk_support", "trk_len",
)  # fmt: skip
SOLVER_FEATURES: tuple[str, ...] = (
    "trk_member",
    "trk_agree",
    "trk_support",
    "trk_len",
    "trk_state",
)
HGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 250,
    "l2_regularization": 2.0,
    "min_samples_leaf": 25,
    "random_state": 0,
}
LABEL_COL = "label"
FORBIDDEN_PREFIXES = (
    "yolo",
    "sam",
    "b1_",
    "cov",
    "n_src_cov",
)  # features that must never be in the list (coverage identifies the match)


def check_features(features: tuple[str, ...] | list[str]) -> None:
    bad = [
        f for f in features if f.startswith(FORBIDDEN_PREFIXES) or f.endswith("_cov")
    ]
    if bad:
        raise ValueError(
            f"features not available in production / identify the match: {bad}"
        )


def register_split_of(match: str) -> str:
    """Match-level split from the register: ``train``, ``eval_heldout_match`` or ``unregistered`` (fail closed: never trains)."""
    from footy_track import splits  # noqa: PLC0415

    return (
        "unregistered"
        if splits.unregistered_match(match, None)
        else str(splits.split_of(match, None))
    )


def make_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    return HistGradientBoostingClassifier(**HGB_PARAMS)


def _solver_matrix(df: pd.DataFrame) -> np.ndarray:
    g = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(float)  # noqa: E731
    return np.stack(
        [
            g("trk_member"),
            g("trk_agree"),
            np.log1p(g("trk_support")),
            np.log1p(g("trk_len")),
            (g("trk_state") == 0).astype(float),
        ],
        1,
    )


def fit_solver(df: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    return LogisticRegression(C=1.0, max_iter=2000).fit(
        _solver_matrix(df), df[LABEL_COL].to_numpy(int)
    )


# ---------------------------------------------------------------------------------------------- metrics
def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(y: np.ndarray, s: np.ndarray, *, prob: bool = True) -> dict:
    from sklearn.metrics import (  # noqa: PLC0415
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
    )

    out = {"n": int(len(y)), "pos": int(y.sum())}
    if len(np.unique(y)) < 2:
        return {**out, "auroc": None, "ap": None}
    out["auroc"] = float(roc_auc_score(y, s))
    out["ap"] = float(average_precision_score(y, s))
    prec, rec, _ = precision_recall_curve(y, s)
    for r in (0.5, 0.8):
        ok = rec >= r
        out[f"p_at_r{int(r * 100)}"] = float(prec[ok].max()) if ok.any() else None
    if prob:
        out["ece"] = ece(y, np.clip(s, 0, 1))
    return out


def lomo(
    table: pd.DataFrame,
    features: tuple[str, ...] | list[str],
    *,
    train_matches: set[str],
    eval_matches: list[str] | None = None,
    respect_clip_split: bool = True,
) -> dict:
    """Leave-one-match-out. For every match ``m`` in ``eval_matches`` (default: all matches in ``table``): fit on the clean
    nodes of ``train_matches - {m}`` (only clips whose own split is ``train`` when ``respect_clip_split``: in-match eval
    clips never train) and score every clean node of ``m``. A match in ``eval_matches`` that is not in ``train_matches`` (a
    heldout match) is a pure TEST fold. ``table`` needs ``match, label, contam``, ``clip_split`` (optional) and the features.

    Reported per match and pooled out-of-fold, on E1 (every labelled node) and E2 (nodes with an RT-DETR candidate, the
    apples-to-apples comparison with RT-DETR confidence), each next to the RT-DETR confidence baseline."""
    t = table[(table[LABEL_COL] >= 0) & (~table["contam"].astype(bool))]
    matches = eval_matches or sorted(t["match"].unique())
    oof = pd.Series(np.nan, index=t.index)
    folds = {}
    for m in matches:
        tr = t[t["match"].isin(train_matches - {m})]
        if respect_clip_split and "clip_split" in t:
            tr = tr[tr["clip_split"] == "train"]
        te = t[t["match"] == m]
        if te.empty:
            continue
        if tr[LABEL_COL].nunique() < 2:
            folds[m] = {"skipped": "no training data"}
            continue
        clf = make_hgb().fit(tr[list(features)].astype(float), tr[LABEL_COL])
        oof[te.index] = clf.predict_proba(te[list(features)].astype(float))[:, 1]
        folds[m] = {"trained_on": sorted(tr["match"].unique()), "n_train": int(len(tr))}
    d = t[oof.notna()]
    res: dict = {"folds": folds, "per_match": {}, "pooled": {}}
    for name, sel in (("E1", slice(None)), ("E2", d["rt_n"].fillna(0).to_numpy() > 0)):
        dd = d[sel] if not isinstance(sel, slice) else d
        pooled = {
            "model": metrics(
                dd[LABEL_COL].to_numpy(int), oof[dd.index].to_numpy(float)
            ),
            "rtdetr_conf": metrics(
                dd[LABEL_COL].to_numpy(int), dd["rt_score"].fillna(0.0).to_numpy(float)
            ),
        }
        res["pooled"][name] = pooled
        for m, g in dd.groupby("match"):
            res["per_match"].setdefault(m, {})[name] = {
                "model": metrics(
                    g[LABEL_COL].to_numpy(int), oof[g.index].to_numpy(float), prob=False
                ),
                "rtdetr_conf": metrics(
                    g[LABEL_COL].to_numpy(int),
                    g["rt_score"].fillna(0.0).to_numpy(float),
                    prob=False,
                ),
            }
    return res


# ---------------------------------------------------------------------------------------------- train / save / load
def table_digest(df: pd.DataFrame, features: tuple[str, ...] | list[str]) -> str:
    h = hashlib.sha256()
    cols = ["match", "clip", "frame", LABEL_COL, *features]
    h.update(
        pd.util.hash_pandas_object(df[cols].astype(str), index=False)
        .to_numpy()
        .tobytes()
    )
    return h.hexdigest()[:16]


def version_name(
    digest: str, features: tuple[str, ...] | list[str], date: str | None = None
) -> str:
    key = hashlib.sha1(
        (
            digest + json.dumps(list(features)) + json.dumps(HGB_PARAMS, sort_keys=True)
        ).encode()
    ).hexdigest()[:8]
    return f"hgb-rtm-{date or time.strftime('%Y%m%d')}-{key}"


@dataclass
class Scorer:
    version: str
    features: tuple[str, ...]
    hgb: object
    solver_lr: object
    manifest: dict

    def apply(self, nodes: pd.DataFrame) -> pd.DataFrame:
        """Return ``nodes`` with ``p_real``, ``solver_post`` and ``scorer_version`` filled in.

        Detection nodes get the model's probability. ``tracker_gap`` nodes carry no detector evidence, so their ``p_real`` is the
        gap's plausibility (their one source score, see ``track.gap_plausibility``): the miner's M bucket ranks gaps by it."""
        df = nodes.copy()
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"nodes lack scorer features {missing}")
        gap = (
            df["gap_frame"]
            .map(lambda v: v is True or (isinstance(v, (bool, np.bool_)) and bool(v)))
            .to_numpy(bool)
        )
        p = np.zeros(len(df), np.float64)
        sp = np.full(len(df), np.nan)
        if (~gap).any():
            x = df.loc[~gap, list(self.features)].astype(float)
            p[~gap] = self.hgb.predict_proba(x)[:, 1]
            sp[~gap] = self.solver_lr.predict_proba(_solver_matrix(df.loc[~gap]))[:, 1]
        if gap.any():
            p[gap] = (
                df.loc[gap, "source_scores"]
                .map(lambda v: float(v[0]) if len(v) else 0.0)
                .to_numpy(float)
            )
        df["p_real"] = np.clip(p, 0.0, 1.0).astype(np.float32)
        df["solver_post"] = sp.astype(np.float32)
        df["scorer_version"] = self.version
        return df


def train(
    table: pd.DataFrame,
    *,
    features: tuple[str, ...] = FEATURES,
    split_of=None,
    extra_train_matches: set[str] | None = None,
    code_sha: str | None = None,
    source: dict | None = None,
) -> tuple[Scorer, dict]:
    """Fit the scorer on the clean labelled nodes of the TRAIN matches of ``table`` and compute its LOMO metrics.

    ``split_of(match) -> "train" | "heldout" | ...`` decides which matches may train (default: the match split register,
    ``footy_track.match_splits``; fail closed when a match is unknown). A heldout match is evaluated as a test fold only.
    ``extra_train_matches`` forces matches in (recorded in the manifest as an override; rule zero says do not)."""
    check_features(features)
    split_of = split_of or register_split_of
    matches = sorted(table["match"].unique())
    kind = {m: split_of(m) for m in matches}
    train_matches = {m for m, k in kind.items() if k == "train"} | set(
        extra_train_matches or ()
    )
    clean = table[(table[LABEL_COL] >= 0) & (~table["contam"].astype(bool))]
    train_matches = {m for m in train_matches if m in set(clean["match"])}
    if not train_matches:
        raise ValueError(
            f"no train-register match with clean labelled nodes in {matches} ({kind})"
        )
    tr = clean[clean["match"].isin(train_matches)]
    if "clip_split" in tr:
        tr = tr[
            tr["clip_split"] == "train"
        ]  # clips of a train match that are in-match eval never train
    if tr[LABEL_COL].nunique() < 2:
        raise ValueError("training nodes have a single class")
    report = lomo(table, features, train_matches=train_matches)
    # the same folds with heldout matches ALLOWED to train (the experiment's regime): reference only, never shipped
    ref = (
        lomo(
            table, features, train_matches=set(clean["match"]), respect_clip_split=False
        )
        if set(clean["match"]) - train_matches
        else None
    )
    hgb = make_hgb().fit(tr[list(features)].astype(float), tr[LABEL_COL].to_numpy(int))
    solver = fit_solver(tr)
    digest = table_digest(tr, features)
    version = version_name(digest, features)
    import sklearn  # noqa: PLC0415

    manifest = {
        "scorer_version": version, "mode": MODE, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": "HistGradientBoostingClassifier",
        "hgb_params": HGB_PARAMS, "features": list(features), "sklearn": sklearn.__version__, "code_sha": code_sha,
        "training": {
            "digest": digest, "train_matches": sorted(train_matches), "split_of": kind,
            "heldout_trained": sorted(set(extra_train_matches or ()) & {m for m, k in kind.items() if k != "train"}),
            "n_nodes": int(len(tr)), "n_pos": int(tr[LABEL_COL].sum()), "n_clips": int(tr["clip"].nunique()),
            "per_match": {m: {"nodes": int(len(g)), "pos": int(g[LABEL_COL].sum()), "clips": int(g["clip"].nunique())} for m, g in tr.groupby("match")},
            "clean_only": True, "excluded_contaminated_nodes": int((table["contam"].astype(bool) & (table[LABEL_COL] >= 0)).sum()),
            "source": source or {},
        },
        "leakage_policy": "heldout matches (match_splits.json) never train the scorer; they are leave-one-match-out TEST folds only",
        "lomo": report,
        "lomo_reference_heldout_allowed_to_train": ref,
        "notes": [
            "shadow/ranking mode: p_real orders cards; auto-accept is decided by Learn-then-Test on the A-bucket audits, not by a fixed scorer threshold",
            "no YOLO / SAM / coverage features: coverage identified the match in the experiment",
        ],
    }  # fmt: skip
    return Scorer(version, tuple(features), hgb, solver, manifest), manifest


def save(
    scorer: Scorer, root: str | Path = DEFAULT_ROOT, *, make_latest: bool = True
) -> Path:
    import joblib  # noqa: PLC0415

    d = Path(root) / scorer.version
    if d.exists():
        raise FileExistsError(f"{d} exists: scorer versions are immutable")
    tmp = d.with_name(f".{d.name}.tmp-{os.getpid()}")
    tmp.mkdir(parents=True)
    joblib.dump(
        {
            "hgb": scorer.hgb,
            "solver_lr": scorer.solver_lr,
            "features": list(scorer.features),
        },
        tmp / "model.joblib",
    )
    (tmp / "features.json").write_text(json.dumps(list(scorer.features), indent=1))
    (tmp / "manifest.json").write_text(
        json.dumps(scorer.manifest, indent=1, sort_keys=True)
    )
    (tmp / "lomo.json").write_text(
        json.dumps(scorer.manifest["lomo"], indent=1, sort_keys=True)
    )
    os.replace(tmp, d)
    if make_latest:
        p = Path(root) / "LATEST"
        t = p.with_name(f".LATEST.tmp-{os.getpid()}")
        t.write_text(scorer.version + "\n")
        os.replace(t, p)
    return d


def latest_version(root: str | Path = DEFAULT_ROOT) -> str | None:
    try:
        v = (Path(root) / "LATEST").read_text().strip()
    except OSError:
        return None
    return v if v and (Path(root) / v / "model.joblib").is_file() else None


def load(version: str = "latest", root: str | Path = DEFAULT_ROOT) -> Scorer:
    import joblib  # noqa: PLC0415

    if version in ("", "latest"):
        v = latest_version(root)
        if not v:
            raise FileNotFoundError(f"no scorer under {root} (LATEST missing or empty)")
        version = v
    d = Path(root) / version
    blob = joblib.load(d / "model.joblib")
    manifest = json.loads((d / "manifest.json").read_text())
    feats = tuple(json.loads((d / "features.json").read_text()))
    if list(feats) != list(blob["features"]):
        raise ValueError(f"{d}: features.json disagrees with the model")
    check_features(feats)
    return Scorer(version, feats, blob["hgb"], blob["solver_lr"], manifest)
