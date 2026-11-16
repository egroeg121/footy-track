"""Per-match pipeline status for the ``/matches`` dashboard.

One row per match, one cell per stage (download, segments, detection, tracking, calibration,
projection). Everything is derived from cheap signals - small manifests, directory listings and
``stat`` calls - and the whole scan is cached for ``CACHE_TTL_S`` seconds. Prefect flow-run links
are best effort (read-only API, short timeout) and silently omitted when the API is unreachable.

Signals per stage (see ``scan`` below):

* download    ``full_matches/manifest.jsonl`` (latest record per post_url), ``work/locks/*.lock``
              (live lock => running, carries the Prefect run id), ``work/<host>/*.part`` (bytes),
              ``links/*.jsonl`` (queued, not started); pre-existing ``<match>/original_video/``.
* segments    ``clips/<prefix>_*.mp4``, ``<match>/broadcast_segments|split_video_broadcast_frames*/*.mp4``;
              total from ``clips/segments.json`` when present.
* detection   ``machine_labels*/<prefix>_*.jsonl`` (per-clip detector output) / segment count.
* tracking    ``tracklets/<prefix>_*.jsonl`` / segment count; plus ``/mnt/storage/footy/runs/*``
              (footy-infer pipeline store) whose ingest source is the match.
* calibration ``ball_gt_marks/calibration_proposals/<prefix>_*.json`` (+ ``calibration_gt`` accepted).
* projection  not implemented.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from base64 import b64encode
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

CACHE_TTL_S = 30
LOCK_STALE_S = 600  # matches footy_matches.LOCK_STALE_S (holder refreshes every 60 s)

#: Clip-name prefixes that differ from the match directory name.
PREFIX_ALIASES: dict[str, list[str]] = {
    "arsenal_mancity": ["mancity"],
    "arsenal_bournmouth_1st_half": ["arsenal_bournemouth_1st", "bournemouth_1st"],
    "arsenal_bournmouth_2nd_half": ["arsenal_bournemouth_2nd", "bournemouth_2nd"],
}

STAGES = ["download", "segments", "detection", "tracking", "calibration", "projection"]


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def data_root() -> Path:
    return _env_path("FOOTY_DATA_DIR", "/mnt/storage/footy_data")


def runs_root() -> Path:
    return _env_path("FOOTY_RUNS_DIR", "/mnt/storage/footy/runs")


def prefect_api() -> str:
    return os.environ.get(
        "MATCHES_PREFECT_API", "http://100.72.126.53:4200/api"
    ).rstrip("/")


def _stage(
    state: str = "none",
    text: str = "",
    pct: float | None = None,
    run_id: str | None = None,
    **extra,
) -> dict:
    return {"state": state, "text": text, "pct": pct, "run_id": run_id, **extra}


def _hsize(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u in ("B", "KB") else f"{n:.1f} {u}"
        n /= 1024
    return f"{n} B"


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _iso_mtime(p: Path) -> str | None:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, UTC).isoformat(
            timespec="seconds"
        )
    except OSError:
        return None


def _names(d: Path) -> list[str]:
    """File names in ``d`` (empty when missing). One scandir, no per-file stat."""
    try:
        with os.scandir(d) as it:
            return [e.name for e in it]
    except OSError:
        return []


def _count(names: list[str], prefixes: list[str], suffix: str) -> int:
    pre = tuple(p + "_" for p in prefixes)
    return sum(1 for n in names if n.startswith(pre) and n.endswith(suffix))


# ------------------------------------------------------------------ Prefect (best effort)
_api_cache: dict = {"t": 0.0, "runs": []}


def _auth_header() -> dict[str, str]:
    s = os.environ.get("PREFECT_API_AUTH_STRING", "")
    if not s:
        try:
            for line in (
                (Path.home() / "deploy" / "prefect.env").read_text().splitlines()
            ):
                if line.startswith("PREFECT_API_AUTH_STRING="):
                    s = line.split("=", 1)[1].strip().strip("\"'")
        except OSError:
            pass
    return {"Authorization": "Basic " + b64encode(s.encode()).decode()} if s else {}


def _flow_runs() -> list[dict]:
    """Recent match-download / footy-infer-pipeline runs; cached, [] on any failure."""
    if time.time() - _api_cache["t"] < CACHE_TTL_S:
        return _api_cache["runs"]
    runs: list[dict] = []
    try:
        body = json.dumps(
            {
                "flows": {"name": {"any_": ["match-download", "footy-infer-pipeline"]}},
                "flow_runs": {
                    "state": {
                        "type": {
                            "any_": [
                                "RUNNING",
                                "COMPLETED",
                                "FAILED",
                                "CRASHED",
                                "CANCELLED",
                            ]
                        }
                    }
                },
                "sort": "START_TIME_DESC",
                "limit": 200,
            }
        ).encode()
        req = urllib.request.Request(
            prefect_api() + "/flow_runs/filter",
            data=body,
            headers={"content-type": "application/json", **_auth_header()},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            runs = [
                {
                    "id": x["id"],
                    "state": (x.get("state") or {}).get("type", ""),
                    "params": x.get("parameters") or {},
                }
                for x in json.load(r)
            ]
    except Exception:  # noqa: BLE001 - links are optional
        runs = []
    _api_cache.update(t=time.time(), runs=runs)
    return runs


def _run_link(run_id: str | None) -> str | None:
    if not run_id:
        return None
    return prefect_api().removesuffix("/api") + "/runs/flow-run/" + run_id


def _find_run(runs: list[dict], **want) -> str | None:
    """Latest run (list is newest first) whose parameters equal all non-empty ``want`` values;
    a COMPLETED/RUNNING run beats a failed one."""
    hits = [
        r for r in runs if any(v and r["params"].get(k) == v for k, v in want.items())
    ]
    good = [r for r in hits if r["state"] in ("COMPLETED", "RUNNING")]
    return (good or hits or [{}])[0].get("id")


# ------------------------------------------------------------------ downloads
_speed: dict[str, tuple[float, int, float | None]] = {}  # key -> (t, bytes, bytes/s)


def _slug(post_url: str) -> str:
    s = post_url.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"-full-match", "", s)


def _abbr_match(slug: str, part_name: str) -> bool:
    """``arsenal-vs-chelsea-2026-27`` vs ``ars.v.che.full...rar.part``."""
    m = re.match(r"([a-z]{3})\.v\.([a-z]{3})\.", part_name.lower())
    sides = re.split(r"-vs-", slug)
    if not m or len(sides) != 2:
        return False
    a = re.sub(r"[^a-z]", "", sides[0])[:3]
    b = re.sub(r"[^a-z]", "", sides[1])[:3]
    return (a, b) == m.groups()


def _manifest(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    try:
        with open(root / "manifest.jsonl") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("post_url"):
                    out[r["post_url"]] = r
    except OSError:
        pass
    return out


def _locks(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    d = root / "work" / "locks"
    now = time.time()
    for n in _names(d):
        p = d / n
        j = _read_json(p)
        try:
            age = now - p.stat().st_mtime
        except OSError:
            continue
        if isinstance(j, dict) and j.get("post_url") and age < LOCK_STALE_S:
            out[j["post_url"]] = j
    return out


def _parts(root: Path) -> list[tuple[str, str, int, float]]:
    """(host, name, bytes, mtime) of every .part file."""
    out = []
    w = root / "work"
    for host in _names(w):
        if host == "locks":
            continue
        for n in _names(w / host):
            if n.endswith(".part"):
                try:
                    st = (w / host / n).stat()
                except OSError:
                    continue
                out.append((host, n, st.st_size, st.st_mtime))
    return out


def _download_rows(root: Path, runs: list[dict]) -> list[dict]:  # noqa: PLR0915
    man, locks, parts = _manifest(root), _locks(root), _parts(root)
    sizes = [
        r["size"] for r in man.values() if r.get("status") == "done" and r.get("size")
    ]
    typical = median(sizes) if sizes else None
    now = time.time()
    rows: list[dict] = []
    seen = set()

    def row(post_url: str, rec: dict | None, link: dict | None) -> dict:
        key = _slug(post_url)
        title = (rec or {}).get("title") or (link or {}).get("title") or key
        mp4 = (rec or {}).get("mp4")
        st = (rec or {}).get("status")
        lock = locks.get(post_url)
        dl: dict
        if st == "done":
            ok = bool(mp4) and os.path.exists(mp4)
            dl = _stage(
                "done" if ok else "failed",
                f"{_hsize(rec.get('size') or 0)} - {rec.get('host') or rec.get('source', '')}".strip(
                    " -"
                )
                + ("" if ok else " (mp4 missing)"),
                1.0 if ok else None,
                when=rec.get("finished"),
            )
            dl["run_id"] = _find_run(
                runs, url=rec.get("url") or rec.get("mega_url"), post_url=post_url
            )
        elif lock or st == "downloading":
            host = (lock or {}).get("host") or (rec or {}).get("host") or ""
            cand = [p for p in parts if p[0] == host and _abbr_match(key, p[1])]
            if not cand:  # fallback: the freshest part file on that host, only if not claimed by another lock
                cand = [
                    p
                    for p in parts
                    if p[0] == host
                    and now - p[3] < LOCK_STALE_S
                    and not any(_abbr_match(_slug(u), p[1]) for u in locks)
                ]
                cand = cand[:1]
            got = cand[0][2] if cand else 0
            speed = None
            if lock:
                t0, b0, s0 = _speed.get(key, (0.0, 0, None))
                if t0 and now - t0 >= 5 and got >= b0:
                    speed = (got - b0) / (now - t0)
                    _speed[key] = (now, got, speed)
                elif t0:
                    speed = s0
                else:
                    _speed[key] = (now, got, None)
            pct = min(got / typical, 0.99) if (typical and got) else None
            txt = f"{host} {_hsize(got)}" + (
                f" (~{pct:.0%} of typical {_hsize(typical)})" if pct is not None else ""
            )
            if speed:
                txt += f" at {speed / 1e6:.1f} MB/s"
            if lock:
                dl = _stage(
                    "running",
                    txt,
                    pct,
                    (lock or {}).get("run_id"),
                    when=(lock or {}).get("claimed"),
                )
            else:
                dl = _stage(
                    "partial",
                    "no live lock (stalled?) " + txt,
                    pct,
                    when=(rec or {}).get("started"),
                )
        elif st in ("partial", "failed"):
            dl = _stage(
                "failed",
                f"{st}: {((rec or {}).get('error') or '')[:100]}",
                when=(rec or {}).get("finished"),
            )
        else:
            dl = _stage("queued", "queued, not started")
        return {
            "key": key,
            "title": title,
            "source": Path(mp4).name
            if mp4
            else post_url.rstrip("/").rsplit("/", 1)[-1],
            "origin": "download",
            "mp4": mp4,
            "post_url": post_url,
            "stages": {"download": dl},
            "prefixes": [key] + ([Path(mp4).stem] if mp4 else []),
        }

    for url, rec in man.items():
        seen.add(url)
        rows.append(row(url, rec, None))
    links: dict[str, dict] = {}
    try:
        for lp in sorted((root / "links").glob("*.jsonl")):
            with open(lp) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("post_url") and r["post_url"] not in seen:
                        links.setdefault(r["post_url"], r)
    except OSError:
        pass
    for url, r in links.items():
        rows.append(row(url, None, r))
    return rows


# ------------------------------------------------------------------ existing matches + downstream stages
def _store_runs(runs_dir: Path) -> list[dict]:
    out = []
    for n in _names(runs_dir):
        ing = _read_json(runs_dir / n / "ingest.json")
        if not isinstance(ing, dict) or not ing.get("source"):
            continue
        clips = _read_json(runs_dir / n / "clips.json") or {}
        n_clips = len(clips.get("clips", []))
        done = sum(
            1
            for c in _names(runs_dir / n / "out")
            if (runs_dir / n / "out" / c / "DONE.json").exists()
        )
        out.append(
            {
                "run_id": n,
                "source": ing["source"],
                "n_clips": n_clips,
                "done": done,
                "finished": (runs_dir / n / "finish.done.json").exists(),
            }
        )
    return out


def _existing_rows(root: Path) -> list[dict]:
    rows = []
    for n in sorted(_names(root)):
        ov = root / n / "original_video"
        vids = [v for v in _names(ov) if not v.startswith(".")]
        if not vids:
            continue
        p = ov / vids[0]
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        rows.append(
            {
                "key": n,
                "title": n,
                "source": vids[0],
                "origin": "existing",
                "mp4": str(p),
                "post_url": None,
                "stages": {
                    "download": _stage(
                        "done",
                        f"original_video, {_hsize(size)}",
                        1.0,
                        when=_iso_mtime(p),
                    )
                },
                "prefixes": [n] + PREFIX_ALIASES.get(n, []),
            }
        )
    return rows


def _progress(state_done: int, total: int | None, noun: str) -> dict:
    if not state_done:
        return _stage("none", "")
    if total and state_done >= total:
        return _stage("done", f"{state_done}/{total} {noun}", 1.0)
    return _stage(
        "partial",
        f"{state_done}/{total} {noun}" if total else f"{state_done} {noun}",
        state_done / total if total else None,
    )


def _fill_downstream(
    rows: list[dict], root: Path, store: list[dict], runs: list[dict]
) -> None:
    clips = _names(root / "clips")
    seg_json = _read_json(root / "clips" / "segments.json") or {}
    seg_stems = Path(str(seg_json.get("source", ""))).stem
    ml_dirs = [
        (d, _names(root / d))
        for d in sorted(_names(root))
        if d.startswith("machine_labels")
    ]
    tracklets = _names(root / "tracklets")
    proposals = _names(root / "ball_gt_marks" / "calibration_proposals")
    accepted = _names(root / "ball_gt_marks" / "calibration_gt")

    for r in rows:
        pf = r["prefixes"]
        st = r["stages"]
        mdir = root / r["key"]
        # segments: best of every location; total only when clips/segments.json describes this match
        n_seg = max(
            [_count(clips, pf, ".mp4")]
            + [
                sum(1 for n in _names(mdir / sub) if n.endswith(".mp4"))
                for sub in ["broadcast_segments"]
                + [
                    d
                    for d in _names(mdir)
                    if d.startswith("split_video_broadcast_frames")
                ]
            ]
        )
        total = (
            seg_json.get("segments") and len(seg_json["segments"])
            if seg_stems and seg_stems.startswith(r["key"])
            else None
        )
        mine = [s for s in store if s["source"] == r.get("mp4")]
        # Segments are "done" once their count is stable; we cannot know the true total for
        # most matches, so only mark done with a known total (segments.json) else show the count.
        if n_seg:
            seg = _progress(n_seg, total, "segments")
            if not total:
                seg = _stage("done", f"{n_seg} segments", 1.0)
        elif mine and mine[0]["n_clips"]:
            seg = _stage(
                "done" if mine[0]["finished"] else "partial",
                f"{mine[0]['n_clips']} clips (infer run)",
                None,
                _find_run(runs, run_id=mine[0]["run_id"]),
            )
        else:
            seg = _stage("none", "")
        st["segments"] = seg
        n_total = total or n_seg or None
        # detection
        det = _stage("none", "")
        for d, names in ml_dirs:
            c = _count(names, pf, ".jsonl")
            if c:
                det = _progress(c, n_total, "clips")
                m = _read_json(root / d / "_manifest.json") or {}
                if m.get("model_id"):
                    det["text"] += f" - {m['model_id']}"
                det["when"] = m.get("finished")
                break
        st["detection"] = det
        # tracking
        trk = _progress(_count(tracklets, pf, ".jsonl"), n_total, "clips")
        if mine:
            m0 = mine[0]
            if trk["state"] == "none":
                trk = _stage(
                    "done" if m0["finished"] else "partial",
                    f"{m0['done']}/{m0['n_clips']} clips (infer run {m0['run_id']})",
                    m0["done"] / m0["n_clips"] if m0["n_clips"] else None,
                    _find_run(runs, run_id=m0["run_id"]),
                )
        extra = [
            s
            for s in store
            if s["source"] != r.get("mp4")
            and Path(s["source"]).name.startswith(tuple(p + "_" for p in pf))
        ]
        if extra:
            trk["text"] = (
                trk["text"] + "; " if trk["text"] else ""
            ) + f"{len(extra)} clip infer run(s) in store"
            if trk["state"] == "none":
                trk["state"] = "partial"
            trk["run_id"] = trk["run_id"] or _find_run(runs, run_id=extra[0]["run_id"])
        st["tracking"] = trk
        # calibration
        # no meaningful total (only a sample of clips is ever calibrated): show counts, never "done"
        n_prop, acc = _count(proposals, pf, ".json"), _count(accepted, pf, ".json")
        cal = (
            _stage(
                "partial",
                f"{n_prop} clips w/ proposals" + (f", {acc} accepted" if acc else ""),
            )
            if (n_prop or acc)
            else _stage("none", "")
        )
        st["calibration"] = cal
        st["projection"] = _stage("na", "not implemented")


# ------------------------------------------------------------------ public API
_cache: dict = {"t": 0.0, "data": None}
_lock = threading.Lock()


def scan() -> dict:
    root = data_root()
    runs = _flow_runs()
    rows = _download_rows(root / "full_matches", runs) + _existing_rows(root)
    _fill_downstream(rows, root, _store_runs(runs_root()), runs)
    for r in rows:
        for k in STAGES:
            r["stages"].setdefault(k, _stage("none", ""))
            r["stages"][k]["link"] = _run_link(r["stages"][k].get("run_id"))
        r.pop("prefixes", None)

    def rank(r: dict) -> tuple:
        s = [r["stages"][k]["state"] for k in STAGES]
        return (
            0
            if "running" in s
            else 1
            if r["origin"] == "download" and s[0] != "queued"
            else 2
            if r["origin"] == "existing"
            else 3,
            r["key"],
        )

    rows.sort(key=rank)
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "stages": STAGES,
        "links_available": bool(runs),
        "matches": rows,
    }


def get_status(force: bool = False) -> dict:
    with _lock:
        if force or _cache["data"] is None or time.time() - _cache["t"] >= CACHE_TTL_S:
            _cache.update(data=scan(), t=time.time())
        return _cache["data"]
