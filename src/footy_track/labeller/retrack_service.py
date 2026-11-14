"""Re-tracking inside the Ball Check review server (``source=list``).

Enabled by ``FOOTY_RETRACK_CACHE`` (the round's detector candidate cache folder, one
``<run clip>.jsonl`` per clip). After every verdict the clip's whole-clip constrained
Viterbi path is recomputed in a background thread (the verdict POST does not wait), and
when it finishes up to :data:`PIN_PER_TRACK` follow-up cards are *pinned*: the next cards
of the queue after the ones already prefetched. A follow-up is either a card of the list
that is still unjudged (moved up) or, where the list has none near that frame, a new
blind id-keyed card for the path's box at the frame. The queue is otherwise the round's
own list, in its own order: nothing waits on a re-track, and the blind audits, controls
and random frames (never pinned, never hidden) keep their places in the base order, with
one base card after every :data:`PINS_PER_BASE` pinned ones, so they are all served.

Optional ``FOOTY_RETRACK_AUTO_DONE=<conf>`` (default off): an unjudged, non-audit card
whose frame the path answers with a detected box on that very box at confidence >= conf
is not served. Unserved cards write nothing by themselves at ingest; the path's labels do.

State lives under ``<gt dir>/retrack/``: ``cards.json`` (the new cards), ``state.json``
(round, pinned ids). The path itself is a pure function of the cache and the verdict log
and is recomputed at ingest, so nothing here is needed to be exact on a restart.
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from footy_track.autolabel import retrack as rt
from footy_track.ball_eval.detection_replay import CandidateCache, read_cache
from footy_track.ball_trackers.detection_tracker import DETECTED

LOGGER = logging.getLogger(__name__)

PIN_PER_TRACK = 2  # follow-up cards pinned per finished re-track
MAX_PINNED = 6  # unserved pinned cards at most
MAX_DYNAMIC_PER_CLIP = 10  # new cards per clip: the round must stay finite
PINS_PER_BASE = 2  # pinned cards served per card of the base order
MIXED_ROLES = frozenset({"audit", "control", "random"})  # blind samples: never moved
#: a card whose role is unknown (no meta file) is treated like a blind sample
NO_MOVE = MIXED_ROLES | {"?"}


class RetrackService:
    def __init__(
        self,
        *,
        cache_dir: Path,
        state_dir: Path,
        round_id: str,
        meta_path: Path | None = None,
        auto_done: float | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.state_dir = state_dir
        self.round_id = round_id
        self.auto_done = auto_done
        self.roles: dict[str, str] = {}
        if meta_path and meta_path.exists():
            try:
                self.roles = {
                    k: v.get("role", "")
                    for k, v in json.loads(meta_path.read_text()).items()
                }
            except (OSError, ValueError, AttributeError):
                LOGGER.warning("retrack: card meta %s unreadable", meta_path)
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrack")
        self._running: set[str] = set()
        self._dirty: set[str] = set()
        self._caches: dict[str, CandidateCache] = {}
        self._points: dict[str, dict] = {}  # run clip -> {frame: BallPoint}
        self._sols: dict[str, object] = {}
        self._dynamic: dict[str, dict] = {}
        self._pinned: list[str] = []
        self._streak = 0  # follow-up cards judged in a row (see arrange)
        self.stats: dict[str, dict] = {}
        self.version = 0
        self._load()

    # ------------------------------------------------------------------ state

    def _load(self) -> None:
        try:
            for c in json.loads((self.state_dir / "cards.json").read_text()):
                self._dynamic[c["id"]] = c
            self._pinned = list(
                json.loads((self.state_dir / "state.json").read_text()).get(
                    "pinned", []
                )
            )
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self._save()

    def _save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        for name, obj in (
            ("cards.json", list(self._dynamic.values())),
            (
                "state.json",
                {
                    "round": self.round_id,
                    "pinned": self._pinned,
                    "auto_done": self.auto_done,
                },
            ),
        ):
            tmp = self.state_dir / (name + ".tmp")
            tmp.write_text(json.dumps(obj, indent=1))
            tmp.replace(self.state_dir / name)

    def dynamic_cards(self) -> list[dict]:
        with self._lock:
            return list(self._dynamic.values())

    # ------------------------------------------------------------- scheduling

    def on_verdict(
        self, run_clip: str | None, card_id: str | None = None, *, undo: bool = False
    ) -> None:
        """A verdict (or an undo) landed on a card of ``run_clip``: re-track it, async."""
        with self._lock:
            if card_id is not None and not undo:
                self._streak = self._streak + 1 if card_id in self._pinned else 0
            elif undo:
                self._streak = 0
            if not run_clip:
                return
            if run_clip in self._running:
                self._dirty.add(run_clip)
                return
            self._running.add(run_clip)
        self._pool.submit(self._run, run_clip)

    def _run(self, clip: str) -> None:
        while True:
            try:
                self._solve_once(clip)
            except Exception:  # noqa: BLE001 - a failed re-track must never break the review
                LOGGER.exception("retrack: %s failed", clip)
            with self._lock:
                if clip in self._dirty:
                    self._dirty.discard(clip)
                    continue
                self._running.discard(clip)
                return

    def pending(self) -> int:
        with self._lock:
            return len(self._running)

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until no re-track is running (tests, scripts)."""
        end = time.time() + timeout
        while time.time() < end:
            if not self.pending():
                return True
            time.sleep(0.02)
        return False

    # ------------------------------------------------------------------ solve

    def _cache(self, clip: str) -> CandidateCache | None:
        if clip not in self._caches:
            path = self.cache_dir / f"{clip}.jsonl"
            if not path.exists():
                return None
            self._caches[clip] = read_cache(path)
        return self._caches[clip]

    def _solve_once(self, clip: str) -> None:
        bc = importlib.import_module("footy_track.labeller.ball_check")  # import cycle

        cache = self._cache(clip)
        if cache is None:
            return
        t0 = time.perf_counter()
        all_cards = bc._list_items()
        cards = [
            c for c in all_cards if c.get("run_clip") == clip and c.get("frame_size")
        ]
        judged = bc._read_verdicts()
        cons = rt.collect_constraints(cards, judged).get(clip, [])
        sol, _ = rt.solve(cache, cons)
        ms = round((time.perf_counter() - t0) * 1000)
        pts = {p.frame_index: p for p in sol.points}
        with self._lock:
            changed = rt.changed_frames(self._points.get(clip), sol)
            self._points[clip], self._sols[clip] = pts, sol
            self.stats[clip] = {
                "ms": ms,
                "frames": len(cache.frames),
                "constraints": len(cons),
                "changed": len(changed),
                "at": round(time.time(), 1),
            }
            self._pin_followups(
                clip, cards, all_cards, judged, sol, changed, cache.frame_size
            )
            self.version += 1
            self._save()
        LOGGER.info(
            "retrack: %s %d constraints, %d ms, %d frames changed",
            clip,
            len(cons),
            ms,
            len(changed),
        )

    def _pin_followups(
        self, clip, cards, all_cards, judged, sol, changed, size
    ) -> None:
        def is_judged(c: dict) -> bool:
            return ("list", c["clip"], c["frame_index"], c["box_index"]) in judged

        open_ids = {c["id"] for c in all_cards if c.get("id") and not is_judged(c)}
        self._pinned = [i for i in self._pinned if i in open_ids]
        live = set(self._pinned)
        free = MAX_PINNED - len(self._pinned)
        if free <= 0:
            return
        blocked = [c["frame_index"] for c in cards if is_judged(c)]
        # blind samples are never moved and never duplicated; pinned ones are queued already
        blocked += [
            c["frame_index"]
            for c in cards
            if not is_judged(c)
            and (self.roles.get(c["id"], "?") in NO_MOVE or c["id"] in live)
        ]
        frames = rt.followup_frames(
            sol, changed, blocked, limit=min(PIN_PER_TRACK, free)
        )
        stem = next((c["clip"] for c in cards), clip)
        n_dyn = sum(1 for c in self._dynamic.values() if c["run_clip"] == clip)
        for f in frames:
            near = [
                c
                for c in cards
                if abs(c["frame_index"] - f) <= rt.DEDUPE_FRAMES
                and not is_judged(c)
                and c["id"] not in live
                and self.roles.get(c["id"], "?") not in NO_MOVE
            ]
            if near:
                card = min(near, key=lambda c: abs(c["frame_index"] - f))
            else:
                if n_dyn >= MAX_DYNAMIC_PER_CLIP:
                    continue
                n_dyn += 1
                card = rt.dynamic_card(
                    self.round_id, clip, stem, size, self._points[clip][f]
                )
                card["box_index"] = rt.card_box_index(card["id"])
                self._dynamic[card["id"]] = card
            self._pinned.append(card["id"])
            live.add(card["id"])

    # ------------------------------------------------------------------ queue

    def _hidden(self, card: dict) -> bool:
        """Auto-done: not served because the path is confident on this very box."""
        if self.auto_done is None or self.roles.get(card["id"], "?") in NO_MOVE:
            return False
        p = self._points.get(card.get("run_clip", ""), {}).get(card["frame_index"])
        sol = self._sols.get(card.get("run_clip", ""))
        if p is None or sol is None or p.source != DETECTED:
            return False
        return (
            card["frame_index"] not in sol.anchored
            and (p.confidence or 0.0) >= self.auto_done
            and rt.box_matches(card, p, card["frame_size"])
        )

    def arrange(self, todo: list[dict]) -> tuple[list[dict], int]:
        """The served order of the unjudged cards ``todo`` (the list's own order) and the
        number hidden as auto-done."""
        with self._lock:
            by_id = {c["id"]: c for c in todo if c.get("id") is not None}
            self._pinned = [i for i in self._pinned if i in by_id]
            pinned = [by_id[i] for i in self._pinned]
            hidden = [c for c in todo if c not in pinned and self._hidden(c)]
            hid = {id(c) for c in hidden}
            base = [c for c in todo if id(c) not in hid]
            return (
                rt.arrange(
                    base,
                    pinned,
                    every=PINS_PER_BASE,
                    base_first=self._streak >= PINS_PER_BASE,
                ),
                len(hidden),
            )

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": True,
                "round": self.round_id,
                "pending": len(self._running),
                "version": self.version,
                "pinned": len(self._pinned),
                "dynamic_cards": len(self._dynamic),
                "auto_done_conf": self.auto_done,
                "clips": self.stats,
            }
