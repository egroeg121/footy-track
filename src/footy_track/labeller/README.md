# Footy Track Labeller — features & requirements

Web app for semi-automatic video labelling: mark boxes by hand, propagate them
through the clip with VitTrack, review/correct the results. FastAPI backend
(`server.py`), Konva.js frontends (`web/index.html` labeller, `web/review.html`
review), VitTrack SOT propagation backend (`video_utils.py`).

Run with:

```bash
uv run uvicorn footy_track.labeller.server:app --reload
```

| Route | Page |
|---|---|
| `/` (alias `/main`) | Hub |
| `/labeller` | Frame labeller (mark + propagate) |
| `/object_review` | Tinder-style crop review/correction |
| `/ingest` | Clip ingestion |

## Label hierarchy

Every box records where it came from in its `model` field, persisted in the
JSONL sidecar as the second tag: `tags: [label, model]`:

| `model` | Meaning | Authority |
|---|---|---|
| `labeller` | Hand-drawn / hand-corrected — ground truth | Never auto-overwritten |
| `vittrack` | VitTrack SOT propagation output | Overwritten by re-runs; ignored on frames that have GT |
| `yolo` | YOLO autodetect output | Same as vittrack |
| `sam3` | Legacy SAM3 propagation (back-compat with old sidecars) | Same as vittrack |

**Rules:**

- **Display order follows the hierarchy**: GT (hand-crafted) first, then
  model-assisted (machine boxes a human has seen/kept), then pure model
  output — everywhere labels are listed or drawn.
- **Newer hand-crafted labels win**: when the user hand-marks a frame, those
  boxes replace any pre-existing labels of the same or lower level (old GT is
  replaced by new GT; machine boxes are replaced outright). Machine output
  never replaces anything above its own level.
- A propagation run seeds from the timeline at the start frame (canvas is
  committed as GT first, so Run/Restart always starts from what's on screen).
- Downstream, `merge_propagated` keeps GT frames untouched — new machine boxes
  are discarded there. The run reports which frames kept hand marks
  ("kept your marks on frames 14–17") instead of skipping silently.
- On clip load, the full JSONL sidecar is restored into the timeline **with the
  original `model` tag preserved** (machine boxes must not be promoted to
  `labeller` GT by a save/reload cycle).

## Persistence

- Sidecar: one `<clip_stem>.jsonl` per clip under the GT-marks dir; one JSON
  object per line: `{frame_index, bbox|null, center|null, tags}`.
  Skip markers: `no_ball`, `not_broadcast` (frame recorded, no box).
- Flush is debounced (2 s) and forced on clip switch; all model tags persist.
- Downstream, sidecars + Roboflow datasets are ingested into the DuckDB
  feature store (`feature_store/ingest_gt.py`) and exported as leakage-free
  YOLO training datasets (`scripts/export_training_dataset.py`).

## Labeller UI features

- **Clip picker** (left sidebar): grey `✓` = clip has some marks; green `✅` =
  fully labelled (server-checked completion; refreshes every 20 s).
- **Timeline bar** under the scrubber: shows which frames have ball / player
  labels; updates live during propagation runs; red stripe = player-only frame
  (ball status unknown).
- **Run / Pause / Restart** propagation loop with anomaly handback: run
  auto-pauses when a track moves implausibly or VitTrack confidence drops
  below the handback threshold, so you correct and restart.
- On clip load: saved marks are shown if frame 0 has boxes, otherwise YOLO
  autodetect seeds frame 0. Run-control state is fully reset on clip switch.
- **Detected objects pane** (right): per-box class dropdown, delete, and
  "Clear all" (confirm + undoable) for the current frame.
- **Keyboard**: `g` next detection · `n` no-ball · `b` not-broadcast ·
  `r` re-run autodetect · `1–6` class hotkeys · arrow keys/scrub to navigate ·
  Backspace deletes selection.
- Model warm-start: the VitTrack ONNX session is cached process-wide, so
  "Compiling model…" happens once per server start, not on every run.

## Review UI features (`/object_review`)

- Grid of detection crops grouped by class; batch accept / relabel / delete.
- Accept/relabel marks items as **seen** (green border) rather than removing
  them; "Hide reviewed" toggle (default on) filters them.
- Modal: side-by-side crop + full frame, Konva box editing, YOLO re-run,
  `a`/`d` navigation, Backspace delete; relabel stays in the modal.
- Seen state persists in localStorage; auto-save on navigate/close.

## Requirements (standing)

1. Hand marks are sacred: no automatic process may modify or discard
   `labeller`-provenance boxes, and nothing may silently promote machine boxes
   to `labeller` GT.
2. Everything the UI shows must survive save → reload with identical
   `model` tags, labels, and geometry (round-trip fidelity — see
   `tests/feature_store/test_roundtrip_fidelity.py` for the store leg).
3. On video load, all pre-existing labels (any `model` tag) must be scanned and
   made visible/navigable immediately, so the user can skip around (`g`,
   timeline bar) rather than re-labelling blind.
4. Propagation runs must be restartable from any corrected frame, and it must
   be visible which frames a run did not touch.
5. All timestamps follow the ContinuousTime / GameTime conventions
   (`docs/timings.md`).

## Backlog / open items

- **Feature-store scan on load**: the load path only reads the JSONL sidecar;
  labels that exist only in the DuckDB feature store for the clip are not yet
  surfaced.
- **Roboflow upload leg**: local YOLO export is built and fidelity-tested;
  pushing an export back to Roboflow as a new dataset version is not yet wired
  (see PR #18's skip-by-default live test).
- **VitTrack handback threshold**: 0.5 on main; a local experiment suggested
  0.3 — unresolved which is right (parked in stash).
- **Per-machine clip symlinks**: `eval_data/clips/*` symlink targets are
  machine-specific (macOS iCloud vs Linux `/mnt/storage`) and get clobbered
  when either machine "fixes" them; needs a per-machine config + regeneration
  script.
- **GPU propagation loop** (`ft-8vx` epic): paused by request.
