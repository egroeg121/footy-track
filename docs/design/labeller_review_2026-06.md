# Labeller — Deep Review (Implementation + Design)

**Bead:** ft-31z · **Date:** 2026-06-21 · **Reviewer:** polecat ruby
**Scope:** `src/footy_track/labeller/` (`app.py`, `server.py`, `video_utils.py`,
`web/index.html`, `_canvas_compat.py`) plus how the GT-marking / bake-off and the
SAM3 Run-loop integrate. All citations are `file:line` against the branch at
review time.

> **Headline:** there are **two complete, divergent labeller front-ends** in the
> tree (a Streamlit app and a FastAPI+Konva web app) sharing one backend engine
> (`video_utils.py`). The web app is the intended go-forward UX for mark-once GT
> + bake-off, but it is **missing an export path** — it can mark and propagate
> but cannot save the result. That is the single blocker for marking clips. There
> is **no `ball_eval/` module yet** (the bead assumed one exists); the bake-off
> harness is not present in this tree.

---

## 1. Component map & data flow

```
                         ┌────────────────────────────────────────┐
                         │  video_utils.py  (shared backend)        │
                         │  • Sam3VideoLabeller  – SAM3 propagation │
                         │  • BackgroundLabeller – daemon-thread    │
                         │       run/pause/restart + anomaly stop   │
                         │  • yolo_seed_objects  – YOLO seeding      │
                         │  • export_frames_json – disk export      │
                         └───────────────┬──────────────┬──────────┘
                                         │              │
              ┌──────────────────────────┘              └──────────────────────┐
              │                                                                 │
   ┌──────────▼───────────┐                              ┌─────────────────────▼──────────┐
   │ app.py (Streamlit)    │                              │ server.py (FastAPI)             │
   │ • server-side state in │                              │ • Session.timeline = single     │
   │   st.session_state     │                              │   source of truth (per-frame)   │
   │ • st_canvas drawing    │                              │ • provenance tags PROV_*        │
   │ • _canvas_compat shim   │                              │ • HTTP + WebSocket              │
   │ • export via Save JSON │   ◄── only this side exports │ • web/index.html (Konva canvas) │
   └────────────────────────┘                              └─────────────────────────────────┘
        run_labeller.py entrypoint                          uvicorn ...server:app  (no entrypoint script)
```

**Pipeline position.** The labeller produces `FrameDetections` JSON whose `uri`
stem encodes the absolute frame index (`<stem>_frame_000123`). `feature_store`
ingests exactly this format: `importers.parse_labeller_uri`
(`feature_store/importers.py:93`) parses `<clip>_frame_<idx>` back out, and
`import_labeller_json` tags rows `source='sam3'`. So the contract between
labeller output and the feature store is **the `export_frames_json` file shape**,
not any in-memory object — which is why the web app's missing export is fatal to
the downstream flow.

### Module responsibilities

| File | Responsibility | LOC |
|---|---|---|
| `video_utils.py` | SAM3 propagation engine, threaded run loop, YOLO seeding, anomaly heuristic, JSON export | 830 |
| `app.py` | Streamlit UI — full interaction, state in `st.session_state` | 911 |
| `server.py` | FastAPI — authoritative timeline w/ provenance, run/pause/restart over WS | 406 |
| `web/index.html` | Konva-based browser client for `server.py` | 428 |
| `_canvas_compat.py` | Monkeypatch restoring `image_to_url` for `st_canvas` on Streamlit ≥1.40 (Streamlit-only) | 44 |

---

## 2. Design assessment

### 2a. The two front-ends are the central design problem

`app.py` and `server.py` are **two ~full implementations of the same feature**.
They share `video_utils.py` but duplicate everything above it and embody two
*different* state models:

- **`app.py`** keeps the working set in `st.session_state.objects` and re-derives
  view mode every rerun (`app.py:331-336`). State is implicit, scattered across
  ~14 session keys (`app.py:69-93`), and there is no single per-frame timeline —
  corrections live only in `BackgroundLabeller.frames` plus transient session
  state.
- **`server.py`** introduces the *right* abstraction the Streamlit app lacks: a
  single authoritative `Session.timeline[idx]` of `ObjectDetection` with
  **provenance** (`PROV_LABELLER`/`PROV_YOLO`/`PROV_SAM3`, `server.py:44-47`).
  Labeller boxes are ground truth and survive re-propagation via the merge rule
  in `merge_propagated` (`server.py:106-113`). This is a cleaner, more correct
  model than the Streamlit app's.

**Recommendation:** treat `server.py` + `web/` as the go-forward labeller and
**deprecate/retire `app.py`** once export parity exists (§5, P0/P1). Maintaining
both doubles the surface for the morning's marking work and they have *already*
drifted (see consistency bugs below). Do not invest further in `app.py`.

### 2b. Duplicated logic across the two front-ends

`yolo_seed_objects` and `_nms_filter` exist **twice**, verbatim:
`video_utils.py:743-808` (used by `server.py`) and `app.py:96-160`
(`app.py`-local copies). The `app.py` copies should be deleted and the
`video_utils.py` versions imported (`app.py` already imports other helpers from
`video_utils`). Pure dead duplication; risk of one drifting from the other.

### 2c. `video_utils.py` engine — solid, with sharp edges

The engine is the strongest part of the labeller:

- **Hot-predictor cache** (`get_cached_predictor`, `video_utils.py:146-188`)
  keeps SAM3 weights + compiled kernels resident across pause/restart — this is
  what makes the correct-and-restart loop usable. Good.
- **Restart-mid-clip via temp video** (`iter_frames_from`,
  `video_utils.py:416-490`): because `SAM3VideoPredictor` seeds tracks from the
  *first frame it sees*, restarting at frame N writes frames `[N, end]` to a temp
  `.mp4` and re-streams. Clever and necessary given the predictor's statefulness.
- **Seed frame is ground truth** (`_seed_frame_detections`,
  `video_utils.py:492-517`; `offset == 0` branch at `:477`): the user's exact
  boxes are emitted verbatim rather than SAM3's re-segmentation. Correct call —
  prevents a corrected box from being mangled on restart.

Sharp edges:

- **Object-index → label mapping is positional and fragile**
  (`_result_to_frame`, `video_utils.py:519-554`). SAM3 only preserves object
  *order*, so label assignment is `self.objects[obj_idx].label`. If SAM3 drops a
  track or returns masks in a different count/order than seeds, **labels silently
  shift** (`"unknown"` fallback at `:535`). For the ball specifically — the one
  object most likely to be lost — a dropped ball track means every subsequent
  object inherits the wrong label. This is a real correctness risk for GT.
- **Temp-clip rewrite is lossy and slow** (`video_utils.py:449-465`):
  re-encodes with `mp4v` at the source fps, doubling I/O and risking codec drift
  vs the original frames. For long clips this is the dominant cost on every
  restart.
- **Anomaly heuristic is plausible but untuned** (`_track_anomaly_reason`,
  `video_utils.py:683-728`): nearest-same-label matching with a 40%-of-diagonal
  jump threshold and 8× area ratio. No tests; thresholds are magic numbers. It
  matches by *label* not *track identity*, so with two players it can match the
  wrong pair and false-positive. Fine as a "stop and let me look" aid, not
  something to trust silently.

### 2d. Threading / async correctness in `server.py`

- `Session.timeline` is guarded by `_tl_lock` (`server.py:67`) — good. But
  `BackgroundLabeller` exposes `running`, `last_completed_frame`,
  `anomaly_frame` as plain attributes read from the asyncio loop
  (`_stream_frames`, `server.py:298-337`) while written under `self._lock` in the
  worker thread (`video_utils.py:652-676`). These cross-thread reads are
  unsynchronized. In CPython they won't corrupt, but `anomaly_frame` is *both*
  read and cleared from the async side (`server.py:328`) — a genuine race with
  the worker setting it (`video_utils.py:666-668`). Low-probability, but it can
  drop or double-fire an anomaly pause.
- **`SESSION` is a single module-global** (`server.py:147`). The server is
  strictly **single-session / single-user**. Two browser tabs share one
  timeline and one `bg`; the second `load` silently wipes the first. Acceptable
  for solo morning marking, but must be stated, and the WS handler assumes it
  (`server.py:343` captures `bg = SESSION.bg` then re-reads `SESSION.bg` later —
  inconsistent).

### 2e. Frame access pattern is O(n) seeks everywhere

Every frame fetch opens a fresh `cv2.VideoCapture`, seeks, reads one frame, and
releases — `frame_jpeg` (`server.py:132-144`), and in `app.py` at `:405-408`,
`:427-431`, `:485-488`, `:730-732`. On a long clip, scrubbing is a storm of
open/seek/release. `video_utils.py` has no frame cache. For the morning's UX
(scrub back to fix a box, scrub forward) this will feel sluggish. A small
LRU frame cache in the `Session` (web) would help most.

---

## 3. Bug / gap register (ordered by impact on "marking clips in the morning")

### 🔴 B1 — Web labeller cannot export (BLOCKER)
`server.py` has **no save/export endpoint**. The timeline is never written to
disk; `export_frames_json` (`video_utils.py:811-817`) is only wired into
`app.py` (`app.py:743-745`). A user can mark + propagate in the browser and then
**lose everything** — there is no path from `Session.timeline` to the
`FrameDetections` JSON that `feature_store` ingests. Without this, the web app
produces no GT. **Must fix before marking.**

### 🔴 B2 — Web default-model caption contradicts the actual default
`web/index.html:54` tells the user the default is `sam3.1_multiplex.pt`. The
backend default is `sam3.pt`, and `_default_model_uri` (`video_utils.py:100-132`)
explicitly documents that `sam3.1_multiplex.pt` returns **whole-frame masks for
every object on this footage** and is deliberately NOT used. A user trusting the
caption and typing that checkpoint gets garbage GT. Fix the caption text.

### 🟠 B3 — Hardcoded class list in JS can drift from `DETECTION_CLASSES`
`web/index.html:101` hardcodes `CLASSES = [...]` and `:102-103` hardcodes
`COLORS`. The source of truth is `schema.DETECTION_CLASSES`
(`schema.py:25-34`) and `detectors.utils.color_map`. They currently match, but
any class change updates Python and silently leaves the JS stale → boxes drawn
with the wrong class/colour, corrupting GT labels. Serve the class+colour list
from an endpoint (`/classes`) instead of duplicating it.

### 🟠 B4 — `idle_results` save-on-leave only triggers if mode is set, but it never is on Run-completion-then-edit
In `web/index.html`, `saveCurrentEdits` (`:306-311`) persists edits when
`mode==="paused"||mode==="idle_results"`. After a run completes, `uiIdle()` sets
`mode="idle_results"` (`:370`) — good. But the **seed frame** path: on first
load `mode="idle"` (`:280`), and `autodetect` draws editable boxes; if the user
edits those frame-0 boxes and immediately scrubs, `saveCurrentEdits` does
**nothing** (mode is `"idle"`, not in the allowed set), so the edit is lost
until they hit Run (which calls `commitCanvasTo`). Subtle data-loss footgun
during seeding. Either include `"idle"` or persist on every box change.

### 🟠 B5 — Positional label shift on dropped SAM3 tracks (see §2c)
`video_utils.py:530-536`. If SAM3 returns fewer masks than seeds, the mask→label
zip misaligns and downstream labels are wrong with no warning. Needs SAM3 to
return per-object IDs, or a guard that pauses when mask count ≠ seed count.

### 🟡 B6 — Anomaly `anomaly_frame` cross-thread race (see §2d)
`server.py:320-328` vs `video_utils.py:666-668`. Read+clear of `anomaly_frame`
from the async loop without the worker's lock.

### 🟡 B7 — Streamlit app: `_load_frame_objects` reads `completed` captured once per run
`app.py:353-361` closes over `completed` (`app.py:310`) computed at the top of
`main()`. During a paused scrub the list is fixed for that run, which is fine,
but the prev/next handlers (`:557-572`) call it after `st.rerun()` boundaries —
behaviour depends on rerun timing. Works today; fragile.

### 🟡 B8 — `_canvas_compat` is Streamlit-only debt
`_canvas_compat.py` exists solely to keep `st_canvas` alive on modern Streamlit.
If `app.py` is retired (§2a) this whole shim + the `streamlit-drawable-canvas`
dependency can go. Tracked as cleanup.

### 🟡 B9 — No tests anywhere in `labeller/`
No test file exercises `Sam3VideoLabeller`, the anomaly heuristic, the merge
rule, or `export_frames_json`. The merge rule (`merge_propagated`) and
`_track_anomaly_reason` are pure functions and trivially unit-testable; they
should be, given GT correctness rides on them.

---

## 4. Where GT-marking + the 4 bake-off methods plug in

The bead assumes a `ball_eval/` module (GT marking + bake-off harness). **It
does not exist in this tree** — `find src -iname '*ball*' -o -iname '*eval*'`
returns nothing, and there are no references to `ball_eval`/`bake-off` anywhere
in `src` or `docs`. So this section is about *where the seams are* for that work,
not a review of existing code.

**Mark-once GT** is well-served by `server.py`'s provenance model: a box tagged
`PROV_LABELLER` is ground truth and is preserved across re-propagation
(`merge_propagated`, `server.py:106-113`). The natural GT artifact is the
`Session.timeline` filtered to `model == PROV_LABELLER`. **But** (B1) there is no
endpoint to emit it. The first bake-off prerequisite is therefore an export
endpoint that writes the timeline (or just the labeller-provenance boxes) as
`FrameDetections` JSON.

**The 4 bake-off methods** (detection/tracking variants competing against GT)
would plug in at the `/autodetect` seam (`server.py:212-251`) — today that seam
hardcodes a single YOLO detector via `yolo_seed_objects`. To bake off N methods
you need: (a) a method registry, (b) per-method runs against the same frames,
(c) a metric vs the `PROV_LABELLER` GT. None of that scaffolding exists yet. The
cleanest insertion is a new `ball_eval/` module that consumes the exported GT
JSON + each method's output and computes IoU/precision-style metrics — keeping it
*out* of the labeller server (which should stay a marking tool).

---

## 5. Consistency check

| Concern | Status |
|---|---|
| `ObjectDetection` schema (`schema.py:57-65`) ↔ web wire format | ✅ server maps `model`→`source` over the wire (`server.py:173-186`) and back (`:155-170`); normalized xywh both sides. Consistent. |
| `DETECTION_CLASSES` ↔ JS `CLASSES` | ⚠️ duplicated, currently aligned (B3). |
| `color_map` ↔ JS `COLORS` | ⚠️ duplicated, currently aligned (B3). |
| Frame-index encoding ↔ feature-store importer | ✅ `<stem>_frame_<idx>` written by `video_utils.py:498,521` and parsed by `importers.parse_labeller_uri` (`importers.py:93`). |
| Time conventions (ContinuousTime/GameTime) | ➖ labeller works in raw frame indices only; no GameTime mapping. Fine for GT marking, but the exported JSON has no period/clock info — downstream must derive it. Worth noting for the broadcast→pitch→tagging vision. |
| `provenance` (`PROV_*`) ↔ `ObjectDetection.model` | ✅ provenance is stored in the `model` field (`server.py:56`, `:167`). Slight semantic overload — `model` doubles as "which model produced this" and "is this ground truth" — but documented. |

---

## 6. Recommendations (prioritised — each a candidate bead)

| # | Priority | Recommendation | Why |
|---|---|---|---|
| R1 | **P0** | Add a `/session/export` endpoint to `server.py` that writes `Session.timeline` via `export_frames_json` | B1 — unblocks all GT marking; nothing downstream works without it |
| R2 | **P0** | Fix `web/index.html:54` default-model caption (`sam3.1_multiplex.pt` → `sam3.pt`) | B2 — one-line fix preventing whole-frame-mask garbage GT |
| R3 | **P1** | Decide the canonical front-end; deprecate `app.py` (Streamlit) in favour of `server.py`+`web/` | §2a — stop maintaining two drifting UIs before marking begins |
| R4 | **P1** | Serve classes+colours from a `/classes` endpoint; delete JS duplication | B3 — prevents silent label/colour corruption |
| R5 | **P1** | Persist seeding-mode edits in web (include `"idle"` in `saveCurrentEdits` or save-on-change) | B4 — data-loss footgun during marking |
| R6 | **P2** | Guard `_result_to_frame` when SAM3 mask count ≠ seed count (pause + warn instead of silent label shift) | B5 — protects GT label integrity |
| R7 | **P2** | Add unit tests for `merge_propagated`, `_track_anomaly_reason`, `export_frames_json` | B9 — GT correctness rides on these pure functions |
| R8 | **P2** | Add an LRU frame cache in `Session` to make scrubbing responsive | §2e — UX during correct-and-restart |
| R9 | **P2** | If `app.py` retired: delete `_canvas_compat.py`, the duplicated `yolo_seed_objects`/`_nms_filter`, and drop `streamlit-drawable-canvas` | B8, §2b — dead-code removal |
| R10 | **P3** | Scaffold `ball_eval/`: consume exported GT JSON + N method outputs, compute IoU/precision metrics; register methods at the `/autodetect` seam | §4 — the actual bake-off, after R1 lands |
| R11 | **P3** | Synchronize cross-thread reads of `BackgroundLabeller` anomaly state | B6 — correctness hardening |

**If only two things ship before the morning's marking: R1 and R2.** They are the
difference between "can produce ground truth" and "cannot."
