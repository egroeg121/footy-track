# Labelling architecture review (read-only) and consolidation proposal

Scope: `src/footy_track/labeller/` plus `web/`, as deployed from `/home/george/footy-track-main` (HEAD `b2f8faa`, PR #71, one commit past the `1e6baa6` named in the brief). Measurements: GET-only against `http://mac-mini-i7:8011`, load average 13 to 17 on the mini at the time (all timings are inflated by that, but the ratios hold). Nothing was written to any label store, any repo or the live site.

KB pages relied on (Step 0 searches run for "footy labeller", "labelling site", "ball check", "labelling jobs", "prefect jobs"):
- `wiki/footy-labeller-gotchas.md`: GT truncation history, provenance trap, Ball Check design, "`{"ok": false}` in a 200", sampler perf, iOS Range.
- `wiki/prefect-job-system.md`: queue, pools and worker model.
- `wiki/footy-claude-autolabel-loop.md`: the round queue and its labelling server.
- `wiki/footy-data-plane-reconciliation.md`: annotations authority is the feature store ("ingest, not files").
- `raw/labeller-jobs-api.md`: one server, `?dataset=`, jobs API, ContextVar gotcha.
- `raw/footy-ballcheck-prepare-2026-11-17.md`: render cache and prepare numbers.
- `raw/labeller-pause.md`: event-loop starvation fix (#61).
- `raw/footy-offline-ball-tracking-2026-11-15-labeller.md`: kf10 re-encode is what makes seeks fast, and a restart empties the in-memory caches.

Hooks branch: `/home/george/footy-track-hooks` (`feat/labelling-hooks`) is at `b2f8faa` with a clean tree, so it has no code or design doc yet. My hooks recommendations come from the brief plus the precursors already on main (the `prepare` state machine in `jobs.py`, `task_done.done_when`, the adaptive sampler in `candidates.py`, and the Prefect automation in `~/code/jobs/broadcast_review_automation.py`).

---

## 0. Executive summary

1. The site is functionally rich but is really 9 mini-apps sharing a process. Each owns its own store format, "live label" fold, progress rule, queue order, render path and page scaffold. There is no common kind abstraction; `jobs.py` (1418 lines) has the only cross-kind switchboard, as `if kind ==` ladders in `compute`, `deep_link`, `_item_units` and `_normalise_items`.
2. Live-verdict logic exists 3 times (`ball_check._read_verdicts`, `jobs._verdicts`, `feature_store.ball_checks.read_latest_verdicts`), atomic file writes 9 times, video lookup and frame decode about 6 times, and clip-alias resolution twice (`jobs._gt_stem` and `ball_check._gt_sidecar_stem`).
3. The speed-critical parts are good and must be preserved: Ball Check's optimistic advance, 4-crop and 3-clip prefetch, memory LRU plus content-keyed disk cache, mtime-keyed file caches in `jobs.py`/`candidates.py`, and kf10 clips.
4. The slow spots are almost all accidental rather than structural:
   - `/api/tasks` (the home page) takes 1.2 to 1.9 s because it re-reads one JSON file per item.
   - `/clips/status` blocks the event loop for 4 to 6 s.
   - Calibration frames take 1.4 to 1.8 s every time with no cache.
   - `/review/queue` ships 6.3 MB per load.
   - The first `source=candidates` request after a restart takes 13 s.
5. Real bugs found (section 2.3), the most important being:
   - Ball Check's page never checks the HTTP result of a verdict save, so a server error silently drops the verdict.
   - `/clips/status` stalls every other request.
   - `/review/correct`/`delete` and Ball Check `save_box` can be silently erased by a stale `Session` flush.
   - Calibration GT read-modify-write is unlocked, while its sibling stores are locked.
6. Target: a `LabelKind` plugin interface (unit address, item schema, store adapter, progress rule, order policy, render provider, page config), one `LabelStore` read/append API implemented as adapters over today's files, a single `LabelEvent` bus fed by every write, one job engine with hooks as the extension point, one `RenderService`, and a small shared frontend shell.
7. Do not migrate file formats first. Pinned autolabel checkouts read `gt/ball_checks/*.jsonl` directly, and `ingest_gt` and the LAB tests pin the sidecar format. Adapt over files, emit events, and let the event log feed versioned snapshots (and later LanceDB).
8. For George's priority (versioned datasets and per-version metrics): a version is `(dataset, event watermark, split manifest)`, and a snapshot is `LabelStore.live()` as of that watermark. That is the one function the consolidation introduces that versioning needs.
9. Ordering: (a) a bug-fix PR (S); (b) extract shared primitives with a parity test (M); (c) store/event interface; then render service, job engine plus hooks, frontend shell, and task-board-as-jobs.
10. Explicitly not consolidated: the propagation/WS tool core (`Session`, `video_utils`, `motion_tracker`), calibration geometry, identity clustering, the sampler math, `matches_status`, the Prefect flows, and the page interaction designs.

---

## 1. Inventory

Dataset resolution is uniform: `datasets.DatasetMiddleware` (`datasets.py:412`) sets a ContextVar from `?dataset=`; registry `/mnt/storage/footy_data/labeller_datasets.json` (5 datasets live: `calib`, `nottinghamforest`, `nottinghamforest_disagree`, `nottinghamforest_vittrack`, `broadcast_review`). "marks" below means `<dataset.marks_dir>`.

| UI / page | Routes (module) | Storage (format, location) | Fed by | Cache | Label consumers |
|---|---|---|---|---|---|
| Task board `/` (`main.html`, 278 lines) | `/`, `/main`, `/api/tasks`, `/api/server_info`, `/api/datasets` (`server.py`, `tasks.py`, `task_done.py`, `jobs.board_tasks`) | `labelling_tasks.json` (+ `.done.json`) and the jobs dir; read-only view | CLI `labelling_tasks`, open jobs | `done_when` results 5 s TTL (`task_done.py:CACHE_TTL`) | none |
| Ball placement `/labeller` (`index.html`, 1155 lines) | `/clips`, `/clips/status`, `/session/load`, `/frame/{n}.jpg`, `/marks`, `/autodetect`, `/edit`, `/no-ball`, `/not-broadcast`, `/propagate`, `/ws` (`server.py`, `session.py`, `video_utils.py`, `run_stream.py`) | `marks/<clip>.jsonl`: one line per box (`tags=[label, provenance, markers]`) plus skip-marker lines. Whole file rewritten from in-memory timeline, debounced 2 s (`session.py:_do_flush`) | clip dir listing, or `?job=` via `/api/jobs/{id}/frames` (job mode, J/K stepping) | none (`session.frame_jpeg` opens cv2, seeks, encodes per request) | `ingest_gt` to DuckDB, `export_training_dataset`, job progress |
| Object review `/object_review` (`review.html`, 857) | `/review/queue`, `/review/crop`, `/review/correct`, `/review/delete`, `/review/frame`, `/review/yolo` (`review.py`) | edits the same `marks/<clip>.jsonl` in place by positional `box_index` (`write_text`, non-atomic) | all sidecars scanned per request | crop LRU 200 (`review.py:58`) | same as above |
| Ball Check `/ball_check` (`ball_check.html`, 665) | `/ball_check/queue,stats,crop,crop_meta,clip,step,save_box,verdict,undo` (`ball_check.py` 1635 lines, `candidates.py`, `ball_check_cache.py`) | verdicts: `marks/ball_checks/<clip>.jsonl`, append-only, `verdict:null` tombstone undo. Corrected boxes also go to the sidecar (`save_box`) | 4 sources: sidecar (all human/machine boxes), `cand` (S3 machine labels via `candidates_dir`), `list` (registry `ball_check_list` JSON), job inline cards (`box_index >= 2**30`) | memory LRU crops 2000 and clips 600; disk render cache `ballcheck_renders` (sha256 content keys, job refs, GC, 5 GB cap); candidate pool and file caches; list cache by mtime | `feature_store/ball_checks.py` to `detection_review`, autolabel-ingest (pinned checkouts read the log dir), job results |
| Calibration `/calibration` (`calibration.html`, 964) | `/calibration/api/{clips,frame,fit,warp,auto_points,gt,frame_type,too_few_points,pitch_model}` (`calibration.py` 905) | `marks/calibration_gt/<canon>.json` ({frames:{idx:rec}}); `marks/frame_type/<canon>.json` (not_broadcast, too_few_points flags) | clip dir, deep links, jobs (deep link only: no stepping mode) | auto-points LRU 64 (`calibration.py:716`); frames not cached | `ingest_calibration` |
| Calibration review `/calibration/review` (`calibration_review.html`, 525) | `/calibration/api/proposals/...` accept/reject/not_broadcast/too_few_points (`calibration_proposals.py`) | `marks/calibration_proposals/<canon>.json` (runs, then frames, then status); accept writes `calibration_gt` | `prelabel_calibration` flow | client prefetch scheduler (`calibration_review.html:187`) | `ingest_calibration` |
| Identity `/identity`, `/identity/merge` | `/identity/*` (`identity_routes.py` 613) | `identity_labels_dir/identity_pairs.jsonl`, `identity_reviews.jsonl`, append-only (`identity/store.py`) | tracklets dir | none (decode per crop) | `identity/clusters`, `score_links` |
| Broadcast review `/broadcast_review` (97 lines) | `/broadcast_review/{frame,api/job,api/verdict}` (`broadcast_review.py`) | `marks/broadcast_review/<match>.json` ({frames:{n:{verdict,...}}}), atomic whole-file | job only (`?job=`); frames pre-rendered JPEG on disk | HTTP `max-age=3600` on frames | `scripts/export_broadcast_review.py` (imports `labeller.broadcast_review`) |
| Matches `/matches` | `/api/matches` (`matches_status.py` 664) | none (derived from manifests, dirs, Prefect API) | filesystem scan | whole scan 30 s TTL | none |
| Ingest `/ingest` | `/ingest/upload`, `/ingest/run`; the page route 500s (OPEN-1, measured 2026-11-17) | n/a | n/a | n/a | n/a |
| Jobs API | `/api/jobs*` (`jobs_routes.py`, `jobs.py`) | `labelling_jobs/<id>.json`: spec, items (up to 300 to 2000) and mutable state in ONE file, rewritten whole | Prefect flows (`footy_broadcast_review.py` POSTs; `footy_ballcheck.py` drives the `prepare_*` endpoints), CLI | `_FILE_CACHE` by (mtime, size) for label files; `_INLINE_CACHE` | results API `labelling-job-results/v1` |

Cross-cutting facts: one uvicorn worker (PID started 455 s before my first call, RSS 718 MB); everything shares one GIL and one event loop. The home page links, `?job=` deep links and Prefect flows (via `FOOTY_LABELLER_URL`) are the only integration surface. The autolabel queue (`~/code/jobs/footy_autolabel_queue.py:310`) still launches its own labeller per round from a pinned worktree, with env vars (`FOOTY_BALL_CHECK_LIST`, `FOOTY_RETRACK_*`) that main does not read for `RETRACK`. That is a second deployment path next to the "one site" rule.

---

## 2. Duplication and divergence

### 2.1 Repeated patterns (with where they diverge)

| Pattern | Copies (file:line) | Divergence that matters |
|---|---|---|
| Live-verdict fold (last wins, `null` = undo, default `source="sidecar"`) | `ball_check.py:204-245` (`_read_verdicts`, all files, key includes clip, **no cache**); `jobs.py:655-677` (`_verdicts`, per clip, mtime-cached, key without clip); `feature_store/ball_checks.py:95-113` (docstring says "same resolution rule" but has no shared code) | Three code paths must be kept in step by hand. Cache behaviour differs, and the per-request full-directory parse in `ball_check_queue` (`ball_check.py:466` and `:606`) never benefits from the `jobs.py` cache. |
| Atomic JSON write | `calibration.py:125-140`, `calibration/frame_type.py:46-58`, `calibration/prelabel/proposals.py:41-55`, `jobs.py:153-180`, `broadcast_review.py:71-85`, `tasks.py:80-96`, `task_done.py:196-210`, `datasets.py:360-378`, `session.py:303-314` | `fsync` is in 5 of 9, chmod handling differs (0600 vs 0664), and only `jobs.py` supports exclusive create. Sidecar edits in `review.py:341`/`:364` use plain `write_text` (non-atomic). |
| Read-modify-write locking | `frame_type.py:31` and `proposals.py` use a process `_lock`; `jobs._LOCK`, `broadcast_review._LOCK` exist; `calibration.py` GT (`_read_gt`/`_write_gt`) has none; `review.py` has none | See bug B5 and B4. |
| Clip to video file | `review.py:118` (4 extensions, `clips_dir` only), `calibration.py:595` (exact, then canonical alias, then `extra_clips_dirs`), `identity_routes.py:373` and `:509` (`.mp4` only), `session.load` (path from request) | Ball Check on `calib` cannot see clips that only live in `extra_clips_dirs` (the calibration page can). Identity cannot open `.mov`/`.mkv`. |
| Frame decode | `session.py` `frame_jpeg` (cv2 open/seek/encode), `review.py:280-300` and `:369-390`, `ball_check.render_crop_bytes`/`_render_mp4`, `identity_routes.py:373`, `calibration/video.py:84` (`read_frame`: cv2 then ffmpeg when the frame is black, per-video reader memo) | Only calibration has the black-frame fallback. The others return black stills for those clips (latent). JPEG quality differs per path (default, 80, 88, 90). |
| Clip alias to GT stem | `jobs.py:561` `_gt_stem` and `ball_check.py:1488` `_gt_sidecar_stem` (identical logic); `canonical_clip_id` used by calibration for file names, but verdict logs and sidecars use raw stems | Verdicts are keyed by the raw clip name, so the same video judged under two aliases has two verdict logs. T-03 fixed this only at export. |
| Dataset-keyed caches | `ball_check.py:98,120,127,260`, `review.py:57`, `candidates.py:89,584`, `jobs.py:541,1010`, `calibration.py:716`, `task_done.py:_cache`, `matches_status` TTL | At least 9 unrelated cache implementations. Only the Ball Check disk layer is bounded by bytes and GC'd. Keys differ: the Ball Check memory key omits box geometry (`ball_check.py:826-835`) while its disk key includes it. |
| Progress counting | `jobs.compute` (`jobs.py:706`, per kind), `ball_check_queue` `remaining/judged` (`ball_check.py:515-600`), `task_done._evaluate` (`task_done.py:160`), `ball_check_stats`, `/clips/status` `_clip_completion` (`server.py:201`, "complete" = reached last frame and has a player mark), calibration `list_clips` counts, proposals `total`, `matches_status` | Seven definitions of "done". Example: for `ball_place` a frame is done when a human ball box or `no_ball`/`not_broadcast` marker exists; for `calibration` it is GT or a `frame_type` flag. "Not broadcast" is stored twice and the two stores are never synced (`server.py:431` writes only the Session marker; `calibration.mark_not_broadcast` writes only `frame_type`). |
| Job integration | Per-kind and inconsistent: `/labeller?job=` (frames endpoint plus J/K stepping), `/ball_check?job=` (queue filter, prep progress), `/broadcast_review?job=` (own `api/job`), `calibration`/`frame_type`/`identity` (deep link only, no stepping, so "next unlabelled" is only the first link) | Same `kind` field, five different consumption paths. |
| Queue ordering/shuffle | `ball_check._shuffle_key` + `_spread_order` (`:320,363`, clip-second buckets), `candidates.stratified_order`/`adaptive_weight` (confidence band, pull-model, re-estimates tau per fetch), `review._build_review_queue` (`:200`, low-conf YOLO first, IoU dedup), list = file order, job = item order, proposals = score sort, identity `risky-frames`, broadcast review = job order | Order is decided in each endpoint. A reorder hook has nowhere to attach except the job doc. |
| Undo | Ball Check: tombstone append (`ball_check.py:1613`); labeller: client-side `pushUndo`; broadcast review: hard `DELETE` of the verdict; calibration: `DELETE gt` and `clear_not_broadcast` (restores displaced GT); review: none (delete rewrites the file); identity: none | Four semantics. |
| Keyboard/phone scaffolding | 7 pages each define their own `keydown` map (`ball_check.html:657`, `broadcast_review.html`, `index.html:573`, `calibration*.html`, `review.html:800`, `merge.html:151`), own CSS tokens (`:root` per page), own `fetch` wrappers. Only `ball_check.html` and `calibration_review.html` have prefetch; `broadcast_review.html` has none. `dataset.js` (74 lines) patches `fetch/WebSocket/img` globally but `broadcast_review.html` and `matches.html` do their own `dsq` | `konva` is loaded from `unpkg.com` on 3 pages (`index.html:7`, `review.html:7`, `calibration.html:8`). Pages are `no-store` (every navigation re-downloads 26 to 53 KB of inline HTML+JS). |

### 2.2 Where behaviour is inconsistent in ways users see
- Error signalling: `ball_check`/`review`/`identity` routes return HTTP 200 with `{"ok": false}`; `calibration` and `broadcast_review` use real status codes (the KB flagged this exact anti-pattern for calibration on 2026-09-24).
- Save feedback: `broadcast_review.html` checks `r.ok` and alerts; `ball_check.html:377` does not (B3).
- Job stepping exists for 3 of 7 kinds; calibration jobs open at the first missing frame and then the user must navigate manually.

### 2.3 Bugs and latent bugs (evidence)

| # | Severity | Finding | Evidence |
|---|---|---|---|
| B1 | High (perf/availability) | `/clips/status` is `async def` with blocking file and `cv2.VideoCapture` work per clip. It stalls the whole event loop (all phones). Measured: `/clips/status` 4.2 to 6.3 s; a concurrent `/api/server_info` waited 3.65 s (idle it is 0.009 s). Opening `/labeller` on a laptop (it calls this endpoint at load) pauses Ball Check on the phone. | `server.py:258-270`, `web/index.html` fetch of `/clips/status` |
| B2 | Medium (perf) | Home page `/api/tasks` takes 1.2 to 1.9 s (9 broadcast-review jobs). `jobs.compute` calls `br.read_store` once per item (300 items, 9 jobs, about 2.7k file reads and JSON parses) and the store has no cache. `broadcast_review.job_view` has the same bug: `stores.setdefault(it["clip"], read_store(...))` evaluates `read_store` eagerly for every item. Measured per-job `/api/jobs/<id>` 0.12 s. | `jobs.py:761`, `broadcast_review.py:143` |
| B3 | High (silent data loss) | Ball Check `judge()` removes the card, then `await fetch('/ball_check/verdict')` and only treats a thrown network error as failure. A 500 or `{"ok": false}` (the endpoint answers 200 for validation errors) is treated as saved, and the card is gone. `broadcast_review.html` checks `r.ok`. | `web/ball_check.html:377-396`; `ball_check.py:1584-1590` |
| B4 | Medium to High (lost update) | `Session._do_flush` rewrites the whole sidecar from memory with no mtime/version check. `/review/correct`, `/review/delete` (`review.py:341,364`, plain `write_text`, no lock, positional `box_index`) and Ball Check `save_box` (rewrite via `review_correct`, or append) edit the same file with no coordination. If `/labeller` still has that clip loaded and the user edits it afterwards, the next flush erases those corrections. Concurrent review deletes also shift indices (a stale page can delete the wrong box). Same data-loss class as the 2026-08 GT truncation in the KB. | `session.py:256-314`, `review.py:309-366`, `ball_check.py:1404-1486` |
| B5 | Medium (latent) | Calibration GT is read-modify-written with no lock (`_read_gt` then `_write_gt` inside `to_thread`), unlike `frame_type`/`proposals`, which use `_lock`. Concurrent saves for the same clip (or `mark_not_broadcast` and `accept` racing) lose an update. | `calibration.py:218-243,894-905`, `mark_not_broadcast` `:158-180`, `calibration_proposals.py:241-243`; `frame_type.py:31,67` |
| B6 | Low (LAN only) | Unsanitised clip name from the request body used in a path: `/ball_check/verdict` (`_verdict_path(rec["clip"])`, `mkdir` plus append) and `/review/correct` (`marks_dir / f"{clip}.jsonl"`). Identity and broadcast review sanitise (`_safe_clip`, `check_match`); these do not. | `ball_check.py:200,251`, `review.py:~315` |
| B7 | Low | `GET /ingest` 500s (OPEN-1, still open on live). `/clips/add` POST has no route (OPEN-2). | live curl 500; README section 14 |
| B8 | Medium (perf) | All identity routes block the event loop (zero `to_thread` in the module); `track-video` encodes up to 150 frames inline. `label_pair` loads all labels and rebuilds clusters inline. The same starvation PR #61 fixed for `/frame`. | `identity_routes.py:352-373,467-540,538-560` |
| B9 | Low (latent) | Black-frame fallback exists only in `calibration/video.py:84`. The labeller, Ball Check, review and identity use raw cv2, so a clip OpenCV decodes as black yields black cards. | section 2.1 |
| B10 | Low to Medium | "Not broadcast" is stored in two places (sidecar marker in `<clip>.jsonl` and `frame_type/<canon>.json`) with no link. A frame flagged in one tool is unflagged in the other, and job progress for `ball_place` vs `calibration` diverges. | `server.py:427-451` vs `calibration.py:158-180`; `jobs.py:610-638` |
| B11 | Low | GETs mutate: `GET /api/jobs/{ref}` and `/api/tasks` call `refresh_status` (rewrites the whole 70 KB job doc at the first label and at 100%); `/api/tasks` writes `.done.json`. | `jobs.py:849,917`; `task_done.py:annotate` |
| B12 | Low | `source=auto/candidates` first request after a restart took 13.3 s (then 0.14 s). `labeller_deploy.sh` warms only `source=list` of `nottinghamforest`, not candidate pools or the sidecar queue. | measured; `labeller_deploy.sh:41` (`WARM_DATASETS`) |
| B13 | Low | `/review/queue` returns every box in every sidecar (6.3 MB, 3.3 s) on every page load; the sidecar-source Ball Check queue re-scans all sidecars every fetch (1.5 to 1.9 s, no cache). | measured; `review.py:241`, `ball_check.py:581` |

---

## 3. Performance

### 3.1 Measured (GET only, load avg 13 to 17, so absolute times are pessimistic)

| Endpoint | Time | Note |
|---|---|---|
| Static pages (`/main`, `/ball_check`, `/identity/`, `/calibration`) | 4 to 12 ms | inline HTML, `no-store`; `/labeller` 107 ms (53 KB) |
| `/api/datasets`, `/api/server_info`, `/api/jobs/<ball_check id>` | 12 to 25 ms | |
| `/api/jobs?status=open` | 0.44 to 0.68 s | computes progress for all 11 jobs |
| **`/api/tasks`** (home page) | **1.2 to 1.9 s** | B2 |
| `/clips` | 51 ms | name listing only |
| **`/clips/status`** | **4.2 to 6.3 s, blocks the loop** | B1 |
| `/ball_check/queue?source=sidecar&limit=5` | 1.5 to 2.0 s (every call) | re-scans all sidecars |
| `/ball_check/queue?source=list` (Forest) | 0.05 to 0.28 s | mtime-cached list |
| `/ball_check/queue?source=candidates` (Forest) | 13.3 s cold, **0.14 s warm** | pool build; binned sampler works (KB: 1.45 s to 0.018 s) |
| `/ball_check/stats` | 35 to 70 ms | full verdict dir parse, 463 lines (158 KB) today |
| Cached Ball Check crop (prepared job, `pad=6&reticle=false`) | 18 to 83 ms (33 KB) | disk render cache |
| Cached Ball Check clip (mp4, Range-capable) | 84 ms (325 KB) | |
| **`/calibration/api/frame/<clip>/<idx>.jpg`** | **1.4 to 1.8 s, repeated identically** | no cache, no `Cache-Control` |
| `/calibration/api/clips` | 0.10 to 1.5 s | 3 file reads per clip per request, sync inside `async def` |
| `/review/queue` | 3.3 s, 6.3 MB | B13 |
| `/api/matches` | 0.95 s | 30 s TTL, fine |
| `/broadcast_review/api/job/<id>` | 0.11 s | B2 (eager `setdefault`) |

Cold renders (not re-measured here to avoid writing cache blobs; from `raw/footy-ballcheck-prepare-2026-11-17.md` and `raw/footy-offline-ball-tracking-2026-11-15-labeller.md`): 8.8 CPU-s per card for still plus clip on the loaded mini, 2.2 cards/s on the desktop; stream-copy clips with 2 to 4 keyframes make a seek cost 7 to 45 s per card, kf10 re-encodes bring a still to about 0.3 s.

### 3.2 What makes each page fast today (keep these)
- **Ball Check**: optimistic UI (card advances before the POST returns, `ball_check.html:366-376`), `PREFETCH=4` crops and `CLIP_PREFETCH=3` clips, refill at 12 cards, deterministic hash order (no reshuffle on restart), memory LRU in front of a content-keyed disk cache (`ball_check_cache.py`) with in-flight coalescing (`Coalescer`), `Cache-Control: private, max-age=600` on clips, the candidate binned pool (0.018 s), `source=list` and job `_INLINE_CACHE` keyed by mtime, warm script after restarts.
- **Calibration review**: a client request scheduler with urgent-vs-prefetch priorities.
- **Labeller**: `asyncio.to_thread` for frame/load/pause (#61), 2 s debounced flush, kf10 clips.
- **Broadcast review**: images are pre-rendered JPEGs served as static files with `max-age=3600` (cheapest path), but there is no next-frame prefetch.
- **Jobs**: `_cached(path, parse)` by (mtime_ns, size) (`jobs.py:544`).

### 3.3 Consolidation choices that would risk speed (avoid)
1. A generic `LabelStore.live()` that re-parses every verdict file per request. Keep per-file (mtime, size) caches, and for append-only logs add incremental tail-reads (offset-keyed) so a 100x bigger log stays flat.
2. Routing frame/crop decode through one shared lock, a small worker pool or a global semaphore. Today Ball Check renders in parallel; prepare is bounded separately. Put a per-dataset concurrency limit on render work only, not on the request path.
3. Replacing the Ball Check in-memory LRU with disk-only. The 0.018 s warm path depends on memory; the disk layer is for restart survival.
4. Moving the pages onto a framework or a JS bundle. Inline pages load in about 10 ms; any shared shell should be 2 plain cacheable static files (`shell.css`, `shell.js`) with a hash in the URL.
5. Running hook actions in-request. A Platt refit or reorder must run after the response (background task or Prefect), with event emission non-blocking and bounded.
6. A rewrite of `jobs.py` that keeps whole-job-document rewrites. Progress events already rewrite a 70 KB doc; splitting spec from state (below) makes that cheaper, not slower.

---

## 4. Target architecture

### 4.1 Shape

```
 pages (phone/laptop)            static shell: shell.css + shell.js (tokens, api(), keys, prefetch, toast, dataset passthrough)
   |  HTTP                       per-kind page = page-config + small kind module
   v
 FastAPI app  (one process, one loop; blocking work only in to_thread)
   |-- DatasetMiddleware (unchanged)
   |-- KindRegistry --------------------------+
   |     ball_place | ball_check | calibration | frame_type | identity | broadcast_review | ...
   |     each = LabelKind (section 4.2)       |
   |-- JobEngine  (spec + state, queue builder, results, board)
   |       ^                                  |
   |       | events                           v
   |   HookRunner  <----  LabelEvent bus  <---- LabelStore.append()   (every label write, every kind)
   |       |                       |
   |       |                       +--> events/<dataset>/<yyyy-mm>.jsonl   (append-only change feed)
   |       +--> actions: in-process registry (refit, reorder, append items) | Prefect (prepare, calibrate, ...)
   |-- RenderService (frame/crop/clip providers, memory LRU -> disk cache, coalescing, GC by job refs)
   +-- Session/WS propagation tool (unchanged core; persists via SidecarStore guard)

 files on disk (unchanged formats):  <clip>.jsonl | ball_checks/*.jsonl | calibration_gt/*.json | frame_type/*.json
                                     calibration_proposals/*.json | broadcast_review/*.json | identity_*.jsonl
 consumers:  feature_store ingest (reads files today; reads event feed tomorrow) -> DuckDB / LanceDB; snapshots; exports
```

### 4.2 `LabelKind` interface (what each kind supplies)

```python
class LabelKind(Protocol):
    name: str                                   # "ball_check"
    item_schema: type[BaseModel]                # job item shapes today in jobs._normalise_items
    def unit_keys(self, item) -> list[UnitKey]  # today jobs._item_units
    store: KindStore                            # adapter (4.3)
    def progress(self, items, live) -> Progress # done/total/next: replaces jobs.compute branches + task_done
    def order(self, pool, live, hint) -> list   # OrderPolicy: spread/hash, stratified, score-asc, job order
    def render(self, item, variant) -> RenderSpec | None   # key + pure fn (for RenderService/prepare)
    page: PageConfig                            # url, verdict set + keys, undo mode, prefetch depth, deep_link(job, next)
    def to_event(self, write) -> LabelEvent     # normalised view for hooks / feature store
```

`UnitKey = (kind, clip_canonical, frame, sub)` plus the dataset (from the ContextVar); `sub` is `box_index` or `source+box_index` for Ball Check, `None` for frames, `track_id` for identity, so job progress and hooks use one address. Canonicalise at the boundary with `clip_id.canonical_clip_id`, never rename existing files.

### 4.3 One label store abstraction, adapting over files (recommended) vs migrating

`LabelStore` API: `live(kind, clip=None) -> dict[UnitKey, Label]`, `append(kind, unit, payload, actor) -> LabelEvent`, `events(since_watermark)`. Backed by adapters that keep today's bytes on disk:

| Adapter | Kinds | Behaviour |
|---|---|---|
| `JsonlLog` | ball_check verdicts, identity | append with newline-safe prefix (`ball_check.py:1470` pattern), tombstones, incremental tail read, one fold (replaces the 3 live-verdict copies) |
| `JsonMap` | calibration_gt, frame_type, proposals, broadcast_review | one lock-guarded read-modify-write, one atomic writer (fsync, mode keep), mtime cache |
| `SidecarStore` | ball_place and object review | wraps `<clip>.jsonl`. Today's `Session` stays the editor but flushes through a guard: per-file lock, mtime check at flush, merge-or-refuse on external change (fixes B4). Review edits go through the same guard (atomic write). |

Why not migrate formats now:
- Pinned autolabel checkouts read `gt/ball_checks` directly (`footy_autolabel.py:601`: `--verdicts-dir`), and `ensure_checkout(sha)` pins old code, so the log format is a contract with code that cannot be edited.
- `ingest_gt` and LAB-1xx pin the sidecar format; 386 test functions in `tests/labeller` encode current behaviour.
- The migration value is in having one event stream, not in having one file format.

**Event log.** Every adapter `append` also writes one `LabelEvent {id (ULID), dataset, kind, unit, op (set|clear), payload, actor, job_id?, schema_v, ts}` to `events/<dataset>/<month>.jsonl`. It is a secondary write: the files stay authoritative until the event log has been verified equal for a few weeks (compare by replay, per dataset), then the feature store can ingest from events instead of scanning 5 file formats.

### 4.4 Serving "versioned datasets + per-version metrics (+ LanceDB)"
- A dataset version = `(labeller dataset, event watermark W, split manifest)`; its content is `LabelStore.live()` replayed to W. Snapshots are cheap and reproducible: `labels snapshot --dataset D --at W` writes parquet (and later a Lance table whose version is W) plus a manifest of inputs. Metrics are keyed by version id, so "metrics per version" is a join on W.
- Today the feature store is DuckDB (`feature_store/store.py`), and the docs defer lancedb to ANN needs (`docs/design/feature_store.md:725`). The design is storage-agnostic: ingest consumes `LabelEvent` records (flat, Arrow-friendly), so swapping DuckDB for LanceDB changes one sink.
- `human_checked`, provenance tier, `source`, `job_id`, `actor` stay as columns in the event payload, so tiers are filterable per version (the provenance trap in `footy-labeller-gotchas` is a consumer-side filtering bug this prevents).

### 4.5 One job/queue engine, hooks as the extension point
- Split the job doc: immutable spec (items, kind, dataset, requester) in `<id>.json`, mutable state (status, prepare, order override, counters, hook state) in `<id>.state.json`. Today every `prepare_event` progress tick rewrites 70 KB (`jobs.py:1239`), and `all_jobs()` parses every doc on each call. Add a `JobIndex` cached by dir mtime.
- Queue builder = `kind.order(pool, live, hint)` where `hint` includes the job's `order_override` (a list of unit keys, written by hooks). This unifies the pull-model sampler (`candidates.adaptive_weight`) and the push-model "reorder unanswered items" without changing either.
- Fold `prepare` into hooks: the `prepare` field, `prepare_event`, `maybe_reprepare` and `DISPATCH` (`jobs.py:1071-1300`) are already a hard-wired on-submit and on-demand-evicted hook with a Prefect action. Re-express as `HookConfig(on_submit="prefect:ballcheck-prepare", on_demand="reprepare")` so Ball Check no longer owns a private state machine. That is where the in-flight hooks branch should land: `HookRunner` subscribes to the `LabelEvent` bus; per-job triggers `every_n`, `min_interval`, `on_done`; actions from a registry (`broadcast_calibrate`, `reorder_uncertain`, `append_items` within caps, `prefect:*`); the mutation API (reorder/append) writes `state.json`.
- Hooks run after the response via a bounded queue and per-job single-flight, never inline with the write.
- Tasks become jobs of kind `other` with `done_when` (the board then has one list; `task_done` shapes become progress rules). Optional, last.

### 4.6 Shared render cache
`RenderService.get(RenderSpec)` = memory LRU, then `ball_check_cache.RenderCache` (disk, content key, job refs, GC), then render via a provider (`still`, `crop`, `clip`, `frame`). All providers share one `read_frame` (cv2 with ffmpeg black-frame fallback), one `find_video` (extensions, aliases, `extra_clips_dirs`), one size cache, a fixed JPEG quality per variant, and `Cache-Control` headers. `ball_check_cache.py` is already stdlib-only and generic; promote it rather than rewriting. Prepared flows stay unchanged (`render_crop_bytes` is already pure).

### 4.7 Shared frontend shell
Static `shell.css`/`shell.js` (hash-versioned, cacheable): tokens (the `:root` colours repeated per page), header plus progress bar, `api()` with `r.ok`/`ok:false` handling, retry and an offline queue (fixes B3 for all pages), key map helper, prefetch helper, toast, `dataset` passthrough (replace the monkey-patching in `dataset.js` with explicit `api()`/`url()` helpers, keeping the shim as a fallback during migration), self-hosted konva. Per-kind pages keep their own interaction code. Do not rewrite the Ball Check page; extract its helpers into the shell.

---

## 5. Migration plan (each step one mergeable PR, behaviour-preserving)

Ordered by value over risk. Sizes: S under 150 LOC net, M 150 to 600, L over 600.

| # | PR | Value | Test strategy (safety net) | Size |
|---|---|---|---|---|
| 1 | **Bug-fix pass**: B1 `to_thread` for `/clips/status` (and identity routes, B8); B2 per-clip store cache in `jobs.compute` and `job_view`; B3 check `r.ok`/`ok:false` in `judge()`; B5 lock calibration GT RMW; B6 sanitise clip names; B7 remove the `/ingest` page route (OPEN-1) or add the page; atomic writes in `review.py` | Fixes real stalls, silent loss and a 1.5 s home page | Existing `test_ball_check.py`, `test_review_endpoints.py`, `test_calibration.py`, `test_task_board.py`, `test_jobs.py`; add one test each: concurrent GT saves, `/clips/status` does not block (event-loop latency probe), `compute` read count (spy on `read_store`) | S to M |
| 2 | **Shared primitives, no behaviour change**: `labeller/common/{fsio,clips,verdicts}.py` (atomic write + lock helper, `find_video` + `read_frame` + clip size, the single live-verdict fold used by `ball_check`, `jobs` and `feature_store`). Delete the duplicates (`_gt_stem`/`_gt_sidecar_stem`, 9 writers, 3 folds) | Removes the divergence class; the fold becomes cacheable once | Existing LAB suites unchanged; add a **parity test** running the 3 old folds against a golden JSONL corpus (tombstones, missing `source`, bad lines) before deletion; feature_store tests (`tests/feature_store/`) | M |
| 3 | **`LabelStore` read side + `LabelKind` skeleton**: kinds implement `unit_keys`/`progress`; `jobs.compute`, `deep_link`, `board_tasks`, `task_done` call the kind. Keep every existing route and the results schema `labelling-job-results/v1` | One progress rule per kind; removes the `if kind ==` ladders | `test_jobs.py` (554 lines), `test_ball_check_prepare.py` (964), `test_task_board.py` (345), golden `GET /api/jobs/<id>/results` snapshots per kind | M to L |
| 4 | **Write side and `LabelEvent` log (dual-write, off by default)** + sidecar guard (B4: per-file lock, mtime check at flush, review/save_box through the guard) | Closes the lost-update path; creates the change feed | `test_session_dataloss.py`, `test_session_roundtrip.py`, `test_per_box_promotion.py`, new interleaving tests; replay test (events replayed equal `live()`) | M |
| 5 | **`RenderService`**: promote `ball_check_cache`; move Ball Check onto it unchanged first, then calibration frames, review crops, identity crops, `/frame/{n}` (small LRU of recent frames). Add `Cache-Control` | Calibration frame 1.4 s to milliseconds on repeat; identity decode off the loop; one cache policy | `test_ball_check.py`, `test_ball_check_prepare.py`, `test_calibration.py`; byte-identity test for prepared vs served stills (exists in prepare tests); latency probes | M |
| 6 | **JobEngine: spec/state split + `JobIndex` + `order_override`; `prepare` expressed as a hook** (keep the `prepare_*` endpoints as shims for the Prefect flow) | Cheaper progress events and board; one extension point | `test_ball_check_prepare.py` is the net; contract tests against the jobs repo flows' HTTP calls (`footy_ballcheck.py:56-72`) | L |
| 7 | **Hooks onto the bus** (land the `feat/labelling-hooks` work here): `HookRunner`, registry, `every_n`/`min_interval`/`on_done`, mutation API, `broadcast_calibrate` | The in-flight work, now with a single event source for every kind | its own tests plus an event-fan-out test per kind | M |
| 8 | **Frontend shell**: ship `shell.js/css`; migrate `broadcast_review` (trivial) then `matches`, `identity`, `calibration_review`, `calibration`, `review`, `index`, finally `ball_check` (extract only); self-host konva; job stepping for calibration and identity jobs via a shared "job bar" | One keymap/toast/save-feedback behaviour; no CDN dependency | the JS-in-Python tests (`test_calibration_editor_js.py`, `test_calibration_review_js.py`), `test_pages_carry_the_dataset_shim`, a manual phone smoke list per page | M per page (L total) |
| 9 | **Snapshot CLI + event-fed ingest** for versioned datasets/metrics (parquet first; Lance table later) | George's top priority becomes a function over `live()` and a watermark | round-trip fidelity tests in `tests/feature_store/`, snapshot reproducibility test (same W gives same hash) | M |
| 10 | **Tasks as jobs of kind `other` with `done_when`**; retire `labelling_tasks.json` | One board list | `test_task_board.py` | M (optional) |

**Explicitly NOT consolidated**
- `Session`/`BackgroundLabeller`/`motion_tracker`/WS run protocol: a stateful interactive tool with its own protocol, and the most data-loss-sensitive code in the repo. Only its persistence is guarded (step 4).
- Calibration geometry (`fit`, `warp`, `auto_points`, proposals accept/refit semantics), identity clustering and track-video, the confidence sampler math (`candidates.py`): kind-specific logic. They share I/O, frames and ordering hooks only.
- `matches_status`: a read-only status dashboard. It shares only the shell.
- On-disk label formats and verdict log naming (pinned consumers, tests, 112 existing verdict files): keep, normalise at read.
- Page interaction designs (swipe, konva editors, tracklet viewer): share the shell, not the UX.
- Prefect flows in `~/code/jobs`: keep them as HTTP clients of the job API (`/api/jobs*`); do not import labeller code there.
- The per-round autolabel review server: decide separately (open question 1) rather than silently folding it in.

---

## 6. Risks and open questions for George

1. **Second deployment path.** `footy_autolabel_queue.py` launches a labeller per round from a pinned worktree with `FOOTY_BALL_CHECK_LIST` and `FOOTY_RETRACK_*` (not read on main). Should rounds become inline-card jobs on :8011 (the jobs API already supports hidden keys and prepare)? Until then any API change must stay compatible with pinned checkouts.
2. **Verdict identity.** Verdicts are keyed by raw clip name (112 log files). Normalise at read (recommended) or migrate to canonical ids? Migration touches pinned consumers.
3. **Event log vs direct LanceDB.** Is the append-only JSONL change feed acceptable as the source for versioned snapshots, or do you want writes to go straight to a table? The feed is cheaper and reversible; direct writes need a durability story on the mini.
4. **Hooks placement and cost.** In-process actions share one uvicorn worker's GIL on a thermally limited machine. Rule proposed: anything over about 100 ms or any I/O-heavy refit goes to Prefect; only tiny actions (reorder, counters) run in-process. Confirm.
5. **Single global `Session` per dataset.** Two clients (laptop and phone) share one timeline; loading clip B replaces clip A. Acceptable, or per-clip sessions?
6. **GET mutates** (`refresh_status`, done sidecar). Move to a write path or a background tick?
7. **Results schema stability.** `labelling-job-results/v1` and the item shapes are consumed by flows and branches; the plan keeps them, but kind-specific `meta` fields need an explicit version policy.
8. **No auth.** CORS `*` on `/api/` GETs (`server.py:122`) was for :8011 to :8012 and is now moot; the single-site assumption makes it removable. Confirm before touching.
9. **Test baselines.** README section 15's traceability table (138 tests) is stale; the suite is about 386 test functions across 25 files. Update it as part of step 2 so the LAB IDs remain a usable contract. One pre-existing failure per the KB: `test_datasets.py::test_auto_points_follow_the_dataset`.
10. **Hooks worktree is empty.** I could not review its direction beyond the brief; if it already has a design doc elsewhere, reconcile with section 4.5 before step 6 to avoid two job-state models.

---

## Appendix: key file:line index
- Dataset resolution: `datasets.py:412` (`DatasetMiddleware`), `datasets.py:224-335` (registry cache).
- Sidecar persistence: `session.py:256-314` (`_do_flush`), `session.py:96-145` (load, `_load_dropped` guard).
- Review in-place edits: `review.py:309-366`; positional index `review.py:171-190`.
- Ball Check: verdict log `ball_check.py:196-257`; queue `:451-603`; crop `:808-875`; clip `:1202-1245`; save_box `:1404-1486`; prepare_plan `:1520-1578`; verdict/undo `:1579-1635`.
- Render cache: `ball_check_cache.py` (Coalescer `:124`, RenderCache `:176`, gc `:365`).
- Jobs: `jobs.py:154` (atomic write), `:183-210` (read/all), `:544` (`_cached`), `:561` (`_gt_stem`), `:655` (`_verdicts`), `:706` (`compute`), `:866` (`deep_link`), `:964` (`board_tasks`), `:1010-1050` (inline index), `:1071-1300` (prepare machinery).
- Tasks: `tasks.py:190-203`, `task_done.py:160-195`.
- Calibration: `calibration.py:125-140` (`_write_gt`), `:218-243` (`gt_save`), `:648-659` (frame), `:716-770` (auto cache); `calibration/frame_type.py:31,67`; `calibration_proposals.py:241`.
- Identity: `identity_routes.py:352-373, 467-540, 538-565`.
- Broadcast review: `broadcast_review.py:58-112` (store), `:132-152` (job view).
- Consumers: `feature_store/ball_checks.py:95-113`, `ingest_gt.py`, `ingest_calibration.py`, `scripts/export_broadcast_review.py`, `scripts/export_training_dataset.py`; jobs repo `footy_broadcast_review.py:141`, `footy_ballcheck.py:56-72`, `footy_autolabel.py:596-606`, `footy_autolabel_queue.py:310`.
