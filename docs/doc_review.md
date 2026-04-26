# Documentation Review — `docs/`

A holistic alignment audit of every Markdown file under `docs/` against the
canonical system description.

**Note on the reference doc.** The originating issue (`footy_track-5h0`)
asks for alignment against `system_design.md`, but no such file exists in
the repo. The closest match — and the file explicitly self-identifying as
"the single authoritative reference for contributors and agents"
(`docs/system_overview.md:3`) — is **`system_overview.md`**, which is what
this review uses as the canonical baseline. **Recommend either renaming
`system_overview.md` → `system_design.md` or updating downstream docs/beads
to use the existing name.**

This is a flag-only report; fixes are out of scope.

---

## Summary table

| Doc | Aligned with system_overview.md? | Code drift? | Internal contradictions |
|---|---|---|---|
| `index.md` | Partial | Naming inconsistency | Footy Scan vs footy-track vs football-scan |
| `development.md` | Partial | `cd football-scan` ≠ package name | Same naming issue |
| `pipeline_architecture.md` | Good | None | Tracking status unclear |
| `pipelines.md` | **Poor — describes a 6-stage design that does not exist in code** | Massive | Contradicts system_overview.md outright |
| `timings.md` | Good | None | None |
| `data_formats.md` | Poor | Entry points + class list wrong | Class list differs from training.md |
| `training.md` | Partial | "keeper" class doesn't exist in code | Class list differs from data_formats.md |
| `training/notable_runs.md` | Good | None | Reference run only reproducible with the (unversioned) local binary dataset |
| `design/player_tracking_format.md` | Good (explicitly DRAFT) | None — Parquet schema not yet implemented | None |
| `agent_guidelines.md` | Good | None | Has a typo on line 1 |

---

## Critical findings (action recommended)

### 1. `pipelines.md` describes a system that doesn't exist
`docs/pipelines.md:13–175` describes a six-stage video pipeline (frame
embeddings, overhead-shot classification via clustering, OCR for the in-game
clock, field-line matching, homography estimation, camera-geometry
synchronisation). None of this exists in `src/`:

- no embeddings module
- no OCR
- no field-line / homography code
- no overhead-shot classifier in the live pipeline (`classifier.py` is used
  for Roboflow dataset *labelling*, not live filtering — `labelling.py`)

`system_overview.md:7–31` describes a three-stage pipeline
(InputConsumer → Processor (Detection/Tracking/Event) → OutputProducer)
which matches the actual code.

Neither doc labels its status. A reader cannot tell which is current and
which is aspirational. **Either delete `pipelines.md`, mark it
`DRAFT / future-state` at the top, or fold its concrete bits into
`pipeline_architecture.md`.**

### 2. Detection-class lists disagree across three docs and the code
The detection class taxonomy is given inconsistently in three places:

| Source | Classes |
|---|---|
| `system_overview.md:121–135` | `person`, `ball`, `in_play_ball`, `out_of_play_ball`, `player`, `player_sub`, `referee`, `coach` (**8**) |
| `training.md:90` | `coach`, `in_play_ball`, `person`, `player`, `player_sub`, `referee`, `+ keeper` (**7+keeper**) |
| `data_formats.md:83–84` | `player`, `player_sub`, `coach`, `referee`, `keeper`, `in_play_ball`, `person` (**7+keeper**) |
| `src/footy_track/constants.py:9–16` (truth) | `ball`, `person`, `out_of_play_ball`, `in_play_ball`, `player`, `player_sub`, `referee`, `coach` (**8**) |

`keeper`/`goalkeeper` is **not** a constant. `detectors/ultralytics.py`
collapses goalkeeper detections to `PLAYER_TAG`. **Two docs invent a
`keeper` class that the code doesn't have.** Pick one canonical taxonomy
(probably `constants.py`) and align the other docs.

### 3. Non-existent CLI entry points referenced
`docs/data_formats.md:35` tells the reader to run:

```bash
uv run footy-track-extract-frames data/arsenal_mancity/original_video/match.mp4 ...
```

The only entry point defined in `pyproject.toml` is
`scripts.footy-track = "footy_track:main"`. There is no
`footy-track-extract-frames`, no `footy-track-classify-frames`, and no
training-script entry point. Anyone copy-pasting this fails immediately.
The actual scripts live at `src/footy_track/scripts/*.py` and must be
invoked as `uv run python src/footy_track/scripts/extract_frames.py …`.

### 4. `agent_guidelines.md` has a typo on line 1
`docs/agent_guidelines.md:1` reads `Rea# Agent and Tooling Guidelines` —
the leading `Rea` is garbage and breaks the H1. Should be
`# Agent and Tooling Guidelines`.

---

## Other gaps and inconsistencies

### 5. Tracking status is unclear
`system_overview.md:49–54` calls Stage 2 (Tracking) "planned" — not yet
implemented. `pipeline_architecture.md:13–14` describes tracking as one of
three optionally-decomposable stages without saying whether it currently
exists. There's no single doc stating the current status, the roadmap, or
how a contributor would enable it. The repo *does* have a `tracking/`
concept referenced (e.g. `lap` for Hungarian assignment per
`/Users/georgebarnett/code/CLAUDE.md`) but no doc describes the actual
implementation surface.

### 6. Broadcast classifier role is mis-described
`system_overview.md:60–64` and the design notes around line 214 imply the
broadcast-frame classifier is an optional pre-filter for live detection.
In code, `classifier.py` + `labelling.py` is a *dataset-curation* tool that
labels Roboflow uploads as `Yes` / `No`. There is no live filtering happening
in the detection pipeline.

### 7. Metaflow is acknowledged but not documented
`system_overview.md:204–206` mentions `metaflow/` exists for "batch
processing pipelines" but says nothing more. No doc describes which flows
exist, how to run them, or whether they're production / experimental. The
`metaflow/` directory in the repo has flow definitions but no README and no
mkdocs nav entry.

### 8. Naming drift: `Footy Scan` vs `footy-track` vs `football-scan`
- `index.md:3` and `index.md:7`: "Footy Scan" (marketing).
- `pyproject.toml:9`: package name `footy-track`.
- `development.md:10`: tells you to `cd football-scan` (a path that
  doesn't match the package). The aakevy06 W&B run was historically also
  produced from a `football-scan` directory.

This is mostly cosmetic but onboarding contributors will be briefly
confused. Pick one external name and one repo/dir name and flatten the
others.

### 9. SAM3 detector — undocumented config surface
`system_overview.md:45–46` describes the SAM3 detector capabilities
(text prompts, per-prompt confidence thresholds, centre-distance NMS) but
not that prompts and thresholds are hard-coded in `_prompt_specs`
(`detectors/ultralytics.py:113–131`) or that there is a `bbox_padding_percent`
parameter (line 102). For anyone trying to retune it, this matters.

### 10. `training.md` reference run note is missing
`training/notable_runs.md` is internally consistent and the reference-run
analysis is solid. One small gap: it doesn't note that `aakevy06`'s
reproducer (`nsrl1x1g`) only succeeded against a *local* binary dataset
that is not under DVC or version control. There is no path to reproduce
`aakevy06` from a clean checkout. **Recommend filing a bead to put the
binary v10 dataset under DVC or to cut a Roboflow `version=11` with the
`Unlabeled` class removed (already discussed in `notable_runs.md`).**

---

## Recommended priority order for follow-up beads

1. **(high)** Resolve `pipelines.md` vs `system_overview.md`: either mark
   `pipelines.md` as future-state or delete it.
2. **(high)** Single-source the detection class taxonomy from
   `constants.py` and update `system_overview.md`, `training.md`,
   `data_formats.md` to match.
3. **(high)** Fix the broken `footy-track-extract-frames` invocation in
   `data_formats.md:35` (and audit other docs for similarly-stale
   entry-point examples).
4. **(low)** Fix the `Rea# …` typo in `agent_guidelines.md:1`.
5. **(med)** Decide whether `system_overview.md` should be renamed to
   `system_design.md` so the bead language matches reality, or update the
   bead/process language to match the existing filename.
6. **(med)** Document `metaflow/` flows; either give them their own page
   or expand `system_overview.md:204–206`.
7. **(med)** Clarify tracking status in one canonical place.
8. **(low)** Flatten the `Footy Scan` / `footy-track` / `football-scan`
   naming.
