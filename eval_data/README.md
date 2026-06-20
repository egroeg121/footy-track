# Ball-Tracking Eval Dataset

Ground-truth annotated clips used to score competing ball-tracking methods
in the bake-off (ft-1my). All methods are evaluated on the same clips so
results are directly comparable.

## Why human-anchored GT?

Current automated detectors (YOLO, SAM3) are the *subject* of evaluation —
using them to produce ground truth would create circular scores
("agrees with broken detector"). Instead, a human manually marks the ball
center on sparse frames, exploiting temporal context (scrubbing) to find the
ball even when per-frame detectors fail.

**Sparse is enough.** Judging accuracy needs far fewer labelled frames than
training. A few hundred human-marked frames across 3-5 hard clips is sufficient
for a trustworthy ranking.

## Directory layout

```
eval_data/
    clips/
        <clip_name>.mp4          ← Video file (broadcast footage)
        <clip_name>.jsonl        ← Ground-truth labels (one JSON line per labelled frame)
    README.md                    ← This file
```

## Ground-truth JSONL format

One JSON object per **labelled** frame (unlabelled frames are simply absent).
Three valid entry types:

**Center-only** (preferred for human-anchored GT — just click the ball):
```json
{"frame_index": 5, "center": [0.451, 0.312], "tags": []}
{"frame_index": 10, "center": [0.452, 0.314], "tags": ["small_ball"]}
```

**Full bbox** (automated annotation or careful manual box drawing):
```json
{"frame_index": 0, "bbox": [0.44, 0.30, 0.02, 0.03], "tags": []}
```

**Ball absent**:
```json
{"frame_index": 1, "bbox": null, "tags": ["ball_not_visible"]}
```

Fields:
- `frame_index` (int): Zero-based frame number.
- `center` (list[float] | absent): Normalised `[cx, cy]` ball center `[0.0, 1.0]`.
  Preferred for human labelling — no box drawing needed.
- `bbox` (list[float] | null): Normalised `[x, y, w, h]` top-left bbox.
  Used when full box available; `null` means ball absent.
- `tags` (list[str]): Challenge tags (see below).

Standard tags:
- `occlusion` — ball partially or fully hidden by a player/post/crowd
- `motion_blur` — ball is blurred due to fast movement
- `small_ball` — ball appears small (far end of pitch or high camera angle)
- `crowd_background` — ball against complex crowd/advertising background
- `ball_not_visible` — ball is completely out of frame or cut from broadcast

## Human labelling workflow

Use the sparse labelling script:

```bash
uv run python scripts/label_ball_centers.py VIDEO.mp4 --step 5 --tags small_ball
```

Controls:
- **←/→** or **A/D** — move between frames (skips by `--step`)
- **Click** — mark ball center at that pixel
- **N** — mark ball as not visible
- **U** — undo label on current frame
- **S** — save progress
- **Q / Esc** — quit and save

Output: `VIDEO.jsonl` (one center entry per labelled frame).

## Adding new clips

1. Cut a short clip (10–30 s of broadcast footage, MP4).
2. Pick **hard cases**: small/distant ball, occlusion, motion blur, crowd background.
3. Place it in `eval_data/clips/`.
4. Label it: `uv run python scripts/label_ball_centers.py eval_data/clips/CLIP.mp4 --step 5`
5. Aim for ≥ 50 labelled frames per clip; ≥ 200 frames total across the dataset.

## Hard-case coverage

The eval set should include at least:

| Hard case | Why it matters |
|---|---|
| Small/distant ball | Most failures in the existing YOLO baseline (ft-55d.1 analysis) |
| Occlusion | Used to measure occlusion_recovery_rate metric |
| Motion blur | Fast-moving ball at high frame rate |
| Crowd background | Common false-positive / loss-of-track scenario |

## Metrics produced by the harness

| Metric | Priority | Description |
|---|---|---|
| `center_within_radius_pct` | **PRIMARY** | % frames where predicted center is within ball-radius (or 10px) of GT center |
| `mean_center_dist_px` | **PRIMARY** | Mean pixel distance between predicted and GT centers |
| `max_track_streak` | Key | Longest consecutive run of within-radius predictions (continuity) |
| `catastrophic_failure_rate` | Key | % predictions where center error > 50px (tracking wrong object) |
| `occlusion_recovery_rate` | Key | % of occlusion events where tracking resumed ≤3 frames later |
| `tracking_failures` | Support | Count of frames: GT present, tracker returned None |
| `mean_iou` | Secondary | Mean per-frame IoU — unreliable for tiny balls, kept for reference |
| `recall_pct` | Support | % ball-present frames where tracker returned a bbox |
| `precision_pct` | Support | % tracker predictions where GT bbox exists (anti-hallucination) |
| `fps` | Support | Tracker throughput (inference only, not data loading) |
| `peak_vram_mb` | Support | Peak GPU VRAM during inference (0 if CPU/MPS) |
| `effective_resolution_px` | Support | Median crop height fed to the tracker |

## Human visual review

In addition to numbers, render overlay videos for each method:

```python
from footy_track.ball_eval import compare_methods, EvalDataset

dataset = EvalDataset.from_dir("eval_data/clips/")
table = compare_methods(results, overlay_output_dir="eval_data/overlays/", dataset=dataset)
```

Each overlay video shows:
- **Green circle** — GT ball center (human-anchored)
- **Red circle** — predicted ball center
- **Blue box** — predicted bounding box
- **Distance text** — pixel error per frame

Watch ~30 s per method: jitter? rock solid? following a sock? instant qualitative signal.
