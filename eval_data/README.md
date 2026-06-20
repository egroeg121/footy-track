# Ball-Tracking Eval Dataset

Ground-truth annotated clips used to score competing ball-tracking methods
in the bake-off (ft-1my). All methods are evaluated on the same clips so
results are directly comparable.

## Directory layout

```
eval_data/
    clips/
        <clip_name>.mp4          ← Video file (broadcast footage)
        <clip_name>.jsonl        ← Ground-truth labels (one JSON line per frame)
    README.md                    ← This file
```

## Ground-truth JSONL format

One JSON object per line (no blank lines):

```json
{"frame_index": 0, "bbox": [0.45, 0.30, 0.02, 0.03], "tags": []}
{"frame_index": 1, "bbox": null, "tags": ["ball_not_visible"]}
{"frame_index": 2, "bbox": [0.46, 0.31, 0.02, 0.03], "tags": ["occlusion"]}
```

Fields:
- `frame_index` (int): Zero-based frame number.
- `bbox` (list[float] | null): Normalised `[x, y, w, h]` where `(x, y)` is
  the **top-left** corner of the ball bounding box, in `[0.0, 1.0]`.
  `null` means the ball is absent or not visible in this frame.
- `tags` (list[str]): Zero or more tags describing the challenge. Standard tags:
  - `occlusion` — ball partially or fully hidden by a player/post/crowd
  - `motion_blur` — ball is blurred due to fast movement
  - `small_ball` — ball appears small (far end of pitch or high camera angle)
  - `crowd_background` — ball against complex crowd/advertising background
  - `ball_not_visible` — ball is completely out of frame or cut from broadcast

## Adding new clips

1. Cut a short clip (10–30 s of broadcast footage, MP4).
2. Place it in `eval_data/clips/`.
3. Create the matching `.jsonl` using the labeller tool or manually.
4. Include hard cases: occluded ball, motion blur, small/distant ball,
   crowd background. At least one clip per hard-case category is recommended.

## Labelling tips

- Use the footy-track labeller (`make run` → labeller UI) to mark bboxes.
- Export via `bd labelling.py` → ground-truth JSONL format above.
- Aim for ≥ 50 labelled frames per clip; ≥ 200 frames total across the dataset.
- Include frames where the ball is absent (`bbox: null`) to test tracker precision.

## Hard-case coverage

The eval set should include at least:

| Hard case | Why it matters |
|---|---|
| Small/distant ball | Most failures in the existing YOLO baseline (ft-55d.1 analysis) |
| Occlusion | Used to measure occlusion_recovery_rate metric |
| Motion blur | Fast-moving ball at high frame rate |
| Crowd background | Common false-positive / loss-of-track scenario |

## Metrics produced by the harness

| Metric | Description |
|---|---|
| `mean_iou` | Mean per-frame IoU (GT∩Pred / GT∪Pred) |
| `recall_pct` | % ball-present frames where tracker returned a bbox |
| `precision_pct` | % tracker predictions where GT bbox exists (anti-hallucination) |
| `tracking_failures` | Count of frames: GT present, tracker returned None |
| `occlusion_recovery_rate` | % of occlusion events where tracking resumed ≤3 frames later |
| `fps` | Tracker throughput (inference only, not data loading) |
| `peak_vram_mb` | Peak GPU VRAM during inference (0 if CPU/MPS) |
| `effective_resolution_px` | Median crop height fed to the tracker (proxy for resolution used) |
