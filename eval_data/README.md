# Eval Data

Evaluation clips and GT labels for the ball-tracking bake-off.

## Layout

```
eval_data/
  clips/
    <clip_name>.mp4       # video clip (any length; 5-30s recommended)
    <clip_name>.jsonl     # GT labels sidecar (auto-created by the GT marking UI)
  README.md
```

## Adding Clips

Drop `.mp4` (or `.avi`, `.mov`, `.mkv`) files in `eval_data/clips/`.
The GT marking UI at http://localhost:8000/gt will discover them automatically.

### Recommended clip types for the bake-off

| Clip name prefix | What to capture |
|---|---|
| `hard_occlusion` | Ball passes behind a player or post |
| `small_distant` | Ball visible but small (far end of pitch) |
| `crowd_background` | Ball against a packed crowd |
| `motion_blur` | Fast shot or clearance |
| `easy_control` | Clear open-play tracking (control clip) |

Aim for 4-5 clips of ~10-20s each. Mark every 5th frame (≈150-250 marks total).

## GT JSONL Format

One JSON object per line, one line per labelled frame:

```jsonl
{"frame_index": 0, "center": [0.52, 0.43], "bbox": null, "tags": []}
{"frame_index": 5, "center": [0.54, 0.41], "bbox": null, "tags": []}
{"frame_index": 10, "bbox": null, "center": null, "tags": ["ball_not_visible"]}
```

Fields:
- `frame_index`: absolute frame number (0-based)
- `center`: normalised `[cx, cy]` ball center — primary format for bake-off
- `bbox`: normalised `[x, y, w, h]` top-left box — optional, not required
- `tags`: list of tags; standard values: `occlusion`, `motion_blur`, `small_ball`,
  `crowd_background`, `ball_not_visible`

Frames with no JSONL entry are unlabelled — the harness skips them for metrics.

## Running the Bake-off

Once you have GT marks, run the shootout (bead ft-5hd) to score all 4 methods:
- SOT VitTrack (method A, ft-ztw)
- SAM2 ROI (method B, ft-xps)
- ROI-YOLO (method C, ft-1d9)
- SAM3 baseline (method D, ft-76b)

The harness scores only frames that have GT marks; partial GT is fine.
