# Eval Data

Human-anchored ground-truth for ball-tracking evaluation.

## Layout

```
eval_data/
    clips/
        <clip_name>.mp4        # video clip to label
        <clip_name>.jsonl      # sidecar: one JSON object per labelled frame
    README.md
```

## How to add a clip

Drop any `.mp4` (or `.avi`, `.mov`, `.mkv`) into `eval_data/clips/`.
Then open the web labeller at **http://localhost:8000/gt** to mark ball centers.

Run the server with:
```bash
uv run uvicorn footy_track.labeller.server:app --reload
```

## Labelling workflow

1. Open http://localhost:8000/gt in your browser
2. Pick your clip from the left sidebar
3. For each frame: **click the ball center** on the canvas
   - Arrow keys / A-D to navigate frames
   - `N` = ball not visible this frame
   - `U` = undo / clear the current frame's mark
4. Marks save **incrementally** as you go — no explicit save needed

## JSONL format

Each line in `<clip>.jsonl` is one JSON object:

```jsonl
{"frame_index": 0, "bbox": null, "tags": [], "center": [0.512, 0.334]}
{"frame_index": 5, "bbox": null, "tags": ["ball_not_visible"]}
```

- `center`: normalised `[cx, cy]` from top-left (range 0–1 for both axes)
- `bbox`: always `null` for center-only labels; set for full-box annotations
- `tags`: `["ball_not_visible"]` when ball is absent/occluded

## Using labels in code

```python
from footy_track.ball_eval import EvalDataset

dataset = EvalDataset.from_dir("eval_data/clips/")
for clip in dataset:
    for fi, lbl in clip.labels.items():
        if lbl.is_ball_visible():
            cx, cy = lbl.ball_center()
            print(f"{clip.name} frame {fi}: ball at ({cx:.3f}, {cy:.3f})")
```
