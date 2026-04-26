# Data Formats

This document describes the expected directory layouts for raw match data and
Roboflow training datasets, and the environment variables required by the
training scripts.

---

## Raw match layout

Match data lives under `data/` and is organised by match name:

```
data/
└── <match_name>/               # e.g. arsenal_mancity
    ├── original_video/         # source video file(s) for the match
    │   └── *.mp4               # one file per half / recording chunk
    └── full_video_frames/      # frames extracted from the source video
        └── *.jpg               # one image per extracted frame (default JPEG)
```

### `original_video/`

Contains the raw `.mp4` recording(s) for a single match.
Long recordings should be split into chunks first using `split_video.py` (see
`src/footy_track/scripts/README.md`); the resulting parts are stored here.

### `full_video_frames/`

Contains the per-frame images produced by `extract_frames.py`.
Default format is JPEG at 1 FPS; PNG and WEBP are also supported.

```bash
# Extract frames from a source video into the expected directory
uv run footy-track-extract-frames \
    data/arsenal_mancity/original_video/match.mp4 \
    --outdir data/arsenal_mancity/full_video_frames
```

Frames can optionally carry EXIF metadata (`--embed-metadata`):

| EXIF tag | Content |
|---|---|
| `video_id` | Match identifier string |
| `frame_index` | Zero-based frame number |
| `timestamp_seconds` | Wall-clock position in the source video |

Matches used in practice (referenced in notebooks):

| Match | Directory name |
|---|---|
| Arsenal vs Manchester City | `arsenal_mancity` |
| Arsenal vs Norwich | `arsenal_norwich` |
| Arsenal vs Aston Villa | `arsenal_astonvilla` |
| Arsenal vs Bournemouth (1st half) | `arsenal_bournmouth_1st_half` |
| Arsenal vs Bournemouth (2nd half) | `arsenal_bournmouth_2nd_half` |

---

## Roboflow dataset layout

Training datasets are downloaded from Roboflow into `data/` when the training
scripts run. The layout differs between the two model types.

### Detection dataset

- **Roboflow workspace:** `egroeg121`
- **Project:** `footy-track-detection`
- **Download path:** `data/detection_dataset/roboflow_dataset_<version>/`

```
data/detection_dataset/
└── roboflow_dataset_<version>/     # e.g. roboflow_dataset_3
    ├── data.yaml                   # YOLO dataset config (paths + class names)
    ├── train/
    │   ├── images/                 # training images
    │   └── labels/                 # YOLO .txt annotations (normalised xywh)
    └── val/
        ├── images/                 # validation images
        └── labels/
```

**Classes (7):** `player`, `player_sub`, `coach`, `referee`, `keeper`,
`in_play_ball`, `person`

Current default version: **3** (162 train / 49 val images).

### Classifier dataset

- **Roboflow workspace:** `egroeg121`
- **Project:** `footy-track-broadcast-frame`
- **Download path:** `data/classifier_dataset/roboflow_dataset_<version>/`

```
data/classifier_dataset/
└── roboflow_dataset_<version>/     # e.g. roboflow_dataset_10
    ├── train/
    │   ├── yes/                    # broadcast / in-play frames
    │   └── no/                     # non-broadcast frames
    ├── val/
    │   ├── yes/
    │   └── no/
    └── test/
        ├── yes/
        └── no/
```

**Classes (2):** `yes` (broadcast frame), `no` (non-broadcast frame)

Current default version: **10**.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ROBOFLOW_API_KEY` | **Yes** | Roboflow API key. Used by all training scripts and upload utilities to authenticate against the Roboflow API. Obtain it from your Roboflow workspace settings. |

The Roboflow SDK also reads the key from `~/.config/roboflow/config.json` if
you have previously run `roboflow login`.

Training metrics are logged to [Weights & Biases](https://wandb.ai) if the
`wandb` CLI is authenticated (`wandb login`). No explicit env var is required
beyond the standard `WANDB_API_KEY` that the W&B SDK reads automatically.

### W&B projects

| Script | W&B project |
|---|---|
| `train_object_detector.py` | `footy_scan_detection` |
| `train_classifier.py` | `footy_scan_classifier` |
