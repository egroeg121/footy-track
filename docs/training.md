# Object Detector Training

This document covers how to run the object detector training script and documents baseline training results.

## Script: `train_object_detector.py`

**Location:** `src/footy_track/scripts/train_object_detector.py`

Downloads a labelled dataset from Roboflow, fine-tunes a YOLO model on it, and saves the best weights.

### Prerequisites

Set the Roboflow API key:

```bash
export ROBOFLOW_API_KEY=<your-api-key>
```

The key can be found in your Roboflow workspace settings. Alternatively the SDK reads it from `~/.config/roboflow/config.json` if you have previously run `roboflow login`.

### Usage

```bash
uv run python src/footy_track/scripts/train_object_detector.py \
  --model yolo11n \
  --dataset-version 3 \
  --freeze 9 \
  --epochs 5
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `yolo11n` | YOLO model variant (e.g. `yolo11n`, `yolo11m`). `.pt` is appended automatically. |
| `--dataset-version` | `3` | Roboflow dataset version to download. |
| `--freeze` | `9` | Number of backbone layers to freeze. Freezing early layers speeds up training when fine-tuning from a pretrained checkpoint. |
| `--epochs` | `50` | Number of training epochs. |

### Outputs

Weights and plots are saved to:
```
footy_scan_detection/<run-name>/weights/best.pt
footy_scan_detection/<run-name>/weights/last.pt
```

`<run-name>` encodes the timestamp and hyperparameters, for example:
```
2026-04-26_08-33_model_name=yolo11n_dataset_version=3_epochs=5_freeze_layers=9
```

Training metrics are also logged to Weights & Biases under the `footy_scan_detection` project.

---

## Dataset

- **Source:** Roboflow workspace `egroeg121`, project `footy-track-detection`
- **Version 3 split:** 162 train / 49 val images
- **Classes (7):** `coach`, `in_play_ball`, `person`, `player`, `player_sub`, `referee`, `+ keeper`
- **Downloaded to:** `data/detection_dataset/roboflow_dataset_<version>/`

---

## Baseline Run Results

**Run:** `2026-04-26_08-33_model_name=yolo11n_dataset_version=3_epochs=5_freeze_layers=9`

### Configuration

| Parameter | Value |
|---|---|
| Model | yolo11n (YOLO11 nano) |
| Parameters | 2,591,205 (2.6M) |
| GFLOPs | 6.4 |
| Dataset version | 3 |
| Epochs | 5 |
| Frozen layers | 9 |
| Image size | 640 |
| Device | MPS (Apple M4) |
| Optimizer | AdamW (lr=0.000909, momentum=0.9) |
| Augmentation | RandAugment + mosaic |
| Training time | ~1.3 min (0.022 hours) |

### Epoch-by-Epoch Metrics (validation)

| Epoch | box_loss | cls_loss | dfl_loss | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 1 | 1.286 | 4.163 | 0.926 | 0.0454 | 0.0135 |
| 2 | 1.223 | 3.500 | 0.882 | 0.0412 | 0.0111 |
| 3 | 1.251 | 3.026 | 0.886 | 0.0667 | 0.0214 |
| 4 | 1.251 | 2.610 | 0.876 | 0.0767 | 0.0379 |
| 5 | 1.240 | 2.276 | 0.897 | 0.0892 | 0.0482 |

### Final Validation (best.pt)

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **all** | 49 | 921 | 0.022 | 0.267 | 0.091 | 0.047 |
| player | 49 | 778 | 0.122 | 0.730 | 0.481 | 0.241 |
| coach | 8 | 13 | 0.003 | 0.615 | 0.026 | 0.020 |
| referee | 47 | 77 | 0.004 | 0.130 | 0.021 | 0.009 |
| player_sub | 3 | 8 | 0.003 | 0.125 | 0.019 | 0.010 |
| in_play_ball | 44 | 44 | 0.000 | 0.000 | 0.000 | 0.000 |
| person | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |

**Inference speed:** 76.3 ms/image (preprocess: 0.1 ms, postprocess: 65.4 ms)

### Observations

- **Player detection** is the strongest class (mAP50 = 0.481), which is expected given it dominates the training set (778 of 921 validation instances).
- **Ball and person** classes have zero mAP after 5 epochs — they are underrepresented or difficult at this scale. More epochs and/or a larger model will be needed for those classes.
- **Classification loss is still decreasing** across all 5 epochs, indicating more training would improve results. This run is a baseline only.
- **5 epochs with 9 frozen layers** is a minimal sanity-check run. For a production checkpoint, use more epochs (≥50) and consider unfreezing more layers or using a larger model variant (`yolo11m`, `yolo11l`).

---

## W&B Run

Metrics, curves and model artifacts are synced to Weights & Biases:

- **Project:** `footy_scan_detection`
- **Baseline run:** `ze990tv6`
