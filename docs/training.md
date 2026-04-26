# Training

This document covers how to run the training scripts and documents baseline results.

---

# Object Detector Training

This document covers how to run the object detector training script and documents baseline training results.

## Script: `train_object_detector.py`

**Location:** `src/footy_track/scripts/train_object_detector.py`

Downloads a labelled dataset from Roboflow, fine-tunes a YOLO model on it, and saves the best weights.

### Prerequisites

**Environment variables — `.envrc`**

A `.envrc` file (managed by [direnv](https://direnv.net/)) is provided at the repo root to set local environment variables. It is listed in `.gitignore` and is not committed.

```bash
# .envrc (already present in the repo root)
export DATA_ROOT="$PWD/data"
```

`DATA_ROOT` points to the `data/` directory at the project root, which is where downloaded datasets and other local data files are stored. Run `direnv allow` once after cloning to activate it.

Set the Roboflow API key:

```bash
export ROBOFLOW_API_KEY=<your-api-key>
```

The key can be found in your Roboflow workspace settings. Alternatively the SDK reads it from `~/.config/roboflow/config.json` if you have previously run `roboflow login`.

### Usage

The script reads `DATA_ROOT` from the environment (set by `.envrc`) and downloads the dataset to `$DATA_ROOT/detection_dataset/roboflow_dataset_<version>/`.

```bash
uv run python src/footy_track/scripts/train_object_detector.py \
  --model yolo11n \
  --dataset-version 3 \
  --freeze 9 \
  --epochs 5
```

To run without direnv active, pass `DATA_ROOT` inline:

```bash
DATA_ROOT="$PWD/data" uv run python src/footy_track/scripts/train_object_detector.py \
  --model yolo11n \
  --dataset-version 3 \
  --freeze 9 \
  --epochs 1
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
- **Downloaded to:** `$DATA_ROOT/detection_dataset/roboflow_dataset_<version>/` (defaults to `data/detection_dataset/...` if `DATA_ROOT` is unset)

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

---

# Classifier Training

This section covers the broadcast-frame classifier: a model that determines whether a video frame shows the game in play.

## Script: `train_classifier.py`

**Location:** `src/footy_track/scripts/train_classifier.py`

Downloads a labelled dataset from Roboflow, fine-tunes a YOLO classification model on it, and saves the best weights.

### Prerequisites

Same `.envrc` and `ROBOFLOW_API_KEY` setup as the object detector (see above).

### Usage

```bash
uv run python src/footy_track/scripts/train_classifier.py \
  --model yolo11n-cls \
  --dataset-version 10 \
  --freeze 9 \
  --epochs 5
```

To run without direnv active, pass `DATA_ROOT` inline:

```bash
DATA_ROOT="$PWD/data" uv run python src/footy_track/scripts/train_classifier.py \
  --model yolo11n-cls \
  --dataset-version 10 \
  --freeze 9 \
  --epochs 5
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `yolo11n-cls` | YOLO classification model variant. `.pt` is appended automatically. |
| `--dataset-version` | `10` | Roboflow dataset version to download. |
| `--freeze` | `9` | Number of backbone layers to freeze. |
| `--epochs` | `50` | Number of training epochs. |

### Outputs

Weights and plots are saved to:
```
footy_scan_classifier/<run-name>/weights/best.pt
footy_scan_classifier/<run-name>/weights/last.pt
```

`<run-name>` encodes the timestamp and hyperparameters, for example:
```
2026-04-26_09-16_model_name=yolo11n-cls_dataset_version=10_epochs=5_freeze_layers=9
```

---

## Dataset

- **Source:** Roboflow workspace `egroeg121`, project `footy-track-broadcast-frame`
- **Version 10 split:**

| Split | Images | Classes present |
|---|---|---|
| train | 392 | No (165), Unlabeled (5), Yes (222) |
| valid | 104 | No (46), Yes (58) |
| test | 100 | No (47), Yes (53) |

- **Classes:** `No` (frame not in play), `Yes` (frame in play), `Unlabeled` (ambiguous — train only)
- **Note:** `Unlabeled` is absent from val and test splits. Accuracy metrics during training are reported on the two-class (`No` / `Yes`) val set.
- **Downloaded to:** `$DATA_ROOT/classifier_dataset/roboflow_dataset_<version>/`

---

## Baseline Run Results

**Run:** `2026-04-26_09-16_model_name=yolo11n-cls_dataset_version=10_epochs=5_freeze_layers=9`

### Configuration

| Parameter | Value |
|---|---|
| Model | yolo11n-cls (YOLO11 nano classifier) |
| Parameters | 1,534,947 (1.5M) |
| GFLOPs | 3.3 |
| Dataset version | 10 |
| Epochs | 5 |
| Frozen layers | 9 |
| Image size | 224 |
| Device | MPS (Apple M4) |
| Optimizer | AdamW (lr=0.001429, momentum=0.9) |
| Augmentation | RandAugment |
| Training time | ~36 s (0.010 hours) |

### Epoch-by-Epoch Metrics (validation)

| Epoch | train_loss | top1_acc | top5_acc |
|---|---|---|---|
| 1 | 0.734 | 0.356 | 1.0 |
| 2 | 0.288 | 0.404 | 1.0 |
| 3 | 0.185 | 0.404 | 1.0 |
| 4 | 0.190 | 0.413 | 1.0 |
| 5 | 0.149 | 0.413 | 1.0 |

### Final Validation (best.pt)

| Metric | Value |
|---|---|
| top1_acc | 0.413 |
| top5_acc | 1.0 |

**Inference speed:** 0.1 ms preprocess, 2.9 ms inference, 0.0 ms postprocess per image

### Observations

- **top1_acc plateaus at 41.3%** after epoch 2 — well below a useful threshold for production use. With only the classifier head unfrozen (9 of 10 backbone layers frozen), the model has limited capacity to adapt to this task.
- **top5_acc is trivially 1.0** — there are only 3 classes, so a top-5 prediction always includes the correct label. This metric is not meaningful here.
- **Val loss diverges** while train loss falls (train: 0.149 vs W&B final val: ~5.97), indicating overfitting even at 5 epochs. The training set is small (392 images) and the val set is missing the `Unlabeled` class.
- **`Unlabeled` class** has only 5 training samples — the model cannot learn this class reliably. Consider merging it into `No` or removing it from the dataset for future runs.
- **For a production checkpoint**, unfreeze more layers (e.g. `--freeze 0`), train for more epochs (≥50), and consider a larger model (`yolo11s-cls`, `yolo11m-cls`). Data augmentation and a larger dataset would also help.

---

## W&B Runs

Metrics, curves and model artifacts are synced to Weights & Biases:

- **Project:** `footy_scan_classifier`
- **Baseline run (5 epochs):** `ds9q9dr6`
- **50-epoch reproduction run:** `n5fh28pv`

---

## Reference Run: W&B `aakevy06` (January 2026)

The best previously-known classifier result is W&B run `aakevy06`, trained in January 2026 from the `football-scan` project path. A side-by-side W&B-API hyperparameter diff against the April reproduction (`n5fh28pv`) and root-cause analysis of the 0.981 → 0.433 regression are recorded in [training/notable_runs.md](training/notable_runs.md).

### Configuration (from W&B)

| Parameter | Value |
|---|---|
| Model | yolo11n-cls |
| Epochs | 50 |
| Frozen layers | 9 |
| Dataset version | 10 |
| Image size | 224 |
| Optimizer | auto (AdamW) |
| Device | MPS |
| Runtime | ~316 s (0.088 hours) |

### Results

| Metric | Value |
|---|---|
| top1_acc | **0.981** |
| top5_acc | 1.0 |
| train/loss | 0.024 |
| val/loss | 0.023 |

---

## 50-Epoch Reproduction Run

Attempted to reproduce `aakevy06` using identical hyperparameters against the current dataset version 10.

**Run:** `2026-04-26_09-56_model_name=yolo11n-cls_dataset_version=10_epochs=50_freeze_layers=9`

### Configuration

| Parameter | Value |
|---|---|
| Model | yolo11n-cls |
| Parameters | 1,534,947 (1.5M) |
| GFLOPs | 3.3 |
| Dataset version | 10 |
| Epochs | 50 |
| Frozen layers | 9 |
| Image size | 224 |
| Device | MPS (Apple M4) |
| Optimizer | AdamW (lr=0.001429, momentum=0.9) |
| Training time | ~0.095 hours (~5.7 min) |

### Epoch-by-Epoch Metrics (validation top1_acc, selected epochs)

| Epoch | top1_acc |
|---|---|
| 1 | 0.356 |
| 2 | 0.404 |
| 5 | 0.413 |
| 7 | **0.442** ← best |
| 10–49 | 0.433 (flat) |
| 50 | 0.433 |

### Final Validation (best.pt)

| Metric | Value |
|---|---|
| top1_acc | 0.442 |
| top5_acc | 1.0 |

**Inference speed:** 0.1 ms preprocess, 1.8 ms inference, 0.0 ms postprocess per image

### Results vs Reference

| | Reference `aakevy06` | This run |
|---|---|---|
| top1_acc | **0.981** | 0.442 |
| train/loss | 0.024 | 0.030 |
| val/loss | 0.023 | 8.018 |

### Why the Results Differ

The reproduction run achieved **0.442 top1_acc**, far below the reference run's **0.981**. The cause is a dataset class mismatch:

- **Training set (current):** 3 classes — `No` (165), `Unlabeled` (5), `Yes` (222)
- **Val/test set (current):** 2 classes — `No` and `Yes` only; `Unlabeled` is absent

Ultralytics warns about this at startup (`found 2 classes, requires 3`) but proceeds. The 3-class model is evaluated against a 2-class val set, so any image predicted as `Unlabeled` is counted wrong. Additionally, 0.442 is below the majority-class baseline of 0.558 (58/104 val images are `Yes`), suggesting the frozen backbone's learned representations do not transfer well to this task with only 5 `Unlabeled` training samples distorting the class boundaries.

The January 2026 reference run (`aakevy06`) likely used a snapshot of dataset version 10 where the val/test splits also contained `Unlabeled` samples (making evaluation consistent), or an earlier version where the dataset was binary. The dataset has since diverged.

### Recommended Fix

To reproduce reference-level accuracy:
1. **Remove or merge `Unlabeled`** — relabel `Unlabeled` images as `No` or drop them. This makes training and validation consistent (binary `No`/`Yes`).
2. **Re-download a corrected dataset version** from Roboflow once the split is fixed.
3. Consider using `--freeze 0` to allow full fine-tuning for better adaptation to this domain.
