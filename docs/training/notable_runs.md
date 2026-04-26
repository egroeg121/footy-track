# Notable Classifier Training Runs

A curated log of classifier training runs for the broadcast-frame classifier
(`footy-track-broadcast-frame` Roboflow project). Each entry records the
hyperparameters, key metrics, and observations needed to understand why a run
was notable and how to reproduce it.

Use `/train-classifier` to run training and auto-append a new entry here.

---

### aakevy06 — 2026-01-25

| Parameter | Value |
|---|---|
| Model | yolo11n-cls |
| Epochs | 50 |
| Frozen layers | 9 |
| Dataset version | 10 |
| Optimizer | AdamW (auto) |
| Image size | 224 |
| Device | MPS |
| Training time | ~316 s (0.088 hours) |

**Metrics (best.pt on val set):**

| Metric | Value |
|---|---|
| top1_acc | **0.981** |
| top5_acc | 1.0 |
| train_loss | 0.024 |
| val_loss | 0.023 |
| Inference speed | 4.17 ms/image |

**W&B:** `george-barnett-121/footy_scan_classifier/aakevy06`

**Observations:** Best known result. Trained from the `football-scan` project path against a snapshot of dataset v10 where val/test splits likely contained `Unlabeled` samples (or the dataset was binary at the time), making train/val class counts consistent. Not reproducible against the current dataset v10 — see `n5fh28pv` for the reproduction attempt and root-cause analysis.

---

### ds9q9dr6 — 2026-04-26

| Parameter | Value |
|---|---|
| Model | yolo11n-cls |
| Epochs | 5 |
| Frozen layers | 9 |
| Dataset version | 10 |
| Optimizer | AdamW (lr=0.001429, momentum=0.9) |
| Image size | 224 |
| Device | MPS (Apple M4) |
| Training time | ~36 s (0.010 hours) |

**Metrics (best.pt on val set):**

| Metric | Value |
|---|---|
| top1_acc | 0.413 |
| top5_acc | 1.0 |
| train_loss | 0.149 |
| val_loss | 5.972 |
| Inference speed | 2.9 ms/image |

**W&B:** `george-barnett-121/footy_scan_classifier/ds9q9dr6`

**Observations:** Minimal 5-epoch baseline run to verify end-to-end pipeline with `DATA_ROOT` env var. Accuracy stalls at 41.3% due to train/val class mismatch (3-class train, 2-class val). Established as the baseline for further runs.

---

### n5fh28pv — 2026-04-26

| Parameter | Value |
|---|---|
| Model | yolo11n-cls |
| Epochs | 50 |
| Frozen layers | 9 |
| Dataset version | 10 |
| Optimizer | AdamW (lr=0.001429, momentum=0.9) |
| Image size | 224 |
| Device | MPS (Apple M4) |
| Training time | ~342 s (0.095 hours) |

**Metrics (best.pt on val set):**

| Metric | Value |
|---|---|
| top1_acc | 0.442 |
| top5_acc | 1.0 |
| train_loss | 0.030 |
| val_loss | 8.018 |
| Inference speed | 1.8 ms/image |

**W&B:** `george-barnett-121/footy_scan_classifier/n5fh28pv`

**Observations:** Attempted reproduction of `aakevy06` using identical hyperparameters. Top1_acc reached only 0.442 vs reference 0.981. Root cause: current dataset v10 has `Unlabeled` class in train (5 samples) but not in val/test, creating a 3-class model evaluated against a 2-class val set. Accuracy actually falls below the 55.8% majority-class baseline, indicating the model is confused by the extra class. Fix: remove or merge `Unlabeled` into `No` in the Roboflow dataset, then re-export.

---

## Regression Analysis: `aakevy06` vs `n5fh28pv` (0.981 → 0.433)

| | `aakevy06` (reference) | `n5fh28pv` (reproduction) |
|---|---|---|
| Date | 2026-01-25 | 2026-04-26 |
| W&B project | `footy_scan_classifier` | `footy_scan_classifier` |
| top1_acc | **0.981** | **0.433** |
| train/loss | 0.024 | 0.030 |
| val/loss | 0.022 | 8.018 |
| Model parameters | 1,533,666 | 1,534,947 |
| Runtime | 316 s | 348 s |

### Hyperparameters compared via the W&B API

Both run configs were fetched via `wandb.Api()` and diffed. **Every training
hyperparameter is identical**, including:

- `model: yolo11n-cls.pt`
- `epochs: 50`, `freeze: 9`, `imgsz: 224`, `batch: 16`
- `optimizer: auto`, `lr0: 0.01`, `lrf: 0.01`, `momentum: 0.937`,
  `weight_decay: 0.0005`, `warmup_epochs: 3`, `warmup_momentum: 0.8`
- `device: mps`, `cache: True`, `augment: True`, `auto_augment: randaugment`,
  `dataset_version: 10`, `seed: 0`, `deterministic: True`

The only config differences are environmental: `data` path, `save_dir`, and
the auto-generated run `name` (the `_freeze_layers=92` suffix on aakevy06's
name is YOLO's directory-collision counter — _not_ the `freeze` value, which
is `9` in both runs' configs).

### Root cause: the dataset behind `version=10` changed

The +1,281-parameter delta in the trained model (1,533,666 → 1,534,947) is
the smoking gun: the classifier head grew by one output class. Both runs ask
Roboflow for `footy-track-broadcast-frame` version 10, but the dataset was
re-uploaded between January and April 2026 to add a third class
(`Unlabeled`) which is present in `train/` only:

| Split | Classes |
|---|---|
| train | `No` (165), `Unlabeled` (5), `Yes` (222) |
| valid | `No` (46), `Yes` (58) |
| test | `No` (47), `Yes` (53) |

Ultralytics warns at startup (`found 2 classes, requires 3`) but proceeds.
The 3-output model is then evaluated against a 2-class val set: any image
predicted as `Unlabeled` is wrong by construction, and the val loss
(8.018) reflects an unnormalised softmax over a never-seen class. The 0.433
result is below the 0.558 majority-class baseline (58/104 val images are
`Yes`).

### Reproducing aakevy06's accuracy

Re-running `train_classifier.py` with aakevy06's config does **not** reach
0.981 — `n5fh28pv` is exactly that re-run, with byte-identical training
hyperparameters. The fix is on the dataset, not the script:

1. Drop or merge the `Unlabeled` class in Roboflow so train/val/test are
   consistently binary.
2. Cut a new dataset version (e.g. v11) from the corrected source.
3. Re-train with the same hyperparameters as aakevy06.

If reproducing the historical result against the current API is needed,
download the dataset snapshot used by aakevy06 directly from W&B
(`api.run(...).file('train_batch0_*.jpg')`) rather than re-pulling
`version=10` from Roboflow.

### W&B references

- Reference: <https://wandb.ai/george-barnett-121/footy_scan_classifier/runs/aakevy06>
- Reproduction: <https://wandb.ai/george-barnett-121/footy_scan_classifier/runs/n5fh28pv>

Issue: `footy_track-117`.
