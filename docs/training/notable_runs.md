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
