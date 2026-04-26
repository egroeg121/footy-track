---
description: Run classifier training with configurable hyperparameters and log results to docs/training/notable_runs.md
allowed-tools: Bash(uv run python*), Bash(uv run wandb*), Bash(grep*), Bash(echo*), Bash(date*), Read, Edit, Write
argument-hint: [--model <name>] [--dataset-version <n>] [--epochs <n>] [--freeze <n>]
---

Run classifier training using `train_classifier.py` with `DATA_ROOT` env var, then log results to `docs/training/notable_runs.md`.

Arguments: $ARGUMENTS

## Step 1: Parse arguments

Extract values from $ARGUMENTS. Use these defaults if not provided:
- `--model`: `yolo11n-cls`
- `--dataset-version`: `10`
- `--epochs`: `50`
- `--freeze`: `9`

## Step 2: Run training

Run the classifier training script with `DATA_ROOT` set to `$PWD/data`:

```bash
DATA_ROOT="$PWD/data" uv run python src/footy_track/scripts/train_classifier.py \
  --model <model> \
  --dataset-version <dataset-version> \
  --epochs <epochs> \
  --freeze <freeze>
```

Capture the full output. The script prints the run name, epoch-by-epoch metrics, and W&B run URL.

## Step 3: Extract key metrics from output

From the captured output, extract:

1. **Run name** — printed as `Starting training for run: <name>`
2. **W&B run ID** — from the line `🚀 View run ... at: https://wandb.ai/.../runs/<run-id>`
3. **Training time** — from `N epochs completed in X.XXX hours.`
4. **Final best.pt top1_acc** — the `all` row in the final validation block (after `Validating .../weights/best.pt...`)
5. **Per-epoch top1_acc** — the `all` value printed after each epoch validation pass

Then fetch final summary metrics from W&B:

```bash
uv run python -c "
import wandb
api = wandb.Api()
run = api.run('george-barnett-121/footy_scan_classifier/<run-id>')
s = run.summary._json_dict
print('top1_acc:', s.get('metrics/accuracy_top1'))
print('train_loss:', s.get('train/loss'))
print('val_loss:', s.get('val/loss'))
print('speed_ms:', s.get('model/speed_PyTorch(ms)'))
"
```

## Step 4: Append entry to docs/training/notable_runs.md

Read `docs/training/notable_runs.md` and append a new entry. If the file does not exist, create it first (see structure below).

The entry format is:

```markdown
### <run-id> — <YYYY-MM-DD>

| Parameter | Value |
|---|---|
| Model | <model> |
| Epochs | <epochs> |
| Frozen layers | <freeze> |
| Dataset version | <dataset-version> |
| Optimizer | AdamW (auto) |
| Image size | 224 |
| Device | MPS |
| Training time | ~<X> hours |

**Metrics (best.pt on val set):**

| Metric | Value |
|---|---|
| top1_acc | <value> |
| train_loss | <value> |
| val_loss | <value> |
| Inference speed | <speed_ms> ms/image |

**W&B:** `george-barnett-121/footy_scan_classifier/<run-id>`

**Observations:** <brief observations about this run>
```

## File structure (if creating from scratch)

```markdown
# Notable Classifier Training Runs

A curated log of classifier training runs for the broadcast-frame classifier
(`footy-track-broadcast-frame` Roboflow project). Each entry records the
hyperparameters, key metrics, and observations needed to understand why a run
was notable and how to reproduce it.

---

<!-- entries appended here -->
```
