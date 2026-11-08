# Development Guide

This guide covers how to set up, develop, and work with the Footy Scan codebase.

## Setting Up Development Environment

1. Clone the repository:
```bash
git clone <repository-url>
cd football-scan
```

2. Python Environment:
We use `uv` for Python environment and dependency management. First, install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install the package in development mode:
```bash
uv pip install -e .
```

4. Install additional dependencies for ML and video processing (if needed):
```bash
uv pip install "torch" "opencv-python" "tqdm"
```

## Using the Makefile (recommended)

Common repo-wide tasks are captured in the Makefile to standardize workflows:

make setup
make deps-update

## Development Tools

### Documentation
We use MkDocs with the Material theme for documentation. To work with docs:

```bash
# Serve docs locally (with live reload)
uv run mkdocs serve

# Build the documentation site
uv run mkdocs build

# Deploy to GitHub Pages (if configured)
uv run mkdocs gh-deploy
```

The documentation will be available at http://127.0.0.1:8000 when serving locally.

### Running Tests
```bash
uv run pytest
```

### Code Organization

```
football-scan/
├── docs/               # Documentation
├── src/
│   └── footy_track/   # Main package code
│       ├── __init__.py
│       └── main_entrypoint.py
├── tests/             # Test files
├── mkdocs.yml        # Documentation configuration
└── pyproject.toml    # Project configuration
```

## Project Conventions

### Time Handling
The project uses two main time formats:
- ContinuousTime: Continuous seconds from game start
- GameTime: Referee/broadcast clock time

See [Time Formats](timings.md) for detailed conversion rules and examples.

### Pipeline Architecture
The processing pipeline consists of three main components:
- InputConsumer: Video frame input
- Processor: Detection and tracking
- OutputProducer: Data serialization

See [Pipeline Architecture](pipeline_architecture.md) for detailed design information.

### Automated Tools
When using automated tools or agents with this codebase, follow the guidelines in [Agent Guidelines](agent_guidelines.md) to maintain consistency.

### Docs
To change the docs, read [How to contribute to the docs](contributing-docs.md).

## Environment notes

Who this is for: anyone setting up a new machine, especially an older or unusual one. These notes were measured on the review's build machine on 2026-11-08.

- **Old CPUs without AVX2 crash on import.** The build machine's CPU lacks AVX2. The newest `duckdb` and the stock `polars` die with an illegal instruction there. The workaround is `duckdb==1.3.2` and `polars-lts-cpu` in the virtual environment only. `pyproject.toml` asks for `duckdb>=1.5.3`, and `polars` arrives through `ultralytics`; do not edit `pyproject.toml` or `uv.lock` to work around a local CPU.
- **Open the feature store read-only when inspecting it:** `duckdb.connect(path, read_only=True)`. See [Feature store design, section 7.1](design/feature_store.md#71-keys-as-implemented-and-what-is-not-in-the-store).
- **Weights are not in git.** `model_saves/sam3/sam3.pt` and the detector checkpoints come from the object store or DVC/LFS. Tests that need them fail without the file (see below).
- **No GPU is needed for the tests.** No test uses CUDA (checked 2026-11-08). The slow ones are model tests that run on the CPU.
- **Where the data is** is covered in [Datasets and labels](data.md#where-things-live); the labeller finds it through `FOOTY_CLIPS_DIR`, `FOOTY_GT_MARKS_DIR` and `FOOTY_BALL_CANDIDATES_DIR`.

## Running the tests

The suite has 462 tests (collected 2026-11-08). The whole run took about 4 m 30 s before the speed-up and about 52 s after it (measured on the build machine, same date).

**Why it was slow.** The guess was a GPU problem. It was not: no test uses CUDA. Three things took almost all the time: two SAM3 tests (about 75 to 80 s each, and they fail anyway when the weights file is missing) and the first `fiftyone` test (about 17 s of start-up).

**What fixes it.** The branch `chore/faster-tests` is pushed but **not merged yet**. Once it is merged you get:

| Command | What runs | Time (measured on that branch) |
|---|---|---|
| `uv run pytest` or `make tests` | Everything | about 52 s |
| `make test-fast` | Skips tests marked `slow` or `needs_weights` | about 25 s |

The `needs_weights` marker is available once `chore/faster-tests` is merged. It is for tests that need model weights that are not in the repo; they skip themselves when the file is missing. The branch also makes the `extract_frames` test fixture create its output folder first, because `ffmpeg` does not.

```mermaid
flowchart LR
  classDef code fill:#dbeafe,stroke:#2563eb,color:#111
  classDef new fill:#ede9fe,stroke:#7c3aed,color:#111
  classDef data fill:#e5e7eb,stroke:#6b7280,color:#111

  ALL["make tests<br/>462 tests, about 52 s"]:::code
  FAST["make test-fast<br/>about 25 s"]:::new
  SAM["2 SAM3 tests<br/>about 75 to 80 s each<br/>needs_weights, slow"]:::new
  FO["fiftyone tests<br/>about 17 s start-up<br/>slow"]:::new
  W[("sam3.pt weights<br/>not in git")]:::data

  ALL --> SAM
  ALL --> FO
  FAST -. "skips" .-> SAM
  FAST -. "skips" .-> FO
  W --> SAM
```

Legend: blue is what exists on `main` today, purple is new on the unmerged `chore/faster-tests` branch, grey is data not in git. The diagram shows that the full run includes the two SAM3 tests and the `fiftyone` tests, and that `make test-fast` skips both groups.

CI does not run the tests today ([T-08](todo.md)).
