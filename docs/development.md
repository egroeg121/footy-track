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
uv run pytest        # everything (also: make tests)
make test-fast       # skips tests marked slow or needs_weights
```

Tests marked `needs_weights` skip themselves when the model weights
(e.g. `model_saves/sam3/sam3.pt`) have not been pulled from DVC/LFS.

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


## Checks before you commit

Two kinds of automatic check protect `main`: small, fast ones that run on your
machine each time you commit, and a fuller set that runs on GitHub for every
pull request (PR). The fast ones catch most mistakes before anyone else sees
them.

**One-time setup.** Run `make hooks`. This installs a git hook using
`uvx pre-commit`, so there is nothing new to add to the project.

**What runs when you commit.** Each commit runs these hooks on the files you
changed:

- `ruff` checks the Python for mistakes and fixes what it safely can.
- `ruff-format` lays the code out in the project style.
- Small file checks: a newline at the end of each file, no trailing spaces,
  valid YAML and TOML, no leftover merge-conflict markers, and no file over
  500 KB.

If a hook changes a file, the commit stops. Look at the change, `git add` the
files again and commit again. If a hook reports a problem it cannot fix, fix it
by hand and commit again.

**Run everything yourself.**

```bash
make check    # lint + fast tests, the same things CI runs
make lint     # ruff check + ruff format --check only
make format   # let ruff fix formatting and simple lint problems
```

**What runs on GitHub.** Every PR, and every push to `main`, runs three jobs:

| Job | What it does |
| --- | --- |
| `lint` | Runs the pre-commit hooks on every file in the repo. |
| `tests` | Installs the project and runs `make test-fast` (skips slow tests and tests that need model weights or API keys). |
| `docs` | Builds the docs site, to catch broken pages. It is not strict yet, because `system_design.md` has two known broken links. |

A new push to the same branch cancels the run that is still going.

```mermaid
flowchart TD
  classDef you fill:#dbeafe,stroke:#2563eb,color:#111
  classDef auto fill:#e5e7eb,stroke:#6b7280,color:#111
  A["Edit code and docs"]:::you --> B["git commit:<br/>pre-commit hooks run"]:::auto
  B -->|"a hook fails or fixes a file"| A
  B -->|"all hooks pass"| C["git push and open a PR"]:::you
  C --> D["CI runs: lint, tests, docs build"]:::auto
  D -->|"a job fails"| A
  D -->|"all jobs pass"| E["Review"]:::you
  E --> F["Merge to main"]:::you
  F --> G["Docs deploy to GitHub Pages"]:::auto
```

Legend: blue boxes are things you do; grey boxes happen by themselves. The
arrows going back to "Edit" show that a failed check sends you back to fix it.

Never skip the hooks with `git commit --no-verify`. If a check is wrong, fix
the check in its own PR.

Planned: once the docs branch with `scripts/check_docs_diagrams.py` is merged,
add a hook that runs it on changed docs.
