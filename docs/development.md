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