# Footy Scan

Footy Scan is an application for tracking players, actions, and events in a football match. It consumes video, applies ML-based object detection and tracking, and emits structured, time-accurate match data suitable for analytics or downstream pipelines.

📚 [**Read the full documentation**](https://yourorg.github.io/football-scan)

## Overview

- Purpose: detect and persistently track players (and other relevant objects) in video, infer actions/events, and export those observations in a simple machine-readable format.
- Input: video files or live streams that are unwound frame-by-frame and processed over time by the pipeline.
- Output: continuous, time-based records that include timestamps, player IDs, bounding boxes, confidences and event metadata.

## Quick Start

We use `uv` for Python environment and dependency management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package
uv pip install -e .

# Run the documentation server
uv run mkdocs serve
```

Visit http://127.0.0.1:8000 for the full documentation.

## Makefile quick start

You can use the Makefile for common tasks:

```bash
# Discover available commands
make help

# Install deps and project in editable mode
make setup
```

## Key points about outputs

- Outputs may be produced at variable frequency and timing — events and per-frame outputs are time-accurate and should be treated as continuous-time submissions.
- Each record MUST include an exact timestamp (wall-clock or video timecode) so data can be synchronized, merged, or downsampled reliably.
- Analysis or downstream systems can resample or aggregate this continuous data to fixed frequencies if required (for example, 25 Hz, 1 Hz, or per-event).
- Preferred serialisation formats:
  - JSON: for rich, nested event objects (recommended for event and trajectory records).
  - CSV: for flattened, tabular exports that are easy to ingest into analytics tools. Treat CSV as a table view of the same underlying objects.

## Time formats and conversion

All timestamps in stored outputs use ContinuousTime (seconds from game start). The project also recognizes GameTime (referee/broadcast clock) which resets at half-time. Use GameMetadata from video sources to convert GameTime to ContinuousTime when ingesting frames.

- ContinuousTime: continuous seconds from game start (0.0 = kickoff). Does not reset at half-time; includes stoppage time — i.e. second-half ContinuousTime will not necessarily start at 45:00 because the first half may include stoppage.
- GameTime: clock shown in broadcast/referee feed; resets at half-time and reflects per-half stoppage.

For video inputs, parse GameTime and use GameMetadata to compute ContinuousTime per frame. For first-half frames, ContinuousTime typically equals GameTime. For second-half frames, add the second-half kickoff offset (ContinuousTime when second half began) so ContinuousTime remains continuous.

See `docs/timings.md` for more details and examples.

## Pipeline

The high-level pipeline is intentionally small and modular:

- InputConsumer
  - Reads video (file or stream), decodes frames, and supplies them with timestamps to the Processor.

- Processor
  - Core processing stage. Implementations may be a single monolithic processor or split into the three sub-steps below:
    1. Detection — run object detection models to locate players, ball, referees and other objects per frame.
    2. Tracking — associate detections over time to assign persistent IDs to players and objects (multi-object tracker).
    3. Event extraction — infer actions and events (passes, shots, tackles, set-pieces, substitutions) using rules and/or ML classifiers that consume tracked trajectories and detection history.
  - Note: these three steps can be combined into a single Processor implementation or separated into distinct components depending on performance and design goals.

- OutputProducer
  - Serializes the results from the Processor and writes them to storage, a message bus, or an API. It is responsible for formatting timestamps and choosing JSON/CSV/table outputs.

See `docs/pipeline_architecture.md` for design and extension notes.

## Documentation

- 📖 [Getting Started Guide](docs/development.md)
- 🏗️ [Pipeline Architecture](docs/pipeline_architecture.md)
- ⏱️ [Time Formats and Conversion](docs/timings.md)
- 🤖 [Agent Guidelines](docs/agent_guidelines.md)

## Development

See our [Development Guide](docs/development.md) for complete setup instructions. We use:

- `uv` for Python environment and dependency management
- MkDocs with Material theme for documentation
- Python 3.12+ for implementation
- Core ML dependencies: torch, opencv-python, tqdm

```bash
# Quick development setup
uv pip install -e .
uv pip install "torch" "opencv-python" "tqdm"  # ML dependencies

# Run documentation locally
uv run mkdocs serve
```

If you prefer pinning versions, create a small requirements file (for example `requirements-ml.txt`) and install from that.

## Running

- The project exposes a simple script entry point (configured in `pyproject.toml`). Run the package's main entry point to verify the install:

   footy-track

This will run the package entry point defined in `pyproject.toml` and print a simple startup message until you replace `main()` with your pipeline.
