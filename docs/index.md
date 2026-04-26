# Footy Scan Documentation

Welcome to the Footy Scan documentation! This documentation will help you understand, use, and contribute to the Footy Scan project.

## What is Footy Scan?

Footy Scan is an application that:
- Tracks players and objects in football match videos
- Applies ML-based object detection and tracking
- Produces time-accurate structured data for analytics
- Supports both file-based and live stream inputs

## Getting Started

→ [System Design](system_design.md): Guiding reference for all pipeline stages
→ [Development Guide](development.md): Complete setup and development instructions
→ [Pipeline Architecture](pipeline_architecture.md): Understanding the system design
→ [Time Formats](timings.md): How we handle match timing data
→ [Data Formats](data_formats.md): Raw match layout, Roboflow dataset structure, and env vars

## Quick Development Setup

```bash
# Install the package
uv pip install -e .

# Install ML dependencies
uv pip install "torch" "opencv-python" "tqdm"

# Run the entry point
uv run footy-track
```

## Key Documentation Sections

- **[System Design](system_design.md)**
  - End-to-end pipeline stages (Input → Broadcast Classifier → Calibration → Detection → Tracking → 2D Projection → Output)
  - Per-stage inputs, outputs, and implemented vs planned status
  - Cross-stage invariants and where new component designs fit

- **[Development Guide](development.md)**
  - Environment setup
  - Development workflow
  - Project organization
  - Running tests

- **[Pipeline Architecture](pipeline_architecture.md)**
  - System components
  - Data flow
  - Extension points

- **[Time Formats](timings.md)**
  - Time representation
  - Format conversion
  - Video sync details

- **[Data Formats](data_formats.md)**
  - Raw match directory layout (`original_video/`, `full_video_frames/`)
  - Roboflow dataset structure (detection + classifier)
  - Required environment variables

- **[Agent Guidelines](agent_guidelines.md)**
  - Automated tooling rules
  - Repository conventions
