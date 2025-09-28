Rea# Agent and Tooling Guidelines

This repository includes a short developer guideline for automated agents and maintainers who run programmatic edits.

- Always check available tools and the MCP/context before making automated changes.
  - Verify which tool APIs (for example: file editors, search, terminal) are available in your environment.
  - Confirm access and permissions for any remote services or repos you'll interact with.
- When converting or processing data, prefer using the repository's `docs/timings.md` and `docs/pipeline_architecture.md` for canonical time and pipeline conventions.
- Ensure any automated edits preserve existing content by reading files first and applying minimal, well-documented changes.
- Include a brief explanation of edits in commit messages or pull request descriptions.

These guidelines are intended to reduce accidental changes and help maintain consistent behaviour when using automated coding agents against this codebase.
