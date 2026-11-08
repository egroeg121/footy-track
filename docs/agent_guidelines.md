Rea# Agent and Tooling Guidelines

This repository includes a short developer guideline for automated agents and maintainers who run programmatic edits.

- Always check available tools and the MCP/context before making automated changes.
  - Verify which tool APIs (for example: file editors, search, terminal) are available in your environment.
  - Confirm access and permissions for any remote services or repos you'll interact with.
- When converting or processing data, prefer using the repository's `docs/timings.md` and `docs/pipeline_architecture.md` for canonical time and pipeline conventions.
- Ensure any automated edits preserve existing content by reading files first and applying minimal, well-documented changes.
- Include a brief explanation of edits in commit messages or pull request descriptions.

These guidelines are intended to reduce accidental changes and help maintain consistent behaviour when using automated coding agents against this codebase.

## Docs

- Read [How to contribute to the docs](contributing-docs.md) before you change any page.
- Every change updates the docs in the same PR. If your change makes a page wrong, fix the page.
- Use plain language and Mermaid diagrams, following that guide.
- Run the docs build and look at a screenshot of every diagram you add or change.
- Never put secrets, private hostnames, IP addresses or private bucket names in docs. The site is public.
- When a TODO item starts, moves or finishes, update `docs/todo.md`.

## Checks

Every agent and contributor must run `make check` (or have the pre-commit hook
installed with `make hooks`) before committing. Never use `--no-verify` to skip
the hooks. If you could not run a check (for example the machine cannot run the
tests), say so in the commit message or PR description and name the check. See
[Checks before you commit](development.md#checks-before-you-commit).
