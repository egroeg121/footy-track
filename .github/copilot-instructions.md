

Always use your tasks to check pre-commit (using uv run prek) and then fix the results of the changed.
Where possible, write tests for the code you've written.

Write a small haiku about football and coding at the start of each session when you read these instructions.

Run pre-commit locally whenever you have made changes. Use `make pcr` to do this (which will use the prek and makefile).

Run relevant tests to check your results. You can run `uv run pytest ...` with relevant paths to run tests. You can use `-k` to run specific tests e.g. `uv run pytest "tests/<path_to_file.py>::<test_name>"` to run a specific file and test

Use the todos tool to break up tasks into smaller steps where needed and to keep track of them.

Follow the guidance within:
* docs/agent_guidelines.md
* development.md

Add instructions to those files when useful
