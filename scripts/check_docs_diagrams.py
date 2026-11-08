"""Quick text checks on the Mermaid blocks in docs/**/*.md. Exits 1 on any problem."""
import pathlib
import re
import sys

MAX_NODES = 25
bad = 0
for md in sorted(pathlib.Path("docs").rglob("*.md")):
    for n, m in enumerate(re.finditer(r"```mermaid\n(.*?)```", md.read_text(), re.S), 1):
        body, where, problems = m.group(1), f"{md} diagram {n}", []
        if not body.strip():
            problems.append("empty")
        for line in body.splitlines():
            if line.count('"') % 2:
                problems.append(f"unbalanced quote: {line.strip()}")
            for label in re.findall(r'"([^"]*)"', line):
                if re.search(r"&(?!(amp|lt|gt|quot|#\d+);)|[<>](?!br/?>)", label.replace("<br/>", "")):
                    problems.append(f"unescaped & < > in label: {label}")
        nodes = set(re.findall(r"^\s*([A-Za-z_]\w*)\s*[\[\(\{>]", body, re.M))
        if len(nodes) > MAX_NODES:
            problems.append(f"{len(nodes)} nodes (max {MAX_NODES})")
        for p in problems:
            print(f"{where}: {p}")
        bad += bool(problems)
sys.exit(1 if bad else 0)
