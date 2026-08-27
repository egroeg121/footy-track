#!/usr/bin/env bash
# Runs ON the box. Roboflow writes an ABSOLUTE `path:` into every data.yaml,
# pointing at whatever machine downloaded it. Rsync'd to a rented box that path
# does not exist and ultralytics fails with a confusing "dataset not found".
# Rewrite it to point at wherever the yaml actually lives now. Idempotent.
set -euo pipefail
REMOTE_DIR="${1:-/root/footy_track}"
DATA_ROOT="$REMOTE_DIR/data/training_datasets"
[ -d "$DATA_ROOT" ] || { echo "[fix-paths] no dataset dir — nothing to do"; exit 0; }

find "$DATA_ROOT" -name 'data.yaml' -print0 | while IFS= read -r -d '' yaml; do
  python3 - "$yaml" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
want = str(p.parent)
txt = p.read_text()
new, n = re.subn(r'(?m)^path:.*$', f'path: {want}', txt)
if n and new != txt:
    p.write_text(new)
    print(f"[fix-paths] {p}: path -> {want}")
else:
    print(f"[fix-paths] {p}: already correct")
PY
done
