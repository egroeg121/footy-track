#!/usr/bin/env bash
# Runs ON the Vast box. Idempotent — safe to re-run on an existing instance.
# Invoked by up.sh; nothing here may assume prior state.
set -euo pipefail

REMOTE_DIR="/root/footy_track"
VENV="/venv/main"

info() { printf '\033[36m[provision]\033[0m %s\n' "$*"; }

# The image ships torch built against the host driver. Reinstalling it via the
# repo's deps would pull a generic wheel and can silently lose CUDA, so we
# install INTO the image venv and let uv see torch as already satisfied.
[ -d "$VENV" ] || { echo "expected image venv at $VENV" >&2; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

info "baseline: $(python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())')"

command -v uv >/dev/null 2>&1 || { info "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"

info "installing footy-track deps into $VENV"
cd "$REMOTE_DIR"
uv pip install --python "$VENV/bin/python" -e . 2>&1 | tail -5

info "verifying CUDA survived the install"
python - <<'PY'
import torch, ultralytics
assert torch.cuda.is_available(), "CUDA disappeared after dep install — check the torch wheel"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0),
      "|", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("ultralytics", ultralytics.__version__)
PY

# Ephemeral discipline: anything not written here and shipped off-box is lost on
# destroy. down.sh pulls these two directories and nothing else.
mkdir -p "$REMOTE_DIR/runs" "$REMOTE_DIR/model_saves"

# Make the venv + env vars present for non-interactive ssh too, not just login
# shells (the image only wires nvm/venv for interactive use).
grep -q 'vast-provision' /root/.bashrc 2>/dev/null || cat >> /root/.bashrc <<'RC'
# vast-provision
source /venv/main/bin/activate
[ -f /root/.footy_env ] && set -a && . /root/.footy_env && set +a
cd /root/footy_track 2>/dev/null || true
RC

bash "$REMOTE_DIR/scripts/vast/fix_dataset_paths.sh" "$REMOTE_DIR"

if [ -d "$REMOTE_DIR/data/training_datasets" ]; then
  info "done — dataset present ($(du -sh "$REMOTE_DIR/data/training_datasets" | cut -f1))"
else
  info "done — NO dataset on box; run: scripts/vast/sync.sh up --data"
fi
