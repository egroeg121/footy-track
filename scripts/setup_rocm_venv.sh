#!/usr/bin/env bash
# Idempotent build of the merlin iGPU (Radeon 780M / gfx1103) venv: torch from the ROCm wheel index ALONE, then ultralytics --no-deps,
# then the rest from PyPI. ~17 GB. Needs no system ROCm, sudo or apt.
#   scripts/setup_rocm_venv.sh [--rocm 7.2] [--dest ~/.cache/footy_rocm_venv] [--ultralytics 8.4.215] [--no-selftest]
# Pitfalls (verified): never add PyPI as an extra index for torch (pulls CUDA torch); never build under /tmp (11 GB tmpfs); the 6.4 wheel +
# fp32 hangs the GPU; runtime needs HSA_OVERRIDE_GFX_VERSION=11.0.2 and fp16 (scripts/rocm_run.sh sets both).
# ultralytics: the repo lock has 8.3.241, but 8.4.215 is the version verified with torch 2.14.1+rocm7.2 (5.5 fps, matches the desktop). The two
# Python defaults to 3.14 (what the verified torch 2.14.1+rocm7.2 venv uses; override with FOOTY_ROCM_PYTHON).
# The two can give different box counts on one image, so candidates from this venv are not bit-identical to the desktop's; pass --ultralytics to change.
set -euo pipefail
ROCM=7.2; DEST="$HOME/.cache/footy_rocm_venv"; ULTRA=8.4.215; SELFTEST=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rocm) ROCM="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --ultralytics) ULTRA="$2"; shift 2;;
    --no-selftest) SELFTEST=0; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
case "$DEST" in /tmp/*) echo "refusing a venv under /tmp (tmpfs)" >&2; exit 2;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DEST/bin/python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
if [[ ! -x "$PY" ]]; then uv venv --python "${FOOTY_ROCM_PYTHON:-3.14}" "$DEST"; fi
have() { "$PY" -c "import $1" 2>/dev/null; }
# 1 torch, ROCm index only
if ! "$PY" -c "import torch,sys; sys.exit(0 if (torch.version.hip and '+rocm$ROCM' in torch.__version__) else 1)" 2>/dev/null; then
  uv pip install --python "$PY" --reinstall --index-url "https://download.pytorch.org/whl/rocm$ROCM" torch torchvision
fi
# 2 ultralytics without deps (its deps would drag CUDA torch in)
if ! "$PY" -c "import ultralytics,sys; sys.exit(0 if ultralytics.__version__=='$ULTRA' else 1)" 2>/dev/null; then
  uv pip install --python "$PY" --no-deps "ultralytics==$ULTRA"
fi
# 3 the rest, from PyPI (torch is already satisfied, so nothing replaces it)
uv pip install --python "$PY" numpy opencv-python-headless pillow pyyaml requests scipy matplotlib polars psutil ultralytics-thop pydantic pandas pyarrow duckdb
# 4 whatever footy_track.ball_mining.candidates / scripts.ball_pool import that is still missing
for i in $(seq 1 20); do
  miss=$(PYTHONPATH="$REPO/src" "$PY" - <<'PYEOF' 2>&1 || true
import importlib, sys
try:
    importlib.import_module("footy_track.ball_mining.candidates")
    importlib.import_module("footy_track.scripts.ball_pool")
    importlib.import_module("footy_track.datasets.detector_run")
except ModuleNotFoundError as e:
    print(e.name.split(".")[0]); sys.exit(1)
PYEOF
)
  [[ -z "$miss" ]] && break
  echo "missing module $miss: installing from PyPI"
  case "$miss" in cv2) miss=opencv-python-headless;; PIL) miss=pillow;; yaml) miss=pyyaml;; sklearn) miss=scikit-learn;; esac
  uv pip install --python "$PY" "$miss"
done
PYTHONPATH="$REPO/src" "$PY" -c "import footy_track.ball_mining.candidates, footy_track.scripts.ball_pool, footy_track.datasets.detector_run" \
  || { echo "footy_track imports still fail" >&2; exit 1; }
"$PY" -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)"
if [[ "$SELFTEST" == 1 ]]; then
  W="${FOOTY_SELFTEST_WEIGHTS:-/mnt/storage/models/rtdetr_1920_v11/best.pt}"
  if [[ -n "$(journalctl -k --since "-10 min" 2>/dev/null | grep -i "gpu reset" || true)" ]]; then echo "GPU reset in the last 10 min: skipping self-test, retry later" >&2; exit 3; fi
  echo "self-test: torch sees the GPU, then ONE RT-DETR fp16 frame (timeout 120; no retry on failure)"
  HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.2}" FOOTY_HALF=1 PYTHONPATH="$REPO/src" timeout 120 "$PY" - "$W" <<'PYEOF'
import sys, time, numpy as np, torch
assert torch.cuda.is_available(), "torch does not see the GPU"
print("device:", torch.cuda.get_device_name(0))
from footy_track.ball_mining import candidates as C
det = C.make_detector(sys.argv[1])
assert det.device == "cuda", det.device
frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
t = time.time(); det.predict_frames([frame]); print(f"1 frame fp16 ok in {time.time() - t:.1f} s (incl. warm-up)")
PYEOF
  echo "self-test passed"
fi
