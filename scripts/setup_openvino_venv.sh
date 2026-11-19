#!/usr/bin/env bash
# Idempotent build of the merlin CPU/OpenVINO venv (cpu torch + openvino + pinned ultralytics + footy_track imports) and the OpenVINO
# export of the given RT-DETR weights, then a short self-test. No GPU, no ROCm, no sudo.
#   scripts/setup_openvino_venv.sh [--weights W.pt] [--dest ~/.cache/footy_ov_venv] [--export-dir DIR] [--ultralytics 8.3.241]
#                                  [--imgsz 1280] [--no-selftest]
# Export goes to --export-dir, default ~/.cache/footy_ov_models/<sha8 of W.pt>/ (a dir named best_openvino_model inside it is what
# --weights of ball_eval_panel suggest / ball_pool candidates takes). Static batch 1, fp32. Load it with RTDETR(dir), NEVER YOLO(dir)
# (the YOLO postprocess gives garbage); footy_track.detectors.ultralytics does that. OpenVINO ignores OMP caps: run via scripts/ov_run.sh.
# ultralytics is pinned to the repo-lock version 8.3.241 (the version the export + 0.876 top-1@2px / 280-of-282 recall were measured with).
set -euo pipefail
W="${FOOTY_OV_WEIGHTS:-/mnt/storage/models/rtdetr_1920_v11/best.pt}"
DEST="$HOME/.cache/footy_ov_venv"; ULTRA=8.3.241; IMGSZ=1280; SELFTEST=1; EXPORT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights) W="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --export-dir) EXPORT_DIR="$2"; shift 2;;
    --ultralytics) ULTRA="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --no-selftest) SELFTEST=0; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
case "$DEST" in /tmp/*) echo "refusing a venv under /tmp (tmpfs)" >&2; exit 2;; esac
[[ -f "$W" ]] || { echo "weights not found: $W" >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DEST/bin/python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
if [[ ! -x "$PY" ]]; then uv venv --python "${FOOTY_OV_PYTHON:-3.12}" "$DEST"; fi
# 1 CPU torch (the cpu index, so no CUDA wheels), then ultralytics without deps, then the rest from PyPI
if ! "$PY" -c "import torch,sys; sys.exit(1 if torch.version.cuda or torch.version.hip else 0)" 2>/dev/null; then
  uv pip install --python "$PY" --reinstall --index-url https://download.pytorch.org/whl/cpu torch torchvision
fi
if ! "$PY" -c "import ultralytics,sys; sys.exit(0 if ultralytics.__version__=='$ULTRA' else 1)" 2>/dev/null; then
  uv pip install --python "$PY" --no-deps "ultralytics==$ULTRA"
fi
uv pip install --python "$PY" openvino numpy opencv-python-headless pillow pyyaml requests scipy matplotlib polars psutil ultralytics-thop pydantic pandas pyarrow duckdb
# 2 whatever footy_track.ball_mining.candidates / ball_pool / ball_eval_panel import that is still missing
IMPORTS='footy_track.ball_mining.candidates footy_track.scripts.ball_pool footy_track.scripts.ball_eval_panel footy_track.ball_eval.panel_suggest'
for i in $(seq 1 20); do
  miss=$(PYTHONPATH="$REPO/src" "$PY" - $IMPORTS <<'PYEOF' 2>&1 || true
import importlib, sys
try:
    for m in sys.argv[1:]:
        importlib.import_module(m)
except ModuleNotFoundError as e:
    print(e.name.split(".")[0]); sys.exit(1)
PYEOF
)
  [[ -z "$miss" ]] && break
  echo "missing module $miss: installing from PyPI"
  case "$miss" in cv2) miss=opencv-python-headless;; PIL) miss=pillow;; yaml) miss=pyyaml;; sklearn) miss=scikit-learn;; esac
  uv pip install --python "$PY" "$miss"
done
PYTHONPATH="$REPO/src" "$PY" -c "import $(echo $IMPORTS | sed 's/ /, /g')" || { echo "footy_track imports still fail" >&2; exit 1; }
# 3 export (cached by the weights sha)
SHA=$(sha256sum "$W" | cut -d' ' -f1)
[[ -n "$EXPORT_DIR" ]] || EXPORT_DIR="$HOME/.cache/footy_ov_models/${SHA:0:8}"
OVDIR="$EXPORT_DIR/best_openvino_model"
if [[ ! -f "$OVDIR/best.xml" || ! -f "$OVDIR/footy_source.json" ]]; then
  mkdir -p "$EXPORT_DIR"
  cp -f "$W" "$EXPORT_DIR/best.pt"   # ultralytics writes <stem>_openvino_model next to the .pt
  (cd "$EXPORT_DIR" && "$PY" -c "from ultralytics import RTDETR; print(RTDETR('best.pt').export(format='openvino', imgsz=$IMGSZ, half=False, dynamic=False, batch=1))")
  rm -f "$EXPORT_DIR/best.pt"
  printf '{"sha256": "%s", "weights": "%s", "imgsz": %s, "ultralytics": "%s"}\n' "$SHA" "$W" "$IMGSZ" "$ULTRA" > "$OVDIR/footy_source.json"
fi
echo "openvino model: $OVDIR"
# 4 self-test: load via the repo detector (must pick RTDETR/openvino) and run 3 frames, capped like ov_run.sh
if [[ "$SELFTEST" == 1 ]]; then
  echo "self-test (taskset 0-3, timeout 300)"
  OMP_NUM_THREADS=4 PYTHONPATH="$REPO/src" taskset -c 0-3 timeout 300 "$PY" - "$OVDIR" <<'PYEOF'
import sys, time, numpy as np
from footy_track.ball_mining import candidates as C
det = C.make_detector(sys.argv[1])
assert det.backend == "openvino" and det.device == "cpu", (det.backend, det.device)
frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
det.predict_frames([frame])  # warm-up
t = time.time(); r = det.predict_frames([frame] * 3); dt = time.time() - t
assert len(r) == 3
print(f"3 frames ok, {3 / dt:.2f} fps, backend={det.backend}")
PYEOF
  echo "self-test passed"
fi
