#!/usr/bin/env bash
# Run python in the merlin OpenVINO CPU venv, pinned to 4 cores and under a timeout. For SMALL jobs only (eval-panel suggestions,
# one-off single-clip ball candidates): ~2 fps for RT-DETR-L @1280. Full matches stay on the desktop GPU.
#   scripts/ov_run.sh -m footy_track.scripts.ball_eval_panel suggest --match M --weights <OpenVINO dir or .pt>
# Env (overridable): FOOTY_OV_VENV (~/.cache/footy_ov_venv), FOOTY_OV_CPUS (0-3), FOOTY_OV_THREADS (4),
#   FOOTY_OV_TIMEOUT (seconds, default 7200 = 2 h), FOOTY_SRC (<repo>/src, the PYTHONPATH).
# OpenVINO ignores OMP/torch thread caps, so taskset is what actually bounds it. Only ONE of these at a time on merlin.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FOOTY_OV_VENV:-$HOME/.cache/footy_ov_venv}"
[[ -x "$VENV/bin/python" ]] || { echo "ov_run.sh: no venv at $VENV (run scripts/setup_openvino_venv.sh)" >&2; exit 2; }
N="${FOOTY_OV_THREADS:-4}"
export OMP_NUM_THREADS="$N" MKL_NUM_THREADS="$N" OPENBLAS_NUM_THREADS="$N" BALL_POOL_THREADS="$N"
export FOOTY_DEVICE=cpu FOOTY_HALF=0
export PYTHONPATH="${FOOTY_SRC:-$REPO/src}${PYTHONPATH:+:$PYTHONPATH}"
exec taskset -c "${FOOTY_OV_CPUS:-0-3}" timeout "${FOOTY_OV_TIMEOUT:-7200}" "$VENV/bin/python" "$@"
