#!/usr/bin/env bash
# Run python in the merlin ROCm venv (Radeon 780M iGPU, gfx1103) with the env it needs, under a timeout.
#   scripts/rocm_run.sh -m footy_track.scripts.ball_pool --root ... candidates --match M --weights W
# Env (all overridable): FOOTY_ROCM_VENV (~/.cache/footy_rocm_venv), HSA_OVERRIDE_GFX_VERSION (11.0.2),
#   FOOTY_SRC (<repo>/src, the PYTHONPATH), FOOTY_ROCM_TIMEOUT (seconds, default 21600 = 6 h).
# fp16 only: FOOTY_HALF is forced to 1 and FOOTY_HALF=0 is an error (fp32 crashed conv2d and reset the GPU).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FOOTY_ROCM_VENV:-$HOME/.cache/footy_rocm_venv}"
if [[ "${FOOTY_HALF:-1}" != "1" ]]; then
  echo "rocm_run.sh: refusing FOOTY_HALF=${FOOTY_HALF} (fp32 on the iGPU resets the GPU); fp16 only" >&2
  exit 2
fi
[[ -x "$VENV/bin/python" ]] || { echo "rocm_run.sh: no venv at $VENV (run scripts/setup_rocm_venv.sh)" >&2; exit 2; }
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.2}"
export FOOTY_HALF=1
export PYTHONPATH="${FOOTY_SRC:-$REPO/src}${PYTHONPATH:+:$PYTHONPATH}"
exec timeout "${FOOTY_ROCM_TIMEOUT:-21600}" "$VENV/bin/python" "$@"
