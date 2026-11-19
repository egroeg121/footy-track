#!/usr/bin/env bash
# Deploy / operate the labelling server. It runs ONLY from $SERVE_DIR on main.
# ONE service (calib, :8011) serves every dataset (calib, nottinghamforest, ...): the dataset is the
# ?dataset= query parameter and the datasets live in the registry
# $DATA/labeller_datasets.json (see footy_track/labeller/datasets.py), created on first
# restart/deploy by `labeller_datasets init` if missing, never overwritten.
#
#   scripts/labeller_deploy.sh status
#   scripts/labeller_deploy.sh deploy  [svc...]   # ff-only to origin/main, restart, wait for sha
#   scripts/labeller_deploy.sh restart <svc> [--force]
#   scripts/labeller_deploy.sh logs    <svc> [tail args, e.g. -f]
#
# Global flag --dry-run prints every mutating command instead of running it
# (read-only checks still run). --force lets restart kill a non-tmux process
# that still holds the port. --no-warm skips the Ball Check cache warm-up that
# otherwise runs in the background after restarting a service in WARM_SVCS.
#
# Env overrides: SERVE_DIR, LABELLER_PY, TMUX_SOCKET, LABELLER_LOG_DIR, DEPLOY_TIMEOUT.
# Retire the old nottinghamforest server (:8012) yourself after the first deploy: `tmux -L footy kill-session -t footy-norwich`.
set -euo pipefail

SERVE_DIR="${SERVE_DIR:-/home/george/footy-track-main}"
PY="${LABELLER_PY:-/home/george/footy-track/.venv/bin/python}"
TMUX_SOCKET="${TMUX_SOCKET:-footy}"
LOG_DIR="${LABELLER_LOG_DIR:-$HOME/.cache/footy-labeller}"
HEALTH_TIMEOUT="${DEPLOY_TIMEOUT:-60}"
HOST="${LABELLER_HEALTH_HOST:-127.0.0.1}"
DATA=/mnt/storage/footy_data

# ---- service table: name -> port, extra env (one KEY=VALUE per line) --------
SERVICES=(calib)
declare -A SVC_PORT=([calib]=8011)
declare -A SVC_ENV
# No FOOTY_*_DIR here: per-dataset dirs come from the registry. (Setting FOOTY_CLIPS_DIR /
# FOOTY_GT_MARKS_DIR / ... would override the DEFAULT dataset's fields and hide registry edits.)
# Broadcast-view classifier the labeller pre-classifies with (classifier.default_classifier_path).
# Production = whatever the jobs repo last promoted: its promotion step mirrors the Prefect Variable
# `broadcast-classifier-production` into $PRODUCTION_MIRROR ({"run", "sha256", "path", ...}); production_weights()
# reads it at every (re)start. FALLBACK_WEIGHTS is used only while the mirror does not exist / points at a missing file.
PRODUCTION_MIRROR="${PRODUCTION_MIRROR:-/mnt/storage/models/production_classifier.json}"
FALLBACK_WEIGHTS=/mnt/storage/models/broadcast_cls_20261118_v2dcef14/last.pt
# -----------------------------------------------------------------------------

# Path of the production broadcast classifier's last.pt: the promotion mirror's file if it exists, else the fallback.
production_weights() {
  local p=""
  if [[ -r "$PRODUCTION_MIRROR" ]]; then
    p="$(python3 -c "import sys,json; print(json.load(open(sys.argv[1])).get('path') or '')" "$PRODUCTION_MIRROR" 2>/dev/null || true)"
  fi
  if [[ -n "$p" && -f "$p" ]]; then
    echo "$p"
  else
    [[ -z "$p" ]] || echo "warning: $PRODUCTION_MIRROR points at missing $p; using $FALLBACK_WEIGHTS" >&2
    echo "$FALLBACK_WEIGHTS"
  fi
}

# service -> datasets whose Ball Check crop/clip caches are warmed after a restart (they
# serve a ball_check_list; a restart empties the in-memory caches).
declare -A WARM_DATASETS=([calib]="nottinghamforest")
# service -> datasets whose detector-candidate pool is built after a restart (the first
# `source=candidates` request otherwise pays ~13 s; later ones 0.14 s).
declare -A WARM_POOL_DATASETS=([calib]="nottinghamforest")

DRY=0
FORCE=0
WARM=1

die() { echo "error: $*" >&2; exit 1; }

# Run a mutating command, or just print it under --dry-run.
run() {
  if ((DRY)); then
    if [[ "$1" == tm ]]; then shift; set -- tmux -L "$TMUX_SOCKET" "$@"; fi
    printf '+ '; printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

tm() { tmux -L "$TMUX_SOCKET" "$@"; }
session_of() { echo "footy-$1"; }
port_of() { echo "${SVC_PORT[$1]:-}"; }

check_svc() {
  [[ -n "$(port_of "$1")" ]] || die "unknown service '$1' (known: ${SERVICES[*]})"
}

have_session() { tm has-session -t "$(session_of "$1")" 2>/dev/null; }

# PID(s) listening on a TCP port (empty if none).
listener_pids() {
  ss -H -ltnp "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true
}

# JSON field of a service's /api/server_info; empty on any failure.
served_field() {
  local port="$1" field="$2"
  curl -fsS --max-time 3 "http://$HOST:$port/api/server_info" 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get(sys.argv[1]) or '')" "$field" 2>/dev/null || true
}

origin_main_sha() {
  local s
  s="$(git -C "$SERVE_DIR" ls-remote origin main 2>/dev/null | cut -f1 || true)"
  [[ -n "$s" ]] || s="$(git -C "$SERVE_DIR" rev-parse origin/main 2>/dev/null || true)"
  echo "$s"
}

cmd_status() {
  local origin head branch
  origin="$(origin_main_sha)"
  head="$(git -C "$SERVE_DIR" rev-parse HEAD 2>/dev/null || echo '?')"
  branch="$(git -C "$SERVE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "checkout $SERVE_DIR: ${head:0:9} ($branch)  origin/main: ${origin:0:9}"
  if [[ -n "$(git -C "$SERVE_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo "  WARNING: checkout has uncommitted changes (deploy will refuse)"
  fi
  local svc port sess pid served verdict
  for svc in "${SERVICES[@]}"; do
    port="$(port_of "$svc")"
    sess="-"; have_session "$svc" && sess="$(session_of "$svc")"
    pid="$(listener_pids "$port" | tr '\n' ',' | sed 's/,$//')"
    served="$(served_field "$port" sha)"
    if [[ -z "$served" ]]; then
      verdict="DOWN (no /api/server_info; pre-dates it or not running)"
    elif [[ -n "$origin" && "$served" == "$origin" ]]; then
      verdict="up to date"
    else
      verdict="STALE (serving ${served:0:9}, origin/main ${origin:0:9})"
    fi
    printf '%-8s :%s tmux=%-14s pid=%-8s %s\n' "$svc" "$port" "$sess" "${pid:--}" "$verdict"
  done
}

build_start_cmd() {
  local svc="$1" port line
  port="$(port_of "$svc")"
  local -a envs=("PYTHONPATH=src" "OMP_NUM_THREADS=2" "FOOTY_SERVICE_NAME=$svc" "FOOTY_SERVICE_PORT=$port"
                 "FOOTY_CLASSIFIER_WEIGHTS=$(production_weights)")
  while IFS= read -r line; do [[ -n "$line" ]] && envs+=("$line"); done <<<"${SVC_ENV[$svc]:-}"
  local log="$LOG_DIR/$svc.log"
  printf 'cd %q && exec env' "$SERVE_DIR"
  printf ' %q' "${envs[@]}"
  printf ' nice -n 5 %q -m uvicorn footy_track.labeller.server:app --host 0.0.0.0 --port %q >> %q 2>&1' "$PY" "$port" "$log"
}

# True if PID is $2 or one of its descendants.
pid_under() {
  local pid="$1" root="$2" n=0
  while [[ -n "$pid" && "$pid" -gt 1 && $n -lt 64 ]]; do
    [[ "$pid" == "$root" ]] && return 0
    pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
    n=$((n + 1))
  done
  return 1
}

# Print the PIDs holding svc's port that are NOT inside svc's own tmux session.
foreign_listeners() {
  local svc="$1" pid root ok
  local -a roots=()
  if have_session "$svc"; then mapfile -t roots < <(tm list-panes -t "$(session_of "$svc")" -F '#{pane_pid}')
  fi
  for pid in $(listener_pids "$(port_of "$svc")"); do
    ok=0
    for root in "${roots[@]}"; do pid_under "$pid" "$root" && ok=1 && break; done
    ((ok)) || echo "$pid"
  done
}

# Abort BEFORE anything is restarted if any service's port is held by a
# non-tmux process (unless --force), so one service is never restarted while
# the next one then fails.
preflight_ports() {
  local svc pids bad=0
  for svc in "$@"; do
    pids="$(foreign_listeners "$svc" | tr '\n' ' ')"
    if [[ -n "${pids// /}" ]]; then
      if ((FORCE)); then
        echo "preflight: $svc port $(port_of "$svc") held by non-tmux PID(s) $pids; --force will kill"
      else
        echo "preflight: $svc port $(port_of "$svc") is held by a process outside tmux session $(session_of "$svc") (PID(s) $pids)" >&2
        bad=1
      fi
    fi
  done
  ((bad == 0)) || die "nothing was restarted. Stop those process(es) yourself or rerun with --force."
}

cmd_restart() {
  local svc="$1" port sess pids
  check_svc "$svc"
  port="$(port_of "$svc")"; sess="$(session_of "$svc")"
  run mkdir -p "$LOG_DIR"
  ensure_registry
  if have_session "$svc"; then
    run tm kill-session -t "$sess"
  fi
  if ((DRY)); then
    echo "# would wait up to 10s for :$port to free; a non-tmux holder aborts unless --force"
  else
    local i
    for i in $(seq 1 20); do
      [[ -z "$(listener_pids "$port")" ]] && break
      sleep 0.5
    done
    pids="$(listener_pids "$port" | tr '\n' ' ')"
    if [[ -n "${pids// /}" ]]; then
      if ((FORCE)); then
        echo "port $port still held by PID(s) $pids; --force: killing"
        # shellcheck disable=SC2086
        kill $pids || true
        sleep 2
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        sleep 1
      else
        die "port $port is held by a non-tmux process (PID(s) $pids). Stop it yourself or rerun with --force."
      fi
    fi
  fi
  run tm new-session -d -s "$sess" -c "$SERVE_DIR" "$(build_start_cmd "$svc")"
  start_warm "$svc"
}

# Create the dataset registry from the built-in seed if it does not exist; never overwrites.
# Aborts (before anything is restarted) if that fails, so the server cannot come up pointing
# at the wrong data.
ensure_registry() {
  run bash -c 'cd "$1" && PYTHONPATH=src exec "$2" -m footy_track.scripts.labeller_datasets init --data-root "$3"' \
    _ "$SERVE_DIR" "$PY" "$DATA" || die "could not ensure the dataset registry; nothing was restarted"
}

# Background, sequential, nice 19: GET every not-yet-verdicted Ball Check card's
# crop + clip so the first phone to open it does not pay for the decode.
start_warm() {
  local svc="$1" ds port log
  ((WARM)) || return 0
  port="$(port_of "$svc")"
  for ds in ${WARM_DATASETS[$svc]:-}; do
    log="$LOG_DIR/$svc-warm-$ds.log"
    if ((DRY)); then
      echo "# would warm $svc's Ball Check cards for dataset $ds in the background (disable with --no-warm):"
      printf '+ (cd %q && PYTHONPATH=src nohup nice -n 19 %q -m footy_track.scripts.warm_ball_check --base http://%s:%s --dataset %s >> %q 2>&1 &)\n' \
        "$SERVE_DIR" "$PY" "$HOST" "$port" "$ds" "$log"
      continue
    fi
    pkill -f "footy_track.scripts.warm_ball_check --base http://$HOST:$port --dataset $ds" 2>/dev/null || true
    (cd "$SERVE_DIR" && PYTHONPATH=src nohup nice -n 19 "$PY" -m footy_track.scripts.warm_ball_check \
      --base "http://$HOST:$port" --dataset "$ds" >>"$log" 2>&1 &)
    echo "$svc: warming Ball Check caches of dataset $ds in the background (log: $log)"
  done
  for ds in ${WARM_POOL_DATASETS[$svc]:-}; do
    log="$LOG_DIR/$svc-warm-pool-$ds.log"
    if ((DRY)); then
      echo "# would build $svc's candidate pool for dataset $ds in the background (disable with --no-warm):"
      printf '+ (cd %q && PYTHONPATH=src nohup nice -n 19 %q -m footy_track.scripts.warm_ball_check --source candidates --base http://%s:%s --dataset %s >> %q 2>&1 &)\n' \
        "$SERVE_DIR" "$PY" "$HOST" "$port" "$ds" "$log"
      continue
    fi
    (cd "$SERVE_DIR" && PYTHONPATH=src nohup nice -n 19 "$PY" -m footy_track.scripts.warm_ball_check \
      --source candidates --base "http://$HOST:$port" --dataset "$ds" >>"$log" 2>&1 &)
    echo "$svc: building the candidate pool of dataset $ds in the background (log: $log)"
  done
}

wait_healthy() {
  local svc="$1" want="$2" port got i
  port="$(port_of "$svc")"
  if ((DRY)); then
    echo "# would poll http://$HOST:$port/api/server_info up to ${HEALTH_TIMEOUT}s until sha == ${want:0:9}"
    return 0
  fi
  for i in $(seq 1 "$HEALTH_TIMEOUT"); do
    got="$(served_field "$port" sha)"
    if [[ "$got" == "$want" ]]; then echo "$svc: healthy, serving ${got:0:9}"; return 0; fi
    sleep 1
  done
  echo "$svc: NOT healthy after ${HEALTH_TIMEOUT}s (served '${got:0:9}', wanted ${want:0:9}); see: $0 logs $svc" >&2
  return 1
}

cmd_deploy() {
  local -a svcs=("$@")
  ((${#svcs[@]})) || svcs=("${SERVICES[@]}")
  local s
  for s in "${svcs[@]}"; do check_svc "$s"; done
  if [[ -n "$(git -C "$SERVE_DIR" status --porcelain --untracked-files=no)" ]]; then
    git -C "$SERVE_DIR" status --short --untracked-files=no >&2
    die "$SERVE_DIR has uncommitted changes; refusing to deploy"
  fi
  preflight_ports "${svcs[@]}"
  run git -C "$SERVE_DIR" fetch origin
  local target
  target="$(git -C "$SERVE_DIR" rev-parse origin/main)"  # what HEAD will be after the ff (re-read below when live)
  if git -C "$SERVE_DIR" symbolic-ref -q HEAD >/dev/null; then
    local br
    br="$(git -C "$SERVE_DIR" rev-parse --abbrev-ref HEAD)"
    [[ "$br" == main ]] || die "$SERVE_DIR is on branch '$br', not main; live servers run only main"
    run git -C "$SERVE_DIR" merge --ff-only origin/main
  else
    run git -C "$SERVE_DIR" checkout --detach origin/main
  fi
  ((DRY)) || target="$(git -C "$SERVE_DIR" rev-parse HEAD)"
  if ((!DRY)); then
    # bash has already parsed THIS copy of the script (service table included) from before the
    # pull; re-exec the freshly checked-out copy so a changed service table takes effect now.
    local -a fwd=()
    ((FORCE)) && fwd+=(--force)
    ((WARM)) || fwd+=(--no-warm)
    exec bash "$SERVE_DIR/scripts/labeller_deploy.sh" "${fwd[@]+"${fwd[@]}"}" _restart_all "$target" "${svcs[@]}"
  fi
  restart_all "$target" "${svcs[@]}"
}

restart_all() {   # restart_all <sha> <svc...>: restart each, then wait until each serves <sha>
  local target="$1"; shift
  local s rc=0
  for s in "$@"; do check_svc "$s"; done
  for s in "$@"; do cmd_restart "$s"; done
  for s in "$@"; do wait_healthy "$s" "$target" || rc=1; done
  return $rc
}

cmd_logs() {
  local svc="$1"; shift
  check_svc "$svc"
  local log="$LOG_DIR/$svc.log"
  [[ -f "$log" ]] || die "no log yet at $log"
  if (($#)); then tail "$@" "$log"; else tail -n 100 "$log"; fi
}

main() {
  local -a args=()
  local a
  for a in "$@"; do
    case "$a" in
      --dry-run) DRY=1 ;;
      --force) FORCE=1 ;;
      --no-warm) WARM=0 ;;
      *) args+=("$a") ;;
    esac
  done
  set -- "${args[@]+"${args[@]}"}"
  local cmd="${1:-}"; shift || true
  case "$cmd" in
    status) cmd_status ;;
    deploy) cmd_deploy "$@" ;;
    restart)
      [[ $# -ge 1 ]] || die "usage: restart <svc> [--force]"
      check_svc "$1"; preflight_ports "$1"; cmd_restart "$1"
      wait_healthy "$1" "$(git -C "$SERVE_DIR" rev-parse HEAD)" ;;
    _restart_all) restart_all "$@" ;;   # internal: deploy re-execs the updated script into this
    logs) [[ $# -ge 1 ]] || die "usage: logs <svc> [tail args]"; cmd_logs "$@" ;;
    *) sed -n '2,13p' "$0"; exit 2 ;;
  esac
}

main "$@"
