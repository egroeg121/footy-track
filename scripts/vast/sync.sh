#!/usr/bin/env bash
# Push code (and optionally the dataset) up to the box; pull artifacts back down.
#
#   sync.sh up          code only (fast — do this after every edit)
#   sync.sh up --data   code + data/training_datasets (~136 MB, first run only)
#   sync.sh down        pull runs/ and model_saves/ back to the rig
#
# The dataset is rsync'd rather than re-pulled from Roboflow on purpose: a fresh
# Roboflow download can hand you a different version, and the ball benchmarks are
# only comparable against the exact v11 val split (see wiki/footy-ball-detection-findings).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CMD="${1:-}"; shift || true
WITH_DATA=0
for a in "$@"; do [ "$a" = "--data" ] && WITH_DATA=1; done

RSYNC=(rsync -az --info=stats1 -e "ssh -o BatchMode=yes")

case "$CMD" in
  up)
    info "pushing code -> $SSH_ALIAS:$REMOTE_DIR"
    ssh "$SSH_ALIAS" "mkdir -p $REMOTE_DIR"
    "${RSYNC[@]}" --delete \
      --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
      --exclude='*.pyc' --exclude='*.png' --exclude='notebooks/' \
      --exclude='metaflow/' --exclude='docs/' \
      --exclude='data/' --exclude='model_saves/' --exclude='runs/' \
      "$RIG_DIR/" "$SSH_ALIAS:$REMOTE_DIR/"

    if [ "$WITH_DATA" = 1 ]; then
      info "pushing data/training_datasets (~$(du -sh "$RIG_DIR/data/training_datasets" | cut -f1))"
      ssh "$SSH_ALIAS" "mkdir -p $REMOTE_DIR/data"
      "${RSYNC[@]}" "$RIG_DIR/data/training_datasets" "$SSH_ALIAS:$REMOTE_DIR/data/"
      # rsync restores the mini-absolute path every push, so re-fix after each one
      ssh "$SSH_ALIAS" "bash $REMOTE_DIR/scripts/vast/fix_dataset_paths.sh $REMOTE_DIR"
    fi

    # Secrets go in a mode-600 env file, never into the rsync'd tree — .envrc
    # lives above the rig and would otherwise ride along in a code push.
    if [ -f "$RIG_DIR/../../.envrc" ]; then
      info "installing ROBOFLOW_API_KEY into /root/.footy_env"
      grep -h '^export ROBOFLOW_API_KEY=' "$RIG_DIR/../../.envrc" | sed 's/^export //' \
        | ssh "$SSH_ALIAS" "cat > /root/.footy_env && chmod 600 /root/.footy_env"
    fi
    ;;

  down)
    # The whole point of ephemeral: if it isn't pulled, destroy eats it.
    info "pulling runs/ + model_saves/ <- $SSH_ALIAS"
    mkdir -p "$RIG_DIR/runs" "$RIG_DIR/model_saves"
    for d in runs model_saves; do
      if ssh "$SSH_ALIAS" "[ -d $REMOTE_DIR/$d ]"; then
        "${RSYNC[@]}" "$SSH_ALIAS:$REMOTE_DIR/$d/" "$RIG_DIR/$d/"
      else
        info "no $d/ on the box — skipping"
      fi
    done
    ;;

  *) die "usage: sync.sh {up [--data]|down}" ;;
esac
info "sync $CMD complete"
