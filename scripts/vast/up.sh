#!/usr/bin/env bash
# Rent a GPU box, wire ssh, push code, provision. Ephemeral by design:
# NO volume is created or attached — everything lives on container disk and dies
# with the instance. Pull anything you care about with sync.sh down.
#
#   up.sh                                  cheapest A4000-class box, 60 GB disk
#   up.sh --gpu RTX_4090 --disk 120 --max-price 0.40
#   up.sh --reuse                          re-provision the existing instance
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_key

GPU="${VAST_GPU:-RTX_A4000}"
DISK="${VAST_DISK:-60}"
MAX_PRICE="${VAST_MAX_PRICE:-0.20}"
IMAGE="${VAST_IMAGE:-vastai/pytorch:cuda-12.8.1-auto}"
REUSE=0
WITH_DATA="--data"

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)       GPU="$2"; shift 2 ;;
    --disk)      DISK="$2"; shift 2 ;;
    --max-price) MAX_PRICE="$2"; shift 2 ;;
    --image)     IMAGE="$2"; shift 2 ;;
    --reuse)     REUSE=1; shift ;;
    --no-data)   WITH_DATA=""; shift ;;
    *) die "unknown flag: $1" ;;
  esac
done

PUBKEY_FILE="$HOME/.ssh/id_ed25519.pub"
[ -f "$PUBKEY_FILE" ] || die "no ssh public key at $PUBKEY_FILE"

if [ "$REUSE" = 1 ]; then
  ID="$(resolve_instance)"
  info "reusing instance $ID"
else
  info "searching: $GPU, 1 GPU, >=${DISK}GB disk, <= \$$MAX_PRICE/hr"
  # Direct ssh needs a host with open ports; a proxy-only host would leave us
  # unable to connect, because our key is attached per-instance and the Vast
  # proxy only honours account-level keys (which are 2FA-gated on this account).
  ASK="$(vastai search offers \
      "gpu_name=$GPU num_gpus=1 rentable=true verified=true direct_port_count>=2 disk_space>=$DISK dph<=$MAX_PRICE" \
      -o 'dph+' --raw 2>/dev/null \
    | python3 -c '
import json, sys
offers = json.load(sys.stdin)
if not offers:
    sys.exit("no matching offers — relax --gpu / --max-price / --disk")
o = offers[0]
print(o["id"], file=sys.stderr)
print(f"  picked ask {o[\"id\"]}: {o[\"num_gpus\"]}x {o[\"gpu_name\"]} "
      f"${o[\"dph_total\"]:.4f}/hr  {o[\"geolocation\"]}  {o[\"inet_down\"]:.0f}Mbps down",
      file=sys.stderr)
print(o["id"])
')"
  info "creating instance on ask $ASK (disk ${DISK}GB, image $IMAGE)"
  ID="$(vastai create instance "$ASK" --image "$IMAGE" --disk "$DISK" --ssh --direct --raw 2>/dev/null \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); sys.exit("create failed: "+str(r)) if not r.get("success") else print(r["new_contract"])')"
  info "instance $ID created"
fi

echo "$ID" > "$STATE_FILE"

info "attaching ssh key to instance (account-level keys need 2FA; per-instance does not)"
vastai attach ssh "$ID" "$(cat "$PUBKEY_FILE")" >/dev/null

info "waiting for instance to report running and expose port 22"
for _ in $(seq 1 60); do
  if CONN="$(instance_conn "$ID" 2>/dev/null)"; then
    read -r _ STATUS IP PORT <<<"$CONN"
    [ "$STATUS" = "running" ] && break
  fi
  sleep 10
done
[ "${STATUS:-}" = "running" ] || die "instance $ID never reached running state"

write_ssh_alias "$IP" "$PORT" "$ID"
wait_for_ssh 300

"$(dirname "${BASH_SOURCE[0]}")/sync.sh" up $WITH_DATA
info "running provision on the box"
ssh "$SSH_ALIAS" "bash $REMOTE_DIR/scripts/vast/provision.sh"

cat <<EOF

  instance $ID ready — ssh $SSH_ALIAS
  EPHEMERAL: container disk only, no volume. destroy wipes everything.
  pull artifacts:  scripts/vast/sync.sh down
  tear down:       scripts/vast/down.sh
EOF
