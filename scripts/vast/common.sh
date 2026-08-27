#!/usr/bin/env bash
# Shared helpers for the ephemeral Vast.ai workflow.
# Nothing on a Vast box survives destroy — see wiki/vast-ai-gpu-rental.md.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

RIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_ALIAS="${VAST_SSH_ALIAS:-vast}"
REMOTE_DIR="/root/footy_track"
STATE_FILE="$HOME/.config/vastai/current-instance"

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }

need_key() {
  [ -f "$HOME/.config/vastai/vast_api_key" ] || die "no Vast API key; run: vastai set api-key <KEY>"
}

# Echo the id of the instance to act on: $1, else $VAST_INSTANCE, else the saved
# one, else the only running instance. Fails loudly if ambiguous.
resolve_instance() {
  local want="${1:-${VAST_INSTANCE:-}}"
  [ -z "$want" ] && [ -f "$STATE_FILE" ] && want="$(cat "$STATE_FILE")"
  vastai show instances --raw 2>/dev/null | python3 -c '
import json, sys
want = sys.argv[1] or None
rows = json.load(sys.stdin)
if want:
    rows = [i for i in rows if str(i["id"]) == want]
    if not rows:
        sys.exit(f"instance {want} not found on this account")
elif len(rows) != 1:
    sys.exit(f"expected exactly 1 instance, found {len(rows)}: "
             f"{[i['id'] for i in rows]} — pass an id or set VAST_INSTANCE")
print(rows[0]["id"])
' "$want"
}

# Print "<id> <status> <ip> <ssh_port>" for an instance.
instance_conn() {
  vastai show instances --raw 2>/dev/null | python3 -c '
import json, sys
want = sys.argv[1]
for i in json.load(sys.stdin):
    if str(i["id"]) != want:
        continue
    # Port 22 is mapped to a random external port; both IP and port change on
    # every rebuild, so never cache these.
    m = (i.get("ports") or {}).get("22/tcp") or []
    port = next((p["HostPort"] for p in m if p.get("HostIp") == "0.0.0.0"), None)
    if not port:
        sys.exit(f"instance {want} exposes no direct port 22 — is it still loading?")
    print(i["id"], i["actual_status"], i["public_ipaddr"], port)
    break
else:
    sys.exit(f"instance {want} not found")
' "$1"
}

# Rewrite the managed ssh_config block so `ssh $SSH_ALIAS` always points at the
# current instance. Idempotent.
write_ssh_alias() {
  local ip="$1" port="$2" id="$3"
  local cfg="$HOME/.ssh/config" begin="# BEGIN vast-$SSH_ALIAS" end="# END vast-$SSH_ALIAS"
  touch "$cfg"; chmod 600 "$cfg"
  python3 - "$cfg" "$begin" "$end" <<PY
import sys
path, begin, end = sys.argv[1:4]
block = f'''{begin}
# Managed by scripts/vast/. Regenerated on every up.sh — do not hand-edit.
# instance $id
Host $SSH_ALIAS
    HostName $ip
    Port $port
    User root
    StrictHostKeyChecking accept-new
    UserKnownHostsFile ~/.ssh/known_hosts_vast
    ConnectTimeout 15
    ServerAliveInterval 30
{end}'''
lines, out, skip = open(path).read().splitlines(), [], False
for l in lines:
    if l.strip() == begin: skip = True
    if not skip: out.append(l)
    if l.strip() == end: skip = False
while out and not out[-1].strip(): out.pop()
open(path, "w").write("\n".join(out + ["", block, ""]).lstrip("\n"))
PY
  # Host key changes with every rebuild at a recycled ip:port — a stale entry
  # would look like a MITM warning and block the connection.
  ssh-keygen -R "[$ip]:$port" -f "$HOME/.ssh/known_hosts_vast" >/dev/null 2>&1 || true
  info "ssh alias '$SSH_ALIAS' -> root@$ip:$port (instance $id)"
}

wait_for_ssh() {
  local deadline=$(( SECONDS + ${1:-300} ))
  while (( SECONDS < deadline )); do
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_ALIAS" true 2>/dev/null; then
      info "ssh is up"; return 0
    fi
    sleep 10
  done
  die "ssh did not come up within ${1:-300}s"
}
