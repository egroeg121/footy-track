#!/usr/bin/env bash
# Pull artifacts, then destroy. Destroy is irreversible and wipes the container
# filesystem — there is no volume backing any of this.
#
#   down.sh              sync artifacts down, confirm, destroy
#   down.sh --force      skip the confirmation prompt
#   down.sh --no-sync    destroy without pulling (you will lose runs/)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_key

FORCE=0; DO_SYNC=1
while [ $# -gt 0 ]; do
  case "$1" in
    --force)   FORCE=1; shift ;;
    --no-sync) DO_SYNC=0; shift ;;
    *) die "unknown flag: $1" ;;
  esac
done

ID="$(resolve_instance)"
read -r _ STATUS IP PORT <<<"$(instance_conn "$ID")"
info "instance $ID ($STATUS) at $IP:$PORT"

if [ "$DO_SYNC" = 1 ]; then
  "$(dirname "${BASH_SOURCE[0]}")/sync.sh" down
else
  info "--no-sync: skipping artifact pull"
fi

if [ "$FORCE" != 1 ]; then
  printf '\033[33mdestroy instance %s? this is irreversible [y/N] \033[0m' "$ID"
  read -r reply
  [[ "$reply" =~ ^[Yy]$ ]] || die "aborted — instance still running and still billing"
fi

vastai destroy instance "$ID" >/dev/null
rm -f "$STATE_FILE"
info "instance $ID destroyed"

# A dead alias pointing at a recycled ip:port is worse than no alias.
python3 - "$HOME/.ssh/config" "# BEGIN vast-$SSH_ALIAS" "# END vast-$SSH_ALIAS" <<'PY'
import sys
path, begin, end = sys.argv[1:4]
out, skip = [], False
for l in open(path).read().splitlines():
    if l.strip() == begin: skip = True
    if not skip: out.append(l)
    if l.strip() == end: skip = False
open(path, "w").write("\n".join(out).rstrip("\n") + "\n")
PY
info "ssh alias '$SSH_ALIAS' removed"

vastai show volumes --raw 2>/dev/null | python3 -c '
import json, sys
v = json.load(sys.stdin)
if v:
    print(f"\033[33mnote:\033[0m {len(v)} volume(s) still billing: {[x[\"id\"] for x in v]}")
'
