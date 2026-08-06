#!/usr/bin/env bash
# Run commands on the RunPod pod over a PTY session and strip terminal control noise.
set -uo pipefail
POD="${POD:-v2nto8bubwbodg-64411ff0@ssh.runpod.io}"
{ cat; printf '\nexit\n'; } | timeout "${POD_TIMEOUT:-180}" ssh -tt \
  -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 "$POD" 2>&1 \
  | sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\x1b\][0-9];[^\x07]*\x07//g' -e 's/\r//g'
