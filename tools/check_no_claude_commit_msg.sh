#!/usr/bin/env bash
# Reject commit messages that mention Claude (e.g. co-author trailers).
set -euo pipefail

msg_file="$1"

if grep -qi "claude" "$msg_file"; then
    echo "error: commit message must not mention 'Claude'" >&2
    exit 1
fi
