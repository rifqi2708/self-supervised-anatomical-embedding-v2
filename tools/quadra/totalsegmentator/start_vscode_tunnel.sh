#!/usr/bin/env bash
set -Eeuo pipefail

if ! command -v code >/dev/null 2>&1; then
  echo "VS Code CLI is not installed. Install it during RunPod Stage 2." >&2
  exit 2
fi

exec code tunnel --accept-server-license-terms "$@"
