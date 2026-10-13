#!/usr/bin/env bash
# Purpose: run the existing environment or tool entry point.
# Inputs: existing environment/profile settings. Outputs: existing logs/environment destinations.
# Dependencies: bash and the selected Python environment. Role: launcher.
# Usage: see the README in this workflow; execution remains explicit.
set -Eeuo pipefail

if ! command -v code >/dev/null 2>&1; then
  echo "VS Code CLI is not installed. Install it during RunPod Stage 2." >&2
  exit 2
fi

exec code tunnel --accept-server-license-terms "$@"
