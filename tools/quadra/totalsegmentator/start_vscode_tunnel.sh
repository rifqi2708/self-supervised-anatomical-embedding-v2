#!/usr/bin/env bash
# Compatibility launcher; canonical script: publication/masks/segmentation/start_vscode_tunnel.sh
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${REPO_ROOT}/publication/masks/segmentation/start_vscode_tunnel.sh" "$@"
