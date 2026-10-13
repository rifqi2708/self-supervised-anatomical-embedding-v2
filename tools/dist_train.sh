#!/usr/bin/env bash
# Compatibility launcher; canonical script: shared/dist_train.sh
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${REPO_ROOT}/shared/dist_train.sh" "$@"
