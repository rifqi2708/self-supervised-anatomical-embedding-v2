#!/usr/bin/env bash
# Compatibility launcher; canonical script: publication/masks/postprocessing/setup_review_env.sh
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${REPO_ROOT}/publication/masks/postprocessing/setup_review_env.sh" "$@"
