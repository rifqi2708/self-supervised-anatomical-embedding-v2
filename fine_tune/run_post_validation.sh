#!/usr/bin/env bash
# Compatibility launcher; canonical script: master_project/fine_tune/run_post_validation.sh
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${REPO_ROOT}/master_project/fine_tune/run_post_validation.sh" "$@"
