#!/usr/bin/env bash
# Purpose: run the existing environment or tool entry point.
# Inputs: existing environment/profile settings. Outputs: existing logs/environment destinations.
# Dependencies: bash and the selected Python environment. Role: launcher.
# Usage: see the README in this workflow; execution remains explicit.
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/review-venv" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$1"

if [[ "${VENV_PATH}" != /* ]]; then
  echo "The review environment path must be absolute." >&2
  exit 2
fi

python3 -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements-review.txt"
"${VENV_PATH}/bin/python" -c \
  'import matplotlib, nibabel, numpy, pandas, streamlit, yaml; print("Mask review environment PASS")'
