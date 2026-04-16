#!/usr/bin/env bash # Use bash from the user's environment.
set -euo pipefail # Exit on error, unset var usage, and pipe failures.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Resolve this script's directory.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)" # Resolve repository root (parent of script dir).
cd "${REPO_ROOT}" # Ensure all relative paths are resolved from repo root.

CONFIG="${CONFIG:-configs/fine_tune/sam_quadra_fine_tune_a6000.py}" # Default config file, overridable by env.
WORK_DIR="${WORK_DIR:-work_dirs/sam_quadra_fine_tune_a6000}" # Default training work directory, overridable by env.
GPUS="${GPUS:-1}" # Default number of GPUs for post-validation, overridable by env.

# Choose checkpoint in this priority: script arg1 > CHECKPOINT env > latest checkpoint path in WORK_DIR.
CHECKPOINT="${1:-${CHECKPOINT:-${WORK_DIR}/latest.pth}}" # Compute effective checkpoint path.

if [[ ! -f "${CHECKPOINT}" ]]; then # Validate that checkpoint file exists before launching evaluation.
  echo "[post-val] checkpoint not found: ${CHECKPOINT}" >&2 # Print missing-checkpoint error to stderr.
  echo "[post-val] pass a checkpoint path as arg1 or set CHECKPOINT env." >&2 # Print usage hint to stderr.
  exit 1 # Stop script early when checkpoint is missing.
fi # End checkpoint existence check.

echo "[post-val] repo root: ${REPO_ROOT}" # Print resolved repository root for debugging.
echo "[post-val] config: ${CONFIG}" # Print config path being used.
echo "[post-val] checkpoint: ${CHECKPOINT}" # Print checkpoint path being used.
echo "[post-val] gpus: ${GPUS}" # Print GPU count being used.
echo "[post-val] output path (from config): work_dirs/sam_quadra_fine_tune_a6000/post_val_embeddings/" # Print expected embedding output folder.

if [[ "${GPUS}" -gt 1 ]]; then # Use distributed test script when more than one GPU is requested.
  bash tools/dist_test.sh "${CONFIG}" "${CHECKPOINT}" "${GPUS}" # Launch distributed post-validation export.
else # Use single-GPU test entrypoint when one GPU is requested.
  eval_cmd=( # Build single-GPU evaluation command as an array.
    python # Python interpreter.
    tools/test_sam.py # Repo test/evaluation entrypoint.
    "${CONFIG}" # Config path.
    "${CHECKPOINT}" # Checkpoint path.
  )
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${eval_cmd[@]}" # Run command with repo on PYTHONPATH.
fi # End GPU mode branch.
