#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/fine_tune/sam_quadra_fine_tune_a6000.py}"
WORK_DIR="${WORK_DIR:-work_dirs/sam_quadra_fine_tune_a6000}"
GPUS="${GPUS:-1}"

# Priority: arg1 > CHECKPOINT env > latest checkpoint under WORK_DIR
CHECKPOINT="${1:-${CHECKPOINT:-${WORK_DIR}/latest.pth}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[post-val] checkpoint not found: ${CHECKPOINT}" >&2
  echo "[post-val] pass a checkpoint path as arg1 or set CHECKPOINT env." >&2
  exit 1
fi

echo "[post-val] repo root: ${REPO_ROOT}"
echo "[post-val] config: ${CONFIG}"
echo "[post-val] checkpoint: ${CHECKPOINT}"
echo "[post-val] gpus: ${GPUS}"
echo "[post-val] output path (from config): work_dirs/sam_quadra_fine_tune_a6000/post_val_embeddings/"

if [[ "${GPUS}" -gt 1 ]]; then
  bash tools/dist_test.sh "${CONFIG}" "${CHECKPOINT}" "${GPUS}"
else
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  python tools/test_sam.py "${CONFIG}" "${CHECKPOINT}"
fi
