#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/fine_tune/sam_quadra_fine_tune_a6000.py}"
WORK_DIR="${WORK_DIR:-work_dirs/sam_quadra_fine_tune_a6000}"
GPUS="${GPUS:-1}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-8}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-8}"
SEED="${SEED:-42}"
PORT="${PORT:-29500}"

echo "[fine-tune] repo root: ${REPO_ROOT}"
echo "[fine-tune] config: ${CONFIG}"
echo "[fine-tune] work_dir: ${WORK_DIR}"
echo "[fine-tune] gpus: ${GPUS}"
echo "[fine-tune] samples_per_gpu: ${SAMPLES_PER_GPU}"
echo "[fine-tune] workers_per_gpu: ${WORKERS_PER_GPU}"
echo "[fine-tune] seed: ${SEED}"

if [[ "${GPUS}" -gt 1 ]]; then
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  python -m torch.distributed.launch \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    tools/train_sam.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --auto-resume \
    --no-validate \
    --launcher pytorch \
    --seed "${SEED}" \
    --cfg-options \
      data.samples_per_gpu="${SAMPLES_PER_GPU}" \
      data.workers_per_gpu="${WORKERS_PER_GPU}" \
    "$@"
else
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  python tools/train_sam.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --auto-resume \
    --no-validate \
    --seed "${SEED}" \
    --cfg-options \
      data.samples_per_gpu="${SAMPLES_PER_GPU}" \
      data.workers_per_gpu="${WORKERS_PER_GPU}" \
    "$@"
fi
