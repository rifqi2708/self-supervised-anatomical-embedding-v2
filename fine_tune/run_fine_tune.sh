#!/usr/bin/env bash # Use bash from the user's environment.
set -euo pipefail # Exit on error, unset var usage, and pipe failures.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Resolve this script's directory.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)" # Resolve repository root (parent of script dir).
cd "${REPO_ROOT}" # Ensure all relative paths are resolved from repo root.

CONFIG="${CONFIG:-configs/fine_tune/sam_quadra_fine_tune_a6000.py}" # Default config file, overridable by env.
WORK_DIR="${WORK_DIR:-work_dirs/sam_quadra_fine_tune_a6000}" # Default output/checkpoint directory, overridable by env.
GPUS="${GPUS:-1}" # Default number of GPUs to use, overridable by env.
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-8}" # Default per-GPU batch size, overridable by env.
WORKERS_PER_GPU="${WORKERS_PER_GPU:-8}" # Default dataloader workers per GPU, overridable by env.
LR="${LR:-0.002}" # Default fine-tuning learning rate, overridable by env.
MAX_ITERS="${MAX_ITERS:-2000}" # Default max training iterations for fine-tuning, overridable by env.
CKPT_INTERVAL="${CKPT_INTERVAL:-100}" # Default checkpoint save interval in iterations, overridable by env.
LOG_INTERVAL="${LOG_INTERVAL:-10}" # Default logger interval in iterations, overridable by env.
SEED="${SEED:-42}" # Default random seed, overridable by env.
PORT="${PORT:-29500}" # Default distributed master port, overridable by env.

echo "[fine-tune] repo root: ${REPO_ROOT}" # Print resolved repository root for debugging.
echo "[fine-tune] config: ${CONFIG}" # Print config path being used.
echo "[fine-tune] work_dir: ${WORK_DIR}" # Print output directory being used.
echo "[fine-tune] gpus: ${GPUS}" # Print GPU count being used.
echo "[fine-tune] samples_per_gpu: ${SAMPLES_PER_GPU}" # Print effective batch size per GPU.
echo "[fine-tune] workers_per_gpu: ${WORKERS_PER_GPU}" # Print effective dataloader workers per GPU.
echo "[fine-tune] lr: ${LR}" # Print effective learning rate override.
echo "[fine-tune] max iters: ${MAX_ITERS}" # Print effective max training iterations override.
echo "[fine-tune] checkpoint interval: ${CKPT_INTERVAL}" # Print effective checkpoint save interval override.
echo "[fine-tune] log interval: ${LOG_INTERVAL}" # Print effective log interval override.
echo "[fine-tune] seed: ${SEED}" # Print effective random seed.

if [[ "${GPUS}" -gt 1 ]]; then # Use distributed launch when more than one GPU is requested.
  train_cmd=( # Build distributed training command as an array to preserve quoting safely.
    python # Python interpreter.
    -m # Run a module as a script.
    torch.distributed.launch # Torch distributed launcher module.
    --nproc_per_node="${GPUS}" # Number of processes/GPUs on this node.
    --master_port="${PORT}" # Port for process group initialization.
    tools/train_sam.py # Repo training entrypoint.
    "${CONFIG}" # Fine-tune config path.
    --work-dir # Start of work directory argument.
    "${WORK_DIR}" # Work directory value.
    --auto-resume # Resume from latest checkpoint in work dir when available.
    --no-validate # Match original SAM workflow: disable in-training validation hook.
    --launcher # Start of launcher argument.
    pytorch # Use PyTorch distributed backend mode in script.
    --seed # Start of seed argument.
    "${SEED}" # Seed value.
    --cfg-options # Start inline config overrides.
    data.samples_per_gpu="${SAMPLES_PER_GPU}" # Override samples_per_gpu at runtime.
    data.workers_per_gpu="${WORKERS_PER_GPU}" # Override workers_per_gpu at runtime.
    optimizer.lr="${LR}" # Override learning rate at runtime for fine-tuning.
    runner.max_iters="${MAX_ITERS}" # Override max training iterations at runtime.
    checkpoint_config.interval="${CKPT_INTERVAL}" # Override checkpoint interval at runtime.
    log_config.interval="${LOG_INTERVAL}" # Override logger interval at runtime.
  )
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${train_cmd[@]}" "$@" # Run command with repo on PYTHONPATH and forward extra CLI args.
else # Use single-GPU/non-distributed mode when one GPU is requested.
  train_cmd=( # Build single-GPU training command as an array.
    python # Python interpreter.
    tools/train_sam.py # Repo training entrypoint.
    "${CONFIG}" # Fine-tune config path.
    --work-dir # Start of work directory argument.
    "${WORK_DIR}" # Work directory value.
    --auto-resume # Resume from latest checkpoint in work dir when available.
    --no-validate # Match original SAM workflow: disable in-training validation hook.
    --seed # Start of seed argument.
    "${SEED}" # Seed value.
    --cfg-options # Start inline config overrides.
    data.samples_per_gpu="${SAMPLES_PER_GPU}" # Override samples_per_gpu at runtime.
    data.workers_per_gpu="${WORKERS_PER_GPU}" # Override workers_per_gpu at runtime.
    optimizer.lr="${LR}" # Override learning rate at runtime for fine-tuning.
    runner.max_iters="${MAX_ITERS}" # Override max training iterations at runtime.
    checkpoint_config.interval="${CKPT_INTERVAL}" # Override checkpoint interval at runtime.
    log_config.interval="${LOG_INTERVAL}" # Override logger interval at runtime.
  )
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${train_cmd[@]}" "$@" # Run single-GPU command and forward extra CLI args.
fi # End GPU mode branch.
