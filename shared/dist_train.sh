#!/usr/bin/env bash
# Purpose: launch the existing distributed model training entry point.
# Inputs: config, GPU count and original launcher options; evaluation also takes a checkpoint.
# Outputs: unchanged model work/export destinations. Dependencies: bash, Torch environment.
# Role: launcher. Usage: resources/training.md and master_project/README.md.

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" \
    "${SCRIPT_DIR}/train_sam.py" "$CONFIG" --auto-resume --no-validate --launcher pytorch "${@:3}"
