#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/mnt/pfs/zhangjiyao/yiming/kai0"
readonly STORAGE_ROOT="${REPO_ROOT}"

# Keep all downloaded data, caches, temporary files, W&B runs, and checkpoints
# on the persistent /mnt/pfs/zhangjiyao/yiming volume. Do not use /tmp.
export HF_HOME="${STORAGE_ROOT}/hf_home"
export HF_DATASETS_CACHE="${STORAGE_ROOT}/hf_datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="${STORAGE_ROOT}/cache"
export TORCH_HOME="${STORAGE_ROOT}/torch_home"
export TMPDIR="${STORAGE_ROOT}/runtime_tmp"
export WANDB_DIR="${STORAGE_ROOT}/wandb/runs"
export WANDB_CACHE_DIR="${STORAGE_ROOT}/wandb/cache"
export WANDB_MODE="online"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export JAX_PLATFORMS="cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"

readonly EXP_NAME="${EXP_NAME:-towel_fold_advantage_v1}"
readonly RESUME="${RESUME:-0}"

mkdir -p \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" \
    "${HF_HUB_CACHE}" \
    "${XDG_CACHE_HOME}" \
    "${TORCH_HOME}" \
    "${TMPDIR}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}"

cd "${REPO_ROOT}"

resume_args=()
if [[ "${RESUME}" == "1" ]]; then
    resume_args+=(--resume)
fi

exec .venv/bin/torchrun \
    --standalone \
    --nproc_per_node=2 \
    scripts/train_pytorch.py \
    ADVANTAGE_TORCH_KAI0_TOWEL_FOLD \
    --exp-name="${EXP_NAME}" \
    "${resume_args[@]}" \
    "$@"
