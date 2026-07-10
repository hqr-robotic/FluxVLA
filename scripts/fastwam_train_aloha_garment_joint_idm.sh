#!/usr/bin/env bash
# Train the real ALOHA garment FastWAM joint and IDM variants sequentially.
#
# Default output layout is next to the existing uncond garment run:
#   /root/projects/ryanhu/FluxVLA/FastWAM/fastwam_joint_aloha_garment_full_finetune_ddp_stride4_bs12_20260704_20260706/
#   /root/projects/ryanhu/FluxVLA/FastWAM/fastwam_idm_aloha_garment_full_finetune_ddp_stride4_bs12_20260704_20260706/
#
# Useful overrides:
#   NPROC_PER_NODE=8
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   RUN_NAME_SUFFIX=_my_experiment
#   DRY_RUN=1
#   RESUME=1
#   ALLOW_EXISTING_WORK_DIR=1
#   PER_DEVICE_BATCH_SIZE=8
#   TRAIN_CFG_OPTIONS="runner.max_steps=1000"
#   TRAIN_EXTRA_ARGS="--resume-from /path/to/checkpoint.safetensors"

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/projects/FluxVLA}"
BASE_RUN_DIR="${BASE_RUN_DIR:-/root/projects/ryanhu/FluxVLA/FastWAM/fastwam_aloha_garment_full_finetune_ddp_stride4_bs12_20260704_20260706}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "${BASE_RUN_DIR}")}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-_ddp_stride4_bs12_20260704_20260706}"
SESSION_TS="${SESSION_TS:-$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-fluxvla}"
USE_CONDA="${USE_CONDA:-auto}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
ALLOW_EXISTING_WORK_DIR="${ALLOW_EXISTING_WORK_DIR:-0}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-12}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${MLP_WORKER_GPU:-}" ]]; then
    NPROC_PER_NODE="${MLP_WORKER_GPU}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#_visible_gpus[@]}"
  else
    NPROC_PER_NODE=8
  fi
fi
export NPROC_PER_NODE
export MASTER_PORT="${MASTER_PORT:-29500}"

RUN_PREFIX=()
if [[ "${USE_CONDA}" == "1" || "${USE_CONDA}" == "auto" ]]; then
  CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
  if [[ -z "${CONDA_BIN}" && -x "/root/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/root/miniconda3/bin/conda"
  fi
  if [[ -n "${CONDA_BIN}" && "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    RUN_PREFIX=("${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}")
  elif [[ "${USE_CONDA}" == "1" && -z "${CONDA_BIN}" ]]; then
    echo "[fastwam-aloha-joint-idm] conda not found, but USE_CONDA=1 was requested." >&2
    exit 1
  fi
fi

VARIANT_NAMES=(
  "fastwam_joint_aloha_garment_full_finetune"
  "fastwam_idm_aloha_garment_full_finetune"
)
VARIANT_CONFIGS=(
  "configs/fastwam/fastwam_joint_aloha_garment_full_finetune.py"
  "configs/fastwam/fastwam_idm_aloha_garment_full_finetune.py"
)

mkdir -p "${OUTPUT_ROOT}"
DRIVER_LOG="${OUTPUT_ROOT}/fastwam_aloha_garment_joint_idm_${SESSION_TS}.log"
if [[ "${DRY_RUN}" != "1" ]]; then
  exec > >(tee -a "${DRIVER_LOG}") 2>&1
fi

log() {
  echo "[$(date '+%F %T')] $*"
}

split_env_args() {
  local value="${1:-}"
  local -n out_ref="$2"
  out_ref=()
  if [[ -n "${value}" ]]; then
    read -r -a out_ref <<< "${value}"
  fi
}

resolve_latest_ckpt() {
  local work_dir="$1"
  local checkpoint_dir="${work_dir}/checkpoints"

  if [[ -e "${checkpoint_dir}/latest-checkpoint.safetensors" ]]; then
    readlink -f "${checkpoint_dir}/latest-checkpoint.safetensors"
    return 0
  fi
  if [[ -e "${checkpoint_dir}/latest-checkpoint.pt" ]]; then
    readlink -f "${checkpoint_dir}/latest-checkpoint.pt"
    return 0
  fi
  if [[ -d "${checkpoint_dir}" ]]; then
    local ckpt
    ckpt="$(find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.safetensors' \
      ! -name 'latest-checkpoint.safetensors' -printf '%T@ %p\n' \
      | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
    if [[ -n "${ckpt}" ]]; then
      readlink -f "${ckpt}"
      return 0
    fi

    ckpt="$(find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.pt' \
      ! -name 'latest-checkpoint.pt' -printf '%T@ %p\n' \
      | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
    if [[ -n "${ckpt}" ]]; then
      readlink -f "${ckpt}"
      return 0
    fi
  fi

  return 1
}

ensure_work_dir_is_safe() {
  local work_dir="$1"
  local resume_ckpt="$2"

  if [[ "${DRY_RUN}" == "1" || "${RESUME}" == "1" || "${ALLOW_EXISTING_WORK_DIR}" == "1" ]]; then
    return 0
  fi
  if [[ -d "${work_dir}" && -n "$(find "${work_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[fastwam-aloha-joint-idm] Refusing to train into non-empty work dir: ${work_dir}" >&2
    echo "[fastwam-aloha-joint-idm] Set RUN_NAME_SUFFIX to a new value, or set ALLOW_EXISTING_WORK_DIR=1." >&2
    if [[ -n "${resume_ckpt}" ]]; then
      echo "[fastwam-aloha-joint-idm] Existing checkpoint found. Set RESUME=1 to resume from: ${resume_ckpt}" >&2
    fi
    exit 1
  fi
}

run_cmd() {
  printf '[fastwam-aloha-joint-idm] command:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

split_env_args "${TRAIN_CFG_OPTIONS:-}" TRAIN_CFG_ARGS
split_env_args "${TRAIN_EXTRA_ARGS:-}" TRAIN_ARGS

cd "${REPO_ROOT}"

log "repo=${REPO_ROOT}"
log "base_run_dir=${BASE_RUN_DIR}"
log "output_root=${OUTPUT_ROOT}"
log "session_ts=${SESSION_TS}"
log "nproc_per_node=${NPROC_PER_NODE}"
log "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
log "pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
log "run_prefix=${RUN_PREFIX[*]:-(none)}"
log "driver_log=${DRIVER_LOG}"

for idx in "${!VARIANT_NAMES[@]}"; do
  variant="${VARIANT_NAMES[$idx]}"
  config="${VARIANT_CONFIGS[$idx]}"
  run_name="${variant}${RUN_NAME_SUFFIX}"
  work_dir="${OUTPUT_ROOT}/${run_name}"
  train_log="${work_dir}/${variant}_${SESSION_TS}.train.log"

  log "========== ${variant} =========="
  log "config=${config}"
  log "work_dir=${work_dir}"
  log "train_log=${train_log}"

  existing_ckpt=""
  if existing_ckpt="$(resolve_latest_ckpt "${work_dir}")"; then
    log "existing_ckpt=${existing_ckpt}"
  else
    existing_ckpt=""
  fi

  ensure_work_dir_is_safe "${work_dir}" "${existing_ckpt}"
  mkdir -p "${work_dir}"

  train_cmd=("${RUN_PREFIX[@]}" bash scripts/train.sh "${config}" "${work_dir}")
  if [[ "${RESUME}" == "1" ]]; then
    if [[ -z "${existing_ckpt}" ]]; then
      echo "[fastwam-aloha-joint-idm] RESUME=1 but no checkpoint exists under ${work_dir}/checkpoints" >&2
      exit 1
    fi
    train_cmd+=("--resume-from" "${existing_ckpt}")
  fi
  train_cmd+=(
    "--cfg-options"
    "train_dataloader.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
    "${TRAIN_CFG_ARGS[@]}"
  )
  train_cmd+=("${TRAIN_ARGS[@]}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    run_cmd "${train_cmd[@]}"
  else
    run_cmd "${train_cmd[@]}" 2>&1 | tee -a "${train_log}"
  fi
done

log "all requested variants finished"
log "driver_log=${DRIVER_LOG}"
