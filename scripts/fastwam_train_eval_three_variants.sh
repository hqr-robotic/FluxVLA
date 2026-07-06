#!/usr/bin/env bash
# Train and evaluate the three full-suite FastWAM variants sequentially.
#
# Default output layout mirrors the historical FastWAM runs:
#   /root/projects/ryanhu/FluxVLA/FastWAM/<run_name>/
#   /root/projects/ryanhu/FluxVLA/FastWAM/evaluate_results/libero/<run_name>/<timestamp>/
#
# Useful overrides:
#   NPROC_PER_NODE=8
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   RUN_NAME_SUFFIX=_my_experiment
#   SKIP_TRAIN=1
#   SKIP_EVAL=1
#   RESUME=1
#   TRAIN_EXTRA_ARGS="--cfg-options runner.max_epochs=1"
#   NUM_TRIALS_PER_TASK=10 SAVE_ROLLOUT_VIDEOS=False

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/projects/FluxVLA}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/projects/ryanhu/FluxVLA/FastWAM}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_ROOT}/evaluate_results/libero}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-_30hz_statefix_ddp_stride6_bs16}"
SESSION_TS="${SESSION_TS:-$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-fluxvla}"
USE_CONDA="${USE_CONDA:-auto}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
RESUME="${RESUME:-0}"
ALLOW_EXISTING_WORK_DIR="${ALLOW_EXISTING_WORK_DIR:-0}"
EVAL_CONFIG_SOURCE="${EVAL_CONFIG_SOURCE:-work_dir}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

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
export NUM_GPUS="${NUM_GPUS:-${NPROC_PER_NODE}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29690}"

RUN_PREFIX=()
if [[ "${USE_CONDA}" == "1" || "${USE_CONDA}" == "auto" ]]; then
  CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
  if [[ -z "${CONDA_BIN}" && -x "/root/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/root/miniconda3/bin/conda"
  fi
  if [[ -n "${CONDA_BIN}" && "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    RUN_PREFIX=("${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}")
  elif [[ "${USE_CONDA}" == "1" && -z "${CONDA_BIN}" ]]; then
    echo "[fastwam-3x] conda not found, but USE_CONDA=1 was requested." >&2
    exit 1
  fi
fi

VARIANT_NAMES=(
  "fastwam_libero_full_finetune"
  "fastwam_idm_libero_full_finetune"
  "fastwam_joint_libero_full_finetune"
)
VARIANT_CONFIGS=(
  "configs/fastwam/fastwam_libero_full_finetune.py"
  "configs/fastwam/fastwam_idm_libero_full_finetune.py"
  "configs/fastwam/fastwam_joint_libero_full_finetune.py"
)

mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}"
DRIVER_LOG="${OUTPUT_ROOT}/fastwam_3variants_${SESSION_TS}.log"
exec > >(tee -a "${DRIVER_LOG}") 2>&1

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

  if [[ "${SKIP_TRAIN}" == "1" || "${RESUME}" == "1" ]]; then
    return 0
  fi
  if [[ "${ALLOW_EXISTING_WORK_DIR}" == "1" ]]; then
    return 0
  fi
  if [[ -d "${work_dir}" && -n "$(find "${work_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[fastwam-3x] Refusing to train into non-empty work dir: ${work_dir}" >&2
    echo "[fastwam-3x] Set RUN_NAME_SUFFIX to a new value, or set ALLOW_EXISTING_WORK_DIR=1." >&2
    if [[ -n "${resume_ckpt}" ]]; then
      echo "[fastwam-3x] Existing checkpoint found. Set RESUME=1 to resume from: ${resume_ckpt}" >&2
    fi
    exit 1
  fi
}

run_with_log() {
  local label="$1"
  local log_file="$2"
  shift 2

  mkdir -p "$(dirname "${log_file}")"
  log "START ${label}"
  printf '[fastwam-3x] command:' | tee -a "${log_file}"
  printf ' %q' "$@" | tee -a "${log_file}"
  printf '\n' | tee -a "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
  log "DONE ${label}"
}

split_env_args "${TRAIN_EXTRA_ARGS:-}" TRAIN_ARGS
split_env_args "${EVAL_EXTRA_ARGS:-}" EVAL_ARGS

cd "${REPO_ROOT}"

log "repo=${REPO_ROOT}"
log "output_root=${OUTPUT_ROOT}"
log "eval_root=${EVAL_ROOT}"
log "session_ts=${SESSION_TS}"
log "nproc_per_node=${NPROC_PER_NODE}"
log "run_prefix=${RUN_PREFIX[*]:-(none)}"
log "driver_log=${DRIVER_LOG}"

for idx in "${!VARIANT_NAMES[@]}"; do
  variant="${VARIANT_NAMES[$idx]}"
  config="${VARIANT_CONFIGS[$idx]}"
  run_name="${variant}${RUN_NAME_SUFFIX}"
  work_dir="${OUTPUT_ROOT}/${run_name}"
  train_log="${work_dir}/${variant}_${SESSION_TS}.train.log"
  eval_dir="${EVAL_ROOT}/${run_name}/${SESSION_TS}"
  eval_log="${eval_dir}/manager_${SESSION_TS}.log"

  log "========== ${variant} =========="
  log "config=${config}"
  log "work_dir=${work_dir}"
  log "eval_dir=${eval_dir}"

  existing_ckpt=""
  if existing_ckpt="$(resolve_latest_ckpt "${work_dir}")"; then
    log "existing_ckpt=${existing_ckpt}"
  else
    existing_ckpt=""
  fi

  ensure_work_dir_is_safe "${work_dir}" "${existing_ckpt}"
  mkdir -p "${work_dir}"

  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    train_cmd=("${RUN_PREFIX[@]}" bash scripts/train.sh "${config}" "${work_dir}")
    if [[ "${RESUME}" == "1" ]]; then
      if [[ -z "${existing_ckpt}" ]]; then
        echo "[fastwam-3x] RESUME=1 but no checkpoint exists under ${work_dir}/checkpoints" >&2
        exit 1
      fi
      train_cmd+=("--resume-from" "${existing_ckpt}")
    fi
    train_cmd+=("${TRAIN_ARGS[@]}")
    run_with_log "train ${variant}" "${train_log}" "${train_cmd[@]}"
  else
    log "skip train for ${variant}"
  fi

  if [[ "${SKIP_EVAL}" == "1" ]]; then
    log "skip eval for ${variant}"
    continue
  fi

  ckpt="$(resolve_latest_ckpt "${work_dir}")"
  log "eval_ckpt=${ckpt}"

  eval_config="${config}"
  if [[ "${EVAL_CONFIG_SOURCE}" == "work_dir" ]]; then
    if [[ -f "${work_dir}/config.yaml" ]]; then
      eval_config="${work_dir}/config.yaml"
    elif [[ -f "${work_dir}/config.json" ]]; then
      eval_config="${work_dir}/config.json"
    fi
  fi

  mkdir -p "${eval_dir}"
  eval_cmd=(env "OUTPUT_DIR=${eval_dir}" "${RUN_PREFIX[@]}" bash scripts/eval_libero_manager.sh "${eval_config}" "${ckpt}")
  eval_cmd+=("${EVAL_ARGS[@]}")
  run_with_log "eval ${variant}" "${eval_log}" "${eval_cmd[@]}"
done

log "all variants finished"
log "driver_log=${DRIVER_LOG}"
