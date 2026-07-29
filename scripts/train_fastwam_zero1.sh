#!/usr/bin/env bash
set -euo pipefail

CONFIG='configs/fastwam/fastwam_libero_full_finetune.py'
WORK_DIR='work_dirs/fastwam_libero_full_finetune'
if (( $# > 0 )); then
  CONFIG=$1
  shift
fi
if (( $# > 0 )); then
  WORK_DIR=$1
  shift
fi

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NUM_MACHINES=${NNODES:-1}
MACHINE_RANK=${NODE_RANK:-0}
MAIN_PROCESS_IP=${MASTER_ADDR:-127.0.0.1}
MAIN_PROCESS_PORT=${MASTER_PORT:-29500}
FASTWAM_CONDA_ENV=${FASTWAM_CONDA_ENV:-fastwam_fluxvla_parity}

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
WORK_DIR=$(realpath -m "${WORK_DIR}")
FASTWAM_TEXT_CACHE_DIR=${FASTWAM_TEXT_CACHE_DIR:-${WORK_DIR}/text_cache}
FASTWAM_TEXT_CACHE_DIR=$(realpath -m "${FASTWAM_TEXT_CACHE_DIR}")
export FASTWAM_TEXT_CACHE_DIR
FASTWAM_HF_DATASETS_CACHE=${FASTWAM_HF_DATASETS_CACHE:-\
${PROJECT_ROOT}/work_dirs/.fastwam_hf_datasets_cache/datasets-3.6.0}
HF_DATASETS_CACHE=$(realpath -m "${FASTWAM_HF_DATASETS_CACHE}")
export HF_DATASETS_CACHE

conda run --no-capture-output -n "${FASTWAM_CONDA_ENV}" accelerate launch \
  --config_file scripts/accelerate_configs/fastwam_zero1.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  scripts/train.py \
  --config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  "$@"
