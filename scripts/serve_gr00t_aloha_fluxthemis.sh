#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_ROOT="${GR00T_ALOHA_CHECKPOINT_ROOT:-/root/projects/ryanhu/checkpoints/gr00t_flod_cloth_aloha_only}"
CHECKPOINT_PATH="${GR00T_ALOHA_CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/checkpoints/step-047720-epoch-20-loss=0.0209.safetensors}"
NORM_STATS_PATH="${CHECKPOINT_ROOT}/dataset_statistics.json"
TOKENIZER_PATH="${CHECKPOINT_ROOT}/tokenizer"
PRETRAINED_MODEL_PATH="${GR00T_PRETRAINED_MODEL_PATH:-/mnt/data/oss/users/liyinhao/projects/GR00T-N1.5-3B}"
CONFIG_PATH="${REPO_ROOT}/configs/gr00t/gr00t_eagle_3b_aloha_fold_cloth_zmq.py"
SERVER_HOST="${FLUXVLA_ZMQ_HOST:-127.0.0.1}"
SERVER_PORT="${FLUXVLA_ZMQ_PORT:-3333}"
DEVICE="${FLUXVLA_DEVICE:-cuda:0}"
DTYPE="${FLUXVLA_DTYPE:-bf16}"

if [[ -n "${FLUXVLA_PYTHON:-}" ]]; then
    PYTHON_BIN="${FLUXVLA_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN='python'
fi

for required_file in "${CONFIG_PATH}" "${CHECKPOINT_PATH}" \
    "${NORM_STATS_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

for required_dir in "${TOKENIZER_PATH}" "${PRETRAINED_MODEL_PATH}"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "Required directory not found: ${required_dir}" >&2
        exit 1
    fi
done

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export GR00T_PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH}"

if ! "${PYTHON_BIN}" -c 'import mmengine, torch, transformers' \
    >/dev/null 2>&1; then
    echo "Selected Python lacks FluxVLA dependencies: ${PYTHON_BIN}" >&2
    echo "Set FLUXVLA_PYTHON to the fluxvla environment interpreter." >&2
    exit 1
fi

if [[ "${1:-}" == '--dry-run' ]]; then
    if [[ "$#" -ne 1 ]]; then
        echo "--dry-run does not accept additional arguments" >&2
        exit 1
    fi
    "${PYTHON_BIN}" -c \
        'import sys; from mmengine import Config; Config.fromfile(sys.argv[1])' \
        "${CONFIG_PATH}"
    echo "Config OK: ${CONFIG_PATH}"
    echo "Checkpoint OK: ${CHECKPOINT_PATH}"
    echo "Normalization statistics OK: ${NORM_STATS_PATH}"
    echo "Tokenizer OK: ${TOKENIZER_PATH}"
    echo "Pretrained model assets OK: ${PRETRAINED_MODEL_PATH}"
    exit 0
fi

exec "${PYTHON_BIN}" -m fluxvla.engines.runners.serving.serve \
    --config "${CONFIG_PATH}" \
    --ckpt-path "${CHECKPOINT_PATH}" \
    --dataset-key inference \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    "$@"
