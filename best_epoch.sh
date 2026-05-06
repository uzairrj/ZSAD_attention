#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_FILE="${MAIN_FILE:-main.py}"
MODE="${MODE:-test}"

GPU0_DEVICE="${GPU0_DEVICE:-cuda:0}"
GPU1_DEVICE="${GPU1_DEVICE:-cuda:1}"

DEFAULT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR:-./checkpoints}"
VISA_OUTPUT_DIR="${VISA_OUTPUT_DIR:-./checkpoints_mvtec}"

LOG_DIR="${LOG_DIR:-./best_epoch_logs}"
LOG_PREFIX="${LOG_PREFIX:-best_epoch_}"

declare -A BEST_EPOCHS=(
    ["mvtec"]=8
    ["visa"]=6
    ["btad"]=8
    ["mpdd"]=3
    ["ksdd"]=4
    ["ksdd2"]=8
    ["dagm"]=4
    ["dtd"]=3
)

mkdir -p "${LOG_DIR}"

run_dataset() {
    local DATASET="$1"
    local DEVICE="$2"

    local EPOCH="${BEST_EPOCHS[${DATASET}]}"
    local START_EPOCH=$((EPOCH - 1))
    local END_EPOCH="${EPOCH}"
    local OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"

    if [[ "${DATASET}" == "visa" ]]; then
        OUTPUT_DIR="${VISA_OUTPUT_DIR}"
    fi

    local CHECKPOINT="${OUTPUT_DIR}/model_epoch_${EPOCH}.pth"
    local LOG_FILE="${LOG_DIR}/${LOG_PREFIX}${DATASET}.log"

    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Missing checkpoint for ${DATASET}: ${CHECKPOINT}" >&2
        exit 1
    fi

    echo "Running dataset: ${DATASET} | epoch: ${EPOCH} | device: ${DEVICE} | checkpoint dir: ${OUTPUT_DIR}"

    "${PYTHON_BIN}" "${MAIN_FILE}" \
        --mode "${MODE}" \
        --dataset_name "${DATASET}" \
        --start_epochs "${START_EPOCH}" \
        --end_epochs "${END_EPOCH}" \
        --device "${DEVICE}" \
        --output_dir "${OUTPUT_DIR}" \
        > "${LOG_FILE}" 2>&1

    echo "Finished dataset: ${DATASET} | device: ${DEVICE} | log: ${LOG_FILE}"
}

run_worker() {
    local DEVICE="$1"
    shift

    for DATASET in "$@"; do
        run_dataset "${DATASET}" "${DEVICE}"
    done
}

GPU0_DATASETS=("mpdd")
GPU1_DATASETS=("ksdd")

run_worker "${GPU0_DEVICE}" "${GPU0_DATASETS[@]}" &
PID0=$!

run_worker "${GPU1_DEVICE}" "${GPU1_DATASETS[@]}" &
PID1=$!

STATUS=0

if ! wait "${PID0}"; then
    echo "GPU 0 worker failed." >&2
    STATUS=1
fi

if ! wait "${PID1}"; then
    echo "GPU 1 worker failed." >&2
    STATUS=1
fi

if [[ "${STATUS}" -ne 0 ]]; then
    echo "At least one worker failed. Check logs in ${LOG_DIR}" >&2
    exit "${STATUS}"
fi

echo "All datasets finished successfully."