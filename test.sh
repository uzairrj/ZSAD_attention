#!/bin/bash

DATASET="visa"

LOG_DIR="./logs_all"

mkdir -p "${LOG_DIR}"


echo "Running dataset: ${DATASET}"

python main.py \
    --mode test \
    --dataset_name "${DATASET}" \
    --start_epochs 0 \
    --end_epochs 30 \
    --device cuda:1 \
    --output_dir "./checkpoints_mvtec" \
    > "${LOG_DIR}/${DATASET}.log"

echo "Finished dataset: ${DATASET}"
