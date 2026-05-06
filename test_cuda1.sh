#!/bin/bash

DATASETS=("brainmri" "clinicdb" "colondb" "kvasir" "endo" "br35h" "ISIC" "dagm")

LOG_DIR="./new_logs"

mkdir -p "${LOG_DIR}"

for DATASET in "${DATASETS[@]}"
do
    echo "Running dataset: ${DATASET}"

    python main.py \
        --mode test \
        --dataset_name "${DATASET}" \
        --start_epochs 0 \
        --end_epochs 30 \
        --device cuda:1 \
        --output_dir "./checkpoints_visa" \
        > "${LOG_DIR}/${DATASET}.log"

    echo "Finished dataset: ${DATASET}"
done