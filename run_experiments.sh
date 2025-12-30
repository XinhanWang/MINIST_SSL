#!/bin/bash

# Settings
DATASETS=("MNIST" "FashionMNIST")
COUNTS=(20 50 100)
METHODS=("Supervised" "Consistency" "PseudoLabel") 

# 1. Run Baselines and Semi-Supervised
for DATASET in "${DATASETS[@]}"; do
    # Full Supervised Baseline
    echo "Running Full Supervised on $DATASET"
    python train.py --dataset $DATASET --method Supervised --n_labeled -1 --epochs 20

    for COUNT in "${COUNTS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            echo "Running $METHOD on $DATASET with $COUNT labels"
            python train.py --dataset $DATASET --method $METHOD --n_labeled $COUNT --epochs 20
        done
    done
done

# 2. Plot Results
python plot_results.py
