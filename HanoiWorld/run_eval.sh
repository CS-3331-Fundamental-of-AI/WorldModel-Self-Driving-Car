#!/bin/bash

# Simple wrapper to run DreamerV3 evaluation
# Usage: ./run_eval.sh

python evaluate.py \
    --logdir test \
    --config racetrack \
    --device cpu