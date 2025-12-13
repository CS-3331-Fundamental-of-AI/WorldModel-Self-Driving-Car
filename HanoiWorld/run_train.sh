#!/bin/bash
set -e

# Usage: ./run_train.sh [CONFIG] [STEPS] [LOGDIR] [RENDER_EVERY]
# Defaults: CONFIG=racetrack, STEPS=10000, LOGDIR=runs/train, RENDER_EVERY=0 (no render)

CONFIG=${1:-roundabout}
STEPS=${2:-10000}
LOGDIR=${3:-runs/train}
RENDER_EVERY=${4:-0}

python train_actor_rssm.py \
  --config "$CONFIG" \
  --logdir "$LOGDIR" \
  --steps "$STEPS" \
  --render_every "$RENDER_EVERY"

