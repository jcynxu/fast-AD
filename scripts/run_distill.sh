#!/bin/bash

# Fast-AD training launcher
# Usage: bash scripts/run_distill.sh [config_path] [epochs] [device] [log_dir]

# Defaults
CONFIG_PATH=${1:-"configs/cifar100_config.yaml"}
EPOCHS=${2:-200}
DEVICE=${3:-"cuda"}
LOG_DIR=${4:-"./logs"}

# Check whether config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Warning: Config file $CONFIG_PATH not found, using default config"
    CONFIG_PATH=""
fi

# Run training
echo "Starting Fast-AD training..."
echo "Config: $CONFIG_PATH"
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "Log Dir: $LOG_DIR"
echo ""

python main.py \
    --config "$CONFIG_PATH" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --log-dir "$LOG_DIR"

echo "Training completed!"

