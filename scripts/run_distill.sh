#!/bin/bash

# Fast-AD 训练启动脚本
# Usage: bash scripts/run_distill.sh [config_path] [epochs] [device]

# 默认参数
CONFIG_PATH=${1:-"configs/cifar100_config.yaml"}
EPOCHS=${2:-200}
DEVICE=${3:-"cuda"}
LOG_DIR=${4:-"./logs"}

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Warning: Config file $CONFIG_PATH not found, using default config"
    CONFIG_PATH=""
fi

# 运行训练
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

