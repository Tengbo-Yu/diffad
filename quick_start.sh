#!/bin/bash
# Quick Start Script for Video Prediction Training
# This script helps you set up and start training

echo "=========================================="
echo "Video Prediction - Quick Start"
echo "=========================================="

# Step 1: Check Python version
echo ""
echo "[Step 1/5] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Step 2: Install dependencies
echo ""
echo "[Step 2/5] Installing dependencies..."
read -p "Install required packages? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install -r requirements_video_pred.txt
    echo "Dependencies installed!"
else
    echo "Skipping dependency installation..."
fi

# Step 3: Prepare data
echo ""
echo "[Step 3/5] Preparing dataset..."
read -p "Do you need to prepare data? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    read -p "Enter your data root path (e.g., data/bench2drive): " data_root
    python prepare_video_data.py \
        --data_root "$data_root" \
        --output_dir data/infos \
        --val_ratio 0.2 \
        --verify
    echo "Data preparation complete!"
else
    echo "Skipping data preparation..."
fi

# Step 4: Update config
echo ""
echo "[Step 4/5] Configuring training..."
echo "Please edit configs/config_video_prediction.yaml to set your data paths"
read -p "Press Enter after you've checked the config file..." dummy

# Step 5: Start training
echo ""
echo "[Step 5/5] Starting training..."
echo ""
echo "Choose training mode:"
echo "1) Single GPU"
echo "2) Multi-GPU (4 GPUs)"
echo "3) Multi-GPU (8 GPUs)"
echo "4) Custom"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Starting single GPU training..."
        python train_video_pred.py --config configs/config_video_prediction.yaml
        ;;
    2)
        echo "Starting 4-GPU training..."
        torchrun --nproc_per_node=4 train_video_pred.py --config configs/config_video_prediction.yaml
        ;;
    3)
        echo "Starting 8-GPU training..."
        torchrun --nproc_per_node=8 train_video_pred.py --config configs/config_video_prediction.yaml
        ;;
    4)
        read -p "Enter number of GPUs: " num_gpus
        echo "Starting $num_gpus-GPU training..."
        torchrun --nproc_per_node=$num_gpus train_video_pred.py --config configs/config_video_prediction.yaml
        ;;
    *)
        echo "Invalid choice. Please run the training command manually:"
        echo "python train_video_pred.py --config configs/config_video_prediction.yaml"
        ;;
esac

echo ""
echo "=========================================="
echo "To monitor training:"
echo "tensorboard --logdir logs/video_prediction"
echo "=========================================="

