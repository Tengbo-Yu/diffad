#!/bin/bash
# Single GPU evaluation
# Add --use_train_set flag to evaluate on training set (more diverse scenarios)
CUDA_VISIBLE_DEVICES=4 python eval_video_pred.py \
    --config configs/config_video_prediction.yaml \
    --ckpt checkpoints/video_prediction/step_50000.pt \
    --num_samples 100 \
    --save_visualizations \
    --num_inference_steps 50 \
    --use_train_set
