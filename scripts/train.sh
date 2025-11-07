#!/bin/bash

# Activate conda environment
source /media/raid/workspace/tengbo/anaconda3/etc/profile.d/conda.sh
conda activate diff

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
torchrun --nproc_per_node=2 --master_port=29510 train_video_pred.py \
    --config configs/config_video_prediction.yaml \
    --Global.load_from /media/raid/workspace/tengbo/DiffAD-main/checkpoints/overfit/step_70000.pt \
    --Global.save_path checkpoints/overfit_scratch \
    --Global.log_dir logs/overfit_scratch \
    --Global.tb_path logs/overfit_scratch \
    --Global.wandb_run_name overfit_scratch \
    --Dataset.train.data_root b2d_data/overfit \
    --Dataset.train.ann_file b2d_data/overfit/VehicleTurningRoute_Town15_Route443_Weather1/infos/b2d_infos_train.pkl \
    --Dataset.eval.data_root b2d_data/overfit \
    --Dataset.eval.ann_file b2d_data/overfit/VehicleTurningRoute_Town15_Route443_Weather1/infos/b2d_infos_val.pkl \