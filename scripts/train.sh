CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train_video_pred.py \
    --config configs/config_video_prediction.yaml