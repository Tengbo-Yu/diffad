# DiffAD Video Prediction

A diffusion-based video prediction model for autonomous driving, trained on Bench2Drive dataset.

## Quick Start

### 1. Download Chencpoints
Download the VAE from HuggingFace, and modify the path in config.yaml
```bash
huggingface-cli download --resume-download stabilityai/sd-vae-ft-mse --local-dir /path
```

### 2. Data Preprocessing

Process raw Bench2Drive data to create training annotations:

```bash
python prepare_video_data.py \
    --data_root b2d_data/your_scenario \
    --output_dir b2d_data/your_scenario/infos \
    --verify
```

**For overfitting test (single scenario):**
```bash
python prepare_video_data.py \
    --data_root b2d_data/overfit \
    --output_dir b2d_data/overfit/infos \
    --overfit \
    --verify
```

### 3. Training

**Single GPU:**
```bash
python train_video_pred.py --config configs/config_video_prediction.yaml
```

**Multi-GPU (recommended):**
```bash
torchrun --nproc_per_node=2 train_video_pred.py \
    --config configs/config_video_prediction.yaml
```

**Resume from checkpoint:**
```bash
torchrun --nproc_per_node=2 train_video_pred.py \
    --config configs/config_video_prediction.yaml \
    --resume checkpoints/video_prediction/step_50000.pt
```

### 4. Evaluation

```bash
python eval_video_pred.py \
    --config configs/config_video_prediction.yaml \
    --ckpt checkpoints/video_prediction/step_50000.pt \
    --num_samples 100 \
    --save_visualizations \
    --num_inference_steps 50
```

**Evaluate on training set:**
```bash
python eval_video_pred.py \
    --config configs/config_video_prediction.yaml \
    --ckpt checkpoints/video_prediction/step_50000.pt \
    --num_samples 100 \
    --save_visualizations \
    --num_inference_steps 50 \
    --use_train_set
```


## Configuration

Edit `configs/config_video_prediction.yaml` to customize:
- Model architecture (DiT hidden size, depth, heads)
- Training hyperparameters (learning rate, batch size, epochs)
- Dataset paths and frame settings
- Logging and checkpointing intervals

## Requirements
- See `requirements.txt` for full dependencies

## Directory Structure

```
├── configs/                    # Configuration files
├── dataset/                    # Dataset loaders
├── model/                      # Model definitions
├── diffusion/                  # Diffusion/flow matching algorithms
├── checkpoints/                # Saved model checkpoints
├── logs/                       # Training logs
├── prepare_video_data.py       # Data preprocessing
├── train_video_pred.py         # Training script
├── eval_video_pred.py          # Evaluation script
└── inference_demo.py           # Inference demo
```

