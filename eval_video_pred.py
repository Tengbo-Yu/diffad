"""
Evaluation script for Front Camera Video Prediction
Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
    train_video_prediction.py \
    --config configs/config_video_prediction.yaml --ckpt checkpoints/video_prediction/step_50000.pt
"""
import argparse
import yaml
import os
import torch
from tqdm import tqdm
import torchvision
import numpy as np
from video_prediction_trainer import VideoPredictionTrainer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def denormalize_frames(frames):
    """
    Denormalize frames from ImageNet normalization
    Args:
        frames: Tensor [B, T, 3, H, W] or [T, 3, H, W]
    Returns:
        frames: Denormalized and clamped to [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    if frames.ndim == 5:
        mean = mean.view(1, 1, 3, 1, 1)
        std = std.view(1, 1, 3, 1, 1)
    else:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    
    frames = frames * std + mean
    frames = torch.clamp(frames, 0, 1)
    return frames


def compute_metrics(pred_frames, gt_frames):
    """
    Compute evaluation metrics
    Args:
        pred_frames: [B, T, 3, H, W]
        gt_frames: [B, T, 3, H, W]
    Returns:
        dict with metrics
    """
    # MSE
    mse = torch.mean((pred_frames - gt_frames) ** 2).item()
    
    # PSNR
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
    
    # MAE
    mae = torch.mean(torch.abs(pred_frames - gt_frames)).item()
    
    # SSIM (simplified version using correlation)
    # For proper SSIM, consider using pytorch-msssim library
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae
    }


def save_video_comparison(past_frames, gt_future, pred_future, save_path, clip_id):
    """
    Save a video showing: past frames | ground truth future | predicted future
    """
    # Denormalize all frames
    past_frames = denormalize_frames(past_frames.cpu())  # [B, T_past, 3, H, W]
    gt_future = denormalize_frames(gt_future.cpu())  # [B, T_future, 3, H, W]
    pred_future = denormalize_frames(pred_future.cpu())  # [B, T_future, 3, H, W]
    
    B = past_frames.shape[0]
    
    for b in range(B):
        # Get frames for this sample
        past = past_frames[b]  # [T_past, 3, H, W]
        gt = gt_future[b]  # [T_future, 3, H, W]
        pred = pred_future[b]  # [T_future, 3, H, W]
        
        # Concatenate all frames
        all_frames = torch.cat([past, gt, pred], dim=0)  # [T_past + 2*T_future, 3, H, W]
        
        # Create grid
        grid = torchvision.utils.make_grid(all_frames, nrow=4, padding=2, normalize=False)
        
        # Save image
        sample_save_path = save_path.replace('.png', f'_{clip_id[b]}.png')
        torchvision.utils.save_image(grid, sample_save_path)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Video Prediction Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of denoising steps')
    parser.add_argument('--use_train_set', action='store_true',
                       help='Use training set instead of validation set for evaluation (more diverse scenarios)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['Global']['load_from'] = args.ckpt
    config['Train']['num_inference_steps'] = args.num_inference_steps
    
    # Create trainer (which loads the model)
    print("Initializing model...")
    trainer = VideoPredictionTrainer(config)
    
    # Set to eval mode
    trainer.ddp_model.eval()
    trainer.ema.eval()
    
    # Choose which dataloader to use
    if args.use_train_set:
        print("Using TRAINING set for evaluation (more diverse scenarios)")
        eval_loader = trainer.train_loader
        dataset_name = 'train'
    else:
        print("Using VALIDATION set for evaluation")
        eval_loader = trainer.eval_loader
        dataset_name = 'val'
    
    # Create output directory
    output_dir = os.path.join(config['Global']['save_path'], f'evaluation_{dataset_name}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating on {args.num_samples} samples from {dataset_name} set...")
    
    # Evaluation loop
    all_metrics = {
        'mse': [],
        'psnr': [],
        'mae': []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader)):
            if batch_idx >= args.num_samples:
                break
            
            past_frames = batch['past_frames'].to(trainer.device)
            future_frames = batch['future_frames'].to(trainer.device)
            task_label = batch['task_label'].to(trainer.device)
            ego_status = batch['ego_status'].to(trainer.device)
            command = batch['command'].to(trainer.device)
            
            # Generate predictions using EMA model
            predicted_frames = trainer.ema.sample(
                past_frames,
                task_label,
                ego_status,
                command,
                num_inference_steps=args.num_inference_steps
            )
            
            # Compute metrics
            metrics = compute_metrics(predicted_frames, future_frames)
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            # Save visualizations
            if args.save_visualizations and batch_idx < 20:
                save_path = os.path.join(output_dir, f'sample_{batch_idx:04d}.png')
                save_video_comparison(
                    past_frames,
                    future_frames,
                    predicted_frames,
                    save_path,
                    batch['clip_ids']
                )
    
    # Compute average metrics
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    
    for key, values in all_metrics.items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        print(f"{key.upper():8s}: {avg_value:.6f} ± {std_value:.6f}")
    
    print("="*50)
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*50 + "\n")
        for key, values in all_metrics.items():
            avg_value = np.mean(values)
            std_value = np.std(values)
            f.write(f"{key.upper():8s}: {avg_value:.6f} ± {std_value:.6f}\n")
        f.write("="*50 + "\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Num samples: {len(all_metrics['mse'])}\n")
        f.write(f"Num inference steps: {args.num_inference_steps}\n")
    
    print(f"\nMetrics saved to {metrics_file}")
    if args.save_visualizations:
        print(f"Visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

