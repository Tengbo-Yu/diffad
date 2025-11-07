"""
Training script for Front Camera Video Prediction
Usage:
    # Single GPU
    python train_video_pred.py --config configs/config_video_prediction.yaml
    
    # Multi-GPU (e.g., 4 GPUs)
    torchrun --nproc_per_node=4 train_video_pred.py --config configs/config_video_prediction.yaml
"""
import argparse
import yaml
from video_prediction_trainer import VideoPredictionTrainer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Front Camera Video Prediction Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.resume:
        config['Global']['load_from'] = args.resume
        config['Train']['resume_training'] = True
    
    # Create trainer and start training
    trainer = VideoPredictionTrainer(config)
    trainer.train_loop()


if __name__ == '__main__':
    main()

