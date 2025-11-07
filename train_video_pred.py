"""
Training script for Front Camera Video Prediction
Usage:
    # Single GPU
    python train_video_pred.py --config configs/config_video_prediction.yaml
    
    # Multi-GPU (e.g., 4 GPUs)
    torchrun --nproc_per_node=4 train_video_pred.py --config configs/config_video_prediction.yaml
    
    # Override config with command line arguments
    torchrun --nproc_per_node=2 train_video_pred.py \
        --config configs/config_video_prediction.yaml \
        --Global.save_path checkpoints/overfit \
        --Model.past_frames 8
"""
import argparse
import yaml
from video_prediction_trainer import VideoPredictionTrainer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def update_config_from_args(config, args):
    """Update config dict with command line arguments"""
    def smart_type_convert(value):
        """Convert string value to appropriate type"""
        if not isinstance(value, str):
            return value
        
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try to convert to bool
        if value.lower() in ['true', 'yes', '1']:
            return True
        elif value.lower() in ['false', 'no', '0']:
            return False
        
        # Try to parse as list (e.g., "[256, 448]")
        if value.startswith('[') and value.endswith(']'):
            try:
                import ast
                return ast.literal_eval(value)
            except:
                pass
        
        # Return as string if no conversion worked
        return value
    
    for arg_name, arg_value in vars(args).items():
        if arg_name in ['config', 'resume'] or arg_value is None:
            continue
        
        # Parse nested config keys (e.g., "Global.save_path" -> config['Global']['save_path'])
        if '.' in arg_name:
            keys = arg_name.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            # Convert value to appropriate type
            current[keys[-1]] = smart_type_convert(arg_value)
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Front Camera Video Prediction Model',
                                     allow_abbrev=False)
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume training from')
    
    # Parse known args first to get config-specific overrides
    args, unknown = parser.parse_known_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Add additional arguments dynamically for config overrides
    for arg in unknown:
        if arg.startswith('--'):
            arg_name = arg[2:]
            parser.add_argument(f'--{arg_name}', type=str)
    
    # Parse all arguments
    args = parser.parse_args()
    
    # Override with command line arguments
    if args.resume:
        config['Global']['load_from'] = args.resume
        config['Train']['resume_training'] = True
    
    # Update config with additional command line arguments
    config = update_config_from_args(config, args)
    
    # Create trainer and start training
    trainer = VideoPredictionTrainer(config)
    trainer.train_loop()


if __name__ == '__main__':
    main()

