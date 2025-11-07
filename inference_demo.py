"""
Demo script for video prediction inference
Load a trained model and generate future frames from sample images
"""
import argparse
import torch
import yaml
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import os


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config_path, checkpoint_path):
    """Load trained model from checkpoint"""
    from model.video_prediction_diffusion import VideoPredictionDiffusion
    
    config = load_config(config_path)
    model_config = config['Model']
    
    print("Loading model...")
    model = VideoPredictionDiffusion(
        vae_model_path=model_config['vae_model_path'],
        latent_channels=model_config['latent_channels'],
        past_frames=model_config['past_frames'],
        future_frames=model_config['future_frames'],
        img_size=tuple(model_config['img_size']),
        dit_config=model_config.get('dit_config', {})
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema'], strict=False)
        print("Loaded EMA model")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
        print("Loaded model")
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    return model, config


def load_images(image_paths, img_size=(256, 448)):
    """
    Load images from paths
    Args:
        image_paths: List of image paths
        img_size: Target size (H, W)
    Returns:
        Tensor [T, 3, H, W]
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
    
    return torch.stack(images, dim=0)  # [T, 3, H, W]


def denormalize(frames):
    """Denormalize frames"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    frames = frames * std + mean
    frames = torch.clamp(frames, 0, 1)
    return frames


@torch.no_grad()
def predict(model, past_frames, device='cuda', num_steps=50):
    """
    Generate future frames
    Args:
        model: Trained model
        past_frames: [T_past, 3, H, W]
        device: Device to run on
        num_steps: Number of denoising steps
    Returns:
        future_frames: [T_future, 3, H, W]
    """
    model = model.to(device)
    past_frames = past_frames.unsqueeze(0).to(device)  # [1, T_past, 3, H, W]
    
    print(f"Generating future frames with {num_steps} denoising steps...")
    future_frames = model.sample(past_frames, num_inference_steps=num_steps)
    
    future_frames = future_frames.squeeze(0)  # [T_future, 3, H, W]
    return future_frames


def visualize_results(past_frames, future_frames, save_path):
    """
    Create visualization of past and predicted future frames
    Args:
        past_frames: [T_past, 3, H, W]
        future_frames: [T_future, 3, H, W]
        save_path: Path to save visualization
    """
    # Denormalize
    past_frames = denormalize(past_frames)
    future_frames = denormalize(future_frames)
    
    # Concatenate
    all_frames = torch.cat([past_frames, future_frames], dim=0)  # [T_past+T_future, 3, H, W]
    
    # Create grid
    grid = torchvision.utils.make_grid(all_frames, nrow=4, padding=4, normalize=False)
    
    # Save
    torchvision.utils.save_image(grid, save_path)
    print(f"Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Video Prediction Inference Demo')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                       help='Paths to input images (4 images for past frames)')
    parser.add_argument('--output', type=str, default='prediction.png',
                       help='Output path for visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of denoising steps (more=better quality but slower)')
    
    args = parser.parse_args()
    
    # Check inputs
    if len(args.images) != 4:
        print("Error: Please provide exactly 4 input images for past frames")
        return
    
    # Load model
    model, config = load_model(args.config, args.ckpt)
    img_size = tuple(config['Model']['img_size'])
    
    # Load input images
    print(f"Loading images from: {args.images}")
    past_frames = load_images(args.images, img_size)
    print(f"Input shape: {past_frames.shape}")
    
    # Predict
    future_frames = predict(
        model, 
        past_frames, 
        device=args.device,
        num_steps=args.num_steps
    )
    print(f"Output shape: {future_frames.shape}")
    
    # Visualize
    visualize_results(past_frames, future_frames, args.output)
    
    print("\nDone! Check the visualization at:", args.output)


if __name__ == '__main__':
    # Example usage:
    # python inference_demo.py \
    #     --config configs/config_video_prediction.yaml \
    #     --ckpt checkpoints/video_prediction/step_50000.pt \
    #     --images frame1.png frame2.png frame3.png frame4.png \
    #     --output prediction.png \
    #     --num_steps 50
    
    main()

