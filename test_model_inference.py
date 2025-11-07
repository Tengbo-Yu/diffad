"""
Test script to verify model inference is working correctly
"""
import torch
import yaml
from model.video_prediction_diffusion import VideoPredictionDiffusion
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_inference():
    """Test model inference with dummy data"""
    
    # Load config
    with open('configs/config_video_prediction.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['Model']
    
    # Create model
    logger.info("Creating model...")
    model = VideoPredictionDiffusion(
        vae_model_path=model_config['vae_model_path'],
        latent_channels=model_config['latent_channels'],
        past_frames=config['Dataset']['train']['past_frames'],
        future_frames=config['Dataset']['train']['future_frames'],
        img_size=tuple(config['Dataset']['train']['img_size']),
        dit_config=model_config['dit_config']
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on {device}")
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Total parameters: {total_params:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params:.2f}M")
    
    # Create dummy input (normalized)
    B = 1
    T_past = config['Dataset']['train']['past_frames']
    T_future = config['Dataset']['train']['future_frames']
    H, W = config['Dataset']['train']['img_size']
    
    logger.info(f"\nTesting with: B={B}, T_past={T_past}, T_future={T_future}, H={H}, W={W}")
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    
    # Random frames in [0, 1], then normalize
    past_frames_raw = torch.rand(B, T_past, 3, H, W)
    past_frames = (past_frames_raw - mean) / std
    past_frames = past_frames.to(device)
    
    future_frames_raw = torch.rand(B, T_future, 3, H, W)
    future_frames = (future_frames_raw - mean) / std
    future_frames = future_frames.to(device)
    
    task_label = torch.randn(B, 4).to(device)
    ego_status = torch.randn(B, 9).to(device)
    command = torch.zeros(B, 6).to(device)
    command[:, 0] = 1  # One-hot
    
    logger.info(f"Input ranges - Past: [{past_frames.min():.3f}, {past_frames.max():.3f}]")
    logger.info(f"Input ranges - Future: [{future_frames.min():.3f}, {future_frames.max():.3f}]")
    
    # Test forward pass (training)
    logger.info("\n=== Testing forward pass (training) ===")
    with torch.no_grad():
        loss_dict = model(past_frames, future_frames, task_label, ego_status, command)
    
    logger.info(f"Loss: {loss_dict['loss'].item():.6f}")
    logger.info(f"Noise pred shape: {loss_dict['noise_pred'].shape}")
    logger.info(f"Noise true shape: {loss_dict['noise_true'].shape}")
    
    # Test sampling (inference)
    logger.info("\n=== Testing sampling (inference) with 5 steps ===")
    with torch.no_grad():
        predicted_frames = model.sample(
            past_frames,
            task_label,
            ego_status,
            command,
            num_inference_steps=5  # Fast test
        )
    
    logger.info(f"Predicted frames shape: {predicted_frames.shape}")
    logger.info(f"Predicted frames range: [{predicted_frames.min():.3f}, {predicted_frames.max():.3f}]")
    
    # Denormalize and check
    predicted_denorm = (predicted_frames * std.to(device) + mean.to(device)).clamp(0, 1)
    logger.info(f"Denormalized predicted range: [{predicted_denorm.min():.3f}, {predicted_denorm.max():.3f}]")
    
    # Test with more steps
    logger.info("\n=== Testing sampling (inference) with 20 steps ===")
    with torch.no_grad():
        predicted_frames_20 = model.sample(
            past_frames,
            task_label,
            ego_status,
            command,
            num_inference_steps=20
        )
    
    logger.info(f"Predicted frames (20 steps) range: [{predicted_frames_20.min():.3f}, {predicted_frames_20.max():.3f}]")
    predicted_denorm_20 = (predicted_frames_20 * std.to(device) + mean.to(device)).clamp(0, 1)
    logger.info(f"Denormalized predicted (20 steps) range: [{predicted_denorm_20.min():.3f}, {predicted_denorm_20.max():.3f}]")
    
    # Check if predictions are reasonable
    if predicted_denorm_20.min() < 0 or predicted_denorm_20.max() > 1:
        logger.warning("⚠️ Predicted frames are out of [0, 1] range after denormalization!")
    else:
        logger.info("✓ Predicted frames are in valid range [0, 1]")
    
    # Check if predictions are not just noise
    pred_mean = predicted_denorm_20.mean().item()
    pred_std = predicted_denorm_20.std().item()
    logger.info(f"Predicted statistics - Mean: {pred_mean:.3f}, Std: {pred_std:.3f}")
    
    if pred_mean < 0.2 or pred_mean > 0.8:
        logger.warning(f"⚠️ Predicted mean {pred_mean:.3f} is unusual (expected around 0.3-0.7)")
    if pred_std < 0.05 or pred_std > 0.4:
        logger.warning(f"⚠️ Predicted std {pred_std:.3f} is unusual (expected around 0.1-0.3)")
    
    logger.info("\n✓ Model inference test completed successfully!")
    return True

if __name__ == '__main__':
    try:
        test_model_inference()
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

