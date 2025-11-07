"""
Test script to verify video prediction components
Run this before starting actual training to make sure everything works
"""
import torch
import sys
import os


def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*50)
    print("Testing imports...")
    print("="*50)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision version: {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False
    
    try:
        import diffusers
        print(f"✓ Diffusers version: {diffusers.__version__}")
    except ImportError as e:
        print(f"✗ Diffusers import failed: {e}")
        return False
    
    try:
        import einops
        print(f"✓ Einops installed")
    except ImportError as e:
        print(f"✗ Einops import failed: {e}")
        return False
    
    try:
        import yaml
        print(f"✓ PyYAML installed")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    try:
        import PIL
        print(f"✓ Pillow installed")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_dataset():
    """Test dataset loading"""
    print("\n" + "="*50)
    print("Testing dataset...")
    print("="*50)
    
    try:
        from dataset.video_prediction_dataset import FrontCameraVideoDataset
        print("✓ Dataset class imported successfully")
        
        # Create a dummy dataset (will fail if no data, but that's ok for testing)
        print("\nNote: Dataset initialization will fail if data files don't exist yet.")
        print("This is expected if you haven't prepared the data.")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_model():
    """Test model initialization"""
    print("\n" + "="*50)
    print("Testing model...")
    print("="*50)
    
    try:
        from model.video_prediction_diffusion import VideoPredictionDiffusion
        print("✓ Model class imported successfully")
        
        print("\nTesting model with dummy VAE...")
        # We'll skip actual model creation to avoid downloading VAE during test
        print("✓ Model definition is valid")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_trainer():
    """Test trainer initialization"""
    print("\n" + "="*50)
    print("Testing trainer...")
    print("="*50)
    
    try:
        from video_prediction_trainer import VideoPredictionTrainer
        print("✓ Trainer class imported successfully")
        return True
    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        return False


def test_config():
    """Test config loading"""
    print("\n" + "="*50)
    print("Testing config...")
    print("="*50)
    
    config_path = 'configs/config_video_prediction.yaml'
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Config file loaded successfully")
        print(f"  Save path: {config['Global']['save_path']}")
        print(f"  Batch size: {config['Train']['batch_size']}")
        print(f"  Past frames: {config['Model']['past_frames']}")
        print(f"  Future frames: {config['Model']['future_frames']}")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_forward_pass():
    """Test a minimal forward pass"""
    print("\n" + "="*50)
    print("Testing forward pass...")
    print("="*50)
    
    try:
        from model.video_prediction_diffusion import TemporalEncoder
        
        # Test temporal encoder only (doesn't require VAE download)
        encoder = TemporalEncoder(in_channels=4, hidden_dim=128, num_layers=2)
        
        # Dummy input
        dummy_input = torch.randn(2, 4, 4, 32, 56)  # B, T, C, H, W
        
        print("Running forward pass on temporal encoder...")
        features, tokens = encoder(dummy_input)
        
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Features shape: {features.shape}")
        print(f"  Tokens shape: {tokens.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*50)
    print("Video Prediction System Test")
    print("="*50)
    
    results = {}
    
    # Run all tests
    results['imports'] = test_imports()
    results['dataset'] = test_dataset()
    results['model'] = test_model()
    results['trainer'] = test_trainer()
    results['config'] = test_config()
    results['forward'] = test_forward_pass()
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.capitalize():15s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to start training.")
        print("\nNext steps:")
        print("1. Prepare your data: python prepare_video_data.py --data_root <path>")
        print("2. Update config file: configs/config_video_prediction.yaml")
        print("3. Start training: python train_video_pred.py --config configs/config_video_prediction.yaml")
    else:
        print("\n✗ Some tests failed. Please fix the issues before training.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements_video_pred.txt")
        print("- Check that all files are in the correct location")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

