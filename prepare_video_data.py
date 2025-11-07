"""
Data preparation script for Bench2Drive video prediction
This script helps you prepare the annotation files from raw Bench2Drive data

Usage:
    python prepare_video_data.py --data_root data/bench2drive --output_dir data/infos
"""
import argparse
import os
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np


def load_bench2drive_annotations(data_root):
    """
    Load Bench2Drive dataset annotations
    Expected structure:
        data_root/
            SceneName_Town_Route_Weather/
                camera/
                    rgb_front/
                        *.png
                    rgb_front_left/
                    ...
                anno/
                lidar/
            ...
    """
    print(f"Loading annotations from {data_root}")
    
    all_infos = []
    scenes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_path = os.path.join(data_root, scene)
        
        # Check if camera/rgb_front directory exists
        rgb_front_path = os.path.join(scene_path, 'camera', 'rgb_front')
        if not os.path.exists(rgb_front_path):
            # print(f"Warning: {rgb_front_path} does not exist, skipping...")
            continue
        
        # Look for annotation file
        ann_file = os.path.join(scene_path, 'anno', 'annotations.json')
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
        else:
            # If no annotation file, create minimal info from image files
            # print(f"No annotation file found in {scene_path}, creating from image files...")
            annotations = create_minimal_annotations(rgb_front_path, scene)
        
        # Process annotations
        for ann in annotations:
            info = {
                'scene_token': ann.get('scene_token', scene),
                'clip_id': ann.get('clip_id', scene),
                'timestamp': ann.get('timestamp', 0),
                'frame_id': ann.get('frame_id', 0),
                'cams': {
                    'cam_front': {
                        'data_path': ann.get('rgb_front_path', ''),
                        'type': 'camera',
                        'sample_data_token': ann.get('token', '')
                    }
                }
            }
            
            # Add other camera views if available
            for cam_name in ['cam_front_left', 'cam_front_right', 
                            'cam_back', 'cam_back_left', 'cam_back_right']:
                if cam_name in ann:
                    info['cams'][cam_name] = {
                        'data_path': ann[cam_name],
                        'type': 'camera'
                    }
            
            # Add 3D bounding boxes if available
            if 'gt_boxes' in ann:
                info['gt_boxes'] = ann['gt_boxes']
            if 'gt_names' in ann:
                info['gt_names'] = ann['gt_names']
            
            all_infos.append(info)
    
    print(f"Loaded {len(all_infos)} frame annotations")
    return all_infos


def create_minimal_annotations(rgb_front_path, scene_name):
    """
    Create minimal annotations from image files when no annotation file exists
    """
    annotations = []
    
    # Get all PNG and JPG files sorted by name
    image_files = sorted(list(Path(rgb_front_path).glob('*.png')) + 
                        list(Path(rgb_front_path).glob('*.jpg')))
    
    for idx, img_file in enumerate(image_files):
        # Extract frame info from filename
        # Assuming format like: frame_000123.png or 000123.png
        filename = img_file.name
        
        # Try to extract frame number
        try:
            if 'frame_' in filename:
                frame_id = int(filename.split('frame_')[1].split('.')[0])
            else:
                # Remove extension and try to parse as number
                frame_id = int(filename.split('.')[0])
        except:
            frame_id = idx
        
        # Create relative path: scene_name/camera/rgb_front/filename
        rel_path = os.path.join(scene_name, 'camera', 'rgb_front', filename)
        
        annotation = {
            'scene_token': scene_name,
            'clip_id': scene_name,
            'timestamp': frame_id * 0.1,  # Assume 10Hz
            'frame_id': frame_id,
            'rgb_front_path': rel_path,
            'token': f"{scene_name}_{frame_id}"
        }
        
        annotations.append(annotation)
    
    return annotations


def split_train_val(infos, val_ratio=0.2, overfit=False, seed=42):
    """
    Split data into train and validation sets
    
    Args:
        infos: List of annotation dictionaries
        val_ratio: Ratio of validation data
        overfit: If True, use the same data for train and val (for overfitting tests)
        seed: Random seed (only used if shuffling is needed)
    """
    np.random.seed(seed)
    
    # Group by scene/clip
    scenes = {}
    for info in infos:
        scene_token = info.get('scene_token', info.get('clip_id', 'unknown'))
        if scene_token not in scenes:
            scenes[scene_token] = []
        scenes[scene_token].append(info)
    
    # Sort frames within each scene by timestamp/frame_id for temporal consistency
    for scene_token in scenes:
        scenes[scene_token] = sorted(
            scenes[scene_token], 
            key=lambda x: (x.get('timestamp', 0), x.get('frame_id', 0))
        )
    
    scene_list = sorted(scenes.keys())  # Sort scene names for deterministic split
    
    if overfit:
        # Use all scenes for both train and val
        print("OVERFIT MODE: Using same data for train and validation")
        train_scenes = scene_list
        val_scenes = scene_list
    else:
        # Split scenes by ratio (deterministic, no shuffle)
        split_idx = int(len(scene_list) * (1 - val_ratio))
        train_scenes = scene_list[:split_idx]
        val_scenes = scene_list[split_idx:]
    
    # Collect infos
    train_infos = []
    val_infos = []
    
    for scene in train_scenes:
        train_infos.extend(scenes[scene])
    
    for scene in val_scenes:
        val_infos.extend(scenes[scene])
    
    print(f"Train scenes: {len(train_scenes)}, frames: {len(train_infos)}")
    print(f"Val scenes: {len(val_scenes)}, frames: {len(val_infos)}")
    
    return train_infos, val_infos


def save_pkl(data, output_path):
    """Save data to pickle file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Wrap in expected format
    data_dict = {
        'infos': data,
        'metadata': {
            'version': 'v1.0',
            'num_samples': len(data)
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Saved {len(data)} samples to {output_path}")


def verify_data(infos, data_root):
    """
    Verify that image files exist
    """
    print("Verifying data...")
    missing_count = 0
    
    for info in tqdm(infos[:100], desc="Checking"):  # Check first 100 samples
        cam_front_path = info['cams']['cam_front']['data_path']
        full_path = os.path.join(data_root, cam_front_path)
        
        if not os.path.exists(full_path):
            missing_count += 1
            if missing_count <= 5:  # Print first 5 missing files
                print(f"Missing: {full_path}")
    
    if missing_count > 0:
        print(f"Warning: {missing_count} files missing out of 100 checked")
    else:
        print("All checked files exist!")
    
    return missing_count == 0


def main():
    parser = argparse.ArgumentParser(description='Prepare Bench2Drive data for video prediction')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of Bench2Drive dataset')
    parser.add_argument('--output_dir', type=str, default='data/infos',
                       help='Output directory for processed annotations')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set ratio (ignored if --overfit is set)')
    parser.add_argument('--overfit', action='store_true',
                       help='Use same data for train and val (for overfitting tests)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify that all files exist')
    
    args = parser.parse_args()
    
    print("="*50)
    print("Bench2Drive Video Prediction Data Preparation")
    print("="*50)
    
    # Load annotations
    all_infos = load_bench2drive_annotations(args.data_root)
    
    if len(all_infos) == 0:
        print("Error: No data found!")
        print(f"Please check that {args.data_root} contains the Bench2Drive dataset")
        return
    
    # Verify data if requested
    if args.verify:
        if not verify_data(all_infos, args.data_root):
            print("Warning: Some files are missing. Training may fail.")
    
    # Split train/val
    train_infos, val_infos = split_train_val(
        all_infos, 
        val_ratio=args.val_ratio,
        overfit=args.overfit
    )
    
    # Save
    train_output = os.path.join(args.output_dir, 'b2d_infos_train.pkl')
    val_output = os.path.join(args.output_dir, 'b2d_infos_val.pkl')
    
    save_pkl(train_infos, train_output)
    save_pkl(val_infos, val_output)
    
    print("\n" + "="*50)
    print("Data preparation completed!")
    print("="*50)
    print(f"Train set: {train_output}")
    print(f"Val set: {val_output}")
    print("\nYou can now run training with:")
    print("python train_video_pred.py --config configs/config_video_prediction.yaml")


if __name__ == '__main__':
    main()

