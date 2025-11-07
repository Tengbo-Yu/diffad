"""
Front Camera Video Prediction Dataset for Bench2Drive
Predicts future frames from historical front camera images
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FrontCameraVideoDataset(Dataset):
    """
    Dataset for front camera video prediction task
    
    Args:
        data_root (str): Root directory of Bench2Drive dataset
        ann_file (str): Annotation file path (e.g., b2d_infos_train.pkl)
        past_frames (int): Number of historical frames to use as input (default: 4)
        future_frames (int): Number of future frames to predict (default: 4)
        sample_interval (int): Interval between frames (default: 5, i.e., 0.5s)
        img_size (tuple): Target image size (H, W) (default: (256, 448))
        is_train (bool): Training mode flag
    """
    
    def __init__(self,
                 data_root,
                 ann_file,
                 past_frames=4,
                 future_frames=4,
                 sample_interval=5,
                 img_size=(256, 448),
                 is_train=True):
        
        self.data_root = data_root
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.sample_interval = sample_interval
        self.img_size = img_size
        self.is_train = is_train
        
        # Load annotation file
        with open(ann_file, 'rb') as f:
            self.data_infos = pickle.load(f)['infos']
        
        print(f"Total raw samples: {len(self.data_infos)}")
        
        # Build valid sample indices
        # Each sample needs: past_frames + future_frames consecutive frames
        self.valid_samples = self._build_valid_samples()
        print(f"Valid samples for video prediction: {len(self.valid_samples)}")
        
        # Data transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _build_valid_samples(self):
        """
        Build list of valid sample indices
        A sample is valid if it has enough past and future frames in the same scene
        """
        valid_samples = []
        
        # Group frames by scene/clip_id
        scene_dict = {}
        for idx, info in enumerate(self.data_infos):
            clip_id = info.get('scene_token', info.get('clip_id', 'unknown'))
            timestamp = info.get('timestamp', idx)
            
            if clip_id not in scene_dict:
                scene_dict[clip_id] = []
            scene_dict[clip_id].append({
                'idx': idx,
                'timestamp': timestamp,
                'info': info
            })
        
        # Sort each scene by timestamp
        for clip_id in scene_dict:
            scene_dict[clip_id] = sorted(scene_dict[clip_id], key=lambda x: x['timestamp'])
        
        # Find valid temporal windows
        total_frames_needed = self.past_frames + self.future_frames
        
        for clip_id, frames in scene_dict.items():
            if len(frames) < total_frames_needed:
                continue
            
            # Sliding window through the scene
            for i in range(len(frames) - total_frames_needed + 1):
                window = frames[i:i + total_frames_needed]
                
                # Check if frames are consecutive (considering sample_interval)
                timestamps = [f['timestamp'] for f in window]
                indices = [f['idx'] for f in window]
                
                # Check if indices are consecutive (simple and reliable)
                # Allow for small gaps (e.g., missing frames)
                is_consecutive = all(indices[j+1] - indices[j] <= self.sample_interval + 2 
                                   for j in range(len(indices)-1))
                
                if is_consecutive:
                    valid_samples.append({
                        'clip_id': clip_id,
                        'past_indices': [f['idx'] for f in window[:self.past_frames]],
                        'future_indices': [f['idx'] for f in window[self.past_frames:]]
                    })
        
        return valid_samples
    
    def _load_image(self, img_path):
        """Load and preprocess a single image"""
        full_path = os.path.join(self.data_root, img_path)
        try:
            img = Image.open(full_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (1600, 928), color=(0, 0, 0))
            return self.transform(img)
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'past_frames': Tensor [T_past, C, H, W] - historical frames
                'future_frames': Tensor [T_future, C, H, W] - ground truth future frames
                'task_label': Tensor [4] - task labels
                'ego_status': Tensor [9] - ego vehicle status
                'command': Tensor [6] - navigation command (one-hot)
                'clip_id': str - scene identifier
                'metadata': dict - additional metadata
            }
        """
        sample = self.valid_samples[idx]
        
        # Load past frames (input)
        past_frames = []
        for frame_idx in sample['past_indices']:
            info = self.data_infos[frame_idx]
            # Get front camera image path
            cam_front_path = info['cams']['cam_front']['data_path']
            img_tensor = self._load_image(cam_front_path)
            past_frames.append(img_tensor)
        
        # Load future frames (ground truth)
        future_frames = []
        for frame_idx in sample['future_indices']:
            info = self.data_infos[frame_idx]
            cam_front_path = info['cams']['cam_front']['data_path']
            img_tensor = self._load_image(cam_front_path)
            future_frames.append(img_tensor)
        
        # Stack into tensors
        past_frames = torch.stack(past_frames, dim=0)  # [T_past, C, H, W]
        future_frames = torch.stack(future_frames, dim=0)  # [T_future, C, H, W]
        
        # Get conditioning information from the last past frame
        last_past_info = self.data_infos[sample['past_indices'][-1]]
        
        # Extract task label (dummy for now, should be extracted from dataset)
        # Task label format: [lane_change, overtake, turn, stop]
        task_label = torch.zeros(4, dtype=torch.float32)
        task_label[0] = 1.0  # Default: no special task
        
        # Extract ego status (dummy for now, should be extracted from dataset)
        # Ego status format: [x, y, z, vx, vy, vz, yaw, pitch, roll]
        ego_status = torch.zeros(9, dtype=torch.float32)
        if 'ego_status' in last_past_info:
            ego_status = torch.tensor(last_past_info['ego_status'], dtype=torch.float32)
        
        # Extract command (dummy for now, should be extracted from dataset)
        # Command format: one-hot [follow_lane, turn_left, turn_right, go_straight, stop, change_lane]
        command = torch.zeros(6, dtype=torch.float32)
        command[0] = 1.0  # Default: follow lane
        if 'command' in last_past_info:
            cmd_idx = last_past_info['command']
            if 0 <= cmd_idx < 6:
                command = torch.zeros(6, dtype=torch.float32)
                command[cmd_idx] = 1.0
        
        return {
            'past_frames': past_frames,
            'future_frames': future_frames,
            'task_label': task_label,
            'ego_status': ego_status,
            'command': command,
            'clip_id': sample['clip_id'],
            'metadata': {
                'past_indices': sample['past_indices'],
                'future_indices': sample['future_indices']
            }
        }


def collate_video_batch(batch):
    """
    Collate function for video prediction dataloader
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        dict: Batched data
    """
    past_frames = torch.stack([item['past_frames'] for item in batch], dim=0)
    future_frames = torch.stack([item['future_frames'] for item in batch], dim=0)
    task_label = torch.stack([item['task_label'] for item in batch], dim=0)
    ego_status = torch.stack([item['ego_status'] for item in batch], dim=0)
    command = torch.stack([item['command'] for item in batch], dim=0)
    
    return {
        'past_frames': past_frames,  # [B, T_past, C, H, W]
        'future_frames': future_frames,  # [B, T_future, C, H, W]
        'task_label': task_label,  # [B, 4]
        'ego_status': ego_status,  # [B, 9]
        'command': command,  # [B, 6]
        'clip_ids': [item['clip_id'] for item in batch],
        'metadata': [item['metadata'] for item in batch]
    }


if __name__ == '__main__':
    # Test the dataset
    dataset = FrontCameraVideoDataset(
        data_root='data/bench2drive',
        ann_file='data/infos/b2d_infos_train.pkl',
        past_frames=4,
        future_frames=4,
        sample_interval=5,
        is_train=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Past frames shape: {sample['past_frames'].shape}")
        print(f"Future frames shape: {sample['future_frames'].shape}")
        print(f"Clip ID: {sample['clip_id']}")

