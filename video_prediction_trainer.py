"""
Trainer for Video Prediction Task
"""
import os
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torchvision

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from dataset.video_prediction_dataset import FrontCameraVideoDataset, collate_video_batch
from model.video_prediction_diffusion import VideoPredictionDiffusion
from util.dist_util import (
    cleanup,
    init_dist,
    setup_logger,
    set_seed
)


class VideoPredictionTrainer:
    """
    Trainer for front camera video prediction using diffusion models
    """
    def __init__(self, config):
        # Initialize distributed training
        self.rank, self.device = init_dist()
        
        self.config = config
        self.global_config = config['Global']
        self.train_config = config['Train']
        self.dataset_config = config['Dataset']
        self.model_config = config['Model']
        
        # Set seed
        set_seed(seed=self.global_config['global_seed'])
        
        # Setup logger
        self.logger, self.writer = setup_logger(self.global_config, self.rank)
        
        # Initialize wandb (only on rank 0)
        self.use_wandb = WANDB_AVAILABLE and self.global_config.get('use_wandb', False)
        if self.use_wandb and self.rank == 0:
            wandb.init(
                project=self.global_config.get('wandb_project', 'video-prediction'),
                name=self.global_config.get('wandb_run_name', None),
                config=config,
                resume='allow'
            )
            self.logger.info("Wandb initialized successfully")
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.total_epoch = self.train_config['max_epoch']
        self.log_interval = self.train_config['log_every_step']
        self.save_interval = self.train_config['save_every_step']
        self.visualize_interval = self.train_config.get('visualize_every_step', 1000)
        
        # Load datasets
        self.train_loader = self._load_data('train')
        self.eval_loader = self._load_data('eval')
        
        # Build model
        if self.rank == 0:
            self.logger.info("Building model...")
        self.model = VideoPredictionDiffusion(
            vae_model_path=self.model_config['vae_model_path'],
            latent_channels=self.model_config['latent_channels'],
            past_frames=self.model_config['past_frames'],
            future_frames=self.model_config['future_frames'],
            img_size=tuple(self.model_config['img_size']),
            dit_config=self.model_config.get('dit_config', {})
        ).to(self.device)
        if self.rank == 0:
            self.logger.info("Model built successfully")
        
        # Setup EMA
        if self.rank == 0:
            self.logger.info("Setting up EMA model...")
        self.ema = deepcopy(self.model)
        self.requires_grad(self.ema, False)
        if self.rank == 0:
            self.logger.info("EMA model setup complete")
        
        # Setup optimizer
        self.optimizer = self._build_optimizer()
        
        # Load checkpoint if specified
        if self.global_config['load_from']:
            self.load_checkpoint(self.global_config['load_from'])
        
        # Wrap model with DDP (if in distributed mode)
        if dist.is_available() and dist.is_initialized():
            if self.rank == 0:
                self.logger.info("Wrapping model with DDP...")
            self.ddp_model = DDP(self.model, device_ids=[self.device])
            if self.rank == 0:
                self.logger.info("DDP wrapper complete")
        else:
            # Single GPU mode - no DDP wrapper needed
            print("Using single GPU mode - no DDP wrapper")
            self.ddp_model = self.model
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.ddp_model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in self.ddp_model.parameters() if p.requires_grad) / 1e6
        self.logger.info(f"Total parameters: {total_params:.2f}M")
        self.logger.info(f"Trainable parameters: {trainable_params:.2f}M")
    
    def _load_data(self, split):
        """Load dataset and create dataloader"""
        config = self.dataset_config[split]
        
        dataset = FrontCameraVideoDataset(
            data_root=config['data_root'],
            ann_file=config['ann_file'],
            past_frames=config['past_frames'],
            future_frames=config['future_frames'],
            sample_interval=config['sample_interval'],
            img_size=tuple(config['img_size']),
            is_train=(split == 'train')
        )
        
        # Use DistributedSampler only in distributed mode
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=(split == 'train'),
                seed=self.global_config['global_seed']
            )
            shuffle = False
        else:
            sampler = None
            shuffle = (split == 'train')
        
        loader = DataLoader(
            dataset,
            batch_size=self.train_config['batch_size'] if split == 'train' else 1,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.train_config['num_workers'],
            pin_memory=True,
            drop_last=(split == 'train'),
            collate_fn=collate_video_batch
        )
        
        self.logger.info(f"{split} dataset contains {len(dataset):,} samples")
        return loader
    
    def _build_optimizer(self):
        """Build optimizer"""
        # Only optimize DiT parameters, VAE is frozen
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.train_config['lr'],
            weight_decay=self.train_config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        return optimizer
    
    @staticmethod
    def requires_grad(model, flag):
        """Set requires_grad for all parameters in model"""
        for p in model.parameters():
            p.requires_grad = flag
    
    @torch.no_grad()
    def update_ema(self, decay=0.9999):
        """Update EMA model"""
        ema_params = OrderedDict(self.ema.named_parameters())
        model_params = OrderedDict(self.ddp_model.module.named_parameters())
        
        for name, param in model_params.items():
            if name in ema_params:
                ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def save_checkpoint(self, filename=None):
        """Save checkpoint"""
        if self.rank == 0:
            checkpoint = {
                'model': self.ddp_model.module.state_dict(),
                'ema': self.ema.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'step': self.step,
                'epoch': self.epoch
            }
            
            if filename is None:
                filename = f"step_{self.step}.pt"
            
            save_path = os.path.join(self.global_config['save_path'], filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save(checkpoint, save_path)
            self.logger.info(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, path):
        """Load checkpoint"""
        if self.rank == 0:
            self.logger.info(f"Loading checkpoint from {path}")
        
        # Convert device to proper format for map_location
        if isinstance(self.device, int):
            map_location = f'cuda:{self.device}'
        else:
            map_location = self.device
        
        checkpoint = torch.load(path, map_location=map_location)
        
        # Load model state
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'], strict=False)
        
        # Load EMA state
        if 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'], strict=False)
        
        # Load optimizer state
        if 'optimizer' in checkpoint and self.train_config.get('resume_training', False):
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load training state
        if 'step' in checkpoint:
            self.step = checkpoint['step']
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        
        if self.rank == 0:
            self.logger.info(f"Loaded checkpoint (step={self.step}, epoch={self.epoch})")
    
    def train_step(self, batch):
        """Single training step"""
        past_frames = batch['past_frames'].to(self.device)  # [B, T_past, 3, H, W]
        future_frames = batch['future_frames'].to(self.device)  # [B, T_future, 3, H, W]
        task_label = batch['task_label'].to(self.device)  # [B, 4]
        ego_status = batch['ego_status'].to(self.device)  # [B, 9]
        command = batch['command'].to(self.device)  # [B, 6]
        
        # Forward pass
        loss_dict = self.ddp_model(past_frames, future_frames, task_label, ego_status, command)
        loss = loss_dict['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.train_config.get('gradient_clip_val', None):
            torch.nn.utils.clip_grad_norm_(
                self.ddp_model.parameters(),
                self.train_config['gradient_clip_val']
            )
        
        self.optimizer.step()
        
        # Update EMA
        self.update_ema()
        
        return loss_dict
    
    @torch.no_grad()
    def eval_step(self, batch):
        """Single evaluation step"""
        past_frames = batch['past_frames'].to(self.device)
        future_frames = batch['future_frames'].to(self.device)
        task_label = batch['task_label'].to(self.device)
        ego_status = batch['ego_status'].to(self.device)
        command = batch['command'].to(self.device)
        
        # Generate predictions using EMA model
        predicted_frames = self.ema.sample(
            past_frames,
            task_label,
            ego_status,
            command,
            num_inference_steps=self.train_config.get('num_inference_steps', 20)
        )
        
        # Compute metrics
        mse = torch.mean((predicted_frames - future_frames) ** 2)
        psnr = 10 * torch.log10(1.0 / mse)
        
        return {
            'predicted_frames': predicted_frames,
            'mse': mse.item(),
            'psnr': psnr.item()
        }
    
    def train_loop(self):
        """Main training loop"""
        # Initialize EMA with model weights
        self.logger.info("Initializing EMA model with current model weights...")
        self.update_ema(decay=0)
        
        # Verify EMA initialization
        model_param_norm = sum(p.norm().item() for p in self.ddp_model.module.parameters() if p.requires_grad)
        ema_param_norm = sum(p.norm().item() for p in self.ema.parameters())
        self.logger.info(f"Model param norm: {model_param_norm:.4f}, EMA param norm: {ema_param_norm:.4f}")
        
        self.logger.info(f"Starting training for {self.total_epoch} epochs...")
        
        for epoch in range(self.epoch, self.total_epoch):
            self.epoch = epoch
            self.logger.info(f"Epoch {self.epoch + 1}/{self.total_epoch}")
            
            # Set epoch for sampler
            self.train_loader.sampler.set_epoch(self.epoch)
            
            # Train
            self.ddp_model.train()
            self.ema.eval()
            
            for batch_idx, batch in enumerate(self.train_loader):
                loss_dict = self.train_step(batch)
                self.step += 1
                
                # Log
                if self.step % self.log_interval == 0:
                    loss_value = loss_dict['loss'].item()
                    self.logger.info(
                        f"Epoch {self.epoch + 1}, Step {self.step}, Loss: {loss_value:.6f}"
                    )
                    
                    if self.rank == 0:
                        self.writer.add_scalar('train/loss', loss_value, self.step)
                        
                        # Log to wandb
                        if self.use_wandb:
                            log_dict = {
                                'train/loss': loss_value,
                                'train/epoch': self.epoch,
                                'train/step': self.step
                            }
                            wandb.log(log_dict, step=self.step)
                
                # Visualize predictions
                if self.step % self.visualize_interval == 0 and self.rank == 0:
                    self._visualize_training_batch(batch)
                
                # Save checkpoint
                if self.step % self.save_interval == 0:
                    self.save_checkpoint()
                    dist.barrier()
            
            # Evaluation
            if (self.epoch + 1) % self.train_config.get('eval_every_epoch', 1) == 0:
                self.evaluate()
        
        # Final save
        self.save_checkpoint(filename='final.pt')
        self.logger.info("Training completed!")
        cleanup()
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluation loop"""
        self.logger.info("Running evaluation...")
        self.ddp_model.eval()
        self.ema.eval()
        
        total_mse = 0
        total_psnr = 0
        num_samples = 0
        
        save_dir = os.path.join(self.global_config['save_path'], 'eval', f'epoch_{self.epoch}')
        os.makedirs(save_dir, exist_ok=True)
        
        for batch_idx, batch in enumerate(tqdm(self.eval_loader, disable=(self.rank != 0))):
            results = self.eval_step(batch)
            
            total_mse += results['mse']
            total_psnr += results['psnr']
            num_samples += 1
            
            # Save visualization for first few batches
            if batch_idx < 10 and self.rank == 0:
                self._save_visualization(
                    batch,
                    results['predicted_frames'],
                    os.path.join(save_dir, f'sample_{batch_idx}.png')
                )
        
        # Average metrics
        avg_mse = total_mse / num_samples
        avg_psnr = total_psnr / num_samples
        
        if self.rank == 0:
            self.logger.info(f"Evaluation - MSE: {avg_mse:.6f}, PSNR: {avg_psnr:.2f} dB")
            self.writer.add_scalar('eval/mse', avg_mse, self.epoch)
            self.writer.add_scalar('eval/psnr', avg_psnr, self.epoch)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'eval/mse': avg_mse,
                    'eval/psnr': avg_psnr,
                    'eval/epoch': self.epoch
                }, step=self.step)
        
        self.ddp_model.train()
    
    def _save_visualization(self, batch, predicted_frames, save_path):
        """Save visualization of predictions"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        
        past_frames = batch['past_frames'][:1].cpu() * std + mean  # [1, T_past, 3, H, W]
        future_frames = batch['future_frames'][:1].cpu() * std + mean  # [1, T_future, 3, H, W]
        predicted_frames = predicted_frames[:1].cpu() * std + mean  # [1, T_future, 3, H, W]
        
        # Clamp to [0, 1]
        past_frames = torch.clamp(past_frames, 0, 1)
        future_frames = torch.clamp(future_frames, 0, 1)
        predicted_frames = torch.clamp(predicted_frames, 0, 1)
        
        # Create grid: [past frames | ground truth future | predicted future]
        # Rearrange to [N, 3, H, W]
        past_grid = past_frames[0]  # [T_past, 3, H, W]
        gt_grid = future_frames[0]  # [T_future, 3, H, W]
        pred_grid = predicted_frames[0]  # [T_future, 3, H, W]
        
        # Concatenate
        all_frames = torch.cat([past_grid, gt_grid, pred_grid], dim=0)  # [T_past + 2*T_future, 3, H, W]
        
        # Make grid
        grid = torchvision.utils.make_grid(all_frames, nrow=4, padding=2, normalize=False)
        
        # Save
        torchvision.utils.save_image(grid, save_path)
        
        # Log to wandb if enabled
        if self.use_wandb and self.rank == 0:
            # Convert to numpy for wandb (HWC format)
            grid_np = grid.permute(1, 2, 0).numpy()  # [H, W, 3]
            grid_np = (grid_np * 255).astype(np.uint8)
            
            wandb.log({
                'eval/predictions': wandb.Image(
                    grid_np,
                    caption=f"Epoch {self.epoch} - Past | GT | Predicted"
                )
            }, step=self.step)
    
    @torch.no_grad()
    def _visualize_training_batch(self, batch):
        """Visualize predictions during training"""
        num_steps = self.train_config.get('num_inference_steps', 20)
        self.logger.info(f"Generating visualization at step {self.step} with {num_steps} inference steps...")
        
        # Use EMA model for inference
        self.ema.eval()
        
        past_frames = batch['past_frames'][:1].to(self.device)  # Take first sample
        future_frames = batch['future_frames'][:1].to(self.device)
        task_label = batch['task_label'][:1].to(self.device)
        ego_status = batch['ego_status'][:1].to(self.device)
        command = batch['command'][:1].to(self.device)
        
        # Debug: check input ranges
        self.logger.info(f"Past frames range: [{past_frames.min():.3f}, {past_frames.max():.3f}]")
        self.logger.info(f"Future frames range: [{future_frames.min():.3f}, {future_frames.max():.3f}]")
        
        # Generate predictions
        predicted_frames = self.ema.sample(
            past_frames,
            task_label,
            ego_status,
            command,
            num_inference_steps=num_steps
        )
        
        # Debug: check prediction ranges
        self.logger.info(f"Predicted frames range: [{predicted_frames.min():.3f}, {predicted_frames.max():.3f}]")
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(self.device)
        
        past_frames_vis = (past_frames * std + mean).clamp(0, 1)
        future_frames_vis = (future_frames * std + mean).clamp(0, 1)
        predicted_frames_vis = (predicted_frames * std + mean).clamp(0, 1)
        
        # Create comparison visualization
        vis_images = []
        T_past = past_frames_vis.shape[1]
        T_future = future_frames_vis.shape[1]
        
        # Add past frames
        for t in range(T_past):
            vis_images.append(past_frames_vis[0, t])
        
        # Add GT and predicted frames side by side
        for t in range(T_future):
            vis_images.append(future_frames_vis[0, t])
            vis_images.append(predicted_frames_vis[0, t])
        
        # Stack all images
        all_frames = torch.stack(vis_images, dim=0)  # [N, 3, H, W]
        
        # Create grid
        nrow = T_past + 2  # Past frames + 2 columns for GT/Pred pairs
        grid = torchvision.utils.make_grid(all_frames, nrow=nrow, padding=4, normalize=False)
        
        if self.use_wandb:
            # Convert to numpy for wandb
            grid_np = grid.cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
            grid_np = (grid_np * 255).astype(np.uint8)
            
            wandb.log({
                'train/visualization': wandb.Image(
                    grid_np,
                    caption=f"Step {self.step} - Past frames | GT/Pred pairs"
                )
            }, step=self.step)
        
        self.logger.info(f"Visualization logged to wandb")


if __name__ == '__main__':
    # Test
    print("Video Prediction Trainer initialized successfully!")

