"""
Video Prediction Model using Diffusion
Predicts future front camera frames from historical frames
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from diffusers.models import AutoencoderKL
from model.dit_modules.dit import DiT


class TemporalEncoder(nn.Module):
    """
    Encode temporal sequence of past frames into BEV-like representation
    Outputs bevfeature compatible with DiT
    """
    def __init__(self, in_channels=4, bevfeat_dim=256, num_layers=4):
        super().__init__()
        self.bevfeat_dim = bevfeat_dim
        
        # Conv3D for temporal encoding
        self.conv3d_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels if i == 0 else bevfeat_dim, 
                         bevfeat_dim, 
                         kernel_size=(3, 3, 3), 
                         stride=(1, 1, 1),
                         padding=1),
                nn.GroupNorm(32, bevfeat_dim),
                nn.SiLU(),
            ) for i in range(num_layers)
        ])
        
        # Temporal pooling
        self.temporal_pool = nn.Conv3d(bevfeat_dim, bevfeat_dim, 
                                       kernel_size=(3, 1, 1), 
                                       stride=(1, 1, 1), 
                                       padding=(1, 0, 0))
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] - past frames in latent space
        Returns:
            bevfeature: [B, bevfeat_dim, H, W] - BEV-like temporal features
        """
        B, T, C, H, W = x.shape
        
        # Rearrange for Conv3D: [B, C, T, H, W]
        x = rearrange(x, 'b t c h w -> b c t h w')
        
        # Apply 3D convolutions
        for layer in self.conv3d_layers:
            x = layer(x)
        
        # Pool over temporal dimension
        x = self.temporal_pool(x)  # [B, bevfeat_dim, T, H, W]
        
        # Average over time to get single feature map
        bevfeature = x.mean(dim=2)  # [B, bevfeat_dim, H, W]
        
        return bevfeature


class VideoDiT(nn.Module):
    """
    Diffusion Transformer for Video Prediction
    Uses the original DiT architecture with all conditioning inputs
    """
    def __init__(self, 
                 input_size=(32, 56),  # Latent spatial size (H/8, W/8)
                 in_channels=4,  # VAE latent channels
                 hidden_size=1152,
                 depth=28,
                 num_heads=16,
                 patch_size=2,
                 mlp_ratio=4.0,
                 bevfeat_dim=256,  # BEV feature dimension from temporal encoder
                 command_dim=6,
                 learn_sigma=False):
        super().__init__()
        
        # Use original DiT architecture
        self.dit = DiT(
            depth=depth,
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_heads=num_heads,
            input_size=input_size,
            in_channels=in_channels,
            bevfeat_dim=bevfeat_dim,
            command_dim=command_dim,
            mlp_ratio=mlp_ratio,
            learn_sigma=learn_sigma
        )
        
    def forward(self, x, t, bevfeature, task_label, ego_status, command):
        """
        Args:
            x: [B, C, H, W] - noisy latent of future frame
            t: [B] - diffusion timestep
            bevfeature: [B, bevfeat_dim, H, W] - temporal features from encoder
            task_label: [B, 4] - task labels
            ego_status: [B, 9] - ego vehicle status
            command: [B, command_dim] - navigation command
        Returns:
            noise_pred: [B, C, H, W] - predicted noise
        """
        # Forward through original DiT with all conditions
        # prop_prev_x_start is None for initial prediction
        output = self.dit(
            x=x,
            timestep=t,
            bevfeature=bevfeature,
            prop_prev_x_start=None,
            task_label=task_label,
            ego_status=ego_status,
            command=command
        )
        
        return output


class VideoPredictionDiffusion(nn.Module):
    """
    Complete Video Prediction Model
    """
    def __init__(self, 
                 vae_model_path,
                 latent_channels=4,
                 past_frames=4,
                 future_frames=4,
                 img_size=(256, 448),
                 dit_config=None):
        super().__init__()
        
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.latent_channels = latent_channels
        self.img_size = img_size
        
        # VAE for encoding/decoding images
        print(f"Loading VAE from {vae_model_path}...")
        self.vae = AutoencoderKL.from_pretrained(vae_model_path, local_files_only=True)
        print("VAE loaded successfully")
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Calculate latent size (VAE downsamples by 8)
        latent_h = img_size[0] // 8
        latent_w = img_size[1] // 8
        
        # Temporal encoder for past frames
        bevfeat_dim = dit_config.get('bevfeat_dim', 256) if dit_config else 256
        self.temporal_encoder = TemporalEncoder(
            in_channels=latent_channels,
            bevfeat_dim=bevfeat_dim,
            num_layers=4
        )
        
        # DiT model for diffusion
        dit_config = dit_config or {}
        self.dit = VideoDiT(
            input_size=(latent_h, latent_w),
            in_channels=latent_channels,
            hidden_size=dit_config.get('hidden_size', 1152),
            depth=dit_config.get('depth', 28),
            num_heads=dit_config.get('num_heads', 16),
            patch_size=dit_config.get('patch_size', 2),
            mlp_ratio=dit_config.get('mlp_ratio', 4.0),
            bevfeat_dim=bevfeat_dim,
            command_dim=dit_config.get('command_dim', 6),
            learn_sigma=dit_config.get('learn_sigma', False)
        )
        
        self.vae_scale_factor = 0.18215
        
    @torch.no_grad()
    def encode_frames(self, frames):
        """
        Encode frames to latent space
        Args:
            frames: [B, T, C, H, W] or [B, C, H, W]
        Returns:
            latents: same shape with C replaced by latent_channels
        """
        if frames.ndim == 5:
            B, T, C, H, W = frames.shape
            frames_flat = rearrange(frames, 'b t c h w -> (b t) c h w')
            latents = self.vae.encode(frames_flat).latent_dist.sample()
            latents = latents * self.vae_scale_factor
            latents = rearrange(latents, '(b t) c h w -> b t c h w', b=B, t=T)
        else:
            latents = self.vae.encode(frames).latent_dist.sample()
            latents = latents * self.vae_scale_factor
        
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents):
        """
        Decode latents to images
        Args:
            latents: [B, T, C, H, W] or [B, C, H, W]
        Returns:
            frames: same shape with C=3
        """
        if latents.ndim == 5:
            B, T, C, H, W = latents.shape
            latents_flat = rearrange(latents, 'b t c h w -> (b t) c h w')
            latents_flat = latents_flat / self.vae_scale_factor
            frames = self.vae.decode(latents_flat).sample
            frames = rearrange(frames, '(b t) c h w -> b t c h w', b=B, t=T)
        else:
            latents = latents / self.vae_scale_factor
            frames = self.vae.decode(latents).sample
        
        return frames
    
    def forward(self, past_frames, future_frames, task_label, ego_status, command, noise=None):
        """
        Forward pass for training
        Args:
            past_frames: [B, T_past, 3, H, W] - historical frames
            future_frames: [B, T_future, 3, H, W] - ground truth future frames
            task_label: [B, 4] - task labels
            ego_status: [B, 9] - ego vehicle status
            command: [B, command_dim] - navigation command
            noise: Optional pre-sampled noise
        Returns:
            loss_dict: Dictionary containing losses
        """
        B, T_future, _, _, _ = future_frames.shape
        device = past_frames.device
        
        # Encode past and future frames to latent space
        with torch.no_grad():
            past_latents = self.encode_frames(past_frames)  # [B, T_past, C, H, W]
            future_latents = self.encode_frames(future_frames)  # [B, T_future, C, H, W]
        
        # Encode temporal information from past frames to bevfeature
        bevfeature = self.temporal_encoder(past_latents)  # [B, bevfeat_dim, H, W]
        
        # For each future frame, apply diffusion
        # We'll predict all future frames together for efficiency
        # Flatten temporal dimension
        future_latents_flat = rearrange(future_latents, 'b t c h w -> (b t) c h w')
        
        # Repeat bevfeature and other conditions for all future frames
        bevfeature_repeated = repeat(bevfeature, 'b c h w -> (b repeat) c h w', repeat=T_future)
        task_label_repeated = repeat(task_label, 'b d -> (b repeat) d', repeat=T_future)
        ego_status_repeated = repeat(ego_status, 'b d -> (b repeat) d', repeat=T_future)
        command_repeated = repeat(command, 'b d -> (b repeat) d', repeat=T_future)
        
        # Sample timesteps
        timesteps = torch.randint(
            0, 1000, (B * T_future,), device=device
        ).long()
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(future_latents_flat)
        
        # Add noise to future latents (forward diffusion)
        # Using DDPM formulation: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        alphas_cumprod = self._get_alphas_cumprod(device)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[timesteps])
        
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1)
        
        noisy_latents = (
            sqrt_alphas_cumprod * future_latents_flat +
            sqrt_one_minus_alphas_cumprod * noise
        )
        
        # Predict noise using DiT with all conditions
        noise_pred = self.dit(
            x=noisy_latents,
            t=timesteps,
            bevfeature=bevfeature_repeated,
            task_label=task_label_repeated,
            ego_status=ego_status_repeated,
            command=command_repeated
        )
        
        # Compute loss (simple MSE between predicted and true noise)
        loss = F.mse_loss(noise_pred, noise, reduction='mean')
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise_true': noise
        }
    
    def _get_alphas_cumprod(self, device, num_timesteps=1000):
        """
        Get cumulative alpha values for DDPM
        """
        # Linear beta schedule
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod
    
    @torch.no_grad()
    def sample(self, past_frames, task_label, ego_status, command, num_inference_steps=50):
        """
        Generate future frames from past frames
        Args:
            past_frames: [B, T_past, 3, H, W]
            task_label: [B, 4] - task labels
            ego_status: [B, 9] - ego vehicle status
            command: [B, command_dim] - navigation command
            num_inference_steps: Number of denoising steps
        Returns:
            future_frames: [B, T_future, 3, H, W]
        """
        B, T_past, C, H, W = past_frames.shape
        device = past_frames.device
        
        # Encode past frames
        past_latents = self.encode_frames(past_frames)
        
        # Encode temporal information to bevfeature
        bevfeature = self.temporal_encoder(past_latents)  # [B, bevfeat_dim, H, W]
        
        # Initialize with random noise for future frames
        latent_h, latent_w = H // 8, W // 8
        future_latents_shape = (B * self.future_frames, self.latent_channels, latent_h, latent_w)
        latents = torch.randn(future_latents_shape, device=device)
        
        # Repeat conditions for all future frames
        bevfeature_repeated = repeat(bevfeature, 'b c h w -> (b repeat) c h w', repeat=self.future_frames)
        task_label_repeated = repeat(task_label, 'b d -> (b repeat) d', repeat=self.future_frames)
        ego_status_repeated = repeat(ego_status, 'b d -> (b repeat) d', repeat=self.future_frames)
        command_repeated = repeat(command, 'b d -> (b repeat) d', repeat=self.future_frames)
        
        # DDIM sampling
        alphas_cumprod = self._get_alphas_cumprod(device)
        timesteps = torch.linspace(999, 0, num_inference_steps, device=device).long()
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B * self.future_frames)
            
            # Predict noise
            noise_pred = self.dit(
                x=latents,
                t=t_batch,
                bevfeature=bevfeature_repeated,
                task_label=task_label_repeated,
                ego_status=ego_status_repeated,
                command=command_repeated
            )
            
            # DDIM update with proper formulation
            alpha_t = alphas_cumprod[t]
            alpha_t_prev = alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            
            # Predict x0 from x_t and noise
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            pred_x0 = (latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            # Note: No clipping here as it would destroy the signal at early timesteps
            # where pred_x0 can have large values (e.g., [-200, 200] at t=999)
            
            # Compute direction pointing to x_t
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
            
            # DDIM deterministic update
            latents = sqrt_alpha_t_prev * pred_x0 + sqrt_one_minus_alpha_t_prev * noise_pred
        
        # Reshape back to [B, T_future, C, H, W]
        latents = rearrange(latents, '(b t) c h w -> b t c h w', b=B, t=self.future_frames)
        
        # Decode to images
        future_frames = self.decode_latents(latents)
        
        return future_frames


if __name__ == '__main__':
    # Test the model
    print("Testing VideoPredictionDiffusion model...")
    
    model = VideoPredictionDiffusion(
        vae_model_path='stabilityai/sd-vae-ft-mse',
        past_frames=4,
        future_frames=4,
        img_size=(256, 448)
    )
    
    # Dummy input
    past_frames = torch.randn(2, 4, 3, 256, 448)
    future_frames = torch.randn(2, 4, 3, 256, 448)
    task_label = torch.randn(2, 4)
    ego_status = torch.randn(2, 9)
    command = torch.zeros(2, 6)
    command[:, 0] = 1  # One-hot for first command
    
    # Forward pass
    loss_dict = model(past_frames, future_frames, task_label, ego_status, command)
    print(f"Loss: {loss_dict['loss'].item():.4f}")
    
    # Sampling
    with torch.no_grad():
        predicted_frames = model.sample(past_frames, task_label, ego_status, command, num_inference_steps=10)
    print(f"Predicted frames shape: {predicted_frames.shape}")

