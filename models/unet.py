"""
UNet for diffusion (simplified).

For real experiments, consider using a stronger UNet (e.g., from `diffusers`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic UNet block."""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SimpleUNet(nn.Module):
    """
    A minimal UNet for diffusion.

    Note: this is a toy implementation meant for structure and integration;
    swap it with a production-grade diffusion backbone for actual results.
    """
    def __init__(self, image_channels=3, time_emb_dim=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, 64, 3, padding=1)
        
        # Downsampling
        self.down1 = Block(64, 128, time_emb_dim, up=False)
        self.down2 = Block(128, 256, time_emb_dim, up=False)
        self.down3 = Block(256, 512, time_emb_dim, up=False)
        
        # Bottleneck
        self.bot1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bot2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bot3 = nn.Conv2d(1024, 512, 3, padding=1)
        
        # Upsampling
        self.up1 = Block(512, 256, time_emb_dim, up=True)
        self.up2 = Block(256, 128, time_emb_dim, up=True)
        self.up3 = Block(128, 64, time_emb_dim, up=True)
        
        # Output
        self.output = nn.Conv2d(64, image_channels, 1)
        
        # Precompute alphas_cumprod for DDIM sampling
        self.register_buffer('alphas_cumprod', self._linear_beta_schedule(1000))
        
    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """Linear beta schedule."""
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod

    def forward(self, x, timestep, context=None):
        """
        Args:
            x: input image tensor [B, C, H, W]
            timestep: timestep tensor [B]
            context: optional conditioning (e.g., text embeddings)
        """
        # Time embedding
        t = self.time_mlp(timestep)
        
        # Initial conv
        x = self.conv0(x)
        
        # U-Net
        down1 = self.down1(x, t)
        down2 = self.down2(down1, t)
        down3 = self.down3(down2, t)
        
        # Bottleneck
        bot1 = F.relu(self.bot1(down3))
        bot2 = F.relu(self.bot2(bot1))
        bot3 = F.relu(self.bot3(bot2))
        
        # Upsampling
        up1 = self.up1(bot3, t)
        up2 = self.up2(up1, t)
        up3 = self.up3(up2, t)
        
        # Output
        output = self.output(up3)
        return output


class UNet(nn.Module):
    """
    UNet wrapper to keep a stable interface.
    """
    def __init__(self, image_channels=3, time_emb_dim=32, num_classes=None):
        super().__init__()
        self.model = SimpleUNet(image_channels, time_emb_dim)
        self.alphas_cumprod = self.model.alphas_cumprod
        
    def forward(self, x, timestep, context=None):
        """
        Args:
            x: input tensor [B, C, H, W]
            timestep: timestep [B] or int
            context: optional conditioning
        """
        if isinstance(timestep, int):
            timestep = torch.full((x.size(0),), timestep, device=x.device, dtype=torch.long)
        elif isinstance(timestep, torch.Tensor) and timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
            
        return self.model(x, timestep, context)

