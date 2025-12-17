"""
UNet 模型定义：用于扩散模型
简化版本，实际使用时建议使用 diffusers 库或更完整的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步位置编码"""
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
    """UNet 基础块"""
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
    简化的 UNet 用于扩散模型
    注意：这是一个基础实现，实际使用时建议使用更完整的 UNet 或 diffusers 库
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
        
        # 初始化 alphas_cumprod (用于 DDIM 采样)
        self.register_buffer('alphas_cumprod', self._linear_beta_schedule(1000))
        
    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """线性 beta schedule"""
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod

    def forward(self, x, timestep, context=None):
        """
        Args:
            x: 输入图像 [B, C, H, W]
            timestep: 时间步 [B]
            context: 可选的上下文嵌入 (如文本提示)
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
    UNet 包装类，兼容不同的接口
    """
    def __init__(self, image_channels=3, time_emb_dim=32, num_classes=None):
        super().__init__()
        self.model = SimpleUNet(image_channels, time_emb_dim)
        self.alphas_cumprod = self.model.alphas_cumprod
        
    def forward(self, x, timestep, context=None):
        """
        Args:
            x: 输入图像 [B, C, H, W]
            timestep: 时间步 [B] 或 int
            context: 可选的上下文嵌入
        """
        if isinstance(timestep, int):
            timestep = torch.full((x.size(0),), timestep, device=x.device, dtype=torch.long)
        elif isinstance(timestep, torch.Tensor) and timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
            
        return self.model(x, timestep, context)

