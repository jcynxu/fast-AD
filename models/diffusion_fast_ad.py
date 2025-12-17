"""
Fast-AD: Fast Adversarial Distillation via Confidence-Aware Dynamic Rectification
核心算法实现：CADR 引导 + DDIM 加速采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FastADConfig:
    """Fast-AD 超参数配置类"""
    def __init__(self, config_dict: Optional[dict] = None):
        # 论文 4.1 节参数
        self.lambda_max = 1.5      # 最大引导强度
        self.eta = 0.1             # 梯度缩放因子 (Gradient Scaling Factor)
        self.tau_ent = 0.4         # 熵阈值 (CIFAR-100为0.4, ImageNet为0.6)
        self.k_sigmoid = 10.0      # Sigmoid 的陡峭程度
        self.ddim_steps = 50       # DDIM 采样步数 (20x 加速)
        self.xi = 1e-6             # 梯度归一化时的数值稳定常数
        self.gamma = 1.0           # CE Loss 的权重
        self.T = 1000              # 原始扩散模型的总步数
        
        # 如果提供了配置字典，则更新参数
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def to_dict(self):
        """转换为字典格式"""
        return {
            'lambda_max': self.lambda_max,
            'eta': self.eta,
            'tau_ent': self.tau_ent,
            'k_sigmoid': self.k_sigmoid,
            'ddim_steps': self.ddim_steps,
            'xi': self.xi,
            'gamma': self.gamma,
            'T': self.T
        }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算分类器输出的熵
    
    Args:
        logits: [B, C] 分类器 logits
        
    Returns:
        entropy: [B] 每个样本的熵值
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy


class FastADGenerator:
    """
    Fast-AD 生成器：实现 CADR 引导的 DDIM 采样
    """
    def __init__(self, diffusion_model, teacher_model, config: FastADConfig, 
                 device='cuda', image_size: Tuple[int, int] = (32, 32), bn_loss_fn=None):
        """
        Args:
            diffusion_model: 预训练的扩散模型 (UNet)
            teacher_model: 预训练的 Teacher 分类器
            config: FastADConfig 配置对象
            device: 计算设备
            image_size: 图像尺寸 (H, W)
            bn_loss_fn: BN Loss 计算函数 (可选)
        """
        self.diffusion = diffusion_model
        self.teacher = teacher_model
        self.cfg = config
        self.device = device
        self.image_size = image_size
        self.bn_loss_fn = bn_loss_fn
        self.teacher.eval()  # Teacher 始终固定

        # 获取 DDIM 的 alpha 序列
        # 假设 diffusion_model 提供了 alphas_cumprod
        if hasattr(self.diffusion, 'alphas_cumprod'):
            self.alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        else:
            # 如果没有，使用线性 schedule 生成
            self.alphas_cumprod = self._linear_beta_schedule(self.cfg.T).to(device)
        
    def _linear_beta_schedule(self, timesteps: int, beta_start: float = 0.0001, 
                              beta_end: float = 0.02) -> torch.Tensor:
        """线性 beta schedule，用于生成 alphas_cumprod"""
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod

    def get_alpha_t(self, t: int) -> torch.Tensor:
        """获取第 t 步的 alpha_cumprod"""
        if isinstance(t, torch.Tensor):
            t = t.item() if t.numel() == 1 else t
        if isinstance(t, (list, tuple)):
            # 批量处理
            indices = torch.tensor(t, device=self.device)
            return self.alphas_cumprod[indices].view(-1, 1, 1, 1)
        return self.alphas_cumprod[t].view(-1, 1, 1, 1)

    def cadr_guidance(self, x_t: torch.Tensor, t: int, noise_pred: torch.Tensor, 
                     target_labels: torch.Tensor, bn_loss_fn=None) -> torch.Tensor:
        """
        论文核心：Confidence-Aware Dynamic Rectification (CADR)
        
        实现论文公式 1, 2, 4, 5, 6，返回用于公式 8 的修正项。
        
        Args:
            x_t: 当前时间步的噪声图像 [B, C, H, W]
            t: 当前时间步
            noise_pred: 预测的噪声 ε_θ(x_t, t, c) [B, C, H, W]
            target_labels: 目标标签 [B]
            bn_loss_fn: BN Loss 计算函数 (可选)
            
        Returns:
            rectified_grads: 修正后的梯度项 g_rect = λ(t) * g_norm * ||x_t|| * η
                            用于公式 8: ε̃_t = ε_θ - √(1-α_t) * g_rect
        """
        # 必须开启梯度计算以获取 Guidance
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)
            
            # --- 步骤 1: State Sensing - Tweedie's Formula 估计 Clean Image (公式 4) ---
            # x̂₀|t = (x_t - √(1-α_t) * ε_θ) / √α_t
            alpha_t = self.get_alpha_t(t)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            # 公式 4: Tweedie's Formula
            x0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            # 将 x0_hat 裁剪到有效范围 [-1, 1]
            x0_hat = torch.clamp(x0_hat, -1, 1)
            
            # --- 步骤 2: 计算 Teacher 熵 (状态感知) ---
            # H = Entropy(T(x̂₀|t))
            teacher_logits = self.teacher(x0_hat)
            entropy = compute_entropy(teacher_logits)
            
            # --- 步骤 3: Gating Decision - 门控决策 (公式 5) ---
            # λ(t) = λ_max / (1 + exp(-k * (H - τ_ent)))
            # 当熵 H 高 (Teacher 不确定) -> exp 变小 -> 分母变小 -> λ 变大 (强引导)
            # 当熵 H 低 (Teacher 自信) -> exp 变大 -> 分母变大 -> λ 变小 (弱引导)
            gate_val = self.cfg.lambda_max / (1 + torch.exp(-self.cfg.k_sigmoid * (entropy - self.cfg.tau_ent)))
            gate_val = gate_val.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            
            # --- 步骤 4: 计算引导梯度 (公式 1, 2) ---
            # LCE(x, y) = CrossEntropy(T(x), y) (公式 2)
            loss_ce = F.cross_entropy(teacher_logits, target_labels)
            
            # LBN(x) = Σ_l ||μ_l(x) - μ^T_l||²_2 + ||σ²_l(x) - σ²^T_l||²_2 (公式 1)
            # BN Loss (如果提供了计算函数)
            if bn_loss_fn is not None:
                # bn_loss_fn 是 BNLoss 实例，直接调用 forward
                loss_bn = bn_loss_fn(x0_hat)
            else:
                loss_bn = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Lguide = LBN + γLCE
            total_loss = loss_bn + self.cfg.gamma * loss_ce
            
            # 计算梯度 grad = d(Loss)/d(x_t)
            grads = torch.autograd.grad(total_loss, x_t, create_graph=False)[0]
            
            # --- 步骤 5: Adaptive Normalization - 自适应梯度归一化 (公式 6) ---
            # g = ∇_{x_t} Lguide(x̂₀|t) = ∇(LBN + γLCE)
            # g_norm = g / (||g||₂ + ξ)  (归一化，防止梯度爆炸)
            grad_norm = torch.norm(grads.view(grads.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
            normalized_grads = grads / (grad_norm + self.cfg.xi)
            
            # signal_strength = ||x_t||₂  (信号强度，用于相对缩放)
            signal_strength = torch.norm(x_t.view(x_t.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
            
            # 公式 6: g_rect = λ(t) * (g / (||g||₂ + ξ)) * ||x_t||₂ * η
            rectified_grads = gate_val * normalized_grads * signal_strength * self.cfg.eta
            
            return rectified_grads

    @torch.no_grad()
    def sample(self, batch_size: int, target_labels: torch.Tensor, 
               prompts_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        基于 DDIM 的加速采样循环 (K=50)
        
        Args:
            batch_size: 批次大小
            target_labels: 目标标签 [B]
            prompts_embeddings: 可选的文本提示嵌入
            
        Returns:
            x: 生成的合成图像 [B, C, H, W]
        """
        # 初始化随机噪声
        img_shape = (batch_size, 3, self.image_size[0], self.image_size[1])
        x = torch.randn(img_shape, device=self.device)
        
        # 定义采样时间步序列 (从大到小)
        step_size = self.cfg.T // self.cfg.ddim_steps
        time_steps = list(range(0, self.cfg.T, step_size))
        time_steps = reversed(time_steps)  # [999, 979, ..., 0]
        
        for i, t in enumerate(time_steps):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # 1. 预测噪声 epsilon_theta
            if hasattr(self.diffusion, '__call__'):
                # 假设 diffusion model 的 forward 签名: (x, t, context=None)
                if prompts_embeddings is not None:
                    noise_pred = self.diffusion(x, t_tensor, context=prompts_embeddings)
                else:
                    noise_pred = self.diffusion(x, t_tensor)
            else:
                # 如果 diffusion 是 UNet，直接调用
                noise_pred = self.diffusion(x, t_tensor)
            
            # 2. 计算 CADR 修正梯度 (核心创新)
            # cadr_guidance 返回 g_rect = λ(t) * g_norm * ||x_t|| * η
            cadr_grad = self.cadr_guidance(x, t, noise_pred, target_labels, self.bn_loss_fn)
            
            # 3. 修正噪声 (公式 8)
            # ε̃_t = ε_θ(x_t, t, c) - √(1-α_t) * [λ_max/(1+e^(-k(H-τ))) * ∇(LBN+γLCE)/(||∇||+ξ) * ||x_t|| * η]
            alpha_t = self.get_alpha_t(t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            rectified_noise = noise_pred - sqrt_one_minus_alpha_t * cadr_grad
            
            # 4. DDIM Update Step (公式 7)
            # x_{t-1} = √α_{t-1} * x̂₀ + √(1-α_{t-1}) * ε̃_t
            # 获取当前步和下一步的 alpha
            alpha_t = self.get_alpha_t(t)
            prev_t = max(0, t - step_size)
            alpha_prev = self.get_alpha_t(prev_t) if prev_t >= 0 else torch.tensor(1.0, device=self.device).view(-1, 1, 1, 1)
            
            # 预测 x0 (使用修正后的噪声)
            # x̂₀ = (x_t - √(1-α_t) * ε̃_t) / √α_t
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * rectified_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)  # 裁剪到有效范围
            
            # 公式 7: x_{t-1} = √α_{t-1} * x̂₀ + √(1-α_{t-1}) * ε̃_t
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * rectified_noise
            
        return x  # 返回生成的合成图像

