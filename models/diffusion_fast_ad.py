"""
Fast-AD: Fast Adversarial Distillation via Confidence-Aware Dynamic Rectification

Core implementation: CADR guidance + DDIM accelerated sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FastADConfig:
    """Fast-AD hyperparameter container."""
    def __init__(self, config_dict: Optional[dict] = None):
        # Paper hyperparameters (Sec. 4.1)
        self.lambda_max = 1.5      # Max rectification strength
        self.eta = 0.1             # Gradient scaling factor
        self.tau_ent = 0.4         # Entropy threshold (CIFAR-100: 0.4, ImageNet: 0.6)
        self.k_sigmoid = 10.0      # Sigmoid steepness
        self.ddim_steps = 50       # Number of DDIM steps (~20x speedup)
        self.xi = 1e-6             # Epsilon for numerical stability (grad norm)
        self.gamma = 1.0           # CE loss weight
        self.T = 1000              # Original diffusion timesteps
        
        # Allow overrides from a dict
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def to_dict(self):
        """Convert config to a serializable dict."""
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
    Compute prediction entropy from classifier logits.

    Args:
        logits: [B, C] logits
        
    Returns:
        entropy: [B] per-sample entropy
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy


class FastADGenerator:
    """
    Fast-AD generator: DDIM sampling with CADR guidance.
    """
    def __init__(self, diffusion_model, teacher_model, config: FastADConfig, 
                 device='cuda', image_size: Tuple[int, int] = (32, 32), bn_loss_fn=None):
        """
        Args:
            diffusion_model: pretrained diffusion backbone (UNet)
            teacher_model: pretrained teacher classifier
            config: FastADConfig
            device: device
            image_size: (H, W)
            bn_loss_fn: optional BN loss function for guidance
        """
        self.diffusion = diffusion_model
        self.teacher = teacher_model
        self.cfg = config
        self.device = device
        self.image_size = image_size
        self.bn_loss_fn = bn_loss_fn
        self.teacher.eval()  # Teacher is fixed

        # DDIM noise schedule (alphas_cumprod)
        if hasattr(self.diffusion, 'alphas_cumprod'):
            self.alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        else:
            # Fallback schedule if the backbone doesn't provide one
            self.alphas_cumprod = self._linear_beta_schedule(self.cfg.T).to(device)
        
    def _linear_beta_schedule(self, timesteps: int, beta_start: float = 0.0001, 
                              beta_end: float = 0.02) -> torch.Tensor:
        """Linear beta schedule used to derive `alphas_cumprod`."""
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod

    def get_alpha_t(self, t: int) -> torch.Tensor:
        """Return alpha_cumprod at timestep t."""
        if isinstance(t, torch.Tensor):
            t = t.item() if t.numel() == 1 else t
        if isinstance(t, (list, tuple)):
            # Batch lookup
            indices = torch.tensor(t, device=self.device)
            return self.alphas_cumprod[indices].view(-1, 1, 1, 1)
        return self.alphas_cumprod[t].view(-1, 1, 1, 1)

    def cadr_guidance(self, x_t: torch.Tensor, t: int, noise_pred: torch.Tensor, 
                     target_labels: torch.Tensor, bn_loss_fn=None) -> torch.Tensor:
        """
        Confidence-Aware Dynamic Rectification (CADR).

        Implements Eq. 1, 2, 4, 5, 6 and returns the term used in Eq. 8.
        
        Args:
            x_t: noisy image at timestep t [B, C, H, W]
            t: current timestep
            noise_pred: predicted noise ε_θ(x_t, t, c) [B, C, H, W]
            target_labels: target class labels [B]
            bn_loss_fn: optional BN loss function
            
        Returns:
            rectified_grads: g_rect = λ(t) * g_norm * ||x_t|| * η
                             used by Eq. 8: ε̃_t = ε_θ - √(1-α_t) * g_rect
        """
        # Must enable gradients to compute guidance
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)
            
            # --- Step 1: State sensing (Tweedie's formula, Eq. 4) ---
            # x̂₀|t = (x_t - √(1-α_t) * ε_θ) / √α_t
            alpha_t = self.get_alpha_t(t)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            # Eq. 4: Tweedie's formula
            x0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            # Clamp to valid range
            x0_hat = torch.clamp(x0_hat, -1, 1)
            
            # --- Step 2: teacher entropy (state estimate) ---
            # H = Entropy(T(x̂₀|t))
            teacher_logits = self.teacher(x0_hat)
            entropy = compute_entropy(teacher_logits)
            
            # --- Step 3: gating decision (Eq. 5) ---
            # λ(t) = λ_max / (1 + exp(-k * (H - τ_ent)))
            # High entropy => strong guidance; low entropy => weak guidance.
            gate_val = self.cfg.lambda_max / (1 + torch.exp(-self.cfg.k_sigmoid * (entropy - self.cfg.tau_ent)))
            gate_val = gate_val.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            
            # --- Step 4: guidance loss gradient (Eq. 1, 2) ---
            # LCE(x, y) = CrossEntropy(T(x), y) (Eq. 2)
            loss_ce = F.cross_entropy(teacher_logits, target_labels)
            
            # LBN(x) = Σ_l ||μ_l(x) - μ^T_l||²_2 + ||σ²_l(x) - σ²^T_l||²_2 (Eq. 1)
            # BN loss (optional)
            if bn_loss_fn is not None:
                # bn_loss_fn is a BNLoss module
                loss_bn = bn_loss_fn(x0_hat)
            else:
                loss_bn = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Lguide = LBN + γLCE
            total_loss = loss_bn + self.cfg.gamma * loss_ce
            
            # grad = d(Lguide)/d(x_t)
            grads = torch.autograd.grad(total_loss, x_t, create_graph=False)[0]
            
            # --- Step 5: adaptive normalization (Eq. 6) ---
            # g = ∇_{x_t} Lguide(x̂₀|t) = ∇(LBN + γLCE)
            # g_norm = g / (||g||₂ + ξ)
            grad_norm = torch.norm(grads.view(grads.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
            normalized_grads = grads / (grad_norm + self.cfg.xi)
            
            # signal_strength = ||x_t||₂
            signal_strength = torch.norm(x_t.view(x_t.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
            
            # Eq. 6: g_rect = λ(t) * g_norm * ||x_t||₂ * η
            rectified_grads = gate_val * normalized_grads * signal_strength * self.cfg.eta
            
            return rectified_grads

    @torch.no_grad()
    def sample(self, batch_size: int, target_labels: torch.Tensor, 
               prompts_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        DDIM accelerated sampling loop (K steps).
        
        Args:
            batch_size: batch size
            target_labels: target labels [B]
            prompts_embeddings: optional prompt embeddings (conditioning)
            
        Returns:
            x: synthetic images [B, C, H, W]
        """
        # Initialize Gaussian noise
        img_shape = (batch_size, 3, self.image_size[0], self.image_size[1])
        x = torch.randn(img_shape, device=self.device)
        
        # Timesteps (high -> low)
        step_size = self.cfg.T // self.cfg.ddim_steps
        time_steps = list(range(0, self.cfg.T, step_size))
        time_steps = reversed(time_steps)  # [999, 979, ..., 0]
        
        for i, t in enumerate(time_steps):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # 1) Predict noise ε_θ
            if hasattr(self.diffusion, '__call__'):
                # Expected forward: (x, t, context=None)
                if prompts_embeddings is not None:
                    noise_pred = self.diffusion(x, t_tensor, context=prompts_embeddings)
                else:
                    noise_pred = self.diffusion(x, t_tensor)
            else:
                # Fallback
                noise_pred = self.diffusion(x, t_tensor)
            
            # 2) CADR term (core)
            cadr_grad = self.cadr_guidance(x, t, noise_pred, target_labels, self.bn_loss_fn)
            
            # 3) Rectified noise (Eq. 8)
            # ε̃_t = ε_θ(x_t, t, c) - √(1-α_t) * [λ_max/(1+e^(-k(H-τ))) * ∇(LBN+γLCE)/(||∇||+ξ) * ||x_t|| * η]
            alpha_t = self.get_alpha_t(t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            rectified_noise = noise_pred - sqrt_one_minus_alpha_t * cadr_grad
            
            # 4) DDIM update (Eq. 7)
            prev_t = max(0, t - step_size)
            alpha_prev = self.get_alpha_t(prev_t) if prev_t >= 0 else torch.tensor(1.0, device=self.device).view(-1, 1, 1, 1)
            
            # Predict x0 using rectified noise
            # x̂₀ = (x_t - √(1-α_t) * ε̃_t) / √α_t
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * rectified_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Eq. 7
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * rectified_noise
            
        return x

