"""
Loss functions: BN regularization and KD (KL divergence).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


def compute_bn_loss(teacher_model: nn.Module, input_images: torch.Tensor, 
                   enable_grad: bool = True) -> torch.Tensor:
    """
    BN regularization loss (Eq. 1):
    LBN(x) = Σ_l ||μ_l(x) - μ^T_l||²_2 + ||σ²_l(x) - σ²^T_l||²_2
    
    Args:
        teacher_model: teacher network
        input_images: input images [B, C, H, W]
        enable_grad: keep gradients for guidance (True in CADR)
        
    Returns:
        bn_loss: scalar tensor
    """
    bn_loss = 0.0
    count = 0
    
    # Register forward hooks to capture per-BN activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output
        return hook
    
    hooks = []
    for name, layer in teacher_model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            hook = layer.register_forward_hook(get_activation(name))
            hooks.append(hook)
    
    # Forward to populate activations (keep graph if enable_grad=True)
    if enable_grad:
        _ = teacher_model(input_images)
    else:
        with torch.no_grad():
            _ = teacher_model(input_images)
    
    # Compute BN loss across layers
    for name, layer in teacher_model.named_modules():
        if isinstance(layer, nn.BatchNorm2d) and name in activations:
            activation = activations[name]
            
            # Current batch statistics
            if activation.dim() == 4:  # [B, C, H, W]
                current_mean = activation.mean([0, 2, 3])  # [C]
                current_var = activation.var([0, 2, 3], unbiased=False)  # [C]
            else:
                continue
            
            # Teacher running stats (constants)
            target_mean = layer.running_mean.detach()
            target_var = layer.running_var.detach()
            
            # Eq. 1: squared L2 distance
            mean_loss = torch.sum((current_mean - target_mean) ** 2)
            var_loss = torch.sum((current_var - target_var) ** 2)
            bn_loss += mean_loss + var_loss
            count += 1
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    if count > 0:
        bn_loss = bn_loss / count
    
    return bn_loss


class BNLoss(nn.Module):
    """
    BN loss module wrapper (keeps gradients for guidance).
    """
    def __init__(self, teacher_model: nn.Module):
        super().__init__()
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        
    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        Compute BN loss (with gradients enabled).
        """
        # CADR needs gradients w.r.t. inputs
        return compute_bn_loss(self.teacher_model, input_images, enable_grad=True)


def compute_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                   temperature: float = 4.0) -> torch.Tensor:
    """
    Knowledge distillation loss (Eq. 9):
    LKD = τ_kd² · KL(softmax(S(x)/τ_kd) || softmax(T(x)/τ_kd))

    Where:
    - S(x): student logits
    - T(x): teacher logits
    - τ_kd: distillation temperature
    - KL: Kullback-Leibler divergence

    This encourages the student to match the teacher's *soft* distribution (dark knowledge),
    not just hard labels.

    Args:
        student_logits: student logits [B, C]
        teacher_logits: teacher logits [B, C]
        temperature: distillation temperature τ_kd (default: 4.0)
        
    Returns:
        kd_loss: KL divergence loss (scaled by τ_kd²)
    """
    # KL(P||Q) = Σ P(x) log(P(x)/Q(x))
    # KLDivLoss expects:
    # - input: log P = log_softmax(S(x)/τ)
    # - target: log Q = log_softmax(T(x)/τ)   (log_target=True)
    criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
    loss = criterion(
        F.log_softmax(student_logits / temperature, dim=1),  # log P
        F.log_softmax(teacher_logits / temperature, dim=1)     # log Q
    ) * (temperature ** 2)  # scale by τ_kd²
    return loss


class KDLoss(nn.Module):
    """
    KD loss module wrapper (Eq. 9).
    """
    def __init__(self, temperature: float = 4.0):
        """
        Args:
            temperature: distillation temperature τ_kd (default: 4.0)
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KD loss (Eq. 9).
        
        Args:
            student_logits: [B, C] student logits S(x)
            teacher_logits: [B, C] teacher logits T(x)
            
        Returns:
            loss: KD loss LKD
        """
        return self.criterion(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.log_softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

