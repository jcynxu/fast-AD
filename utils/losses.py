"""
损失函数实现：BN Loss, CE Loss, KD Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


def compute_bn_loss(teacher_model: nn.Module, input_images: torch.Tensor, 
                   enable_grad: bool = True) -> torch.Tensor:
    """
    计算 BN 正则化 Loss (论文公式 1)
    LBN(x) = Σ_l ||μ_l(x) - μ^T_l||²_2 + ||σ²_l(x) - σ²^T_l||²_2
    
    Args:
        teacher_model: Teacher 模型
        input_images: 输入图像 [B, C, H, W]
        enable_grad: 是否启用梯度计算 (用于 CADR 引导时需要 True)
        
    Returns:
        bn_loss: BN 损失值
    """
    bn_loss = 0.0
    count = 0
    
    # 注册 forward hook 来获取每一层的统计量
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
    
    # Forward pass 获取激活值
    # 注意：如果 enable_grad=True，需要保持梯度计算
    if enable_grad:
        _ = teacher_model(input_images)
    else:
        with torch.no_grad():
            _ = teacher_model(input_images)
    
    # 计算 BN Loss
    for name, layer in teacher_model.named_modules():
        if isinstance(layer, nn.BatchNorm2d) and name in activations:
            activation = activations[name]
            
            # 获取当前输入的统计量
            if activation.dim() == 4:  # [B, C, H, W]
                # 计算当前 batch 的均值和方差
                current_mean = activation.mean([0, 2, 3])  # [C]
                current_var = activation.var([0, 2, 3], unbiased=False)  # [C]
            else:
                continue
            
            # 获取 Teacher 存储的统计量 (running_mean 和 running_var)
            # 这些是固定的，不需要梯度
            target_mean = layer.running_mean.detach()
            target_var = layer.running_var.detach()
            
            # L2 Loss (论文公式 1)
            # ||μ_l(x) - μ^T_l||²_2 + ||σ²_l(x) - σ²^T_l||²_2
            mean_loss = torch.sum((current_mean - target_mean) ** 2)
            var_loss = torch.sum((current_var - target_var) ** 2)
            bn_loss += mean_loss + var_loss
            count += 1
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    if count > 0:
        bn_loss = bn_loss / count
    
    return bn_loss


class BNLoss(nn.Module):
    """
    BN Loss 的 Module 封装，支持梯度计算
    """
    def __init__(self, teacher_model: nn.Module):
        super().__init__()
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        
    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        计算 BN Loss (支持梯度计算)
        
        Args:
            input_images: [B, C, H, W] 输入图像
            
        Returns:
            loss: BN 损失值
        """
        # 在 CADR 引导中需要计算梯度，所以 enable_grad=True
        return compute_bn_loss(self.teacher_model, input_images, enable_grad=True)


def compute_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                   temperature: float = 4.0) -> torch.Tensor:
    """
    计算知识蒸馏 Loss (论文公式 9)
    
    论文公式 9:
    LKD = τ_kd² · KL(softmax(S(x)/τ_kd) || softmax(T(x)/τ_kd))
    
    其中：
    - S(x): Student 模型的 logits
    - T(x): Teacher 模型的 logits
    - τ_kd: 蒸馏温度 (distillation temperature)
    - KL: Kullback-Leibler 散度
    
    这个损失函数强制 Student 学习 Teacher 的"暗知识"（soft probability distribution），
    而不仅仅是硬标签。
    
    Args:
        student_logits: Student 模型输出 [B, C]
        teacher_logits: Teacher 模型输出 [B, C]
        temperature: 蒸馏温度 τ_kd (默认 4.0)
        
    Returns:
        kd_loss: KL 散度损失 (已乘以 τ_kd²)
    """
    # KL(P||Q) = Σ P(x) log(P(x)/Q(x))
    # 使用 KLDivLoss，其中：
    # - input: log P = log_softmax(S(x)/τ)
    # - target: log Q = log_softmax(T(x)/τ) (log_target=True)
    criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
    loss = criterion(
        F.log_softmax(student_logits / temperature, dim=1),  # log P
        F.log_softmax(teacher_logits / temperature, dim=1)     # log Q
    ) * (temperature ** 2)  # 乘以 τ_kd²
    return loss


class KDLoss(nn.Module):
    """
    知识蒸馏 Loss 的 Module 封装 (论文公式 9)
    
    实现 LKD = τ_kd² · KL(softmax(S(x)/τ_kd) || softmax(T(x)/τ_kd))
    """
    def __init__(self, temperature: float = 4.0):
        """
        Args:
            temperature: 蒸馏温度 τ_kd (默认 4.0)
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        计算 KD Loss (公式 9)
        
        Args:
            student_logits: [B, C] Student 输出 S(x)
            teacher_logits: [B, C] Teacher 输出 T(x)
            
        Returns:
            loss: KD 损失值 LKD
        """
        return self.criterion(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.log_softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

