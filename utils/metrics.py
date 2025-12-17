"""
评估指标：FID, Accuracy 等
"""

import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: tuple = (1,)) -> list:
    """
    计算分类准确率
    
    Args:
        logits: 模型输出 [B, C]
        targets: 真实标签 [B]
        topk: 计算 top-k 准确率
        
    Returns:
        accuracies: top-k 准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def compute_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """
    计算 FID (Fréchet Inception Distance)
    简化版本，实际使用时建议使用 pytorch-fid 库
    
    Args:
        real_features: 真实图像的特征 [N, D]
        fake_features: 生成图像的特征 [N, D]
        
    Returns:
        fid: FID 分数
    """
    # 计算均值和协方差
    mu1 = real_features.mean(dim=0)
    mu2 = fake_features.mean(dim=0)
    
    sigma1 = torch.cov(real_features.T)
    sigma2 = torch.cov(fake_features.T)
    
    # 计算 FID
    diff = mu1 - mu2
    covmean = torch.sqrt(sigma1 @ sigma2)
    
    fid = torch.norm(diff) ** 2 + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.item()


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                   device: str = 'cuda') -> dict:
    """
    评估模型性能
    
    Args:
        model: 待评估模型
        dataloader: 数据加载器
        device: 计算设备
        
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    correct = 0
    total = 0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    top1_acc = compute_accuracy(all_logits, all_targets, topk=(1,))[0]
    top5_acc = compute_accuracy(all_logits, all_targets, topk=(5,))[0] if all_logits.size(1) >= 5 else top1_acc
    
    return {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'correct': correct,
        'total': total
    }

