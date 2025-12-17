"""
训练日志与可视化工具
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional
import torch


class Logger:
    """训练日志记录器"""
    def __init__(self, log_dir: str = './logs', exp_name: Optional[str] = None):
        """
        Args:
            log_dir: 日志目录
            exp_name: 实验名称
        """
        if exp_name is None:
            exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.log_dir, 'metrics.json')
        self.metrics_history = []
        
    def log(self, epoch: int, metrics: Dict[str, float]):
        """
        记录训练指标
        
        Args:
            epoch: 当前 epoch
            metrics: 指标字典
        """
        log_entry = {
            'epoch': epoch,
            **metrics
        }
        self.metrics_history.append(log_entry)
        
        # 保存到文件
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # 打印到控制台
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        print(f"Epoch {epoch}: {metrics_str}")
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, filepath: Optional[str] = None):
        """
        保存模型检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前 epoch
            filepath: 保存路径
        """
        if filepath is None:
            filepath = os.path.join(self.log_dir, f'checkpoint_epoch_{epoch}.pth')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, model: torch.nn.Module, 
                       optimizer: Optional[torch.optim.Optimizer] = None):
        """
        加载模型检查点
        
        Args:
            filepath: 检查点路径
            model: 模型
            optimizer: 优化器 (可选)
        """
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint.get('epoch', 0)

