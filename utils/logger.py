"""
Training logger and checkpoint utilities.
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional
import torch


class Logger:
    """Simple experiment logger (JSON metrics + checkpoints)."""
    def __init__(self, log_dir: str = './logs', exp_name: Optional[str] = None):
        """Create a new log directory under `log_dir/exp_name`."""
        if exp_name is None:
            exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.log_dir, 'metrics.json')
        self.metrics_history = []
        
    def log(self, epoch: int, metrics: Dict[str, float]):
        """Append metrics for an epoch and persist to `metrics.json`."""
        log_entry = {
            'epoch': epoch,
            **metrics
        }
        self.metrics_history.append(log_entry)
        
        # Persist to disk
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Print to console
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        print(f"Epoch {epoch}: {metrics_str}")
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, filepath: Optional[str] = None):
        """Save a model checkpoint (model + optimizer + epoch)."""
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
        """Load a checkpoint into model (and optimizer if provided)."""
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint.get('epoch', 0)

