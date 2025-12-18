"""
Online distillation pipeline (paper Sec. 3.5).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import yaml

from models.diffusion_fast_ad import FastADGenerator, FastADConfig
from utils.buffer import ReplayBuffer
from utils.losses import BNLoss, KDLoss
from utils.logger import Logger
from utils.metrics import evaluate_model


class FastADTrainer:
    """
    Fast-AD trainer: alternates between synthesis and student updates.
    """
    def __init__(self, teacher: nn.Module, student: nn.Module, diffusion_model: nn.Module,
                 config: FastADConfig, device: str = 'cuda', num_classes: int = 100,
                 image_size: tuple = (32, 32), logger: Optional[Logger] = None):
        """Initialize trainer with teacher/student/diffusion and config."""
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.diffusion = diffusion_model.to(device)
        self.config = config
        self.device = device
        self.num_classes = num_classes
        self.image_size = image_size
        self.logger = logger
        
        # Modes: teacher/diffusion fixed, student trainable
        self.teacher.eval()
        self.diffusion.eval()
        self.student.train()
        
        # BN regularization loss used by CADR guidance
        self.bn_loss_fn = BNLoss(self.teacher)
        
        # Generator + replay buffer
        self.generator = FastADGenerator(
            self.diffusion, self.teacher, config, device, image_size, self.bn_loss_fn
        )
        self.buffer = ReplayBuffer(max_size=config.buffer_size if hasattr(config, 'buffer_size') else 4096)
        
        # KD loss (Eq. 9): τ_kd² * KL(softmax(S/τ_kd) || softmax(T/τ_kd))
        kd_temp = getattr(config, 'kd_temperature', 4.0)  # default: 4.0
        self.kd_loss_fn = KDLoss(temperature=kd_temp)
        
    def train_one_epoch(self, optimizer: optim.Optimizer, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                       synthesis_batch_size: int = 64, train_batch_size: int = 64,
                       train_iterations: int = 100, epoch: int = 0):
        """Train one epoch (synthesis -> buffer -> student update)."""
        self.student.train()
        
        # --- Phase 1: Synthesis ---
        syn_targets = torch.randint(0, self.num_classes, (synthesis_batch_size,), device=self.device)
        
        # TODO: plug in LLM-driven prompt embeddings for better diversity
        # prompts = get_llm_prompts(syn_targets)
        prompts_embeddings = None
        
        print(f"Epoch {epoch+1}: Synthesizing {synthesis_batch_size} images...")
        with torch.no_grad():
            # Generate rectified images x_syn (Eq. 7)
            syn_images = self.generator.sample(synthesis_batch_size, syn_targets, prompts_embeddings)
        
        # Push into FIFO buffer
        self.buffer.push(syn_images, syn_targets)
        print(f"Buffer size: {len(self.buffer)}/{self.buffer.max_size}")
        
        # --- Phase 2: Student update ---
        if len(self.buffer) < train_batch_size:
            print("Buffer not full enough, skipping training phase")
            return {'loss': 0.0, 'buffer_size': len(self.buffer)}
        
        total_loss = 0.0
        for iteration in range(train_iterations):
            # Sample x ~ B
            inputs, targets = self.buffer.sample(train_batch_size)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward: get logits
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            student_logits = self.student(inputs)
            
            # KD loss (Eq. 9)
            loss = self.kd_loss_fn(student_logits, teacher_logits)
            
            # Backprop: update student
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration+1}/{train_iterations}, Loss: {loss.item():.4f}")
        
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = total_loss / train_iterations
        metrics = {
            'loss': avg_loss,
            'buffer_size': len(self.buffer),
            'lr': optimizer.param_groups[0]['lr']
        }
        
        if self.logger:
            self.logger.log(epoch + 1, metrics)
        else:
            print(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}, Buffer Size: {len(self.buffer)}")
        
        return metrics
    
    def train(self, epochs: int, optimizer: optim.Optimizer,
              scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
              synthesis_batch_size: int = 64, train_batch_size: int = 64,
              train_iterations: int = 100, save_interval: int = 10):
        """Full training loop."""
        print("Starting Fast-AD Distillation...")
        print(f"Teacher: {type(self.teacher).__name__}")
        print(f"Student: {type(self.student).__name__}")
        print(f"Epochs: {epochs}, Buffer Size: {self.buffer.max_size}")
        
        for epoch in range(epochs):
            metrics = self.train_one_epoch(
                optimizer, scheduler, synthesis_batch_size,
                train_batch_size, train_iterations, epoch
            )
            
            # Save checkpoints
            if (epoch + 1) % save_interval == 0 and self.logger:
                self.logger.save_checkpoint(self.student, optimizer, epoch + 1)
        
        print("Training completed!")
    
    def evaluate(self, dataloader: Optional[torch.utils.data.DataLoader] = None) -> dict:
        """Evaluate the student model."""
        if dataloader is None:
            # If no dataloader is provided, evaluate on a buffer sample
            if len(self.buffer) == 0:
                return {'error': 'No data available for evaluation'}
            
            # Sample from buffer
            inputs, targets = self.buffer.sample(min(100, len(self.buffer)))
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.student.eval()
            with torch.no_grad():
                logits = self.student(inputs)
                _, predicted = torch.max(logits.data, 1)
                correct = (predicted == targets).sum().item()
                total = targets.size(0)
                accuracy = 100.0 * correct / total
            
            return {'accuracy': accuracy, 'correct': correct, 'total': total}
        else:
            from utils.metrics import evaluate_model
            return evaluate_model(self.student, dataloader, self.device)


def train_fast_ad(teacher: nn.Module, student: nn.Module, diffusion_model: nn.Module,
                  config_path: Optional[str] = None, epochs: int = 200,
                  device: str = 'cuda', num_classes: int = 100, image_size: tuple = (32, 32),
                  log_dir: str = './logs'):
    """Top-level training helper used by `main.py`."""
    # Load config
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        fast_ad_config = config_dict.get('fast_ad', {})
        distillation_config = config_dict.get('distillation', {})
        config = FastADConfig(fast_ad_config)
        config.buffer_size = distillation_config.get('buffer_size', 4096)
        config.kd_temperature = distillation_config.get('kd_temperature', 4.0)
    else:
        config = FastADConfig()
        config.buffer_size = 4096
    
    # Logger
    logger = Logger(log_dir=log_dir)
    
    # Trainer
    trainer = FastADTrainer(
        teacher, student, diffusion_model, config, device,
        num_classes, image_size, logger
    )
    
    # Optimizer + scheduler
    optimizer = optim.SGD(
        student.parameters(),
        lr=distillation_config.get('lr', 0.1) if config_path else 0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    milestones = distillation_config.get('milestones', [100, 150]) if config_path else [100, 150]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Training params
    synthesis_batch_size = distillation_config.get('synthesis_batch_size', 64) if config_path else 64
    train_batch_size = distillation_config.get('train_batch_size', 64) if config_path else 64
    train_iterations = distillation_config.get('train_iterations', 100) if config_path else 100
    
    # Train
    trainer.train(
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        synthesis_batch_size=synthesis_batch_size,
        train_batch_size=train_batch_size,
        train_iterations=train_iterations
    )
    
    return trainer

