"""
Fast-AD entry point: argument parsing and training setup.
"""

import argparse
import torch
import torch.nn as nn
import yaml
import os

from models.resnet import resnet18, resnet34
from models.unet import UNet
from models.diffusion_fast_ad import FastADConfig
from trainer import train_fast_ad


def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_pretrained_model(model: nn.Module, checkpoint_path: str, device: str = 'cuda'):
    """Load a pretrained checkpoint into `model`."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found, using random initialization")
    return model


def create_models(config: dict, device: str = 'cuda'):
    """Create Teacher / Student / Diffusion models from config."""
    distillation_config = config.get('distillation', {})
    num_classes = distillation_config.get('num_classes', 100)
    
    # Teacher
    teacher_arch = distillation_config.get('teacher_arch', 'resnet34')
    if teacher_arch == 'resnet34':
        teacher = resnet34(num_classes=num_classes)
    elif teacher_arch == 'resnet18':
        teacher = resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported teacher architecture: {teacher_arch}")
    
    # Student
    student_arch = distillation_config.get('student_arch', 'resnet18')
    if student_arch == 'resnet18':
        student = resnet18(num_classes=num_classes)
    elif student_arch == 'resnet34':
        student = resnet34(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported student architecture: {student_arch}")
    
    # Diffusion (UNet)
    image_size = distillation_config.get('image_size', [32, 32])
    diffusion = UNet(image_channels=3, time_emb_dim=32)
    
    # Optional pretrained weights
    teacher_path = distillation_config.get('teacher_checkpoint', None)
    if teacher_path:
        teacher = load_pretrained_model(teacher, teacher_path, device)
    
    diffusion_path = distillation_config.get('diffusion_checkpoint', None)
    if diffusion_path:
        diffusion = load_pretrained_model(diffusion, diffusion_path, device)
    
    return teacher, student, diffusion


def main():
    parser = argparse.ArgumentParser(description='Fast-AD: Fast Adversarial Distillation')
    parser.add_argument('--config', type=str, default='configs/cifar100_config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory to save logs and checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Config
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        print(f"Warning: Config file {args.config} not found, using default config")
        config = {
            'distillation': {
                'teacher_arch': 'resnet34',
                'student_arch': 'resnet18',
                'num_classes': 100,
                'image_size': [32, 32],
                'epochs': 200,
                'buffer_size': 4096,
                'lr': 0.1,
                'milestones': [100, 150],
                'synthesis_batch_size': 64,
                'train_batch_size': 64,
                'train_iterations': 100
            },
            'fast_ad': {
                'lambda_max': 1.5,
                'eta': 0.1,
                'tau_ent': 0.4,
                'k_sigmoid': 10.0,
                'ddim_steps': 50,
                'xi': 1e-6,
                'gamma': 1.0,
                'T': 1000
            }
        }
    
    # Models
    print("Creating models...")
    teacher, student, diffusion = create_models(config, device)
    
    # Optionally resume student weights
    if args.resume:
        student = load_pretrained_model(student, args.resume, device)
    
    # Training params
    distillation_config = config.get('distillation', {})
    epochs = args.epochs if args.epochs is not None else distillation_config.get('epochs', 200)
    num_classes = distillation_config.get('num_classes', 100)
    image_size = tuple(distillation_config.get('image_size', [32, 32]))
    
    # Train
    print("Starting training...")
    trainer = train_fast_ad(
        teacher=teacher,
        student=student,
        diffusion_model=diffusion,
        config_path=args.config if os.path.exists(args.config) else None,
        epochs=epochs,
        device=device,
        num_classes=num_classes,
        image_size=image_size,
        log_dir=args.log_dir
    )
    
    print("Training finished!")


if __name__ == '__main__':
    main()

