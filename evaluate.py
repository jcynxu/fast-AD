"""
Evaluate a trained student model checkpoint.

Default: CIFAR-100 test set, report Top-1 / Top-5 accuracy.
"""

import argparse
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from torchvision import datasets, transforms
except Exception as e:  # pragma: no cover
    raise RuntimeError("torchvision is required for evaluation. Please install torchvision.") from e

from models.resnet import resnet18, resnet34
from utils.metrics import evaluate_model


def _pick_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def load_config_num_classes(config_path: str, fallback: int = 100) -> int:
    """Load num_classes from YAML config (if available)."""
    if not config_path or not os.path.exists(config_path) or yaml is None:
        return fallback
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return int(cfg.get("distillation", {}).get("num_classes", fallback))


def build_student(arch: str, num_classes: int) -> torch.nn.Module:
    """Build a student model by name."""
    arch = (arch or "resnet18").lower()
    if arch == "resnet18":
        return resnet18(num_classes=num_classes)
    if arch == "resnet34":
        return resnet34(num_classes=num_classes)
    raise ValueError(f"Unsupported student architecture: {arch}")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    """Load checkpoint weights into model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)


def build_cifar100_test_loader(data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    """Create CIFAR-100 test DataLoader."""
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    ds = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Fast-AD student checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--config", type=str, default="configs/cifar100_config.yaml", help="Path to YAML config")
    parser.add_argument("--student-arch", type=str, default="resnet18", help="Student architecture (resnet18/resnet34)")
    parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset (currently: cifar100)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    args = parser.parse_args()

    device = _pick_device(args.device)
    num_classes = load_config_num_classes(args.config, fallback=100)

    if args.dataset.lower() != "cifar100":
        raise ValueError("Only cifar100 is supported by this evaluation script right now.")

    model = build_student(args.student_arch, num_classes=num_classes).to(device)
    load_checkpoint(model, args.checkpoint, device=device)

    loader = build_cifar100_test_loader(args.data_dir, args.batch_size, args.num_workers)

    print(f"Device: {device}")
    print(f"Dataset: CIFAR-100 (test), size={len(loader.dataset)}")
    print(f"Student: {args.student_arch}, num_classes={num_classes}")
    print(f"Checkpoint: {args.checkpoint}")

    metrics: Dict[str, float] = evaluate_model(model, loader, device=device)
    print(f"Top-1 Accuracy: {metrics['top1_acc']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['top5_acc']:.2f}%")
    print(f"Correct: {metrics['correct']} / {metrics['total']}")


if __name__ == "__main__":
    main()