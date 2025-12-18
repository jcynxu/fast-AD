# Fast-AD: Fast Adversarial Distillation via Confidence-Aware Dynamic Rectification

Fast-AD is a **diffusion-based Data-Free Knowledge Distillation (DFKD)** framework. It leverages **Confidence-Aware Dynamic Rectification (CADR)** and **DDIM accelerated sampling** to synthesize high-quality surrogate data and distill a compact student model without access to the original training set.

## 📋 Project Structure

```
fast-AD/
├── configs/                    # Configuration files
│   └── cifar100_config.yaml    # CIFAR-100 hyperparameters
├── data/                       # Data (optional; e.g., downloads/eval)
│   └── __init__.py
├── models/                     # Model definitions
│   ├── __init__.py
│   ├── unet.py                 # Diffusion backbone (UNet; simplified)
│   ├── diffusion_fast_ad.py    # [Core] FastADGenerator, CADR guidance, DDIM sampling
│   └── resnet.py               # Teacher/Student network (ResNet)
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── buffer.py               # [Core] ReplayBuffer (FIFO)
│   ├── losses.py               # BN Loss, KD Loss, etc.
│   ├── logger.py               # Training logger & checkpoints
│   └── metrics.py              # Accuracy / (optional) FID utilities
├── scripts/                    # Scripts
│   └── run_distill.sh          # Training launcher
├── evaluate.py                 # Evaluation script (top-1/top-5 on CIFAR-100)
├── main.py                     # Entry point (arg parsing & setup)
├── trainer.py                  # [Core] Online distillation loop (Synthesis -> Buffer -> Student Update)
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## 🚀 Quickstart

### 1. Setup Environment

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Or (Windows)
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Pretrained Models

Before running Fast-AD, you need:

- **Teacher model**: a pretrained classifier (e.g., ResNet-34)
- **Diffusion model**: a pretrained diffusion backbone (UNet)

Save the weights as `.pth` files and set their paths in the config:

```yaml
distillation:
  teacher_checkpoint: "path/to/teacher.pth"
  diffusion_checkpoint: "path/to/diffusion.pth"
```

### 3. Configure Hyperparameters

Edit `configs/cifar100_config.yaml` to match your task:

```yaml
distillation:
  teacher_arch: "resnet34"
  student_arch: "resnet18"
  num_classes: 100
  epochs: 200
  buffer_size: 4096
  # ...

fast_ad:
  lambda_max: 1.5
  eta: 0.1
  tau_ent: 0.4
  ddim_steps: 50
  # ...
```

### 4. Train

```bash
# Default config
python main.py

# Specify a config
python main.py --config configs/cifar100_config.yaml

# Override some args
python main.py --config configs/cifar100_config.yaml --epochs 200 --device cuda --log-dir ./logs
```

Or use the provided script:

```bash
bash scripts/run_distill.sh
```

### 5. Evaluate

After training, evaluate the student model with:

```bash
python evaluate.py --checkpoint logs/YYYYMMDD_HHMMSS/checkpoint_epoch_200.pth --config configs/cifar100_config.yaml
```

Example output:
```
Top-1 Accuracy: 76.65%
Top-5 Accuracy: 88.32%
```

## 🔬 Core Method

### CADR (Confidence-Aware Dynamic Rectification)

The key innovation in Fast-AD is **CADR guidance**, which includes:

1. **State Sensing (Tweedie’s formula)**: estimate a clean preview \( \hat{x}_{0|t} \) from noisy \(x_t\) (Eq. 4)
2. **Gating Decision**: compute teacher entropy on \( \hat{x}_{0|t} \) to adapt guidance strength (Eq. 5)
3. **Adaptive Normalization**: normalize the gradient and scale by signal strength to avoid explosions (Eq. 6)
4. **Noise Rectification**: apply the rectification term to the predicted noise (Eq. 8)

See `cadr_guidance()` in `models/diffusion_fast_ad.py`.


### DDIM Accelerated Sampling

DDIM reduces the typical 1000-step DDPM sampling to ~50 steps for ~20× speedup.

### Online Distillation Pipeline

1. **Synthesis**: generate synthetic data using `FastADGenerator`
2. **Buffering**: push samples into a FIFO `ReplayBuffer`
3. **Student update**: sample from the buffer and optimize KD loss

See `FastADTrainer` in `trainer.py`.

## 📊 Key Hyperparameters

### Fast-AD Parameters

- `lambda_max`: maximum rectification strength (default: 1.5)
- `eta`: gradient scaling factor (default: 0.1)
- `tau_ent`: entropy threshold (CIFAR-100: 0.4, ImageNet: 0.6)
- `ddim_steps`: number of DDIM steps (default: 50)
- `gamma`: CE loss weight (default: 1.0)

### Training Parameters

- `epochs`: training epochs (default: 200)
- `buffer_size`: replay buffer capacity (default: 4096)
- `synthesis_batch_size`: synthetic images per epoch (default: 64)
- `train_batch_size`: student update batch size (default: 64)

## 📝 Code Map

### Core Files

1. **`models/diffusion_fast_ad.py`**: 
   - `FastADGenerator`: CADR guidance + DDIM sampling
   - `FastADConfig`: hyperparameter container

2. **`utils/buffer.py`**: 
   - `ReplayBuffer`: FIFO buffer for synthetic samples

3. **`trainer.py`**: 
   - `FastADTrainer`: online training loop
   - `train_fast_ad()`: training entry function

4. **`utils/losses.py`**: 
   - `compute_bn_loss()`: BN regularization
   - `compute_kd_loss()`: KD loss (KL divergence)

## 🔧 Extensions & Customization

### Using a Custom Diffusion Backbone

If you use `diffusers` or another implementation, adapt the interface accordingly:

```python
# In FastADGenerator.__init__()
# Ensure diffusion_model provides alphas_cumprod
# or implement get_alpha_t()
```

### Plugging in LLM-Generated Prompts

In `train_one_epoch()` in `trainer.py`:

```python
# TODO: plug LLM-generated prompt embeddings here
prompts = get_llm_prompts(syn_targets)
syn_images = self.generator.sample(synthesis_batch_size, syn_targets, prompts)
```

### Metrics

Use functions in `utils/metrics.py` to evaluate:

```python
from utils.metrics import evaluate_model, compute_fid

# accuracy
metrics = evaluate_model(student, dataloader, device)

# FID (requires a feature extractor)
fid = compute_fid(real_features, fake_features)
```

