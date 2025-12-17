# Fast-AD: Fast Adversarial Distillation via Confidence-Aware Dynamic Rectification

Fast-AD 是一个基于扩散模型的数据无知识蒸馏 (Data-Free Knowledge Distillation) 框架，通过置信度感知动态修正 (CADR) 机制和 DDIM 加速采样，实现高效的数据合成和模型蒸馏。

## 📋 项目结构

```
fast-AD/
├── configs/                    # 配置文件目录
│   └── cifar100_config.yaml    # CIFAR-100 超参数配置
├── data/                       # 数据相关
│   └── __init__.py
├── models/                     # 模型定义
│   ├── __init__.py
│   ├── unet.py                 # 扩散模型骨干 (UNet)
│   ├── diffusion_fast_ad.py    # 【核心】FastADGenerator, CADR 引导, DDIM 采样
│   └── resnet.py               # Teacher 和 Student 网络结构
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── buffer.py               # 【核心】ReplayBuffer (FIFO 队列)
│   ├── losses.py               # BN Loss, KD Loss, CrossEntropy Loss
│   ├── logger.py               # 训练日志与可视化工具
│   └── metrics.py              # FID, Accuracy 等评估指标
├── scripts/                    # 运行脚本
│   └── run_distill.sh          # 启动训练的 Shell 脚本
├── main.py                     # 主入口，负责解析参数和初始化
├── trainer.py                  # 【核心】训练循环 (Data Synthesis -> Buffer -> Student Update)
├── requirements.txt            # 项目依赖
└── README.md                   # 项目说明
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备预训练模型

在使用 Fast-AD 之前，你需要准备：

- **Teacher 模型**: 预训练的分类器 (如 ResNet-34)
- **Diffusion 模型**: 预训练的扩散模型 (UNet)

将模型权重保存为 `.pth` 文件，并在配置文件中指定路径：

```yaml
distillation:
  teacher_checkpoint: "path/to/teacher.pth"
  diffusion_checkpoint: "path/to/diffusion.pth"
```

### 3. 配置参数

编辑 `configs/cifar100_config.yaml` 以适配你的任务：

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

### 4. 运行训练

```bash
# 使用默认配置
python main.py

# 指定配置文件
python main.py --config configs/cifar100_config.yaml

# 指定其他参数
python main.py --config configs/cifar100_config.yaml --epochs 200 --device cuda --log-dir ./logs
```

或使用提供的脚本：

```bash
bash scripts/run_distill.sh
```

### 5. 评估模型

训练完成后，使用评估脚本评估模型性能：

```bash
python evaluate.py --checkpoint logs/YYYYMMDD_HHMMSS/checkpoint_epoch_200.pth --config configs/cifar100_config.yaml
```

评估输出示例：
```
Top-1 Accuracy: 76.65%
Top-5 Accuracy: 88.32%
```

## 🔬 核心算法

### CADR (Confidence-Aware Dynamic Rectification)

Fast-AD 的核心创新是 **CADR 引导机制**，它包含以下步骤：

1. **State Sensing (Tweedie's Formula)**: 从噪声图像 $x_t$ 估计干净图像 $x_0$ (公式 4)
2. **Gating Decision**: 在估计的 $x_0$ 上计算 Teacher 的熵，动态调整引导强度 (公式 5)
3. **Adaptive Normalization**: 归一化梯度并乘以信号强度，防止梯度爆炸 (公式 6)
4. **Noise Rectification**: 将修正项应用到预测噪声 (公式 8)

详见 `models/diffusion_fast_ad.py` 中的 `cadr_guidance()` 函数。


### DDIM 加速采样

使用 DDIM 将原本 1000 步的采样过程压缩到 50 步，实现 20x 加速。

### 在线蒸馏流程

1. **合成阶段**: 使用 FastADGenerator 生成合成数据
2. **缓冲阶段**: 将生成的数据存入 FIFO ReplayBuffer
3. **训练阶段**: 从 Buffer 采样数据训练 Student 模型

详见 `trainer.py` 中的 `FastADTrainer` 类。

## 📊 主要参数说明

### Fast-AD 算法参数

- `lambda_max`: 最大引导强度 (默认: 1.5)
- `eta`: 梯度缩放因子 (默认: 0.1)
- `tau_ent`: 熵阈值，CIFAR-100 为 0.4，ImageNet 为 0.6
- `ddim_steps`: DDIM 采样步数 (默认: 50)
- `gamma`: CE Loss 的权重 (默认: 1.0)

### 训练参数

- `epochs`: 训练轮数 (默认: 200)
- `buffer_size`: ReplayBuffer 最大容量 (默认: 4096)
- `synthesis_batch_size`: 每个 epoch 合成的图像数量 (默认: 64)
- `train_batch_size`: 训练批次大小 (默认: 64)

## 📝 代码说明

### 核心文件

1. **`models/diffusion_fast_ad.py`**: 
   - `FastADGenerator`: 实现 CADR 引导和 DDIM 采样
   - `FastADConfig`: 超参数配置类

2. **`utils/buffer.py`**: 
   - `ReplayBuffer`: FIFO 队列，管理合成数据

3. **`trainer.py`**: 
   - `FastADTrainer`: 训练循环控制器
   - `train_fast_ad()`: 训练主函数

4. **`utils/losses.py`**: 
   - `compute_bn_loss()`: BN 正则化损失
   - `compute_kd_loss()`: 知识蒸馏损失

## 🔧 扩展与定制

### 使用自定义扩散模型

如果你的扩散模型来自 `diffusers` 库或其他框架，需要适配接口：

```python
# 在 FastADGenerator.__init__() 中
# 确保 diffusion_model 有 alphas_cumprod 属性
# 或实现 get_alpha_t() 方法
```

### 接入 LLM 生成的 Prompts

在 `trainer.py` 的 `train_one_epoch()` 方法中：

```python
# TODO: 这里可以接入 LLM 生成的 Prompts embedding
prompts = get_llm_prompts(syn_targets)
syn_images = self.generator.sample(synthesis_batch_size, syn_targets, prompts)
```

### 评估指标

使用 `utils/metrics.py` 中的函数评估模型性能：

```python
from utils.metrics import evaluate_model, compute_fid

# 评估准确率
metrics = evaluate_model(student, dataloader, device)

# 计算 FID (需要特征提取器)
fid = compute_fid(real_features, fake_features)
```


## ⚠️ 注意事项

1. **预训练模型**: 确保 Teacher 和 Diffusion 模型已正确加载
2. **显存需求**: 根据 `synthesis_batch_size` 和 `train_batch_size` 调整，避免 OOM
3. **扩散模型**: 当前实现使用简化的 UNet，实际使用时建议使用更完整的实现或 `diffusers` 库
4. **BN Loss**: BN Loss 的计算需要注册 forward hook，可能影响性能，可根据需要优化
5. **Loss 波动**: 由于在线训练和 Buffer 的动态更新，Loss 可能会有波动，这是正常的

## 🐛 已知问题

- UNet 实现较为简化，建议在实际使用时替换为更完整的实现
- BN Loss 计算可能较慢，可考虑缓存或优化

