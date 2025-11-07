# Front Camera Video Prediction for Bench2Drive

这是一套完整的视频预测训练流程，用于预测自动驾驶前置摄像头的未来帧图像。

## 📋 项目概述

**任务目标**: 根据历史4帧前置摄像头图像，预测未来4帧图像

**技术方案**:
- 使用扩散模型(Diffusion Model)进行视频生成
- 采用DiT (Diffusion Transformer)架构
- 通过VAE进行图像编码/解码
- 支持多GPU分布式训练

**数据配置**:
- 输入: 4个历史帧 (past_frames=4)
- 输出: 4个未来帧 (future_frames=4)
- 采样间隔: 5帧 (0.5秒间隔)
- 图像尺寸: 256x448 (从原始928x1600下采样)

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision diffusers einops pyyaml tqdm pillow
pip install tensorboard  # 用于训练可视化
```

### 2. 数据准备

假设你有原始的Bench2Drive数据集，结构如下：

```
data/bench2drive/
├── Town01/
│   ├── rgb_front/
│   │   ├── frame_000000.png
│   │   ├── frame_000001.png
│   │   └── ...
│   └── annotations.json (可选)
├── Town02/
└── ...
```

**步骤1: 生成训练/验证集标注文件**

```bash
python prepare_video_data.py \
    --data_root data/bench2drive \
    --output_dir data/infos \
    --val_ratio 0.2 \
    --verify
```

这会生成:
- `data/infos/b2d_infos_train.pkl` - 训练集
- `data/infos/b2d_infos_val.pkl` - 验证集

**注意**: 如果你已经有现成的 `b2d_infos_train.pkl` 和 `b2d_infos_val.pkl`，可以跳过这一步。

### 3. 配置文件

编辑 `configs/config_video_prediction.yaml`:

```yaml
Global:
  save_path: 'checkpoints/video_prediction'  # 模型保存路径
  tb_path: 'logs/video_prediction'  # TensorBoard日志路径
  
Dataset:
  train:
    data_root: 'data/bench2drive'  # 修改为你的数据路径
    ann_file: 'data/infos/b2d_infos_train.pkl'
  eval:
    data_root: 'data/bench2drive'
    ann_file: 'data/infos/b2d_infos_val.pkl'

Model:
  vae_model_path: 'stabilityai/sd-vae-ft-mse'  # 会自动从HuggingFace下载
```

### 4. 训练模型

**单GPU训练**:
```bash
python train_video_pred.py --config configs/config_video_prediction.yaml
```

**多GPU训练 (推荐)**:
```bash
# 使用4个GPU
torchrun --nproc_per_node=4 train_video_pred.py \
    --config configs/config_video_prediction.yaml
```

**从检查点恢复训练**:
```bash
python train_video_pred.py \
    --config configs/config_video_prediction.yaml \
    --resume checkpoints/video_prediction/step_50000.pt
```

### 5. 监控训练

启动TensorBoard查看训练进度：

```bash
tensorboard --logdir logs/video_prediction
```

在浏览器中打开 `http://localhost:6006`

### 6. 评估模型

```bash
python eval_video_pred.py \
    --config configs/config_video_prediction.yaml \
    --ckpt checkpoints/video_prediction/step_50000.pt \
    --num_samples 100 \
    --save_visualizations \
    --num_inference_steps 50
```

评估结果会保存在 `checkpoints/video_prediction/evaluation/`:
- `metrics.txt` - 评估指标 (MSE, PSNR, MAE)
- `sample_*.png` - 可视化结果

---

## 📁 文件说明

### 核心代码文件

| 文件 | 说明 |
|------|------|
| `dataset/video_prediction_dataset.py` | 数据集类，负责加载历史帧和未来帧 |
| `model/video_prediction_diffusion.py` | 视频预测扩散模型 |
| `video_prediction_trainer.py` | 训练器，包含训练和评估逻辑 |
| `train_video_pred.py` | 训练入口脚本 |
| `eval_video_pred.py` | 评估入口脚本 |
| `prepare_video_data.py` | 数据预处理脚本 |

### 配置文件

| 文件 | 说明 |
|------|------|
| `configs/config_video_prediction.yaml` | 主配置文件 |

---

## ⚙️ 配置参数详解

### 训练参数

```yaml
Train:
  max_epoch: 100              # 训练轮数
  batch_size: 4               # 每个GPU的batch size
  num_workers: 8              # 数据加载线程数
  lr: 1.0e-4                  # 学习率
  weight_decay: 1.0e-2        # 权重衰减
  gradient_clip_val: 1.0      # 梯度裁剪
  log_every_step: 100         # 每N步记录日志
  save_every_step: 5000       # 每N步保存检查点
  eval_every_epoch: 5         # 每N轮评估一次
  num_inference_steps: 50     # 评估时的去噪步数
```

### 模型参数

```yaml
Model:
  past_frames: 4              # 历史帧数量
  future_frames: 4            # 预测帧数量
  img_size: [256, 448]        # 图像尺寸 (H, W)
  
  dit_config:
    hidden_size: 1152         # 隐藏层维度
    depth: 28                 # Transformer层数
    num_heads: 16             # 注意力头数
```

### 数据集参数

```yaml
Dataset:
  train:
    past_frames: 4            # 输入历史帧数
    future_frames: 4          # 预测未来帧数
    sample_interval: 5        # 帧采样间隔
```

---

## 🔧 高级用法

### 修改预测帧数

如果想预测更多未来帧（例如预测8帧）：

1. 修改配置文件:
```yaml
Model:
  future_frames: 8

Dataset:
  train:
    future_frames: 8
  eval:
    future_frames: 8
```

2. 重新训练模型

### 修改输入历史帧数

如果想使用更多历史信息（例如8帧）：

```yaml
Model:
  past_frames: 8

Dataset:
  train:
    past_frames: 8
  eval:
    past_frames: 8
```

### 调整图像分辨率

```yaml
Model:
  img_size: [512, 896]  # 更高分辨率

Dataset:
  train:
    img_size: [512, 896]
  eval:
    img_size: [512, 896]
```

**注意**: 更高分辨率需要更多显存

### 使用不同的VAE

可以使用其他预训练VAE模型：

```yaml
Model:
  vae_model_path: 'stabilityai/sd-vae-ft-ema'  # 或其他HuggingFace模型
```

---

## 📊 评估指标说明

- **MSE (Mean Squared Error)**: 均方误差，越小越好
- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，越大越好 (通常>20dB为可接受)
- **MAE (Mean Absolute Error)**: 平均绝对误差，越小越好

---

## 🐛 常见问题

### 1. 数据集加载失败

**问题**: `FileNotFoundError: Cannot find image file`

**解决**:
- 检查 `data_root` 路径是否正确
- 运行 `prepare_video_data.py --verify` 验证数据完整性
- 确保图像路径在 pkl 文件中是相对于 `data_root` 的

### 2. 显存不足 (Out of Memory)

**解决方案**:
- 减小 `batch_size` (例如改为2或1)
- 减小图像尺寸 `img_size: [128, 224]`
- 减少模型参数：
  ```yaml
  dit_config:
    hidden_size: 768  # 从1152降低
    depth: 12         # 从28降低
  ```
- 启用梯度检查点（已默认开启）

### 3. 训练速度慢

**优化方案**:
- 增加 `num_workers` (例如16)
- 使用多GPU训练
- 减少 `num_inference_steps` 在评估时（不影响训练）
- 使用mixed precision training (需要修改代码添加)

### 4. VAE下载失败

如果无法从HuggingFace下载VAE模型：

1. 手动下载模型：
```bash
git lfs install
git clone https://huggingface.co/stabilityai/sd-vae-ft-mse
```

2. 修改配置指向本地路径：
```yaml
Model:
  vae_model_path: './sd-vae-ft-mse'
```

### 5. 生成的视频质量差

**可能原因和解决**:
- 训练不足：继续训练更多epochs
- 采样步数太少：增加 `num_inference_steps` 到100
- 数据质量问题：检查数据集是否包含足够的运动场景
- 学习率问题：尝试降低学习率到 `5e-5`

---

## 📈 训练建议

### 推荐训练策略

1. **阶段1**: 小规模验证
   - 使用少量数据（10%）训练5个epoch
   - 验证代码是否正常运行
   - 检查loss是否下降

2. **阶段2**: 完整训练
   - 使用全部数据
   - 训练至少30-50个epoch
   - 定期检查验证集指标

3. **阶段3**: 精调
   - 降低学习率到原来的1/10
   - 继续训练10-20个epoch

### 超参数调优

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| batch_size | 2-8 | 取决于GPU显存 |
| lr | 1e-5 to 5e-4 | 从小开始尝试 |
| num_inference_steps | 50-100 | 更多步数=更好质量但更慢 |
| past_frames | 2-8 | 更多历史信息可能提升预测 |
| future_frames | 1-8 | 预测更远的未来更困难 |

---

## 📝 输出示例

训练完成后，可视化输出格式：

```
[Past Frame 1] [Past Frame 2] [Past Frame 3] [Past Frame 4]
[GT Future 1]  [GT Future 2]  [GT Future 3]  [GT Future 4]
[Pred Future 1][Pred Future 2][Pred Future 3][Pred Future 4]
```

其中:
- 第一行: 输入的历史帧
- 第二行: 真实的未来帧 (Ground Truth)
- 第三行: 模型预测的未来帧

---

## 🎯 下一步改进方向

1. **添加条件控制**: 
   - 可以加入方向盘角度、速度等控制信号
   - 实现可控的视频生成

2. **多视角预测**:
   - 扩展到预测其他摄像头视角
   - 实现360度环视预测

3. **更长时序预测**:
   - 预测更远的未来（例如16帧，1.6秒）
   - 使用递归预测策略

4. **优化训练效率**:
   - 实现mixed precision training (AMP)
   - 使用更高效的采样算法 (DDIM, DPM-Solver)

5. **提升生成质量**:
   - 使用感知损失 (Perceptual Loss)
   - 添加对抗训练 (GAN)
   - 实现更好的时序一致性约束

---

## 📧 联系与支持

如有问题，请检查：
1. 是否按照步骤正确配置数据路径
2. 是否安装了所有依赖库
3. GPU显存是否足够

## 📄 许可证

本项目基于DiffAD项目扩展，遵循原项目的许可证。

