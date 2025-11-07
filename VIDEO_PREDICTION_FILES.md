# 视频预测训练系统 - 文件清单

## 📋 创建的文件列表

本文档列出了为视频预测任务创建的所有新文件及其用途。

---

## 核心代码文件

### 1. 数据集 (`dataset/video_prediction_dataset.py`)
**功能**: 
- 加载Bench2Drive数据集的front相机图像
- 构建时序数据对：历史帧 + 未来帧
- 数据增强和预处理
- 支持训练/验证模式

**关键类**:
- `FrontCameraVideoDataset`: 主数据集类
- `collate_video_batch`: 批处理整合函数

---

### 2. 模型 (`model/video_prediction_diffusion.py`)
**功能**:
- 基于扩散模型的视频预测
- 时序编码器处理历史帧
- DiT (Diffusion Transformer) 生成未来帧
- VAE编码解码图像

**关键类**:
- `VideoPredictionDiffusion`: 完整的视频预测模型
- `TemporalEncoder`: 3D卷积 + 注意力的时序编码器
- `VideoDiT`: 视频专用的DiT模型

**模型架构**:
```
历史帧 [B,4,3,H,W]
    ↓ VAE Encoder
历史潜在 [B,4,C,h,w]
    ↓ TemporalEncoder (Conv3D + Attention)
时序特征 [B,C,h,w]
    ↓ + 随机噪声
去噪过程 (DiT)
    ↓ DDIM采样
未来潜在 [B,4,C,h,w]
    ↓ VAE Decoder
未来帧 [B,4,3,H,W]
```

---

### 3. 训练器 (`video_prediction_trainer.py`)
**功能**:
- 完整的训练循环
- 分布式训练支持 (DDP)
- EMA (指数移动平均) 模型
- 评估和可视化
- 检查点保存/加载
- TensorBoard日志记录

**关键类**:
- `VideoPredictionTrainer`: 主训练器类

**主要方法**:
- `train_loop()`: 主训练循环
- `train_step()`: 单步训练
- `eval_step()`: 单步评估
- `evaluate()`: 完整评估
- `save_checkpoint()`: 保存检查点
- `load_checkpoint()`: 加载检查点

---

## 脚本文件

### 4. 训练脚本 (`train_video_pred.py`)
**用途**: 训练入口脚本

**用法**:
```bash
# 单GPU
python train_video_pred.py --config configs/config_video_prediction.yaml

# 多GPU
torchrun --nproc_per_node=4 train_video_pred.py --config configs/config_video_prediction.yaml

# 恢复训练
python train_video_pred.py --config configs/config_video_prediction.yaml --resume checkpoints/xxx.pt
```

---

### 5. 评估脚本 (`eval_video_pred.py`)
**用途**: 模型评估和可视化

**用法**:
```bash
python eval_video_pred.py \
    --config configs/config_video_prediction.yaml \
    --ckpt checkpoints/video_prediction/step_50000.pt \
    --num_samples 100 \
    --save_visualizations \
    --num_inference_steps 50
```

**输出**:
- 评估指标 (MSE, PSNR, MAE)
- 可视化图像（历史帧 | 真值 | 预测）

---

### 6. 推理演示 (`inference_demo.py`)
**用途**: 使用训练好的模型进行推理

**用法**:
```bash
python inference_demo.py \
    --config configs/config_video_prediction.yaml \
    --ckpt checkpoints/video_prediction/step_50000.pt \
    --images frame1.png frame2.png frame3.png frame4.png \
    --output prediction.png \
    --num_steps 50
```

**功能**:
- 从任意4张图像预测未来帧
- 生成可视化结果
- 支持自定义采样步数

---

### 7. 数据预处理 (`prepare_video_data.py`)
**用途**: 准备训练数据的标注文件

**用法**:
```bash
python prepare_video_data.py \
    --data_root data/bench2drive \
    --output_dir data/infos \
    --val_ratio 0.2 \
    --verify
```

**功能**:
- 扫描数据目录
- 生成训练/验证集划分
- 创建 pkl 格式标注文件
- 验证数据完整性

---

### 8. 测试脚本 (`test_video_prediction.py`)
**用途**: 验证所有组件是否正常工作

**用法**:
```bash
python test_video_prediction.py
```

**测试项目**:
- ✓ 依赖包安装
- ✓ 数据集类导入
- ✓ 模型类导入
- ✓ 训练器导入
- ✓ 配置文件加载
- ✓ 前向传播测试

---

## 配置文件

### 9. 配置文件 (`configs/config_video_prediction.yaml`)
**内容**:
```yaml
Global:
  save_path: checkpoints路径
  tb_path: tensorboard日志路径
  load_from: 检查点路径

Train:
  max_epoch: 100
  batch_size: 4
  lr: 1.0e-4
  ...

Model:
  past_frames: 4
  future_frames: 4
  img_size: [256, 448]
  vae_model_path: 'stabilityai/sd-vae-ft-mse'
  dit_config:
    hidden_size: 1152
    depth: 28
    num_heads: 16

Dataset:
  train/eval:
    data_root: 数据路径
    ann_file: 标注文件路径
    ...
```

---

## 辅助文件

### 10. 依赖文件 (`requirements_video_pred.txt`)
**内容**: 所有需要的Python包
```
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0
einops>=0.7.0
pyyaml>=6.0
tqdm>=4.65.0
pillow>=10.0.0
tensorboard>=2.13.0
...
```

**安装**:
```bash
pip install -r requirements_video_pred.txt
```

---

### 11. 快速启动脚本 (`quick_start.sh`)
**用途**: 一键式设置和启动训练

**用法**:
```bash
chmod +x quick_start.sh
./quick_start.sh
```

**功能**:
- 检查Python版本
- 安装依赖
- 准备数据
- 配置训练
- 启动训练（单/多GPU）

---

## 文档文件

### 12. 使用文档 (`VIDEO_PREDICTION_README.md`)
**内容**:
- 📋 项目概述
- 🚀 快速开始指南
- ⚙️ 配置参数详解
- 🔧 高级用法
- 🐛 常见问题解答
- 📈 训练建议
- 📝 输出示例

### 13. 文件清单 (`VIDEO_PREDICTION_FILES.md`)
**内容**: 本文档，列出所有文件及用途

---

## 📁 完整目录结构

```
DiffAD-main/
├── dataset/
│   └── video_prediction_dataset.py       # 数据集
├── model/
│   └── video_prediction_diffusion.py     # 模型
├── configs/
│   └── config_video_prediction.yaml      # 配置
├── video_prediction_trainer.py           # 训练器
├── train_video_pred.py                   # 训练脚本
├── eval_video_pred.py                    # 评估脚本
├── inference_demo.py                     # 推理演示
├── prepare_video_data.py                 # 数据预处理
├── test_video_prediction.py              # 测试脚本
├── requirements_video_pred.txt           # 依赖
├── quick_start.sh                        # 快速启动
├── VIDEO_PREDICTION_README.md            # 使用文档
└── VIDEO_PREDICTION_FILES.md             # 本文件
```

---

## 🎯 使用流程

### 第一次使用（完整流程）

1. **安装依赖**
   ```bash
   pip install -r requirements_video_pred.txt
   ```

2. **测试环境**
   ```bash
   python test_video_prediction.py
   ```

3. **准备数据**
   ```bash
   python prepare_video_data.py --data_root data/bench2drive --output_dir data/infos
   ```

4. **修改配置**
   ```bash
   # 编辑 configs/config_video_prediction.yaml
   # 设置正确的数据路径
   ```

5. **开始训练**
   ```bash
   # 单GPU
   python train_video_pred.py --config configs/config_video_prediction.yaml
   
   # 或多GPU
   torchrun --nproc_per_node=4 train_video_pred.py --config configs/config_video_prediction.yaml
   ```

6. **监控训练**
   ```bash
   tensorboard --logdir logs/video_prediction
   ```

7. **评估模型**
   ```bash
   python eval_video_pred.py \
       --config configs/config_video_prediction.yaml \
       --ckpt checkpoints/video_prediction/step_50000.pt \
       --num_samples 100 \
       --save_visualizations
   ```

8. **推理测试**
   ```bash
   python inference_demo.py \
       --config configs/config_video_prediction.yaml \
       --ckpt checkpoints/video_prediction/step_50000.pt \
       --images frame1.png frame2.png frame3.png frame4.png \
       --output prediction.png
   ```

---

## 📊 输出文件

训练和评估过程会产生以下输出：

### 训练输出
```
checkpoints/video_prediction/
├── step_5000.pt          # 检查点
├── step_10000.pt
├── ...
└── final.pt

logs/video_prediction/
└── events.out.tfevents.* # TensorBoard日志
```

### 评估输出
```
checkpoints/video_prediction/evaluation/
├── metrics.txt           # 评估指标
├── sample_0000.png       # 可视化
├── sample_0001.png
└── ...
```

---

## 🔄 修改建议

如果您想修改系统行为，主要需要改动这些文件：

| 需求 | 修改文件 | 位置 |
|------|---------|------|
| 改变帧数 | `config_video_prediction.yaml` | Model/Dataset部分 |
| 改变图像大小 | `config_video_prediction.yaml` | Model.img_size |
| 改变模型大小 | `config_video_prediction.yaml` | Model.dit_config |
| 改变学习率 | `config_video_prediction.yaml` | Train.lr |
| 添加新的损失函数 | `video_prediction_diffusion.py` | forward() |
| 修改数据增强 | `video_prediction_dataset.py` | __init__() |
| 改变采样策略 | `video_prediction_diffusion.py` | sample() |

---

## ✅ 验证清单

使用前请确认：

- [ ] 所有文件都已创建
- [ ] 依赖包已安装
- [ ] 测试脚本通过
- [ ] 数据已准备好
- [ ] 配置文件路径正确
- [ ] GPU可用且显存足够
- [ ] 磁盘空间足够保存检查点

---

## 📞 获取帮助

如果遇到问题：

1. 运行 `python test_video_prediction.py` 检查环境
2. 查看 `VIDEO_PREDICTION_README.md` 常见问题部分
3. 检查TensorBoard日志了解训练状态
4. 验证数据路径和标注文件格式

---

最后更新: 2024年11月
版本: v1.0

