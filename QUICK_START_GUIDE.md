# 🚀 视频预测训练 - 快速上手指南

这是一个**极简版**的使用指南，让您在10分钟内开始训练。

---

## ✅ 准备工作（5分钟）

### 1. 检查GPU
```bash
nvidia-smi
```
确保有可用的GPU和足够显存（建议≥16GB）

### 2. 安装依赖
```bash
pip install -r requirements_video_pred.txt
```

### 3. 测试环境
```bash
python test_video_prediction.py
```
看到 "✓ All tests passed!" 即可继续

---

## 📦 准备数据（3分钟）

### 情况A: 您已有 pkl 标注文件
如果已经有 `b2d_infos_train.pkl` 和 `b2d_infos_val.pkl`，跳过这步。

### 情况B: 需要准备数据
```bash
python prepare_video_data.py \
    --data_root /path/to/your/bench2drive \
    --output_dir data/infos \
    --val_ratio 0.2
```

将 `/path/to/your/bench2drive` 替换为您的实际数据路径。

---

## ⚙️ 配置训练（2分钟）

编辑 `configs/config_video_prediction.yaml`:

```yaml
Dataset:
  train:
    data_root: '/path/to/your/bench2drive'  # ← 改这里
    ann_file: 'data/infos/b2d_infos_train.pkl'
  eval:
    data_root: '/path/to/your/bench2drive'  # ← 改这里
    ann_file: 'data/infos/b2d_infos_val.pkl'
```

**仅需修改** `data_root` 路径！其他保持默认即可。

---

## 🎯 开始训练

### 单GPU训练
```bash
python train_video_pred.py --config configs/config_video_prediction.yaml
```

### 多GPU训练（推荐）
```bash
# 4个GPU
torchrun --nproc_per_node=4 train_video_pred.py --config configs/config_video_prediction.yaml

# 8个GPU
torchrun --nproc_per_node=8 train_video_pred.py --config configs/config_video_prediction.yaml
```

---

## 📊 监控训练

**方法1: 命令行输出**
训练过程会每100步打印一次loss

**方法2: TensorBoard（推荐）**
新开一个终端：
```bash
tensorboard --logdir logs/video_prediction
```
然后在浏览器打开: http://localhost:6006

---

## 🎬 测试模型

训练几个小时后，测试一下效果：

```bash
python eval_video_pred.py \
    --config configs/config_video_prediction.yaml \
    --ckpt checkpoints/video_prediction/step_10000.pt \
    --num_samples 10 \
    --save_visualizations
```

查看生成的图像在: `checkpoints/video_prediction/evaluation/`

---

## 🎨 使用训练好的模型

```bash
python inference_demo.py \
    --config configs/config_video_prediction.yaml \
    --ckpt checkpoints/video_prediction/step_50000.pt \
    --images img1.png img2.png img3.png img4.png \
    --output result.png
```

需要提供4张连续的历史帧图像。

---

## ⏱️ 训练时间参考

| GPU型号 | Batch Size | 每步耗时 | 10K步预计 |
|---------|-----------|---------|----------|
| RTX 3090 | 4 | ~2秒 | ~5.5小时 |
| V100 | 4 | ~2.5秒 | ~7小时 |
| A100 | 8 | ~2秒 | ~5.5小时 |

* 以上为单GPU训练时间，多GPU成比例加速
* 完整训练建议至少50K步（约30-50小时）

---

## 🛠️ 常见问题

### Q1: Out of Memory (OOM)
**解决**: 在配置文件中减小 `batch_size`:
```yaml
Train:
  batch_size: 2  # 从4改为2
```

### Q2: 数据集加载失败
**解决**: 
1. 检查 `data_root` 路径是否正确
2. 检查 `ann_file` 是否存在
3. 运行 `prepare_video_data.py --verify`

### Q3: VAE模型下载失败
**解决**: 设置代理或手动下载
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q4: 训练速度慢
**优化**:
1. 增加 `num_workers: 16`
2. 使用多GPU训练
3. 减少图像尺寸 `img_size: [128, 224]`

---

## 📝 下一步

### 第一次训练建议

1. **先小规模测试** (1-2小时)
   - 训练5000步
   - 验证loss是否下降
   - 检查生成的图像

2. **完整训练** (1-2天)
   - 训练至少50K步
   - 定期评估效果
   - 保存最佳检查点

3. **精调优化** (半天)
   - 降低学习率
   - 继续训练10K步
   - 最终评估

### 进阶功能

想要更多功能？查看完整文档：
- 详细说明: `VIDEO_PREDICTION_README.md`
- 文件清单: `VIDEO_PREDICTION_FILES.md`

---

## 🎉 就是这么简单！

**3条命令开始训练**:
```bash
# 1. 安装
pip install -r requirements_video_pred.txt

# 2. 准备数据（如果需要）
python prepare_video_data.py --data_root /your/data/path --output_dir data/infos

# 3. 开始训练
torchrun --nproc_per_node=4 train_video_pred.py --config configs/config_video_prediction.yaml
```

**或者使用一键脚本**:
```bash
chmod +x quick_start.sh
./quick_start.sh
```

---

祝您训练顺利！🚀

