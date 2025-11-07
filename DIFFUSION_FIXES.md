# Diffusion 模型修复说明

## 问题诊断

你报告的问题：预测结果是彩色噪声（乱码），完全不对。

## 根本原因分析

通过检查代码，发现以下问题：

### 1. **DDIM 采样算法实现错误** ⚠️ 关键问题

**位置**: `model/video_prediction_diffusion.py` 第 344-372 行

**原问题**:
```python
# 错误的 DDIM 更新
alpha_t = alphas_cumprod[t]
alpha_t_prev = alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)

pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
# 没有 clipping！

dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred
latents = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
```

**问题**:
- `pred_x0` 预测没有 clipping，导致数值爆炸
- 在训练早期，模型预测不准确时，`pred_x0` 可能非常大
- 这会导致后续步骤的 `latents` 数值失控，最终产生噪声

**修复**:
```python
# 正确的 DDIM 更新，带 clipping
alpha_t = alphas_cumprod[t]
alpha_t_prev = alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)

# 预测 x0
sqrt_alpha_t = torch.sqrt(alpha_t)
sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
pred_x0 = (latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

# 关键：Clip predicted x0 for stability
pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)

# 计算新的 latents
sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
latents = sqrt_alpha_t_prev * pred_x0 + sqrt_one_minus_alpha_t_prev * noise_pred
```

### 2. **推理步数过多** ⚠️ 次要问题

**位置**: `configs/config_video_prediction.yaml`

**原问题**:
```yaml
num_inference_steps: 50  # 太多了！
```

**问题**:
- 50 步推理对于训练初期的模型太多
- 每步都可能累积误差
- 训练可视化太慢

**修复**:
```yaml
num_inference_steps: 20  # 更合理的步数
```

### 3. **缺少调试信息** ℹ️ 辅助问题

**位置**: `video_prediction_trainer.py`

**添加了**:
- 推理步数日志
- 输入/输出范围检查
- EMA 初始化验证
- 参数范数检查

## 修复总结

### 文件修改清单

1. ✅ `model/video_prediction_diffusion.py`
   - 修复 DDIM 采样算法（添加 x0 clipping）
   - 优化数值稳定性

2. ✅ `configs/config_video_prediction.yaml`
   - 推理步数: 50 → 20
   - 添加 wandb 配置
   - 添加 visualize_every_step 配置

3. ✅ `video_prediction_trainer.py`
   - 添加详细的调试日志
   - 添加输入/输出范围检查
   - 修复 eval_step 推理步数默认值
   - 添加 EMA 初始化验证

4. ✅ 新增测试脚本
   - `test_model_inference.py`: 快速测试模型推理
   - `diagnose_diffusion.py`: 详细诊断采样过程

## 为什么会产生噪声？

在 diffusion 模型中，采样过程是从纯噪声逐步去噪到清晰图像的过程：

```
t=999: [纯噪声] → ... → t=500: [部分去噪] → ... → t=0: [清晰图像]
```

**没有 x0 clipping 时**:
1. 模型预测 `noise_pred`
2. 从 `noise_pred` 推导 `pred_x0` (预测的最终清晰图像)
3. 如果 `pred_x0` 数值很大（比如 100 或 -100），数值就失控了
4. 下一步的 `latents` 会继承这个失控的数值
5. 最终解码出来就是噪声

**有 x0 clipping 时**:
1. 模型预测 `noise_pred`
2. 从 `noise_pred` 推导 `pred_x0`
3. **将 `pred_x0` 限制在 [-3, 3] 范围内**
4. 使用被限制的 `pred_x0` 计算下一步
5. 数值稳定，最终能得到合理的图像

## 如何验证修复

### 方法 1: 运行测试脚本

```bash
cd /media/raid/workspace/tengbo/DiffAD-main
conda activate diff

# 快速测试
python test_model_inference.py

# 详细诊断
python diagnose_diffusion.py
```

**预期输出**（即使模型未训练）:
- Predicted frames range 应该在合理范围内（比如 [-2, 2]）
- Denormalized range 应该在 [0, 1]
- 不应该出现 NaN 或 Inf
- 统计量应该合理（mean ~0.3-0.7, std ~0.1-0.3）

### 方法 2: 继续训练并观察

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_video_pred.py --config configs/config_video_prediction.yaml
```

**观察点**:
1. 训练 loss 应该下降（不是 NaN）
2. Step 1000 时的可视化应该:
   - 不是彩色噪声
   - 能看出图像结构（即使模糊）
   - 随着训练进行逐渐变清晰

### 方法 3: 检查 wandb 日志

如果启用了 wandb:
- 查看 `train/visualization` - 应该能看到图像结构
- 查看 `train/loss` - 应该平滑下降
- 查看 `eval/psnr` - 应该逐渐上升

## 额外建议

### 训练技巧

1. **从小步数开始**
   - 初期使用 10-20 步推理
   - 训练后期可以增加到 50 步

2. **检查点**
   - 定期保存检查点（每 5000 步）
   - 如果出现问题可以回退

3. **学习率**
   - 当前设置 `1e-4` 是合理的
   - 如果 loss 不下降，可以尝试 `1e-5`

### 数值稳定性

当前的 clipping 值 `[-3, 3]` 是基于 VAE latent 的典型范围。如果你发现：

- **太多值被 clip**: 说明模型预测不稳定，考虑：
  - 降低学习率
  - 增加 gradient clipping
  - 检查数据归一化

- **几乎没有值被 clip**: 说明模型训练正常，可以：
  - 保持当前设置
  - 或者逐渐放宽到 `[-4, 4]`

### 监控指标

在训练日志中查找：
```
# 好的迹象
clipped=0 values (或很小的数字)
Final latents range: [-2.5, 2.5]
Predicted frames range: [-1.5, 1.5]
Denormalized: [0.0, 1.0]

# 坏的迹象
clipped=1000+ values (大量 clipping)
Final latents range: [-100, 100] (数值失控)
NaN 或 Inf
```

## 参考资料

- DDIM 论文: "Denoising Diffusion Implicit Models"
- Stable Diffusion 实现中的 x0 clipping 策略
- VAE latent space 通常范围在 [-3, 3] 或 [-4, 4]

## 联系

如果问题仍然存在，检查：
1. 运行 `diagnose_diffusion.py` 获取详细输出
2. 查看训练日志中的 "clipped=XXX values"
3. 检查 VAE 是否正确加载（应该从 `/media/raid/workspace/tengbo/vae`）
4. 确认数据归一化正确（ImageNet mean/std）

