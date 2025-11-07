#!/usr/bin/env python3
"""
测试已有checkpoint的采样质量
"""

import torch
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    from model.video_prediction_diffusion import VideoPredictionDiffusion
    
    # 找到最新的checkpoint
    ckpt_dir = Path("/media/raid/workspace/tengbo/DiffAD-main/logs/video_prediction/checkpoints")
    if not ckpt_dir.exists():
        logger.error(f"Checkpoint directory not found: {ckpt_dir}")
        return
    
    ckpts = list(ckpt_dir.glob("*.pt"))
    if not ckpts:
        logger.error(f"No checkpoints found in {ckpt_dir}")
        return
    
    # 选择最新的
    latest_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading checkpoint: {latest_ckpt}")
    
    # 创建模型
    model = VideoPredictionDiffusion(
        vae_model_path="/media/raid/workspace/tengbo/vae",
        latent_channels=4,
        past_frames=4,
        future_frames=4,
        img_size=(256, 448),
        dit_config={'hidden_size': 768, 'num_heads': 12, 'depth': 12}
    ).cuda()
    
    # 加载checkpoint
    checkpoint = torch.load(latest_ckpt, map_location='cuda:0')
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        logger.info(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    else:
        logger.error("Invalid checkpoint format")
        return
    
    model.eval()
    
    # 创建随机输入（模拟已标准化的数据）
    B = 1
    past_frames = torch.randn(B, 4, 3, 256, 448).cuda() * 0.5
    task_label = torch.randn(B, 256).cuda()
    ego_status = torch.randn(B, 7).cuda()
    command = torch.randn(B, 6).cuda()
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"测试采样质量")
    logger.info(f"{'=' * 70}")
    
    for num_steps in [5, 10, 20, 50]:
        logger.info(f"\n使用 {num_steps} 步采样:")
        
        with torch.no_grad():
            predicted = model.sample(
                past_frames,
                task_label,
                ego_status,
                command,
                num_inference_steps=num_steps
            )
        
        logger.info(f"  输出形状: {predicted.shape}")
        logger.info(f"  输出范围: [{predicted.min():.3f}, {predicted.max():.3f}]")
        logger.info(f"  输出均值: {predicted.mean():.3f}, 标准差: {predicted.std():.3f}")
        
        # 检查是否合理
        pred_mean = predicted.mean().item()
        pred_std = predicted.std().item()
        
        # 标准化后的图像应该在[-3, 3]左右，均值接近0
        is_reasonable = (abs(pred_mean) < 2.0 and 0.1 < pred_std < 2.0)
        
        if is_reasonable:
            logger.info(f"  ✓ 输出看起来合理")
        else:
            logger.warning(f"  ⚠️ 输出可能有问题 - 可能仍然是噪声")
            
            # 检查是否全是高频噪声
            # 高频噪声的特点：相邻像素差异很大
            pred_np = predicted[0, 0].cpu().numpy()  # 取第一帧第一通道
            import numpy as np
            dx = np.abs(np.diff(pred_np, axis=0)).mean()
            dy = np.abs(np.diff(pred_np, axis=1)).mean()
            logger.info(f"  高频分析: dx={dx:.4f}, dy={dy:.4f}")
            
            if dx > 0.5 or dy > 0.5:
                logger.warning(f"  ⚠️ 检测到高频噪声特征")
    
    # 测试训练模式
    logger.info(f"\n{'=' * 70}")
    logger.info(f"测试训练模式")
    logger.info(f"{'=' * 70}")
    
    model.train()
    
    # 创建目标帧（GT）
    future_frames_gt = torch.randn(B, 4, 3, 256, 448).cuda() * 0.5
    
    try:
        with torch.no_grad():
            loss_dict = model(past_frames, future_frames_gt, task_label, ego_status, command)
        logger.info(f"✓ 训练前向传播成功")
        logger.info(f"  Loss: {loss_dict['loss'].item():.6f}")
    except Exception as e:
        logger.error(f"❌ 训练前向传播失败: {e}")
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"测试完成")
    logger.info(f"{'=' * 70}")

if __name__ == '__main__':
    main()


