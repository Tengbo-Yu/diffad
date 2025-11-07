#!/usr/bin/env python3
"""
诊断模型是否能正确工作
"""

import torch
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    from model.video_prediction_diffusion import VideoPredictionDiffusion
    
    logger.info("=" * 60)
    logger.info("模型诊断工具")
    logger.info("=" * 60)
    
    # 创建模型
    model = VideoPredictionDiffusion(
        vae_model_path="/media/raid/workspace/tengbo/vae",
        latent_channels=4,
        past_frames=4,
        future_frames=4,
        img_size=(256, 448),
        dit_config={'hidden_size': 768, 'num_heads': 12, 'depth': 12}
    ).cuda().eval()
    
    # 创建假数据
    B = 1
    past_frames = torch.randn(B, 4, 3, 256, 448).cuda() * 0.5  # 标准化后的范围
    task_label = torch.randn(B, 256).cuda()
    ego_status = torch.randn(B, 7).cuda()
    command = torch.randn(B, 6).cuda()
    
    logger.info(f"\n输入数据:")
    logger.info(f"  past_frames: {past_frames.shape}, range=[{past_frames.min():.3f}, {past_frames.max():.3f}]")
    
    # 测试1: 检查模型是否能正常前向传播（训练模式）
    logger.info(f"\n{'=' * 60}")
    logger.info("测试 1: 训练模式前向传播")
    logger.info("=" * 60)
    
    model.train()
    try:
        with torch.no_grad():
            # 编码
            past_latents = model.encode_frames(past_frames)
            logger.info(f"✓ Past latents: {past_latents.shape}, range=[{past_latents.min():.3f}, {past_latents.max():.3f}]")
            
            # 创建目标latents（模拟GT）
            target_latents = torch.randn_like(past_latents[:, :1]).repeat(1, 4, 1, 1, 1) * 2.0
            logger.info(f"✓ Target latents: {target_latents.shape}")
            
            # 前向传播
            loss_dict = model(past_frames, target_latents, task_label, ego_status, command)
            logger.info(f"✓ Loss: {loss_dict['loss'].item():.6f}")
            
            # 检查梯度
            loss_dict['loss'].backward()
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
            if has_grad:
                logger.info(f"✓ 梯度正常")
            else:
                logger.warning(f"⚠️ 没有梯度!")
                
    except Exception as e:
        logger.error(f"❌ 训练模式失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试2: 检查采样是否工作
    logger.info(f"\n{'=' * 60}")
    logger.info("测试 2: 采样模式（未训练的模型）")
    logger.info("=" * 60)
    
    model.eval()
    model.zero_grad()
    
    for num_steps in [5, 10, 20]:
        logger.info(f"\n--- 使用 {num_steps} 步采样 ---")
        try:
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
                
                # 检查是否是纯噪声
                frame_std = predicted.std(dim=(2, 3, 4)).mean()  # 每帧的空间标准差
                temporal_std = predicted.mean(dim=(2, 3, 4)).std(dim=1).mean()  # 时间标准差
                
                logger.info(f"  帧内标准差: {frame_std:.3f}")
                logger.info(f"  帧间标准差: {temporal_std:.3f}")
                
                if frame_std > 0.1 and frame_std < 2.0:
                    logger.info(f"  ✓ 输出看起来合理")
                else:
                    logger.warning(f"  ⚠️ 输出可能有问题")
                    
        except Exception as e:
            logger.error(f"  ❌ 采样失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试3: 检查采样的中间过程
    logger.info(f"\n{'=' * 60}")
    logger.info("测试 3: 采样过程诊断")
    logger.info("=" * 60)
    
    try:
        with torch.no_grad():
            # 手动运行一步采样来检查中间值
            model.eval()
            
            # 编码过去帧
            past_latents = model.encode_frames(past_frames)
            context_latent = past_latents.mean(dim=1, keepdim=True)  # [B, 1, C, H, W]
            
            # 提取特征
            bevfeature = model.extract_bevfeature(past_frames)
            logger.info(f"BEV特征: {bevfeature.shape}, range=[{bevfeature.min():.3f}, {bevfeature.max():.3f}]")
            
            # 初始化噪声
            B, C, H, W = past_latents.shape[0], model.latent_channels, model.latent_h, model.latent_w
            latents = torch.randn(B, model.future_frames, C, H, W, device=past_frames.device)
            logger.info(f"\n初始噪声: range=[{latents.min():.3f}, {latents.max():.3f}]")
            
            # 重复条件
            from einops import repeat, rearrange
            latents_flat = rearrange(latents, 'b t c h w -> (b t) c h w')
            bevfeature_repeated = repeat(bevfeature, 'b d -> (b repeat) d', repeat=model.future_frames)
            task_label_repeated = repeat(task_label, 'b d -> (b repeat) d', repeat=model.future_frames)
            ego_status_repeated = repeat(ego_status, 'b d -> (b repeat) d', repeat=model.future_frames)
            command_repeated = repeat(command, 'b d -> (b repeat) d', repeat=model.future_frames)
            
            # 测试一步去噪
            alphas_cumprod = model._get_alphas_cumprod(past_frames.device)
            t = 999
            t_batch = torch.tensor([t] * (B * model.future_frames), device=past_frames.device)
            
            # 预测噪声
            noise_pred = model.dit(
                x=latents_flat,
                t=t_batch,
                bevfeature=bevfeature_repeated,
                task_label=task_label_repeated,
                ego_status=ego_status_repeated,
                command=command_repeated
            )
            
            logger.info(f"\n预测的噪声 (t={t}): range=[{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
            logger.info(f"  噪声均值: {noise_pred.mean():.3f}, 标准差: {noise_pred.std():.3f}")
            
            # DDIM更新
            alpha_t = alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            pred_x0 = (latents_flat - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            logger.info(f"\n预测的 x0 (t={t}):")
            logger.info(f"  alpha_t = {alpha_t:.6f}")
            logger.info(f"  pred_x0: range=[{pred_x0.min():.3f}, {pred_x0.max():.3f}]")
            logger.info(f"  pred_x0均值: {pred_x0.mean():.3f}, 标准差: {pred_x0.std():.3f}")
            
            # 检查是否合理
            if abs(pred_x0.mean()) < 5 and pred_x0.std() < 10:
                logger.info(f"  ✓ pred_x0 在合理范围内")
            else:
                logger.warning(f"  ⚠️ pred_x0 范围异常！")
            
            # 解码看看
            decoded = model.decode_latents(pred_x0.reshape(B, model.future_frames, C, H, W))
            logger.info(f"\n解码后的图像: range=[{decoded.min():.3f}, {decoded.max():.3f}]")
            logger.info(f"  解码均值: {decoded.mean():.3f}, 标准差: {decoded.std():.3f}")
            
    except Exception as e:
        logger.error(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"\n{'=' * 60}")
    logger.info("诊断完成")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()


