# Flow Matching Integration - Scheme 1: Dual Loss Hybrid Training

import torch
import torch.nn.functional as F
from torch import nn
from .diffusion_wapper import _DiffusionModel
from .flow_matching import FlowMatchingScheduler, FlowMatchingLoss, FlowMatchingWrapper
from collections import namedtuple

# constants
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class DualLossDiffusionModel(_DiffusionModel):
    """
    扩展的Diffusion模型，支持DDPM和Flow Matching双损失训练
    
    特点:
    - 完全向后兼容原始DDPM训练
    - 通过配置启用/禁用Flow Matching
    - 共享模型参数，提升泛化能力
    - 可动态调整损失权重
    """
    
    def __init__(
        self,
        model,
        timesteps=1000,
        sampling_timesteps=None,
        beta_schedule='cosine',
        loss_type='l2',
        objective='pred_x0',
        data_scale=1.0,
        data_shift=0.0,
        p2_loss_weight_gamma=0.,
        p2_loss_weight_k=1,
        ddim_sampling_eta=1.,
        # Flow Matching specific parameters
        enable_flow_matching=False,
        flow_matching_weight=0.5,
        flow_type='optimal_transport',
        flow_objective='pred_velocity',
        flow_loss_type='mse',
    ):
        # Initialize parent DDPM model
        super().__init__(
            model=model,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            beta_schedule=beta_schedule,
            loss_type=loss_type,
            objective=objective,
            data_scale=data_scale,
            data_shift=data_shift,
            p2_loss_weight_gamma=p2_loss_weight_gamma,
            p2_loss_weight_k=p2_loss_weight_k,
            ddim_sampling_eta=ddim_sampling_eta
        )
        
        # Flow Matching components
        self.enable_flow_matching = enable_flow_matching
        self.flow_matching_weight = flow_matching_weight
        self.ddpm_weight = 1.0 - flow_matching_weight
        
        if self.enable_flow_matching:
            self.fm_scheduler = FlowMatchingScheduler(
                num_timesteps=timesteps,
                flow_type=flow_type
            )
            self.fm_loss = FlowMatchingLoss(loss_type=flow_loss_type)
            self.fm_wrapper = FlowMatchingWrapper(
                scheduler=self.fm_scheduler,
                loss_module=self.fm_loss,
                objective=flow_objective
            )
            self.flow_objective = flow_objective
    
    def forward(self, x_start, t, ret_pred_x=False, noise=None, cond=None):
        """
        前向传播 - 计算混合损失
        
        维度对齐:
        - x_start: [B, D] 干净数据
        - t: [B] 时间步
        - noise: [B, D] 噪声 (可选)
        - cond: dict 条件信息
        
        返回:
        - loss: 标量损失 (DDPM + Flow Matching)
        - unreduced_loss: [B] 未归约的损失
        """
        # 1. 计算原始DDPM损失
        ddpm_loss, unreduced_loss = super().forward(
            x_start, t, ret_pred_x=False, noise=noise, cond=cond
        )
        
        # 2. 如果启用Flow Matching，计算FM损失
        if self.enable_flow_matching:
            fm_loss, fm_info = self.fm_wrapper.compute_loss(
                model=self.model,
                x_start=x_start,
                cond=cond,
                noise=noise
            )
            
            # 3. 混合损失
            total_loss = self.ddpm_weight * ddpm_loss + self.flow_matching_weight * fm_loss
            
            if ret_pred_x:
                # 返回详细信息用于调试
                model_in = (fm_info['x_t'], cond) if cond is not None else fm_info['x_t']
                model_out = self.model(model_in, t)
                return total_loss, fm_info['x_t'], x_start, model_out, unreduced_loss
            else:
                return total_loss, unreduced_loss
        else:
            # 仅使用DDPM
            if ret_pred_x:
                noise = noise if noise is not None else torch.randn_like(x_start)
                x = self.q_sample(x_start=x_start, t=t, noise=noise)
                model_in = (x, cond) if cond is not None else x
                model_out = self.model(model_in, t)
                return ddpm_loss, x, x_start, model_out, unreduced_loss
            else:
                return ddpm_loss, unreduced_loss
    
    def diffusion_model_from_latent(self, x_start, cond=None):
        """
        包装函数 - 与原始接口完全兼容
        
        维度对齐:
        - x_start: [B, D] 干净的潜在向量
        - cond: dict 条件信息
        
        返回: (loss, loss_100, loss_1000, model_out, cond)
        """
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()
        
        loss, x, target, model_out, unreduced_loss = self(x_start, t, cond=cond, ret_pred_x=True)
        loss_100 = unreduced_loss[t < 100].mean().detach()
        loss_1000 = unreduced_loss[t > 100].mean().detach()
        
        return loss, loss_100, loss_1000, model_out, cond


class DiffusionModelGeometry(DualLossDiffusionModel):
    """方案1的便捷包装类"""
    def __init__(self, model, config):
        # 从配置中提取参数
        diff_config = config['diffusion_model_paramerter']['diffusion_config']
        fm_config = config.get('flow_matching', {})
        
        super().__init__(
            model=model,
            enable_flow_matching=fm_config.get('enabled', False),
            flow_matching_weight=fm_config.get('loss_weight', 0.5),
            flow_type=fm_config.get('flow_type', 'optimal_transport'),
            flow_objective=fm_config.get('objective', 'pred_velocity'),
            flow_loss_type=fm_config.get('loss_type', 'mse'),
            **diff_config
        )

