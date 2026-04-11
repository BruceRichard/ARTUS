# Flow Matching Module for ArtFormer
# Non-invasive integration with existing Diffusion framework

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class FlowMatchingScheduler:
    """
    Flow Matching scheduler that can work alongside DDPM
    Implements Conditional Flow Matching (CFM) and Optimal Transport (OT) paths
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        sigma_min: float = 1e-4,
        flow_type: str = 'optimal_transport',  # 'optimal_transport' or 'conditional'
    ):
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.flow_type = flow_type
        
    def get_velocity_target(
        self, 
        x_0: torch.Tensor, 
        x_1: torch.Tensor, 
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute flow matching velocity target
        
        Args:
            x_0: source distribution (noise)
            x_1: target distribution (clean data)
            t: time steps [0, 1]
            noise: optional noise for conditional flow
            
        Returns:
            x_t: interpolated sample
            v_t: velocity target
        """
        # Ensure t is in [0, 1] range
        t = t.float() / self.num_timesteps if t.max() > 1 else t
        t = t.view(-1, 1)  # [B, 1]
        
        if self.flow_type == 'optimal_transport':
            # Optimal Transport path: x_t = t * x_1 + (1 - t) * x_0
            x_t = t * x_1 + (1 - t) * x_0
            v_t = x_1 - x_0  # velocity is constant along OT path
            
        elif self.flow_type == 'conditional':
            # Conditional Flow Matching with Gaussian conditioning
            mu_t = t * x_1
            sigma_t = 1 - (1 - self.sigma_min) * t
            
            if noise is None:
                noise = torch.randn_like(x_0)
            
            x_t = mu_t + sigma_t * noise
            v_t = x_1 - (1 - self.sigma_min) * noise
            
        else:
            raise ValueError(f"Unknown flow type: {self.flow_type}")
            
        return x_t, v_t
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time steps uniformly from [0, num_timesteps)"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching loss module that can be used alongside DDPM loss
    """
    def __init__(
        self,
        loss_type: str = 'mse',  # 'mse' or 'l1'
        weighting: str = 'uniform',  # 'uniform' or 'velocity'
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weighting = weighting
        self.loss_fn = F.mse_loss if loss_type == 'mse' else F.l1_loss
        
    def forward(
        self, 
        pred_velocity: torch.Tensor, 
        target_velocity: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute flow matching loss
        
        Args:
            pred_velocity: predicted velocity from model
            target_velocity: target velocity from flow matching
            t: time steps for weighting (optional)
            
        Returns:
            loss: scalar loss value
        """
        loss = self.loss_fn(pred_velocity, target_velocity, reduction='none')
        
        if self.weighting == 'velocity' and t is not None:
            # Weight by time (similar to P2 weighting in diffusion)
            t_normalized = t.float().view(-1, 1) / 1000.0
            weight = 1.0 / (1.0 + t_normalized)
            loss = loss * weight
            
        return loss.mean()


class FlowMatchingWrapper(nn.Module):
    """
    Wrapper that adds Flow Matching capability to existing diffusion model
    Can be used in parallel with DDPM training
    """
    def __init__(
        self,
        scheduler: FlowMatchingScheduler,
        loss_module: FlowMatchingLoss,
        objective: str = 'pred_velocity',  # 'pred_velocity' or 'pred_x0'
    ):
        super().__init__()
        self.scheduler = scheduler
        self.loss_module = loss_module
        self.objective = objective
        
    def compute_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        cond: Optional[Dict] = None,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute flow matching loss
        
        Args:
            model: the denoising model
            x_start: clean data (x_1 in flow matching)
            cond: conditioning information
            noise: optional noise (x_0 in flow matching)
            
        Returns:
            loss: flow matching loss
            info: dictionary with additional information
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample time steps
        t = self.scheduler.sample_time(batch_size, device)
        
        # Generate noise (source distribution)
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get flow matching targets
        x_t, v_t = self.scheduler.get_velocity_target(noise, x_start, t)
        
        # Forward pass through model
        model_input = (x_t, cond) if cond is not None else x_t
        model_output = model(model_input, t)
        
        # Compute loss based on objective
        if self.objective == 'pred_velocity':
            pred_velocity = model_output
            loss = self.loss_module(pred_velocity, v_t, t)
        elif self.objective == 'pred_x0':
            # Convert x0 prediction to velocity
            pred_x0 = model_output
            t_normalized = t.float().view(-1, 1) / self.scheduler.num_timesteps
            pred_velocity = (pred_x0 - x_t) / (t_normalized + 1e-8)
            loss = self.loss_module(pred_velocity, v_t, t)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        info = {
            'x_t': x_t.detach(),
            'v_t': v_t.detach(),
            'pred_velocity': model_output.detach(),
            't': t,
        }

        return loss, info

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        dim: int,
        device: torch.device,
        cond: Optional[Dict] = None,
        num_steps: int = 100,
        method: str = 'euler',  # 'euler' or 'rk4'
    ) -> torch.Tensor:
        """
        Sample from flow matching model using ODE solver

        Args:
            model: the denoising model
            batch_size: number of samples
            dim: dimension of samples
            device: torch device
            cond: conditioning information
            num_steps: number of integration steps
            method: ODE solver method

        Returns:
            samples: generated samples
        """
        # Start from noise
        x = torch.randn(batch_size, dim, device=device)

        # Time steps from 0 to num_timesteps
        dt = self.scheduler.num_timesteps / num_steps

        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device, dtype=torch.long)

            # Get velocity prediction
            model_input = (x, cond) if cond is not None else x

            if self.objective == 'pred_velocity':
                v = model(model_input, t)
            elif self.objective == 'pred_x0':
                pred_x0 = model(model_input, t)
                t_normalized = t.float().view(-1, 1) / self.scheduler.num_timesteps
                v = (pred_x0 - x) / (t_normalized + 1e-8)

            # ODE integration
            if method == 'euler':
                x = x + v * (dt / self.scheduler.num_timesteps)
            elif method == 'rk4':
                # Runge-Kutta 4th order
                k1 = v

                t_mid = t + dt / 2
                x_mid = x + k1 * (dt / self.scheduler.num_timesteps) / 2
                model_input_mid = (x_mid, cond) if cond is not None else x_mid
                k2 = model(model_input_mid, t_mid.long())

                x_mid = x + k2 * (dt / self.scheduler.num_timesteps) / 2
                model_input_mid = (x_mid, cond) if cond is not None else x_mid
                k3 = model(model_input_mid, t_mid.long())

                t_next = t + dt
                x_next = x + k3 * (dt / self.scheduler.num_timesteps)
                model_input_next = (x_next, cond) if cond is not None else x_next
                k4 = model(model_input_next, t_next.long())

                x = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / self.scheduler.num_timesteps) / 6

        return x

