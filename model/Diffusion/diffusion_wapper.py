# [ArtFormer]: This code is adapted from Diffusion-SDF `https://github.com/princeton-computational-imaging/Diffusion-SDF`
# [ARTUS]: adds `refine`, the partial-noise refinement of a coarse geometry
# latent with the frozen text-conditioned prior (paper Eq. 5-7): the coarse
# prediction is perturbed to an intermediate diffusion step rho* and denoised
# back to 0, instead of restarting generation from pure Gaussian noise.

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from .utils.helpers import *
from collections import namedtuple

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class _DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000, sampling_timesteps = None, beta_schedule = 'cosine',
        loss_type = 'l2', objective = 'pred_x0',
        data_scale = 1.0, data_shift = 0.0,
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()

        self.model = model
        self.objective = objective

        betas = linear_beta_schedule(timesteps) if beta_schedule == 'linear' else cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.loss_fn = F.l1_loss if loss_type=='l1' else F.mse_loss

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))


        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (x0 - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def ddim_sample(
        self,
        dim,
        batch_size,
        noise=None,
        clip_denoised=True,
        traj=False,
        cond=None,
    ):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(0., total_timesteps, steps = sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj_buffer = []

        x_T = default(noise, torch.randn(batch, dim, device = device))

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            model_input = (x_T, cond) if cond is not None else x_T
            with torch.no_grad():
                pred_noise, x_start, *_ = self.model_predictions(model_input, time_cond)

            if clip_denoised:
                x_start.clamp_(-1., 1.)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(x_T) if time_next > 0 else 0.
            x_next = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            x_T = x_next
            traj_buffer.append(x_T.clone())

        return x_T, (traj_buffer if traj else None)

    def sample(
        self,
        dim,
        batch_size,
        noise=None,
        clip_denoised=True,
        traj=False,
        cond=None,
    ):

        batch, device, objective = batch_size, self.betas.device, self.objective

        traj_buffer = []

        x_T = default(noise, torch.randn(batch, dim, device = device))

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):

            time_cond = torch.full((batch,), t, device = device, dtype = torch.long)

            model_input = (x_T, cond) if cond is not None else x_T
            with torch.no_grad():
                pred_noise, x_start, *_ = self.model_predictions(model_input, time_cond)
            if clip_denoised:
                x_start.clamp_(-1., 1.)

            model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x_T, t = time_cond)

            noise = torch.randn_like(x_T) if t > 0 else 0. # no noise if t == 0
            x_next = model_mean + (0.5 * model_log_variance).exp() * noise

            x_T = x_next
            traj_buffer.append(x_T.clone())

        return x_T, (traj_buffer if traj else None)

    @torch.no_grad()
    def refine(
        self,
        coarse_latent,
        start_step,
        cond=None,
        sampler_steps=None,
        clip_denoised=True,
    ):
        """[ARTUS] Partial-noise refinement of a coarse geometry latent.

        The coarse prediction is perturbed to the intermediate diffusion step
        `start_step` (rho*),

            z_rho* = sqrt(alpha_bar_rho*) * v_coarse
                     + sqrt(1 - alpha_bar_rho*) * eps,

        and the frozen prior then runs the text-conditioned reverse process
        back to 0 (paper Eq. 5-7). `start_step == 0` bypasses refinement.

        Args:
            coarse_latent: (B, dim) coarse geometry latent from D_v.
            start_step:    noise index rho* in the training schedule.
            cond:          text condition of the frozen refiner.
            sampler_steps: retained reverse steps; if None or >= start_step,
                           the full ancestral (DDPM) reverse over
                           [0, start_step] is used, otherwise a DDIM schedule.
        """
        device = self.betas.device
        x = coarse_latent.to(device)
        start_step = int(min(max(int(start_step), 0), self.num_timesteps - 1))
        if start_step <= 0:
            return x

        batch = x.shape[0]
        t_start = torch.full((batch,), start_step, device=device, dtype=torch.long)
        x = self.q_sample(x, t_start)

        if sampler_steps is None or int(sampler_steps) >= start_step:
            for t in tqdm(reversed(range(0, start_step + 1)), desc='refinement loop', total=start_step + 1):
                time_cond = torch.full((batch,), t, device=device, dtype=torch.long)
                model_input = (x, cond) if cond is not None else x
                pred_noise, x_start, *_ = self.model_predictions(model_input, time_cond)
                if clip_denoised:
                    x_start.clamp_(-1., 1.)
                model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=time_cond)
                noise = torch.randn_like(x) if t > 0 else 0.
                x = model_mean + (0.5 * model_log_variance).exp() * noise
            return x

        times = torch.linspace(0., start_step, steps=int(sampler_steps) + 2)[:-1]
        times = sorted(set(times.int().tolist()))
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        eta = self.ddim_sampling_eta
        for time, time_next in tqdm(time_pairs, desc='refinement loop'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            model_input = (x, cond) if cond is not None else x
            pred_noise, x_start, *_ = self.model_predictions(model_input, time_cond)
            if clip_denoised:
                x_start.clamp_(-1., 1.)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(x) if time_next > 0 else 0.
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return x

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance


    # "nice property": return x_t given x_0, noise, and timestep
    def q_sample(self, x_start, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        #noise = torch.clamp(noise, min=-6.0, max=6.0)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # main function for calculating loss
    def forward(self, x_start, t, ret_pred_x=False, noise=None, cond=None):
        '''
        x_start: [B, D]
        t: [B]
        '''

        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_in = (x, cond) if cond is not None else x
        model_out = self.model(model_in, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        #loss = reduce(loss, 'b ... -> b (...)', 'mean', b = x_start.shape[0]) # only one dim of latent so don't need this line

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        unreduced_loss = loss.detach().clone().mean(dim=1)

        if ret_pred_x:
            return loss.mean(), x, target, model_out, unreduced_loss
        else:
            return loss.mean(), unreduced_loss

    def model_predictions(self, model_input, t):

        model_output = self.model(model_input, t)

        x = model_input[0] if type(model_input) is tuple else model_input

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)

    # a wrapper function that only takes x_start (clean modulation vector) and condition
    # does everything including sampling timestep and returns loss, loss_100, loss_1000, prediction
    def diffusion_model_from_latent(self, x_start, cond=None):
        # STEP 1: sample timestep
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()

        # STEP 3: pass to forward function
        loss, x, target, model_out, unreduced_loss = self(x_start, t, cond=cond, ret_pred_x=True)
        loss_100 = unreduced_loss[t<100].mean().detach()
        loss_1000 = unreduced_loss[t>100].mean().detach()

        return loss, loss_100, loss_1000, model_out, cond

    def generate_conditional(self, cond):
        self.eval()
        samp, _ = self.sample(
            dim=self.model.dim,
            batch_size=cond['text'].shape[0],
            traj=False,
            cond=cond,
        )

        return samp

    def generate_conditional_ddim(self, cond):
        self.eval()
        samp, _ = self.ddim_sample(
            dim=self.model.dim,
            batch_size=cond['text'].shape[0],
            traj=False,
            cond=cond,
        )

        return samp

    def generate_unconditional(self, num_samples):
        self.eval()
        samp, _ = self.sample(dim=self.model.dim, batch_size=num_samples, traj=False, cond=None)

        return samp

class DiffusionModel(_DiffusionModel):
    def __init__(self, model, config):
        super().__init__(model=model, **config['diffusion_model_paramerter']['diffusion_config'])
