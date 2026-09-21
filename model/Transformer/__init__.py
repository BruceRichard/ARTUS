import math

import torch
import lightning as L

import torch.nn.functional as F

import numpy as np
import utils.mesh as MeshUtils
import wandb
import trimesh
import yaml

from rich import print

from tqdm import tqdm
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
from utils.base import TransArticulatedBaseModule
from .transformer.decoder import TransformerDecoder
from utils.mylogging import Log

from model.SDFAutoEncoder import SDFAutoEncoder
from model.Diffusion import Diffusion

class TransDiffusionCombineModel(TransArticulatedBaseModule):
    """ARTUS joint-latent model (paper Sec. 3).

    The geometry autoencoder and the text-conditioned diffusion refiner R_phi
    are pretrained and frozen; only the joint-latent transformer is trained,
    with the structure-controlled objective

        L = L_end + alpha * L_state + beta * L_geo + lambda * L_scl

    (Eq. 8-10). Channel normalization of the regression targets uses
    optimization-set statistics with standard deviations bounded below by
    1e-6 (Appendix C.2).
    """

    def __init__(self, config):
        super().__init__(config)

        self.automatic_optimization = False

        self._device = config['device']
        self.config = config
        self.op_config = config['optimizer_paramerter']
        self.tf_config = config['transformer_model_paramerter']
        self.part_structure = config['part_structure']

        self.dim_state = (self.part_structure['bounding_box']
                          + self.part_structure['joint_data_origin']
                          + self.part_structure['joint_data_direction']
                          + self.part_structure['limit'])
        self.dim_latent = self.part_structure['latentcode']

        # Frozen text-conditioned geometry refiner R_phi.
        Log.info('Using pretrained diffusion model: %s', config['diffusion_model']['pretrained_model_path'])
        self.diffusion = Diffusion.load_from_checkpoint(config['diffusion_model']['pretrained_model_path'], map_location='cpu')
        self.diff_config = self.diffusion.diff_config
        self.config['diff_config'] = self.diffusion.diff_config
        self.diffusion.eval()
        for param in self.diffusion.parameters():
            param.requires_grad_(False)
        Log.info('Loaded diffusion model (frozen)')

        self.transformer = TransformerDecoder(config)

        self.e_config = config['evaluation']

        try:
            Log.info('Using pretrained SDF model: %s', config['evaluation']['sdf_model_path'])
            self.sdf = SDFAutoEncoder.load_from_checkpoint(self.e_config['sdf_model_path'], map_location='cpu')
        except Exception as e:
            print("DO NOT FOUND CUSTOM CKPT. USE DEFAULT CKPT. : ", e)
            import time; time.sleep(2)
            self.sdf = self.diffusion.sdf

        self.sdf.eval()
        for param in self.sdf.parameters():
            param.requires_grad_(False)
        self.e_config['eval_mesh_output_path'] = Path(self.e_config['eval_mesh_output_path'])
        self.e_config['eval_mesh_output_path'].mkdir(parents=True, exist_ok=True)
        Log.info('Loaded SDF model (frozen)')

        # Channel normalization statistics of the optimization set.
        self.register_buffer('state_mean', torch.zeros(self.dim_state))
        self.register_buffer('state_std', torch.ones(self.dim_state))
        self.register_buffer('latent_mean', torch.zeros(self.dim_latent))
        self.register_buffer('latent_std', torch.ones(self.dim_latent))
        norm_config = config.get('normalization', {})
        self.normalization_enabled = bool(norm_config.get('enabled', False))
        if self.normalization_enabled:
            stats_path = norm_config.get('stats_path')
            if stats_path and Path(stats_path).exists():
                stats = np.load(stats_path)
                self.state_mean.copy_(torch.tensor(stats['state_mean'], dtype=torch.float32))
                self.state_std.copy_(torch.tensor(stats['state_std'], dtype=torch.float32))
                self.latent_mean.copy_(torch.tensor(stats['latent_mean'], dtype=torch.float32))
                self.latent_std.copy_(torch.tensor(stats['latent_std'], dtype=torch.float32))
                Log.info('Loaded channel normalization statistics from %s', stats_path)
            else:
                Log.warning('Normalization enabled but stats file %s not found; '
                            'using identity normalization.', stats_path)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=float(self.op_config['lr']),
            weight_decay=float(self.op_config.get('weight_decay', 0.01)),
            betas=tuple(self.op_config.get('betas', [0.9, 0.999])),
            eps=float(self.op_config.get('eps', 1.0e-8)),
        )
        warmup_steps = int(self.op_config.get('warmup_steps', 5000))
        max_steps = int(self.op_config.get('max_steps', 100000))

        def lr_lambda(step):
            step = step + 1
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def _normalize_state(self, s):
        return (s - self.state_mean) / self.state_std

    def _normalize_latent(self, v):
        return (v - self.latent_mean) / self.latent_std

    def step(self, batch, batch_idx):
        input, output, padding_mask,   \
            raw_end_token_mask, enc_data, enc_data_raw = batch
        '''
            padding_mask:        1 -> not padding token, 0 -> padding token
            raw_end_token_mask:  1 -> not end token,     0 -> end token
        '''
        pred_result = self.transformer(input, padding_mask, enc_data)

        # Do not care about the padding token at the begining.
        end_token_mask = (raw_end_token_mask[padding_mask > 0.5] > 0.5)
        token_output = output['token'][padding_mask > 0.5]

        gt_state = token_output[:, :self.dim_state]
        gt_latent = token_output[:, self.dim_state:]

        #################### L_end: termination loss BEGIN ####################
        end_token_logits = pred_result['is_end_token_logits']
        et_loss = F.binary_cross_entropy_with_logits(end_token_logits, end_token_mask.float(), reduction='mean')
        #################### L_end END ####################


        # Valid parent-child targets J: non-padding, non-terminal positions.
        #################### L_state / L_geo: normalized factor regression BEGIN ####################
        pred_state = pred_result['state'][end_token_mask]
        pred_latent = pred_result['coarse_latent'][end_token_mask]
        gt_state_valid = gt_state[end_token_mask]
        gt_latent_valid = gt_latent[end_token_mask]

        state_loss = F.mse_loss(self._normalize_state(pred_state),
                                self._normalize_state(gt_state_valid), reduction='mean')
        geo_loss = F.mse_loss(self._normalize_latent(pred_latent),
                              self._normalize_latent(gt_latent_valid), reduction='mean')
        #################### L_state / L_geo END ####################

        #################### L_scl: structure-controlled latent loss BEGIN ####################
        # r_hat_k is built from the predicted child state under a
        # teacher-forced pathway; the target r_k comes from the ground-truth
        # state through the same structure-controlled fusion (stop-gradient).
        loss_ratio = self.op_config['loss_ratio']
        scl_weight = float(loss_ratio.get('scl_loss', 0.0))
        if scl_weight > 0.0 and gt_state_valid.shape[0] > 0:
            batch_size, n_part = padding_mask.shape
            s_pred_full = torch.zeros(batch_size, n_part, self.dim_state,
                                      device=pred_state.device, dtype=pred_state.dtype)
            v_pred_full = torch.zeros(batch_size, n_part, self.dim_latent,
                                      device=pred_latent.device, dtype=pred_latent.dtype)
            s_pred_full[padding_mask > 0.5] = pred_result['state']
            v_pred_full[padding_mask > 0.5] = pred_result['coarse_latent']

            r_pred = self.transformer.fusion.fuse_predicted(
                s_pred_full, v_pred_full, input['fa'], pred_result['fusion_h_s'])
            r_pred = r_pred[padding_mask > 0.5][end_token_mask]
            r_target = pred_result['fusion_r'][end_token_mask].detach()
            scl_loss = F.mse_loss(r_pred, r_target, reduction='mean')
        else:
            scl_loss = torch.zeros((), device=end_token_logits.device)
        #################### L_scl END ####################

        loss = loss_ratio['et_loss'] * et_loss             \
             + loss_ratio['state_loss'] * state_loss       \
             + loss_ratio['geo_loss'] * geo_loss           \
             + scl_weight * scl_loss

        data = {
            'loss': loss,
            'et_loss': et_loss,
            'state_loss': state_loss,
            'geo_loss': geo_loss,
            'scl_loss': scl_loss,
            'gt_latent': gt_latent_valid,
            'pred_coarse_latent': pred_latent,
        }

        return data


    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.train()

        data = self.step(batch, batch_idx)

        self.manual_backward(data['loss'])
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
        optimizer.step()

        scheduler = self.lr_schedulers()
        scheduler.step()

        data['transformer_lr'] = optimizer.param_groups[0]['lr']

        del data['pred_coarse_latent']
        del data['gt_latent']

        self.log_dict(data, on_step=True, on_epoch=True, prog_bar=True)


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        data = self.step(batch, batch_idx)

        gt_latent = data['gt_latent']
        pred_latent = data['pred_coarse_latent']

        if batch_idx == 0:
            images = []
            for z in [pred_latent, gt_latent]:

                z_batch = self.e_config['z_batch']
                # import pdb; pdb.set_trace()
                batched_recon_latent = []
                for s in range(0, z.shape[0], z_batch):
                    slice_z = z[s:min(s+z_batch, z.shape[0])]
                    slice_batched_recon_latent = self.sdf.vae_model.decode(slice_z) # reconstruced triplane features
                    batched_recon_latent.append(slice_batched_recon_latent)
                batched_recon_latent = torch.cat(batched_recon_latent, dim=0)

                evaluation_count = min(self.e_config['count'], batched_recon_latent.shape[0], z.shape[0])

                screenshots = [np.random.randn(768, 1024, 3) * 255 for _ in range(evaluation_count)]
                if self.e_config['count'] > batched_recon_latent.shape[0]:
                    Log.warning('`evaluation.count` is greater than batch size. Setting to batch size')

                for batch in tqdm(range(evaluation_count), desc=f'Generating Mesh for Epoch = {batch_idx}'):
                    recon_latent = batched_recon_latent[[batch]] # ([1, D*3, resolution, resolution])
                    output_mesh = (self.e_config['eval_mesh_output_path'] / f'mesh_{self.trainer.current_epoch}_{batch}.ply').as_posix()
                    try:
                        MeshUtils.create_mesh(self.sdf, recon_latent,
                                        output_mesh, N=self.e_config['resolution'],
                                        max_batch=self.e_config['max_batch'],
                                        from_plane_features=True)
                        mesh = trimesh.load(output_mesh)
                        screenshot = MeshUtils.generate_mesh_screenshot(mesh)
                    except Exception as e:
                        Log.error(f"Error while generating mesh: {e}")
                        if "Surface level must be within volume data range" in str(e):
                            break
                        continue
                    screenshots[batch] = screenshot
                image = np.concatenate(screenshots, axis=1)
                images.append(image)
            images = np.concatenate(images, axis=0)
            try: self.logger.log_image(key="Image", images=[wandb.Image(images)])
            except Exception as e: Log.error(f"Error while logging image: {e}")
