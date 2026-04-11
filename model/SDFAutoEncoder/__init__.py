import torch
import wandb
import trimesh
import numpy as np
from tqdm import tqdm
from pathlib import Path
from einops import reduce
from torch.nn import functional as F

from .encoder import Encoder
from .decoder import Decoder
from .intermediate import VAE

import utils.mesh as MeshUtils
from utils.mylogging import Log
from utils.base import TransArticulatedBaseModule

class SDFAutoEncoder(TransArticulatedBaseModule):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs

        self.n_validation = 0

        self.e_config = configs["evaluation"]
        self.e_config['eval_mesh_output_path'] = Path(self.e_config['eval_mesh_output_path'] )
        self.e_config['eval_mesh_output_path'].mkdir(parents=True, exist_ok=True)

        # SDF Encoder Decoder Configs
        sdf_configs = configs["SdfModelSpecs"]
        hidden_dim = sdf_configs["hidden_dim"]
        latent_dim = sdf_configs["latent_dim"]
        skip_connection = sdf_configs["skip_connection"]
        tanh_act = sdf_configs["tanh_act"]
        pn_hidden = sdf_configs["pn_hidden_dim"]

        self.encoder = Encoder(c_dim=latent_dim, hidden_dim=pn_hidden, plane_resolution=64)
        self.decoder = Decoder(latent_size=latent_dim, hidden_dim=hidden_dim, skip_connection=skip_connection, tanh_act=tanh_act)

        # VAE Configs
        modulation_dim = latent_dim * 3
        latent_std = configs["latent_std"]
        hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
        self.vae_model = VAE(in_channels=latent_dim * 3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.configs["sdf_lr"])

    @staticmethod
    def _fit_screenshot_shape(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """
        Ensure screenshot has identical HxW for safe concatenation.
        Uses crop/pad without extra dependencies.
        """
        target_h, target_w = target_hw
        if image is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        img = np.asarray(image)
        if img.ndim != 3:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        if img.shape[2] > 3:
            img = img[:, :, :3]
        if img.shape[2] < 3:
            pad_c = 3 - img.shape[2]
            img = np.pad(img, ((0, 0), (0, 0), (0, pad_c)), mode='constant')

        out = np.zeros((target_h, target_w, 3), dtype=img.dtype)
        copy_h = min(target_h, img.shape[0])
        copy_w = min(target_w, img.shape[1])
        out[:copy_h, :copy_w] = img[:copy_h, :copy_w]
        return out

    def _log_image_compatible(self, image: np.ndarray, key: str = "Image") -> None:
        """
        Log image for both WandB and TensorBoard loggers.
        """
        logger_list = []
        if hasattr(self, "loggers") and self.loggers is not None:
            logger_list.extend(list(self.loggers))
        elif self.logger is not None:
            logger_list.append(self.logger)

        for logger in logger_list:
            if hasattr(logger, "log_image"):
                try:
                    logger.log_image(key=key, images=[wandb.Image(image)])
                    return
                except Exception as e:
                    Log.error(f"Error while logging image with wandb logger: {e}")

            experiment = getattr(logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "add_image"):
                try:
                    # TensorBoard supports HWC via dataformats.
                    experiment.add_image(key, image, global_step=self.global_step, dataformats='HWC')
                    return
                except Exception as e:
                    Log.error(f"Error while logging image with tensorboard logger: {e}")

        Log.warning("No compatible logger found for image logging.")

    def step(self, batch, batch_idx):
        xyz = batch['xyz']
        gt = batch['gt_sdf']
        pc = batch['point_cloud']

        # STEP 1: pointcloud -> triplane features
        plane_features = self.encoder.get_plane_features(pc)

        # STEP 2: triplane features -> z -> triplane features
        original_features = torch.cat(plane_features, dim=1)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, z = out[0], out[-1]

        # STEP 3: triplane features + query points -> SDF
        point_features = self.encoder.forward_with_plane_features(reconstructed_plane_feature, xyz)
        pred_sdf = self.decoder( torch.cat((xyz, point_features),dim=-1) )

        # STEP 4: Loss for VAE and SDF
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.configs["kld_weight"] )
        except Exception as e:
            print(e)
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        return  {"sdf_loss": sdf_loss, "vae_loss": vae_loss, "loss": loss,
                       "reconstructed_plane_feature": reconstructed_plane_feature, 'z': z}

    def training_step(self, batch, batch_idx):
        self.train()
        return_dict = self.step(batch, batch_idx)
        log_dict = {
            'sdf_loss': return_dict['sdf_loss'],
            'vae_loss': return_dict['vae_loss'],
            'loss': return_dict['loss']
        }
        self.log_dict(log_dict, prog_bar=True, enable_graph=False)

        return return_dict["loss"]

    def validation_step(self, batch, batch_idx):
        self.eval()

        return_dict = self.step(batch, batch_idx)
        log_dict = {
            'val_sdf_loss': return_dict['sdf_loss'],
            'val_vae_loss': return_dict['vae_loss'],
            'val_loss': return_dict['loss']
        }
        self.log_dict(log_dict, prog_bar=False, enable_graph=False, sync_dist=True)

        if batch_idx == 0:
            self.n_validation += 1

        # Only rank-0 should run heavy visualization / file IO in DDP.
        should_visualize = (
            batch_idx == 0
            and self.n_validation % self.e_config['vis_epoch_freq'] == 0
            and self.trainer.is_global_zero
        )
        if should_visualize:
            batched_recon_latent = return_dict["reconstructed_plane_feature"]
            evaluation_count = min(self.e_config['count'], batched_recon_latent.shape[0])
            screenshots = [None for _ in range(evaluation_count)]
            target_hw = None
            if self.e_config['count'] > batched_recon_latent.shape[0]:
                Log.warning('`evaluation.count` is greater than batch size. Setting to batch size')
            for batch in tqdm(range(evaluation_count), desc=f'Generating Mesh for Epoch = {batch_idx}'):
                recon_latent = batched_recon_latent[[batch]] # ([1, D*3, resolution, resolution])
                output_mesh = (
                    self.e_config['eval_mesh_output_path']
                    / f'mesh_rank{self.global_rank}_{batch_idx}_{batch}.ply'
                ).as_posix()
                try:
                    MeshUtils.create_mesh(self, recon_latent,
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

                if target_hw is None and isinstance(screenshot, np.ndarray) and screenshot.ndim == 3:
                    target_hw = (int(screenshot.shape[0]), int(screenshot.shape[1]))
                screenshots[batch] = screenshot

            if target_hw is None:
                target_hw = (256, 256)

            screenshots = [self._fit_screenshot_shape(img, target_hw) for img in screenshots]
            image = np.concatenate(screenshots, axis=1)

            if self.logger is not None:
                self._log_image_compatible(image, key="Image")

        return return_dict["loss"]
