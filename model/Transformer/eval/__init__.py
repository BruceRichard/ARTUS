# [ArtFormer]: This file contains the class to make the inference (generation of articulated object) from text or image condition.
import os
import copy
import torch
import json
import time
import pickle

from pathlib import Path

import torch.utils
import torch.nn.functional as F
from tqdm import trange
# from rich import print
from transformers import AutoTokenizer, T5EncoderModel
from ..dataloader import TransDiffusionDataset
from .. import TransDiffusionCombineModel
from model.SDFAutoEncoder import SDFAutoEncoder

from utils import untokenize_part_info, generate_gif_toy
from utils.por_cuda import POR
from utils.mylogging import Log
from utils.z_to_mesh import GenSDFLatentCodeEvaluator
from experiments.decoart.torch_guidance import guide_part_representations

class Evaluater():
    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.device = eval_config['device']
        self.number_of_trial = self.eval_config['number_of_trial']
        self.physics_guidance_cfg = self.eval_config.get('physics_guidance_inference', {})
        self.physics_guidance_enabled = bool(self.physics_guidance_cfg.get('enabled', False))
        self._physics_query_cache = {}
        self.last_guidance_log = []
        self.structured_guidance_cfg = self.eval_config.get('structured_state_guidance', {})
        self.structured_guidance_enabled = bool(self.structured_guidance_cfg.get('enabled', False))
        self.last_structured_guidance_log = []

        Log.info("Loading model %s", TransDiffusionCombineModel)
        self.model = TransDiffusionCombineModel.load_from_checkpoint(eval_config['checkpoint_path'])
        self.model.eval()
        # self.model.diffusion.model.cond_dropout = False
        self.m_config = self.model.config

        self.z_mini_encoder = self.model.z_mini_encoder

        d_configs = self.m_config['dataset_n_dataloader']

        self.dataset = TransDiffusionDataset(dataset_path=d_configs['dataset_path'],
                cut_off=d_configs['cut_off'],
                enc_data_fieldname=d_configs['enc_data_fieldname'],
                cache_data=False)

        self.eval_output_path = Path(self.eval_config['eval_output_path']) / time.strftime("%m-%d-%I%p-%M-%S")
        os.makedirs(self.eval_output_path, exist_ok=True)

        self.start_token = copy.deepcopy(self.dataset.start_token).to(self.device)
        self.end_token = copy.deepcopy(self.dataset.end_token).to(self.device)

        Log.info("Loading model %s", T5EncoderModel)
        self.tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large', cache_dir='cache/t5_cache')
        self.text_encoder = T5EncoderModel.from_pretrained('google-t5/t5-large', cache_dir='cache/t5_cache').to(self.device)
        #TODO: check need to do self.text_encoder.eval() or not
        self.text_encoder.eval()
        self.t5_max_sentence_length = self.eval_config['t5_max_sentence_length']

        # self.equal_part_threshold = self.eval_config['equal_part_threshold']

        # self.latentcode_evaluator = LatentCodeEvaluator(Path(self.dataset.get_onet_ckpt_path()), 100000, 16, self.device)

        Log.info("Loading model %s", SDFAutoEncoder)
        self.gensdf_config = self.eval_config['gensdf_latentcode_evaluator']
        # self.gensdf_config['gensdf_model_path'] = self.dataset.get_best_sdf_ckpt_path()
        self.sdf = self.model.sdf # SDFAutoEncoder.load_from_checkpoint(self.gensdf_config['gensdf_model_path'])
        self.sdf.eval()
        self.latentcode_evaluator = GenSDFLatentCodeEvaluator(self.sdf, eval_mesh_output_path=self.eval_output_path,
                                                             resolution=self.gensdf_config['resolution'],
                                                             max_batch=self.gensdf_config['max_batch'],
                                                             device=self.device)

    @staticmethod
    def _split_token_fields(tokens):
        return {
            'bbox_size': tokens[:, 0:3],
            'bbox_center': tokens[:, 3:6],
            'joint_origin': tokens[:, 6:9],
            'joint_axis': tokens[:, 9:12],
            'joint_limit': tokens[:, 12:16],
        }

    def _get_physics_query_points(self, surface_samples, device):
        key = int(surface_samples)
        if key <= 0:
            key = 1
        if key not in self._physics_query_cache:
            g = torch.Generator(device='cpu')
            g.manual_seed(2026 + key)
            query_points = torch.rand((key, 3), generator=g) * 2.0 - 1.0
            self._physics_query_cache[key] = query_points
        return self._physics_query_cache[key].to(device)

    def _build_scene_proxy_from_parent(self, parent_center, parent_size):
        min_bbox_size = 1.0e-3
        parent_size = torch.clamp(parent_size.abs(), min=min_bbox_size)
        half = parent_size * 0.5

        ex = torch.tensor([1.0, 0.0, 0.0], device=parent_center.device, dtype=parent_center.dtype).view(1, 3)
        ey = torch.tensor([0.0, 1.0, 0.0], device=parent_center.device, dtype=parent_center.dtype).view(1, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=parent_center.device, dtype=parent_center.dtype).view(1, 3)

        px = parent_center + half[:, 0:1] * ex
        nx = parent_center - half[:, 0:1] * ex
        py = parent_center + half[:, 1:2] * ey
        ny = parent_center - half[:, 1:2] * ey
        pz = parent_center + half[:, 2:3] * ez
        nz = parent_center - half[:, 2:3] * ez
        scene_points = torch.stack((px, nx, py, ny, pz, nz), dim=1)

        npx = ex.expand_as(px)
        nnx = (-ex).expand_as(nx)
        npy = ey.expand_as(py)
        nny = (-ey).expand_as(ny)
        npz = ez.expand_as(pz)
        nnz = (-ez).expand_as(nz)
        scene_normals = torch.stack((npx, nnx, npy, nny, npz, nnz), dim=1)

        area_x = parent_size[:, 1] * parent_size[:, 2]
        area_y = parent_size[:, 0] * parent_size[:, 2]
        area_z = parent_size[:, 0] * parent_size[:, 1]
        area = torch.stack((area_x, area_x, area_y, area_y, area_z, area_z), dim=1)
        volume = parent_size.prod(dim=-1, keepdim=True).sqrt()
        robustness = area * volume
        robustness = robustness / (robustness.sum(dim=1, keepdim=True) + 1e-6)
        return scene_points, scene_normals, robustness

    def _build_physics_guidance_context(self, exist_node):
        if not self.physics_guidance_enabled:
            return {'enabled': False}

        cfg = self.physics_guidance_cfg
        token = exist_node['token'][:, :16]
        fa = exist_node['fa'].long()
        batch = token.shape[0]
        if batch == 0:
            return {'enabled': False}

        child = self._split_token_fields(token)
        parent_token = token[fa.clamp(min=0, max=batch - 1)]
        parent = self._split_token_fields(parent_token)

        scene_points, scene_normals, scene_robustness = self._build_scene_proxy_from_parent(
            parent['bbox_center'],
            parent['bbox_size'],
        )

        surface_samples = int(cfg.get('surface_samples', 256))
        topk_neighbors = int(cfg.get('topk_neighbors', 16))
        canonical_query_points = self._get_physics_query_points(surface_samples, token.device)
        canonical_query_points = canonical_query_points.unsqueeze(0).expand(batch, -1, -1)

        child_half = torch.clamp(child['bbox_size'].abs(), min=1.0e-3) * 0.5
        world_query_points = child['bbox_center'].unsqueeze(1) + canonical_query_points * child_half.unsqueeze(1)

        pair_dist = torch.cdist(scene_points, world_query_points)
        k = min(max(1, topk_neighbors), pair_dist.shape[-1])
        knn_dists, knn_indices = torch.topk(pair_dist, k=k, dim=-1, largest=False)

        active_mask = (torch.arange(batch, device=token.device) > 0).float()

        return {
            'enabled': True,
            'canonical_query_points': canonical_query_points,
            'scene_normals': scene_normals,
            'scene_robustness': scene_robustness,
            'knn_indices': knn_indices,
            'knn_dists': knn_dists,
            'active_mask': active_mask,
            'dmax': float(cfg.get('dmax', 0.10)),
            'surface_temperature': float(cfg.get('surface_temperature', 0.05)),
            'normal_weight': float(cfg.get('normal_weight', 0.0)),
        }

    def _physics_guidance_cost(self, x_t, t, cond, guidance_ctx):
        if guidance_ctx is None or not guidance_ctx.get('enabled', False):
            return torch.zeros((), device=x_t.device, dtype=x_t.dtype)

        query_points = guidance_ctx['canonical_query_points']
        if query_points.shape[0] != x_t.shape[0]:
            b = min(query_points.shape[0], x_t.shape[0])
            x_t = x_t[:b]
            query_points = query_points[:b]
            knn_indices = guidance_ctx['knn_indices'][:b]
            knn_dists = guidance_ctx['knn_dists'][:b]
            scene_normals = guidance_ctx['scene_normals'][:b]
            scene_robustness = guidance_ctx['scene_robustness'][:b]
            active_mask = guidance_ctx['active_mask'][:b]
        else:
            knn_indices = guidance_ctx['knn_indices']
            knn_dists = guidance_ctx['knn_dists']
            scene_normals = guidance_ctx['scene_normals']
            scene_robustness = guidance_ctx['scene_robustness']
            active_mask = guidance_ctx['active_mask']

        plane_features = self.sdf.vae_model.decode(x_t)
        point_features = self.sdf.encoder.forward_with_plane_features(plane_features, query_points)
        sdf_values = self.sdf.decoder(torch.cat((query_points, point_features), dim=-1)).squeeze(-1)

        temp = max(float(guidance_ctx.get('surface_temperature', 0.05)), 1e-6)
        surface_scores = torch.exp(-torch.abs(sdf_values) / temp)
        surface_probs = surface_scores / (surface_scores.sum(dim=-1, keepdim=True) + 1e-6)

        scene_count = scene_robustness.shape[1]
        surface_probs_expand = surface_probs.unsqueeze(1).expand(-1, scene_count, -1)
        prob_knn = torch.gather(surface_probs_expand, dim=2, index=knn_indices)

        dmax = max(float(guidance_ctx.get('dmax', 0.10)), 1e-6)
        contact_term = torch.exp(-knn_dists / dmax)

        alignment = torch.ones_like(contact_term)
        normal_weight = float(guidance_ctx.get('normal_weight', 0.0))
        if normal_weight > 0.0 and torch.is_grad_enabled():
            try:
                q_for_normal = query_points.detach().clone().requires_grad_(True)
                pf_for_normal = self.sdf.encoder.forward_with_plane_features(plane_features, q_for_normal)
                sdf_for_normal = self.sdf.decoder(torch.cat((q_for_normal, pf_for_normal), dim=-1)).squeeze(-1)
                grad_query = torch.autograd.grad(
                    sdf_for_normal.sum(),
                    q_for_normal,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                obj_normals = F.normalize(grad_query, dim=-1)
                idx_expand = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
                obj_normals_knn = torch.gather(
                    obj_normals.unsqueeze(1).expand(-1, scene_count, -1, 3),
                    dim=2,
                    index=idx_expand,
                )
                scene_normals_expand = scene_normals.unsqueeze(2)
                alignment = torch.abs((scene_normals_expand * obj_normals_knn).sum(dim=-1))
                alignment = (1.0 - normal_weight) + normal_weight * alignment
            except RuntimeError:
                alignment = torch.ones_like(contact_term)

        robust_term = scene_robustness.unsqueeze(-1)
        term = alignment * contact_term * robust_term * prob_knn

        per_sample = term.mean(dim=(-1, -2))
        per_sample = per_sample * active_mask
        denom = active_mask.sum().clamp_min(1.0)
        j_cost = -(per_sample.sum() / denom)
        return j_cost

    def encode_text(self, text):
        input_ids = self.tokenizer([text], return_tensors="pt", padding='max_length',
                                    max_length=self.t5_max_sentence_length).input_ids
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
        encoded_text = outputs.last_hidden_state.detach()
        return encoded_text

    def generate_non_padding_mask(self, len):
        return torch.ones(1, len).to(self.device)

    def is_end_token(self, token):
        length = token.size(0)
        difference = torch.nn.functional.mse_loss(token[:length], self.end_token[:length])
        Log.info('    - Difference with end token: %s', difference.item())
        return difference < self.equal_part_threshold

    def inference_from_text(self, text, enc_data=None, need_mesh=True):
        Log.info('[1] Inference text: %s', len(text))
        if enc_data is None:
            encoded_text = self.encode_text(text)
        else:
            encoded_text = enc_data.unsqueeze(0).to(self.device)

        exist_node = {
            'fa': torch.tensor([0]).to(self.device),
            'token': copy.deepcopy((self.start_token[:16])).unsqueeze(0).to(self.device),
            'text_hat': torch.zeros((64)).unsqueeze(0).to(self.device),
            'z_hat': torch.zeros((4, 768)).unsqueeze(0).to(self.device),
            'latent': torch.zeros((768)).unsqueeze(0).to(self.device)
        }
        round = 1
        inference_seed = int(torch.initial_seed() % (2 ** 31 - 1))
        max_generation_rounds = int(
            self.eval_config.get(
                'max_generation_rounds',
                max(2, int(self.dataset.max_count_token) + 1),
            )
        )
        Log.info('[2] Generate nodes')
        atten_weights_list = []
        self.last_structured_guidance_log = []

        use_shape_prior = True
        while round <= max_generation_rounds:
            current_length = exist_node['token'].size(0)
            Log.info('   - Generate nodes round: %s, part count: %s', round, exist_node['token'].size(0))
            with torch.no_grad():
                # input: (batch, seq, xxx) ---> (batch|seq, xxx) base on `padding_mask`, the dimension of batch & seq are merged.
                # batch=1 for evaluation.
                output = self.model.transformer({
                                'fa': exist_node['fa'].unsqueeze(0),        # batched.
                                'token': torch.cat((exist_node['token'], exist_node['text_hat']), dim=1).unsqueeze(0),
                            },
                            self.generate_non_padding_mask(current_length),
                            encoded_text) # unbatched.
            atten_weights_list.append(output['cross_attn_weight_list'])
            # Solve End Token.
            # True -> not end token, False -> end token
            end_token_mask = output['is_end_token_logits'] > 0
            Log.info('   - Check end token: %s', output['is_end_token_logits'])
            Log.info('   - Check end token mask: %s', end_token_mask)
            if not torch.any(end_token_mask):
                break

            fa_idx = torch.arange(end_token_mask.shape[0], device=self.device)
            fa_idx = fa_idx[end_token_mask]
            condition = output['condition']
            if self.structured_guidance_enabled:
                candidate_hidden = output['hidden_tokens'][end_token_mask]
                guidance_result = guide_part_representations(
                    candidate_hidden,
                    self.model.transformer.decode_hidden,
                    exist_node['token'][:, :16],
                    exist_node['fa'],
                    fa_idx,
                    self.structured_guidance_cfg,
                    seed=(
                        int(self.structured_guidance_cfg.get('seed', 2026))
                        + inference_seed
                        + round * 1009
                    ),
                )
                articulated_info = guidance_result.decoded['articulated_info']
                condition = guidance_result.decoded['condition']
                round_log = dict(guidance_result.log)
                round_log['round'] = round
                round_log['candidate_count'] = int(candidate_hidden.shape[0])
                self.last_structured_guidance_log.append(round_log)
                Log.info(
                    "   - Structured guidance: active=%s, J %.6f -> %.6f, routes=%s",
                    round_log['active_count'],
                    round_log.get('j_pre', 0.0),
                    round_log.get('j_post', 0.0),
                    round_log.get('route_counts', {}),
                )
            else:
                articulated_info = output['articulated_info'][end_token_mask]
                if isinstance(condition, dict):
                    condition = {
                        key: value[end_token_mask]
                        for key, value in condition.items()
                    }
                else:
                    condition = condition[end_token_mask]

            if isinstance(condition, dict):
                pred_text_hat = condition['text_hat'] # torch.Size([1, 64])
                pred_z_logits = condition['z_logits'] # torch.Size([1, 4, 128])
                q_z, _KL, _perplexity, _logits = self.z_mini_encoder.forward_with_logits_or_x(tau=0.5, logits=pred_z_logits)
                latent_code = None
            else:
                latent_code = condition # torch.Size([1, 768])
                pred_text_hat = torch.zeros((64)).unsqueeze(0).to(self.device)
                q_z = None
                use_shape_prior = False

            result = articulated_info

            exist_node['fa'] = torch.cat((exist_node['fa'], fa_idx), dim=0)
            exist_node['token'] = torch.cat((exist_node['token'], result), dim=0)

            if pred_text_hat is not None:
                exist_node['text_hat'] = torch.cat((exist_node['text_hat'], pred_text_hat), dim=0)
            if q_z is not None:
                exist_node['z_hat'] = torch.cat((exist_node['z_hat'], q_z), dim=0)
            if latent_code is not None:
                exist_node['latent'] = torch.cat((exist_node['latent'], latent_code), dim=0)

            round += 1
        else:
            Log.warning(
                "Reached max_generation_rounds=%s before every branch emitted an end token.",
                max_generation_rounds,
            )

        Log.info('[3] reconstruct latent code with condition')
        if use_shape_prior:
            guidance_ctx = self._build_physics_guidance_context(exist_node)
            latent = self.model.diffusion.model.generate_conditional(
                {
                    'z_hat': exist_node['z_hat'],
                    'text': exist_node['text_hat'],
                },
                guidance_fn=self._physics_guidance_cost if self.physics_guidance_enabled else None,
                guidance_ctx=guidance_ctx,
                guidance_weight=float(self.physics_guidance_cfg.get('weight_gamma', 0.1)),
                guidance_interval=int(self.physics_guidance_cfg.get('interval_n', 2)),
                guidance_enabled=self.physics_guidance_enabled,
                guidance_log=bool(self.physics_guidance_cfg.get('log_cost', True)),
            )
            self.last_guidance_log = list(getattr(self.model.diffusion.model, 'last_guidance_log', []))
            if self.last_guidance_log:
                first_item = self.last_guidance_log[0]
                last_item = self.last_guidance_log[-1]
                Log.info(
                    "[Physics Guidance] steps=%s, first(t=%s, %.6f->%.6f), last(t=%s, %.6f->%.6f)",
                    len(self.last_guidance_log),
                    first_item['t'],
                    first_item['j_before'],
                    first_item['j_after'],
                    last_item['t'],
                    last_item['j_before'],
                    last_item['j_after'],
                )
            exist_node['token'] = torch.cat((exist_node['token'], latent), dim=-1)
        else:
            exist_node['token'] = torch.cat((exist_node['token'], exist_node['latent']), dim=-1)

        processed_nodes = []
        Log.info('[4] Generate mesh')

        for idx in trange(exist_node['fa'].shape[0], desc='   - Generate mesh'):
            dfn_fa = exist_node['fa'][idx].item()
            token  = exist_node['token'][idx].cpu().tolist()
            processed_node = {
                'dfn': idx,
                'dfn_fa': dfn_fa,
            }
            part_info = untokenize_part_info(token)

            z = torch.tensor(part_info['latent_code']).to(self.device)
            if need_mesh:
                part_info['mesh'] = self.latentcode_evaluator.generate_mesh(z.unsqueeze(0))
            # import pdb; pdb.set_trace()
            part_info['z'] = z
            # raw_points_sdf, rho = self.latentcode_evaluator.generate_uniform_point_cloud_inside_mesh(z.unsqueeze(0))
            # part_info['points'], part_info['rho'] = fit_into_bounding_box(raw_points_sdf, rho, part_info['bbx'])

            processed_node.update(part_info)
            processed_nodes.append(processed_node)

        # import pdb; pdb.set_trace()

        # We do not want start token.
        return processed_nodes[1:], atten_weights_list

    @staticmethod
    def _detach_for_pickle(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: Evaluater._detach_for_pickle(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Evaluater._detach_for_pickle(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(Evaluater._detach_for_pickle(v) for v in obj)
        return obj

    def inference_to_output_path(self, text, output_path, enc_data=None, blender_generated_gif=False):
        output_path.mkdir(exist_ok=True, parents=True)
        processed_nodes, atten_weights_list = self.inference_from_text(text, enc_data)

        # for debug only.
        # processed_nodes, atten_weights_list = pickle.load(open('/ssd1/dengzhidong/.sym/final/ArtFormer/elog/Final_OP1_05-27-01PM-29-48/StorageFurniture_45243_1/0/output.dat', 'rb')), None

        if self.eval_config.get('save_output_dat', True):
            output_data_path = output_path / "output.dat"
            with open(output_data_path, 'wb') as f:
                f.write(pickle.dumps(self._detach_for_pickle(processed_nodes)))
            Log.info("[Write] %s", output_data_path)

            if self.last_guidance_log:
                guidance_log_path = output_path / "physics_guidance_log.json"
                guidance_log_path.write_text(json.dumps(self.last_guidance_log, indent=2))
                Log.info("[Write] %s", guidance_log_path)
            if self.last_structured_guidance_log:
                structured_log_path = output_path / "structured_guidance_log.json"
                structured_log_path.write_text(json.dumps(self.last_structured_guidance_log, indent=2))
                Log.info("[Write] %s", structured_log_path)

        output_tex_path = output_path / "input.txt"
        output_tex_path.write_text(text)
        Log.info("[Write] %s", output_tex_path)

        output_gif_path = output_path / "gif"
        generate_gif_toy(processed_nodes, output_gif_path, bar_prompt="   - Generate Frames", blender_generated_gif=blender_generated_gif)
        Log.info("[Write] %s", output_gif_path)

        # output_temp_path : Path = output_path / "temp"
        # output_temp_path.mkdir(exist_ok=True, parents=True)
        # Log.info("[Write] %s", output_temp_path)

        # for ratio in [0, 0.5, 1]:
        #     visualize_obj_high_q(processed_nodes, output_temp_path / str(ratio), output_path / str(ratio), ratio)

        # return atten_weights_list

    def inference_dat_file_only(self, text, output_dat_path, enc_data=None):
        processed_nodes, atten_weights_list = self.inference_from_text(text, need_mesh=False, enc_data=enc_data)
        with open(output_dat_path, 'wb') as f:
            f.write(pickle.dumps(processed_nodes))

    def inference(self, text):
        number_of_trial = self.number_of_trial
        list_processed_nodes = [None] * number_of_trial
        for trial in trange(number_of_trial, desc="Doing trial"):
            processed_nodes = self.inference_from_text(text)
            list_processed_nodes[trial] = {
                'data': processed_nodes,
                'rate': POR(processed_nodes, n_sample=8192),
            }
            rate = list_processed_nodes[trial]['rate']
            output_gif_path = (Path(self.eval_output_path) / f'output_{trial}_{rate}.gif')
            Log.info('[4] Generate Gif: %s', output_gif_path.as_posix())

            generate_gif_toy(processed_nodes, output_gif_path,
                            bar_prompt="   - Generate Frames")
            Log.info('[5] Done')

        output_json_path = (Path(self.eval_output_path) / 'output.json')
        output_json_path.write_text('{"text": "' + text + '"}')

        output_data_path = (Path(self.eval_output_path) / 'output.data')
        with open(output_data_path, 'wb') as f:
            f.write(pickle.dumps(list_processed_nodes))
        Log.info("Saved data checkpoint %s.", output_data_path.as_posix())
