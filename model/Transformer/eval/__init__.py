# [ARTUS]: Autoregressive generation with structure-controlled joint latent
# refinement (paper Sec. 3.3, Algorithm 1).
#
# Each round re-encodes the Existing Articulation Pathway from the same
# snapshot. Every active parent either terminates (and retires from
# expansion) or predicts a child: the articulated state s_k comes from D_s,
# the coarse geometry latent v_tilde_k from D_v, and the frozen
# text-conditioned refiner R_phi enhances v_tilde_k into v_hat_k through
# partial-noise refinement (Eq. 5-7). The enhanced latent is fed back into
# the pathway so that subsequent predictions condition on the realized
# geometry of preceding parts.
import os
import copy
import torch
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

class Evaluater():
    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.device = eval_config['device']
        self.number_of_trial = self.eval_config['number_of_trial']

        # [ARTUS] Partial-noise refinement and pathway feedback (Sec. 3.3).
        self.refinement_cfg = self.eval_config.get('latent_refinement', {})
        self.refinement_enabled = bool(self.refinement_cfg.get('enabled', True))
        self.rho_star = int(self.refinement_cfg.get('rho_star', 250))
        self.sampler_steps = int(self.refinement_cfg.get('sampler_steps', 50))
        self.feedback = bool(self.refinement_cfg.get('feedback', True))

        Log.info("Loading model %s", TransDiffusionCombineModel)
        self.model = TransDiffusionCombineModel.load_from_checkpoint(eval_config['checkpoint_path'])
        self.model.eval()
        self.m_config = self.model.config

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
        self.text_encoder.eval()
        self.t5_max_sentence_length = self.eval_config['t5_max_sentence_length']

        Log.info("Loading model %s", SDFAutoEncoder)
        self.gensdf_config = self.eval_config['gensdf_latentcode_evaluator']
        self.sdf = self.model.sdf
        self.sdf.eval()
        self.latentcode_evaluator = GenSDFLatentCodeEvaluator(self.sdf, eval_mesh_output_path=self.eval_output_path,
                                                             resolution=self.gensdf_config['resolution'],
                                                             max_batch=self.gensdf_config['max_batch'],
                                                             device=self.device)

    def encode_text(self, text, max_length=None):
        max_length = max_length or self.t5_max_sentence_length
        input_ids = self.tokenizer([text], return_tensors="pt", padding='max_length',
                                    max_length=max_length, truncation=True).input_ids
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
        encoded_text = outputs.last_hidden_state.detach()
        return encoded_text

    def generate_non_padding_mask(self, len):
        return torch.ones(1, len).to(self.device)

    @torch.no_grad()
    def _refine_coarse_latent(self, coarse_latent, text):
        """[ARTUS] R_phi: enhance the coarse geometry latent within the frozen
        pretrained geometry space (paper Eq. 5-7).

        The refiner is text-conditioned: the compressed text feature comes
        from the frozen text-condition encoder of the diffusion stage, and no
        geometry sketch is supplied, so the coarse prediction itself carries
        the part-specific geometric information into the prior.
        """
        if not self.refinement_enabled or self.rho_star <= 0:
            return coarse_latent

        diff_config = self.model.diffusion.diff_config
        text_encoding = self.encode_text(text, max_length=int(
            diff_config['text_condition']['padding_length']))
        _, text_hat = self.model.diffusion.text_mini_encoder(text_encoding)

        batch = coarse_latent.shape[0]
        cond = {
            'z_hat': torch.zeros(
                batch,
                int(diff_config['gsemb_latent_dim']),
                int(diff_config['dim_latentcode']),
                device=coarse_latent.device),
            'text': text_hat.expand(batch, -1),
        }
        return self.model.diffusion.model.refine(
            coarse_latent,
            start_step=self.rho_star,
            cond=cond,
            sampler_steps=self.sampler_steps,
        )

    def inference_from_text(self, text, enc_data=None, need_mesh=True):
        Log.info('[1] Inference text: %s', len(text))
        if enc_data is None:
            encoded_text = self.encode_text(text)
        else:
            encoded_text = enc_data.unsqueeze(0).to(self.device)

        # [ARTUS] The pathway token of each part stores its structural state
        # (16) and its pathway geometry latent (768). The pathway latent is the
        # refined latent when feedback is enabled, otherwise the coarse one.
        dim_state = self.model.dim_state
        dim_latent = self.model.dim_latent

        exist_node = {
            'fa': torch.tensor([0]).to(self.device),
            'token': torch.cat((
                self.start_token[:dim_state],
                torch.zeros(dim_latent, device=self.device),
            )).unsqueeze(0).to(self.device),
            # Realized geometry latent used for mesh decoding (the refined
            # latent whenever refinement is enabled).
            'realized': torch.zeros((1, dim_latent), device=self.device),
        }
        # Active parents can still expand; a parent that predicts END is
        # retained in the pathway but retires from expansion (Algorithm 1).
        active = torch.ones(1, dtype=torch.bool, device=self.device)

        round = 1
        max_generation_rounds = int(
            self.eval_config.get(
                'max_generation_rounds',
                max(2, int(self.dataset.max_count_token) + 1),
            )
        )
        max_nodes = int(self.eval_config.get('max_nodes', int(self.dataset.max_count_token)))
        truncated = False
        Log.info('[2] Generate nodes')
        atten_weights_list = []

        while round <= max_generation_rounds:
            current_length = exist_node['token'].size(0)
            Log.info('   - Generate nodes round: %s, part count: %s', round, current_length)
            with torch.no_grad():
                # input: (batch, seq, xxx) ---> (batch|seq, xxx) base on `padding_mask`, the dimension of batch & seq are merged.
                # batch=1 for evaluation.
                output = self.model.transformer({
                                'fa': exist_node['fa'].unsqueeze(0),        # batched.
                                'token': exist_node['token'].unsqueeze(0),
                            },
                            self.generate_non_padding_mask(current_length),
                            encoded_text) # unbatched.
            atten_weights_list.append(output['cross_attn_weight_list'])

            # Termination decisions of active parents. True -> expands, False -> END.
            end_token_mask = (output['is_end_token_logits'] > 0) & active
            Log.info('   - Check end token: %s', output['is_end_token_logits'])
            Log.info('   - Check end token mask: %s', end_token_mask)
            if not torch.any(end_token_mask):
                break

            fa_idx = torch.arange(end_token_mask.shape[0], device=self.device)
            fa_idx = fa_idx[end_token_mask]
            child_state = output['state'][end_token_mask]
            coarse_latent = output['coarse_latent'][end_token_mask]

            # Node cap: a cap-triggered exit is distinguished from all parents
            # predicting END.
            if current_length + child_state.shape[0] > max_nodes:
                Log.warning('Node cap %s reached; marking truncation.', max_nodes)
                truncated = True
                break

            # [ARTUS] Partial-noise refinement with the frozen prior, then
            # feed the realized geometry back into the pathway.
            refined_latent = self._refine_coarse_latent(coarse_latent, text)
            pathway_latent = refined_latent if self.feedback else coarse_latent

            child_token = torch.cat((child_state, pathway_latent), dim=-1)

            exist_node['fa'] = torch.cat((exist_node['fa'], fa_idx), dim=0)
            exist_node['token'] = torch.cat((exist_node['token'], child_token), dim=0)
            exist_node['realized'] = torch.cat((exist_node['realized'], refined_latent), dim=0)

            # Parents that expanded stay active; new children become active;
            # parents that predicted END retire.
            active = torch.cat((
                end_token_mask,
                torch.ones(child_token.shape[0], dtype=torch.bool, device=self.device),
            ), dim=0)

            round += 1
        else:
            truncated = True
            Log.warning(
                "Reached max_generation_rounds=%s before every branch emitted an end token.",
                max_generation_rounds,
            )
        if truncated:
            Log.warning('[ARTUS] Generation ended by truncation, not by termination decisions.')

        Log.info('[3] assemble part tokens with realized geometry latents')
        full_token = torch.cat((exist_node['token'][:, :dim_state], exist_node['realized']), dim=-1)

        processed_nodes = []
        Log.info('[4] Generate mesh')

        for idx in trange(exist_node['fa'].shape[0], desc='   - Generate mesh'):
            dfn_fa = exist_node['fa'][idx].item()
            token  = full_token[idx].cpu().tolist()
            processed_node = {
                'dfn': idx,
                'dfn_fa': dfn_fa,
            }
            part_info = untokenize_part_info(token)

            z = torch.tensor(part_info['latent_code']).to(self.device)
            if need_mesh:
                part_info['mesh'] = self.latentcode_evaluator.generate_mesh(z.unsqueeze(0))
            part_info['z'] = z

            processed_node.update(part_info)
            processed_nodes.append(processed_node)

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

        if self.eval_config.get('save_output_dat', True):
            output_data_path = output_path / "output.dat"
            with open(output_data_path, 'wb') as f:
                f.write(pickle.dumps(self._detach_for_pickle(processed_nodes)))
            Log.info("[Write] %s", output_data_path)

        output_tex_path = output_path / "input.txt"
        output_tex_path.write_text(text)
        Log.info("[Write] %s", output_tex_path)

        output_gif_path = output_path / "gif"
        generate_gif_toy(processed_nodes, output_gif_path, bar_prompt="   - Generate Frames", blender_generated_gif=blender_generated_gif)
        Log.info("[Write] %s", output_gif_path)

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
