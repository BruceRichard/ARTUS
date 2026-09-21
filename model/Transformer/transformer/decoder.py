import torch
from torch import nn

from .layers.decoder_layer import DecoderLayer
from .layers.post_encoder import PostEncoder
from .layers.token import MLPUnTokenizer
from .layers.gated_fusion import StructureGatedFusion

# [ARTUS]: Joint-latent context model F_theta with structure-gated latent
# fusion (Sec. 3.2) and factor-specific prediction heads D_gamma / D_s / D_v
# (Sec. 3.3). Each visible part contributes one joint latent token built from
# its structural state (16) and its geometry latent (768).

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']

        self.config = config
        self.part_structure = self.config['part_structure']
        self.m_config = self.config['transformer_model_paramerter']
        self.d_model = self.m_config['d_model']

        self.dim_state = (self.part_structure['bounding_box']
                          + self.part_structure['joint_data_origin']
                          + self.part_structure['joint_data_direction']
                          + self.part_structure['limit'])
        self.dim_latent = self.part_structure['latentcode']

        fusion_config = self.m_config.get('gated_fusion', {})
        self.fusion = StructureGatedFusion(
            d_structure=self.dim_state,
            d_geometry=self.dim_latent,
            d_hidden=self.m_config['tokenizer_hidden_dim'],
            d_model=self.d_model,
            dropout=self.m_config['tokenizer_dropout'],
            path_hidden=fusion_config.get('path_hidden_dim', 512),
            path_mode=fusion_config.get('path_mode', 'full'),
            gate_source=fusion_config.get('gate_source', 'structure'),
            fusion_mode=fusion_config.get('fusion_mode', 'gated'),
        )

        self.postencoder    = PostEncoder(dim=self.m_config['encoder_kv_dim'], d_model=self.d_model,
                                          dropout=self.m_config['post_encoder_dropout'],
                                          deepth=self.m_config['post_encoder_deepth'])

        # Factor-specific prediction heads (Sec. 3.3): termination D_gamma,
        # articulated state D_s and coarse geometry latent D_v.
        self.end_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1)
        )
        self.state_head     = MLPUnTokenizer(self.dim_state,
                                             d_hidden=self.m_config['tokenizer_hidden_dim'],
                                             d_model=self.d_model,
                                             drop_out=self.m_config['tokenizer_dropout'])
        self.geometry_head  = MLPUnTokenizer(self.dim_latent,
                                             d_hidden=self.m_config['tokenizer_hidden_dim'],
                                             d_model=self.d_model,
                                             drop_out=self.m_config['tokenizer_dropout'])

        self.layers         = nn.ModuleList([
            DecoderLayer(config)
                for _ in range(self.m_config['n_layer'])
        ])


    def generate_mask(self, n_part):

        mask = torch.ones(n_part, n_part, device=self.device, dtype=torch.int16)
        # mask = torch.tril(mask) # no need mask
        return mask

    def decode_hidden(self, tokens):
        """Decode flattened transformer representations into factor predictions."""
        end_token_logits = self.end_head(tokens).squeeze(-1)

        return {
            'is_end_token_logits': end_token_logits,
            'state': self.state_head(tokens),
            'coarse_latent': self.geometry_head(tokens),
        }

    def forward(self, input, padding_mask, enc_data):
        # input['token']: (batch, part_idx, dim_state + dim_latent)
        # input['fa']:    (batch, part_idx) parent indices
        enc_data = self.postencoder(enc_data)

        batch, n_part, _ = input['token'].size()

        # Structure-gated latent fusion: geometry enters the joint latent only
        # through the structure-derived gate (Sec. 3.2).
        fusion = self.fusion(
            input['token'][..., :self.dim_state],
            input['token'][..., self.dim_state:],
            input['fa'],
        )
        tokens = fusion['e']

        attn_mask = self.generate_mask(n_part)

        cross_attn_weight_list = []
        for idx, layer in enumerate(self.layers):
            tokens, cross_attn_weight = layer(tokens, padding_mask, attn_mask, enc_data)
            cross_attn_weight_list.append(cross_attn_weight.detach().cpu().numpy())

        # Skip padding tokens.
        hidden_tokens = tokens[padding_mask > 0.5]
        decoded = self.decode_hidden(hidden_tokens)

        result = {
            **decoded,
            'fusion_r': fusion['r'][padding_mask > 0.5],
            'fusion_h_s': fusion['h_s'],
            'cross_attn_weight_list': cross_attn_weight_list,
        }

        return result
