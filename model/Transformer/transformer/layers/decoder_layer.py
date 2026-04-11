import torch
from torch import nn
import torch.nn.functional as F

from .feedforward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        m_config = config['transformer_model_paramerter']
        self.n_head = m_config['n_head']
        self.d_model = m_config['d_model']
        self.decoder_dropout = m_config['decoder_dropout']
        self.self_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_head,
                                                    dropout=self.decoder_dropout, batch_first=True)

        self.cross_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_head,
                                                     dropout=self.decoder_dropout, batch_first=True)

        self.ffn = PositionWiseFeedForward(d_model=self.d_model, hidden_dim=m_config['ffn_hidden_dim'],
                                           dropout=m_config['ffn_dropout'])

        self.dropout_1 = nn.Dropout(self.decoder_dropout)
        self.norm_1 = nn.LayerNorm(self.d_model)

        self.dropout_2 = nn.Dropout(self.decoder_dropout)
        self.norm_2 = nn.LayerNorm(self.d_model)

        self.dropout_3 = nn.Dropout(self.decoder_dropout)
        self.norm_3 = nn.LayerNorm(self.d_model)

        vggt_cfg = m_config.get('vggt_enhance', {})
        self.vggt_enabled = bool(vggt_cfg.get('enabled', False))
        self.vggt_salient_ratio = float(vggt_cfg.get('salient_ratio', 0.25))
        self.vggt_refresh_alpha = float(vggt_cfg.get('refresh_alpha', 0.35))
        self.vggt_collapse_margin = float(vggt_cfg.get('collapse_margin', 0.80))

    def _apply_anchor_salient_refresh(self, x, before_x, attn_weight, key_padding_mask):
        if not self.vggt_enabled:
            return x

        refreshed_x = x.clone()
        batch_size = x.shape[0]

        for b in range(batch_size):
            valid_idx = torch.where(key_padding_mask[b] > 0.5)[0]
            if valid_idx.numel() <= 1:
                continue

            # FastVGGT-style anchor: preserve first valid token as stable reference.
            anchor_idx = valid_idx[0]
            refreshed_x[b, anchor_idx] = (
                (1.0 - self.vggt_refresh_alpha) * x[b, anchor_idx]
                + self.vggt_refresh_alpha * before_x[b, anchor_idx]
            )

            candidate_idx = valid_idx[1:]
            if candidate_idx.numel() == 0:
                continue

            keep_count = max(1, int(candidate_idx.numel() * self.vggt_salient_ratio))
            to_anchor = attn_weight[b, candidate_idx, anchor_idx]
            from_anchor = attn_weight[b, anchor_idx, candidate_idx]
            saliency = to_anchor + from_anchor
            salient_local = torch.topk(saliency, k=min(keep_count, saliency.numel()), dim=0).indices
            salient_idx = candidate_idx[salient_local]

            refreshed_x[b, salient_idx] = (
                (1.0 - self.vggt_refresh_alpha) * x[b, salient_idx]
                + self.vggt_refresh_alpha * before_x[b, salient_idx]
            )

        return refreshed_x

    def _calc_attention_collapse_loss(self, attn_weight, key_padding_mask):
        if not self.vggt_enabled:
            return torch.zeros((), device=attn_weight.device)

        loss_terms = []
        batch_size = attn_weight.shape[0]
        for b in range(batch_size):
            valid_idx = torch.where(key_padding_mask[b] > 0.5)[0]
            # Keep anchor token, compare only non-anchor token attention diversity.
            if valid_idx.numel() <= 2:
                continue
            valid_idx = valid_idx[1:]

            attn_valid = attn_weight[b, valid_idx][:, valid_idx]
            attn_valid = F.normalize(attn_valid + 1e-8, p=2, dim=-1)
            sim = torch.matmul(attn_valid, attn_valid.transpose(0, 1))
            eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
            off_diag = sim[~eye]
            if off_diag.numel() == 0:
                continue
            loss_terms.append(F.relu(off_diag - self.vggt_collapse_margin).mean())

        if len(loss_terms) == 0:
            return torch.zeros((), device=attn_weight.device)
        return torch.stack(loss_terms).mean()

    def forward(self, x, key_padding_mask, attn_mask, enc_data):
        cross_attn_weight = None
        vggt_reg_loss = torch.zeros((), device=x.device)

        if True:
            before_x = x
            x, attn_weight = self.self_attention(x, x, x,
                                    key_padding_mask=(key_padding_mask == 0),
                                    attn_mask=(attn_mask == 0))
            x = self._apply_anchor_salient_refresh(x, before_x, attn_weight, key_padding_mask)
            vggt_reg_loss = self._calc_attention_collapse_loss(attn_weight, key_padding_mask)

            x = self.dropout_1(x)
            x = self.norm_1(x + before_x)

        if enc_data is not None:
            before_x = x
            # shape of x: (batch, query_len, d_model)
            x, cross_attn_weight = self.cross_attention(x, enc_data, enc_data)

            x = self.dropout_2(x)
            x = self.norm_2(x + before_x)

        if True:
            before_x = x
            x = self.ffn(x)

            x = self.dropout_3(x)
            x = self.norm_3(x + before_x)

        return x, cross_attn_weight, vggt_reg_loss
