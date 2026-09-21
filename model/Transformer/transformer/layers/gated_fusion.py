# [ARTUS]: Structure-gated latent fusion (paper Sec. 3.2).
#
# Factor-specific encoders E_s / E_v, the articulation-path encoder G_s and the
# structure-derived gate that admits geometric features into the joint latent:
#
#   u_i = h_i^s + p_i                                   (structural representation)
#   g_i = sigmoid(W_g u_i + b_g)                        (structure-derived gate)
#   r_i = u_i + g_i ⊙ h_i^v                             (pre-projection fusion)
#   e_i = W_f LN(r_i) + b_f                             (joint latent token)
#
# `path_mode`, `gate_source` and `fusion_mode` expose the controlled
# latent-construction ablations of paper Table 2(a).

import torch
import torch.nn.functional as F
from torch import nn


class FactorEncoder(nn.Module):
    """Two-layer MLP with Mish activation: raw factor -> hidden -> d_model.

    Used as the structural encoder E_s (d_in=16) and the geometry encoder
    E_v (d_in=768).
    """

    def __init__(self, d_in, d_hidden, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


class StructuralPathEncoder(nn.Module):
    """G_s: bidirectional GRU over the root-to-part chain of structural features.

    For part i the ordered path is path_pi(i) = (r, ..., pi(i), i), obtained by
    repeatedly applying the parent map. The final bidirectional states are
    concatenated and projected to the model width, yielding the
    articulation-path summary p_i.
    """

    def __init__(self, d_model, d_hidden=512):
        super().__init__()
        self.gru = nn.GRU(d_model, d_hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d_hidden, d_model)

    @staticmethod
    def build_ancestor_chains(fa):
        """Vectorized root-ward ancestor chains for every node.

        Args:
            fa: (B, N) long parent indices; fa[:, 0] == 0 is the root self-loop.

        Returns:
            chain:   (B, N, N) long; chain[b, i, d] is the d-th ancestor of
                     node i (d=0 is i itself, increasing d walks to the root;
                     entries at/after the root repeat the root index).
            lengths: (B, N) long; depth(i) + 1.
        """
        B, N = fa.shape
        idx = torch.arange(N, device=fa.device).unsqueeze(0).expand(B, N).contiguous()
        chain = [idx]
        cur = idx
        for _ in range(N - 1):
            cur = torch.gather(fa, 1, cur)
            chain.append(cur)
        chain = torch.stack(chain, dim=-1)  # (B, N, N), node -> ... -> root
        is_root = (chain == 0)
        depth = torch.argmax(is_root.to(torch.long), dim=-1)  # first root hit
        lengths = (depth + 1).clamp(min=1, max=N)
        return chain, lengths

    def encode_path_features(self, path_features, lengths):
        """Run the bidirectional GRU over pre-gathered path features.

        Args:
            path_features: (B, N, L, D) features along each node's chain
                           (node first, root last).
            lengths:       (B, N) valid chain lengths.

        Returns:
            p: (B, N, D) articulation-path summaries.
        """
        B, N, L, D = path_features.shape
        flat_feats = path_features.reshape(B * N, L, D)
        flat_lengths = lengths.reshape(B * N).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            flat_feats, flat_lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)                    # (2, B*N, hidden)
        h_cat = torch.cat((h_n[0], h_n[1]), dim=-1)  # final bidirectional states
        return self.proj(h_cat).reshape(B, N, D)

    def forward(self, h_s, fa, path_mode='full'):
        """Articulation-path summaries from per-node structural features.

        Args:
            h_s: (B, N, D) structural features h_i^s.
            fa:  (B, N) long parent indices.
            path_mode: 'full' (root-to-part chain), 'local' (current node
                       only) or 'none' (no path context; returns zeros).
        """
        B, N, D = h_s.shape
        if path_mode == 'none':
            return torch.zeros_like(h_s)
        chain, lengths = self.build_ancestor_chains(fa)
        if path_mode == 'local':
            lengths = torch.ones_like(lengths)
        path_features = torch.gather(
            h_s.unsqueeze(1).expand(B, N, N, D), 2,
            chain.unsqueeze(-1).expand(B, N, N, D))
        return self.encode_path_features(path_features, lengths)


class StructureGatedFusion(nn.Module):
    """Structure-controlled construction of the joint latent (ARTUS Eq. 2-4).

    Geometry enters the joint latent only through the structure-derived gate;
    the articulation path is computed from structural features alone.
    """

    def __init__(self, d_structure, d_geometry, d_hidden, d_model, dropout,
                 path_hidden=512, path_mode='full', gate_source='structure',
                 fusion_mode='gated'):
        super().__init__()
        assert path_mode in ('full', 'local', 'none')
        assert gate_source in ('structure', 'geometry', 'joint', 'constant', 'zero')
        assert fusion_mode in ('gated', 'direct', 'direct_norm', 'early')

        self.d_model = d_model
        self.path_mode = path_mode
        self.gate_source = gate_source
        self.fusion_mode = fusion_mode

        self.structure_encoder = FactorEncoder(d_structure, d_hidden, d_model, dropout)
        self.geometry_encoder = FactorEncoder(d_geometry, d_hidden, d_model, dropout)

        if path_mode != 'none':
            self.path_encoder = StructuralPathEncoder(d_model, path_hidden)
        else:
            self.path_encoder = None

        self.gate_fc = nn.Linear(d_model, d_model)
        self.gate_const = nn.Parameter(torch.zeros(d_model))  # 'constant' gate

        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_proj = nn.Linear(d_model, d_model)

        if fusion_mode == 'early':
            # Early fusion control: a single encoder over concatenated raw
            # factors, with no structural boundary.
            self.early_encoder = FactorEncoder(
                d_structure + d_geometry, d_hidden, d_model, dropout)

    def _gate(self, u, h_v):
        """Channel-wise gate g_i in (0, 1)^d from the configured source."""
        if self.gate_source == 'zero':
            return torch.zeros_like(u)
        if self.gate_source == 'constant':
            logits = self.gate_const.unsqueeze(0).unsqueeze(0).expand_as(u)
        elif self.gate_source == 'structure':
            logits = self.gate_fc(u)
        elif self.gate_source == 'geometry':
            logits = self.gate_fc(h_v)
        else:  # 'joint'
            logits = self.gate_fc(u + h_v)
        return torch.sigmoid(logits)

    def _fuse(self, u, g, h_v):
        """Fuse the structural representation with (gated) geometry."""
        if self.fusion_mode == 'direct':
            r = u + h_v
        elif self.fusion_mode == 'direct_norm':
            r = u + F.layer_norm(h_v, h_v.shape[-1:])
        else:  # 'gated'
            r = u + g * h_v
        e = self.fusion_proj(self.fusion_norm(r))
        return r, e

    def forward(self, s, v, fa):
        """Build joint latent tokens for a visible pathway snapshot.

        Args:
            s:  (B, N, d_structure) structural states (box, joint, limits).
            v:  (B, N, d_geometry) geometry latents of visible parts.
            fa: (B, N) long parent indices.

        Returns:
            dict with joint tokens 'e', pre-projection representation 'r'
            (L_scl target), structural representation 'u', gate 'g' and
            structural features 'h_s' (context for fuse_predicted).
        """
        if self.fusion_mode == 'early':
            e = self.early_encoder(torch.cat((s, v), dim=-1))
            return {
                'e': e,
                'r': e,
                'u': torch.zeros_like(e),
                'g': torch.ones_like(e),
                'h_s': torch.zeros_like(e),
            }

        h_s = self.structure_encoder(s)
        h_v = self.geometry_encoder(v)
        if self.path_encoder is not None:
            p = self.path_encoder(h_s, fa, path_mode=self.path_mode)
        else:
            p = torch.zeros_like(h_s)
        u = h_s + p
        g = self._gate(u, h_v)
        r, e = self._fuse(u, g, h_v)
        return {'e': e, 'r': r, 'u': u, 'g': g, 'h_s': h_s}

    def fuse_predicted(self, s_pred, v_pred, fa, h_s_ctx):
        """Pre-projection fusion r_k for predicted children (L_scl, Eq. 9-10).

        The articulation path of a predicted child is built from the
        ground-truth structural features of its ancestors (teacher forcing);
        only the child's own feature comes from the predicted state.

        Args:
            s_pred:  (B, N, d_structure) predicted structural states.
            v_pred:  (B, N, d_geometry) predicted coarse geometry latents.
            fa:      (B, N) long parent indices.
            h_s_ctx: (B, N, D) ground-truth structural features h_i^s.

        Returns:
            r: (B, N, D) pre-projection joint representation built from the
               predicted child states.
        """
        if self.fusion_mode == 'early':
            return self.early_encoder(torch.cat((s_pred, v_pred), dim=-1))

        B, N, D = h_s_ctx.shape
        h_s_pred = self.structure_encoder(s_pred)
        h_v_pred = self.geometry_encoder(v_pred)

        if self.path_encoder is not None and self.path_mode != 'none':
            chain, lengths = self.path_encoder.build_ancestor_chains(fa)
            if self.path_mode == 'local':
                lengths = torch.ones_like(lengths)
            path_features = torch.gather(
                h_s_ctx.unsqueeze(1).expand(B, N, N, D), 2,
                chain.unsqueeze(-1).expand(B, N, N, D))
            # Chain position 0 is the node itself: use the predicted feature.
            path_features[:, :, 0] = h_s_pred
            p = self.path_encoder.encode_path_features(path_features, lengths)
        else:
            p = torch.zeros_like(h_s_pred)

        u = h_s_pred + p
        g = self._gate(u, h_v_pred)
        r, _ = self._fuse(u, g, h_v_pred)
        return r
