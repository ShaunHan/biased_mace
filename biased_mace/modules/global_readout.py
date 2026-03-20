from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from e3nn import o3

from biased_mace.tools.scatter import scatter_mean, scatter_sum


class GlobalReadoutBlock(nn.Module):
    """
    Permutation-invariant global readout built from all MACE equivariant descriptors.

    The input node feature tensor is assumed to be the concatenation of the per-layer
    outputs stored in MACE.forward() as node_feats_out = torch.cat(node_feats_concat, dim=-1).

    For each layer slice we:
      - keep scalar (l=0) channels directly,
      - convert each non-scalar irrep copy to an invariant norm,
      - include pairwise dot products between copies of the same irrep block.

    This gives a compact invariant token per atom, per layer, while still using the
    non-scalar information.

    Parameters
    ----------
    layer_irreps
        List of irreps strings, one per interaction/product output slice.
        Example: [str(prod.target_irreps) for prod in self.products]
    hidden_dim
        Transformer hidden size.
    descriptor_dim
        Dimension of the latent global descriptor.
    depth
        Number of transformer encoder layers.
    num_heads
        Number of attention heads.
    dropout
        Transformer dropout.
    """

    def __init__(
        self,
        layer_irreps: Sequence[str],
        hidden_dim: int = 256,
        descriptor_dim: int = 128,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_irreps_str = list(layer_irreps)
        self.layer_irreps = [o3.Irreps(ir) for ir in self.layer_irreps_str]
        self.hidden_dim = hidden_dim
        self.descriptor_dim = descriptor_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dropout = dropout

        # Precompute tokenization specs for TorchScript friendliness.
        # Each entry is a list of [start, dim, l, mul, ir_dim] for one layer.
        # l == 0 means scalar block.
        self.layer_specs: List[List[List[int]]] = []
        self.layer_dims: List[int] = []
        token_dim = 0

        for irreps in self.layer_irreps:
            layer_spec: List[List[int]] = []
            cursor = 0
            layer_dim = int(irreps.dim)
            self.layer_dims.append(layer_dim)

            layer_token_dim = 0
            for mul, ir in irreps:
                block_dim = int(mul * ir.dim)
                layer_spec.append([cursor, block_dim, int(ir.l), int(mul), int(ir.dim)])
                cursor += block_dim

                if ir.l == 0:
                    layer_token_dim += block_dim
                else:
                    # norms: mul
                    # pairwise dot products between copies: mul * (mul - 1) / 2
                    layer_token_dim += mul * (mul + 1) // 2

            self.layer_specs.append(layer_spec)
            token_dim += layer_token_dim

        self.token_dim = token_dim

        self.input_norm = nn.LayerNorm(self.token_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.pool_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.descriptor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, descriptor_dim),
            nn.SiLU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

        self.energy_head = nn.Linear(descriptor_dim, 1)
        nn.init.zeros_(self.energy_head.weight)
        nn.init.zeros_(self.energy_head.bias)

    @staticmethod
    def _pack_graphs(
        x: torch.Tensor,
        batch: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert concatenated atoms into padded per-graph sequences.

        Returns
        -------
        padded : (B, max_n, D)
        key_padding_mask : (B, max_n), True means ignore
        """
        device = x.device
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        counts = torch.bincount(batch, minlength=num_graphs)

        seqs: List[torch.Tensor] = []
        start = 0
        for g in range(num_graphs):
            n = int(counts[g].item())
            end = start + n
            tokens_g = x[start:end]
            if node_mask is not None:
                keep_g = node_mask[start:end].to(torch.bool)
                tokens_g = tokens_g[keep_g]
            if tokens_g.numel() == 0:
                raise ValueError(
                    "A graph had zero selected atoms for the global descriptor."
                )
            seqs.append(tokens_g)
            start = end

        # pad_sequence is fine here because atoms are already grouped per graph
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        max_n = padded.shape[1]
        key_padding_mask = torch.ones(
            (num_graphs, max_n), dtype=torch.bool, device=device
        )
        for g, seq in enumerate(seqs):
            key_padding_mask[g, : seq.shape[0]] = False
        return padded, key_padding_mask

    def _tokenize_layer(self, x_layer: torch.Tensor, layer_spec: List[List[int]]) -> torch.Tensor:
        """
        Convert one layer slice to invariant per-atom tokens.

        For each irrep block:
          - l=0: keep the scalar channels directly
          - l>0: keep the norm of each multiplicity copy and the pairwise copy-dot-products
        """
        parts: List[torch.Tensor] = []
        for start, block_dim, l, mul, ir_dim in layer_spec:
            block = x_layer[:, start : start + block_dim]

            if l == 0:
                parts.append(block)
                continue

            block = block.reshape(block.shape[0], mul, ir_dim)
            norms = torch.linalg.norm(block, dim=-1)
            parts.append(norms)

            if mul > 1:
                gram = torch.einsum("nmc,nkc->nmk", block, block) / float(ir_dim)
                tri = torch.triu_indices(mul, mul, device=block.device)
                parts.append(gram[:, tri[0], tri[1]])

        return torch.cat(parts, dim=-1)

    def forward(
        self,
        node_tokens: torch.Tensor,
        batch: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        node_tokens
            Concatenated per-layer MACE node features, shape (N, D_total).
        batch
            Graph index per atom.
        node_mask
            Optional boolean mask selecting a subset of atoms for the global descriptor.
        edge_index, edge_feats
            Kept for API compatibility; not used in this implementation.

        Returns
        -------
        graph_descriptor : (B, descriptor_dim)
        graph_energy : (B,)
        """
        del edge_index, edge_feats

        if len(self.layer_specs) != len(self.layer_dims):
            raise RuntimeError("GlobalReadoutBlock layer specs are inconsistent.")

        x_layers = torch.split(node_tokens, self.layer_dims, dim=-1)
        tokens_per_layer = [
            self._tokenize_layer(x_layer, spec)
            for x_layer, spec in zip(x_layers, self.layer_specs)
        ]
        x = torch.cat(tokens_per_layer, dim=-1)

        x = self.input_proj(self.input_norm(x))
        x, key_padding_mask = self._pack_graphs(x, batch, node_mask=node_mask)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        valid = (~key_padding_mask).unsqueeze(-1)
        counts = valid.sum(dim=1).clamp_min(1)

        graph_sum = (x * valid).sum(dim=1)
        graph_mean = graph_sum / counts

        scores = self.pool_score(x).squeeze(-1)
        scores = scores.masked_fill(key_padding_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        graph_attn = torch.sum(attn.unsqueeze(-1) * x, dim=1)

        graph_max = x.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        graph_max = torch.where(torch.isfinite(graph_max), graph_max, torch.zeros_like(graph_max))

        graph_emb = torch.cat([graph_sum, graph_mean, graph_attn, graph_max], dim=-1)
        graph_descriptor = self.descriptor_head(graph_emb)
        graph_energy = self.energy_head(graph_descriptor).squeeze(-1)
        return graph_descriptor, graph_energy
