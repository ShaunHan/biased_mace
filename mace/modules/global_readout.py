from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from e3nn import o3
from torch.nn.utils.rnn import pad_sequence

from mace.tools.scatter import scatter_sum


class EquivariantScalarContraction(nn.Module):
    """Convert irreps-valued MACE node features to invariant scalar features.

    This module is used only when ``global_readout_from_invariants_only=False``.
    It has two outputs:

    1. ``node_scalars``: per-atom scalar tokens. 0e channels are passed through
       unchanged. All other irrep blocks are converted to channel Gram matrices
       ``sum_m h[a, m] h[b, m] / (2l + 1)`` and flattened over ``a <= b``.

    2. ``graph_pair_scalars``: explicit all-pair, off-diagonal contractions
       ``mean_{i != j} sum_m h_i[a, m] h_j[b, m] / (2l + 1)`` for each graph.
       These are the global relative-orientation invariants.
    """

    def __init__(self, irreps_list: Sequence[o3.Irreps]):
        super().__init__()
        self.irreps_list = [o3.Irreps(str(irreps)) for irreps in irreps_list]
        self.block_specs: List[List[Tuple[int, int, int, int, int]]] = []
        node_dim = 0
        graph_dim = 0

        for irreps in self.irreps_list:
            layer_specs: List[Tuple[int, int, int, int, int]] = []
            start = 0
            for mul, ir in irreps:
                mul_int = int(mul)
                ell = int(ir.l)
                parity = int(ir.p)
                block_dim = mul_int * (2 * ell + 1)
                end = start + block_dim
                layer_specs.append((start, end, mul_int, ell, parity))

                gram_dim = mul_int * (mul_int + 1) // 2
                if ell == 0 and parity == 1:
                    node_dim += mul_int
                else:
                    node_dim += gram_dim
                graph_dim += gram_dim
                start = end
            self.block_specs.append(layer_specs)

        self.node_output_dim = int(node_dim)
        self.graph_output_dim = int(graph_dim)

    @staticmethod
    def _upper_triangle(x: torch.Tensor) -> torch.Tensor:
        idx = torch.triu_indices(x.shape[-2], x.shape[-1], device=x.device)
        return x[..., idx[0], idx[1]]

    def forward(
        self,
        node_feats_list: List[torch.Tensor],
        batch: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

        dtype = node_feats_list[0].dtype
        device = node_feats_list[0].device
        if node_mask is None:
            weights = torch.ones((batch.shape[0],), dtype=dtype, device=device)
        else:
            weights = node_mask.to(dtype=dtype, device=device)

        counts = scatter_sum(weights, batch, dim=0, dim_size=num_graphs).clamp_min(1.0)
        pair_counts = (counts * (counts - 1.0)).clamp_min(1.0)
        has_pairs = (counts > 1.0).view(-1, 1, 1)

        node_blocks: List[torch.Tensor] = []
        graph_blocks: List[torch.Tensor] = []

        for feats, layer_specs in zip(node_feats_list, self.block_specs):
            for start, end, mul, ell, parity in layer_specs:
                irrep_dim = 2 * ell + 1
                block = feats[:, start:end].reshape(feats.shape[0], mul, irrep_dim)
                block = block * weights.view(-1, 1, 1)
                norm = float(irrep_dim)

                if ell == 0 and parity == 1:
                    node_blocks.append(block[:, :, 0])
                else:
                    node_gram = torch.einsum("nam,nbm->nab", block, block) / norm
                    node_blocks.append(self._upper_triangle(node_gram))

                sum_block = scatter_sum(
                    block.reshape(block.shape[0], mul * irrep_dim),
                    batch,
                    dim=0,
                    dim_size=num_graphs,
                ).reshape(num_graphs, mul, irrep_dim)
                local_gram = torch.einsum("nam,nbm->nab", block, block) / norm
                local_gram_sum = scatter_sum(
                    local_gram.reshape(local_gram.shape[0], mul * mul),
                    batch,
                    dim=0,
                    dim_size=num_graphs,
                ).reshape(num_graphs, mul, mul)

                # Sum over all ordered i != j pairs:
                #   sum_{i != j} h_i dot h_j = (sum_i h_i) dot (sum_j h_j)
                #                            - sum_i h_i dot h_i.
                all_pair_gram = (
                    torch.einsum("gam,gbm->gab", sum_block, sum_block) / norm
                )
                offdiag_pair_mean = (all_pair_gram - local_gram_sum) / pair_counts.view(
                    -1, 1, 1
                )
                offdiag_pair_mean = torch.where(
                    has_pairs, offdiag_pair_mean, torch.zeros_like(offdiag_pair_mean)
                )
                graph_blocks.append(self._upper_triangle(offdiag_pair_mean))

        node_scalars = torch.cat(node_blocks, dim=-1)
        graph_pair_scalars = torch.cat(graph_blocks, dim=-1)
        return node_scalars, graph_pair_scalars


class GlobalReadoutBlock(nn.Module):
    """Permutation-invariant global readout over scalar node features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        descriptor_dim: int = 128,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        graph_input_dim: int = 0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.graph_input_dim = int(graph_input_dim)

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.depth)

        self.pool_score = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        descriptor_input_dim = 4 * self.hidden_dim + self.graph_input_dim
        self.descriptor_head = nn.Sequential(
            nn.LayerNorm(descriptor_input_dim),
            nn.Linear(descriptor_input_dim, self.descriptor_dim),
            nn.SiLU(),
            nn.Linear(self.descriptor_dim, self.descriptor_dim),
        )

        self.energy_head = nn.Linear(self.descriptor_dim, 1)
        nn.init.zeros_(self.energy_head.weight)
        nn.init.zeros_(self.energy_head.bias)

    @staticmethod
    def _pack_graphs(
        x: torch.Tensor,
        batch: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        counts = torch.bincount(batch, minlength=num_graphs)

        seqs = []
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

        padded = pad_sequence(seqs, batch_first=True)
        max_n = padded.shape[1]
        key_padding_mask = torch.ones(
            (num_graphs, max_n), dtype=torch.bool, device=device
        )
        for g, seq in enumerate(seqs):
            key_padding_mask[g, : seq.shape[0]] = False
        return padded, key_padding_mask

    def forward(
        self,
        node_feats: torch.Tensor,
        batch: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
        graph_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del edge_index, edge_feats

        if node_feats.dim() != 2:
            node_feats = node_feats.reshape(node_feats.shape[0], -1)

        if node_feats.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"GlobalReadoutBlock expected input_dim={self.input_dim}, "
                f"but got node_feats.shape[-1]={node_feats.shape[-1]}"
            )

        x = self.input_proj(self.input_norm(node_feats))
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

        graph_max = x.masked_fill(
            key_padding_mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values
        graph_max = torch.where(
            torch.isfinite(graph_max), graph_max, torch.zeros_like(graph_max)
        )

        graph_emb = torch.cat([graph_sum, graph_mean, graph_attn, graph_max], dim=-1)
        if self.graph_input_dim > 0:
            if graph_feats is None:
                raise RuntimeError(
                    "GlobalReadoutBlock was initialized with graph_input_dim="
                    f"{self.graph_input_dim}, but graph_feats was not provided."
                )
            if graph_feats.shape[-1] != self.graph_input_dim:
                raise RuntimeError(
                    f"GlobalReadoutBlock expected graph_input_dim={self.graph_input_dim}, "
                    f"but got graph_feats.shape[-1]={graph_feats.shape[-1]}"
                )
            graph_emb = torch.cat([graph_emb, graph_feats], dim=-1)

        graph_descriptor = self.descriptor_head(graph_emb)
        graph_energy = self.energy_head(graph_descriptor).squeeze(-1)
        return graph_descriptor, graph_energy
