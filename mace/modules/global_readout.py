from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from mace.tools.scatter import scatter_mean


class GlobalReadoutBlock(nn.Module):
    """Permutation-invariant global readout over node features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        descriptor_dim: int = 128,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

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

        self.descriptor_head = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_dim),
            nn.Linear(4 * self.hidden_dim, self.descriptor_dim),
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
        graph_descriptor = self.descriptor_head(graph_emb)
        graph_energy = self.energy_head(graph_descriptor).squeeze(-1)
        return graph_descriptor, graph_energy
