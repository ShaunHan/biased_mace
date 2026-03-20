from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from biased_mace.tools.scatter import scatter_mean


class GlobalReadoutBlock(nn.Module):
    """
    Permutation-invariant global readout over per-atom invariant tokens.

    Inputs
    ------
    node_tokens : (N, token_dim)
        Per-atom invariant token vectors, usually the output of extract_invariant(...)
        from the final MACE node features.
    batch : (N,)
        Graph index for each atom.
    edge_index : (2, E), optional
        Used only to add a neighbor-context feature per atom.
    edge_feats : (E, edge_dim), optional
        Radial / edge features from MACE. These are pooled to nodes by receiver index.
    node_mask : (N,), optional boolean
        True for atoms that should participate in the global descriptor.
        Use this for bias_indices selection without changing the full system forces.

    Returns
    -------
    graph_descriptor : (B, descriptor_dim)
    graph_energy : (B,)
    """

    def __init__(
        self,
        token_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 256,
        descriptor_dim: int = 128,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.descriptor_dim = descriptor_dim

        in_dim = token_dim + edge_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
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
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, descriptor_dim),
            nn.SiLU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

        self.energy_head = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _pack_graphs(
        x: torch.Tensor,
        batch: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Packs a ragged batch into padded sequences.

        Returns
        -------
        padded : (B, max_n, D)
        key_padding_mask : (B, max_n), True means "ignore"
        """
        device = x.device
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        counts = torch.bincount(batch, minlength=num_graphs)

        seqs = []
        masks = []
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
            masks.append(torch.zeros(tokens_g.shape[0], dtype=torch.bool, device=device))
            start = end

        padded = pad_sequence(seqs, batch_first=True)  # (B, max_n, D)
        max_n = padded.shape[1]
        key_padding_mask = torch.ones((num_graphs, max_n), dtype=torch.bool, device=device)
        for g, seq in enumerate(seqs):
            key_padding_mask[g, : seq.shape[0]] = False
        return padded, key_padding_mask

    def forward(
        self,
        node_tokens: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add a neighbor-context invariant summary if available.
        if edge_index is not None and edge_feats is not None:
            neighbor_ctx = scatter_mean(
                src=edge_feats,
                index=edge_index[1],
                dim=0,
                dim_size=node_tokens.shape[0],
            )
            node_tokens = torch.cat([node_tokens, neighbor_ctx], dim=-1)

        x = self.input_proj(node_tokens)
        x, key_padding_mask = self._pack_graphs(x, batch, node_mask=node_mask)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        scores = self.pool_score(x).squeeze(-1)
        scores = scores.masked_fill(key_padding_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        graph_emb = torch.sum(attn.unsqueeze(-1) * x, dim=1)

        graph_descriptor = self.descriptor_head(graph_emb)
        graph_energy = self.energy_head(graph_descriptor).squeeze(-1)
        return graph_descriptor, graph_energy
