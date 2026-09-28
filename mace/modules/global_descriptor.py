"""Smooth, parameter-free O(3) and permutation invariant MACE descriptors.

No new model parameters or checkpoint keys. Features must carry their actual
irreps AND memory layout; these are independent pieces of information.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Mapping, Sequence

import torch
from e3nn import o3


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    irreps: str
    layout: str = "mul_ir"

    def __post_init__(self):
        o3.Irreps(self.irreps)
        if self.layout not in ("mul_ir", "ir_mul"):
            raise ValueError(f"Unknown feature layout: {self.layout}")


def product_layout(product) -> str:
    """Read the product output layout, not a guessed model-wide convention."""
    config = getattr(product, "cueq_config", None)
    if config is not None and getattr(config, "enabled", False):
        return str(config.layout_str)
    linear = product.linear
    for name in ("layout_out", "layout"):
        layout = getattr(linear, name, None)
        if layout is not None:
            text = str(layout)
            if text in ("mul_ir", "ir_mul"):
                return text
    if linear.__class__.__module__.startswith("cuequivariance"):
        raise ValueError("Cannot determine accelerated product output layout")
    return "mul_ir"


def model_feature_specs(model, include_electrostatics: bool = True):
    """Descriptors for eager MACE products; optional exposed field/density."""
    specs = [
        FeatureSpec(f"layer_{i}", str(p.linear.irreps_out), product_layout(p))
        for i, p in enumerate(model.products)
    ]
    if include_electrostatics and hasattr(model, "potential_irreps"):
        config = getattr(model, "cueq_config", None)
        layout = (
            config.layout_str
            if config is not None and getattr(config, "enabled", False)
            else "mul_ir"
        )
        specs.append(
            FeatureSpec("potential_features", str(model.potential_irreps), layout)
        )
        if hasattr(model, "atomic_multipoles_max_l"):
            specs.append(
                FeatureSpec(
                    "spin_charge_density",
                    str(
                        2 * o3.Irreps.spherical_harmonics(model.atomic_multipoles_max_l)
                    ),
                )
            )
    return specs


@contextmanager
def capture_mace_features(model):
    """Capture attached tensors, including models that do not return node_feats.

    Hooks live only during one eager forward and are removed on exceptions.
    As with an ASE calculator itself, do not share the model across threads.
    """
    features = {}
    handles = []
    try:
        for i, product in enumerate(model.products):

            def save(_module, _args, output, key=f"layer_{i}"):
                features[key] = output

            handles.append(product.register_forward_hook(save))
        yield features
    finally:
        for handle in handles:
            handle.remove()


@lru_cache(maxsize=32)
def _symmetric_indices(size, device, dtype):
    i, j = torch.triu_indices(size, size, device=device)
    factor = torch.ones(i.shape, dtype=dtype, device=device)
    factor[i != j] = math.sqrt(2)
    return i, j, factor


def _symmetric_vector(matrix):
    """sqrt(2) off-diagonals preserve the full Frobenius distance."""
    i, j, factor = _symmetric_indices(matrix.shape[-1], matrix.device, matrix.dtype)
    return matrix[..., i, j] * factor


class InvariantMomentDescriptor(torch.nn.Module):
    """Species-resolved first and second moments of irrep-valued features.

    For each source and each (l, parity), include scalar means (0e), means of
    local channel Gram matrices, and the Gram matrix of species-pooled tensors
    (l>0 or 0o). Repeated equal irreps in a source are joined, never unequal l
    or parity. Different layers remain distinct sources.

    """

    def __init__(
        self,
        specs: Sequence[FeatureSpec],
        species: Sequence[int],
    ):
        super().__init__()
        self.specs = tuple(specs)
        self.species = tuple(sorted(set(int(z) for z in species)))
        if not self.specs or not self.species:
            raise ValueError("Nonempty feature specs and species are required")
        if len({s.key for s in self.specs}) != len(self.specs):
            raise ValueError("Feature keys must be unique")
        self._plans = {}
        for spec in self.specs:
            irreps = o3.Irreps(spec.irreps)
            groups = {}
            for (mul, ir), sl in zip(irreps, irreps.slices()):
                groups.setdefault((ir.l, ir.p), []).append((mul, ir.dim, sl))
            self._plans[spec.key] = (irreps.dim, tuple(groups.items()))

    def _blocks(self, features, spec):
        size, groups = self._plans[spec.key]
        features = features.reshape(features.shape[0], -1)
        if features.shape[-1] != size:
            raise ValueError(
                f"{spec.key}: expected {size} features, got {features.shape[-1]}"
            )
        for key, entries in groups:
            blocks = []
            for mul, dim, sl in entries:
                block = features[:, sl]
                if spec.layout == "mul_ir":
                    block = block.reshape(features.shape[0], mul, dim)
                else:
                    block = block.reshape(features.shape[0], dim, mul).transpose(-1, -2)
                blocks.append(block)
            yield key, blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=1)

    def calibration_statistics(self, features):
        """Underlying tensor mean squares and moment orders, in segment order.

        Scale pooled contractions using local tensor amplitudes. A symmetric
        molecule can have a zero pooled vector despite large local vectors;
        dividing by that pooled value would create a spuriously stiff bias.
        """
        statistics = []
        for spec in self.specs:
            for (ell, parity), block in self._blocks(features[spec.key], spec):
                power = block.detach().square().mean()
                if ell == 0 and parity == 1:
                    statistics.append((power, 1))
                statistics.append((power, 2))
                if ell > 0 or parity == -1:
                    statistics.append((power, 2))
        return statistics

    def atom_groups(self, numbers, batch=None, device=None):
        """Build graph/species indices once for a fixed ordered composition.

        ASE supplies numbers on the CPU. Keeping this metadata out of the
        differentiable forward avoids GPU synchronization on every force call.
        The general batched API also accepts tensor inputs on any device.
        """
        numbers = torch.as_tensor(numbers).detach().to(device="cpu", dtype=torch.long)
        if numbers.ndim != 1:
            raise ValueError("numbers must give one atomic number per feature row")
        if batch is None:
            batch = torch.zeros_like(numbers)
        else:
            batch = torch.as_tensor(batch).detach().to(device="cpu", dtype=torch.long)
        if batch.shape != numbers.shape or batch.numel() == 0:
            raise ValueError("Empty/invalid graph batch")
        if set(numbers.tolist()) - set(self.species):
            raise ValueError("A species is outside the fixed descriptor species set")
        labels = sorted(set(batch.tolist()))
        if labels != list(range(len(labels))):
            raise ValueError("Batch graph labels must be contiguous and nonempty")
        return tuple(
            tuple(
                ((batch == graph) & (numbers == z))
                .nonzero()
                .flatten()
                .to(device=device)
                for z in self.species
            )
            for graph in labels
        )

    def moment_factors(self, features: Mapping[str, torch.Tensor], groups):
        """Scalar means and factors X whose Gram matrix X X^T is a moment.

        Factoring does not discard channels or change the descriptor. It lets
        the target distance avoid constructing large channel-by-channel Grams.
        """
        blocks = [
            item
            for spec in self.specs
            for item in self._blocks(features[spec.key], spec)
        ]
        graphs = []
        for graph in groups:
            pieces = []
            for (ell, parity), block in blocks:
                dim, channels = 2 * ell + 1, block.shape[1]
                means, factors = [], []
                for indices in graph:
                    selected = block.index_select(0, indices)
                    count = indices.numel()
                    means.append(selected.sum(0) / max(count, 1))
                    factors.append(
                        selected.transpose(0, 1).reshape(channels, count * dim)
                        / math.sqrt(max(count, 1) * dim)
                    )
                mean = torch.stack(means)
                if ell == 0 and parity == 1:
                    pieces.append(mean.flatten())
                pieces.append(tuple(factors))
                if ell > 0 or parity == -1:
                    pieces.append((mean.flatten(0, 1) / math.sqrt(dim),))
            graphs.append(pieces)
        return graphs

    @staticmethod
    def segments_from_moments(moments):
        graphs = [
            [
                (
                    part
                    if torch.is_tensor(part)
                    else torch.cat([_symmetric_vector(x @ x.T).flatten() for x in part])
                )
                for part in graph
            ]
            for graph in moments
        ]
        return [torch.stack(p) for p in zip(*graphs)]

    def segments(self, features, numbers, batch=None):
        """Return invariant blocks [n_graphs, block_dim] before fixed scaling."""
        first = features[self.specs[0].key]
        if len(numbers) != first.shape[0]:
            raise ValueError("numbers must give one atomic number per feature row")
        groups = self.atom_groups(numbers, batch, first.device)
        return self.segments_from_moments(self.moment_factors(features, groups))

    def forward(self, features, numbers, batch=None):
        return torch.cat(self.segments(features, numbers, batch), dim=-1)
