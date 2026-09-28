"""Smooth, parameter-free O(3) and permutation invariant MACE descriptors.

No new model parameters or checkpoint keys. Features must carry their actual
irreps AND memory layout; these are independent pieces of information.
"""

from contextlib import contextmanager
from dataclasses import dataclass
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


def _symmetric_vector(matrix):
    """sqrt(2) off-diagonals preserve the full Frobenius distance."""
    i, j = torch.triu_indices(matrix.shape[-1], matrix.shape[-1], device=matrix.device)
    factor = torch.where(i == j, 1.0, 2.0**0.5).to(matrix)
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

    @staticmethod
    def _blocks(features, spec):
        irreps = o3.Irreps(spec.irreps)
        features = features.reshape(features.shape[0], -1)
        if features.shape[-1] != irreps.dim:
            raise ValueError(
                f"{spec.key}: expected {irreps.dim} features, got {features.shape[-1]}"
            )
        grouped = {}
        for (mul, ir), sl in zip(irreps, irreps.slices()):
            block = features[:, sl]
            if spec.layout == "mul_ir":
                block = block.reshape(-1, mul, ir.dim)
            else:
                block = block.reshape(-1, ir.dim, mul).transpose(-1, -2)
            grouped.setdefault((ir.l, ir.p), []).append(block)
        return [(key, torch.cat(value, dim=1)) for key, value in grouped.items()]

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

    def segments(
        self,
        features: Mapping[str, torch.Tensor],
        numbers,
        batch=None,
    ):
        """Return invariant blocks [n_graphs, block_dim] before fixed scaling."""
        first = features[self.specs[0].key]
        numbers = torch.as_tensor(numbers, device=first.device, dtype=torch.long)
        if numbers.ndim != 1 or numbers.numel() != first.shape[0]:
            raise ValueError("numbers must give one atomic number per feature row")
        if batch is None:
            batch = torch.zeros_like(numbers)
        if batch.shape != numbers.shape or batch.numel() == 0:
            raise ValueError("Empty/invalid graph batch")
        ngraphs = int(batch.max()) + 1
        if set(numbers.tolist()) - set(self.species):
            raise ValueError("A species is outside the fixed descriptor species set")

        pieces_per_graph = []
        for graph in range(ngraphs):
            select = batch == graph
            if not bool(select.any()):
                raise ValueError("Batch graph labels must be contiguous and nonempty")
            z = numbers[select]
            weights = torch.stack([(z == a).to(first) for a in self.species], dim=1)
            counts = weights.sum(dim=0).clamp_min(1)
            weights = weights / counts
            pieces = []
            for spec in self.specs:
                for (ell, parity), block in self._blocks(
                    features[spec.key][select], spec
                ):
                    dim = 2 * ell + 1
                    mean = torch.einsum("iz,iam->zam", weights, block)
                    if ell == 0 and parity == 1:
                        pieces.append(mean.flatten())
                    local = torch.einsum("iz,iam,ibm->zab", weights, block, block) / dim
                    pieces.append(_symmetric_vector(local).flatten())
                    if ell > 0 or parity == -1:
                        global_tensor = mean.flatten(0, 1)
                        pieces.append(
                            _symmetric_vector(global_tensor @ global_tensor.T / dim)
                        )
            pieces_per_graph.append(pieces)
        return [torch.stack(p) for p in zip(*pieces_per_graph)]

    def forward(self, features, numbers, batch=None):
        return torch.cat(self.segments(features, numbers, batch), dim=-1)
