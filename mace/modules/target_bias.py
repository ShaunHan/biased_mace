"""Fixed descriptor metric and quadratic target bias (autograd intact)."""

import math
from typing import Sequence

import torch


class TargetDescriptorMetric(torch.nn.Module):
    """Calibrate ONCE from a target and optional initial/reference structure.

    Every invariant block contributes its mean squared scaled error. If a
    distinct reference is supplied, its distance is set to one. No statistics
    are updated during MD. Reuse the same state_dict on exact restarts.
    """

    def __init__(
        self,
        target_segments: Sequence[torch.Tensor],
        reference_segments=None,
        scales=None,
    ):
        super().__init__()
        if not target_segments or any(t.shape[0] != 1 for t in target_segments):
            raise ValueError("Calibrate against exactly one target graph")
        reference = (
            target_segments if reference_segments is None else reference_segments
        )
        if len(reference) != len(target_segments):
            raise ValueError("Target and reference block counts differ")
        if scales is None:
            scales = [t.new_tensor(1.0) for t in target_segments]
        if len(scales) != len(target_segments):
            raise ValueError("One fixed scale per descriptor block is required")
        scale_parts = []
        for t, r, rms in zip(target_segments, reference, scales):
            if t.shape != r.shape or not bool(
                torch.isfinite(t).all() & torch.isfinite(r).all()
            ):
                raise ValueError("Nonfinite or mismatched calibration descriptors")
            if not bool(torch.isfinite(rms)) or float(rms) <= 0:
                raise ValueError("Descriptor scales must be finite and positive")
            scale = rms * math.sqrt(t.shape[-1] * len(target_segments))
            scale_parts.append(torch.ones_like(t) * scale)
        target = torch.cat(target_segments, dim=-1).detach().clone()
        scale = torch.cat(scale_parts, dim=-1).detach().clone()
        if reference_segments is not None:
            ref = torch.cat(reference_segments, dim=-1).detach()
            d2 = ((ref - target) / scale).square().sum()
            if float(d2) < 1e-12:
                raise ValueError(
                    "Reference and target are indistinguishable; omit reference_atoms or use a distinct start"
                )
            scale *= d2.sqrt()
        self.register_buffer("target", target)
        self.register_buffer("scale", scale)

    def forward(self, descriptor):
        if descriptor.shape[-1] != self.target.shape[-1]:
            raise ValueError("Descriptor dimension changed after calibration")
        return ((descriptor - self.target) / self.scale).square().sum(dim=-1)


class _GramDistance(torch.nn.Module):
    """Exact squared Frobenius distance to a fixed factored Gram matrix.

    A thin QR of the TARGET reduces work when channels outnumber samples.
    The residual formula is a sum of squares, avoiding cancellation near the
    target. No eigenvalue truncation, approximation, or custom backward.
    """

    def __init__(self, target_factor):
        super().__init__()
        x = target_factor.detach()
        if 0 < x.shape[1] * 4 < x.shape[0]:
            q, _ = torch.linalg.qr(x, mode="reduced")
            projected = q.T @ x
            target = projected @ projected.T
        else:
            q = None
            target = x @ x.T
        self.register_buffer("basis", q, persistent=False)
        self.register_buffer("gram", target, persistent=False)

    def forward(self, x):
        if self.basis is None:
            return (x @ x.T - self.gram).square().sum()
        a = self.basis.T @ x
        residual = x - self.basis @ a
        return (
            (a @ a.T - self.gram).square().sum()
            + 2 * (a @ residual.T).square().sum()
            + (residual.T @ residual).square().sum()
        )


class MomentTargetMetric(torch.nn.Module):
    """Evaluate the existing descriptor metric without exporting its vector.

    Targets and scales remain in TargetDescriptorMetric's original state_dict
    format. Uniform per-block scales permit Frobenius distances; manually
    edited nonuniform scales select the original vector path instead.
    """

    def __init__(self, target_moments, metric):
        super().__init__()
        self.metric = metric
        self.distances = torch.nn.ModuleList()
        self.slices = []
        offset = 0
        for part in target_moments:
            if torch.is_tensor(part):
                size = part.numel()
                self.distances.append(torch.nn.ModuleList())
            else:
                size = sum(x.shape[0] * (x.shape[0] + 1) // 2 for x in part)
                self.distances.append(
                    torch.nn.ModuleList([_GramDistance(x) for x in part])
                )
            self.slices.append(slice(offset, offset + size))
            offset += size
        if offset != metric.target.numel():
            raise ValueError("Moment layout differs from the calibrated descriptor")
        self.refresh_scales()

    def refresh_scales(self):
        self.compatible = all(
            not distances
            or bool(
                (
                    self.metric.scale[:, sl]
                    == self.metric.scale[:, sl.start : sl.start + 1]
                ).all()
            )
            for sl, distances in zip(self.slices, self.distances)
        )

    def forward(self, moments):
        if not self.compatible:
            raise ValueError("Nonuniform scales require the full descriptor path")
        values = []
        for graph in moments:
            parts = []
            for part, sl, distances in zip(graph, self.slices, self.distances):
                if not distances:
                    parts.append(
                        ((part - self.metric.target[0, sl]) / self.metric.scale[0, sl])
                        .square()
                        .sum()
                    )
                else:
                    norm = sum(distance(x) for distance, x in zip(distances, part))
                    parts.append(norm / self.metric.scale[0, sl.start].square())
            values.append(torch.stack(parts).sum())
        return torch.stack(values)


def target_potential(distance_squared):
    """Harmonic restraint, v=d^2/2; no square root or cusp at the target."""
    return 0.5 * distance_squared


def biased_autograd(
    physical_energy,
    unit_potential,
    weight,
    positions,
    displacement=None,
    cell=None,
    pbc=None,
):
    """One forward, one reverse pass; fixed weight in MODEL energy units.

    All differentiable field/density responses remain in the total gradient.
    """
    if not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("bias_weight must be finite and nonnegative")
    total = physical_energy + weight * unit_potential
    total = total + 0 * positions.sum()
    inputs = [positions] if displacement is None else [positions, displacement]
    grads = torch.autograd.grad(total.sum(), inputs, allow_unused=True)
    forces = -grads[0] if grads[0] is not None else torch.zeros_like(positions)
    stress = None
    virials = None
    if displacement is not None:
        strain_gradient = (
            grads[1] if grads[1] is not None else torch.zeros_like(displacement)
        )
        volume = torch.linalg.det(cell.reshape(-1, 3, 3)).abs()
        periodic = volume > 0
        if pbc is not None:
            periodic = periodic & pbc.reshape(-1, 3).any(dim=-1)
        safe_volume = torch.where(periodic, volume, torch.ones_like(volume))
        stress = torch.where(
            periodic[:, None, None], strain_gradient / safe_volume[:, None, None], 0
        )
        virials = -strain_gradient
    return total, forces, stress, virials
