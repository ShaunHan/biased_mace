"""Small utilities for a conservative adaptive target bias."""

import math


class BiasForceError(FloatingPointError):
    """A trial geometry exceeds the bias-force limit; no clipped forces exist."""

    def __init__(self, force, limit):
        self.force = float(force)
        self.limit = float(limit)
        super().__init__(
            f"Atomic bias force {force:.6g} eV/Ang exceeds max_bias_force={limit:.6g}. "
            "Reject this trial geometry; do not use clipped forces."
        )


def validate_bias_weight(value):
    """Require an explicit finite nonnegative floating-point bias weight."""
    if type(value) is not float:
        raise TypeError("bias_weight must be a float, e.g. 1.0")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("bias_weight must be finite and nonnegative")
    return value


def partition_bias_energy(energy_scale, unit_potential, balance, *,
                          unit_force_max=0.0, force_limit=None):
    """Partition one escape energy with an optional initial force constraint.

    In adaptive mode ``balance`` is the dimensionless ratio
    V(d=1) / K before the force constraint. For V = w v,

        K = E / (1 + balance v0)
        w = balance K.

    Returns ``(w, K)``.  The caller decides how the returned motion energy is
    used; this module has no dependency on any particular sampling algorithm.
    """
    balance = validate_bias_weight(balance)
    energy_scale = float(energy_scale)
    unit_potential = float(unit_potential)
    if not math.isfinite(energy_scale) or energy_scale < 0.0:
        raise ValueError("energy_scale must be finite and nonnegative")
    if not math.isfinite(unit_potential) or unit_potential < 0.0:
        raise ValueError("unit_potential must be finite and nonnegative")
    if not math.isfinite(unit_force_max) or unit_force_max < 0:
        raise ValueError("unit_force_max must be finite and nonnegative")
    if force_limit is not None and (not math.isfinite(force_limit) or force_limit <= 0):
        raise ValueError("force_limit must be finite and positive or None")
    if balance == 0.0 or energy_scale == 0.0:
        return 0.0, energy_scale
    denominator = 1.0 + balance * unit_potential
    motion_energy = energy_scale / denominator
    weight = balance * motion_energy
    if force_limit is not None and unit_force_max > 0:
        weight = min(weight, force_limit / unit_force_max)
        # A smaller restraint leaves more of the SAME budget for motion.
        motion_energy = max(0.0, energy_scale - weight * unit_potential)
    return weight, motion_energy
