"""Small utilities for a conservative adaptive target bias."""

import math


def validate_bias_weight(value):
    """Require an explicit finite nonnegative floating-point bias weight."""
    if type(value) is not float:
        raise TypeError("bias_weight must be a float, e.g. 1.0")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("bias_weight must be finite and nonnegative")
    return value


def partition_bias_energy(energy_scale, distance_squared, balance):
    """Partition one energy scale between motion and a harmonic target bias.

    In adaptive mode ``balance`` is the dimensionless ratio
    V(d=1) / K.  For V = w d^2 / 2,

        K = E / (1 + balance d0^2)
        w = 2 balance K.

    Returns ``(w, K)``.  The caller decides how the returned motion energy is
    used; this module has no dependency on any particular sampling algorithm.
    """
    balance = validate_bias_weight(balance)
    energy_scale = float(energy_scale)
    distance_squared = float(distance_squared)
    if not math.isfinite(energy_scale) or energy_scale < 0.0:
        raise ValueError("energy_scale must be finite and nonnegative")
    if not math.isfinite(distance_squared) or distance_squared < 0.0:
        raise ValueError("distance_squared must be finite and nonnegative")
    if balance == 0.0 or energy_scale == 0.0:
        return 0.0, energy_scale
    denominator = 1.0 + balance * distance_squared
    motion_energy = energy_scale / denominator
    return 2.0 * balance * motion_energy, motion_energy
