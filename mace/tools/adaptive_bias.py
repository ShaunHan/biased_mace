"""Partition one MH escape-energy budget between motion and a harmonic bias."""
import math


def kinetic_bias_weight(kinetic_energy: float) -> float:
    """w=2*K_launch: the harmonic bias at the fixed reference equals K_launch."""
    energy = float(kinetic_energy)
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("Escape kinetic energy must be finite and nonnegative")
    weight = 2.0 * energy
    if not math.isfinite(weight):
        raise ValueError("Escape kinetic energy overflows the bias weight")
    return weight


def partition_escape_energy(energy: float, distance_squared: float):
    """Return (w, K_launch) with K_launch + w*d_start**2/2 = energy.

    Together with w/2=K_launch (unit reference distance), this fixes the
    partition without a force cap, saturation scale, or another controller.
    The input is a budget, NOT the kinetic energy retained after this call.
    This is an energy-allocation convention, not a barrier estimate.
    """
    energy, distance_squared = float(energy), float(distance_squared)
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("Escape energy budget must be finite and nonnegative")
    if not math.isfinite(distance_squared) or distance_squared < 0:
        raise ValueError("Squared descriptor distance must be finite and nonnegative")
    kinetic = energy / (1.0 + distance_squared)
    return kinetic_bias_weight(kinetic), kinetic
