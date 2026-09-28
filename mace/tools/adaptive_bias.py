"""Match a harmonic target bias to the existing MH escape energy scale."""

import math


def kinetic_bias_weight(kinetic_energy: float) -> float:
    """Return w=2*K (eV), with the reference descriptor distance fixed to 1.

    K is the escape kinetic energy assigned by the driver, in eV. MH already
    adapts that energy through its temperature/visited-minima feedback. This
    adds no second feedback loop or fitted constants. It is an energy-scale
    heuristic, not a barrier estimator or an optimal-weight guarantee.
    """
    energy = float(kinetic_energy)
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("Escape kinetic energy must be finite and nonnegative")
    weight = 2.0 * energy
    if not math.isfinite(weight):
        raise ValueError("Escape kinetic energy overflows the bias weight")
    return weight
