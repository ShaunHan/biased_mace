"""A dimensionless guidance/exploration ratio sharing one MH energy budget."""
import math


def validate_bias_weight(value):
    """Only floating-point values are accepted; strings, bools and ints are errors."""
    if not isinstance(value, float):
        raise TypeError("bias_weight must be a float, e.g. 1.0; strings/'auto' and integers are not accepted")
    if not math.isfinite(value) or value < 0:
        raise ValueError("bias_weight must be finite and nonnegative")
    return value


def partition_escape_energy(energy: float, distance_squared: float, bias_weight: float = 1.0):
    """Return (w [eV], K [eV]) for V=w*d**2/2 and V(d=1)/K=bias_weight.

    K=E/(1+bias_weight*d_start**2), w=2*bias_weight*K.
    The weight is evaluated only at an escape boundary, never inside MD.
    Equal allocation at unit distance is one choice (1.0), not equipartition.
    """
    gamma = validate_bias_weight(bias_weight)
    energy, squared = float(energy), float(distance_squared)
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("Escape energy must be finite and nonnegative")
    if not math.isfinite(squared) or squared < 0:
        raise ValueError("Squared target distance must be finite and nonnegative")
    if gamma == 0 or energy == 0:
        return 0.0, energy
    # Scaled algebra avoids overflow in gamma*d_start**2.
    scale = max(1.0, gamma)
    denominator = 1.0 / scale + (gamma / scale) * squared
    kinetic = (energy / scale) / denominator
    weight = 2.0 * ((energy * (gamma / scale)) / denominator)
    if not math.isfinite(weight):
        raise ValueError("Bias energy scale overflows floating point")
    return weight, kinetic
