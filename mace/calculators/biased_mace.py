"""Conservative target bias for eager MACE energy models and committees."""

import math
from copy import deepcopy

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from mace.calculators.mace import MACECalculator
from mace.modules.global_descriptor import (
    InvariantMomentDescriptor,
    capture_mace_features,
    model_feature_specs,
)
from mace.modules.target_bias import (
    MomentTargetMetric,
    TargetDescriptorMetric,
    biased_autograd,
    target_potential,
)
from mace.tools import torch_tools
from mace.tools.adaptive_bias import partition_bias_energy, validate_bias_weight


class BiasedMACECalculator(MACECalculator):
    """Conservative target restraint ``V = w d^2 / 2`` for eager MACE models.

    ``bias_weight`` is always a float.  In fixed mode it is the harmonic
    coefficient ``w`` in eV.  In adaptive mode it is the dimensionless
    guidance/exploration ratio ``V(d=1) / K``.  An external driver supplies
    an energy scale through :meth:`adapt_bias`; the calculator then fixes ``w``
    until :meth:`adapt_bias` is called again.
    """

    adaptive_bias_protocol = "energy_partition_ratio_v1"

    def __init__(
        self,
        *args,
        target_atoms,
        bias_weight=0.0,
        use_adaptive_bias=False,
        reference_atoms=None,
        store_descriptor=False,
        **kwargs,
    ):
        validate_bias_weight(bias_weight)
        if type(use_adaptive_bias) is not bool:
            raise TypeError("use_adaptive_bias must be True or False")
        self._adaptive = use_adaptive_bias
        self._bias_ready = False
        if kwargs.get("compile_mode") is not None:
            raise ValueError(
                "Target bias currently requires eager inference (compile_mode=None)"
            )
        if kwargs.get("compute_atomic_stresses", False) or kwargs.get(
            "compute_bec", False
        ):
            raise ValueError("Target bias does not define atomic stresses or BECs")
        super().__init__(*args, **kwargs)
        if kwargs.get("head") is not None and kwargs["head"] != self.head:
            raise ValueError(
                f"Requested head {kwargs['head']!r} is not in {self.available_heads}"
            )
        if self.pad_num_atoms or self.pad_num_edges:
            raise ValueError("Target bias currently requires unpadded graphs")
        if "energy" not in self.implemented_properties:
            raise ValueError("Target bias requires an energy-producing MACE model")
        if self.length_units_to_A != 1.0:
            raise ValueError(
                "Use MACE models with Angstrom length units for target bias"
            )
        if not math.isfinite(self.energy_units_to_eV) or self.energy_units_to_eV <= 0:
            raise ValueError("energy_units_to_eV must be finite and positive")
        self.store_descriptor = bool(store_descriptor)
        for model in self.models:
            if isinstance(model, torch.jit.ScriptModule):
                raise ValueError(
                    "Use an eager .model checkpoint; feature hooks require eager modules"
                )
            model.eval()
        self._bias_weight = 0.0
        self.bias_weight = bias_weight
        # A global bias has no unique atomic partition. Do not present physical
        # per-atom energies as the decomposition of the biased total.
        self.implemented_properties = [
            p
            for p in self.implemented_properties
            if p not in ("energies", "node_energy")
        ]
        if use_adaptive_bias and bias_weight > 0 and reference_atoms is None:
            raise ValueError("Adaptive bias requires distinct reference_atoms")
        self.set_target(target_atoms, reference_atoms)

    @property
    def bias_weight(self):
        """User setting: eV in fixed mode, dimensionless ratio in adaptive mode."""
        return self._bias_strength

    @bias_weight.setter
    def bias_weight(self, value):
        self._bias_strength = validate_bias_weight(value)
        self._bias_weight = 0.0 if self._adaptive else value
        self._adaptive_ready = not self._adaptive or value == 0.0
        self.reset()

    @property
    def use_adaptive_bias(self):
        return self._adaptive

    @use_adaptive_bias.setter
    def use_adaptive_bias(self, value):
        if type(value) is not bool:
            raise TypeError("use_adaptive_bias must be True or False")
        if (value and self.bias_weight > 0 and getattr(self, "_bias_ready", False)
                and not self._normalized_reference):
            raise ValueError("Adaptive bias requires distinct reference_atoms")
        self._adaptive = value
        self.bias_weight = self.bias_weight  # Invalidate cached energies and adaptive state.

    @property
    def current_bias_weight(self):
        """Actual harmonic coefficient ``w`` in eV used by the calculator."""
        return self._bias_weight

    def _probe_target(self, atoms, gradients=False):
        """Independent diagnostic: no mutation of ASE result caches or adaptive state."""
        self._validate_structure(atoms)
        groups = self._groups_for_atoms(atoms)
        distances, energies, unit_forces = [], [], []
        with torch.enable_grad(), torch_tools.default_dtype(self.default_dtype):
            for index, model in enumerate(self.models):
                batch = self._prepare_reference(atoms)
                self._validate_context(index, batch)
                batch["positions"].requires_grad_(True)
                out, features = self._evaluate_features(model, batch, {
                    "compute_force": False, "compute_stress": False,
                    "compute_virials": False, "training": False,
                })
                with torch.set_grad_enabled(gradients):
                    moments = self._descriptors[index].moment_factors(features, groups)
                    if self._moment_metrics[index].compatible:
                        squared = self._moment_metrics[index](moments)
                    else:
                        descriptor = torch.cat(
                            self._descriptors[index].segments_from_moments(moments), -1)
                        squared = self._metrics[index](descriptor)
                if gradients:
                    unit = 0.5 * squared.sum() + 0.0 * batch["positions"].sum()
                    derivative, = torch.autograd.grad(unit, batch["positions"])
                    unit_forces.append(-derivative.detach().cpu().numpy())
                distances.append(float(squared.detach().sum()))
                energies.append(float(out["energy"].detach().sum()) * self.energy_units_to_eV)
                del out, features, batch, moments, squared
        squared = float(np.mean(distances))
        if not math.isfinite(squared) or squared < 0:
            raise FloatingPointError("Invalid descriptor distance")
        return squared, float(np.mean(energies)), (np.mean(unit_forces, axis=0) if gradients else None)

    def _distance_squared(self, atoms):
        return self._probe_target(atoms)[0]

    def get_bias_energy(self, atoms):
        """Bias on an arbitrary geometry at the SAME current weight (acceptance)."""
        if not self._adaptive_ready:
            raise RuntimeError("Call adapt_bias before evaluating an adaptive bias")
        if self.current_bias_weight == 0:
            return 0.0
        return 0.5 * self.current_bias_weight * self._distance_squared(atoms)

    def get_bias_diagnostics(self, atoms, *, forces=False):
        """Report progress and, optionally, the exact bias-only force at a boundary.

        Force diagnostics cost one extra reverse pass per committee member. They
        are not used as a force cap, a learned correction, or a dynamics update.
        """
        if not self._adaptive_ready:
            raise RuntimeError("Call adapt_bias before bias diagnostics")
        squared, energy, unit_force = self._probe_target(atoms, gradients=forces)
        result = dict(distance=math.sqrt(squared), distance_squared=squared,
                      physical_energy=energy, bias_energy=0.5*self.current_bias_weight*squared,
                      weight=self.current_bias_weight)
        if forces:
            force = self.current_bias_weight * unit_force
            result.update(bias_force_rms=float(np.sqrt(np.mean(force**2))),
                          bias_force_max=float(np.linalg.norm(force, axis=1).max(initial=0.0)))
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("Nonfinite bias diagnostic")
        return result

    def adapt_bias(self, atoms, *, energy_scale):
        """Configure one fixed adaptive bias from an external energy scale.

        In adaptive mode ``bias_weight`` is ``gamma = V(d=1)/K``.  At the
        supplied geometry, with squared target distance ``d0^2``,

            K = E / (1 + gamma d0^2),
            w = 2 gamma K.

        The method returns ``K``.  It does not modify positions, velocities, or
        any sampling state, and it has no dependency on a particular driver.
        Fixed mode simply returns ``energy_scale`` and keeps ``w=bias_weight``.
        """
        self._validate_structure(atoms)
        energy_scale = float(energy_scale)
        if not math.isfinite(energy_scale) or energy_scale < 0.0:
            raise ValueError("energy_scale must be finite and nonnegative")
        if self._adaptive and self.bias_weight > 0.0 and not self._normalized_reference:
            raise ValueError("Adaptive bias requires distinct reference_atoms")

        self._adaptive_ready = False
        self.reset()
        distance_squared = self._distance_squared(atoms)
        if self._adaptive:
            weight, motion_energy = partition_bias_energy(
                energy_scale, distance_squared, self.bias_weight
            )
        else:
            weight, motion_energy = self.bias_weight, energy_scale

        self._bias_weight = weight
        self.adaptive_distance_squared = distance_squared
        self.adaptive_motion_energy = motion_energy
        self._adaptive_ready = True
        return motion_energy

    def set_electrostatic_pbcs(self, pbc_handling):
        if getattr(self, "_bias_ready", False):
            raise ValueError(
                "Recreate the biased calculator after changing electrostatic boundary conditions"
            )
        super().set_electrostatic_pbcs(pbc_handling)

    def _prepare_reference(self, atoms):
        if self.model_type == "PolarMACE":
            self._validate_electrostatic_pbcs(atoms)
        batch = self._atoms_to_batch(atoms).to_dict()
        if self.external_field is not None:
            batch["external_field"] = torch.as_tensor(
                self.external_field,
                dtype=batch["positions"].dtype,
                device=batch["positions"].device,
            ).reshape(1, 3)
        return batch

    def _evaluate_features(self, model, batch, kwargs):
        with capture_mace_features(model) as features:
            out = model(batch, **kwargs)
        for key in ("potential_features", "spin_charge_density"):
            if out.get(key) is not None:
                features[key] = out[key]
        return out, features

    def _groups_for_atoms(self, atoms):
        if self._group_numbers is None or not np.array_equal(
            atoms.numbers, self._group_numbers
        ):
            self._group_numbers = atoms.numbers.copy()
            self._current_groups = self._descriptors[0].atom_groups(
                atoms.numbers, device=self.device
            )
        return self._current_groups

    def _clone_batch(self, batch):
        # A fresh, unpadded eager graph has exactly one consumer in this case.
        return batch if self.num_models == 1 else super()._clone_batch(batch)

    def set_target(self, target_atoms, reference_atoms=None):
        """Copy target and freeze the metric. This defines a NEW bias protocol."""
        if len(target_atoms) == 0:
            raise ValueError("target_atoms is empty")
        self._bias_ready = False
        self._target_atoms = target_atoms.copy()
        self._composition = np.sort(target_atoms.numbers)
        self._target_pbc = target_atoms.pbc.copy()
        if reference_atoms is not None:
            self._validate_structure(reference_atoms)
        self._descriptors = [
            InvariantMomentDescriptor(
                model_feature_specs(m),
                self._composition,
            )
            for m in self.models
        ]
        self._metrics = []
        self._moment_metrics = []
        self._group_numbers = None
        self._last_context = None
        self._reference_contexts = []
        self.target_energy = 0.0
        with torch.enable_grad(), torch_tools.default_dtype(self.default_dtype):
            for index, model in enumerate(self.models):
                batch = self._prepare_reference(self._target_atoms)
                out, features = self._evaluate_features(
                    model,
                    batch,
                    {
                        "compute_force": False,
                        "compute_stress": False,
                        "compute_virials": False,
                        "training": False,
                    },
                )
                with torch.no_grad():
                    target_moments = self._descriptors[index].moment_factors(
                        features, self._groups_for_atoms(self._target_atoms)
                    )
                    target = self._descriptors[index].segments_from_moments(
                        target_moments
                    )
                target_stats = self._descriptors[index].calibration_statistics(features)
                reference_stats = target_stats
                context = {
                    k: batch[k].detach().clone()
                    for k in ("total_charge", "total_spin", "external_field")
                    if k in batch
                }
                self._reference_contexts.append(context)
                self.target_energy += (
                    float(out["energy"].detach().sum())
                    * self.energy_units_to_eV
                    / self.num_models
                )
                # Only fixed moments survive calibration. Release the target
                # forward graph before allocating the reference forward graph.
                del out, features, batch
                reference = None
                if reference_atoms is not None:
                    refbatch = self._prepare_reference(reference_atoms)
                    self._validate_context(index, refbatch)
                    refout, reffeatures = self._evaluate_features(
                        model,
                        refbatch,
                        {
                            "compute_force": False,
                            "compute_stress": False,
                            "compute_virials": False,
                            "training": False,
                        },
                    )
                    with torch.no_grad():
                        reference = self._descriptors[index].segments_from_moments(
                            self._descriptors[index].moment_factors(
                                reffeatures, self._groups_for_atoms(reference_atoms)
                            )
                        )
                    reference_stats = self._descriptors[index].calibration_statistics(
                        reffeatures
                    )
                    del refout, reffeatures, refbatch
                scales = []
                for (power_t, order), (power_r, _) in zip(
                    target_stats, reference_stats
                ):
                    sigma = ((power_t + power_r) / 2).sqrt()
                    # An identically zero tensor has no calibratable amplitude.
                    sigma = torch.where(
                        sigma > torch.finfo(sigma.dtype).eps ** 0.5,
                        sigma,
                        torch.ones_like(sigma),
                    )
                    scales.append(sigma**order)
                metric = TargetDescriptorMetric(target, reference, scales)
                self._metrics.append(metric)
                self._moment_metrics.append(
                    MomentTargetMetric(target_moments[0], metric)
                )
        self._normalized_reference = reference_atoms is not None
        if self._adaptive and self.bias_weight > 0:
            self._adaptive_ready = False
        self._bias_ready = True
        self.reset()

    @property
    def target_atoms(self):
        return self._target_atoms.copy()

    def _validate_structure(self, atoms):
        if not np.array_equal(np.sort(atoms.numbers), self._composition):
            raise ValueError("Target bias requires identical elemental composition")
        if not np.array_equal(atoms.pbc, self._target_pbc):
            raise ValueError(
                "Target and current structures must use identical PBC flags"
            )

    def _validate_context(self, index, batch):
        for key, value in self._reference_contexts[index].items():
            if (
                key not in batch
                or batch[key].numel() != value.numel()
                or not torch.allclose(
                    batch[key].reshape(-1).to(value),
                    value.reshape(-1),
                    rtol=0,
                    atol=1e-12,
                )
            ):
                raise ValueError(f"{key} changed: recreate/recalibrate the target bias")

    def check_state(self, atoms, tol=1e-15):
        # Upstream ignores numpy-valued info entries. They include external_field.
        state = super().check_state(atoms, tol)
        if self.atoms is not None:
            for key in self.info_keys.values():
                a, b = self.atoms.info.get(key), atoms.info.get(key)
                if not np.array_equal(a, b) and "info" not in state:
                    state.append("info")
            for key in self.arrays_keys.values():
                if not np.array_equal(
                    self.atoms.arrays.get(key), atoms.arrays.get(key)
                ):
                    if "arrays" not in state:
                        state.append("arrays")
        return state

    def _model_forward(self, model, batch_dict, model_kwargs):
        index = next(i for i, candidate in enumerate(self.models) if candidate is model)
        if not self._context_unchanged:
            self._validate_context(index, batch_dict)
        positions = batch_dict["positions"]
        positions.requires_grad_(True)
        cell = batch_dict["cell"]
        stress_requested = bool(model_kwargs.get("compute_stress"))
        kwargs = dict(
            model_kwargs,
            compute_force=False,
            compute_stress=False,
            compute_virials=False,
            compute_displacement=stress_requested,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
            training=False,
        )
        with torch.enable_grad():
            out, features = self._evaluate_features(model, batch_dict, kwargs)
            moments = self._descriptors[index].moment_factors(
                features, self._current_groups
            )
            descriptor = None
            if self.store_descriptor or not self._moment_metrics[index].compatible:
                descriptor = torch.cat(
                    self._descriptors[index].segments_from_moments(moments), dim=-1
                )
                distance_squared = self._metrics[index](descriptor)
            else:
                distance_squared = self._moment_metrics[index](moments)
            unit = target_potential(distance_squared)
            physical_energy = out["energy"]
            total, forces, stress, virials = biased_autograd(
                physical_energy,
                unit,
                self.current_bias_weight / self.energy_units_to_eV,
                positions,
                out.get("displacement") if stress_requested else None,
                cell,
                batch_dict.get("pbc"),
            )
        out = dict(out, energy=total, forces=forces, stress=stress, virials=virials)
        self._bias_records.append(
            {
                "values": torch.stack(
                    [
                        physical_energy.detach().sum() * self.energy_units_to_eV,
                        unit.detach().sum(),
                        distance_squared.detach().sum(),
                    ]
                ),
                "global_descriptor": (
                    descriptor.detach() if self.store_descriptor else None
                ),
            }
        )
        return out

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if not self._adaptive_ready:
            raise RuntimeError(
                "Adaptive bias is not configured: call adapt_bias() first"
            )
        atoms = self.atoms if atoms is None else atoms
        self._validate_structure(atoms)
        self._groups_for_atoms(atoms)
        context = {
            **{("info", k): atoms.info.get(k) for k in self.info_keys.values()},
            **{("arrays", k): atoms.arrays.get(k) for k in self.arrays_keys.values()},
            ("calculator", "external_field"): self.external_field,
        }
        self._context_unchanged = self._last_context is not None and all(
            k in self._last_context and np.array_equal(v, self._last_context[k])
            for k, v in context.items()
        )
        self._bias_records = []
        with torch.enable_grad():
            super().calculate(atoms, properties, system_changes)
        self._last_context = deepcopy(context)
        values = (
            torch.stack([r["values"] for r in self._bias_records]).mean(0).cpu().numpy()
        )
        for key, value in zip(
            ("unbiased_energy", "unit_bias_energy", "bias_distance_squared"), values
        ):
            self.results[key] = float(value)
        self.results["bias_distance"] = float(
            np.sqrt(self.results["bias_distance_squared"])
        )
        self.results["bias_energy"] = (
            self.current_bias_weight * self.results["unit_bias_energy"]
        )
        self.results["bias_weight"] = self.current_bias_weight
        self.results["target_energy"] = self.target_energy
        # Latent coordinates from independently trained models have no shared
        # basis: keep each descriptor separate; average energies, never features.
        if self.store_descriptor:
            ds = [r["global_descriptor"].cpu().numpy()[0] for r in self._bias_records]
            self.results["global_descriptor"] = ds[0] if self.num_models == 1 else ds
        self._bias_records.clear()
        self.results.pop("energies", None)
        if "node_energy" in self.results:
            self.results["unbiased_node_energy"] = self.results.pop("node_energy")
        for key in ("energy", "forces", "unit_bias_energy"):
            if not np.isfinite(self.results[key]).all():
                self.results.clear()
                raise FloatingPointError(f"Nonfinite biased result: {key}")

    def get_global_descriptor(self, atoms):
        """Return invariant descriptors; this is an ordinary cached evaluation."""
        if not self.store_descriptor:
            raise ValueError(
                "Construct with store_descriptor=True to export descriptors"
            )
        self.get_potential_energy(atoms)
        return deepcopy(self.results["global_descriptor"])

    def get_hessian(self, atoms=None):
        raise NotImplementedError(
            "Biased Hessians are not exposed; finite-difference the biased forces if needed"
        )

    def bias_state_dict(self):
        """Return the target metric and bias settings for reproducible reuse."""
        return {
            "bias_weight": self.bias_weight,
            "use_adaptive_bias": self.use_adaptive_bias,
            "current_bias_weight": self.current_bias_weight,
            "adaptive_ready": self._adaptive_ready,
            "normalized_reference": self._normalized_reference,
            "protocol": self.adaptive_bias_protocol,
            "metrics": [deepcopy(metric.state_dict()) for metric in self._metrics],
        }

    def load_bias_state_dict(self, state):
        if state.get("protocol") != self.adaptive_bias_protocol:
            raise ValueError("Incompatible adaptive-bias state")
        if state.get("use_adaptive_bias") != self.use_adaptive_bias:
            raise ValueError("Adaptive-bias mode differs from saved state")
        if float(state.get("bias_weight")) != self.bias_weight:
            raise ValueError("bias_weight differs from saved state")
        if len(state["metrics"]) != len(self._metrics):
            raise ValueError("Committee size differs from saved state")
        for metric, saved in zip(self._metrics, state["metrics"]):
            if not torch.allclose(
                metric.target, saved["target"].to(metric.target), rtol=1e-9, atol=1e-12
            ):
                raise ValueError("Target descriptor differs from saved state")
            metric.load_state_dict(saved)
        for metric in self._moment_metrics:
            metric.refresh_scales()
        weight = float(state["current_bias_weight"])
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Invalid saved bias coefficient")
        self._bias_weight = weight
        self._adaptive_ready = bool(state["adaptive_ready"])
        self._normalized_reference = bool(state["normalized_reference"])
        self.reset()

