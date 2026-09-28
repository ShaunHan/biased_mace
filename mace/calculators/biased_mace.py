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
    TargetDescriptorMetric,
    biased_autograd,
    target_potential,
)
from mace.tools import torch_tools
from mace.tools.adaptive_bias import kinetic_bias_weight


class BiasedMACECalculator(MACECalculator):
    """E(R) = U(R) + w*d(R,target)^2/2, with exact autograd forces.

    bias_weight: nonnegative eV, or "auto". Auto uses begin_hop() to match
    the bias energy scale to the MH escape kinetic energy. The weight stays
    fixed during each hop, including softening, MD and preliminary relaxation.
    Patched MH calls begin_hop automatically; ordinary ASE drivers may call it
    at explicit trajectory boundaries. No new trainable parameters are added.
    """

    def __init__(
        self,
        *args,
        target_atoms,
        bias_weight=0.0,
        reference_atoms=None,
        store_descriptor=False,
        **kwargs,
    ):
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
        if bias_weight == "auto" and reference_atoms is None:
            raise ValueError("bias_weight=auto requires distinct reference_atoms")
        self.set_target(target_atoms, reference_atoms)

    @property
    def bias_weight(self):
        return "auto" if self._automatic else self._bias_weight

    @bias_weight.setter
    def bias_weight(self, value):
        if isinstance(value, str) and value == "auto":
            self._automatic, self._hop_ready = True, False
            self._bias_weight = 0.0
        else:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("bias_weight must be 'auto' or a nonnegative eV value")
            self._automatic, self._hop_ready = False, True
            self._bias_weight = value
        self.reset()

    @property
    def current_bias_weight(self):
        """The numeric weight (eV) used for the current hop."""
        return self._bias_weight

    def begin_hop(self, atoms, *, kinetic_energy):
        """Called once before softening/MD, never from energy/force evaluation.

        With d(reference,target)=1, w=2*K makes the bias at the reference K.
        Fixed float weights are left unchanged. Repeated ASE property requests
        therefore always see the same energy and force for the same geometry.
        """
        self._validate_structure(atoms)
        if self._automatic:
            if not self._normalized_reference:
                raise ValueError("Auto bias requires distinct reference_atoms")
            self._bias_weight = kinetic_bias_weight(kinetic_energy)
            self._hop_ready = True
            self.reset()
        return self._bias_weight

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

    def _segments(self, model_index, features, batch):
        numbers = self.models[model_index].atomic_numbers[
            batch["node_attrs"].argmax(-1)
        ]
        return self._descriptors[model_index].segments(
            features, numbers, batch["batch"]
        )

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
                target = [t.detach() for t in self._segments(index, features, batch)]
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
                reference = None
                if reference_atoms is not None:
                    refbatch = self._prepare_reference(reference_atoms)
                    self._validate_context(index, refbatch)
                    _, reffeatures = self._evaluate_features(
                        model,
                        refbatch,
                        {
                            "compute_force": False,
                            "compute_stress": False,
                            "compute_virials": False,
                            "training": False,
                        },
                    )
                    reference = [
                        t.detach() for t in self._segments(index, reffeatures, refbatch)
                    ]
                    reference_stats = self._descriptors[index].calibration_statistics(
                        reffeatures
                    )
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
                self._metrics.append(TargetDescriptorMetric(target, reference, scales))
        self._normalized_reference = reference_atoms is not None
        if self._automatic:
            self._hop_ready = False
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
            descriptor = torch.cat(self._segments(index, features, batch_dict), dim=-1)
            distance_squared = self._metrics[index](descriptor)
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
                "unbiased_energy": physical_energy.detach().sum().item()
                * self.energy_units_to_eV,
                "unit_bias_energy": unit.detach().sum().item(),
                "bias_distance_squared": distance_squared.detach().sum().item(),
                **(
                    {"global_descriptor": descriptor.detach().cpu().numpy()[0]}
                    if self.store_descriptor
                    else {}
                ),
            }
        )
        return out

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if not self._hop_ready:
            raise RuntimeError(
                "Auto bias is not initialized: call begin_hop or use patched MH"
            )
        atoms = self.atoms if atoms is None else atoms
        self._validate_structure(atoms)
        self._bias_records = []
        with torch.enable_grad():
            super().calculate(atoms, properties, system_changes)
        for key in (
            "unbiased_energy",
            "unit_bias_energy",
            "bias_distance_squared",
        ):
            self.results[key] = np.mean([r[key] for r in self._bias_records], axis=0)
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
            ds = [r["global_descriptor"] for r in self._bias_records]
            self.results["global_descriptor"] = ds[0] if self.num_models == 1 else ds
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
        """Metric for reproducible restarts with identical model/target/settings."""
        return {
            "bias_weight": self.bias_weight,
            "current_bias_weight": self.current_bias_weight,
            "hop_ready": self._hop_ready,
            "normalized_reference": self._normalized_reference,
            "metrics": [deepcopy(m.state_dict()) for m in self._metrics],
        }

    def load_bias_state_dict(self, state):
        if len(state["metrics"]) != len(self._metrics):
            raise ValueError("Committee size differs from restart state")
        for metric, saved in zip(self._metrics, state["metrics"]):
            if not torch.allclose(
                metric.target, saved["target"].to(metric.target), rtol=1e-9, atol=1e-12
            ):
                raise ValueError("Model/target descriptor differs from restart state")
            metric.load_state_dict(saved)
        self.bias_weight = state["bias_weight"]
        weight = float(state["current_bias_weight"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid saved bias weight")
        self._bias_weight = weight
        self._hop_ready = bool(state["hop_ready"])
        self._normalized_reference = bool(state["normalized_reference"])
        self.reset()
