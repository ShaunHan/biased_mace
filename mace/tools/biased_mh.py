"""Small reusable launcher for independent Biased-MACE walkers and one MH server.

Optional dependency: the accompanying ase_mh integration. No model is loaded on
rank zero. Run identity and calibration are fixed; only completed minima restart.
"""
import hashlib
import json
import os
import time
from pathlib import Path


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime():
    """Time remaining in THIS allocation, measured after calculator initialization."""
    seconds = os.environ.get("BMH_RUNTIME_SECONDS")
    if seconds is not None:
        remaining = int(seconds)
    elif "SLURM_JOB_ID" in os.environ:
        end = os.environ.get("SLURM_JOB_END_TIME")
        if end is None:
            import datetime
            import subprocess
            text = subprocess.check_output(
                ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
            value = next(x.split("=", 1)[1] for x in text.split() if x.startswith("EndTime="))
            end = datetime.datetime.fromisoformat(value).timestamp()
        remaining = int(float(end) - time.time()) - 900  # Scheduler/checkpoint margin, seconds.
    else:
        return "infinite"
    if remaining <= 0:
        raise RuntimeError("No runtime remains after the shutdown margin; increase allocation time")
    days, rem = divmod(remaining, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_biased_mh(*, starts, targets, model, output, bias_weight=1.0,
                  use_adaptive_bias=True, head=None, device="cuda", enable_cueq=True,
                  totalsteps=100000, seed=1234, **mh_parameters):
    """Run unchanged for first start/resubmission; errors terminate MPI nonzero.

    starts/targets are ordered filenames, one pair per walker. Numeric bias_weight
    is dimensionless in adaptive mode and eV in fixed mode. totalsteps is a
    cumulative per-worker distinct-proposal limit, not a new budget on each job.
    """
    from mpi4py import MPI
    try:
        return _run(starts, targets, model, output, bias_weight, use_adaptive_bias,
                    head, device, enable_cueq, totalsteps, seed, mh_parameters)
    except BaseException as exc:
        import sys
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(code or 1)
        raise


def _run(starts, targets, model, output, bias_weight, adaptive, head, device,
         cueq, totalsteps, seed, parameters):
    import fcntl
    import inspect
    import numpy as np
    import torch
    from mpi4py import MPI
    from ase.io import read
    from mace.calculators import BiasedMACECalculator, mace_polar
    from minimahopping.minhop import Minimahopping
    from minimahopping.mh.restart import atomic_json, atomic_write
    from mace.tools.adaptive_bias import validate_bias_weight

    validate_bias_weight(bias_weight)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    comm = MPI.COMM_WORLD
    worker = comm.size == 1 or comm.rank != 0
    offset = int(os.environ.get("BMH_WALKER_OFFSET", "0"))
    count = max(1, comm.size - 1)
    index = offset + (comm.rank - 1 if comm.size > 1 else 0)
    starts, targets = list(map(Path, starts)), list(map(Path, targets))
    if len(starts) != len(targets) or offset < 0 or offset + count > len(starts):
        raise ValueError("Need one valid start/target pair per requested worker")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    try:
        if comm.rank == 0:
            lock = (output / "run.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pairs = [(str(a.resolve()), _hash(a), str(b.resolve()), _hash(b))
                     for a, b in zip(starts[offset:offset+count], targets[offset:offset+count])]
            signature = dict(schema=2, model_sha256=_hash(model), head=head,
                pairs=pairs, seed=seed, ranks=comm.size, bias_weight=bias_weight,
                use_adaptive_bias=adaptive, protocol=BiasedMACECalculator.auto_bias_protocol, enable_cueq=cueq,
                parameters={k: v for k, v in parameters.items() if k not in
                            ("run_time", "logLevel", "verbose_output")})
            # Normalize tuples/NumPy-free JSON before comparison.
            signature = json.loads(json.dumps(signature, allow_nan=False))
            manifest = output / "run_manifest.json"
            if manifest.exists():
                if json.loads(manifest.read_text()) != signature:
                    raise ValueError("Run definition changed. Keep model/targets/bias/ranks fixed on restart or use a new output directory")
            elif (output / "output").exists():
                raise ValueError("Legacy output has no v2 run manifest. Use a new directory for the new bias/acceptance protocol")
            else:
                atomic_json(manifest, signature)
            print(f"RUN_DIRECTORY {output}; model_sha256={signature['model_sha256']}", flush=True)
        comm.Barrier()
        initial = read(starts[index if worker else offset], parallel=False)
        physical = biased = None
        local = comm.Split_type(MPI.COMM_TYPE_SHARED)
        flags = local.allgather(worker)
        ordinal = sum(flags[:local.rank])
        gpu = None
        if worker and device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            ordinal %= torch.cuda.device_count()
            device = f"cuda:{ordinal}"
            torch.cuda.set_device(ordinal)
            gpu = str(getattr(torch.cuda.get_device_properties(ordinal), "uuid", ordinal))
        assigned = local.allgather(gpu)
        allowed = int(os.environ.get("BMH_WALKERS_PER_GPU", "1"))
        if any(assigned.count(g) > allowed for g in assigned if g is not None):
            raise RuntimeError(f"GPU sharing exceeds BMH_WALKERS_PER_GPU={allowed}")
        for slot in range(local.size):
            if local.rank == slot and worker:
                target = read(targets[index], parallel=False)
                for atoms in (initial, target):
                    atoms.info.update(charge=0.0, spin=1.0, external_field=np.zeros(3))
                if initial.pbc.any() or target.pbc.any():
                    raise ValueError("This molecular launcher expects nonperiodic structures")
                np.random.seed(seed + index)
                torch.manual_seed(seed + index)
                physical = mace_polar(model=str(model), head=head, device=device,
                    enable_cueq=cueq and device.startswith("cuda"), default_dtype="float64",
                    pbc_handling="realspace", compute_stress=False)
                if head is not None and physical.head != head:
                    raise ValueError(f"Requested head {head!r} was not selected: {physical.head!r}")
                biased = BiasedMACECalculator(models=physical.models, model_type="PolarMACE",
                    head=physical.head, target_atoms=target, reference_atoms=initial,
                    bias_weight=bias_weight, use_adaptive_bias=adaptive, device=device,
                    default_dtype="float64", pbc_handling="realspace", compute_stress=False)
                metric = output / "bias_metrics" / f"rank_{comm.rank}.pt"
                if metric.exists():
                    biased.load_bias_state_dict(torch.load(metric, map_location="cpu", weights_only=False))
                else:
                    atomic_write(metric, lambda stream: torch.save(biased.bias_state_dict(), stream))
                initial.calc = physical
                print(f"WALKER rank={comm.rank} index={index} {starts[index].name} -> {targets[index].name} "
                      f"device={device} bias_weight={bias_weight} adaptive={adaptive}", flush=True)
                print(f"SOURCES {inspect.getfile(BiasedMACECalculator)} | {inspect.getfile(Minimahopping)}", flush=True)
                import resource
                gpu_mib = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0.0
                print(f"MEMORY rank={comm.rank} host_peak_MiB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} "
                      f"torch_device_peak_MiB={gpu_mib:.1f}", flush=True)
            local.Barrier()
        local.Free()
        os.chdir(output)
        comm.Barrier()
        parameters = dict(parameters, use_MPI=comm.size > 1)
        if "run_time" not in parameters:
            parameters["run_time"] = _runtime()
        with Minimahopping(initial, md_calculator=biased, **parameters) as mh:
            mh(totalsteps=totalsteps)
        comm.Barrier()
    finally:
        if lock is not None:
            lock.close()
