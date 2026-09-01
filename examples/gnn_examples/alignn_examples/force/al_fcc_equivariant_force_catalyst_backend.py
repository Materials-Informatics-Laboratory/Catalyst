"""
Al FCC equivariant force-learning example using the Catalyst training backend.

Recommended location:
    catalyst/examples/gnn_examples/alignn_examples/force/al_fcc_equivariant_force_catalyst_backend.py

This example is the force-learning analogue of the Catalyst-backend energy
example. It does NOT define local train_one_epoch(...), evaluate(...), or manual
checkpoint/inference loops.

Training, validation, checkpointing, DDP, and inference are delegated to:

    catalyst.ml.training.run_training
    catalyst.ml.inference.run_inference
    catalyst.ml.gnn.GNN.GNN

The example owns only:
    - ASE/EMT MD data generation,
    - alignnd graph generation with equivariant fields,
    - train/validation/test sample dictionaries,
    - force target normalization,
    - equivariant vector model construction,
    - optional plotting of backend-generated inference outputs.

Model target
------------
This trains one vector-valued model head:

    pred shape   = [total_atoms_in_batch, 3]
    target shape = [total_atoms_in_batch, 3]

It does NOT train Fx, Fy, and Fz as separate scalar targets. The Catalyst loss is
a single tensor MSE over the full vector field.

Important config
----------------

    task = GNNTask.node_vector(
        target_key="target_vector",
        output_key="vector",
        accumulate_loss="node",
        vector_channels=1,
    )

    cat = Catalyst(
        parameter_file=CONFIG_PATH,
        task=task,
    )

and the model is built with:

    build_task_model(
        task=task,
        model_type="equivariant",
        return_dict=False,
        ...
    )

The task sets output_type="vector", output_level="node", and out_dim=1. The raw
equivariant vector decoder emits [N_atoms, 1, 3], and the task's vector adapter
squeezes that to [N_atoms, 3] before the Catalyst loss sees it.

Run
---
    CATALYST_AL_FCC_FORCE_CONFIG=al_fcc_equivariant_force_catalyst_config.json \\
    python al_fcc_equivariant_force_catalyst_backend.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet

from catalyst.data.utils import load_dictionary, save_dictionary
from catalyst.graph.alignnd import alignn_gen
from catalyst.ml.gnn.GNN import GNN
from catalyst.ml.gnn.tasks import GNNTask, build_task_model
from catalyst.ml.inference import run_inference
from catalyst.ml.training import run_training
from catalyst.ml.utils.distributed import cuda_destroy
from catalyst.observer.params import Catalyst


# =============================================================================
# CONFIGURATION
# =============================================================================


CONFIG_PATH = Path(
    os.environ.get(
        "CATALYST_AL_FCC_FORCE_CONFIG",
        Path(__file__).with_name("al_fcc_equivariant_force_catalyst_config.json"),
    )
)


def load_json_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Could not find Catalyst Al FCC force config file: {config_path}\n"
            "Set CATALYST_AL_FCC_FORCE_CONFIG=/path/to/config.json or place "
            "al_fcc_equivariant_force_catalyst_config.json next to this script."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


CONFIG = load_json_config()

WORKFLOW = CONFIG["workflow"]
AL_CONFIG = CONFIG["al_fcc"]
MODEL_CONFIG = CONFIG["model_architecture"]
FORCE_CONFIG = CONFIG["force_normalization"]
TRAINING_OVERRIDES = CONFIG["training_overrides"]

DEVICE = CONFIG["catalyst_parameters"]["device_dict"]["device"]

RUN_GENERATE_GRAPHS = WORKFLOW["generate_graphs"]
RUN_GENERATE_SAMPLES = WORKFLOW["generate_samples"]
RUN_NORMALIZE_TARGETS = WORKFLOW.get("normalize_targets", True)
RUN_TRAINING = WORKFLOW["train"]
RUN_RETRAINING = WORKFLOW["retrain"]
RUN_TESTING = WORKFLOW["test"]
RUN_PLOT_TRAINING = WORKFLOW["plot_training"]
RUN_PLOT_TEST = WORKFLOW["plot_test"]
RUN_PREDICTIONS = WORKFLOW["predictions"]

TRAINING_BATCH_SIZE = TRAINING_OVERRIDES["training_batch_size"]
TRAINING_NUM_EPOCHS_OVERRIDE = TRAINING_OVERRIDES["num_epochs"]
TRAINING_DELTA_OVERRIDE = TRAINING_OVERRIDES["train_delta"]
TRAINING_TOLERANCE_OVERRIDE = TRAINING_OVERRIDES["train_tolerance"]


def get_figures_dir(cat: Catalyst) -> Path:
    """Return the example figure directory, creating it if needed."""
    figures_dir = Path(cat.parameters["io_dict"]["main_path"]) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


# =============================================================================
# PARAMETER AND MODEL BUILDERS
# =============================================================================


def build_regression_task() -> GNNTask:
    """
    Build the generic task contract for this example.

    This is a node-level vector regression task. The physical meaning of the
    target is intentionally not encoded in the task name. The task learns one
    complete 3D vector per node, not three independent scalar targets.
    """
    return GNNTask.node_vector(
        target_key="target_vector",
        output_key="vector",
        accumulate_loss="node",
        vector_channels=1,
        squeeze_single_vector_channel=True,
    )






def latest_checkpoint(checkpoint_dir: Path, checkpoint_pattern: str = "checkpoint_epoch_*.pt") -> str:
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    epoch_pattern = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")
    checkpoint_matches = []

    for checkpoint_path in checkpoint_dir.glob(checkpoint_pattern):
        match = epoch_pattern.match(checkpoint_path.name)
        if match is None:
            continue
        checkpoint_matches.append((int(match.group(1)), checkpoint_path))

    if not checkpoint_matches:
        raise FileNotFoundError(
            f"No checkpoint files matching {checkpoint_pattern!r} were found in: {checkpoint_dir}"
        )

    latest_epoch, latest_path = max(checkpoint_matches, key=lambda item: item[0])
    print(f"Loading checkpoint from epoch {latest_epoch}: {latest_path}")
    return str(latest_path)




def build_regression_model(device: str = DEVICE) -> GNN:
    """
    Build one equivariant node_vector model wrapped in the high-level Catalyst GNN.

    The task interface owns the model-output contract:
        output_type="vector"
        output_level="node"
        out_dim=1

    The raw equivariant decoder emits:
        [total_atoms_in_batch, 1, 3]

    GNNTask.node_vector(...) wraps the model with VectorChannelAdapter, which
    squeezes that to:
        [total_atoms_in_batch, 3]

    That matches graph.target_vector and graph.y exactly.
    """
    task = build_regression_task()

    model = build_task_model(
        task=task,
        model_type="equivariant",
        return_dict=False,
        num_species=AL_CONFIG["num_species"],
        cutoff=AL_CONFIG["cutoff"],
        dim=MODEL_CONFIG["hidden_dim"],
        num_convs=MODEL_CONFIG["n_convs"],
        act=torch.nn.SiLU(),
    )

    return GNN(model=model, device=device)


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def reset_dir(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_dir(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def first_match(pattern: os.PathLike[str] | str) -> str:
    matches = glob.glob(str(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return matches[0]


def safe_torch_load(file_name: os.PathLike[str] | str, map_location: str | torch.device | None = None):
    try:
        return torch.load(file_name, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(file_name, map_location=map_location)


def as_numpy_tensor(value) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def run_distributed_or_single(cat: Catalyst, target, *args) -> None:
    if cat.parameters["device_dict"]["run_ddp"]:
        processes = []
        for rank in range(cat.parameters["device_dict"]["world_size"]):
            process = mp.Process(target=target, args=(rank, *args))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        cuda_destroy()
    else:
        target(0, *args)


# =============================================================================
# ASE MD / GRAPH GENERATION
# =============================================================================


def build_al_fcc_supercell():
    atoms = bulk(
        "Al",
        "fcc",
        a=AL_CONFIG["lattice_constant"],
        cubic=True,
    )
    atoms = atoms.repeat(tuple(AL_CONFIG["repeat"]))
    atoms.pbc = True
    atoms.calc = EMT()
    return atoms


def run_md_frames() -> list:
    atoms = build_al_fcc_supercell()

    MaxwellBoltzmannDistribution(atoms, temperature_K=AL_CONFIG["temperature_K"])
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = VelocityVerlet(
        atoms,
        timestep=AL_CONFIG["timestep_fs"] * units.fs,
    )

    frames = []
    force_rms_values = []

    for step in range(AL_CONFIG["md_steps"] + 1):
        if step > 0:
            dyn.run(1)

        if step < AL_CONFIG["equilibration_steps"]:
            continue

        if (step - AL_CONFIG["equilibration_steps"]) % AL_CONFIG["sample_every"] != 0:
            continue

        frame = atoms.copy()
        frame.calc = EMT()
        forces = frame.get_forces()
        force_rms = float(np.sqrt(np.mean(forces ** 2)))

        frames.append(frame)
        force_rms_values.append(force_rms)

    print(f"Generated {len(frames)} MD frames.")
    print(
        "Force RMS range: "
        f"{min(force_rms_values):.6f} to {max(force_rms_values):.6f} eV/Ang"
    )

    return frames


def _as_single_graph(obj):
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise RuntimeError("alignn_gen returned an empty list.")
        return obj[0]
    return obj


def finalize_graph_metadata(graph):
    """
    Make graphs safe for the equivariant model path.
    """
    if getattr(graph, "z", None) is not None:
        graph.num_nodes = int(graph.z.size(0))
    elif getattr(graph, "pos", None) is not None:
        graph.num_nodes = int(graph.pos.size(0))
    elif getattr(graph, "x_atm", None) is not None:
        graph.num_nodes = int(graph.x_atm.size(0))

    if getattr(graph, "edge_index", None) is None and getattr(graph, "edge_index_G", None) is not None:
        graph.edge_index = graph.edge_index_G

    if getattr(graph, "shifts", None) is None and getattr(graph, "edge_index", None) is not None:
        graph.shifts = torch.zeros(
            (graph.edge_index.size(1), 3),
            dtype=torch.long,
            device=graph.edge_index.device,
        )

    if getattr(graph, "pbc", None) is None:
        graph.pbc = torch.tensor([True, True, True], dtype=torch.bool)

    return graph


def atoms_to_equivariant_graph(atoms, forces_eVA: np.ndarray, gid: str):
    graph = alignn_gen(
        {
            "type": "alignnd",
            "raw_data": atoms,
            "node_labels": None,
            "element_list": ["Al"],
            "neighbor_params": [AL_CONFIG["cutoff"], AL_CONFIG["neighbor_k"]],
            "is_dihedral": AL_CONFIG.get("include_dihedrals", False),
            "store_raw_data": False,
            "use_pt": False,
            "include_angs": True,
            "cpu_cores": 1,
            "store_atoms_type": "ase-atoms",

            # Required for the equivariant backend.
            "include_equivariant_fields": True,
            "include_edge_geometry": True,

            # Retry controls.
            "auto_retry_graph": True,
            "max_graph_attempts": AL_CONFIG.get("max_graph_attempts", 6),
            "cutoff_scale": AL_CONFIG.get("cutoff_scale", 1.15),
            "max_cutoff": AL_CONFIG.get("max_cutoff", 5.0),
            "require_bonds": True,
            "require_angles": False,
            "require_dihedrals": False,
            "retry_verbose": False,
        }
    )

    graph = _as_single_graph(graph)
    graph = finalize_graph_metadata(graph)

    forces = torch.as_tensor(forces_eVA, dtype=torch.float32).reshape(-1, 3)

    graph.gid = gid

    # Raw force labels for reporting/denormalization.
    graph.forces_eVA = forces.clone()

    # Target fields consumed by the updated Catalyst GNN/predict stack.
    # normalize_targets(...) overwrites these with normalized values.
    graph.target_vector = forces.clone()
    graph.y = forces.clone()

    return graph


def generate_data(cat: Catalyst) -> None:
    data_dir = reset_dir(cat.parameters["io_dict"]["data_dir"])
    frames = run_md_frames()

    for idx, atoms in enumerate(frames):
        forces = atoms.get_forces()
        gid = f"al_fcc_md_force_{idx:05d}"
        force_rms = float(np.sqrt(np.mean(forces ** 2)))

        print(
            f"Building graph {idx + 1:4d}/{len(frames):4d}: "
            f"{gid}, force_rms={force_rms:.6f} eV/Ang"
        )

        graph = atoms_to_equivariant_graph(
            atoms=atoms,
            forces_eVA=forces,
            gid=gid,
        )

        torch.save(graph, data_dir / f"{gid}.pt")


# =============================================================================
# SAMPLING / NORMALIZATION
# =============================================================================


def sample_data(cat: Catalyst) -> None:
    data_files = sorted(glob.glob(os.path.join(cat.parameters["io_dict"]["data_dir"], "*.pt")))
    if not data_files:
        raise FileNotFoundError(
            f"No graph files found in data_dir={cat.parameters['io_dict']['data_dir']}"
        )

    samples_dir = reset_dir(Path(cat.parameters["io_dict"]["main_path"]) / "samples")
    model_samples_dir = reset_dir(samples_dir / "model_samples")
    cat.set_params({'io_dict': {'samples_dir': str(samples_dir)}}, save_params=False)

    gids = [Path(path).stem for path in data_files]
    rng = np.random.default_rng(CONFIG.get("sampling", {}).get("sampling_seed", 112358))
    indices = np.arange(len(gids))
    rng.shuffle(indices)

    test_fraction = float(CONFIG["sampling"]["test_fraction"])
    validation_fraction_of_remaining = float(CONFIG["sampling"]["validation_fraction_of_remaining"])

    n_total = len(indices)
    n_test = int(round(test_fraction * n_total))
    test_idx = indices[:n_test]
    remaining_idx = indices[n_test:]

    n_remaining = len(remaining_idx)
    n_validation = int(round(validation_fraction_of_remaining * n_remaining))
    validation_idx = remaining_idx[:n_validation]
    training_idx = remaining_idx[n_validation:]

    test_gids = [gids[i] for i in test_idx]
    training_gids = [gids[i] for i in training_idx]
    validation_gids = [gids[i] for i in validation_idx]

    save_dictionary(
        samples_dir / "test_data.npy",
        {
            "gids": test_gids,
            "projections": [],
        },
    )

    save_dictionary(
        model_samples_dir / "train_valid_split.npy",
        {
            "training": training_gids,
            "validation": validation_gids,
            "training_projections": [],
            "validation_projections": [],
        },
    )

    # Split-specific test_data.npy files so run_inference can be reused for
    # training, validation, and test parity plots without manual inference loops.
    split_for_inference = {
        "training": training_gids,
        "validation": validation_gids,
        "test": test_gids,
    }

    for split_name, split_gids in split_for_inference.items():
        split_samples_dir = reset_dir(samples_dir / f"inference_{split_name}")
        save_dictionary(
            split_samples_dir / "test_data.npy",
            {
                "gids": split_gids,
                "projections": [],
            },
        )

    with (samples_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_total": len(gids),
                "n_training": len(training_gids),
                "n_validation": len(validation_gids),
                "n_test": len(test_gids),
                "training": training_gids,
                "validation": validation_gids,
                "test": test_gids,
            },
            handle,
            indent=2,
        )

    print(
        "Dataset split: "
        f"train={len(training_gids)}, "
        f"validation={len(validation_gids)}, "
        f"test={len(test_gids)}"
    )


def _graph_path_from_gid(cat: Catalyst, gid: str) -> Path:
    return Path(cat.parameters["io_dict"]["data_dir"]) / f"{gid}.pt"


def compute_force_normalization_from_training(cat: Catalyst) -> tuple[torch.Tensor, torch.Tensor]:
    samples_file = Path(cat.parameters["io_dict"]["main_path"]) / "samples" / "model_samples" / "train_valid_split.npy"
    split = load_dictionary(samples_file)
    training_gids = split["training"]

    if not training_gids:
        raise RuntimeError("Training split is empty; cannot compute force normalization.")

    all_forces = []
    for gid in training_gids:
        graph = safe_torch_load(_graph_path_from_gid(cat, gid), map_location="cpu")
        all_forces.append(graph.forces_eVA.reshape(-1, 3).float())

    all_forces = torch.cat(all_forces, dim=0)

    mode = str(FORCE_CONFIG.get("mode", "scalar")).lower()

    if mode == "component":
        force_mean = all_forces.mean(dim=0)
        force_std = all_forces.std(dim=0)
    elif mode == "scalar":
        force_mean = all_forces.reshape(-1).mean().view(1)
        force_std = all_forces.reshape(-1).std().view(1)
    else:
        raise ValueError(f"force_normalization.mode must be 'scalar' or 'component', got {mode!r}.")

    force_std = torch.where(force_std < 1.0e-12, torch.ones_like(force_std), force_std)

    return force_mean.float(), force_std.float()


def normalize_targets(cat: Catalyst) -> None:
    """
    Normalize force vector targets using training-set statistics.

    This writes normalized graph.target_vector and graph.y with shape [N_atoms, 3].
    """
    force_mean, force_std = compute_force_normalization_from_training(cat)

    data_files = sorted(glob.glob(os.path.join(cat.parameters["io_dict"]["data_dir"], "*.pt")))
    for path in data_files:
        graph = safe_torch_load(path, map_location="cpu")

        forces = graph.forces_eVA.float().reshape(-1, 3)
        normalized = (forces - force_mean) / force_std

        graph.target_vector = normalized.clone()
        graph.y = normalized.clone()
        graph.force_normalization = {
            "force_mean": force_mean.cpu().tolist(),
            "force_std": force_std.cpu().tolist(),
            "force_units": "eV/Ang",
            "mode": str(FORCE_CONFIG.get("mode", "scalar")).lower(),
        }

        torch.save(graph, path)

    norm_data = {
        "force_mean": force_mean.cpu().tolist(),
        "force_std": force_std.cpu().tolist(),
        "force_units": "eV/Ang",
        "mode": str(FORCE_CONFIG.get("mode", "scalar")).lower(),
    }

    with (Path(cat.parameters["io_dict"]["main_path"]) / "force_normalization.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(norm_data, handle, indent=2)

    print("Force normalization mode:", norm_data["mode"])
    print("Force mean:", norm_data["force_mean"])
    print("Force std:", norm_data["force_std"])


# =============================================================================
# BACKEND TRAINING / INFERENCE
# =============================================================================


def train_model(cat: Catalyst) -> None:
    """
    Train using Catalyst's backend. No local training loop lives here.
    """
    cat.set_params({'io_dict': {'samples_dir': str(Path(cat.parameters['io_dict']['main_path']) / 'samples' / 'model_samples')}}, save_params=False)
    cat.set_model(build_regression_model(DEVICE))
    run_distributed_or_single(cat, run_training, cat)


def retrain_model(cat: Catalyst, use_latest_checkpoint: bool = False) -> None:
    cat.set_params({'io_dict': {'samples_dir': str(Path(cat.parameters['io_dict']['main_path']) / 'samples' / 'model_samples')}}, save_params=False)

    model_pattern = "checkpoint_epoch_*.pt"
    model_dir = Path(cat.parameters["io_dict"]["main_path"]) / "models" / "training"
    loaded_model_name = (
        latest_checkpoint(model_dir, model_pattern)
        if use_latest_checkpoint
        else first_match(model_dir / model_pattern)
    )

    cat.set_params({'io_dict': {'model_dir': str(model_dir), 'loaded_model_name': loaded_model_name}}, save_params=False)

    run_distributed_or_single(cat, run_training, cat)


def run_inference_for_split(
    cat: Catalyst,
    split_name: str,
    samples_subdir: str,
    model_dir: Path,
    results_dir: Path,
    use_latest_checkpoint: bool = True,
) -> None:
    loaded_model_name = (
        latest_checkpoint(model_dir, "checkpoint_epoch_*.pt")
        if use_latest_checkpoint
        else first_match(model_dir / "checkpoint_epoch_*.pt")
    )

    cat.set_params({'io_dict': {'write_indv_pred': True, 'samples_dir': str(Path(cat.parameters['io_dict']['main_path']) / 'samples' / samples_subdir), 'results_dir': str(reset_dir(results_dir)), 'model_dir': str(model_dir), 'loaded_model_name': loaded_model_name}}, save_params=False)

    cat.set_model(build_regression_model(DEVICE))

    print(f"Running backend inference for {split_name} split...")

    if cat.parameters["device_dict"]["run_ddp"]:
        processes = []
        for rank in range(cat.parameters["device_dict"]["world_size"]):
            process = mp.Process(target=run_inference, args=(loaded_model_name, rank, cat, True))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        cuda_destroy()
    else:
        run_inference(model_name=loaded_model_name, cat=cat, test=True)


def test_model(cat: Catalyst) -> None:
    """
    Use Catalyst run_inference for training, validation, and test splits.

    The split-specific folders each contain a test_data.npy file so the existing
    backend inference path can be reused unchanged.
    """
    main_path = Path(cat.parameters["io_dict"]["main_path"])
    model_dir = main_path / "models" / "training"

    split_specs = {
        "training": "inference_training",
        "validation": "inference_validation",
        "test": "inference_test",
    }

    for split_name, samples_subdir in split_specs.items():
        run_inference_for_split(
            cat=cat,
            split_name=split_name,
            samples_subdir=samples_subdir,
            model_dir=model_dir,
            results_dir=main_path / "testing" / split_name,
            use_latest_checkpoint=True,
        )


def predict(cat: Catalyst) -> None:
    main_path = Path(cat.parameters["io_dict"]["main_path"])
    model_dir = main_path / "models" / "training"
    loaded_model_name = latest_checkpoint(model_dir, "checkpoint_epoch_*.pt")

    cat.set_params({'io_dict': {'write_indv_pred': False, 'samples_dir': str(main_path / 'samples' / 'inference_test'), 'results_dir': str(reset_dir(main_path / 'testing' / 'predict')), 'model_dir': str(model_dir), 'loaded_model_name': loaded_model_name}}, save_params=False)

    cat.set_model(build_regression_model(DEVICE))

    if cat.parameters["device_dict"]["run_ddp"]:
        processes = []
        for rank in range(cat.parameters["device_dict"]["world_size"]):
            process = mp.Process(target=run_inference, args=(loaded_model_name, rank, cat, False))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        cuda_destroy()
    else:
        run_inference(model_name=loaded_model_name, cat=cat, test=False)


# =============================================================================
# PLOTTING / OUTPUT FROM BACKEND INFERENCE FILES
# =============================================================================


def load_force_normalization(cat: Catalyst) -> tuple[np.ndarray, np.ndarray]:
    path = Path(cat.parameters["io_dict"]["main_path"]) / "force_normalization.json"
    if not path.is_file():
        return np.asarray([0.0], dtype=float), np.asarray([1.0], dtype=float)

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return (
        np.asarray(data["force_mean"], dtype=float),
        np.asarray(data["force_std"], dtype=float),
    )


def _flatten_prediction_records(obj: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if obj is None:
        return records

    if isinstance(obj, dict):
        if "y" in obj and "pred" in obj:
            return [obj]

        for key in ("records", "data", "results", "predictions", "indv_pred", "items"):
            if key in obj:
                records.extend(_flatten_prediction_records(obj[key]))

        if not records:
            for value in obj.values():
                records.extend(_flatten_prediction_records(value))

        return records

    if isinstance(obj, (list, tuple)):
        for value in obj:
            records.extend(_flatten_prediction_records(value))
        return records

    return records


def _as_float_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        flattened: list[float] = []

        def _collect(x: Any) -> None:
            if torch.is_tensor(x):
                _collect(x.detach().cpu().numpy())
            elif isinstance(x, np.ndarray):
                for item in x.reshape(-1):
                    _collect(item)
            elif isinstance(x, (list, tuple)):
                for item in x:
                    _collect(item)
            else:
                try:
                    flattened.append(float(x))
                except (TypeError, ValueError):
                    return

        _collect(value)
        return np.asarray(flattened, dtype=float)


def _load_backend_force_predictions(results_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(results_dir) / "indv_pred.data"
    if not path.is_file():
        raise FileNotFoundError(f"Missing backend inference output: {path}")

    run_data = load_dictionary(path)
    records = _flatten_prediction_records(run_data)

    if not records:
        raise RuntimeError(f"No prediction records with y/pred were found in: {path}")

    y_norm = []
    pred_norm = []

    for record in records:
        if not isinstance(record, dict) or "y" not in record or "pred" not in record:
            continue

        y_norm.extend(_as_float_array(record["y"]).reshape(-1).astype(float).tolist())
        pred_norm.extend(_as_float_array(record["pred"]).reshape(-1).astype(float).tolist())

    n = min(len(y_norm), len(pred_norm))
    n -= n % 3

    if n <= 0:
        raise RuntimeError(f"No numeric vector force predictions were found in: {path}")

    y = np.asarray(y_norm[:n], dtype=float).reshape(-1, 3)
    pred = np.asarray(pred_norm[:n], dtype=float).reshape(-1, 3)

    return y, pred


def _denormalize_forces(y_norm: np.ndarray, pred_norm: np.ndarray, mean: np.ndarray, std: np.ndarray):
    y = y_norm * std + mean
    pred = pred_norm * std + mean
    return y, pred


def _force_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    diff = pred - y
    component_mae = float(np.mean(np.abs(diff)))
    component_rmse = float(np.sqrt(np.mean(diff ** 2)))

    vector_errors = np.linalg.norm(diff, axis=1)
    vector_mae = float(np.mean(vector_errors))
    vector_rmse = float(np.sqrt(np.mean(vector_errors ** 2)))

    return {
        "component_mae_eVA": component_mae,
        "component_rmse_eVA": component_rmse,
        "vector_mae_eVA": vector_mae,
        "vector_rmse_eVA": vector_rmse,
    }


def plot_training_results(cat: Catalyst) -> None:
    model_dir = Path(cat.parameters["io_dict"]["main_path"]) / "models" / "training"
    run_data = load_dictionary(model_dir / "run_information.npy")

    training_loss = run_data["training_loss"]
    validation_loss = run_data["validation_loss"]
    epochs = np.linspace(1, len(training_loss), len(training_loss))

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    ax.set_title("Training loss")
    ax.set_yscale("log")
    ax.plot(epochs, training_loss, marker="o", label="Training loss")
    ax.plot(epochs, validation_loss, marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized vector-field MSE")
    ax.legend(loc="upper right")
    plt.tight_layout()

    out = get_figures_dir(cat) / "training_force_loss.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_force_parity(cat: Catalyst) -> None:
    main_path = Path(cat.parameters["io_dict"]["main_path"])
    force_mean, force_std = load_force_normalization(cat)

    split_names = ["training", "validation", "test"]
    pretty_names = {
        "training": "Training",
        "validation": "Validation",
        "test": "Test",
    }

    split_data = {}
    all_components = []
    metrics_out = {}

    for split_name in split_names:
        y_norm, pred_norm = _load_backend_force_predictions(main_path / "testing" / split_name)
        y, pred = _denormalize_forces(y_norm, pred_norm, force_mean, force_std)

        metrics = _force_metrics(y, pred)
        split_data[split_name] = (y, pred, metrics)
        metrics_out[split_name] = metrics

        all_components.extend(y.reshape(-1).tolist())
        all_components.extend(pred.reshape(-1).tolist())

    lo = float(min(all_components))
    hi = float(max(all_components))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True, sharey=True)

    for ax, split_name in zip(axes, split_names):
        y, pred, metrics = split_data[split_name]

        ax.plot(
            y.reshape(-1),
            pred.reshape(-1),
            linestyle="",
            marker="o",
            markersize=2.5,
            markeredgecolor="k",
            alpha=0.55,
        )

        ax.plot([lo, hi], [lo, hi], linestyle="-", linewidth=1.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("EMT force component (eV/Ang)")
        ax.set_title(
            f"{pretty_names[split_name]}\n"
            f"Nvec={y.shape[0]}, "
            f"comp MAE={metrics['component_mae_eVA']:.4f}, "
            f"vec MAE={metrics['vector_mae_eVA']:.4f}"
        )

    axes[0].set_ylabel("Equivariant GNN force component (eV/Ang)")
    fig.suptitle("Force-component parity for training, validation, and test splits", y=1.02)
    plt.tight_layout()

    out = get_figures_dir(cat) / "force_component_parity_train_validation_test.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")

    with (main_path / "force_metrics_train_validation_test.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_out, handle, indent=2)

    print("Force metrics:")
    print(json.dumps(metrics_out, indent=2))


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def main() -> None:
    task = build_regression_task()
    cat = Catalyst(
        parameter_file=CONFIG_PATH,
        parameters={
            "loader_dict": {"batch_size": TRAINING_BATCH_SIZE},
            "model_dict": {
                "num_epochs": TRAINING_NUM_EPOCHS_OVERRIDE,
                "train_delta": TRAINING_DELTA_OVERRIDE,
                "train_tolerance": TRAINING_TOLERANCE_OVERRIDE,
            },
        },
        task=task,
    )

    make_dir(cat.parameters["io_dict"]["main_path"])
    make_dir(cat.parameters["io_dict"]["data_dir"])

    if RUN_GENERATE_GRAPHS:
        generate_data(cat)

    if RUN_GENERATE_SAMPLES:
        sample_data(cat)

    if RUN_NORMALIZE_TARGETS:
        normalize_targets(cat)

    if RUN_TRAINING:
        train_model(cat)

        if RUN_PLOT_TRAINING:
            plot_training_results(cat)

    if RUN_RETRAINING:
        cat.set_model(build_regression_model(DEVICE))
        cat.set_params({"model_dict": {"restart_training": True}}, save_params=False)
        retrain_model(cat, use_latest_checkpoint=True)

    if RUN_TESTING:
        test_model(cat)

    if RUN_PLOT_TEST:
        plot_force_parity(cat)

    if RUN_PREDICTIONS:
        cat.set_model(build_regression_model(DEVICE))
        predict(cat)


if __name__ == "__main__":
    main()
