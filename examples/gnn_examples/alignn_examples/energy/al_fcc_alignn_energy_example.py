"""
Al FCC ALIGNN energy-learning example using the Catalyst training backend.

Recommended location:
    catalyst/examples/gnn_example/al_fcc_alignn_energy_catalyst_backend.py

This example follows the same pattern as the full random Catalyst example:

    cat = Catalyst()
    cat.set_params(build_catalyst_parameters(CONFIG))

    cat.set_model(build_regression_model(DEVICE))
    run_distributed_or_single(cat, run_training, cat)

    run_inference(...)
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn

try:
    from torch_geometric.utils import scatter
except ImportError:  # pragma: no cover
    scatter = None

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
        "CATALYST_AL_FCC_CONFIG",
        Path(__file__).with_name("al_fcc_alignn_energy_catalyst_config.json"),
    )
)


def load_json_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Could not find Catalyst Al FCC config file: {config_path}\n"
            "Set CATALYST_AL_FCC_CONFIG=/path/to/config.json or place "
            "al_fcc_alignn_energy_catalyst_config.json next to this script."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


CONFIG = load_json_config()
BASE_DIR = Path(__file__).resolve().parent

# Frequently used settings. Edit the JSON file, not this script.
AL_CONFIG = CONFIG["al_fcc"]
MODEL_CONFIG = CONFIG["model_architecture"]
TRAINING_OVERRIDES = CONFIG["training_overrides"]
WORKFLOW = CONFIG["workflow"]

DEVICE = CONFIG["catalyst_parameters"]["device_dict"]["device"]

RUN_GENERATE_GRAPHS = WORKFLOW["generate_graphs"]
RUN_GENERATE_SAMPLES = WORKFLOW["generate_samples"]
RUN_NORMALIZE_TARGETS = WORKFLOW.get("normalize_targets", True)
RUN_TRAINING = WORKFLOW["train"]
RUN_RETRAINING = WORKFLOW["retrain"]
RUN_TESTING = WORKFLOW["test"]
RUN_PLOT_TEST = WORKFLOW["plot_test"]
RUN_PLOT_TRAINING = WORKFLOW["plot_training"]
RUN_PREDICTIONS = WORKFLOW["predictions"]

TRAINING_BATCH_SIZE = TRAINING_OVERRIDES["training_batch_size"]
TRAINING_NUM_EPOCHS_OVERRIDE = TRAINING_OVERRIDES["num_epochs"]
TRAINING_DELTA_OVERRIDE = TRAINING_OVERRIDES["train_delta"]
TRAINING_TOLERANCE_OVERRIDE = TRAINING_OVERRIDES["train_tolerance"]


# =============================================================================
# PARAMETER AND MODEL BUILDERS
# =============================================================================


def build_regression_task() -> GNNTask:
    """
    Build the generic task contract for this example.

    This is a graph-level scalar regression task. The physical meaning of the
    target is intentionally not encoded in the task name.
    """
    return GNNTask.graph_scalar(
        target_key="target_scalar",
        output_key="scalar",
        accumulate_loss="exact",
    )


def resolve_relative_path(path_value: Optional[str]) -> Optional[str]:
    """Resolve JSON path strings relative to this example script."""
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(BASE_DIR / path)


def build_loss_function(loss_name: str):
    """Convert the JSON loss-function name into a PyTorch loss object."""
    from catalyst.ml.utils.loss import MaxNpercent
    loss_functions = {
        "MSELoss": torch.nn.MSELoss,
        "L1Loss": torch.nn.L1Loss,
        "SmoothL1Loss": torch.nn.SmoothL1Loss,
        "MaxNpercent":MaxNpercent
    }
    if loss_name not in loss_functions:
        raise ValueError(
            f"Unsupported loss function {loss_name!r}. "
            f"Supported options are: {sorted(loss_functions)}"
        )
    return loss_functions[loss_name]()


def latest_checkpoint(checkpoint_dir: Path, checkpoint_pattern: str = "checkpoint_epoch_*.pt") -> str:
    """Find the checkpoint with the largest epoch number."""
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


def build_catalyst_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the Catalyst runtime parameter dictionary from JSON."""
    parameters = dict(config["catalyst_parameters"])

    # Copy nested dictionaries so runtime edits do not mutate CONFIG unexpectedly.
    parameters["device_dict"] = dict(parameters["device_dict"])
    parameters["io_dict"] = dict(parameters["io_dict"])
    parameters["sampling_dict"] = dict(parameters.get("sampling_dict", {}))
    parameters["loader_dict"] = dict(parameters["loader_dict"])
    parameters["model_dict"] = dict(parameters["model_dict"])

    model_dict = parameters["model_dict"]
    model_dict["optimizer_params"] = dict(model_dict["optimizer_params"])
    model_dict["optimizer_params"]["params_group"] = dict(
        model_dict["optimizer_params"]["params_group"]
    )

    # Resolve paths relative to the script location.
    io_dict = parameters["io_dict"]
    for key in ["main_path", "data_dir", "model_dir", "results_dir", "samples_dir", "projection_dir"]:
        io_dict[key] = resolve_relative_path(io_dict.get(key))

    # Reconstruct non-JSON Python objects.
    loss_params = dict(model_dict["loss_params"])
    loss_params["function"] = build_loss_function(loss_params["function"])
    if "sub_function" in loss_params and loss_params["sub_function"] is not None:
        loss_params["sub_function"] = build_loss_function(loss_params["sub_function"])
    model_dict["loss_params"] = loss_params
    model_dict["model"] = None

    # Configure the existing Catalyst backend from the formal generic task
    # contract. This sets accumulate_loss and prediction_params consistently.
    build_regression_task().apply_to_catalyst_parameters(parameters)

    return parameters


def scatter_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int | None = None) -> torch.Tensor:
    """Fallback scatter-sum used only by AtomicEnergyReadout."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    out = values.new_zeros((dim_size,) + tuple(values.shape[1:]))
    if values.numel() > 0:
        out.index_add_(0, index.to(values.device), values)
    return out


class AlignnEnergyReadout(nn.Module):
    """
    Extreme-aware order readout for graph_scalar ALIGNN energy learning.

    The original readout used only mean pooling over atom/bond/angle hidden
    states. That tends to underpredict high-energy MD frames because high-energy
    frames are often controlled by the most distorted local environments, and
    mean pooling can average those rare distortions away.

    This decoder keeps stable mean information, but also exposes max/top-k
    local information to the graph-level MLP.

    Per order it builds:
        mean(projected hidden)
        max(projected hidden)
        mean(local scalar contribution)
        max(local scalar contribution)
        top-k mean(local scalar contribution)

    The output remains shape [B].
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        act=nn.SiLU(),
        topk_fraction: float = 0.10,
        topk_min: int = 4,
        feature_clip: float = 50.0,
    ):
        super().__init__()

        hidden_dim = hidden_dim or dim
        self.hidden_dim = int(hidden_dim)
        self.topk_fraction = float(topk_fraction)
        self.topk_min = int(topk_min)
        self.feature_clip = float(feature_clip)

        def make_proj():
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                act,
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                act,
                nn.LayerNorm(hidden_dim),
            )

        def make_local_head():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                act,
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )

        self.atom_proj = make_proj()
        self.bond_proj = make_proj()
        self.angle_proj = make_proj()

        self.atom_local = make_local_head()
        self.bond_local = make_local_head()
        self.angle_local = make_local_head()

        # Per order: mean_h + max_h + mean_local + max_local + topk_mean_local
        per_order_dim = 2 * hidden_dim + 3
        final_in_dim = 3 * per_order_dim

        self.final = nn.Sequential(
            nn.LayerNorm(final_in_dim),
            nn.Linear(final_in_dim, 2 * hidden_dim),
            act,
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            act,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        # Start near the normalized target mean. This avoids unstable first
        # steps but does not constrain the output range.
        last = self.final[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.bias)

    def _finite(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            x,
            nan=0.0,
            posinf=self.feature_clip,
            neginf=-self.feature_clip,
        ).clamp(min=-self.feature_clip, max=self.feature_clip)

    def _first_existing(self, data, names):
        for name in names:
            if hasattr(data, name):
                value = getattr(data, name)
                if value is not None:
                    return value
        return None

    def _batch(self, data, names, n_items, device):
        for name in names:
            if hasattr(data, name):
                value = getattr(data, name)
                if value is not None:
                    return value.to(device)
        return torch.zeros(n_items, dtype=torch.long, device=device)

    def _empty_order_features(self, template: torch.Tensor, n_graphs: int) -> torch.Tensor:
        return template.new_zeros((n_graphs, 2 * self.hidden_dim + 3))

    def _topk_mean(self, values_1d: torch.Tensor) -> torch.Tensor:
        n = int(values_1d.numel())
        if n == 0:
            return values_1d.new_zeros(())
        k = max(self.topk_min, int(round(self.topk_fraction * n)))
        k = min(k, n)
        return torch.topk(values_1d, k=k, largest=True).values.mean()

    def _order_features(
        self,
        h: torch.Tensor | None,
        batch: torch.Tensor | None,
        proj: nn.Module,
        local_head: nn.Module,
        n_graphs: int,
        template: torch.Tensor,
    ) -> torch.Tensor:
        if h is None or batch is None:
            return self._empty_order_features(template, n_graphs)

        z = self._finite(proj(self._finite(h.float())))
        batch = batch.to(z.device)

        local = self._finite(local_head(z).view(-1))

        graph_features = []
        for graph_idx in range(n_graphs):
            mask = batch == graph_idx
            if not torch.any(mask):
                graph_features.append(z.new_zeros((2 * self.hidden_dim + 3,)))
                continue

            z_g = z[mask]
            local_g = local[mask]

            mean_h = z_g.mean(dim=0)
            max_h = z_g.max(dim=0).values

            mean_local = local_g.mean().view(1)
            max_local = local_g.max().view(1)
            topk_local = self._topk_mean(local_g).view(1)

            graph_features.append(
                torch.cat(
                    [
                        mean_h,
                        max_h,
                        mean_local,
                        max_local,
                        topk_local,
                    ],
                    dim=0,
                )
            )

        return self._finite(torch.stack(graph_features, dim=0))

    def forward(self, data):
        h_atom = self._first_existing(data, ["h_atm", "h_1", "h_scalar"])
        h_bond = self._first_existing(data, ["h_bnd", "h_2", "h_edge"])
        h_angle = self._first_existing(data, ["h_ang", "h_3"])

        if h_atom is None:
            raise AttributeError("Missing atom hidden states: expected h_atm, h_1, or h_scalar.")

        device = h_atom.device

        atom_batch = self._batch(
            data,
            ["x_atm_batch", "x_1_batch", "batch"],
            h_atom.size(0),
            device,
        )
        n_graphs = int(atom_batch.max().item()) + 1 if atom_batch.numel() > 0 else 1

        atom_feat = self._order_features(
            h_atom,
            atom_batch,
            self.atom_proj,
            self.atom_local,
            n_graphs,
            h_atom,
        )

        bond_batch = None
        if h_bond is not None:
            bond_batch = self._batch(
                data,
                ["x_bnd_batch", "x_2_batch", "node_A_batch"],
                h_bond.size(0),
                h_bond.device,
            )

        bond_feat = self._order_features(
            h_bond,
            bond_batch,
            self.bond_proj,
            self.bond_local,
            n_graphs,
            h_atom,
        )

        angle_batch = None
        if h_angle is not None:
            angle_batch = self._batch(
                data,
                ["x_ang_batch", "x_3_batch", "edge_A_batch"],
                h_angle.size(0),
                h_angle.device,
            )

        angle_feat = self._order_features(
            h_angle,
            angle_batch,
            self.angle_proj,
            self.angle_local,
            n_graphs,
            h_atom,
        )

        graph_feat = torch.cat([atom_feat, bond_feat, angle_feat], dim=-1)
        energy = self.final(graph_feat)

        return torch.nan_to_num(energy.view(-1), nan=0.0, posinf=50.0, neginf=-50.0)



def build_regression_model(device: str = DEVICE) -> GNN:
    """
    Build the ALIGNN/order graph_scalar regression model used by Catalyst.

    The task interface owns the backend contract:
        - target_key="target_scalar"
        - accumulate_loss="exact"
        - prediction_params["output_key"]="scalar"

    Training, checkpointing, and inference are still handled by the Catalyst
    backend.
    """
    task = build_regression_task()

    model = build_task_model(
        task=task,
        model_type="gnn_builder",
        apply_task_model_kwargs=False,
        preset="alignn",
        processor_type="order",
        conv_type=MODEL_CONFIG["conv_type"],
        decoder=AlignnEnergyReadout(
            dim=MODEL_CONFIG["hidden_dim"],
            hidden_dim=MODEL_CONFIG["hidden_dim"],
            act=nn.SiLU(),
            topk_fraction=0.10,
            topk_min=4,
            feature_clip=50.0,
        ),
        num_species=AL_CONFIG["num_species"],
        cutoff=AL_CONFIG["cutoff"],
        dim=MODEL_CONFIG["hidden_dim"],
        num_convs=MODEL_CONFIG["n_convs"],
        out_dim=MODEL_CONFIG["regression_out_dim"],
        act=nn.SiLU(),
        aggr_scheme=MODEL_CONFIG.get("aggr_scheme", "add"),
        encode_3body=True,
        dihedral=AL_CONFIG.get("include_dihedrals", False),
    )
    return GNN(model=model, device=device)


# =============================================================================
# UTILITY FUNCTIONS
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
    """Load full Catalyst/PyG graph objects across old and new PyTorch versions."""
    try:
        return torch.load(file_name, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(file_name, map_location=map_location)


def as_numpy_tensor(value) -> np.ndarray:
    """Convert tensors or arrays to a CPU NumPy array."""
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
# GRAPH GENERATION
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
    energies = []

    for step in range(AL_CONFIG["md_steps"] + 1):
        if step > 0:
            dyn.run(1)

        if step < AL_CONFIG["equilibration_steps"]:
            continue

        if (step - AL_CONFIG["equilibration_steps"]) % AL_CONFIG["sample_every"] != 0:
            continue

        frame = atoms.copy()
        frame.calc = EMT()
        energy = float(frame.get_potential_energy())

        frames.append(frame)
        energies.append(energy)

    print(f"Generated {len(frames)} MD frames.")
    print(f"Energy range: {min(energies):.6f} to {max(energies):.6f} eV")

    return frames


def _as_single_graph(obj):
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise RuntimeError("alignn_gen returned an empty list.")
        return obj[0]
    return obj


def finalize_graph_metadata(graph):
    """
    Make graphs safe for the renamed GNNBuilder/equivariant-compatible stack.
    Legacy ALIGNN/order fields are preserved.
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

    return graph


def atoms_to_alignn_graph(atoms, energy_eV: float, gid: str):
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

            # Updated graph fields. Older graph builders may ignore these.
            "include_equivariant_fields": AL_CONFIG.get("include_equivariant_fields", True),
            "include_edge_geometry": AL_CONFIG.get("include_edge_geometry", True),

            # Retry controls.
            "auto_retry_graph": True,
            "max_graph_attempts": AL_CONFIG.get("max_graph_attempts", 6),
            "cutoff_scale": AL_CONFIG.get("cutoff_scale", 1.15),
            "max_cutoff": AL_CONFIG.get("max_cutoff", 5.0),
            "require_bonds": True,
            "require_angles": True,
            "require_dihedrals": False,
            "retry_verbose": False,
        }
    )

    graph = _as_single_graph(graph)
    graph = finalize_graph_metadata(graph)

    graph.gid = gid
    graph.energy_eV = torch.tensor([energy_eV], dtype=torch.float32)

    # Raw target initially; normalize_targets(...) overwrites y/target_scalar.
    graph.y = torch.tensor([energy_eV], dtype=torch.float32)
    graph.target_scalar = torch.tensor([energy_eV], dtype=torch.float32)

    return graph


def generate_data(cat: Catalyst) -> None:
    data_dir = reset_dir(cat.parameters["io_dict"]["data_dir"])
    frames = run_md_frames()

    for idx, atoms in enumerate(frames):
        energy_eV = float(atoms.get_potential_energy())
        gid = f"al_fcc_md_{idx:05d}"

        print(f"Building graph {idx + 1:4d}/{len(frames):4d}: {gid}, E={energy_eV:.6f} eV")

        graph = atoms_to_alignn_graph(
            atoms=atoms,
            energy_eV=energy_eV,
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
    cat.parameters["io_dict"]["samples_dir"] = str(samples_dir)

    gids = [Path(path).stem for path in data_files]
    rng = np.random.default_rng(CONFIG.get("sampling", {}).get("sampling_seed", 112358))
    indices = np.arange(len(gids))
    rng.shuffle(indices)

    test_fraction = CONFIG["sampling"]["test_fraction"]
    validation_fraction_of_remaining = CONFIG["sampling"]["validation_fraction_of_remaining"]

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
            # Placeholder kept for compatibility with projection-based examples.
            "projections": [],
        },
    )

    save_dictionary(
        model_samples_dir / "train_valid_split.npy",
        {
            "training": training_gids,
            "validation": validation_gids,
            # Placeholder keys kept for compatibility with plotting/sampling code.
            "training_projections": [],
            "validation_projections": [],
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


def normalize_targets(cat: Catalyst) -> None:
    """
    Normalize graph.y and graph.target_scalar using training-set statistics.

    Training/inference still flow through Catalyst. This function only prepares
    graph targets before run_training.
    """
    samples_file = Path(cat.parameters["io_dict"]["main_path"]) / "samples" / "model_samples" / "train_valid_split.npy"
    if not samples_file.is_file():
        raise FileNotFoundError(
            f"Cannot normalize targets because training split is missing: {samples_file}"
        )

    split = load_dictionary(samples_file)
    training_gids = split["training"]

    if not training_gids:
        raise RuntimeError("Training split is empty; cannot compute target normalization.")

    training_energies = []
    for gid in training_gids:
        graph = safe_torch_load(_graph_path_from_gid(cat, gid))
        training_energies.append(float(graph.energy_eV.view(-1)[0]))

    target_mean = float(np.mean(training_energies))
    target_std = float(np.std(training_energies))
    if target_std < 1.0e-12:
        target_std = 1.0

    data_files = sorted(glob.glob(os.path.join(cat.parameters["io_dict"]["data_dir"], "*.pt")))
    for path in data_files:
        graph = safe_torch_load(path)
        energy = float(graph.energy_eV.view(-1)[0])
        normalized = (energy - target_mean) / target_std

        graph.y = torch.tensor([normalized], dtype=torch.float32)
        graph.target_scalar = torch.tensor([normalized], dtype=torch.float32)
        graph.target_normalization = {
            "target_mean": target_mean,
            "target_std": target_std,
            "target_units": "eV",
            "target_type": "total_energy",
        }

        torch.save(graph, path)

    with (Path(cat.parameters["io_dict"]["main_path"]) / "target_normalization.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "target_mean": target_mean,
                "target_std": target_std,
                "target_units": "eV",
                "target_type": "total_energy",
            },
            handle,
            indent=2,
        )

    print(f"Target normalization: mean={target_mean:.6f} eV, std={target_std:.6f} eV")


# =============================================================================
# BACKEND WORKFLOW FUNCTIONS
# =============================================================================


def train_model(cat: Catalyst) -> None:
    """
    Train using the Catalyst backend.

    No local epoch loop belongs here. run_training owns:
        GNN.load_data
        GNN.set_dataloader
        GNN.set_optimizer_
        epoch loop
        checkpointing
        DDP behavior
    """
    cat.parameters["io_dict"]["samples_dir"] = str(
        Path(cat.parameters["io_dict"]["main_path"]) / "samples" / "model_samples"
    )
    cat.set_model(build_regression_model(DEVICE))
    run_distributed_or_single(cat, run_training, cat)


def retrain_model(cat: Catalyst, use_latest_checkpoint: bool = False) -> None:
    cat.parameters["io_dict"]["samples_dir"] = str(
        Path(cat.parameters["io_dict"]["main_path"]) / "samples" / "model_samples"
    )

    model_pattern = "checkpoint_epoch_*.pt"
    model_dir = Path(cat.parameters["io_dict"]["main_path"]) / "models" / "training"
    loaded_model_name = (
        latest_checkpoint(model_dir, model_pattern)
        if use_latest_checkpoint
        else first_match(model_dir / model_pattern)
    )

    cat.parameters["io_dict"].update(
        {
            "model_dir": str(model_dir),
            "loaded_model_name": loaded_model_name,
        }
    )

    run_distributed_or_single(cat, run_training, cat)


def run_testing_for_model(
    cat: Catalyst,
    model_dir: Path,
    results_dir: Path,
    model_pattern: str,
    use_latest_checkpoint: bool = False,
) -> None:
    loaded_model_name = (
        latest_checkpoint(model_dir, model_pattern)
        if use_latest_checkpoint
        else first_match(model_dir / model_pattern)
    )

    cat.parameters["io_dict"].update(
        {
            "write_indv_pred": True,
            "results_dir": str(reset_dir(results_dir)),
            "model_dir": str(model_dir),
            "loaded_model_name": loaded_model_name,
        }
    )

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
    main_path = Path(cat.parameters["io_dict"]["main_path"])
    cat.parameters["io_dict"]["samples_dir"] = str(main_path / "samples")
    cat.set_model(build_regression_model(DEVICE))

    run_testing_for_model(
        cat=cat,
        model_dir=main_path / "models" / "training",
        results_dir=main_path / "testing" / "training",
        model_pattern="checkpoint_epoch_*.pt",
        use_latest_checkpoint=True,
    )


def predict(cat: Catalyst) -> None:
    main_path = Path(cat.parameters["io_dict"]["main_path"])
    model_dir = main_path / "models" / "training"
    results_dir = main_path / "testing" / "predict"
    loaded_model_name = latest_checkpoint(model_dir, "checkpoint_epoch_*.pt")

    cat.parameters["io_dict"].update(
        {
            "write_indv_pred": False,
            "results_dir": str(reset_dir(results_dir)),
            "model_dir": str(model_dir),
            "loaded_model_name": loaded_model_name,
        }
    )

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
# OPTIONAL PLOTTING
# =============================================================================


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


def load_target_normalization(cat: Catalyst) -> tuple[float, float]:
    path = Path(cat.parameters["io_dict"]["main_path"]) / "target_normalization.json"
    if not path.is_file():
        return 0.0, 1.0
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return float(data["target_mean"]), float(data["target_std"])


def plot_training_results(cat: Catalyst) -> None:
    model_dir = Path(cat.parameters["io_dict"]["main_path"]) / "models" / "training"
    cat.parameters["io_dict"]["model_dir"] = str(model_dir)

    run_data = load_dictionary(model_dir / "run_information.npy")
    training_loss = run_data["training_loss"]
    validation_loss = run_data["validation_loss"]
    epochs = np.linspace(1, len(training_loss), len(training_loss))

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    ax.set_title("Training loss")
    ax.set_yscale("log")
    ax.plot(epochs, training_loss, color="b", marker="o", label="Training loss")
    ax.plot(epochs, validation_loss, color="r", marker="o", label="Validation loss")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()



def _follow_batch_names(graphs: Sequence[Any]) -> list[str]:
    candidate_names = [
        "x_atm", "x_bnd", "x_ang",
        "x_1", "x_2", "x_3",
        "node_G", "node_A", "edge_A",
    ]
    follow_batch = []
    for name in candidate_names:
        if any(hasattr(graph, name) and getattr(graph, name) is not None for graph in graphs):
            follow_batch.append(name)
    return follow_batch


def _load_graphs_from_gids(cat: Catalyst, gids: Sequence[str]) -> list[Any]:
    graphs = []
    for gid in gids:
        graph = safe_torch_load(_graph_path_from_gid(cat, gid), map_location="cpu")
        graphs.append(graph)
    return graphs


def _predict_graphs_with_checkpoint(
    cat: Catalyst,
    graphs: Sequence[Any],
    checkpoint_path: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    from torch_geometric.loader import DataLoader

    if not graphs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    gnn = build_regression_model(DEVICE)
    gnn.load_checkpoint(
        fname=checkpoint_path,
        map_location=DEVICE,
        load_optimizer=False,
        strict=True,
    )
    gnn.model.eval()

    follow_batch = _follow_batch_names(graphs)
    loader = DataLoader(
        list(graphs),
        batch_size=batch_size,
        shuffle=False,
        follow_batch=follow_batch,
        num_workers=0,
    )

    pred_norm = []
    y_norm = []

    grad_context = torch.enable_grad if gnn._model_requires_grad_forward() else torch.no_grad

    with grad_context():
        for batch in loader:
            batch = batch.to(gnn.device)

            raw_pred = gnn.model(batch)
            preds, y, vec = gnn._accumulate_predictions(
                raw_pred,
                batch,
                cat.parameters,
                return_y=True,
            )
            preds, y = gnn._align_pred_and_target(preds, y)

            pred_norm.extend(as_numpy_tensor(preds).reshape(-1).astype(float).tolist())
            y_norm.extend(as_numpy_tensor(y).reshape(-1).astype(float).tolist())

    n = min(len(pred_norm), len(y_norm))
    return (
        np.asarray(y_norm[:n], dtype=float),
        np.asarray(pred_norm[:n], dtype=float),
    )


def _parity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return mae, rmse


def plot_test_data(cat: Catalyst) -> None:
    """
    Plot parity results for training, validation, and test splits in one figure.

    This bypasses the saved indv_pred.data parser and instead:
        1. loads the latest training checkpoint,
        2. loads graph objects for each split,
        3. runs direct model inference on each split,
        4. denormalizes predictions/targets back to eV,
        5. renders a 1x3 parity-plot figure.
    """
    main_path = Path(cat.parameters["io_dict"]["main_path"])
    model_dir = main_path / "models" / "training"
    checkpoint_path = latest_checkpoint(model_dir, "checkpoint_epoch_*.pt")

    split_train_valid = load_dictionary(main_path / "samples" / "model_samples" / "train_valid_split.npy")
    split_test = load_dictionary(main_path / "samples" / "test_data.npy")

    split_map = {
        "Training": split_train_valid.get("training", []),
        "Validation": split_train_valid.get("validation", []),
        "Test": split_test.get("gids", []),
    }

    target_mean, target_std = load_target_normalization(cat)
    batch_size = int(cat.parameters["loader_dict"]["batch_size"][1])
    if batch_size <= 0:
        batch_size = max(max(len(v) for v in split_map.values()), 1)

    split_results: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
    all_values = []

    for split_name, gids in split_map.items():
        graphs = _load_graphs_from_gids(cat, gids)
        y_norm, pred_norm = _predict_graphs_with_checkpoint(
            cat=cat,
            graphs=graphs,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
        )

        if y_norm.size == 0 or pred_norm.size == 0:
            split_results[split_name] = (
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                float("nan"),
                float("nan"),
            )
            continue

        y_eV = y_norm * target_std + target_mean
        pred_eV = pred_norm * target_std + target_mean
        mae, rmse = _parity_metrics(y_eV, pred_eV)

        split_results[split_name] = (y_eV, pred_eV, mae, rmse)
        all_values.extend(y_eV.tolist())
        all_values.extend(pred_eV.tolist())

    if not all_values:
        raise RuntimeError("Could not compute any parity-plot data for training/validation/test splits.")

    lo = float(min(all_values))
    hi = float(max(all_values))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True, sharey=True)

    for ax, split_name in zip(axes, ["Training", "Validation", "Test"]):
        y_eV, pred_eV, mae, rmse = split_results[split_name]

        if y_eV.size > 0:
            ax.plot(
                y_eV,
                pred_eV,
                linestyle="",
                marker="o",
                markeredgecolor="k",
                markersize=5,
                alpha=0.85,
            )
            ax.plot([lo, hi], [lo, hi], linestyle="-", color="r", linewidth=1.5)
            ax.set_title(f"{split_name}\nN={len(y_eV)}, MAE={mae:.4f} eV, RMSE={rmse:.4f} eV")
        else:
            ax.set_title(f"{split_name}\nNo data")
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("EMT energy (eV)")

    axes[0].set_ylabel("GNN-predicted energy (eV)")
    fig.suptitle("Parity plots for training, validation, and test splits", y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def main() -> None:
    cat = Catalyst()
    cat.set_params(build_catalyst_parameters(CONFIG))

    # Ensure base directories exist.
    make_dir(cat.parameters["io_dict"]["main_path"])
    make_dir(cat.parameters["io_dict"]["data_dir"])

    if RUN_GENERATE_GRAPHS:
        generate_data(cat)

    if RUN_GENERATE_SAMPLES:
        sample_data(cat)

    if RUN_NORMALIZE_TARGETS:
        normalize_targets(cat)

    if RUN_TRAINING:
        cat.parameters["loader_dict"]["batch_size"] = TRAINING_BATCH_SIZE
        cat.parameters["model_dict"]["num_epochs"] = TRAINING_NUM_EPOCHS_OVERRIDE
        cat.parameters["model_dict"]["train_delta"] = TRAINING_DELTA_OVERRIDE
        cat.parameters["model_dict"]["train_tolerance"] = TRAINING_TOLERANCE_OVERRIDE
        train_model(cat)

        if RUN_PLOT_TRAINING:
            plot_training_results(cat)

    if RUN_RETRAINING:
        cat.parameters["loader_dict"]["batch_size"] = TRAINING_BATCH_SIZE
        cat.parameters["model_dict"]["num_epochs"] = TRAINING_NUM_EPOCHS_OVERRIDE
        cat.parameters["model_dict"]["train_delta"] = TRAINING_DELTA_OVERRIDE
        cat.parameters["model_dict"]["train_tolerance"] = TRAINING_TOLERANCE_OVERRIDE
        cat.set_model(build_regression_model(DEVICE))
        cat.parameters["model_dict"]["restart_training"] = True
        retrain_model(cat, use_latest_checkpoint=True)

    if RUN_TESTING:
        cat.parameters["loader_dict"]["batch_size"] = TRAINING_BATCH_SIZE
        cat.set_model(build_regression_model(DEVICE))
        test_model(cat)

    if RUN_PLOT_TEST:
        plot_test_data(cat)

    if RUN_PREDICTIONS:
        cat.set_model(build_regression_model(DEVICE))
        predict(cat)


if __name__ == "__main__":
    main()
