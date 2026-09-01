"""
Al FCC ALIGNN energy-per-atom example using the Catalyst training backend.

This example is intentionally designed as a robust learning demonstration rather
than a narrow single-trajectory benchmark. It generates several families of FCC
Al structures with EMT reference energies:

    * isotropic compression/expansion,
    * uniaxial strain,
    * shear strain,
    * random thermal-like atomic displacements,
    * short MD trajectories at several temperatures.

The GNN predicts normalized energy per atom. The decoder is a local-environment
energy readout: ALIGNN/order message passing updates atom, bond, and angle hidden
states, and the decoder forms per-atom-normalized atomic, pair, and angular energy
contributions. This gives geometry a direct route to the scalar target instead of
forcing every geometric signal through the atom channel alone. The main
optimization/checkpoint/inference workflow uses the public Catalyst training
backend.

The example also includes diagnostics for the common failure mode where every
prediction collapses to the training-target mean:

    * pre-training output/gradient checks,
    * a small direct overfit diagnostic on a diverse training subset,
    * post-training prediction-spread/collapse checks.

The tiny overfit diagnostic is intentionally advisory by default because its
numerical threshold depends on optimizer, hardware, and random initialization.
The post-training collapse check is the authoritative model-quality guard.

All Matplotlib figures are written to <main_path>/figures using the non-interactive
Agg backend.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn


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

# Frequently used settings. Edit the JSON file, not this script.
AL_CONFIG = CONFIG["al_fcc"]
MODEL_CONFIG = CONFIG["model_architecture"]
TRAINING_OVERRIDES = CONFIG["training_overrides"]
WORKFLOW = CONFIG["workflow"]
GENERATION_CONFIG = CONFIG.get("dataset_generation", {})
DIAGNOSTICS_CONFIG = CONFIG.get("diagnostics", {})

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
RUN_PREFLIGHT_DIAGNOSTICS = WORKFLOW.get("preflight_diagnostics", True)
RUN_OVERFIT_SANITY = WORKFLOW.get("overfit_sanity_check", True)
RUN_POSTTRAIN_DIAGNOSTICS = WORKFLOW.get("posttrain_diagnostics", True)

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

    This is a graph-level scalar regression task. The physical meaning of the
    target is intentionally not encoded in the task name.
    """
    return GNNTask.graph_scalar(
        target_key="target_scalar",
        output_key="scalar",
        accumulate_loss="exact",
    )






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




def scatter_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int | None = None) -> torch.Tensor:
    """Small dependency-free graph-wise scatter sum helper."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    out = values.new_zeros((dim_size,) + tuple(values.shape[1:]))
    if values.numel() > 0:
        out.index_add_(0, index.to(values.device), values)
    return out


class LocalEnvironmentEnergyReadout(nn.Module):
    """Energy-per-atom readout using atom, bond, and angle hidden states.

    A monoatomic crystal can make an atom-only readout unnecessarily fragile:
    all atoms start from the same species embedding and geometry must first be
    transferred from edge/angle channels into the atom channel.  This readout
    keeps the physically local character of an energy model while giving the
    learned two- and three-body representations a direct route to the target.

    For each graph it builds three scalar contributions:

        atomic term  = sum_i epsilon_i(h_1) / N_atoms
        pair term    = sum_b epsilon_b(h_2) / N_atoms
        angular term = sum_a epsilon_a(h_3) / N_atoms

    A very small final MLP combines the three terms into normalized E/N.  The
    normalization by N_atoms preserves sensible scaling if the supercell size is
    changed later.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = int(hidden_dim or dim)
        bottleneck = max(hidden_dim // 2, 16)

        def make_head():
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, bottleneck),
                nn.SiLU(),
                nn.Linear(bottleneck, 1),
            )

        self.atom_head = make_head()
        self.bond_head = make_head()
        self.angle_head = make_head()
        self.combine = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

        # Targets are standardized around zero.  Biases start at zero, while
        # weights retain PyTorch's ordinary nonzero initialization so geometry
        # can influence the prediction from the first forward pass.
        for module in (self.atom_head[-1], self.bond_head[-1], self.angle_head[-1], self.combine[-1]):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    @staticmethod
    def _first_existing(data, names):
        for name in names:
            value = getattr(data, name, None)
            if value is not None:
                return value
        return None

    @staticmethod
    def _batch_or_zeros(data, names, n_items: int, device: torch.device) -> torch.Tensor:
        for name in names:
            value = getattr(data, name, None)
            if value is not None:
                return value.to(device=device, dtype=torch.long)
        return torch.zeros(n_items, dtype=torch.long, device=device)

    def _per_atom_normalized_term(
        self,
        hidden: torch.Tensor | None,
        batch_index: torch.Tensor | None,
        head: nn.Module,
        n_graphs: int,
        atom_counts: torch.Tensor,
    ) -> torch.Tensor:
        if hidden is None or batch_index is None or hidden.numel() == 0:
            return atom_counts.new_zeros((n_graphs,))

        local = head(hidden.float()).view(-1)
        total = scatter_sum(local, batch_index, dim_size=n_graphs)
        return total / atom_counts

    def forward(self, data):
        h_atom = self._first_existing(data, ["h_1", "h_atm", "h_g_node", "h_node"])
        h_bond = self._first_existing(data, ["h_2", "h_bnd", "h_a_node", "h_edge"])
        h_angle = self._first_existing(data, ["h_3", "h_ang", "h_a_edge"])

        if h_atom is None:
            raise AttributeError(
                "LocalEnvironmentEnergyReadout requires atom hidden states under "
                "h_1/h_atm/h_g_node/h_node."
            )

        device = h_atom.device
        atom_batch = self._batch_or_zeros(
            data, ["x_atm_batch", "node_G_batch", "batch"], h_atom.size(0), device
        )
        n_graphs = int(atom_batch.max().item()) + 1 if atom_batch.numel() else 1
        atom_counts = scatter_sum(
            torch.ones(h_atom.size(0), dtype=h_atom.dtype, device=device),
            atom_batch,
            dim_size=n_graphs,
        ).clamp_min(1.0)

        bond_batch = None
        if h_bond is not None:
            bond_batch = self._batch_or_zeros(
                data, ["x_bnd_batch", "node_A_batch"], h_bond.size(0), device
            )

        angle_batch = None
        if h_angle is not None:
            angle_batch = self._batch_or_zeros(
                data, ["x_ang_batch", "edge_A_batch"], h_angle.size(0), device
            )

        atom_term = self._per_atom_normalized_term(
            h_atom, atom_batch, self.atom_head, n_graphs, atom_counts
        )
        bond_term = self._per_atom_normalized_term(
            h_bond, bond_batch, self.bond_head, n_graphs, atom_counts
        )
        angle_term = self._per_atom_normalized_term(
            h_angle, angle_batch, self.angle_head, n_graphs, atom_counts
        )

        local_terms = torch.stack([atom_term, bond_term, angle_term], dim=-1)
        return self.combine(local_terms).view(-1)


def build_regression_model(device: str = DEVICE) -> GNN:
    """Build the ALIGNN/order graph-scalar regression model used by Catalyst."""
    task = build_regression_task()
    conv_type = str(MODEL_CONFIG.get("conv_type", "gine")).lower().strip()

    model = build_task_model(
        task=task,
        model_type="gnn_builder",
        apply_task_model_kwargs=False,
        preset="alignn",
        processor_type="order",
        conv_type=conv_type,
        decoder=LocalEnvironmentEnergyReadout(
            dim=MODEL_CONFIG["hidden_dim"],
            hidden_dim=MODEL_CONFIG.get("readout_hidden_dim", MODEL_CONFIG["hidden_dim"]),
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


def _with_emt(atoms):
    atoms = atoms.copy()
    atoms.pbc = True
    atoms.calc = EMT()
    return atoms


def _deform_structure(base_atoms, deformation: np.ndarray):
    """Apply a homogeneous deformation to the cell and scaled coordinates."""
    atoms = base_atoms.copy()
    old_cell = np.asarray(atoms.cell.array, dtype=float)
    new_cell = old_cell @ np.asarray(deformation, dtype=float).T
    atoms.set_cell(new_cell, scale_atoms=True)
    atoms.wrap()
    atoms.calc = EMT()
    return atoms


def _linspace_config(name: str, default_min: float, default_max: float, default_count: int):
    lower = float(GENERATION_CONFIG.get(f"{name}_min", default_min))
    upper = float(GENERATION_CONFIG.get(f"{name}_max", default_max))
    count = int(GENERATION_CONFIG.get(f"{name}_count", default_count))
    if count < 1:
        raise ValueError(f"dataset_generation.{name}_count must be >= 1")
    return np.linspace(lower, upper, count)


def generate_structure_ensemble() -> list[tuple[Any, dict[str, Any]]]:
    """Create a deterministic, structurally diverse FCC-Al regression dataset."""
    base = build_al_fcc_supercell()
    seed = int(GENERATION_CONFIG.get("seed", 112358))
    rng = np.random.default_rng(seed)

    structures: list[tuple[Any, dict[str, Any]]] = []

    # 1) Equation-of-state-like isotropic strains.
    for strain in _linspace_config("isotropic_strain", -0.08, 0.08, 25):
        deformation = np.eye(3) * (1.0 + float(strain))
        atoms = _deform_structure(base, deformation)
        structures.append(
            (atoms, {"family": "isotropic", "parameter": float(strain)})
        )

    # 2) Uniaxial strains along each Cartesian direction.
    uniaxial_values = _linspace_config("uniaxial_strain", -0.06, 0.06, 11)
    for axis in range(3):
        for strain in uniaxial_values:
            if abs(float(strain)) < 1.0e-12:
                continue
            deformation = np.eye(3)
            deformation[axis, axis] = 1.0 + float(strain)
            atoms = _deform_structure(base, deformation)
            structures.append(
                (
                    atoms,
                    {
                        "family": f"uniaxial_{'xyz'[axis]}",
                        "parameter": float(strain),
                    },
                )
            )

    # 3) Simple shears.  These deliberately perturb bond angles as well as
    # distances, making the 3-body ALIGNN pathway useful in the example.
    shear_values = _linspace_config("shear_strain", -0.05, 0.05, 11)
    shear_components = [(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]
    for i, j, label in shear_components:
        for gamma in shear_values:
            if abs(float(gamma)) < 1.0e-12:
                continue
            deformation = np.eye(3)
            deformation[i, j] = float(gamma)
            atoms = _deform_structure(base, deformation)
            structures.append(
                (atoms, {"family": f"shear_{label}", "parameter": float(gamma)})
            )

    # 4) Thermal-like random displacements about the ideal lattice. Remove the
    # center-of-mass translation so the perturbation changes local geometry only.
    sigmas = GENERATION_CONFIG.get(
        "random_displacement_sigmas_A", [0.01, 0.03, 0.05, 0.08, 0.12]
    )
    samples_per_sigma = int(GENERATION_CONFIG.get("random_samples_per_sigma", 8))
    for sigma in sigmas:
        sigma = float(sigma)
        for sample_idx in range(samples_per_sigma):
            atoms = base.copy()
            displacement = rng.normal(0.0, sigma, size=(len(atoms), 3))
            displacement -= displacement.mean(axis=0, keepdims=True)
            atoms.positions += displacement
            atoms.wrap()
            atoms.calc = EMT()
            structures.append(
                (
                    atoms,
                    {
                        "family": "random_displacement",
                        "parameter": sigma,
                        "replicate": int(sample_idx),
                    },
                )
            )

    # 5) Short MD trajectories at several temperatures. These add realistic
    # correlated thermal environments without making the full dataset a single
    # narrow trajectory.
    temperatures = GENERATION_CONFIG.get("md_temperatures_K", [300.0, 600.0, 900.0, 1200.0])
    md_equilibration_steps = int(GENERATION_CONFIG.get("md_equilibration_steps", 50))
    md_sample_stride = int(GENERATION_CONFIG.get("md_sample_stride", 10))
    md_samples_per_temperature = int(GENERATION_CONFIG.get("md_samples_per_temperature", 10))
    timestep_fs = float(GENERATION_CONFIG.get("md_timestep_fs", AL_CONFIG.get("timestep_fs", 1.0)))

    for temp_index, temperature in enumerate(temperatures):
        atoms = _with_emt(base)
        # Use a dedicated RNG stream for deterministic but independent velocities.
        velocity_rng = np.random.default_rng(seed + 1000 + temp_index)
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=float(temperature),
            rng=velocity_rng,
        )
        Stationary(atoms)
        ZeroRotation(atoms)
        dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
        if md_equilibration_steps > 0:
            dyn.run(md_equilibration_steps)

        for sample_idx in range(md_samples_per_temperature):
            if md_sample_stride > 0:
                dyn.run(md_sample_stride)
            frame = _with_emt(atoms)
            structures.append(
                (
                    frame,
                    {
                        "family": "md",
                        "parameter": float(temperature),
                        "replicate": int(sample_idx),
                    },
                )
            )

    energies_per_atom = []
    family_counts: dict[str, int] = {}
    for atoms, metadata in structures:
        energy_per_atom = float(atoms.get_potential_energy()) / float(len(atoms))
        energies_per_atom.append(energy_per_atom)
        family = str(metadata["family"])
        family_counts[family] = family_counts.get(family, 0) + 1

    if not structures:
        raise RuntimeError("Dataset generation produced zero structures.")

    print(f"Generated {len(structures)} structurally diverse FCC-Al structures.")
    print(
        "Energy/atom range: "
        f"{min(energies_per_atom):.6f} to {max(energies_per_atom):.6f} eV/atom; "
        f"std={np.std(energies_per_atom):.6f} eV/atom"
    )
    print("Structure-family counts:")
    for family, count in sorted(family_counts.items()):
        print(f"  {family:24s}: {count}")

    return structures


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


def atoms_to_alignn_graph(
    atoms,
    energy_eV: float,
    energy_per_atom_eV: float,
    gid: str,
    metadata: dict[str, Any],
):
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
            "include_equivariant_fields": AL_CONFIG.get("include_equivariant_fields", True),
            "include_edge_geometry": AL_CONFIG.get("include_edge_geometry", True),
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
    graph.energy_per_atom_eV = torch.tensor([energy_per_atom_eV], dtype=torch.float32)
    graph.n_atoms_reference = torch.tensor([len(atoms)], dtype=torch.long)
    graph.structure_family = str(metadata.get("family", "unknown"))
    graph.structure_parameter = torch.tensor(
        [float(metadata.get("parameter", 0.0))], dtype=torch.float32
    )

    # Raw energy/atom target initially; normalize_targets(...) overwrites these.
    graph.y = torch.tensor([energy_per_atom_eV], dtype=torch.float32)
    graph.target_scalar = torch.tensor([energy_per_atom_eV], dtype=torch.float32)

    return graph


def generate_data(cat: Catalyst) -> None:
    data_dir = reset_dir(cat.parameters["io_dict"]["data_dir"])
    structures = generate_structure_ensemble()
    manifest = []

    for idx, (atoms, metadata) in enumerate(structures):
        energy_eV = float(atoms.get_potential_energy())
        energy_per_atom_eV = energy_eV / float(len(atoms))
        family = str(metadata.get("family", "structure"))
        gid = f"al_fcc_{family}_{idx:05d}"

        print(
            f"Building graph {idx + 1:4d}/{len(structures):4d}: {gid}, "
            f"E/N={energy_per_atom_eV:.6f} eV/atom"
        )

        graph = atoms_to_alignn_graph(
            atoms=atoms,
            energy_eV=energy_eV,
            energy_per_atom_eV=energy_per_atom_eV,
            gid=gid,
            metadata=metadata,
        )
        torch.save(graph, data_dir / f"{gid}.pt")

        manifest.append(
            {
                "gid": gid,
                "family": family,
                "parameter": float(metadata.get("parameter", 0.0)),
                "energy_eV": energy_eV,
                "energy_per_atom_eV": energy_per_atom_eV,
                "n_atoms": len(atoms),
            }
        )

    with (Path(cat.parameters["io_dict"]["main_path"]) / "dataset_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)


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

    test_fraction = float(CONFIG["sampling"]["test_fraction"])
    validation_fraction_of_remaining = float(
        CONFIG["sampling"]["validation_fraction_of_remaining"]
    )

    # Stratify by structure family so train/validation/test each contain the
    # different deformation modes represented in the example.
    family_to_gids: dict[str, list[str]] = {}
    for path, gid in zip(data_files, gids):
        graph = safe_torch_load(path, map_location="cpu")
        family = str(getattr(graph, "structure_family", "unknown"))
        family_to_gids.setdefault(family, []).append(gid)

    training_gids: list[str] = []
    validation_gids: list[str] = []
    test_gids: list[str] = []

    for family, family_gids in sorted(family_to_gids.items()):
        family_gids = list(family_gids)
        rng.shuffle(family_gids)
        n_family = len(family_gids)

        n_test = int(round(test_fraction * n_family))
        if test_fraction > 0 and n_family >= 3:
            n_test = max(1, n_test)
        n_test = min(n_test, max(n_family - 2, 0))

        remaining = family_gids[n_test:]
        n_validation = int(round(validation_fraction_of_remaining * len(remaining)))
        if validation_fraction_of_remaining > 0 and len(remaining) >= 2:
            n_validation = max(1, n_validation)
        n_validation = min(n_validation, max(len(remaining) - 1, 0))

        test_gids.extend(family_gids[:n_test])
        validation_gids.extend(remaining[:n_validation])
        training_gids.extend(remaining[n_validation:])

    rng.shuffle(training_gids)
    rng.shuffle(validation_gids)
    rng.shuffle(test_gids)

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
    Normalize energy-per-atom graph targets using training-set statistics.

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

    training_energies_per_atom = []
    for gid in training_gids:
        graph = safe_torch_load(_graph_path_from_gid(cat, gid))
        training_energies_per_atom.append(float(graph.energy_per_atom_eV.view(-1)[0]))

    target_mean = float(np.mean(training_energies_per_atom))
    target_std = float(np.std(training_energies_per_atom))
    if target_std < 1.0e-12:
        target_std = 1.0

    data_files = sorted(glob.glob(os.path.join(cat.parameters["io_dict"]["data_dir"], "*.pt")))
    for path in data_files:
        graph = safe_torch_load(path)
        energy_per_atom = float(graph.energy_per_atom_eV.view(-1)[0])
        normalized = (energy_per_atom - target_mean) / target_std

        graph.y = torch.tensor([normalized], dtype=torch.float32)
        graph.target_scalar = torch.tensor([normalized], dtype=torch.float32)
        graph.target_normalization = {
            "target_mean": target_mean,
            "target_std": target_std,
            "target_units": "eV/atom",
            "target_type": "energy_per_atom",
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
                "target_units": "eV/atom",
                "target_type": "energy_per_atom",
            },
            handle,
            indent=2,
        )

    print(f"Target normalization: mean={target_mean:.6f} eV/atom, std={target_std:.6f} eV/atom")


# =============================================================================
# LEARNING DIAGNOSTICS
# =============================================================================


def _training_gids(cat: Catalyst) -> list[str]:
    split_path = (
        Path(cat.parameters["io_dict"]["main_path"])
        / "samples"
        / "model_samples"
        / "train_valid_split.npy"
    )
    split = load_dictionary(split_path)
    return list(split.get("training", []))


def _diverse_training_subset(cat: Catalyst, subset_size: int) -> list[Any]:
    """Select approximately evenly spaced training targets for diagnostics."""
    graphs = _load_graphs_from_gids(cat, _training_gids(cat))
    if not graphs:
        raise RuntimeError("Training split is empty; diagnostics cannot run.")

    graphs.sort(key=lambda g: float(g.target_scalar.view(-1)[0]))
    subset_size = max(2, min(int(subset_size), len(graphs)))
    indices = np.linspace(0, len(graphs) - 1, subset_size).round().astype(int)
    return [graphs[int(i)] for i in indices]


def run_preflight_learning_diagnostics(cat: Catalyst) -> None:
    """Check prediction shape and verify that a finite backward signal exists.

    Initial scalar-output separation is reported but is not treated as a hard
    requirement. The subsequent tiny-subset overfit test is the authoritative
    check that the graph/model pair can learn structure-dependent variation.
    """
    from torch_geometric.loader import DataLoader

    graphs = _diverse_training_subset(cat, subset_size=2)
    follow_batch = _follow_batch_names(graphs)
    loader = DataLoader(graphs, batch_size=2, shuffle=False, follow_batch=follow_batch)
    batch = next(iter(loader))

    gnn = build_regression_model(DEVICE)
    batch = batch.to(gnn.device)
    gnn.model.train()

    raw_pred = gnn.model(batch)
    preds, targets, _ = gnn._accumulate_predictions(
        raw_pred, batch, cat.parameters, return_y=True
    )
    preds, targets = gnn._align_pred_and_target(preds, targets)

    pred_values = as_numpy_tensor(preds).reshape(-1)
    target_values = as_numpy_tensor(targets).reshape(-1)
    if pred_values.size < 2 or target_values.size < 2:
        raise RuntimeError("Preflight diagnostic expected two graph predictions/targets.")

    structure_difference = float(abs(pred_values[1] - pred_values[0]))
    tolerance = float(DIAGNOSTICS_CONFIG.get("preflight_output_difference_tolerance", 1.0e-10))

    loss = nn.MSELoss()(preds.reshape(-1), targets.reshape(-1))
    gnn.model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sq = 0.0
    for parameter in gnn.model.parameters():
        if parameter.grad is not None:
            grad_sq += float(torch.sum(parameter.grad.detach() ** 2).cpu())
    grad_norm = math.sqrt(grad_sq)

    print("Preflight learning diagnostics:")
    print(f"  target pair          = {target_values.tolist()}")
    print(f"  initial predictions  = {pred_values.tolist()}")
    print(f"  prediction difference= {structure_difference:.6e}")
    print(f"  gradient norm        = {grad_norm:.6e}")

    # Equal or nearly equal *random initial scalar outputs* are not, by
    # themselves, evidence of a collapsed representation. A zero-centered
    # regression head can legitimately map two different latent states to nearly
    # the same scalar before optimization. The hard preflight requirement is that
    # backpropagation reaches the model; the tiny-subset overfit test below is the
    # stronger end-to-end test of structure-dependent learnability.
    if structure_difference <= tolerance:
        print(
            "  note                 = initial scalar outputs are nearly identical; "
            "continuing to the gradient and tiny-overfit diagnostics"
        )

    if not math.isfinite(grad_norm) or grad_norm <= 1.0e-12:
        raise RuntimeError(
            "The preflight backward pass produced no useful model gradient."
        )


def run_overfit_sanity_check(cat: Catalyst) -> None:
    """Run an advisory tiny-set optimization diagnostic.

    This uses a separate throwaway model and therefore cannot contaminate the
    production Catalyst checkpoint.  Failure to hit a particular loss ratio is
    reported as a warning by default rather than treated as a package/runtime
    failure; set diagnostics.overfit_fail_hard=true to restore strict behavior.
    """
    from torch_geometric.loader import DataLoader

    subset_size = int(DIAGNOSTICS_CONFIG.get("overfit_subset_size", 6))
    steps = int(DIAGNOSTICS_CONFIG.get("overfit_steps", 250))
    learning_rate = float(DIAGNOSTICS_CONFIG.get("overfit_learning_rate", 5.0e-3))
    required_fraction = float(DIAGNOSTICS_CONFIG.get("overfit_required_loss_fraction", 0.75))
    fail_hard = bool(DIAGNOSTICS_CONFIG.get("overfit_fail_hard", False))

    graphs = _diverse_training_subset(cat, subset_size=subset_size)
    follow_batch = _follow_batch_names(graphs)
    loader = DataLoader(
        graphs,
        batch_size=len(graphs),
        shuffle=False,
        follow_batch=follow_batch,
        num_workers=0,
    )
    batch = next(iter(loader))

    gnn = build_regression_model(DEVICE)
    batch = batch.to(gnn.device)
    # Adam without weight decay is intentionally used for a memorization test.
    optimizer = torch.optim.Adam(gnn.model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    def prediction_and_loss():
        raw = gnn.model(batch)
        pred, target, _ = gnn._accumulate_predictions(
            raw, batch, cat.parameters, return_y=True
        )
        pred, target = gnn._align_pred_and_target(pred, target)
        return pred.reshape(-1), target.reshape(-1), loss_fn(pred.reshape(-1), target.reshape(-1))

    gnn.model.eval()
    with torch.no_grad():
        initial_pred, initial_target, initial_loss_tensor = prediction_and_loss()
        initial_loss = float(initial_loss_tensor.detach().cpu())
        initial_pred_std = float(initial_pred.detach().float().std(unbiased=False).cpu())
        target_std = float(initial_target.detach().float().std(unbiased=False).cpu())

    best_loss = initial_loss
    gnn.model.train()
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        pred, target, loss = prediction_and_loss()
        if not torch.isfinite(loss):
            best_loss = float("nan")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.model.parameters(), max_norm=100.0)
        optimizer.step()
        current = float(loss.detach().cpu())
        if current < best_loss:
            best_loss = current

    gnn.model.eval()
    with torch.no_grad():
        final_pred, final_target, final_loss_tensor = prediction_and_loss()
        final_loss = float(final_loss_tensor.detach().cpu())
        final_pred_std = float(final_pred.detach().float().std(unbiased=False).cpu())

    fraction = best_loss / max(initial_loss, 1.0e-15)

    print("Tiny-subset overfit diagnostic:")
    print(f"  subset size        = {len(graphs)}")
    print(f"  optimization steps = {steps}")
    print(f"  target std         = {target_std:.6e}")
    print(f"  initial pred std   = {initial_pred_std:.6e}")
    print(f"  final pred std     = {final_pred_std:.6e}")
    print(f"  initial MSE        = {initial_loss:.6e}")
    print(f"  best MSE           = {best_loss:.6e}")
    print(f"  final MSE          = {final_loss:.6e}")
    print(f"  best/initial       = {fraction:.6f}")

    failed = (
        not math.isfinite(best_loss)
        or not math.isfinite(final_loss)
        or fraction > required_fraction
    )
    if failed:
        message = (
            "Tiny-set optimization did not reach the configured loss-reduction "
            f"target (best/initial={fraction:.4f}, required<={required_fraction:.4f}). "
            "Continuing to the real Catalyst training run because this diagnostic "
            "is optimizer- and initialization-dependent. The post-training "
            "prediction-spread diagnostic will determine whether the trained model "
            "actually collapsed."
        )
        if fail_hard:
            raise RuntimeError(message)
        print(f"WARNING: {message}")


def _prediction_spread_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    target_std = float(np.std(y_true)) if y_true.size else float("nan")
    prediction_std = float(np.std(y_pred)) if y_pred.size else float("nan")
    ratio = prediction_std / target_std if target_std > 1.0e-15 else float("nan")
    mae, rmse = _parity_metrics(y_true, y_pred)

    if y_true.size > 1 and np.var(y_true) > 1.0e-15:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan")

    return {
        "target_std": target_std,
        "prediction_std": prediction_std,
        "prediction_to_target_std_ratio": ratio,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def diagnose_trained_model(cat: Catalyst, fail_on_collapse: bool = True) -> dict[str, dict[str, float]]:
    """Measure prediction spread on every split and detect mean-predictor collapse."""
    main_path = Path(cat.parameters["io_dict"]["main_path"])
    checkpoint_path = latest_checkpoint(main_path / "models" / "training")
    split_train_valid = load_dictionary(main_path / "samples" / "model_samples" / "train_valid_split.npy")
    split_test = load_dictionary(main_path / "samples" / "test_data.npy")
    split_map = {
        "training": split_train_valid.get("training", []),
        "validation": split_train_valid.get("validation", []),
        "test": split_test.get("gids", []),
    }

    target_mean, target_std = load_target_normalization(cat)
    batch_size = int(cat.parameters["loader_dict"]["batch_size"][1])
    batch_size = max(batch_size, 1)
    metrics: dict[str, dict[str, float]] = {}

    print("Post-training prediction-spread diagnostics:")
    for split_name, gids in split_map.items():
        graphs = _load_graphs_from_gids(cat, gids)
        y_norm, pred_norm = _predict_graphs_with_checkpoint(
            cat, graphs, checkpoint_path, batch_size
        )
        y = y_norm * target_std + target_mean
        pred = pred_norm * target_std + target_mean
        stats = _prediction_spread_stats(y, pred)
        metrics[split_name] = stats
        print(
            f"  {split_name:10s}: target_std={stats['target_std']:.6e}, "
            f"pred_std={stats['prediction_std']:.6e}, "
            f"ratio={stats['prediction_to_target_std_ratio']:.4f}, "
            f"MAE={stats['mae']:.6e} eV/atom, R2={stats['r2']:.4f}"
        )

    metrics_path = main_path / "model_diagnostics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Wrote {metrics_path}")

    collapse_threshold = float(DIAGNOSTICS_CONFIG.get("collapse_std_ratio", 0.05))
    training_ratio = metrics.get("training", {}).get(
        "prediction_to_target_std_ratio", float("nan")
    )
    if fail_on_collapse and math.isfinite(training_ratio) and training_ratio < collapse_threshold:
        raise RuntimeError(
            "Prediction collapse detected: the training prediction standard "
            f"deviation is only {training_ratio:.4f} times the target standard "
            "deviation. The model is behaving like a constant mean predictor."
        )

    return metrics


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

    cat.set_params({'io_dict': {'write_indv_pred': True, 'results_dir': str(reset_dir(results_dir)), 'model_dir': str(model_dir), 'loaded_model_name': loaded_model_name}}, save_params=False)

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
    cat.set_params({'io_dict': {'samples_dir': str(main_path / 'samples')}}, save_params=False)
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

    cat.set_params({'io_dict': {'write_indv_pred': False, 'results_dir': str(reset_dir(results_dir)), 'model_dir': str(model_dir), 'loaded_model_name': loaded_model_name}}, save_params=False)

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
    cat.set_params({'io_dict': {'model_dir': str(model_dir)}}, save_params=False)

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
    output_path = get_figures_dir(cat) / "training_energy_loss.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")



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
        4. denormalizes predictions/targets back to eV/atom,
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
            ax.set_title(f"{split_name}\nN={len(y_eV)}, MAE={mae:.4f} eV/atom, RMSE={rmse:.4f} eV/atom")
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
        ax.set_xlabel("EMT energy (eV/atom)")

    axes[0].set_ylabel("GNN-predicted energy (eV/atom)")
    fig.suptitle("Energy-per-atom parity plots for training, validation, and test splits", y=1.02)
    plt.tight_layout()
    output_path = get_figures_dir(cat) / "energy_parity_train_validation_test.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def main() -> None:
    task = build_regression_task()
    cat = Catalyst(
        parameter_file=CONFIG_PATH,
        parameters={
            "loader_dict": {
                "batch_size": TRAINING_BATCH_SIZE,
                "shuffle_loader": True,
            },
            "model_dict": {
                "num_epochs": TRAINING_NUM_EPOCHS_OVERRIDE,
                "train_delta": TRAINING_DELTA_OVERRIDE,
                "train_tolerance": TRAINING_TOLERANCE_OVERRIDE,
            },
        },
        task=task,
    )

    # Ensure base directories exist.
    make_dir(cat.parameters["io_dict"]["main_path"])
    make_dir(cat.parameters["io_dict"]["data_dir"])

    if RUN_GENERATE_GRAPHS:
        generate_data(cat)

    if RUN_GENERATE_SAMPLES:
        sample_data(cat)

    if RUN_NORMALIZE_TARGETS:
        normalize_targets(cat)

    if RUN_PREFLIGHT_DIAGNOSTICS:
        run_preflight_learning_diagnostics(cat)

    if RUN_OVERFIT_SANITY:
        run_overfit_sanity_check(cat)

    if RUN_TRAINING:
        train_model(cat)

        if RUN_POSTTRAIN_DIAGNOSTICS:
            diagnose_trained_model(cat, fail_on_collapse=True)

        if RUN_PLOT_TRAINING:
            plot_training_results(cat)

    if RUN_RETRAINING:
        cat.set_model(build_regression_model(DEVICE))
        cat.set_params({"model_dict": {"restart_training": True}}, save_params=False)
        retrain_model(cat, use_latest_checkpoint=True)

    if RUN_TESTING:
        cat.set_model(build_regression_model(DEVICE))
        test_model(cat)

    if RUN_PLOT_TEST:
        plot_test_data(cat)

    if RUN_PREDICTIONS:
        cat.set_model(build_regression_model(DEVICE))
        predict(cat)


if __name__ == "__main__":
    main()
