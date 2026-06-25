"""
End-to-end ALIGNN-style force-learning example for Al FCC MD.

Recommended location:
    catalyst/examples/gnn_example/al_fcc_alignn_force_example.py

What this example does
----------------------
1. Read a user-editable JSON config.
2. Build an Al FCC supercell with ASE.
3. Run a short ASE/EMT molecular dynamics trajectory.
4. Convert each ASE frame into Catalyst Atomic_Graph_Data using alignn_gen.
5. Assign the EMT atomic forces as the graph target.
6. Split graphs into train/validation/test sets.
7. Build an ALIGNN-style model using the new modular framework:

       build_model(
           preset="alignn",
           conv_type="gated_gcn",
           decoder=AtomicForceVectorDecoder(...),
       )

8. Train the model to predict the direct 3-component force vector per atom.
9. Save a checkpoint.
10. Reload the checkpoint.
11. Evaluate on the test set and write parity plots/CSV outputs.

Important distinction from the energy example
---------------------------------------------
This model predicts a vector per atom:

    output shape = [total_atoms_in_batch, 3]

It does NOT train three separate scalar models. The x/y/z force components are
learned jointly by one decoder head with out_dim=3.

Dependencies
------------
Required:
    ase
    torch
    torch_geometric
    matplotlib
    numpy
    catalyst

Run:
    python al_fcc_alignn_force_example.py

Config:
    Edit al_fcc_alignn_force_config.json next to this script, or set:
        CATALYST_AL_FCC_FORCE_CONFIG=/path/to/config.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet

from catalyst.graph.alignnd import alignn_gen

from catalyst.ml.gnn.modules.models.generic_gnn import build_model


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExampleConfig:
    # Output
    output_dir: str = "al_fcc_alignn_force_example_output"
    random_seed: int = 7

    # Structure / MD
    lattice_constant: float = 4.05
    repeat: Tuple[int, int, int] = (3, 3, 3)
    temperature_K: float = 800.0
    timestep_fs: float = 1.0
    md_steps: int = 300
    sample_every: int = 3
    equilibration_steps: int = 20

    # Graph generation
    cutoff: float = 3.35
    neighbor_k: int = -1
    include_dihedrals: bool = False
    max_graph_attempts: int = 6
    cutoff_scale: float = 1.15
    max_cutoff: float = 5.0

    # Dataset split
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    # Model
    hidden_dim: int = 128
    num_convs: int = 3
    conv_type: str = "gated_gcn"
    aggr_scheme: str = "add"

    # Training
    batch_size: int = 8
    num_epochs: int = 150
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0

    # Force normalization
    # "component" means a separate mean/std for Fx, Fy, Fz but still one vector model.
    # "scalar" means one global force mean/std for all components.
    force_normalization: str = "component"

    # Runtime
    device: str = "auto"
    num_workers: int = 0


CONFIG_PATH = Path(
    os.environ.get(
        "CATALYST_AL_FCC_FORCE_CONFIG",
        Path(__file__).with_name("al_fcc_alignn_force_config.json"),
    )
)


def _tuple3(value, name: str) -> Tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a list/tuple of length 3, got {value!r}.")
    return tuple(int(v) for v in value)


def load_config(config_path: Path = CONFIG_PATH) -> ExampleConfig:
    """
    Load ExampleConfig from JSON.

    If the JSON config does not exist yet, this writes a default config file and
    raises FileNotFoundError so the user can inspect/edit it before running.
    """
    config_path = Path(config_path)

    if not config_path.is_file():
        default_config = ExampleConfig()
        config_path.write_text(
            json.dumps(asdict(default_config), indent=2),
            encoding="utf-8",
        )
        raise FileNotFoundError(
            f"Could not find config file: {config_path}\n"
            f"A default config was written there. Edit it if needed, then rerun."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    valid_fields = {f.name for f in fields(ExampleConfig)}
    unknown = sorted(set(raw) - valid_fields)
    if unknown:
        raise KeyError(
            f"Unknown config keys in {config_path}: {unknown}\n"
            f"Valid keys are: {sorted(valid_fields)}"
        )

    cfg = ExampleConfig(**raw)
    cfg.repeat = _tuple3(cfg.repeat, "repeat")

    if str(cfg.device).lower() in {"auto", "default"}:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.force_normalization = str(cfg.force_normalization).lower()
    if cfg.force_normalization not in {"component", "scalar"}:
        raise ValueError(
            "force_normalization must be either 'component' or 'scalar', "
            f"got {cfg.force_normalization!r}."
        )

    return cfg


CONFIG = load_config()


# =============================================================================
# File helpers
# =============================================================================


def reset_dir(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_torch_load(path: os.PathLike[str] | str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# ASE MD generation
# =============================================================================


def build_al_fcc_supercell(config: ExampleConfig):
    atoms = bulk(
        "Al",
        "fcc",
        a=config.lattice_constant,
        cubic=True,
    )
    atoms = atoms.repeat(config.repeat)
    atoms.pbc = True
    atoms.calc = EMT()
    return atoms


def run_md_frames(config: ExampleConfig) -> List[Any]:
    """
    Run a short ASE/EMT MD trajectory and return sampled ASE Atoms frames.

    EMT is used only to generate a toy supervised dataset. It is not intended to
    be a high-fidelity Al potential for production.
    """
    atoms = build_al_fcc_supercell(config)

    MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature_K)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = VelocityVerlet(
        atoms,
        timestep=config.timestep_fs * units.fs,
    )

    frames = []
    force_rms_values = []

    for step in range(config.md_steps + 1):
        if step > 0:
            dyn.run(1)

        if step < config.equilibration_steps:
            continue

        if (step - config.equilibration_steps) % config.sample_every != 0:
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


# =============================================================================
# Catalyst ALIGNN graph generation
# =============================================================================


def _as_single_graph(obj):
    """
    alignn_gen can return either one graph or a list depending on Catalyst version
    and input form. Normalize that to one graph.
    """
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise RuntimeError("alignn_gen returned an empty list.")
        return obj[0]
    return obj


def atoms_to_alignn_graph(
    atoms,
    forces_eVA: np.ndarray,
    gid: str,
    config: ExampleConfig,
):
    """
    Convert one ASE Atoms frame into Catalyst Atomic_Graph_Data.
    """
    graph = alignn_gen(
        {
            "type": "alignnd",
            "raw_data": atoms,
            "node_labels": None,
            "element_list": ["Al"],
            "neighbor_params": [config.cutoff, config.neighbor_k],
            "is_dihedral": config.include_dihedrals,
            "store_raw_data": False,
            "use_pt": False,
            "include_angs": True,
            "cpu_cores": 1,
            "store_atoms_type": "ase-atoms",

            # New retry controls. Older alignn_gen versions may ignore these if
            # they are not wired through yet.
            "auto_retry_graph": True,
            "max_graph_attempts": config.max_graph_attempts,
            "cutoff_scale": config.cutoff_scale,
            "max_cutoff": config.max_cutoff,
            "require_bonds": True,
            "require_angles": True,
            "require_dihedrals": False,
            "retry_verbose": False,
        }
    )

    graph = _as_single_graph(graph)
    graph.gid = gid

    forces = torch.as_tensor(forces_eVA, dtype=torch.float32)

    # Raw forces for reporting.
    graph.forces_eVA = forces.clone()

    # y will later be overwritten with normalized forces for training.
    # Shape is [N_atoms, 3].
    graph.y = forces.clone()

    return graph


def generate_graph_dataset(config: ExampleConfig, output_dir: Path) -> List[Path]:
    """
    Generate MD frames, convert to graphs, and save graph .pt files.
    """
    graph_dir = reset_dir(output_dir / "graphs")
    frames = run_md_frames(config)

    graph_paths = []

    for idx, atoms in enumerate(frames):
        forces = atoms.get_forces()
        gid = f"al_fcc_md_force_{idx:05d}"

        force_rms = float(np.sqrt(np.mean(forces ** 2)))
        print(
            f"Building graph {idx + 1:4d}/{len(frames):4d}: "
            f"{gid}, force_rms={force_rms:.6f} eV/Ang"
        )

        graph = atoms_to_alignn_graph(
            atoms=atoms,
            forces_eVA=forces,
            gid=gid,
            config=config,
        )

        graph_path = graph_dir / f"{gid}.pt"
        torch.save(graph, graph_path)
        graph_paths.append(graph_path)

    return graph_paths


# =============================================================================
# Sampling / splitting
# =============================================================================


def split_graph_paths(
    graph_paths: Sequence[Path],
    config: ExampleConfig,
    output_dir: Path,
) -> Dict[str, List[str]]:
    """
    Random train/validation/test split.

    This can later be replaced by SODAS/UMAP sampling exactly like the synthetic
    workflow, but the random split is sufficient for validating force learning.
    """
    graph_paths = list(graph_paths)
    rng = np.random.default_rng(config.random_seed)
    indices = np.arange(len(graph_paths))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(round(config.train_fraction * n_total))
    n_valid = int(round(config.validation_fraction * n_total))

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    split = {
        "training": [Path(graph_paths[i]).stem for i in train_idx],
        "validation": [Path(graph_paths[i]).stem for i in valid_idx],
        "test": [Path(graph_paths[i]).stem for i in test_idx],
        "training_files": [str(graph_paths[i]) for i in train_idx],
        "validation_files": [str(graph_paths[i]) for i in valid_idx],
        "test_files": [str(graph_paths[i]) for i in test_idx],
    }

    with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)

    print(
        "Dataset split: "
        f"train={len(split['training_files'])}, "
        f"valid={len(split['validation_files'])}, "
        f"test={len(split['test_files'])}"
    )

    return split


def load_graphs(paths: Sequence[str | Path]) -> List[Any]:
    return [safe_torch_load(path) for path in paths]


def compute_force_normalization(
    graphs: Sequence[Any],
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute force normalization from training graphs only.

    mode="component":
        force_mean/std have shape [3], one value per vector component.

    mode="scalar":
        force_mean/std have shape [1], one global value across all components.

    Either way, the model still predicts one direct 3-component vector per atom.
    """
    all_forces = torch.cat(
        [g.forces_eVA.reshape(-1, 3).float() for g in graphs],
        dim=0,
    )

    if mode == "component":
        mean = all_forces.mean(dim=0)
        std = all_forces.std(dim=0)
    elif mode == "scalar":
        mean = all_forces.reshape(-1).mean().view(1)
        std = all_forces.reshape(-1).std().view(1)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    std = torch.where(std < 1e-12, torch.ones_like(std), std)

    return mean.float(), std.float()


def apply_force_normalization(
    graphs: Sequence[Any],
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
) -> None:
    for graph in graphs:
        forces = graph.forces_eVA.float().reshape(-1, 3)
        graph.y = (forces - force_mean) / force_std


def make_loader(
    graphs: Sequence[Any],
    config: ExampleConfig,
    shuffle: bool,
) -> DataLoader:
    """
    Build a PyG DataLoader with follow_batch fields for atom/bond/angle tensors.

    The force target graph.y is [N_atoms, 3]. PyG batching concatenates this into:
        [total_atoms_in_batch, 3]
    which matches the model output directly.
    """
    follow_batch = ["x_atm", "x_bnd"]

    if len(graphs) > 0 and hasattr(graphs[0], "x_ang"):
        follow_batch.append("x_ang")

    return DataLoader(
        list(graphs),
        batch_size=config.batch_size,
        shuffle=shuffle,
        follow_batch=follow_batch,
        num_workers=config.num_workers,
    )


# =============================================================================
# Model
# =============================================================================


class AtomicForceVectorDecoder(nn.Module):
    """
    Direct 3-component per-atom force decoder.

    It reads atom hidden features and predicts:

        [N_atoms_total_in_batch, 3]

    This is one vector-valued model head, not three separate scalar models.
    """

    def __init__(self, dim: int, act=nn.SiLU()):
        super().__init__()
        self.dim = dim
        self.force_head = nn.Sequential(
            nn.Linear(dim, dim),
            act,
            nn.Linear(dim, dim),
            act,
            nn.Linear(dim, 3),
        )

    def forward(self, data):
        if hasattr(data, "h_atm"):
            h_atom = data.h_atm
        elif hasattr(data, "h_1"):
            h_atom = data.h_1
        else:
            raise AttributeError("AtomicForceVectorDecoder expected data.h_atm or data.h_1.")

        return self.force_head(h_atom)


def build_alignn_force_model(config: ExampleConfig) -> nn.Module:
    """
    Build an ALIGNN-style force model through the public build_model(...) API.

    This uses the modular ALIGNN stack:

        atomic encoder + OrderProcessor(atom/bond/angle updates) + custom force-vector decoder

    The decoder is passed explicitly because this is a direct per-atom vector
    prediction task, not the default scalar decoder.
    """
    return build_model(
        preset="alignn",
        conv_type=config.conv_type,
        decoder=AtomicForceVectorDecoder(
            dim=config.hidden_dim,
            act=nn.SiLU(),
        ),
        num_species=1,
        cutoff=config.cutoff,
        dim=config.hidden_dim,
        num_convs=config.num_convs,
        out_dim=3,
        act=nn.SiLU(),
        aggr_scheme=config.aggr_scheme,
        encode_3body=True,
        dihedral=config.include_dihedrals,
    )


# =============================================================================
# Training / evaluation
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_atoms = 0

    loss_fn = nn.MSELoss()

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(batch)
        target = batch.y.to(pred.device)

        if pred.shape != target.shape:
            raise RuntimeError(
                f"Prediction/target shape mismatch: pred={tuple(pred.shape)}, "
                f"target={tuple(target.shape)}. Expected both to be [N_atoms, 3]."
            )

        loss = loss_fn(pred, target)
        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        n_batch_atoms = int(target.size(0))
        total_loss += float(loss.item()) * n_batch_atoms
        n_atoms += n_batch_atoms

    return total_loss / max(n_atoms, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
) -> Dict[str, Any]:
    model.eval()

    force_mean = force_mean.to(device)
    force_std = force_std.to(device)

    pred_all = []
    true_all = []

    for batch in loader:
        batch = batch.to(device)

        pred_norm = model(batch)
        true_norm = batch.y.to(pred_norm.device)

        if pred_norm.shape != true_norm.shape:
            raise RuntimeError(
                f"Prediction/target shape mismatch during evaluation: "
                f"pred={tuple(pred_norm.shape)}, target={tuple(true_norm.shape)}."
            )

        pred_f = pred_norm * force_std + force_mean
        true_f = true_norm * force_std + force_mean

        pred_all.append(pred_f.detach().cpu())
        true_all.append(true_f.detach().cpu())

    if not pred_all:
        return {
            "component_mae_eVA": float("nan"),
            "component_rmse_eVA": float("nan"),
            "vector_mae_eVA": float("nan"),
            "vector_rmse_eVA": float("nan"),
            "pred_forces_eVA": np.asarray([]),
            "true_forces_eVA": np.asarray([]),
        }

    pred = torch.cat(pred_all, dim=0).numpy()
    true = torch.cat(true_all, dim=0).numpy()

    diff = pred - true

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
        "pred_forces_eVA": pred,
        "true_forces_eVA": true,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: ExampleConfig,
    output_dir: Path,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
) -> Dict[str, List[float]]:
    device = config.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = {
        "epoch": [],
        "train_loss_norm": [],
        "valid_component_mae_eVA": [],
        "valid_component_rmse_eVA": [],
        "valid_vector_mae_eVA": [],
        "valid_vector_rmse_eVA": [],
    }

    best_valid = math.inf
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )

        valid_metrics = evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            force_mean=force_mean,
            force_std=force_std,
        )

        history["epoch"].append(epoch)
        history["train_loss_norm"].append(train_loss)
        history["valid_component_mae_eVA"].append(valid_metrics["component_mae_eVA"])
        history["valid_component_rmse_eVA"].append(valid_metrics["component_rmse_eVA"])
        history["valid_vector_mae_eVA"].append(valid_metrics["vector_mae_eVA"])
        history["valid_vector_rmse_eVA"].append(valid_metrics["vector_rmse_eVA"])

        if valid_metrics["component_mae_eVA"] < best_valid:
            best_valid = valid_metrics["component_mae_eVA"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "force_mean": force_mean.cpu(),
                    "force_std": force_std.cpu(),
                    "config": asdict(config),
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == config.num_epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"train loss={train_loss:.6e} | "
                f"valid comp MAE={valid_metrics['component_mae_eVA']:.6f} eV/Ang | "
                f"valid vec MAE={valid_metrics['vector_mae_eVA']:.6f} eV/Ang"
            )

    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch",
            "train_loss_norm",
            "valid_component_mae_eVA",
            "valid_component_rmse_eVA",
            "valid_vector_mae_eVA",
            "valid_vector_rmse_eVA",
        ])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss_norm"][i],
                history["valid_component_mae_eVA"][i],
                history["valid_component_rmse_eVA"][i],
                history["valid_vector_mae_eVA"][i],
                history["valid_vector_rmse_eVA"][i],
            ])

    return history


def _strip_wrapper_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
                changed = True
            if new_key.startswith("_orig_mod."):
                new_key = new_key[len("_orig_mod."):]
                changed = True
        cleaned[new_key] = value
    return cleaned


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str,
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    state = _strip_wrapper_prefixes(checkpoint["model_state"])

    model.load_state_dict(state, strict=True)
    model.to(device)

    force_mean = checkpoint["force_mean"].float()
    force_std = checkpoint["force_std"].float()

    return model, force_mean, force_std


# =============================================================================
# Plotting / output
# =============================================================================


def plot_training_history(history: Dict[str, List[float]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["valid_component_mae_eVA"], marker="o", label="Component MAE")
    ax.plot(history["epoch"], history["valid_vector_mae_eVA"], marker="s", label="Vector MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (eV/Ang)")
    ax.set_title("Al FCC ALIGNN force validation error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_validation_force_mae.png", dpi=200)
    plt.close(fig)


def plot_force_component_parity(metrics: Dict[str, Any], output_dir: Path) -> None:
    true = metrics["true_forces_eVA"].reshape(-1)
    pred = metrics["pred_forces_eVA"].reshape(-1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(true, pred, linestyle="", marker="o", markersize=3, markeredgecolor="k", alpha=0.7)

    if len(true) > 0:
        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="-")

    ax.set_xlabel("EMT force component (eV/Ang)")
    ax.set_ylabel("ALIGNN-predicted force component (eV/Ang)")
    ax.set_title(
        f"Force component parity: MAE={metrics['component_mae_eVA']:.4f} eV/Ang"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "test_force_component_parity.png", dpi=200)
    plt.close(fig)


def plot_force_vector_magnitude_parity(metrics: Dict[str, Any], output_dir: Path) -> None:
    true = metrics["true_forces_eVA"]
    pred = metrics["pred_forces_eVA"]

    true_mag = np.linalg.norm(true, axis=1)
    pred_mag = np.linalg.norm(pred, axis=1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(true_mag, pred_mag, linestyle="", marker="o", markersize=3, markeredgecolor="k", alpha=0.7)

    if len(true_mag) > 0:
        lo = float(min(true_mag.min(), pred_mag.min()))
        hi = float(max(true_mag.max(), pred_mag.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="-")

    ax.set_xlabel("EMT |F| (eV/Ang)")
    ax.set_ylabel("ALIGNN-predicted |F| (eV/Ang)")
    ax.set_title(
        f"Force vector magnitude parity: vector MAE={metrics['vector_mae_eVA']:.4f} eV/Ang"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "test_force_vector_magnitude_parity.png", dpi=200)
    plt.close(fig)


def write_predictions(metrics: Dict[str, Any], output_dir: Path) -> None:
    true = metrics["true_forces_eVA"]
    pred = metrics["pred_forces_eVA"]
    diff = pred - true

    with (output_dir / "test_force_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "atom_row",
            "true_fx_eVA",
            "true_fy_eVA",
            "true_fz_eVA",
            "pred_fx_eVA",
            "pred_fy_eVA",
            "pred_fz_eVA",
            "err_fx_eVA",
            "err_fy_eVA",
            "err_fz_eVA",
            "vector_error_eVA",
        ])

        for idx in range(true.shape[0]):
            vec_err = float(np.linalg.norm(diff[idx]))
            writer.writerow([
                idx,
                float(true[idx, 0]),
                float(true[idx, 1]),
                float(true[idx, 2]),
                float(pred[idx, 0]),
                float(pred[idx, 1]),
                float(pred[idx, 2]),
                float(diff[idx, 0]),
                float(diff[idx, 1]),
                float(diff[idx, 2]),
                vec_err,
            ])


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    config = CONFIG
    set_random_seeds(config.random_seed)

    output_dir = reset_dir(config.output_dir)

    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    shutil.copy2(CONFIG_PATH, output_dir / "input_config.json")

    print("Config file:", CONFIG_PATH.resolve())
    print("Output directory:", output_dir.resolve())
    print("Device:", config.device)

    # 1. Graph generation
    graph_paths = generate_graph_dataset(config=config, output_dir=output_dir)

    # 2. Sampling/splitting
    split = split_graph_paths(
        graph_paths=graph_paths,
        config=config,
        output_dir=output_dir,
    )

    train_graphs = load_graphs(split["training_files"])
    valid_graphs = load_graphs(split["validation_files"])
    test_graphs = load_graphs(split["test_files"])

    # 3. Force normalization based on training set only.
    force_mean, force_std = compute_force_normalization(
        train_graphs,
        mode=config.force_normalization,
    )

    apply_force_normalization(train_graphs, force_mean, force_std)
    apply_force_normalization(valid_graphs, force_mean, force_std)
    apply_force_normalization(test_graphs, force_mean, force_std)

    print("Force normalization mode:", config.force_normalization)
    print("Force mean:", force_mean.tolist())
    print("Force std:", force_std.tolist())

    train_loader = make_loader(train_graphs, config=config, shuffle=True)
    valid_loader = make_loader(valid_graphs, config=config, shuffle=False)
    test_loader = make_loader(test_graphs, config=config, shuffle=False)

    # 4. Model building
    model = build_alignn_force_model(config)
    print(model)

    # 5. Training
    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=config,
        output_dir=output_dir,
        force_mean=force_mean,
        force_std=force_std,
    )

    plot_training_history(history, output_dir)

    # 6. Reload best checkpoint and test.
    best_model = build_alignn_force_model(config)
    best_model, force_mean, force_std = load_model_checkpoint(
        model=best_model,
        checkpoint_path=output_dir / "best_model.pt",
        device=config.device,
    )

    test_metrics = evaluate(
        model=best_model,
        loader=test_loader,
        device=config.device,
        force_mean=force_mean,
        force_std=force_std,
    )

    print(
        "Test metrics: "
        f"component MAE={test_metrics['component_mae_eVA']:.6f} eV/Ang, "
        f"component RMSE={test_metrics['component_rmse_eVA']:.6f} eV/Ang, "
        f"vector MAE={test_metrics['vector_mae_eVA']:.6f} eV/Ang, "
        f"vector RMSE={test_metrics['vector_rmse_eVA']:.6f} eV/Ang"
    )

    plot_force_component_parity(test_metrics, output_dir)
    plot_force_vector_magnitude_parity(test_metrics, output_dir)
    write_predictions(test_metrics, output_dir)

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "component_mae_eVA": test_metrics["component_mae_eVA"],
                "component_rmse_eVA": test_metrics["component_rmse_eVA"],
                "vector_mae_eVA": test_metrics["vector_mae_eVA"],
                "vector_rmse_eVA": test_metrics["vector_rmse_eVA"],
            },
            handle,
            indent=2,
        )

    print("Done.")
    print(f"Results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
