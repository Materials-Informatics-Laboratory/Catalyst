"""
End-to-end ALIGNN-style energy-learning example for Al FCC MD.

Recommended location:
    catalyst/examples/gnn_example/al_fcc_alignn_energy_example.py

What this example does
----------------------
1. Build an Al FCC supercell with ASE.
2. Run a short ASE/EMT molecular dynamics trajectory.
3. Convert each ASE frame into Catalyst Atomic_Graph_Data using alignn_gen.
4. Assign the EMT total potential energy as the graph target.
5. Split graphs into train/validation/test sets.
6. Build an ALIGNN-style model using the new modular framework:

       build_model(
           preset="alignn",
           conv_type="gated_gcn",
           decoder=AtomicEnergyReadout(...),
       )

7. Train the model to predict total energy.
8. Save a checkpoint.
9. Reload the checkpoint.
10. Evaluate on the test set and write parity/training plots.

Why this is useful
------------------
This validates the new modular model setup using an actually atomistic graph.
The old synthetic random-graph workflow should use encoder_type="generic".
This workflow uses the atomistic/ALIGNN path.

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
    python al_fcc_alignn_energy_example_v2.py

Config:
    Edit al_fcc_alignn_energy_config.json next to this script, or set:
        CATALYST_AL_FCC_CONFIG=/path/to/config.json

Notes
-----
This example intentionally uses a local PyTorch training loop instead of the
Catalyst high-level Trainer/GNN wrapper. That keeps the example focused on
validating:
    ASE atoms -> alignn_gen -> AtomicGraphEncoder -> OrderProcessor -> decoder
without involving Catalyst's accumulation/prediction wrappers.
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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

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
    output_dir: str = "al_fcc_alignn_energy_example_output"
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

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0



CONFIG_PATH = Path(
    os.environ.get(
        "CATALYST_AL_FCC_CONFIG",
        Path(__file__).with_name("al_fcc_alignn_energy_config.json"),
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

    # JSON stores tuples as lists; convert back where needed.
    cfg.repeat = _tuple3(cfg.repeat, "repeat")

    # Let users request automatic device selection in JSON.
    if str(cfg.device).lower() in {"auto", "default"}:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

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


def make_dir(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
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
    energies = []

    for step in range(config.md_steps + 1):
        if step > 0:
            dyn.run(1)

        if step < config.equilibration_steps:
            continue

        if (step - config.equilibration_steps) % config.sample_every != 0:
            continue

        frame = atoms.copy()
        frame.calc = EMT()
        energy = float(frame.get_potential_energy())

        frames.append(frame)
        energies.append(energy)

    print(f"Generated {len(frames)} MD frames.")
    print(f"Energy range: {min(energies):.6f} to {max(energies):.6f} eV")

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
    energy_eV: float,
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

    # Store raw total energy for reporting.
    graph.energy_eV = torch.tensor([energy_eV], dtype=torch.float32)

    # y will later be overwritten with normalized target values for training.
    graph.y = torch.tensor([energy_eV], dtype=torch.float32)

    return graph


def generate_graph_dataset(config: ExampleConfig, output_dir: Path) -> List[Path]:
    """
    Generate MD frames, convert to graphs, and save graph .pt files.
    """
    graph_dir = reset_dir(output_dir / "graphs")
    frames = run_md_frames(config)

    graph_paths = []

    for idx, atoms in enumerate(frames):
        energy_eV = float(atoms.get_potential_energy())
        gid = f"al_fcc_md_{idx:05d}"

        print(f"Building graph {idx + 1:4d}/{len(frames):4d}: {gid}, E={energy_eV:.6f} eV")

        graph = atoms_to_alignn_graph(
            atoms=atoms,
            energy_eV=energy_eV,
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

    This is deliberately simple for this example. If desired, this can later be
    replaced by SODAS/UMAP sampling exactly like the synthetic workflow.
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


def compute_target_normalization(graphs: Sequence[Any]) -> Tuple[float, float]:
    values = np.asarray([float(g.energy_eV.view(-1)[0]) for g in graphs], dtype=float)
    mean = float(values.mean())
    std = float(values.std())

    if std < 1e-12:
        std = 1.0

    return mean, std


def apply_target_normalization(
    graphs: Sequence[Any],
    mean: float,
    std: float,
) -> None:
    for graph in graphs:
        energy = float(graph.energy_eV.view(-1)[0])
        graph.y = torch.tensor([(energy - mean) / std], dtype=torch.float32)


def make_loader(
    graphs: Sequence[Any],
    config: ExampleConfig,
    shuffle: bool,
) -> DataLoader:
    """
    Build a PyG DataLoader with follow_batch fields for atom/bond/angle tensors.

    x_atm_batch is used by AtomicEnergyReadout to sum per-atom energies into
    graph energies for each graph in a batch.
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


class AtomicEnergyReadout(nn.Module):
    """
    Graph-level total-energy decoder.

    It reads atom hidden features and predicts one scalar energy contribution per
    atom, then sums those contributions per graph.

    Output:
        [batch_size, 1] normalized total energy
    """

    def __init__(self, dim: int, act=nn.SiLU()):
        super().__init__()
        self.dim = dim
        self.atom_energy = nn.Sequential(
            # A slightly deeper readout helps for the toy MD dataset.
            nn.Linear(dim, dim),
            act,
            nn.Linear(dim, 1),
        )

    def forward(self, data):
        if hasattr(data, "h_atm"):
            h_atom = data.h_atm
        elif hasattr(data, "h_1"):
            h_atom = data.h_1
        else:
            raise AttributeError("AtomicEnergyReadout expected data.h_atm or data.h_1.")

        per_atom = self.atom_energy(h_atom).view(-1)

        if hasattr(data, "x_atm_batch"):
            batch = data.x_atm_batch
        elif hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(
                per_atom.size(0),
                dtype=torch.long,
                device=per_atom.device,
            )

        graph_energy = scatter(
            per_atom,
            index=batch,
            dim=0,
            reduce="add",
        )

        return graph_energy.view(-1, 1)


def build_alignn_energy_model(config: ExampleConfig) -> nn.Module:
    """
    Build an ALIGNN-style energy model through the public build_model(...) API.

    This uses the modular ALIGNN stack:

        atomic encoder + OrderProcessor(atom/bond/angle updates) + custom energy readout

    The decoder is passed explicitly because this is a graph-level total-energy
    task, not the default positive scalar per-order decoder.
    """
    return build_model(
        preset="alignn",
        processor_type="order",
        conv_type=config.conv_type,
        decoder=AtomicEnergyReadout(
            dim=config.hidden_dim,
            act=nn.SiLU(),
        ),
        num_species=1,
        cutoff=config.cutoff,
        dim=config.hidden_dim,
        num_convs=config.num_convs,
        out_dim=1,
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
    n_graphs = 0

    loss_fn = nn.MSELoss()

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(batch).view(-1)
        target = batch.y.view(-1).to(pred.device)

        loss = loss_fn(pred, target)
        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        batch_size = int(target.numel())
        total_loss += float(loss.item()) * batch_size
        n_graphs += batch_size

    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_mean: float,
    target_std: float,
) -> Dict[str, Any]:
    model.eval()

    pred_all = []
    true_all = []

    for batch in loader:
        batch = batch.to(device)

        pred_norm = model(batch).view(-1)
        true_norm = batch.y.view(-1).to(pred_norm.device)

        pred_e = pred_norm * target_std + target_mean
        true_e = true_norm * target_std + target_mean

        pred_all.append(pred_e.detach().cpu())
        true_all.append(true_e.detach().cpu())

    if not pred_all:
        return {
            "mae_eV": float("nan"),
            "rmse_eV": float("nan"),
            "pred_eV": np.asarray([]),
            "true_eV": np.asarray([]),
        }

    pred = torch.cat(pred_all).numpy()
    true = torch.cat(true_all).numpy()

    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

    return {
        "mae_eV": mae,
        "rmse_eV": rmse,
        "pred_eV": pred,
        "true_eV": true,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: ExampleConfig,
    output_dir: Path,
    target_mean: float,
    target_std: float,
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
        "valid_mae_eV": [],
        "valid_rmse_eV": [],
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
            target_mean=target_mean,
            target_std=target_std,
        )

        history["epoch"].append(epoch)
        history["train_loss_norm"].append(train_loss)
        history["valid_mae_eV"].append(valid_metrics["mae_eV"])
        history["valid_rmse_eV"].append(valid_metrics["rmse_eV"])

        if valid_metrics["mae_eV"] < best_valid:
            best_valid = valid_metrics["mae_eV"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "config": asdict(config),
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == config.num_epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"train loss={train_loss:.6e} | "
                f"valid MAE={valid_metrics['mae_eV']:.6f} eV | "
                f"valid RMSE={valid_metrics['rmse_eV']:.6f} eV"
            )

    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss_norm", "valid_mae_eV", "valid_rmse_eV"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss_norm"][i],
                history["valid_mae_eV"][i],
                history["valid_rmse_eV"][i],
            ])

    return history


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str,
) -> Tuple[nn.Module, float, float]:
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    state = checkpoint["model_state"]

    # Keep checkpoint robust to possible DDP/torch.compile wrappers.
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

    model.load_state_dict(cleaned, strict=True)
    model.to(device)

    return model, float(checkpoint["target_mean"]), float(checkpoint["target_std"])


# =============================================================================
# Plotting / output
# =============================================================================


def plot_training_history(history: Dict[str, List[float]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["valid_mae_eV"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (eV)")
    ax.set_title("Al FCC ALIGNN validation error")
    fig.tight_layout()
    fig.savefig(output_dir / "training_validation_mae.png", dpi=200)
    plt.close(fig)


def plot_test_parity(metrics: Dict[str, Any], output_dir: Path) -> None:
    true = metrics["true_eV"]
    pred = metrics["pred_eV"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(true, pred, linestyle="", marker="o", markeredgecolor="k")

    if len(true) > 0:
        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="-")

    ax.set_xlabel("EMT energy (eV)")
    ax.set_ylabel("ALIGNN-predicted energy (eV)")
    ax.set_title(
        f"Test parity: MAE={metrics['mae_eV']:.4f} eV, "
        f"RMSE={metrics['rmse_eV']:.4f} eV"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "test_energy_parity.png", dpi=200)
    plt.close(fig)


def write_predictions(metrics: Dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_energy_eV", "pred_energy_eV", "error_eV"])
        for true, pred in zip(metrics["true_eV"], metrics["pred_eV"]):
            writer.writerow([float(true), float(pred), float(pred - true)])


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

    # 3. Target normalization based on training set only.
    target_mean, target_std = compute_target_normalization(train_graphs)

    apply_target_normalization(train_graphs, target_mean, target_std)
    apply_target_normalization(valid_graphs, target_mean, target_std)
    apply_target_normalization(test_graphs, target_mean, target_std)

    print(f"Training target normalization: mean={target_mean:.6f} eV, std={target_std:.6f} eV")

    train_loader = make_loader(train_graphs, config=config, shuffle=True)
    valid_loader = make_loader(valid_graphs, config=config, shuffle=False)
    test_loader = make_loader(test_graphs, config=config, shuffle=False)

    # 4. Model building
    model = build_alignn_energy_model(config)
    print(model)

    # 5. Training
    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=config,
        output_dir=output_dir,
        target_mean=target_mean,
        target_std=target_std,
    )

    plot_training_history(history, output_dir)

    # 6. Reload best checkpoint and test.
    best_model = build_alignn_energy_model(config)
    best_model, target_mean, target_std = load_model_checkpoint(
        model=best_model,
        checkpoint_path=output_dir / "best_model.pt",
        device=config.device,
    )

    test_metrics = evaluate(
        model=best_model,
        loader=test_loader,
        device=config.device,
        target_mean=target_mean,
        target_std=target_std,
    )

    print(
        "Test metrics: "
        f"MAE={test_metrics['mae_eV']:.6f} eV, "
        f"RMSE={test_metrics['rmse_eV']:.6f} eV"
    )

    plot_test_parity(test_metrics, output_dir)
    write_predictions(test_metrics, output_dir)

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mae_eV": test_metrics["mae_eV"],
                "rmse_eV": test_metrics["rmse_eV"],
            },
            handle,
            indent=2,
        )

    print("Done.")
    print(f"Results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
