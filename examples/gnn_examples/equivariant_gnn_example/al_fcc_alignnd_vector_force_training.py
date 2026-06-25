#!/usr/bin/env python
"""
Example: Al FCC MD -> equivariant ALIGNN graphs -> learn per-atom vector targets.

Recommended location:
    examples/equivariant/al_fcc_alignnd_vector_force_training.py

This example intentionally keeps the core model language generic:

    output_type="vector"
    target_vector

For an atomistic demonstration, the target_vector is the EMT force vector on
each atom, but the model/decoder does not need to know that these are "forces".
It simply learns a node-level equivariant vector field.

Workflow
--------
1. Build an Al FCC supercell with ASE.
2. Run short EMT MD and sample frames.
3. For each frame, build an updated ALIGNN graph carrying equivariant fields:
       z, pos, edge_index, cell, pbc, shifts
   plus legacy ALIGNN fields:
       x_atm, x_bnd, x_ang, edge_index_G, edge_index_A
4. Attach:
       graph.target_vector = atoms.get_forces()
       graph.target_scalar = atoms.get_potential_energy()
5. Train:
       EquivariantAtomicEncoder
       EquivariantProcessor(equivariant_type="egnn")
       EquivariantDecoder(output_type="vector", output_level="node")
6. Save a checkpoint and diagnostic plots.

Important distinction
---------------------
This is NOT the legacy ALIGNN/order model branch. It only uses alignnd as the
graph builder. The model consumes the equivariant fields, not edge_index_A/x_ang.

The direct vector decoder predicts a vector field directly. This is useful for
generic vector targets. If you want conservative force-field behavior, use
output_type="scalar_gradient" instead and train the gradient of a scalar.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet


# =============================================================================
# Catalyst imports
# =============================================================================


def import_alignn_gen():
    """Import alignn_gen from common Catalyst layouts."""
    try:
        from catalyst.graph.alignnd import alignn_gen
        return alignn_gen
    except ImportError:
        pass

    try:
        from catalyst.src.graph.alignnd import alignn_gen
        return alignn_gen
    except ImportError as exc:
        raise ImportError(
            "Could not import alignn_gen. Tried:\n"
            "  catalyst.graph.alignnd\n"
            "  catalyst.src.graph.alignnd\n\n"
            "Make sure Catalyst is installed in editable mode, for example:\n"
            "  pip install -e .\n"
            "from the repository root."
        ) from exc


def import_equivariant_graph_helpers():
    """Import fallback equivariant field helpers from graph.py."""
    try:
        from catalyst.graph.graph import build_equivariant_atomic_fields, attach_equivariant_fields
        return build_equivariant_atomic_fields, attach_equivariant_fields
    except ImportError:
        pass

    try:
        from catalyst.src.graph.graph import build_equivariant_atomic_fields, attach_equivariant_fields
        return build_equivariant_atomic_fields, attach_equivariant_fields
    except ImportError:
        return None, None


def import_equivariant_modules():
    """Import the new equivariant modules."""
    try:
        from catalyst.ml.gnn.modules.encoders.equivariant_encoders import EquivariantAtomicEncoder
        from catalyst.ml.gnn.modules.processors.equivariant_processor import EquivariantProcessor
        from catalyst.ml.gnn.modules.decoders.equivariant_decoders import EquivariantDecoder
        return EquivariantAtomicEncoder, EquivariantProcessor, EquivariantDecoder
    except ImportError:
        pass

    try:
        from catalyst.src.catalyst.ml.gnn.modules.encoders.equivariant_encoders import EquivariantAtomicEncoder
        from catalyst.src.catalyst.ml.gnn.modules.processors.equivariant_processor import EquivariantProcessor
        from catalyst.src.catalyst.ml.gnn.modules.decoders.equivariant_decoders import EquivariantDecoder
        return EquivariantAtomicEncoder, EquivariantProcessor, EquivariantDecoder
    except ImportError as exc:
        raise ImportError(
            "Could not import the equivariant modules. Expected these files:\n"
            "  catalyst/ml/gnn/modules/encoders/equivariant_encoders.py\n"
            "  catalyst/ml/gnn/modules/processors/equivariant_processor.py\n"
            "  catalyst/ml/gnn/modules/decoders/equivariant_decoders.py\n\n"
            "Make sure the new files are installed in the package and reinstall with:\n"
            "  pip install -e ."
        ) from exc


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# MD data generation
# =============================================================================


def build_al_fcc(
    *,
    lattice_constant: float = 4.05,
    repeat: Tuple[int, int, int] = (3, 3, 3),
):
    """Build Al FCC supercell with ASE EMT."""
    atoms = bulk("Al", "fcc", a=lattice_constant, cubic=True)
    atoms = atoms.repeat(repeat)
    atoms.pbc = True
    atoms.calc = EMT()
    return atoms


def run_md_collect_frames(
    atoms,
    *,
    n_frames: int = 80,
    steps_per_frame: int = 5,
    temperature_K: float = 500.0,
    timestep_fs: float = 1.0,
    seed: int = 7,
):
    """Run NVE MD using EMT and collect frame copies with force/energy targets."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1.")

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = VelocityVerlet(atoms, timestep_fs * units.fs)

    frames = []
    rows = []

    def capture(frame_index: int, step: int):
        # Calculate on current atoms before copying.
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float32)
        temperature = float(atoms.get_temperature())

        frame = atoms.copy()
        frame.calc = EMT()

        frames.append(
            {
                "atoms": frame,
                "energy": energy,
                "forces": forces,
                "step": int(step),
                "time_fs": float(step * timestep_fs),
                "temperature_K": temperature,
            }
        )
        rows.append(
            {
                "frame": int(frame_index),
                "step": int(step),
                "time_fs": float(step * timestep_fs),
                "energy_eV": energy,
                "temperature_K": temperature,
                "force_rms_eV_per_A": float(np.sqrt(np.mean(forces ** 2))),
                "force_max_norm_eV_per_A": float(np.max(np.linalg.norm(forces, axis=1))),
            }
        )

    capture(0, 0)
    for frame_index in range(1, n_frames):
        dyn.run(steps_per_frame)
        capture(frame_index, frame_index * steps_per_frame)

    return frames, rows


# =============================================================================
# Graph construction
# =============================================================================


def finalize_graph_metadata(graph):
    """Make key PyG/equivariant metadata explicit."""
    z = getattr(graph, "z", None)
    pos = getattr(graph, "pos", None)
    x_atm = getattr(graph, "x_atm", None)

    if z is not None:
        graph.num_nodes = int(z.size(0))
    elif pos is not None:
        graph.num_nodes = int(pos.size(0))
    elif x_atm is not None:
        graph.num_nodes = int(x_atm.size(0))

    if getattr(graph, "edge_index", None) is None and getattr(graph, "edge_index_G", None) is not None:
        graph.edge_index = graph.edge_index_G

    edge_index = getattr(graph, "edge_index", None)
    if getattr(graph, "shifts", None) is None and edge_index is not None:
        graph.shifts = torch.zeros((edge_index.size(1), 3), dtype=torch.long, device=edge_index.device)

    return graph


def ensure_equivariant_fields(graph, atoms):
    """Attach equivariant fields if alignn_gen did not do it directly."""
    required = ("z", "pos", "edge_index", "cell", "pbc", "shifts")
    if all(getattr(graph, key, None) is not None for key in required):
        return finalize_graph_metadata(graph)

    build_equivariant_atomic_fields, attach_equivariant_fields = import_equivariant_graph_helpers()
    if build_equivariant_atomic_fields is None or attach_equivariant_fields is None:
        missing = [key for key in required if getattr(graph, key, None) is None]
        raise RuntimeError(
            "Graph is missing equivariant fields and fallback helpers could not be imported. "
            f"Missing fields: {missing}"
        )

    if getattr(graph, "edge_index_G", None) is None:
        raise RuntimeError("Cannot attach fallback equivariant fields because graph.edge_index_G is missing.")

    fields = build_equivariant_atomic_fields(
        atoms,
        graph.edge_index_G.detach().cpu().numpy(),
        dtype=np.float32,
        include_edge_geometry=True,
    )
    graph = attach_equivariant_fields(graph, **fields)
    return finalize_graph_metadata(graph)


def validate_graph(graph, *, name: str = "graph"):
    """Validate fields required by the equivariant model."""
    required = ("z", "pos", "edge_index", "cell", "pbc", "shifts", "num_nodes", "target_vector")
    missing = [key for key in required if getattr(graph, key, None) is None]
    if missing:
        raise ValueError(f"{name} missing required fields: {missing}")

    n_nodes = int(graph.num_nodes)
    n_edges = int(graph.edge_index.size(1))

    if graph.z.size(0) != n_nodes:
        raise ValueError(f"{name}: z length does not match num_nodes.")

    if graph.pos.shape != (n_nodes, 3):
        raise ValueError(f"{name}: pos must have shape [N, 3], got {tuple(graph.pos.shape)}.")

    if graph.target_vector.shape != (n_nodes, 3):
        raise ValueError(
            f"{name}: target_vector must have shape [N, 3], got {tuple(graph.target_vector.shape)}."
        )

    if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
        raise ValueError(f"{name}: edge_index must have shape [2, E].")

    if n_edges > 0:
        if int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= n_nodes:
            raise ValueError(f"{name}: edge_index references nodes outside [0, N).")

    if graph.shifts.shape != (n_edges, 3):
        raise ValueError(f"{name}: shifts must have shape [E, 3], got {tuple(graph.shifts.shape)}.")

    return True


def build_alignnd_graph_dataset(
    frames: Sequence[dict],
    *,
    cutoff: float = 5.0,
    k: int = -1,
    include_angs: bool = True,
    include_dihedrals: bool = False,
    retry_verbose: bool = False,
):
    """Build updated ALIGNN graphs with node-level vector targets."""
    alignn_gen = import_alignn_gen()

    graphs = []
    for frame_index, frame in enumerate(frames):
        atoms = frame["atoms"]

        graph = alignn_gen(
            {
                "type": "alignnd",
                "raw_data": atoms,
                "neighbor_params": [float(cutoff), int(k)],
                "include_angs": bool(include_angs),
                "is_dihedral": bool(include_dihedrals),
                "include_equivariant_fields": True,
                "include_edge_geometry": True,
                "retry_verbose": bool(retry_verbose),
                "require_bonds": True,
                "require_angles": bool(include_angs),
                "require_dihedrals": False,
            }
        )

        graph = ensure_equivariant_fields(graph, atoms)

        # Generic target names.  In this example, target_vector = EMT forces.
        graph.target_vector = torch.as_tensor(frame["forces"], dtype=torch.float)
        graph.target_scalar = torch.as_tensor([frame["energy"]], dtype=torch.float)
        graph.frame_index = torch.as_tensor([frame_index], dtype=torch.long)
        graph.md_step = torch.as_tensor([frame["step"]], dtype=torch.long)
        graph.md_time_fs = torch.as_tensor([frame["time_fs"]], dtype=torch.float)
        graph.temperature_K = torch.as_tensor([frame["temperature_K"]], dtype=torch.float)

        validate_graph(graph, name=f"graph[{frame_index}]")
        graphs.append(graph)

    return graphs


# =============================================================================
# Model
# =============================================================================


class EquivariantVectorModel(nn.Module):
    """Encoder -> equivariant processor -> generic vector decoder."""

    def __init__(
        self,
        *,
        dim: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        rbf_dim: int = 32,
        dropout: float = 0.0,
        use_atom_features: bool = True,
        vector_out_dim: int = 1,
    ):
        super().__init__()

        EquivariantAtomicEncoder, EquivariantProcessor, EquivariantDecoder = import_equivariant_modules()

        self.encoder = EquivariantAtomicEncoder(
            dim=dim,
            max_atomic_number=118,
            use_atom_features=use_atom_features,
            initialize_vector=True,
            dropout=dropout,
            norm=True,
        )

        self.processor = EquivariantProcessor(
            dim=dim,
            num_convs=num_layers,
            equivariant_type="egnn",
            cutoff=cutoff,
            rbf_dim=rbf_dim,
            dropout=dropout,
            residual=True,
            norm=True,
            update_pos=False,
            use_precomputed_geometry=False,
        )

        self.decoder = EquivariantDecoder(
            dim=dim,
            output_type="vector",
            output_level="node",
            out_dim=vector_out_dim,
            reduce="sum",
            squeeze_vector_channels=True,
            return_dict=True,
            vector_key="vector",
        )

    def forward(self, data):
        data = self.encoder(data)
        data = self.processor(data)
        out = self.decoder(data)
        return out


# =============================================================================
# Training / evaluation
# =============================================================================


@dataclass
class Metrics:
    loss: float
    mae: float
    rmse: float
    component_mae: float
    cosine_mean: float


def vector_metrics(pred: torch.Tensor, target: torch.Tensor) -> Metrics:
    """Compute vector-field metrics."""
    diff = pred - target
    loss = F.mse_loss(pred, target)
    mae = torch.mean(torch.linalg.norm(diff, dim=-1))
    rmse = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1)))
    component_mae = torch.mean(torch.abs(diff))

    pred_norm = torch.linalg.norm(pred, dim=-1).clamp_min(1.0e-12)
    target_norm = torch.linalg.norm(target, dim=-1).clamp_min(1.0e-12)
    cosine = torch.sum(pred * target, dim=-1) / (pred_norm * target_norm)
    cosine_mean = torch.mean(cosine)

    return Metrics(
        loss=float(loss.detach().cpu()),
        mae=float(mae.detach().cpu()),
        rmse=float(rmse.detach().cpu()),
        component_mae=float(component_mae.detach().cpu()),
        cosine_mean=float(cosine_mean.detach().cpu()),
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        target = batch.target_vector.to(device)

        optimizer.zero_grad(set_to_none=True)

        out = model(batch)
        pred = out["vector"]

        loss = F.mse_loss(pred, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        n_nodes = int(target.size(0))
        total_loss += float(loss.detach().cpu()) * n_nodes
        total_nodes += n_nodes

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    preds = []
    targets = []

    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        target = batch.target_vector.to(device)

        out = model(batch)
        pred = out["vector"]

        loss = F.mse_loss(pred, target)

        n_nodes = int(target.size(0))
        total_loss += float(loss.detach().cpu()) * n_nodes
        total_nodes += n_nodes

        preds.append(pred.detach().cpu())
        targets.append(target.detach().cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)

    metrics = vector_metrics(pred_all, target_all)
    metrics.loss = total_loss / max(total_nodes, 1)
    return metrics, pred_all, target_all


def split_dataset(graphs, *, train_fraction: float = 0.8, seed: int = 7):
    """Random train/test split."""
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1.")

    indices = list(range(len(graphs)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = max(1, int(round(train_fraction * len(indices))))
    n_train = min(n_train, len(indices) - 1) if len(indices) > 1 else len(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_graphs = [graphs[i] for i in train_indices]
    test_graphs = [graphs[i] for i in test_indices]

    return train_graphs, test_graphs, train_indices, test_indices


# =============================================================================
# Plotting / saving
# =============================================================================


def write_csv_rows(rows: Sequence[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_loss(history: Sequence[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    test = [row["test_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(epochs, train, label="Train")
    ax.plot(epochs, test, label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Vector MSE")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_force_parity(pred: torch.Tensor, target: torch.Tensor, path: Path, *, max_points: int = 5000):
    """Plot predicted vs target vector components."""
    path.parent.mkdir(parents=True, exist_ok=True)

    pred_np = pred.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)

    if pred_np.size > max_points:
        rng = np.random.default_rng(7)
        idx = rng.choice(pred_np.size, size=max_points, replace=False)
        pred_np = pred_np[idx]
        target_np = target_np[idx]

    lo = float(min(pred_np.min(), target_np.min()))
    hi = float(max(pred_np.max(), target_np.max()))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(target_np, pred_np, s=6, alpha=0.35)
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Target vector component")
    ax.set_ylabel("Predicted vector component")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def save_example_outputs(
    *,
    output_dir: Path,
    model: nn.Module,
    args,
    md_rows: Sequence[dict],
    history: Sequence[dict],
    test_pred: torch.Tensor,
    test_target: torch.Tensor,
    graphs: Optional[Sequence] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    graph_dir = output_dir / "graphs"

    write_csv_rows(md_rows, output_dir / "md_frames.csv")
    write_csv_rows(history, output_dir / "training_history.csv")
    plot_loss(history, plot_dir / "vector_loss.png")
    plot_force_parity(test_pred, test_target, plot_dir / "vector_component_parity.png")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "final_test_pred": test_pred,
        "final_test_target": test_target,
    }
    torch.save(checkpoint, output_dir / "equivariant_vector_model.pt")

    if graphs is not None and args.save_graphs:
        graph_dir.mkdir(parents=True, exist_ok=True)
        for i, graph in enumerate(graphs):
            torch.save(graph, graph_dir / f"graph_{i:04d}.pt")


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an equivariant GNN on Al FCC EMT per-atom vector targets using alignnd graphs."
    )

    parser.add_argument("--output-dir", type=Path, default=Path("al_fcc_alignnd_vector_training"))

    parser.add_argument("--repeat", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument("--lattice-constant", type=float, default=4.05)

    parser.add_argument("--n-frames", type=int, default=80)
    parser.add_argument("--steps-per-frame", type=int, default=5)
    parser.add_argument("--temperature-K", type=float, default=500.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)

    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=-1)
    parser.add_argument("--include-angs", action="store_true", default=True)
    parser.add_argument("--no-include-angs", dest="include_angs", action="store_false")
    parser.add_argument("--include-dihedrals", action="store_true")
    parser.add_argument("--retry-verbose", action="store_true")

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--rbf-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-atom-features", dest="use_atom_features", action="store_false")
    parser.set_defaults(use_atom_features=True)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--train-fraction", type=float, default=0.8)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-graphs", action="store_true")

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)

    print("Al FCC equivariant vector training example")
    print("------------------------------------------")
    print(f"device: {device}")
    print(f"output_dir: {args.output_dir}")

    atoms = build_al_fcc(
        lattice_constant=args.lattice_constant,
        repeat=tuple(args.repeat),
    )
    print(f"Built Al FCC supercell with {len(atoms)} atoms.")

    print(f"Running EMT MD: n_frames={args.n_frames}, steps_per_frame={args.steps_per_frame}")
    frames, md_rows = run_md_collect_frames(
        atoms,
        n_frames=args.n_frames,
        steps_per_frame=args.steps_per_frame,
        temperature_K=args.temperature_K,
        timestep_fs=args.timestep_fs,
        seed=args.seed,
    )

    print("Building equivariant alignnd graphs with target_vector...")
    graphs = build_alignnd_graph_dataset(
        frames,
        cutoff=args.cutoff,
        k=args.k,
        include_angs=args.include_angs,
        include_dihedrals=args.include_dihedrals,
        retry_verbose=args.retry_verbose,
    )

    print(f"Built {len(graphs)} graphs.")
    print(
        f"First graph: nodes={graphs[0].num_nodes}, "
        f"edges={graphs[0].edge_index.size(1)}, "
        f"target_vector={tuple(graphs[0].target_vector.shape)}"
    )

    train_graphs, test_graphs, train_indices, test_indices = split_dataset(
        graphs,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    print(f"Train graphs: {len(train_graphs)} | Test graphs: {len(test_graphs)}")

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = EquivariantVectorModel(
        dim=args.dim,
        num_layers=args.num_layers,
        cutoff=args.cutoff,
        rbf_dim=args.rbf_dim,
        dropout=args.dropout,
        use_atom_features=args.use_atom_features,
        vector_out_dim=1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_test = math.inf
    best_state = None

    print("\nTraining")
    print("--------")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_metrics, _, _ = evaluate(model, test_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_metrics.loss,
            "test_mae_vector_norm": test_metrics.mae,
            "test_rmse_vector_norm": test_metrics.rmse,
            "test_component_mae": test_metrics.component_mae,
            "test_cosine_mean": test_metrics.cosine_mean,
        }
        history.append(row)

        if test_metrics.loss < best_test:
            best_test = test_metrics.loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} "
                f"train_mse={train_loss:.6e} "
                f"test_mse={test_metrics.loss:.6e} "
                f"test_vec_mae={test_metrics.mae:.6e} "
                f"test_comp_mae={test_metrics.component_mae:.6e} "
                f"cos={test_metrics.cosine_mean:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics, test_pred, test_target = evaluate(model, test_loader, device)

    print("\nFinal test metrics")
    print("------------------")
    print(f"vector MSE:          {final_metrics.loss:.6e}")
    print(f"vector norm MAE:     {final_metrics.mae:.6e}")
    print(f"vector norm RMSE:    {final_metrics.rmse:.6e}")
    print(f"component MAE:       {final_metrics.component_mae:.6e}")
    print(f"mean cosine:         {final_metrics.cosine_mean:.6f}")

    save_example_outputs(
        output_dir=args.output_dir,
        model=model,
        args=args,
        md_rows=md_rows,
        history=history,
        test_pred=test_pred,
        test_target=test_target,
        graphs=graphs,
    )

    # Save last frame for visual inspection.
    write(args.output_dir / "last_md_frame.xyz", frames[-1]["atoms"])

    print("\nWrote outputs:")
    print(f"  {args.output_dir / 'equivariant_vector_model.pt'}")
    print(f"  {args.output_dir / 'training_history.csv'}")
    print(f"  {args.output_dir / 'md_frames.csv'}")
    print(f"  {args.output_dir / 'plots' / 'vector_loss.png'}")
    print(f"  {args.output_dir / 'plots' / 'vector_component_parity.png'}")


if __name__ == "__main__":
    main()
