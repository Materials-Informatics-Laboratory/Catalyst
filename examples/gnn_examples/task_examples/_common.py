"""Shared utilities for the compact full-backend GNNTask examples.

This module is imported by the task examples and is intentionally not an
example entry point itself.  The public examples use real Catalyst graph
objects, ``build_task_model()``, the high-level ``run_training()`` backend,
checkpoint reload, and ``run_inference()``.  It also provides standardized
headless plotting so every task example writes figures when it is run.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import KDTree

from catalyst.data.utils import load_dictionary, save_dictionary
from catalyst.graph.generic_build import generic_graph_gen
from catalyst.ml.gnn import GNN
from catalyst.ml.inference import run_inference
from catalyst.ml.training import run_training
from catalyst.observer import Catalyst


def load_example_config(script_path: Path) -> dict:
    """Load the JSON configuration stored beside a task example script."""
    config_path = Path(script_path).with_name("config.json")
    if not config_path.is_file():
        raise FileNotFoundError(f"Task example configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a JSON object in {config_path}.")
    return config


def reset_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def scatter_mean(values: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Small torch-only graph mean used by custom pedagogical readouts."""
    out_shape = (int(n_graphs),) + tuple(values.shape[1:])
    sums = values.new_zeros(out_shape)
    sums.index_add_(0, batch, values)

    counts = values.new_zeros((int(n_graphs),))
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=values.dtype))
    view = (int(n_graphs),) + (1,) * (values.dim() - 1)
    return sums / counts.clamp_min(1.0).view(view)


def infer_num_graphs(data) -> int:
    explicit = getattr(data, "num_graphs", None)
    if explicit is not None:
        return int(explicit)
    ptr = getattr(data, "ptr", None)
    if torch.is_tensor(ptr):
        return int(ptr.numel() - 1)
    batch = getattr(data, "batch", None)
    if torch.is_tensor(batch) and batch.numel() > 0:
        return int(batch.max().item()) + 1
    return 1


def make_generic_graph(
    *,
    seed: int,
    n_nodes: int,
    n_types: int = 3,
    k: int = 4,
):
    """Create one deterministic 3-body generic Catalyst graph."""
    rng = np.random.default_rng(int(seed))
    n_nodes = max(int(n_nodes), 3)
    n_types = max(int(n_types), 2)
    k = max(1, min(int(k), n_nodes - 1))

    positions = rng.normal(0.0, 0.75, size=(n_nodes, 3)).astype(np.float32)
    type_ids = rng.integers(0, n_types, size=n_nodes)
    node_features = np.eye(n_types, dtype=np.float32)[type_ids]

    tree = KDTree(positions, metric="euclidean", leaf_size=2)
    distances, indices = tree.query(positions, k=k + 1)

    graph = generic_graph_gen(
        {
            "type": "generic_pairwise",
            "raw_data": positions,
            "params": {
                "dist": distances,
                "ind": indices,
                "g_nodes": node_features,
                "use_raw_data_as_pos": True,
            },
            "line_graph": True,
            "include_angs": True,
            "include_self_edges": False,
            "strict": True,
            "include_equivariant_fields": True,
        }
    )
    graph.gid = f"generic_task_{int(seed):04d}"
    graph.synthetic_type_ids = torch.as_tensor(type_ids, dtype=torch.long)
    return graph


def make_dimer_graph(*, vector: np.ndarray, gid: str):
    """Create an Al--Cu nonperiodic dimer with equivariant Catalyst fields."""
    from ase import Atoms
    from catalyst.graph.alignnd import alignn_gen

    vector = np.asarray(vector, dtype=float).reshape(3)
    atoms = Atoms(
        symbols=["Al", "Cu"],
        positions=np.vstack((-0.5 * vector, 0.5 * vector)),
        cell=np.eye(3) * 12.0,
        pbc=False,
    )

    graph = alignn_gen(
        {
            "type": "alignnd",
            "raw_data": atoms,
            "element_list": ["Al", "Cu"],
            "neighbor_params": [5.0, -1],
            "include_angs": False,
            "is_dihedral": False,
            "store_raw_data": False,
            "use_pt": False,
            "include_equivariant_fields": True,
            "include_edge_geometry": True,
            "auto_retry_graph": True,
            "max_graph_attempts": 3,
            "require_bonds": True,
            "require_angles": False,
            "require_dihedrals": False,
            "retry_verbose": False,
        }
    )
    graph.gid = str(gid)
    return graph


def write_dataset(
    root: Path,
    graphs: Sequence,
    *,
    n_train: int,
    n_validation: int,
):
    """Serialize graphs plus the sample dictionaries expected by Catalyst."""
    root = reset_dir(root)
    data_dir = root / "graphs"
    samples_dir = root / "samples"
    results_dir = root / "results"
    figures_dir = root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    gids = []
    for graph in graphs:
        gid = str(graph.gid)
        gids.append(gid)
        torch.save(graph, data_dir / f"{gid}.pt")

    n_train = int(n_train)
    n_validation = int(n_validation)
    if n_train < 1 or n_validation < 1 or n_train + n_validation >= len(gids):
        raise ValueError("Need nonempty training, validation, and test splits.")

    train = gids[:n_train]
    validation = gids[n_train : n_train + n_validation]
    test = gids[n_train + n_validation :]

    save_dictionary(
        samples_dir / "train_valid_split.npy",
        {"training": train, "validation": validation},
    )
    save_dictionary(
        samples_dir / "test_data.npy",
        {"validation": test, "gids": test},
    )

    return data_dir, samples_dir, results_dir, figures_dir, train, validation, test


def newest_checkpoint(model_dir: Path) -> Path:
    checkpoints = sorted(
        Path(model_dir).glob("checkpoint_epoch_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not checkpoints:
        raise RuntimeError(f"No Catalyst checkpoint was written to {model_dir}.")
    return checkpoints[-1]


def make_backend_parameters(
    *,
    root: Path,
    data_dir: Path,
    samples_dir: Path,
    results_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
):
    return {
        "device_dict": {
            "device": "cpu",
            "run_ddp": False,
            "use_amp": False,
            "pin_memory": False,
        },
        "io_dict": {
            "main_path": str(root),
            "data_dir": str(data_dir),
            "samples_dir": str(samples_dir),
            "results_dir": str(results_dir),
            "graph_read_format": 0,
            "write_indv_pred": False,
            "remove_old_model": False,
            "training_info_nwrite_steps": 1,
        },
        "loader_dict": {
            "shuffle_loader": True,
            "batch_size": [int(batch_size), 1],
            "num_workers": 0,
            "batch_mode": "graphs",
        },
        "model_dict": {
            "num_epochs": int(epochs),
            "train_delta": 0.0,
            "train_tolerance": 0.0,
            "max_deltas": 0,
            "worsen_tolerance": 5.0,
            "strict_loss_policy": False,
            "validation_interval": 1,
            "compile_model": False,
            "loss_params": {"function": "MSELoss"},
            "optimizer_params": {
                "optimizer": "Adam",
                "implementation": "default",
                "params_group": {"lr": float(learning_rate)},
            },
        },
    }


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _normalize_gid(gid_value) -> str:
    if isinstance(gid_value, (list, tuple)):
        if len(gid_value) != 1:
            raise ValueError(f"Expected one gid per batch for plotting, received: {gid_value}")
        return _normalize_gid(gid_value[0])
    if isinstance(gid_value, np.ndarray):
        if gid_value.size != 1:
            raise ValueError(f"Expected one gid per batch for plotting, received array shape {gid_value.shape}")
        return _normalize_gid(gid_value.reshape(-1)[0])
    return str(gid_value)


def _flatten_inference_pairs(inference: dict):
    preds = inference.get("pred", [])
    gids = inference.get("gids", [])
    if len(preds) != len(gids):
        raise RuntimeError("Inference prediction/gid lengths do not match.")

    pairs = []
    for gid_value, pred_value in zip(gids, preds):
        gid = _normalize_gid(gid_value)
        pairs.append((gid, _to_numpy(pred_value)))
    return pairs


def _load_target_for_gid(data_dir: Path, gid: str, target_key: str) -> np.ndarray:
    graph = safe_torch_load(Path(data_dir) / f"{gid}.pt")
    if not hasattr(graph, target_key):
        raise AttributeError(f"Graph '{gid}' does not contain target attribute '{target_key}'.")
    return _to_numpy(getattr(graph, target_key))


def _compute_scalar_metrics(target: np.ndarray, pred: np.ndarray) -> dict:
    target = np.asarray(target, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    mse = float(np.mean((pred - target) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred - target)))
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1.0e-16 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
    }


def _plot_training_history(history: dict, figures_dir: Path) -> None:
    train_loss = [float(value) for value in history.get("training_loss", [])]
    valid_loss = [float(value) for value in history.get("validation_loss", [])]
    epochs_train = np.arange(1, len(train_loss) + 1)
    epochs_valid = np.arange(1, len(valid_loss) + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if train_loss:
        ax.plot(epochs_train, train_loss, marker="o", label="Training loss")
    if valid_loss:
        ax.plot(epochs_valid, valid_loss, marker="s", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training history")
    ax.grid(True, alpha=0.3)
    if train_loss or valid_loss:
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "training_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _parity_limits(target: np.ndarray, pred: np.ndarray):
    lo = float(min(np.min(target), np.min(pred)))
    hi = float(max(np.max(target), np.max(pred)))
    if math.isclose(lo, hi):
        pad = 1.0
    else:
        pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _write_graph_scalar_outputs(root: Path, figures_dir: Path, pairs: list[dict]) -> dict:
    csv_path = root / "predictions.csv"
    targets = []
    preds = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gid", "target", "prediction"])
        for entry in pairs:
            target = float(np.asarray(entry["target"]).reshape(-1)[0])
            pred = float(np.asarray(entry["pred"]).reshape(-1)[0])
            targets.append(target)
            preds.append(pred)
            writer.writerow([entry["gid"], target, pred])

    target_arr = np.asarray(targets, dtype=float)
    pred_arr = np.asarray(preds, dtype=float)
    metrics = _compute_scalar_metrics(target_arr, pred_arr)

    lo, hi = _parity_limits(target_arr, pred_arr)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(target_arr, pred_arr, alpha=0.9)
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title("Graph scalar parity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "graph_scalar_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return metrics


def _write_graph_multiscalar_outputs(root: Path, figures_dir: Path, pairs: list[dict], target_names: Sequence[str] | None) -> dict:
    sample_target = np.asarray(pairs[0]["target"]).reshape(-1)
    n_targets = int(sample_target.size)
    target_names = list(target_names or [f"target_{i}" for i in range(n_targets)])
    if len(target_names) != n_targets:
        raise ValueError("target_names length does not match graph_multiscalar target width.")

    targets = []
    preds = []
    csv_path = root / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gid"] + [f"target_{name}" for name in target_names] + [f"prediction_{name}" for name in target_names])
        for entry in pairs:
            target = np.asarray(entry["target"]).reshape(-1)
            pred = np.asarray(entry["pred"]).reshape(-1)
            targets.append(target)
            preds.append(pred)
            writer.writerow([entry["gid"], *target.tolist(), *pred.tolist()])

    target_arr = np.vstack(targets)
    pred_arr = np.vstack(preds)
    overall = _compute_scalar_metrics(target_arr.reshape(-1), pred_arr.reshape(-1))
    per_target = {
        name: _compute_scalar_metrics(target_arr[:, i], pred_arr[:, i])
        for i, name in enumerate(target_names)
    }

    ncols = min(3, n_targets)
    nrows = int(math.ceil(n_targets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.5 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for i, name in enumerate(target_names):
        ax = axes[i]
        lo, hi = _parity_limits(target_arr[:, i], pred_arr[:, i])
        ax.scatter(target_arr[:, i], pred_arr[:, i], alpha=0.9)
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    for ax in axes[n_targets:]:
        ax.axis("off")
    fig.suptitle("Graph multiscalar parity", y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / "graph_multiscalar_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"overall": overall, "per_target": per_target}


def _write_node_scalar_outputs(root: Path, figures_dir: Path, pairs: list[dict]) -> dict:
    csv_path = root / "predictions.csv"
    targets = []
    preds = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gid", "node_index", "target", "prediction"])
        for entry in pairs:
            target = np.asarray(entry["target"]).reshape(-1)
            pred = np.asarray(entry["pred"]).reshape(-1)
            if target.shape != pred.shape:
                raise RuntimeError(f"Node-scalar prediction/target shape mismatch for {entry['gid']}: {pred.shape} vs {target.shape}")
            for node_index, (t_val, p_val) in enumerate(zip(target, pred)):
                targets.append(float(t_val))
                preds.append(float(p_val))
                writer.writerow([entry["gid"], node_index, float(t_val), float(p_val)])

    target_arr = np.asarray(targets, dtype=float)
    pred_arr = np.asarray(preds, dtype=float)
    metrics = _compute_scalar_metrics(target_arr, pred_arr)

    lo, hi = _parity_limits(target_arr, pred_arr)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(target_arr, pred_arr, alpha=0.75, s=15)
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Target node scalar")
    ax.set_ylabel("Predicted node scalar")
    ax.set_title("Node scalar parity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "node_scalar_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return metrics


def _component_metrics(target_arr: np.ndarray, pred_arr: np.ndarray, labels: Sequence[str]) -> dict:
    return {
        label: _compute_scalar_metrics(target_arr[:, i], pred_arr[:, i])
        for i, label in enumerate(labels)
    }


def _write_graph_vector_outputs(root: Path, figures_dir: Path, pairs: list[dict]) -> dict:
    components = ["x", "y", "z"]
    targets = []
    preds = []
    csv_path = root / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gid", "target_x", "target_y", "target_z", "pred_x", "pred_y", "pred_z", "target_norm", "pred_norm"])
        for entry in pairs:
            target = np.asarray(entry["target"]).reshape(-1)
            pred = np.asarray(entry["pred"]).reshape(-1)
            if target.size != 3 or pred.size != 3:
                raise RuntimeError(f"Graph-vector prediction/target for {entry['gid']} must have three components.")
            targets.append(target)
            preds.append(pred)
            writer.writerow([
                entry["gid"],
                *target.tolist(),
                *pred.tolist(),
                float(np.linalg.norm(target)),
                float(np.linalg.norm(pred)),
            ])

    target_arr = np.vstack(targets)
    pred_arr = np.vstack(preds)
    overall = _compute_scalar_metrics(target_arr.reshape(-1), pred_arr.reshape(-1))
    per_component = _component_metrics(target_arr, pred_arr, components)
    target_norm = np.linalg.norm(target_arr, axis=1)
    pred_norm = np.linalg.norm(pred_arr, axis=1)
    norm_metrics = _compute_scalar_metrics(target_norm, pred_norm)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, label in enumerate(components):
        lo, hi = _parity_limits(target_arr[:, i], pred_arr[:, i])
        axes[i].scatter(target_arr[:, i], pred_arr[:, i], alpha=0.85)
        axes[i].plot([lo, hi], [lo, hi], linestyle="--")
        axes[i].set_xlim(lo, hi)
        axes[i].set_ylim(lo, hi)
        axes[i].set_xlabel(f"Target {label}")
        axes[i].set_ylabel(f"Prediction {label}")
        axes[i].set_title(f"{label}-component")
        axes[i].grid(True, alpha=0.3)
    fig.suptitle("Graph vector component parity", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "graph_vector_component_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    lo, hi = _parity_limits(target_norm, pred_norm)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(target_norm, pred_norm, alpha=0.85)
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Target vector norm")
    ax.set_ylabel("Predicted vector norm")
    ax.set_title("Graph vector norm parity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "graph_vector_norm_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"overall": overall, "per_component": per_component, "norm": norm_metrics}


def _write_scalar_gradient_outputs(root: Path, figures_dir: Path, pairs: list[dict]) -> dict:
    components = ["fx", "fy", "fz"]
    target_rows = []
    pred_rows = []
    csv_path = root / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gid", "atom_index", "target_fx", "target_fy", "target_fz", "pred_fx", "pred_fy", "pred_fz", "target_norm", "pred_norm"])
        for entry in pairs:
            target = np.asarray(entry["target"]).reshape(-1, 3)
            pred = np.asarray(entry["pred"]).reshape(-1, 3)
            if target.shape != pred.shape:
                raise RuntimeError(f"Scalar-gradient prediction/target shape mismatch for {entry['gid']}: {pred.shape} vs {target.shape}")
            for atom_index, (target_vec, pred_vec) in enumerate(zip(target, pred)):
                target_rows.append(target_vec)
                pred_rows.append(pred_vec)
                writer.writerow([
                    entry["gid"],
                    atom_index,
                    *target_vec.tolist(),
                    *pred_vec.tolist(),
                    float(np.linalg.norm(target_vec)),
                    float(np.linalg.norm(pred_vec)),
                ])

    target_arr = np.asarray(target_rows, dtype=float)
    pred_arr = np.asarray(pred_rows, dtype=float)
    overall = _compute_scalar_metrics(target_arr.reshape(-1), pred_arr.reshape(-1))
    per_component = _component_metrics(target_arr, pred_arr, components)
    target_norm = np.linalg.norm(target_arr, axis=1)
    pred_norm = np.linalg.norm(pred_arr, axis=1)
    norm_metrics = _compute_scalar_metrics(target_norm, pred_norm)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, label in enumerate(components):
        lo, hi = _parity_limits(target_arr[:, i], pred_arr[:, i])
        axes[i].scatter(target_arr[:, i], pred_arr[:, i], alpha=0.8, s=18)
        axes[i].plot([lo, hi], [lo, hi], linestyle="--")
        axes[i].set_xlim(lo, hi)
        axes[i].set_ylim(lo, hi)
        axes[i].set_xlabel(f"Target {label}")
        axes[i].set_ylabel(f"Prediction {label}")
        axes[i].set_title(f"{label} parity")
        axes[i].grid(True, alpha=0.3)
    fig.suptitle("Scalar-gradient component parity", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "scalar_gradient_component_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    lo, hi = _parity_limits(target_norm, pred_norm)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(target_norm, pred_norm, alpha=0.8, s=18)
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Target force norm")
    ax.set_ylabel("Predicted force norm")
    ax.set_title("Scalar-gradient norm parity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "scalar_gradient_norm_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"overall": overall, "per_component": per_component, "norm": norm_metrics}


def _write_metrics_json(root: Path, metrics: dict) -> None:
    with (root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def _write_outputs_for_task(
    *,
    plot_kind: str,
    output_dir: Path,
    data_dir: Path,
    figures_dir: Path,
    history: dict,
    inference: dict,
    target_key: str,
    target_names: Sequence[str] | None = None,
) -> dict:
    pairs = []
    for gid, pred in _flatten_inference_pairs(inference):
        target = _load_target_for_gid(data_dir, gid, target_key)
        pairs.append({"gid": gid, "pred": pred, "target": target})

    _plot_training_history(history, figures_dir)

    if plot_kind == "graph_scalar":
        metrics = _write_graph_scalar_outputs(output_dir, figures_dir, pairs)
    elif plot_kind == "graph_multiscalar":
        metrics = _write_graph_multiscalar_outputs(output_dir, figures_dir, pairs, target_names)
    elif plot_kind == "node_scalar":
        metrics = _write_node_scalar_outputs(output_dir, figures_dir, pairs)
    elif plot_kind == "graph_vector":
        metrics = _write_graph_vector_outputs(output_dir, figures_dir, pairs)
    elif plot_kind == "scalar_gradient":
        metrics = _write_scalar_gradient_outputs(output_dir, figures_dir, pairs)
    else:
        raise ValueError(f"Unknown plot_kind: {plot_kind}")

    _write_metrics_json(output_dir, metrics)
    return metrics


def run_backend_task_example(
    *,
    task,
    model,
    graphs: Sequence,
    output_dir: Path,
    n_train: int,
    n_validation: int,
    epochs: int = 20,
    learning_rate: float = 3.0e-3,
    batch_size: int = 8,
    plot_kind: str,
    target_key: str,
    target_names: Sequence[str] | None = None,
):
    """Run the complete public Catalyst training/checkpoint/inference pathway."""
    data_dir, samples_dir, results_dir, figures_dir, train, validation, test = write_dataset(
        Path(output_dir),
        graphs,
        n_train=n_train,
        n_validation=n_validation,
    )

    cat = Catalyst(
        parameters=make_backend_parameters(
            root=Path(output_dir),
            data_dir=data_dir,
            samples_dir=samples_dir,
            results_dir=results_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        ),
        task=task,
    )
    cat.set_model(GNN(model=model, device=torch.device("cpu")))

    run_training(rank=0, cat=cat)
    checkpoint = newest_checkpoint(Path(cat.parameters["io_dict"]["model_dir"]))
    history = load_dictionary(Path(cat.parameters["io_dict"]["model_dir"]) / "run_information.npy")

    inference = run_inference(
        model_name=str(checkpoint),
        rank=0,
        cat=cat,
        test=False,
    )

    train_loss = list(history.get("training_loss", []))
    valid_loss = list(history.get("validation_loss", []))
    if not train_loss or not valid_loss:
        raise RuntimeError("Catalyst training did not record training/validation loss history.")
    if not math.isfinite(float(train_loss[-1])) or not math.isfinite(float(valid_loss[-1])):
        raise RuntimeError("Catalyst training produced a non-finite final loss.")

    metrics = _write_outputs_for_task(
        plot_kind=plot_kind,
        output_dir=Path(output_dir),
        data_dir=data_dir,
        figures_dir=figures_dir,
        history=history,
        inference=inference,
        target_key=target_key,
        target_names=target_names,
    )

    print("\nTask example summary")
    print(f"  task:                 {task.name}")
    print(f"  training graphs:      {len(train)}")
    print(f"  validation graphs:    {len(validation)}")
    print(f"  inference graphs:     {len(test)}")
    print(f"  first training loss:  {float(train_loss[0]):.6e}")
    print(f"  final training loss:  {float(train_loss[-1]):.6e}")
    print(f"  best validation loss: {float(min(valid_loss)):.6e}")
    print(f"  checkpoint:           {checkpoint.name}")
    print(f"  figures directory:    {figures_dir}")
    print(f"  metrics file:         {Path(output_dir) / 'metrics.json'}")
    print(f"  inference vec flag:   {bool(inference.get('vec', False))}")

    return {
        "cat": cat,
        "checkpoint": checkpoint,
        "history": history,
        "inference": inference,
        "metrics": metrics,
        "test_gids": test,
    }


def random_unit_vectors(rng: np.random.Generator, count: int) -> Iterable[np.ndarray]:
    for _ in range(int(count)):
        vector = rng.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm < 1.0e-12:
            vector = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        yield vector / norm
