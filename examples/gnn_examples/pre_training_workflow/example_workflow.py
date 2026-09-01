"""Pre-training generic graph generation, latent projection, and sampling example.

This example intentionally contains no supervised GNN training or testing.
Task-specific training examples now live under ``gnn_examples/task_examples``.
Its purpose is to demonstrate the distinct Catalyst workflow:

    synthetic generic data
        -> generic_graph_gen
        -> SODAS latent encoding
        -> UMAP projection
        -> Catalyst sampling utilities
        -> saved train/validation/test graph IDs

All figures are written to ``figures/`` with a non-interactive Matplotlib
backend.
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch import nn
from torch_geometric.loader import DataLoader
from umap import umap_

from catalyst.characterization.sodas.model.sodas import SODAS
from catalyst.data.utils import save_dictionary
from catalyst.graph.generic_build import generic_graph_gen
from catalyst.ml.gnn.modules.models.gnn_builder import build_model
from catalyst.observer import Catalyst
import catalyst.utilities.sampling as sampling


CONFIG_PATH = Path(
    os.environ.get(
        "CATALYST_EXAMPLE_CONFIG",
        Path(__file__).with_name("catalyst_example_config.json"),
    )
)


def load_json_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Could not find pre-training workflow config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


CONFIG = load_json_config()
N_TYPES = int(CONFIG["synthetic_data"]["n_types"])
N_DATA = int(CONFIG["synthetic_data"]["n_data"])
N_DIM = int(CONFIG["synthetic_data"]["n_dim"])
N_NODES_RANGE = tuple(CONFIG["synthetic_data"]["n_nodes_range"])
NEIGHBOR_RANGE = tuple(CONFIG["synthetic_data"]["neighbor_range"])

CUTOFF = float(CONFIG["model_architecture"]["cutoff"])
N_CONVS = int(CONFIG["model_architecture"]["n_convs"])
PROJECTION_IN_DIM = int(CONFIG["model_architecture"]["projection_in_dim"])
PROJECTION_OUT_DIM = int(CONFIG["model_architecture"]["projection_out_dim"])
CONV_TYPE = str(CONFIG["model_architecture"]["conv_type"])
POOLING = str(CONFIG["model_architecture"]["pooling"])

UMAP_N_NEIGHBORS = int(CONFIG["projection"]["umap_n_neighbors"])
UMAP_MIN_DIST = float(CONFIG["projection"]["umap_min_dist"])
UMAP_N_COMPONENTS = int(CONFIG["projection"]["umap_n_components"])

RUN_GENERATE_GRAPHS = bool(CONFIG["workflow"].get("generate_graphs", True))
RUN_PROJECT_GRAPHS = bool(CONFIG["workflow"].get("project_graphs", True))
RUN_GENERATE_SAMPLES = bool(CONFIG["workflow"].get("generate_samples", True))
VISUALIZE_FINAL_GRAPH = bool(CONFIG["workflow"].get("visualize_final_graph", True))


def reset_dir(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_dir(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_torch_load(path: os.PathLike[str] | str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_figures_dir(cat: Catalyst) -> Path:
    return make_dir(Path(cat.parameters["io_dict"]["main_path"]) / "figures")


def ensure_generic_order_aliases(data):
    alias_pairs = [
        ("node_G", "x_1"),
        ("edge_index_G", "edge_index_2"),
        ("node_A", "x_2"),
        ("edge_index_A", "edge_index_3"),
        ("edge_A", "x_3"),
    ]
    for old_name, new_name in alias_pairs:
        old_value = getattr(data, old_name, None)
        new_value = getattr(data, new_name, None)
        if old_value is not None and new_value is None:
            setattr(data, new_name, old_value)
        elif new_value is not None and old_value is None:
            setattr(data, old_name, new_value)
    return data


def follow_batch_fields(graph) -> list[str]:
    graph = ensure_generic_order_aliases(graph)
    return [
        name
        for name in ("node_G", "node_A", "edge_A")
        if getattr(graph, name, None) is not None
    ]


def build_projection_model() -> SODAS:
    gnn = build_model(
        model_type="generic",
        processor_type="order",
        conv_type=CONV_TYPE,
        encoder_type="generic",
        decoder_type="scalar",
        encode_3body=True,
        num_species=N_TYPES,
        cutoff=CUTOFF,
        dim=PROJECTION_IN_DIM,
        num_convs=N_CONVS,
        out_dim=PROJECTION_OUT_DIM,
        act=nn.SiLU(),
        aggr_scheme="add",
        combine=True,
    )
    return SODAS(
        mod=gnn,
        ls_mod=umap_.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            n_components=UMAP_N_COMPONENTS,
            random_state=112358,
        ),
        pooling=POOLING,
    )


def generate_random_generic_graph(raw_data: np.ndarray, node_labels: np.ndarray, k: int):
    n_nodes = len(raw_data)
    k = max(1, min(int(k), n_nodes - 1))
    tree = KDTree(raw_data, metric="euclidean", leaf_size=2)
    distances, indices = tree.query(raw_data, k=k + 1)

    graph = generic_graph_gen(
        {
            "type": "generic_pairwise",
            "raw_data": raw_data,
            "params": {
                "dist": distances,
                "ind": indices,
                "g_nodes": node_labels,
                "use_raw_data_as_pos": True,
            },
            "line_graph": True,
            "include_angs": True,
            "include_self_edges": False,
            "strict": True,
            "include_equivariant_fields": True,
        }
    )
    return ensure_generic_order_aliases(graph)


def visualize_graph(data, output_path: Path) -> None:
    edge_index = getattr(data, "edge_index_G").detach().cpu().numpy()
    node_features = getattr(data, "node_G").detach().cpu().numpy()
    graph_nx = nx.Graph(list(edge_index.T))
    pos = nx.spring_layout(graph_nx, seed=42)
    type_ids = np.argmax(node_features, axis=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx(
        graph_nx,
        pos,
        node_color=type_ids,
        edgecolors="black",
        width=0.5,
        node_size=120,
        with_labels=False,
        ax=ax,
    )
    ax.set_title("Synthetic generic graph")
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def generate_data(cat: Catalyst, visualize_final: bool = False) -> None:
    data_dir = reset_dir(cat.parameters["io_dict"]["data_dir"])
    node_counts = np.linspace(*N_NODES_RANGE, N_DATA)
    k_values = np.linspace(*NEIGHBOR_RANGE, N_DATA)
    rng = np.random.default_rng(112358)

    last_graph = None
    for sample_idx in range(N_DATA):
        if sample_idx % 500 == 0:
            print(f"Generating graph {sample_idx} / {N_DATA}")
        n_nodes = max(3, math.ceil(node_counts[sample_idx]))
        raw_data = rng.uniform(-1.0, 1.0, size=(n_nodes, N_DIM)).astype(np.float32)
        labels = np.eye(N_TYPES, dtype=np.float32)[rng.integers(0, N_TYPES, size=n_nodes)]
        graph = generate_random_generic_graph(raw_data, labels, math.ceil(k_values[sample_idx]))
        graph.gid = f"generic_sample_{sample_idx:05d}"
        torch.save(graph, data_dir / f"{graph.gid}.pt")
        last_graph = graph

    if visualize_final and last_graph is not None:
        visualize_graph(last_graph, get_figures_dir(cat) / "generic_graph_visualization.png")


def project_data(cat: Catalyst, projector: SODAS):
    data_files = sorted(glob.glob(os.path.join(cat.parameters["io_dict"]["data_dir"], "*.pt")))
    if not data_files:
        raise FileNotFoundError("No generic graph files were found for projection.")

    graphs = [ensure_generic_order_aliases(safe_torch_load(path)) for path in data_files]
    projection_dir = reset_dir(Path(cat.parameters["io_dict"]["main_path"]) / "projections")
    samples_dir = reset_dir(Path(cat.parameters["io_dict"]["main_path"]) / "samples")
    cat.set_params(
        {"io_dict": {"projection_dir": str(projection_dir), "samples_dir": str(samples_dir)}},
        save_params=False,
    )

    loader = DataLoader(
        graphs,
        batch_size=int(cat.parameters["loader_dict"]["batch_size"][0]),
        shuffle=False,
        follow_batch=follow_batch_fields(graphs[0]),
        num_workers=int(cat.parameters["loader_dict"]["num_workers"]),
    )

    device = str(cat.parameters["device_dict"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested by config but unavailable; using CPU for this example.")
        device = "cpu"
        cat.set_params({"device_dict": {"device": "cpu"}}, save_params=False)
    projector.model.to(device)

    print("Performing graph projections...")
    encoded = np.asarray(projector.generate_gnn_latent_space(cat.parameters, loader))
    projector.fit_preprocess(encoded)
    projector.fit_dim_red(encoded)
    projected = projector.project_data(encoded)

    gids = [str(graph.gid) for graph in graphs]
    save_dictionary(
        projection_dir / "projection_data.npy",
        {"projections": projected, "gids": gids},
    )
    return graphs, projected


def sample_data(cat: Catalyst, graph_data: Sequence[Any], projected_data: np.ndarray) -> None:
    rng = np.random.default_rng(int(cat.parameters["sampling_dict"]["sampling_seed"]))
    test_idx, remaining_idx = sampling.run_sampling(
        projected_data,
        sampling_type=cat.parameters["sampling_dict"]["sampling_types"][0],
        split=cat.parameters["sampling_dict"]["split"][0],
        rng=rng,
        params_group=cat.parameters["sampling_dict"]["params_groups"][0],
    )
    test_idx = np.asarray(test_idx, dtype=int).reshape(-1)
    remaining_idx = np.asarray(remaining_idx, dtype=int).reshape(-1)

    remaining_projection = np.asarray([projected_data[i] for i in remaining_idx])
    remaining_graphs = [graph_data[i] for i in remaining_idx]
    train_idx, valid_idx = sampling.run_sampling(
        remaining_projection,
        sampling_type=cat.parameters["sampling_dict"]["sampling_types"][1],
        split=cat.parameters["sampling_dict"]["split"][1],
        rng=rng,
        params_group=cat.parameters["sampling_dict"]["params_groups"][1],
    )
    train_idx = np.asarray(train_idx, dtype=int).reshape(-1)
    valid_idx = np.asarray(valid_idx, dtype=int).reshape(-1)

    samples_dir = Path(cat.parameters["io_dict"]["samples_dir"])
    model_samples_dir = reset_dir(samples_dir / "model_samples")
    test_gids = [str(graph_data[i].gid) for i in test_idx]
    train_gids = [str(remaining_graphs[i].gid) for i in train_idx]
    valid_gids = [str(remaining_graphs[i].gid) for i in valid_idx]

    save_dictionary(samples_dir / "test_data.npy", {"gids": test_gids, "validation": test_gids})
    save_dictionary(
        model_samples_dir / "train_valid_split.npy",
        {"training": train_gids, "validation": valid_gids},
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes[0].scatter(projected_data[:, 0], projected_data[:, 1], s=12)
    axes[0].set_title("All graphs")
    if test_idx.size > 0:
        p = np.asarray([projected_data[i] for i in test_idx])
        axes[1].scatter(p[:, 0], p[:, 1], s=12)
    axes[1].set_title(f"Test: {len(test_gids)}")
    if train_idx.size > 0:
        p = np.asarray([remaining_projection[i] for i in train_idx])
        axes[2].scatter(p[:, 0], p[:, 1], s=12)
    axes[2].set_title(f"Training: {len(train_gids)}")
    for ax in axes:
        ax.set_xlabel("Projection 1")
    axes[0].set_ylabel("Projection 2")
    fig.tight_layout()
    output_path = get_figures_dir(cat) / "sampling_projection_split.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")
    print(f"Sampling complete: train={len(train_gids)}, validation={len(valid_gids)}, test={len(test_gids)}")


def main() -> None:
    cat = Catalyst(parameter_file=CONFIG_PATH)
    projector = build_projection_model()

    graphs = None
    projections = None

    if RUN_GENERATE_GRAPHS:
        generate_data(cat, visualize_final=VISUALIZE_FINAL_GRAPH)

    if RUN_PROJECT_GRAPHS:
        graphs, projections = project_data(cat, projector)

    if RUN_GENERATE_SAMPLES:
        if graphs is None or projections is None:
            raise RuntimeError(
                "generate_samples=True requires project_graphs=True in this graph-generation/sampling example."
            )
        sample_data(cat, graphs, projections)

    print("Pre-training graph generation/projection/sampling workflow completed.")
    print("Supervised GNN training examples are under gnn_examples/task_examples/.")


if __name__ == "__main__":
    main()
