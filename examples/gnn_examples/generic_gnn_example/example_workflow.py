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
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.neighbors import KDTree
from torch import nn
from torch_geometric.loader import DataLoader
from umap import umap_

# Modern package imports. These assume Catalyst has been installed from the repo
# root with `python -m pip install -e .` and that the package uses the src layout.
from catalyst.characterization.sodas.model.sodas import SODAS
from catalyst.data.utils import load_dictionary, save_dictionary
from catalyst.graph.generic_build import generic_graph_gen
from catalyst.ml.gnn.GNN import GNN
from catalyst.ml.gnn.modules.models.gnn_builder import build_model
from catalyst.ml.inference import run_inference
from catalyst.ml.training import run_active_learning, run_training
from catalyst.ml.utils.distributed import cuda_destroy
from catalyst.observer.params import Catalyst
import catalyst.utilities.sampling as sampling

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_PATH = Path(
    os.environ.get(
        "CATALYST_EXAMPLE_CONFIG",
        Path(__file__).with_name("catalyst_example_config.json"),
    )
)


def load_json_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Could not find Catalyst example config file: {config_path}\n"
            "Set CATALYST_EXAMPLE_CONFIG=/path/to/config.json or place "
            "catalyst_example_config.json next to this script."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


CONFIG = load_json_config()
BASE_DIR = Path(__file__).resolve().parent

# Frequently used settings are unpacked once for readability. Edit the JSON file,
# not this script, to change these values.
N_TYPES = CONFIG["synthetic_data"]["n_types"]
N_DATA = CONFIG["synthetic_data"]["n_data"]
N_DIM = CONFIG["synthetic_data"]["n_dim"]
N_NODES_RANGE = tuple(CONFIG["synthetic_data"]["n_nodes_range"])
NEIGHBOR_RANGE = tuple(CONFIG["synthetic_data"]["neighbor_range"])
ACTIVE_NEIGHBOR_RANGE = tuple(CONFIG["synthetic_data"].get("active_neighbor_range", NEIGHBOR_RANGE))

CUTOFF = CONFIG["model_architecture"]["cutoff"]
N_CONVS = CONFIG["model_architecture"]["n_convs"]
PROJECTION_IN_DIM = CONFIG["model_architecture"]["projection_in_dim"]
PROJECTION_OUT_DIM = CONFIG["model_architecture"]["projection_out_dim"]
REGRESSION_IN_DIM = CONFIG["model_architecture"]["regression_in_dim"]
REGRESSION_OUT_DIM = CONFIG["model_architecture"]["regression_out_dim"]
CONV_TYPE = CONFIG["model_architecture"]["conv_type"]
# Supported by the new modular GenericGNN framework via
# catalyst.ml.gnn.modules.conv.factory:
#   "mesh"/"mgn", "gcn"/"gated_gcn", "gated_gcn_v2",
#   "gine", "edge_conditioned"/"nnconv", "pna"
POOLING = CONFIG["model_architecture"]["pooling"]
DEVICE = CONFIG["catalyst_parameters"]["device_dict"]["device"]

UMAP_N_NEIGHBORS = CONFIG["projection"]["umap_n_neighbors"]
UMAP_MIN_DIST = CONFIG["projection"]["umap_min_dist"]
UMAP_N_COMPONENTS = CONFIG["projection"]["umap_n_components"]

RUN_GENERATE_GRAPHS = CONFIG["workflow"]["generate_graphs"]
RUN_PROJECT_GRAPHS = CONFIG["workflow"]["project_graphs"]
RUN_GENERATE_SAMPLES = CONFIG["workflow"]["generate_samples"]
RUN_TRAINING = CONFIG["workflow"]["train"]
RUN_RETRAINING = CONFIG["workflow"]["retrain"]
RUN_TESTING = CONFIG["workflow"]["test"]
RUN_PLOT_TEST = CONFIG["workflow"]["plot_test"]
RUN_PLOT_TRAINING = CONFIG["workflow"]["plot_training"]
RUN_RANKING = CONFIG["workflow"]["ranking"]
RUN_PREDICTIONS = CONFIG["workflow"]["predictions"]
RUN_ACTIVE_LEARNING = CONFIG["workflow"].get("active_learning", False)
VISUALIZE_FINAL_GRAPH = CONFIG["workflow"]["visualize_final_graph"]

TRAINING_BATCH_SIZE = CONFIG["training_overrides"]["training_batch_size"]
ACTIVE_LEARNING_BATCH_SIZE = CONFIG["training_overrides"].get(
    "active_learning_batch_size", TRAINING_BATCH_SIZE
)
TRAINING_NUM_EPOCHS_OVERRIDE = CONFIG["training_overrides"]["num_epochs"]
TRAINING_DELTA_OVERRIDE = CONFIG["training_overrides"]["train_delta"]
TRAINING_TOLERANCE_OVERRIDE = CONFIG["training_overrides"]["train_tolerance"]

ACTIVE_LEARNING_DATA_DIR = BASE_DIR / CONFIG.get("paths", {}).get(
    "active_learning_data_dir", "active_learning_data"
)

# =============================================================================
# PARAMETER AND MODEL BUILDERS
# =============================================================================


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
    loss_functions = {
        "MSELoss": torch.nn.MSELoss,
        "L1Loss": torch.nn.L1Loss,
        "SmoothL1Loss": torch.nn.SmoothL1Loss,
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
    parameters["sampling_dict"] = dict(parameters["sampling_dict"])
    parameters["loader_dict"] = dict(parameters["loader_dict"])
    parameters["model_dict"] = dict(parameters["model_dict"])

    model_dict = parameters["model_dict"]
    model_dict["optimizer_params"] = dict(model_dict["optimizer_params"])
    model_dict["optimizer_params"]["params_group"] = dict(
        model_dict["optimizer_params"]["params_group"]
    )
    model_dict["active_learning_params_group"] = dict(
        model_dict.get("active_learning_params_group", {})
    )
    if model_dict["active_learning_params_group"]:
        model_dict["active_learning_params_group"]["sampling_params_group"] = dict(
            model_dict["active_learning_params_group"].get("sampling_params_group", {})
        )
        model_dict["active_learning_params_group"]["training_params_group"] = dict(
            model_dict["active_learning_params_group"].get("training_params_group", {})
        )

    # Resolve paths relative to the script location.
    io_dict = parameters["io_dict"]
    for key in ["main_path", "data_dir", "model_dir", "results_dir", "samples_dir", "projection_dir"]:
        io_dict[key] = resolve_relative_path(io_dict.get(key))

    active_learning_group = model_dict.get("active_learning_params_group", {})
    if active_learning_group and "training_data_dir" in active_learning_group:
        active_learning_group["training_data_dir"] = resolve_relative_path(
            active_learning_group.get("training_data_dir")
        )

    # Reconstruct non-JSON Python objects.
    loss_params = dict(model_dict["loss_params"])
    loss_params["function"] = build_loss_function(loss_params["function"])
    if "sub_function" in loss_params and loss_params["sub_function"] is not None:
        loss_params["sub_function"] = build_loss_function(loss_params["sub_function"])
    model_dict["loss_params"] = loss_params
    model_dict["model"] = None

    return parameters


def build_regression_model(device: str = DEVICE) -> GNN:
    """Build the GenericGNN regression model used for training/testing/prediction.

    This example generates Generic_Graph_Data objects, not Atomic_Graph_Data
    objects. Therefore the new modular setup should use the generic feature
    encoder rather than the atomic/materials encoder. The processor still follows
    the ALIGNN-style 1/2/3-body update order through the modular
    processors.order_processor.OrderProcessor internally.
    """
    model = build_model(
        processor_type="order",
        conv_type=CONV_TYPE,
        encoder_type="generic",
        decoder_type="positive",
        encode_3body=True,
        num_species=N_TYPES,      # accepted for compatibility; ignored by generic encoder
        cutoff=CUTOFF,            # accepted for compatibility; ignored by generic encoder
        dim=REGRESSION_IN_DIM,
        num_convs=N_CONVS,
        out_dim=REGRESSION_OUT_DIM,
        act=nn.SiLU(),
        aggr_scheme="add",
    )
    return GNN(model=model, device=device)


def build_projection_model() -> SODAS:
    """Build the SODAS projection model used to generate latent-space projections.

    The projection model uses the same generic encoder/processor setup as the
    regression model but switches to the standard scalar decoder so each graph
    order emits latent vectors of size PROJECTION_OUT_DIM.
    """
    projection_gnn = build_model(
        processor_type="order",
        conv_type=CONV_TYPE,
        encoder_type="generic",
        decoder_type="scalar",
        encode_3body=True,
        num_species=N_TYPES,      # accepted for compatibility; ignored by generic encoder
        cutoff=CUTOFF,            # accepted for compatibility; ignored by generic encoder
        dim=PROJECTION_IN_DIM,
        num_convs=N_CONVS,
        out_dim=PROJECTION_OUT_DIM,
        act=nn.SiLU(),
        aggr_scheme="add",
        combine=True,
    )

    return SODAS(
        mod=projection_gnn,
        ls_mod=umap_.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            n_components=UMAP_N_COMPONENTS,
        ),
        pooling=POOLING,
    )

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


def get_graph_attr(data, *names, default=None):
    for name in names:
        if hasattr(data, name):
            value = getattr(data, name)
            if value is not None:
                return value
    return default


def ensure_generic_order_aliases(data):
    """Expose both old Catalyst names and the newer order-based generic names.

    Existing Catalyst GenericGNN/SODAS code generally expects node_G/node_A/edge_A
    and edge_index_G/edge_index_A. The newer generic graph builders may also
    expose x_1/x_2/x_3 and edge_index_2/edge_index_3. This helper makes a graph
    usable with both conventions.
    """
    alias_pairs = [
        ("node_G", "x_1"),
        ("edge_index_G", "edge_index_2"),
        ("node_A", "x_2"),
        ("edge_index_A", "edge_index_3"),
        ("edge_A", "x_3"),
    ]

    for old_name, new_name in alias_pairs:
        old_has = hasattr(data, old_name) and getattr(data, old_name) is not None
        new_has = hasattr(data, new_name) and getattr(data, new_name) is not None
        if old_has and not new_has:
            setattr(data, new_name, getattr(data, old_name))
        elif new_has and not old_has:
            setattr(data, old_name, getattr(data, new_name))

    return data


def make_follow_batch_fields(graph) -> list[str]:
    """Return DataLoader follow_batch fields compatible with current SODAS."""
    graph = ensure_generic_order_aliases(graph)
    fields = []
    for field in ["node_G", "node_A", "edge_A"]:
        if hasattr(graph, field) and getattr(graph, field) is not None:
            fields.append(field)
    return fields


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
        # Pass rank positionally to match Catalyst functions such as
        # run_training(rank, cat), run_active_learning(rank, cat), etc.
        # Using target(rank=0, *args) causes Python to expand *args first,
        # which can assign the first positional argument to rank and then
        # pass rank again as a keyword.
        target(0, *args)

# =============================================================================
# WORKFLOW FUNCTIONS
# =============================================================================


def visualize_graph(data, atomic: bool = False) -> None:
    del atomic  # Retained for compatibility with the original function signature.
    data = ensure_generic_order_aliases(data)

    colors = np.array([
        "aqua",
        "mediumslateblue",
        "peru",
        "limegreen",
        "darkorange",
        "salmon",
        "brown",
        "gold",
    ])

    edge_index_g = as_numpy_tensor(get_graph_attr(data, "edge_index_G", "edge_index_2"))
    edge_index_a = get_graph_attr(data, "edge_index_A", "edge_index_3", default=None)
    node_features = as_numpy_tensor(get_graph_attr(data, "node_G", "x_1"))

    if edge_index_g.size == 0:
        raise ValueError("Cannot visualize graph because edge_index_G/edge_index_2 is empty.")

    graph_g = nx.Graph(list(edge_index_g.T))
    graph_g_pos = nx.spring_layout(graph_g, seed=42)

    color_map = []
    for node in node_features:
        active = np.where(node != 0.0)[0]
        node_type = int(active[0]) if len(active) else 0
        color_map.append(colors[node_type % len(colors)])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    nx.draw_networkx(
        graph_g,
        graph_g_pos,
        edgecolors="black",
        width=0.4,
        font_size=16,
        node_size=100,
        with_labels=False,
        node_color=color_map,
        edge_color="dimgrey",
        arrows=False,
        ax=ax[0],
    )
    ax[0].set_title("Graph G / order 2 graph")

    if edge_index_a is not None:
        edge_index_a_np = as_numpy_tensor(edge_index_a)
        if edge_index_a_np.size > 0:
            graph_a = nx.Graph(list(edge_index_a_np.T))
            graph_a_pos = nx.spring_layout(graph_a, seed=42)
            nx.draw_networkx(
                graph_a,
                graph_a_pos,
                edgecolors="black",
                width=0.4,
                font_size=16,
                node_size=100,
                with_labels=False,
                edge_color="dimgrey",
                arrows=False,
                ax=ax[1],
            )
            ax[1].set_title("Graph A / order 3 graph")
        else:
            ax[1].set_title("Graph A is empty")
            ax[1].axis("off")
    else:
        ax[1].set_title("Graph A not included")
        ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def generate_random_generic_graph(raw_data: np.ndarray, g_node_labels: np.ndarray, k: int):
    """Build one synthetic generic graph with the modern generic_graph_gen API."""
    n_nodes = len(raw_data)
    if n_nodes < 2:
        raise ValueError("At least two nodes are required to build a pairwise graph.")

    k = max(1, min(int(k), n_nodes - 1))
    tree = KDTree(raw_data, metric="euclidean", leaf_size=2)
    distances, indices = tree.query(raw_data, k=k + 1)  # +1 includes self.

    graph = generic_graph_gen(
        {
            "type": "generic_pairwise",
            "raw_data": raw_data,
            "params": {
                "dist": distances,
                "ind": indices,
                "g_nodes": g_node_labels,
            },
            # Current Catalyst-compatible 1/2/3-body random graph.
            "line_graph": True,
            # New generic graph builders may understand this; old ones will ignore it.
            "max_body_order": 3,
            "remove_self_edges": True,
        }
    )

    if graph is None:
        raise RuntimeError("generic_graph_gen returned None for a non-empty random graph.")

    return ensure_generic_order_aliases(graph)


def generate_data(cat: Catalyst, visualize_final: bool = False) -> None:
    data_dir = reset_dir(cat.parameters["io_dict"]["data_dir"])

    n_nodes = np.linspace(*N_NODES_RANGE, N_DATA)
    k_values = np.linspace(*NEIGHBOR_RANGE, N_DATA)
    targets = [np.linspace(0, 1, N_DATA) for _ in range(REGRESSION_OUT_DIM)]

    rng = np.random.default_rng(seed=CONFIG.get("synthetic_data", {}).get("random_seed", None))

    last_graph = None
    for sample_idx in range(N_DATA):
        if sample_idx % 500 == 0:
            print(f"Generating graph {sample_idx} / {N_DATA}")

        n_sample_nodes = max(2, math.ceil(n_nodes[sample_idx]))
        raw_data = rng.uniform(-1.0, 1.0, size=(n_sample_nodes, N_DIM))
        g_node_labels = np.eye(N_TYPES, dtype=np.float32)[
            rng.choice(N_TYPES, n_sample_nodes)
        ]

        graph = generate_random_generic_graph(
            raw_data=raw_data,
            g_node_labels=g_node_labels,
            k=math.ceil(k_values[sample_idx]),
        )

        graph.y = [
            torch.tensor(targets[target_idx][sample_idx], dtype=torch.float)
            for target_idx in range(REGRESSION_OUT_DIM)
        ]

        torch.save(graph, data_dir / f"{graph.gid}.pt")
        last_graph = graph

    if visualize_final and last_graph is not None:
        visualize_graph(last_graph, atomic=False)


def project_data(cat: Catalyst):
    data_files = sorted(glob.glob(os.path.join(cat.parameters["io_dict"]["data_dir"], "*")))
    if not data_files:
        raise FileNotFoundError(
            f"No graph files found in data_dir={cat.parameters['io_dict']['data_dir']}"
        )

    graph_data = [ensure_generic_order_aliases(safe_torch_load(file_name)) for file_name in data_files]

    print("Performing graph projections...")
    projection_dir = reset_dir(Path(cat.parameters["io_dict"]["main_path"]) / "projections")
    samples_dir = reset_dir(Path(cat.parameters["io_dict"]["main_path"]) / "samples")
    cat.parameters["io_dict"]["projection_dir"] = str(projection_dir)
    cat.parameters["io_dict"]["samples_dir"] = str(samples_dir)

    gids = [data.gid for data in graph_data]
    follow_batch = make_follow_batch_fields(graph_data[0])

    loader = DataLoader(
        graph_data,
        batch_size=cat.parameters["loader_dict"]["batch_size"][0],
        shuffle=False,
        follow_batch=follow_batch,
        num_workers=cat.parameters["loader_dict"]["num_workers"],
    )

    encoded_data = cat.parameters["model_dict"]["model"].generate_gnn_latent_space(
        parameters=cat.parameters,
        loader=loader,
    )
    encoded_data = np.asarray(encoded_data)

    cat.parameters["model_dict"]["model"].fit_preprocess(data=encoded_data)
    cat.parameters["model_dict"]["model"].fit_dim_red(data=encoded_data)
    projected_data = cat.parameters["model_dict"]["model"].project_data(data=encoded_data)

    save_dictionary(
        projection_dir / "projection_data.npy",
        {"projections": projected_data, "gids": gids},
    )
    return graph_data, projected_data


def sample_data(cat: Catalyst, graph_data: Sequence[Any], projected_data: np.ndarray) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12, 4))
    ax[0].plot(projected_data[:, 0], projected_data[:, 1], linestyle="", marker="o", color="w", markeredgecolor="k")
    ax[0].set_title("All data")

    rng = np.random.default_rng(seed=cat.parameters["sampling_dict"]["sampling_seed"])

    test_idx, nontest_idx = sampling.run_sampling(
        projected_data,
        sampling_type=cat.parameters["sampling_dict"]["sampling_types"][0],
        split=cat.parameters["sampling_dict"]["split"][0],
        rng=rng,
        params_group=cat.parameters["sampling_dict"]["params_groups"][0],
    )

    stored_test_data = {
        "projections": [projected_data[index] for index in test_idx],
        "gids": [graph_data[index].gid for index in test_idx],
    }

    projected_remaining = np.asarray([projected_data[index] for index in nontest_idx])
    graph_remaining = [graph_data[index] for index in nontest_idx]
    save_dictionary(Path(cat.parameters["io_dict"]["samples_dir"]) / "test_data.npy", stored_test_data)

    if len(stored_test_data["projections"]) > 0:
        test_projection_array = np.asarray(stored_test_data["projections"])
        ax[1].plot(
            test_projection_array[:, 0],
            test_projection_array[:, 1],
            linestyle="",
            marker="o",
            color="r",
            markeredgecolor="k",
        )
    ax[1].set_title("Test data")

    model_dir = reset_dir(Path(cat.parameters["io_dict"]["samples_dir"]) / "model_samples")
    cat.parameters["io_dict"]["model_dir"] = str(model_dir)

    train_idx, valid_idx = sampling.run_sampling(
        projected_remaining,
        sampling_type=cat.parameters["sampling_dict"]["sampling_types"][1],
        split=cat.parameters["sampling_dict"]["split"][1],
        rng=rng,
        params_group=cat.parameters["sampling_dict"]["params_groups"][1],
    )

    partitioned_data = {
        "training_projections": [projected_remaining[index] for index in train_idx],
        "validation_projections": [projected_remaining[index] for index in valid_idx],
        "training": [graph_remaining[index].gid for index in train_idx],
        "validation": [graph_remaining[index].gid for index in valid_idx],
    }

    print("Using the remaining", len(partitioned_data["validation"]), "for validation")
    save_dictionary(model_dir / "train_valid_split.npy", partitioned_data)

    if len(partitioned_data["training_projections"]) > 0:
        training_projection_array = np.asarray(partitioned_data["training_projections"])
        ax[2].plot(
            training_projection_array[:, 0],
            training_projection_array[:, 1],
            linestyle="",
            marker="o",
            color="y",
            markeredgecolor="k",
        )
    ax[2].set_title("Training data")
    plt.tight_layout()
    plt.show()


def train_model(cat: Catalyst) -> None:
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


def active_learning_model(cat: Catalyst) -> None:
    """Run Catalyst active learning if enabled by the config."""
    cat.parameters["loader_dict"]["batch_size"] = ACTIVE_LEARNING_BATCH_SIZE
    cat.set_model(build_regression_model(DEVICE))
    run_distributed_or_single(cat, run_active_learning, cat)


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
    plt.show()


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

    try:
        run_testing_for_model(
            cat=cat,
            model_dir=main_path / "models" / "pretraining",
            results_dir=main_path / "testing" / "pretraining",
            model_pattern="pre*",
            use_latest_checkpoint=False,
        )
    except FileNotFoundError:
        print("No pretraining model found; skipping pretraining test.")

    run_testing_for_model(
        cat=cat,
        model_dir=main_path / "models" / "training",
        results_dir=main_path / "testing" / "training",
        model_pattern="checkpoint_epoch_*.pt",
        use_latest_checkpoint=True,
    )



def _flatten_prediction_records(obj: Any) -> list[dict[str, Any]]:
    """
    Normalize Catalyst inference output into a flat list of prediction records.

    Older Catalyst examples assumed ``indv_pred.data`` was directly a list of
    dictionaries. Some versions save a dictionary whose values contain the
    records. Iterating over that dictionary yields string keys, which causes
    ``TypeError: string indices must be integers``. This helper handles both.
    """
    records: list[dict[str, Any]] = []

    if obj is None:
        return records

    if isinstance(obj, dict):
        if "y" in obj and "pred" in obj:
            return [obj]

        # Common container keys used by serialized result dictionaries.
        for key in ("records", "data", "results", "predictions", "indv_pred", "items"):
            if key in obj:
                records.extend(_flatten_prediction_records(obj[key]))

        # Fall back to searching every value. This covers gid -> record maps and
        # rank -> record-list maps from distributed inference.
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
    """Convert tensors/lists/scalars into a 1D or 2D float array when possible."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        try:
            return value.astype(float, copy=False)
        except (TypeError, ValueError):
            pass

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


def _append_by_target(target_lists: list[list[float]], value: Any, n_targets: int) -> None:
    """
    Append numeric values to per-target lists while tolerating several Catalyst
    inference-output shapes.
    """
    arr = _as_float_array(value)

    if arr.size == 0:
        return

    if arr.ndim == 0:
        target_lists[0].append(float(arr))
        return

    if n_targets == 1:
        target_lists[0].extend(arr.reshape(-1).astype(float).tolist())
        return

    # Preferred shape: [n_samples, n_targets]
    if arr.ndim >= 2 and arr.shape[-1] == n_targets:
        reshaped = arr.reshape(-1, n_targets)
        for target_idx in range(n_targets):
            target_lists[target_idx].extend(reshaped[:, target_idx].astype(float).tolist())
        return

    # Alternative shape: [n_targets, n_samples]
    if arr.ndim >= 2 and arr.shape[0] == n_targets:
        reshaped = arr.reshape(n_targets, -1)
        for target_idx in range(n_targets):
            target_lists[target_idx].extend(reshaped[target_idx].astype(float).tolist())
        return

    # Single vector containing one value per target.
    flat = arr.reshape(-1).astype(float)
    if flat.size == n_targets:
        for target_idx in range(n_targets):
            target_lists[target_idx].append(float(flat[target_idx]))
        return

    # Last-resort behavior: assign everything to target 0 rather than crashing.
    target_lists[0].extend(flat.tolist())


def plot_test_data(cat: Catalyst) -> None:
    results_dir = Path(cat.parameters["io_dict"]["main_path"]) / "testing" / "training"
    cat.parameters["io_dict"]["results_dir"] = str(results_dir)

    run_data = load_dictionary(results_dir / "indv_pred.data")
    records = _flatten_prediction_records(run_data)

    if not records:
        if isinstance(run_data, dict):
            keys = sorted(str(key) for key in run_data.keys())
            raise RuntimeError(
                "No prediction records with 'y' and 'pred' were found in "
                f"{results_dir / 'indv_pred.data'}. Top-level keys: {keys}"
            )
        raise RuntimeError(
            "No prediction records with 'y' and 'pred' were found in "
            f"{results_dir / 'indv_pred.data'}. Loaded type: {type(run_data)}"
        )

    predictions = [[[] for _ in range(REGRESSION_OUT_DIM)] for _ in range(2)]

    skipped = 0
    for item in records:
        if not isinstance(item, dict) or "y" not in item or "pred" not in item:
            skipped += 1
            continue

        _append_by_target(predictions[0], item["y"], REGRESSION_OUT_DIM)
        _append_by_target(predictions[1], item["pred"], REGRESSION_OUT_DIM)

    if skipped:
        print(f"Skipped {skipped} malformed prediction records during plotting.")

    n_targets = len(predictions[0])
    if n_targets > 1:
        fig, axes = plt.subplots(nrows=1, ncols=n_targets, sharex=False, sharey=False)
        axes = np.atleast_1d(axes)
    else:
        fig, single_ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
        axes = np.asarray([single_ax])

    for target_idx, axis in enumerate(axes):
        true_values = np.asarray(predictions[0][target_idx], dtype=float)
        ml_values = np.asarray(predictions[1][target_idx], dtype=float)

        n_pairs = min(true_values.size, ml_values.size)
        if n_pairs == 0:
            axis.set_title(f"Target {target_idx}: no data")
            axis.axis("off")
            continue

        true_values = true_values[:n_pairs]
        ml_values = ml_values[:n_pairs]

        axis.plot(
            true_values,
            ml_values,
            linestyle="",
            color="dodgerblue",
            marker="o",
            markeredgecolor="k",
        )

        lo = float(min(true_values.min(), ml_values.min()))
        hi = float(max(true_values.max(), ml_values.max()))
        axis.plot([lo, hi], [lo, hi], linestyle="-", color="r")
        axis.set_xlabel("True values")
        axis.set_ylabel("ML values")
        axis.set_title(f"Target {target_idx}")

    plt.tight_layout()
    plt.show()

def predict(cat: Catalyst, interpretable: bool) -> None:
    del interpretable  # retained for config compatibility
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
# MAIN WORKFLOW
# =============================================================================


def main() -> None:
    cat = Catalyst()
    cat.set_params(build_catalyst_parameters(CONFIG))

    projection_model = build_projection_model()
    regression_model = build_regression_model(DEVICE)

    raw_data = None
    projections = None

    if RUN_GENERATE_GRAPHS:
        generate_data(cat, visualize_final=VISUALIZE_FINAL_GRAPH)

    if RUN_PROJECT_GRAPHS:
        cat.set_model(projection_model)
        raw_data, projections = project_data(cat)

    if RUN_GENERATE_SAMPLES:
        if raw_data is None or projections is None:
            raise RuntimeError(
                "RUN_GENERATE_SAMPLES=True requires RUN_PROJECT_GRAPHS=True "
                "in this example workflow."
            )
        sample_data(cat, graph_data=raw_data, projected_data=projections)

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
        cat.set_model(regression_model)
        cat.parameters["model_dict"]["restart_training"] = True
        retrain_model(cat, use_latest_checkpoint=True)

    if RUN_ACTIVE_LEARNING:
        active_learning_model(cat)

    if RUN_TESTING and not RUN_ACTIVE_LEARNING:
        cat.parameters["loader_dict"]["batch_size"] = TRAINING_BATCH_SIZE
        cat.set_model(regression_model)
        test_model(cat)

    if RUN_PLOT_TEST:
        plot_test_data(cat)

    if RUN_PREDICTIONS:
        cat.set_model(regression_model)
        predict(cat, interpretable=RUN_RANKING)


if __name__ == "__main__":
    main()
