
from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.neighbors import KDTree
from torch import nn
from torch_geometric.loader import DataLoader
from umap import umap_

from catalyst.src.characterization.sodas.model.sodas import SODAS
from catalyst.src.data.utils import load_dictionary, save_dictionary
from catalyst.src.graph.generic_build import generic_graph_gen
from catalyst.src.ml.gnn.GNN import GNN
from catalyst.src.ml.gnn.modules.models.alignn import (
    ALIGNN,
    Decoder,
    Encoder_atomic,
    PositiveScalarsDecoder,
    Processor,
)
from catalyst.src.ml.inference import run_inference
from catalyst.src.ml.training import run_active_learning, run_training
from catalyst.src.ml.utils.distributed import cuda_destroy
from catalyst.src.observer.params import Catalyst
import catalyst.src.utilities.sampling as sampling

# =============================================================================
# CONFIGURATION
# =============================================================================

# By default, the script expects catalyst_example_config.json next to this file.
CONFIG_PATH = Path(os.environ.get("CATALYST_EXAMPLE_CONFIG", Path(__file__).with_name("catalyst_example_config.json")))


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

# Frequently used settings are unpacked once for readability inside the workflow
# functions. Edit the JSON file, not this script, to change these values.
BASE_DIR = Path(__file__).resolve().parent
N_TYPES = CONFIG["synthetic_data"]["n_types"]
N_DATA = CONFIG["synthetic_data"]["n_data"]
N_DIM = CONFIG["synthetic_data"]["n_dim"]
N_NODES_RANGE = tuple(CONFIG["synthetic_data"]["n_nodes_range"])
NEIGHBOR_RANGE = tuple(CONFIG["synthetic_data"]["neighbor_range"])
ACTIVE_NEIGHBOR_RANGE = tuple(CONFIG["synthetic_data"]["active_neighbor_range"])

CUTOFF = CONFIG["model_architecture"]["cutoff"]
N_CONVS = CONFIG["model_architecture"]["n_convs"]
PROJECTION_IN_DIM = CONFIG["model_architecture"]["projection_in_dim"]
PROJECTION_OUT_DIM = CONFIG["model_architecture"]["projection_out_dim"]
REGRESSION_IN_DIM = CONFIG["model_architecture"]["regression_in_dim"]
REGRESSION_OUT_DIM = CONFIG["model_architecture"]["regression_out_dim"]
CONV_TYPE = CONFIG["model_architecture"]["conv_type"]
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
RUN_ACTIVE_LEARNING = CONFIG["workflow"]["active_learning"]
VISUALIZE_FINAL_GRAPH = CONFIG["workflow"]["visualize_final_graph"]

TRAINING_BATCH_SIZE = CONFIG["training_overrides"]["training_batch_size"]
ACTIVE_LEARNING_BATCH_SIZE = CONFIG["training_overrides"]["active_learning_batch_size"]
TRAINING_NUM_EPOCHS_OVERRIDE = CONFIG["training_overrides"]["num_epochs"]
TRAINING_DELTA_OVERRIDE = CONFIG["training_overrides"]["train_delta"]
TRAINING_TOLERANCE_OVERRIDE = CONFIG["training_overrides"]["train_tolerance"]

ACTIVE_LEARNING_DATA_DIR = BASE_DIR / CONFIG["paths"]["active_learning_data_dir"]


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
            f"Unsupported loss function '{loss_name}'. "
            f"Supported options are: {sorted(loss_functions)}"
        )
    return loss_functions[loss_name]()

def latest_checkpoint(checkpoint_dir: Path, checkpoint_pattern: str = "checkpoint_epoch_*.pt") -> str:
    """
    Find the checkpoint with the largest epoch number.

    Expected checkpoint filename format:
        checkpoint_epoch_<epoch>.pt

    Example:
        checkpoint_epoch_6.pt
        checkpoint_epoch_120.pt
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    epoch_pattern = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")

    checkpoint_matches = []

    for checkpoint_path in checkpoint_dir.glob(checkpoint_pattern):
        match = epoch_pattern.match(checkpoint_path.name)

        if match is None:
            continue

        epoch = int(match.group(1))
        checkpoint_matches.append((epoch, checkpoint_path))

    if not checkpoint_matches:
        raise FileNotFoundError(
            f"No checkpoint files matching '{checkpoint_pattern}' were found in: {checkpoint_dir}"
        )

    latest_epoch, latest_path = max(checkpoint_matches, key=lambda item: item[0])

    print(f"Loading checkpoint from epoch {latest_epoch}: {latest_path}")

    return str(latest_path)


def build_catalyst_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the Catalyst runtime parameter dictionary from JSON.

    JSON cannot directly store Python objects such as torch.nn.MSELoss() or a live
    model instance, so those are reconstructed here after loading the file.
    """
    parameters = dict(config["catalyst_parameters"])

    # Copy nested dictionaries so runtime edits do not mutate CONFIG unexpectedly.
    parameters["device_dict"] = dict(parameters["device_dict"])
    parameters["io_dict"] = dict(parameters["io_dict"])
    parameters["sampling_dict"] = dict(parameters["sampling_dict"])
    parameters["loader_dict"] = dict(parameters["loader_dict"])
    parameters["model_dict"] = dict(parameters["model_dict"])
    parameters["model_dict"]["optimizer_params"] = dict(parameters["model_dict"]["optimizer_params"])
    parameters["model_dict"]["optimizer_params"]["params_group"] = dict(
        parameters["model_dict"]["optimizer_params"]["params_group"]
    )
    parameters["model_dict"]["active_learning_params_group"] = dict(
        parameters["model_dict"]["active_learning_params_group"]
    )
    parameters["model_dict"]["active_learning_params_group"]["sampling_params_group"] = dict(
        parameters["model_dict"]["active_learning_params_group"]["sampling_params_group"]
    )
    parameters["model_dict"]["active_learning_params_group"]["training_params_group"] = dict(
        parameters["model_dict"]["active_learning_params_group"]["training_params_group"]
    )

    # Resolve paths relative to the script location.
    io_dict = parameters["io_dict"]
    for key in ["main_path", "data_dir", "model_dir", "results_dir", "samples_dir", "projection_dir"]:
        io_dict[key] = resolve_relative_path(io_dict.get(key))

    active_learning_group = parameters["model_dict"]["active_learning_params_group"]
    active_learning_group["training_data_dir"] = resolve_relative_path(
        active_learning_group.get("training_data_dir")
    )

    # Reconstruct non-JSON Python objects.
    loss_params = dict(parameters["model_dict"]["loss_params"])
    loss_params["function"] = build_loss_function(loss_params["function"])
    if "sub_function" in loss_params and loss_params["sub_function"] is not None:
        loss_params["sub_function"] = build_loss_function(loss_params["sub_function"])
    parameters["model_dict"]["loss_params"] = loss_params
    parameters["model_dict"]["model"] = None

    return parameters


def build_regression_model(device: str = DEVICE) -> GNN:
    """Build the ALIGNN regression model used for training/testing/prediction."""
    alignn_model = ALIGNN(
        encoder=Encoder_atomic(
            num_species=N_TYPES,
            cutoff=CUTOFF,
            dim=REGRESSION_IN_DIM,
            act=nn.SiLU(),
        ),
        processor=Processor(
            num_convs=N_CONVS,
            dim=REGRESSION_IN_DIM,
            conv_type=CONV_TYPE,
            act=nn.SiLU(),
        ),
        decoder=PositiveScalarsDecoder(dim=REGRESSION_IN_DIM, act=nn.SiLU()),
        # Alternative decoder:
        # decoder=Decoder(
        #     in_dim=REGRESSION_IN_DIM,
        #     out_dim=REGRESSION_OUT_DIM,
        #     act=nn.SiLU(),
        #     combine=False,
        # ),
    )
    return GNN(model=alignn_model, device=device)



def build_projection_model() -> SODAS:
    """Build the SODAS projection model used to generate latent-space projections."""
    return SODAS(
        mod=ALIGNN(
            encoder=Encoder_atomic(
                num_species=N_TYPES,
                cutoff=CUTOFF,
                dim=PROJECTION_IN_DIM,
                act=nn.SiLU(),
            ),
            processor=Processor(
                num_convs=N_CONVS,
                dim=PROJECTION_IN_DIM,
                conv_type=CONV_TYPE,
                act=nn.SiLU(),
            ),
            decoder=Decoder(
                in_dim=PROJECTION_IN_DIM,
                out_dim=PROJECTION_OUT_DIM,
                act=nn.SiLU(),
            ),
        ),
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
        target(rank=0, *args)


# =============================================================================
# WORKFLOW FUNCTIONS
# =============================================================================


def visualize_graph(data, atomic: bool = False) -> None:
    del atomic  # Retained for compatibility with the original function signature.

    graph_options = {
        "edgecolors": "black",
        "width": 0.4,
        "font_size": 16,
        "node_size": 100,
    }

    colors = [
        "aqua",
        "mediumslateblue",
        "peru",
        "limegreen",
        "darkorange",
        "salmon",
        "brown",
        "gold",
    ]

    edge_index_g = data.edge_index_G.numpy()
    graph_g = nx.Graph(list(edge_index_g.T))
    graph_g_pos = nx.spring_layout(graph_g)

    color_map = []
    for node in data.node_G:
        node_type = np.where(node == 1.0)[0][0]
        color_map.append(colors[node_type])

    fig, ax = plt.subplots(1, 2)
    nx.draw_networkx(
        graph_g,
        graph_g_pos,
        **graph_options,
        with_labels=False,
        node_color=color_map,
        edge_color="dimgrey",
        arrows=False,
        ax=ax[0],
    )

    edge_index_a = data.edge_index_A.numpy()
    graph_a = nx.Graph(list(edge_index_a.T))
    graph_a_pos = nx.spring_layout(graph_a)
    nx.draw_networkx(
        graph_a,
        graph_a_pos,
        **graph_options,
        with_labels=False,
        edge_color="dimgrey",
        arrows=False,
        ax=ax[1],
    )

    ax[0].set_title("Graph G (1,2 body graph)")
    ax[1].set_title("Graph A (2,3 body graph)")
    plt.show()



def generate_data(cat: Catalyst, visualize_final: bool = False) -> None:
    data_dir = reset_dir(cat.parameters["io_dict"]["data_dir"])

    n_nodes = np.linspace(*N_NODES_RANGE, N_DATA)
    k_values = np.linspace(*NEIGHBOR_RANGE, N_DATA)
    targets = [np.linspace(0, 1, N_DATA) for _ in range(REGRESSION_OUT_DIM)]

    last_graph = None
    for sample_idx in range(N_DATA):
        if sample_idx % 500 == 0:
            print(f"Generating graph {sample_idx}")

        raw_data = np.random.uniform(
            -1,
            1,
            size=(math.ceil(n_nodes[sample_idx]), N_DIM),
        )
        g_node_labels = np.eye(N_TYPES)[np.random.choice(N_TYPES, len(raw_data))]

        tree = KDTree(raw_data, metric="euclidean", leaf_size=2)
        distances, indices = tree.query(
            raw_data,
            k=math.ceil(k_values[sample_idx]) + 1,  # +1 includes self-interaction
        )

        graph = generic_graph_gen(
            {
                "raw_data": raw_data,
                "params": {
                    "dist": distances,
                    "ind": indices,
                    "g_nodes": g_node_labels,
                },
                "line_graph": True,
                "type": "generic_pairwise",
            }
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
    graph_data = [
        torch.load(file_name)
        for file_name in glob.glob(os.path.join(cat.parameters["io_dict"]["data_dir"], "*"))
    ]

    print("Performing graph projections...")
    projection_dir = reset_dir(Path(cat.parameters["io_dict"]["main_path"]) / "projections")
    samples_dir = reset_dir(Path(cat.parameters["io_dict"]["main_path"]) / "samples")
    cat.parameters["io_dict"]["projection_dir"] = str(projection_dir)
    cat.parameters["io_dict"]["samples_dir"] = str(samples_dir)

    gids = [data.gid for data in graph_data]
    follow_batch = ["node_G", "node_A", "edge_A"] if hasattr(graph_data[0], "edge_A") else ["node_G", "node_A"]

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
    encoded_data = np.array(encoded_data)

    cat.parameters["model_dict"]["model"].fit_preprocess(data=encoded_data)
    cat.parameters["model_dict"]["model"].fit_dim_red(data=encoded_data)
    projected_data = cat.parameters["model_dict"]["model"].project_data(data=encoded_data)

    save_dictionary(
        projection_dir / "projection_data.npy",
        {"projections": projected_data, "gids": gids},
    )
    return graph_data, projected_data



def sample_data(cat: Catalyst, graph_data, projected_data) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
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

    projected_data = [projected_data[index] for index in nontest_idx]
    graph_data = [graph_data[index] for index in nontest_idx]
    save_dictionary(Path(cat.parameters["io_dict"]["samples_dir"]) / "test_data.npy", stored_test_data)

    ax[1].plot(
        np.array(stored_test_data["projections"])[:, 0],
        np.array(stored_test_data["projections"])[:, 1],
        linestyle="",
        marker="o",
        color="r",
        markeredgecolor="k",
    )
    ax[1].set_title("Test data")

    model_dir = reset_dir(Path(cat.parameters["io_dict"]["samples_dir"]) / "model_samples")
    cat.parameters["io_dict"]["model_dir"] = str(model_dir)

    train_idx, valid_idx = sampling.run_sampling(
        projected_data,
        sampling_type=cat.parameters["sampling_dict"]["sampling_types"][1],
        split=cat.parameters["sampling_dict"]["split"][1],
        rng=rng,
        params_group=cat.parameters["sampling_dict"]["params_groups"][1],
    )

    partitioned_data = {
        "training_projections": [projected_data[index] for index in train_idx],
        "validation_projections": [projected_data[index] for index in valid_idx],
        "training": [graph_data[index].gid for index in train_idx],
        "validation": [graph_data[index].gid for index in valid_idx],
    }

    print("Using the remaining", len(partitioned_data["validation"]), "for validation")
    save_dictionary(model_dir / "train_valid_split.npy", partitioned_data)

    ax[2].plot(
        np.array(partitioned_data["training_projections"])[:, 0],
        np.array(partitioned_data["training_projections"])[:, 1],
        linestyle="",
        marker="o",
        color="y",
        markeredgecolor="k",
    )
    ax[2].set_title("Training data")
    plt.show()



def train_model(cat: Catalyst) -> None:
    cat.parameters["io_dict"]["samples_dir"] = str(Path(cat.parameters["io_dict"]["main_path"]) / "samples" / "model_samples")
    cat.set_model(build_regression_model(DEVICE))

    if cat.parameters["device_dict"]["run_ddp"]:
        print("Performing training on model...")
        processes = []
        for rank in range(cat.parameters["device_dict"]["world_size"]):
            process = mp.Process(target=run_training, args=(rank, cat))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    else:
        run_training(rank=0, cat=cat)



def retrain_model(cat: Catalyst,use_latest_checkpoint: bool = False) -> None:
    cat.parameters["io_dict"]["samples_dir"] = str(Path(cat.parameters["io_dict"]["main_path"]) / "samples" / "model_samples")
    model_pattern = "checkpoint_epoch_*.pt"
    model_dir = os.path.join(Path(cat.parameters["io_dict"]["main_path"]),'models','training')
    if use_latest_checkpoint:
        loaded_model_name = latest_checkpoint(
            checkpoint_dir=model_dir,
            checkpoint_pattern=model_pattern,
        )
    else:
        loaded_model_name = first_match(os.path.join(model_dir,model_pattern))
    cat.parameters["io_dict"].update(
        {
            "model_dir": str(model_dir),
            "loaded_model_name": loaded_model_name,
        }
    )

    if cat.parameters["device_dict"]["run_ddp"]:
        print("Performing model retraining on", cat.parameters["io_dict"]["loaded_model_name"])
        processes = []
        for rank in range(cat.parameters["device_dict"]["world_size"]):
            process = mp.Process(target=run_training, args=(rank, cat))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        cuda_destroy()
    else:
        run_training(rank=0, cat=cat)



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
    if use_latest_checkpoint:
        loaded_model_name = latest_checkpoint(
            checkpoint_dir=model_dir,
            checkpoint_pattern=model_pattern,
        )
    else:
        loaded_model_name = first_match(model_dir / model_pattern)

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
            process = mp.Process(
                target=run_inference,
                args=(cat.parameters["io_dict"]['loaded_model_name'],rank, cat,True),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        cuda_destroy()

    else:
        run_inference(model_name=cat.parameters["io_dict"]['loaded_model_name'], cat=cat,test=True)

def test_model(cat: Catalyst) -> None:
    main_path = Path(cat.parameters["io_dict"]["main_path"])
    cat.parameters["io_dict"]["samples_dir"] = str(
        Path(cat.parameters["io_dict"]["main_path"]) / "samples")

    # Test optional pretraining model if present.
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

    # Test the normal training model.
    run_testing_for_model(
        cat=cat,
        model_dir=main_path / "models" / "training",
        results_dir=main_path / "testing" / "training",
        model_pattern="checkpoint_epoch_*.pt",
        use_latest_checkpoint=True,
    )


def plot_test_data(cat: Catalyst) -> None:
    results_dir = Path(cat.parameters["io_dict"]["main_path"]) / "testing" / "training"
    cat.parameters["io_dict"]["results_dir"] = str(results_dir)

    run_data = [load_dictionary(results_dir / "indv_pred.data")][0]
    predictions = [[[] for _ in range(REGRESSION_OUT_DIM)] for _ in range(2)]
    for data in run_data:
        if data["vec"]:
            for target_values in data["y"]:
                for target_idx, value in enumerate(target_values):
                    if data["loss_fn"] == "sum":
                        predictions[0][target_idx].append(value)
                    else:
                        predictions[0][target_idx].extend(value)
            for predicted_values in data["pred"]:
                for target_idx, value in enumerate(predicted_values):
                    if data["loss_fn"] == "sum":
                        predictions[1][target_idx].append(value)
                    else:
                        predictions[1][target_idx].extend(value)
        else:
            for target_values in data["y"]:
                if data["loss_fn"] == "sum":
                    predictions[0][0].append(target_values)
                else:
                    predictions[0][0].extend(target_values)
            for predicted_values in data["pred"]:
                if data["loss_fn"] == "sum":
                    predictions[1][0].append(predicted_values)
                else:
                    predictions[1][0].extend(predicted_values)

    n_targets = len(predictions[0])
    if n_targets > 1:
        fig, ax = plt.subplots(nrows=1, ncols=n_targets, sharex=True, sharey=False)
        axes = ax
    else:
        fig, single_ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
        axes = [single_ax]

    for target_idx, axis in enumerate(axes):
        true_values = predictions[0][target_idx]
        ml_values = predictions[1][target_idx]
        axis.plot(true_values, ml_values, linestyle="", color="dodgerblue", marker="o", markeredgecolor="k")
        axis.plot(true_values, true_values, linestyle="-", color="r")
        axis.set_xlabel("True values")
        axis.set_ylabel("ML values")

    plt.show()



def predict(cat: Catalyst, interpretable: bool) -> None:
    main_path = Path(cat.parameters["io_dict"]["main_path"])
    model_dir = main_path / "models" / "training" / "0"
    results_dir = main_path / "testing" / "predict"

    cat.parameters["io_dict"].update(
        {
            "write_indv_pred": False,
            "results_dir": str(reset_dir(results_dir)),
            "model_dir": str(model_dir),
            "loaded_model_name": first_match(model_dir / "model*"),
        }
    )

    # Requires predict_external to be imported above.
    if cat.parameters["device_dict"]["run_ddp"]:
        processes = []
        for rank in range(cat.parameters["device_dict"]["world_size"]):
            process = mp.Process(target=predict_external, args=(cat, "all", rank, interpretable))  # noqa: F821
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        cuda_destroy()
    else:
        predict_external(cat, "all", rank=0, interpretable=interpretable)  # noqa: F821




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
            raise RuntimeError("RUN_GENERATE_SAMPLES=True requires RUN_PROJECT_GRAPHS=True in this example workflow.")
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
        retrain_model(cat)

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
